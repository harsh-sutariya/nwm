# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                          Projection Heads for AC-REPA                         #
#################################################################################

class ProjectionHead(nn.Module):
    """
    Projection head for mapping features to common alignment space.
    Used for both student (P_s) and teacher (P_t) projections in AC-REPA.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        if hidden_dim is None:
            hidden_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x shape: [B, T*N, input_dim] or [B, T, N, input_dim]
        if x.dim() == 4:  # [B, T, N, input_dim]
            B, T, N, D = x.shape
            x = x.reshape(B, T * N, D)  # Flatten tokens: [B, T*N, input_dim]

        return self.layers(x)


class FeatureProjector(nn.Module):
    """
    Combined projector for AC-REPA alignment.
    Contains both student and teacher projection heads.
    """
    def __init__(self, student_dim=1152, teacher_dim=768, proj_dim=512, hidden_dim=None):
        super().__init__()

        self.proj_dim = proj_dim
        self.student_proj = ProjectionHead(student_dim, proj_dim, hidden_dim)
        self.teacher_proj = ProjectionHead(teacher_dim, proj_dim, hidden_dim)

    def forward_student(self, student_features):
        """
        Project student features: h ∈ ℝ^(T×N×d) → ℝ^(T×N×d')
        Args:
            student_features: [B, T, N, student_dim] or [B, T*N, student_dim]
        Returns:
            projected_features: [B, T*N, proj_dim]
        """
        return self.student_proj(student_features)

    def forward_teacher(self, teacher_features):
        """
        Project teacher features: z_T ∈ ℝ^(T×N_T×d_T) → ℝ^(T×N_T×d')
        Args:
            teacher_features: [B, T, N_T, teacher_dim] or [B, T*N_T, teacher_dim]
        Returns:
            projected_features: [B, T*N_T, proj_dim]
        """
        return self.teacher_proj(teacher_features)

    def forward_both(self, student_features, teacher_features):
        """
        Project both student and teacher features for alignment.
        Returns:
            student_proj: [B, T*N, proj_dim]
            teacher_proj: [B, T*N_T, proj_dim]
        """
        return self.forward_student(student_features), self.forward_teacher(teacher_features)


def align_teacher_to_student_grid(teacher_proj, teacher_shape, target_shape):
    """
    Align teacher tokens to the student's spatio-temporal grid without collapsing spatial structure.

    Args:
        teacher_proj: [B, T*N_T, proj_dim] flattened teacher tokens
        teacher_shape: Tuple describing original teacher layout (B, T, N_T, proj_dim)
        target_shape: Desired (T_target, N_target) shape for the student grid

    Returns:
        aligned_teacher: [B, T_target, N_target, proj_dim]
    """
    B, T_teacher, N_teacher, proj_dim = teacher_shape
    T_target, N_target = target_shape

    if T_teacher != T_target:
        raise ValueError(f"Teacher temporal length {T_teacher} does not match student length {T_target}.")

    teacher_spatial = teacher_proj.reshape(B, T_teacher, N_teacher, proj_dim)

    if N_teacher == N_target:
        return teacher_spatial

    teacher_side = int(math.isqrt(N_teacher))
    target_side = int(math.isqrt(N_target))

    if teacher_side * teacher_side == N_teacher and target_side * target_side == N_target:
        teacher_2d = teacher_spatial.view(B * T_teacher, teacher_side, teacher_side, proj_dim).permute(0, 3, 1, 2)
        resized = F.interpolate(
            teacher_2d,
            size=(target_side, target_side),
            mode='bicubic',
            align_corners=False
        )
        resized = resized.permute(0, 2, 3, 1).reshape(B, T_teacher, N_target, proj_dim)
        return resized

    # Fallback to 1D interpolation along flattened spatial tokens if 2D interpolation fails (when N_teacher != N_target and N_teacher is not a perfect square)
    target_positions = torch.linspace(0, N_teacher - 1, steps=N_target, device=teacher_proj.device, dtype=torch.float32)
    lower_idx = target_positions.floor().long()
    upper_idx = torch.clamp(lower_idx + 1, max=N_teacher - 1)
    alpha = (target_positions - lower_idx.float()).view(1, 1, N_target, 1)

    lower_feat = teacher_spatial.index_select(2, lower_idx)
    upper_feat = teacher_spatial.index_select(2, upper_idx)
    aligned = torch.lerp(lower_feat, upper_feat, alpha)

    same_mask = (upper_idx == lower_idx).view(1, 1, N_target, 1)
    if same_mask.any():
        aligned = torch.where(same_mask, lower_feat, aligned)

    return aligned


class FeatureAlignmentLoss(nn.Module):
    """
    Feature Alignment (FA) loss for AC-REPA.
    Aligns teacher tokens to the student grid and penalizes feature differences.
    """
    def __init__(self, pool_type='mean', use_cross_attention=False):
        super().__init__()
        self.pool_type = pool_type
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.cross_attn = CrossAttentionPooling()

    def forward(self, student_proj, teacher_proj, student_shape, teacher_shape):
        B, T, N, _ = student_shape

        student_spatial = student_proj.reshape(B, T, N, -1)
        teacher_aligned = align_teacher_to_student_grid(
            teacher_proj,
            teacher_shape,
            (T, N)
        )

        if self.pool_type == 'cross_attention' and self.use_cross_attention:
            teacher_aligned = self._cross_attention_pool(teacher_proj, student_spatial, teacher_shape, (T, N))
        elif self.pool_type not in ('mean', 'cross_attention'):
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        fa_loss = torch.mean((student_spatial - teacher_aligned) ** 2)

        return fa_loss, teacher_aligned

    def _cross_attention_pool(self, teacher_proj, student_spatial, teacher_shape, target_shape):
        """Cross-attention pooling using student features as queries."""
        if not self.use_cross_attention:
            raise RuntimeError("Cross-attention pooling requested but use_cross_attention=False")
        return self.cross_attn(teacher_proj, student_spatial, teacher_shape, target_shape)




class CrossAttentionPooling(nn.Module):
    """
    Lightweight cross-attention for pooling teacher features to the student grid while preserving spatial structure.
    """
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, teacher_features, student_features, teacher_shape, target_shape):
        """
        Use student features as queries to attend to aligned teacher tokens.

        Args:
            teacher_features: [B, T*N_T, proj_dim] flattened teacher tokens
            student_features: [B, T, N, proj_dim] student tokens
            teacher_shape: (B, T, N_T, proj_dim) original teacher layout
            target_shape: (T_target, N_target) desired student grid dimensions

        Returns:
            [B, T, N, proj_dim] attention-refined teacher representation
        """
        teacher_aligned = align_teacher_to_student_grid(teacher_features, teacher_shape, target_shape)

        B, T_stud, N_stud, proj_dim = student_features.shape
        student_seq = student_features.reshape(B * T_stud, N_stud, proj_dim)
        teacher_seq = teacher_aligned.reshape(B * T_stud, N_stud, proj_dim)

        attn_output, _ = self.cross_attention(student_seq, teacher_seq, teacher_seq)
        attn_output = self.norm(attn_output)
        attn_output = attn_output.view(B, T_stud, N_stud, proj_dim)

        return attn_output


class ACRepaLoss(nn.Module):
    """
    Complete AC-REPA training objective combining diffusion, FA, and AC-TRD losses.
    """
    def __init__(self, student_dim=1152, teacher_dim=768, proj_dim=512,
                 fa_pool_type='mean', trd_gate_type='temporal',
                 lambda_fa=1.0, lambda_trd=1.0, use_sparse_trd=True, sparse_ratio=0.25):
        super().__init__()

        # Initialize projection heads
        self.projector = FeatureProjector(student_dim, teacher_dim, proj_dim)

        # Initialize alignment losses
        self.fa_loss_fn = FeatureAlignmentLoss(pool_type=fa_pool_type, use_cross_attention=False)
        self.trd_loss_fn = ActionConditionedTRDLoss(gate_type=trd_gate_type,
                                                  use_sparse_tokens=use_sparse_trd,
                                                  sparse_ratio=sparse_ratio)

        # Loss weights
        self.lambda_fa = lambda_fa
        self.lambda_trd = lambda_trd

    def forward(self, student_features, teacher_features, actions=None,
                diffusion_loss=None, compute_gradients=True):
        """
        Compute complete AC-REPA loss.

        Args:
            student_features: [B, T, N, student_dim] - CDiT features
            teacher_features: [B, T, N_T, teacher_dim] - VideoMAE features
            actions: [B, T, action_dim] - Action commands (optional)
            diffusion_loss: Scalar diffusion loss (computed externally)
            compute_gradients: Whether to enable gradients

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Project features to common space
        student_proj, teacher_proj = self.projector.forward_both(student_features, teacher_features)

        # Get dimensions for loss computation
        B, T, N, _ = student_features.shape
        B_T, T_T, N_T, _ = teacher_features.shape

        student_shape = (B, T, N, self.projector.proj_dim)
        teacher_shape = (B_T, T_T, N_T, self.projector.proj_dim)

        # Compute Feature Alignment loss
        fa_loss, _ = self.fa_loss_fn(student_proj, teacher_proj, student_shape, teacher_shape)

        # Compute AC-TRD loss
        trd_loss = self.trd_loss_fn(student_proj, teacher_proj, student_shape, teacher_shape, actions)

        # Combine losses
        total_loss = diffusion_loss + self.lambda_fa * fa_loss + self.lambda_trd * trd_loss

        # Return loss dictionary for logging
        loss_dict = {
            'diffusion_loss': diffusion_loss.item() if torch.is_tensor(diffusion_loss) else diffusion_loss,
            'fa_loss': fa_loss.item(),
            'trd_loss': trd_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


class ActionConditionedTRDLoss(nn.Module):
    """
    Action-Conditioned Token Relation Distillation (AC-TRD) loss for AC-REPA.
    Aligns space-time token relations (cosine similarities) rather than raw features.
    Inspired by VideoREPA for temporal coherence.
    """
    def __init__(self, gate_type='temporal', use_sparse_tokens=True, sparse_ratio=0.25):
        super().__init__()
        self.gate_type = gate_type
        self.use_sparse_tokens = use_sparse_tokens
        self.sparse_ratio = sparse_ratio

    def forward(self, student_proj, teacher_proj, student_shape, teacher_shape, actions=None):
        """
        Compute AC-TRD loss.

        Args:
            student_proj: [B, T*N, proj_dim] - Projected student features
            teacher_proj: [B, T*N_T, proj_dim] or [B, T, N, proj_dim] - Projected teacher features
            student_shape: [B, T, N, proj_dim] - Original student shape
            teacher_shape: [B, T, N_T, proj_dim] or [B, T, N, proj_dim] - Teacher shape
            actions: [B, T, action_dim] - Optional action commands for gating

        Returns:
            trd_loss: Scalar AC-TRD loss
        """
        B, T, N, proj_dim = student_shape

        # If teacher_proj is pooled to student grid, it might already be [B, T, N, proj_dim]
        if teacher_proj.dim() == 3:  # [B, T*N, proj_dim]
            # Pool teacher to student grid first
            teacher_pooled = self._pool_teacher_to_student_grid(
                teacher_proj, teacher_shape, (T, N)
            )
        else:  # Already [B, T, N, proj_dim]
            teacher_pooled = teacher_proj

        # Reshape to spatial grid
        student_spatial = student_proj.reshape(B, T, N, proj_dim)  # [B, T, N, proj_dim]

        # L2 normalize features
        X = torch.nn.functional.normalize(student_spatial, p=2, dim=-1)  # [B, T, N, d']
        Y = torch.nn.functional.normalize(teacher_pooled, p=2, dim=-1)   # [B, T, N, d']

        # Flatten spatial-temporal dimensions: [B, T, N, d'] -> [B, T*N, d']
        X_flat = X.reshape(B, T * N, proj_dim)
        Y_flat = Y.reshape(B, T * N, proj_dim)

        # Compute Gram (relation) matrices: R = XX^T
        # [B, T*N, d'] @ [B, d', T*N] -> [B, T*N, T*N]
        R_s = torch.bmm(X_flat, X_flat.transpose(1, 2))  # [B, T*N, T*N]
        R_T = torch.bmm(Y_flat, Y_flat.transpose(1, 2))  # [B, T*N, T*N]

        # Build action/motion gate
        G = self._build_action_gate(T, N, B, actions, device=student_proj.device)  # [B, T*N, T*N] or [T*N, T*N]

        # Apply sparse token selection if enabled
        if self.use_sparse_tokens:
            # Select top-k tokens based on motion/importance
            num_tokens = int(T * N * self.sparse_ratio)
            token_mask = self._select_sparse_tokens(X_flat, num_tokens)  # [B, T*N]

            # Apply mask to relations
            R_s = self._mask_relations(R_s, token_mask)
            R_T = self._mask_relations(R_T, token_mask)
            G = self._mask_relations(G, token_mask) if G.dim() == 3 else G

        # Compute weighted Frobenius norm
        # L_TRD = (1/||G||_1) * ||G ⊙ (R_s - R_T)||_F^2
        diff = R_s - R_T  # [B, T*N, T*N]

        if G.dim() == 2:  # Shared gate across batch
            G = G.unsqueeze(0).expand(B, -1, -1)

        weighted_diff = G * diff  # Element-wise multiplication
        
        # Frobenius norm squared
        frobenius_norm_sq = torch.sum(weighted_diff ** 2, dim=(1, 2))  # [B]

        # Numerically stable normalization
        gate_sum = torch.sum(G, dim=(1, 2))  # [B]
        
        # Robust normalization with regularization to prevent numerical instability
        # Add adaptive regularization based on tensor magnitude for stability
        gate_reg = torch.clamp(gate_sum.mean() * 1e-3, min=1e-4, max=1e-2)  # Adaptive regularization
        gate_sum_stable = gate_sum + gate_reg  # Regularized sum
        
        # to prevent overflow/underflow
        frobenius_norm_sq_fp64 = frobenius_norm_sq.double()
        gate_sum_stable_fp64 = gate_sum_stable.double()
        
        # Compute normalized loss with numerical stability
        normalized_loss_fp64 = frobenius_norm_sq_fp64 / gate_sum_stable_fp64
        trd_loss = torch.mean(normalized_loss_fp64.float())  # Convert back to original precision

        return trd_loss

    def _pool_teacher_to_student_grid(self, teacher_proj, teacher_shape, target_shape):
        """Pool teacher features to match student spatial grid without collapsing spatial structure."""
        return align_teacher_to_student_grid(teacher_proj, teacher_shape, target_shape)

    def _build_action_gate(self, T, N, B, actions=None, device=None):
        """
        Build action/motion gate that upweights near-time pairs and high-motion frames.

        Args:
            T: Number of time steps
            N: Number of spatial tokens
            B: Batch size
            actions: Optional [B, T, action_dim] action commands
            device: Device to create tensors on (inferred from actions if not provided)

        Returns:
            G: [B, T*N, T*N] or [T*N, T*N] gate matrix
        """
        TN = T * N
        if device is None:
            device = actions.device if actions is not None else 'cpu'

        if self.gate_type == 'temporal':
            t_indices = torch.arange(T, device=device).view(T, 1).expand(T, N).reshape(-1)
            t_dist = torch.abs(t_indices.unsqueeze(0) - t_indices.unsqueeze(1))
            sigma = max(T / 4.0, 1.0)
            return torch.exp(-t_dist.float() ** 2 / (2 * sigma ** 2))

        if self.gate_type == 'action':
            if actions is None or actions.numel() == 0:
                raise ValueError('Action gating selected but no action tensor was provided.')

            actions = actions.to(device)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)

            B_act, T_act, _ = actions.shape
            if T_act == 0:
                raise ValueError('Action gating received zero-length action sequences.')

            action_mag = torch.norm(actions, p=2, dim=-1)
            if T_act > 1:
                min_vals = action_mag.min(dim=1, keepdim=True)[0]
                max_vals = action_mag.max(dim=1, keepdim=True)[0]
                denom = (max_vals - min_vals).clamp(min=1e-6)
                action_mag = (action_mag - min_vals) / denom
            else:
                action_mag = torch.zeros_like(action_mag)

            action_weight = action_mag.unsqueeze(2).expand(-1, -1, N).reshape(B_act, -1)
            G = action_weight.unsqueeze(2) * action_weight.unsqueeze(1)

            if B_act != B:
                if B_act == 1:
                    G = G.expand(B, -1, -1)
                else:
                    raise ValueError('Action batch dimension does not match training batch.')

            return G

        # Default fallback if unknown gate_type
        return torch.ones(TN, TN, device=device)

    def _select_sparse_tokens(self, features, num_tokens):
        """
        Select top-k tokens based on feature magnitude (motion proxy).

        Args:
            features: [B, T*N, d'] - Token features
            num_tokens: Number of tokens to select

        Returns:
            mask: [B, T*N] - Boolean mask for selected tokens
        """
        B, TN, _ = features.shape

        # Compute token importance (L2 norm)
        token_importance = torch.norm(features, p=2, dim=-1)  # [B, T*N]

        # Select top-k tokens
        _, top_indices = torch.topk(token_importance, k=num_tokens, dim=1)  # [B, num_tokens]

        # Create mask
        mask = torch.zeros(B, TN, dtype=torch.bool, device=features.device)
        mask.scatter_(1, top_indices, True)

        return mask

    def _mask_relations(self, relations, token_mask):
        """
        Mask relation matrix to only include selected tokens.

        Args:
            relations: [B, T*N, T*N] - Relation matrix
            token_mask: [B, T*N] - Boolean mask

        Returns:
            masked_relations: [B, T*N, T*N] - Masked relations
        """
        # Apply mask to both dimensions
        mask_2d = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)  # [B, T*N, T*N]
        return relations * mask_2d.float()


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ActionEmbedder(nn.Module):
    """
    Embeds action xy into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        hsize = hidden_size//3
        self.x_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.y_emb = TimestepEmbedder(hsize, frequency_embedding_size)
        self.angle_emb = TimestepEmbedder(hidden_size -2*hsize, frequency_embedding_size)

    def forward(self, xya):
        return torch.cat([self.x_emb(xya[...,0:1]), self.y_emb(xya[...,1:2]), self.angle_emb(xya[...,2:3])], dim=-1)

#################################################################################
#                                 Core CDiT Model                                #
#################################################################################

class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, x_cond):
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=modulate(self.norm2(x), shift_ca_x, scale_ca_x), key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(self.context_size + 1, num_patches, hidden_size), requires_grad=True) # for context and for predicted frame
        self.blocks = nn.ModuleList([CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Initialize action embedding:
        nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
            
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, x_cond, rel_t, return_features=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        return_features: If True, return intermediate features for AC-REPA
        """
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) + self.pos_embed[:self.context_size]  # (N, T, D), where T = H * W / patch_size ** 2.flatten(1, 2)
        x_cond = x_cond.flatten(1, 2)
        t = self.t_embedder(t[..., None])
        y = self.y_embedder(y) 
        time_emb = self.time_embedder(rel_t[..., None])
        c = t + time_emb + y # if training on unlabeled data, dont add y.

        # Process transformer blocks and extract features from contextually-rich intermediate layer (after AdaLN)
        features = None  # Initialize for safety
        # Extract from ~2/3 through the model for optimal contextual processing across all model sizes
        target_layer = max(0, int(len(self.blocks) * 0.67) - 1)  # ~67% through, minimum layer 0
        # Note: For debugging, target layers are: XL(28)→18, L(24)→15, B/S(12)→7
        
        for i, block in enumerate(self.blocks):
            x = block(x, c, x_cond)
            
            # Extract features from target layer after AdaLN contextual processing
            if return_features and i == target_layer:
                features = x  # [N, num_patches, hidden_size] - contextually influenced by past frames & actions
        
        # Save pre-final-layer state for fallback
        pre_final_x = x if return_features else None
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        if return_features:
            # Safety check: ensure features is never None
            if features is None:
                features = pre_final_x  # Fallback to pre-final layer if target layer extraction failed
            return x, features
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                  #
#################################################################################

def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2, 
    'CDiT-L/2':  CDiT_L_2, 
    'CDiT-B/2':  CDiT_B_2, 
    'CDiT-S/2':  CDiT_S_2
}
