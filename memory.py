# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# SAM2-Style Memory Components for Navigation World Models
# References:
# SAM2Act: https://arxiv.org/html/2501.18564v1
# SAM2: https://github.com/facebookresearch/sam2
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple


class TemporalPositionalEncoding(nn.Module):
    """
    1D sinusoidal positional encoding for temporal sequences in memory
    """
    def __init__(self, hidden_size: int, max_len: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, hidden_size]
    
    def forward(self, x: torch.Tensor, temporal_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            temporal_ids: [batch_size, seq_len] - temporal indices for each token
        """
        batch_size, seq_len = temporal_ids.shape
        pos_encoding = self.pe[0, temporal_ids.long()]  # [batch_size, seq_len, hidden_size]
        return x + pos_encoding


class FIFOMemoryBank(nn.Module):
    """
    FIFO Memory Bank for storing temporal latent states with metadata
    Inspired by SAM2's memory mechanism with gradient-aware storage
    """
    def __init__(self, hidden_size: int, max_memory_size: int = 512, patch_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_memory_size = max_memory_size
        self.patch_size = patch_size
        
        # Memory storage (these are buffers - no gradients stored here)
        self.register_buffer('memory_states', torch.zeros(max_memory_size, hidden_size))
        self.register_buffer('memory_timestamps', torch.zeros(max_memory_size, dtype=torch.long))
        self.register_buffer('memory_valid', torch.zeros(max_memory_size, dtype=torch.bool))
        
        # FIFO pointer
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('memory_count', torch.zeros(1, dtype=torch.long))
        
        # Temporal positional encoding
        self.temporal_pos_enc = TemporalPositionalEncoding(hidden_size, max_memory_size)
        
    def add_memory(self, states: torch.Tensor, timestamp: torch.Tensor):
        """
        Add new states to memory bank using FIFO policy
        Args:
            states: [batch_size, num_patches, hidden_size] - encoded latent states
            timestamp: [batch_size] or scalar - frame timestamps
        """
        if states.dim() == 3:
            # Average pool across patches for memory storage
            states = states.mean(dim=1)  # [batch_size, hidden_size]
        
        batch_size = states.shape[0]
        
        # Handle timestamp input
        if not isinstance(timestamp, torch.Tensor):
            timestamp = torch.full((batch_size,), timestamp, device=states.device)
        elif timestamp.dim() == 0:
            timestamp = timestamp.unsqueeze(0).repeat(batch_size)
        
        # Detach states to prevent memory accumulation (SAM2 approach)
        states = states.detach()
        
        for b in range(batch_size):
            ptr = self.memory_ptr[0].item()
            
            # Store state and metadata
            self.memory_states[ptr] = states[b]
            self.memory_timestamps[ptr] = timestamp[b]
            self.memory_valid[ptr] = True
            
            # Update FIFO pointer
            self.memory_ptr[0] = (ptr + 1) % self.max_memory_size
            self.memory_count[0] = min(self.memory_count[0] + 1, self.max_memory_size)
    
    def get_memory_context(self, current_timestamp: torch.Tensor, 
                          max_context: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant memory context based on temporal distance
        Args:
            current_timestamp: [batch_size] - current frame timestamp
            max_context: maximum number of memory entries to return
        Returns:
            memory_features: [batch_size, memory_len, hidden_size] - ready for gradient flow
            memory_features: [batch_size, memory_len, hidden_size] - same as key (for compatibility)
        """
        if not isinstance(current_timestamp, torch.Tensor):
            current_timestamp = torch.tensor([current_timestamp], device=self.memory_states.device)
        elif current_timestamp.dim() == 0:
            current_timestamp = current_timestamp.unsqueeze(0)
            
        batch_size = current_timestamp.shape[0]
        memory_len = min(self.memory_count[0].item(), max_context)
        
        if memory_len == 0:
            # Return empty memory
            empty_memory = torch.zeros(batch_size, 1, self.hidden_size, device=self.memory_states.device)
            return empty_memory, empty_memory
        
        # Get the most recent valid memory entries
        valid_indices = torch.where(self.memory_valid[:self.memory_count[0].item()])[0]
        if len(valid_indices) == 0:
            empty_memory = torch.zeros(batch_size, 1, self.hidden_size, device=self.memory_states.device)
            return empty_memory, empty_memory
            
        # Take the most recent entries (up to max_context)
        recent_indices = valid_indices[-memory_len:]
        valid_memory = self.memory_states[recent_indices]  # [memory_len, hidden_size]
        
        # Add temporal positional encoding (this allows gradients through pos encoding)
        temporal_ids = torch.arange(memory_len, device=self.memory_states.device).unsqueeze(0).repeat(batch_size, 1)
        memory_features = self.temporal_pos_enc(
            valid_memory.unsqueeze(0).repeat(batch_size, 1, 1), 
            temporal_ids
        )
        
        return memory_features, memory_features  # Return same for key and value
    
    def clear_memory(self):
        """Clear all memory contents"""
        self.memory_valid.fill_(False)
        self.memory_ptr.fill_(0)
        self.memory_count.fill_(0)


class MemoryAttention(nn.Module):
    """
    Cross-attention mechanism for attending to memory bank
    Based on SAM2's memory attention design
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, memory_features: torch.Tensor, 
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, query_len, hidden_size] - current frame features
            memory_features: [batch_size, memory_len, hidden_size] - memory bank features
            memory_mask: [batch_size, memory_len] - attention mask for valid memory entries
        """
        batch_size, query_len, _ = query.shape
        memory_len = memory_features.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [batch_size, query_len, hidden_size]
        K = self.k_proj(memory_features)  # [batch_size, memory_len, hidden_size]  
        V = self.v_proj(memory_features)  # [batch_size, memory_len, hidden_size]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch_size, num_heads, query_len, memory_len]
        
        # Apply memory mask if provided
        if memory_mask is not None:
            # Expand mask for all heads
            mask = memory_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, memory_len]
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        # [batch_size, num_heads, query_len, head_dim]
        
        # Concatenate heads and project output
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.hidden_size)
        
        output = self.out_proj(attended_values)
        return output


class MemoryAugmentedAttention(nn.Module):
    """
    Memory-augmented cross-attention that combines context attention with memory attention
    Inspired by SAM2's approach for temporal context integration
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Context cross-attention (existing functionality)
        self.context_attention = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **kwargs
        )
        
        # Memory attention
        self.memory_attention = MemoryAttention(hidden_size, num_heads)
        
        # Output projection to combine context and memory
        self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Gating mechanism to balance context vs memory
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, query, context_key, context_value, memory_key=None, memory_value=None):
        """
        Args:
            query: [batch_size, query_len, hidden_size] - current frame features
            context_key, context_value: Context features (from conditioning images)
            memory_key, memory_value: Memory features from memory bank
        """
        batch_size, query_len, hidden_size = query.shape
        
        # Context attention (original cross-attention)
        context_out, _ = self.context_attention(
            query=query, key=context_key, value=context_value, need_weights=False
        )
        
        # Memory attention (if memory is available)
        if memory_key is not None and memory_value is not None:
            # Compute memory mask based on similarity
            memory_mask = torch.ones(batch_size, memory_key.shape[1], dtype=torch.bool, device=query.device)
            memory_out = self.memory_attention(query, memory_key, memory_mask)
        else:
            # No memory available, use zeros
            memory_out = torch.zeros_like(context_out)
        
        # Apply gating
        context_gate = self.context_gate(query)
        memory_gate = self.memory_gate(query)
        
        # Combine outputs
        gated_context = context_gate * context_out
        gated_memory = memory_gate * memory_out
        
        # Final projection
        combined = torch.cat([gated_context, gated_memory], dim=-1)
        output = self.output_proj(combined)
        
        return output 