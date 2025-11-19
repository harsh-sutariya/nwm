# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

from isolated_nwm_infer import model_forward_wrapper
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import os
import matplotlib.pyplot as plt 
import yaml
from glob import glob


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL

from distributed import init_distributed, init_single_gpu
from models import CDiT_models, ACRepaLoss
from diffusion import create_diffusion
from datasets import TrainingDataset
from misc import transform

# Teacher model for AC-REPA
from transformers import AutoModel, AutoImageProcessor

# W&B integration
import wandb

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler()]
        )
    logger = logging.getLogger(__name__)
    return logger

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup training mode (distributed or single GPU):
    if args.single_gpu:
        world_size, rank, device, is_distributed = init_single_gpu()
        print(f"Starting single GPU training on device {device}")
    else:
        world_size, rank, device, is_distributed = init_distributed()
        print(f"Starting distributed training: rank={rank}, world_size={world_size}")
    
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    
    # Initialize W&B logging
    wandb_run, run_id = init_wandb(args, config, rank, world_size, is_distributed)
    
    # Save run_id for worker nodes if we're the primary node
    if is_distributed and rank == 0 and run_id:
        save_run_id_for_workers(run_id, config)
    
    # Setup an experiment folder:
    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    latent_size = config['image_size'] // 8

    assert config['image_size'] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    num_cond = config['context_size']
    model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4).to(device)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Initialize teacher model and AC-REPA loss if enabled
    teacher_model = None
    teacher_processor = None
    ac_repa_loss_fn = None
    
    if config.get('use_ac_repa', False):
        logger.info("Initializing AC-REPA training objective...")
        ac_repa_config = config.get('ac_repa', {})
        
        # Load teacher model
        teacher_model_name = ac_repa_config.get('teacher_model', 'MCG-NJU/videomae-base-finetuned-kinetics')
        logger.info(f"Loading teacher model: {teacher_model_name}")
        teacher_model = AutoModel.from_pretrained(teacher_model_name).to(device)
        teacher_processor = AutoImageProcessor.from_pretrained(teacher_model_name)
        teacher_model.eval()
        requires_grad(teacher_model, False)
        
        # Initialize AC-REPA loss with dynamic student dimension
        # Automatically infer student_dim from the chosen CDiT model
        actual_student_dim = model.blocks[0].norm1.normalized_shape[0]
        num_layers = len(model.blocks)
        target_extraction_layer = max(0, int(num_layers * 0.67) - 1) + 1  # +1 for human-readable (1-indexed)
        logger.info(f"Auto-detected student model: {num_layers} layers, hidden_dim={actual_student_dim}, extracting from layer {target_extraction_layer}")
        
        # Override config student_dim with actual model dimension for robustness
        if 'student_dim' in ac_repa_config and ac_repa_config['student_dim'] != actual_student_dim:
            logger.warning(f"Config student_dim ({ac_repa_config['student_dim']}) doesn't match model dimension ({actual_student_dim}). Using model dimension.")
        
        ac_repa_loss_fn = ACRepaLoss(
            student_dim=actual_student_dim,  # Use actual model dimension
            teacher_dim=ac_repa_config.get('teacher_dim', 768),
            proj_dim=ac_repa_config.get('proj_dim', 512),
            fa_pool_type=ac_repa_config.get('fa_pool_type', 'mean'),
            trd_gate_type=ac_repa_config.get('trd_gate_type', 'temporal'),
            lambda_fa=ac_repa_config.get('lambda_fa', 1.0),
            lambda_trd=ac_repa_config.get('lambda_trd', 1.0),
            use_sparse_trd=ac_repa_config.get('use_sparse_trd', True),
            sparse_ratio=ac_repa_config.get('sparse_ratio', 0.25)
        ).to(device)
        
        logger.info(f"AC-REPA loss initialized with lambda_fa={ac_repa_config.get('lambda_fa', 1.0)}, lambda_trd={ac_repa_config.get('lambda_trd', 1.0)}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = float(config.get('lr', 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)
    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # Checkpoint loading logic - controlled by command line arguments
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    start_epoch = 0
    train_steps = 0
    
    # Check if resuming is allowed (from config)
    allow_resume = config.get('allow_resume', False)  # Default: False (always start fresh)
    if allow_resume:
        print('Resume is ENABLED - will attempt to load from checkpoints')
        if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
            if os.path.isfile(latest_path) and config.get('from_checkpoint', 0):
                raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")
            latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
            print("Loading model from ", latest_path)
            latest_checkpoint = torch.load(latest_path, map_location=f'cuda:{device}', weights_only=False) 

            if "model" in latest_checkpoint:
                model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['model'].items()}
                res = model.load_state_dict(model_ckp, strict=True)
                print("Loading model weights", res)

                model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
                res = ema.load_state_dict(model_ckp, strict=True)
                print("Loading EMA model weights", res)
            else:
                update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

            if "opt" in latest_checkpoint:
                opt_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['opt'].items()}
                opt.load_state_dict(opt_ckp)
                print("Loading optimizer params")
            
            if "epoch" in latest_checkpoint:
                start_epoch = latest_checkpoint['epoch'] + 1
            
            if "train_steps" in latest_checkpoint:
                train_steps = latest_checkpoint["train_steps"]
            
            if "scaler" in latest_checkpoint:
                scaler.load_state_dict(latest_checkpoint["scaler"])
        else:
            print('No checkpoint found - starting fresh')
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        # Fresh start mode (default behavior)
        print('FRESH START MODE - checkpoint resuming is DISABLED')
        print('Existing checkpoints in', checkpoint_dir, 'will be ignored')
        if os.path.isfile(latest_path):
            print(f'Found existing checkpoint {latest_path} but ignoring due to fresh start mode')
        
        # Always initialize EMA with synced weights for fresh training
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        
    # ~40% speedup but might leads to worse performance depending on pytorch version
    if args.torch_compile:
        model = torch.compile(model)
    
    # Wrap model for distributed training if needed
    if is_distributed:
        model = DDP(model, device_ids=[device])
    
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Log model info to W&B
    if wandb_run and rank == 0:
        wandb_run.log({
            "model/parameters": sum(p.numel() for p in model.parameters()),
            "model/trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "training/start_epoch": start_epoch,
            "training/start_steps": train_steps,
        })

    # Quick debug mode: reduce parameters for faster testing
    quick_debug = config.get('quick_debug', False)
    if quick_debug:
        logger.info("=" * 60)
        logger.info("QUICK DEBUG MODE ENABLED")
        logger.info("=" * 60)
        logger.info("Reducing parameters for fast testing:")
        logger.info(f"  Original batch_size: {config['batch_size']}")
        logger.info(f"  Original num_workers: {config['num_workers']}")
        config['batch_size'] = min(config['batch_size'], 4)  # Reduce batch size
        config['num_workers'] = 1  # Use 1 worker to avoid warnings and speed up
        logger.info(f"  Debug batch_size: {config['batch_size']}")
        logger.info(f"  Debug num_workers: {config['num_workers']}")
        logger.info("Will exit after 3 training steps")
        logger.info("=" * 60)

    train_dataset = []
    test_dataset = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    goals_per_obs = int(data_config["goals_per_obs"])
                    if data_split_type == 'test':
                        goals_per_obs = 4 # standardize testing
                    
                    if "distance" in data_config:
                        min_dist_cat=data_config["distance"]["min_dist_cat"]
                        max_dist_cat=data_config["distance"]["max_dist_cat"]
                    else:
                        min_dist_cat=config["distance"]["min_dist_cat"]
                        max_dist_cat=config["distance"]["max_dist_cat"]

                    if "len_traj_pred" in data_config:
                        len_traj_pred=data_config["len_traj_pred"]
                    else:
                        len_traj_pred=config["len_traj_pred"]

                    dataset = TrainingDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        min_dist_cat=min_dist_cat,
                        max_dist_cat=max_dist_cat,
                        len_traj_pred=len_traj_pred,
                        context_size=config["context_size"],
                        normalize=config["normalize"],
                        goals_per_obs=goals_per_obs,
                        transform=transform,
                        predefined_index=None,
                        traj_stride=1,
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        test_dataset.append(dataset)
                    print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    # combine all the datasets from different robots
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    # Setup data loading with or without distributed sampling
    if is_distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_diffusion_loss = 0
    running_fa_loss = 0
    running_trd_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    epoch = start_epoch  
    logged_batch_shapes = False
    first_step = None  # Track first step for quick_debug early exit
    
    for epoch in range(start_epoch, args.epochs):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y, rel_t in loader:
            # Track first step for quick_debug
            if first_step is None:
                first_step = train_steps
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            
            # OPTIMIZATION: Extract teacher features OUTSIDE autocast for better performance
            # Teacher model is in eval mode and doesn't need gradients, so process separately
            teacher_features = None
            if config.get('use_ac_repa', False) and teacher_model is not None:
                with torch.no_grad():
                    # Use separate autocast for teacher to optimize memory and speed
                    with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                        # x is [B, T, C, H, W] in range [-1, 1]
                        # Only process goal frames to match student processing
                        B, T, C, H, W = x.shape
                        num_goals = T - num_cond
                        x_goals = x[:, num_cond:]  # [B, num_goals, C, H, W] - only goal frames
                        
                        # VideoMAE expects exactly 16 frames
                        videomae_frames = 16
                        
                        # OPTIMIZATION: Convert entire batch from [-1, 1] to [0, 1] for VideoMAE
                        x_normalized = x_goals * 0.5 + 0.5  # [B, num_goals, C, H, W]
                        
                        # Determine teacher sampling indices once to preserve temporal ordering
                        if num_goals == videomae_frames:
                            sample_indices = torch.arange(videomae_frames, device=x_normalized.device)
                        else:
                            base_positions = torch.linspace(0, max(num_goals - 1, 1), steps=videomae_frames, device=x_normalized.device)
                            sample_indices = torch.clamp(base_positions.round().long(), 0, num_goals - 1)

                        x_sampled = x_normalized[:, sample_indices]  # [B, 16, C, H, W]

                        # OPTIMIZATION: More efficient list creation using tensor slicing
                        # Convert to list format expected by processor (processor needs PIL/numpy)
                        # Use list comprehension but with direct tensor slicing (faster than nested loops)
                        batch_frames_list = []
                        for b in range(B):
                            # Direct tensor slicing is faster than nested list comprehension
                            batch_frames_list.append([x_sampled[b, t] for t in range(videomae_frames)])

                        batch_inputs = teacher_processor(batch_frames_list, return_tensors="pt", do_rescale=False)
                        batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()}

                        batch_outputs = teacher_model(**batch_inputs, output_hidden_states=True)
                        batch_teacher_features = batch_outputs.hidden_states[-1][:, 1:, :]  # [B, frames*patches, D]
                        D = batch_teacher_features.shape[-1]

                        total_tokens = batch_teacher_features.shape[1]

                        # Cache teacher config to avoid repeated attribute access
                        teacher_cfg = getattr(teacher_model, "config", None)
                        teacher_image_size = getattr(teacher_cfg, "image_size", config["image_size"]) if teacher_cfg else config["image_size"]
                        teacher_patch_size = getattr(teacher_cfg, "patch_size", 16) if teacher_cfg else 16
                        if isinstance(teacher_image_size, (tuple, list)):
                            teacher_image_size = teacher_image_size[0]
                        if isinstance(teacher_patch_size, (tuple, list)):
                            teacher_patch_size = teacher_patch_size[-1]
                        spatial_tokens = max(1, (teacher_image_size // teacher_patch_size) ** 2)

                        teacher_frames = max(1, round(total_tokens / spatial_tokens))
                        expected_tokens = teacher_frames * spatial_tokens

                        if expected_tokens != total_tokens:
                            if expected_tokens > total_tokens:
                                pad = expected_tokens - total_tokens
                                pad_tokens = batch_teacher_features[:, -1:, :].expand(-1, pad, -1)
                                batch_teacher_features = torch.cat([batch_teacher_features, pad_tokens], dim=1)
                            else:
                                batch_teacher_features = batch_teacher_features[:, :expected_tokens, :]

                        teacher_tokens = batch_teacher_features.view(B, teacher_frames, spatial_tokens, D)

                        teacher_positions = torch.linspace(
                            0, max(num_goals - 1, 0), steps=teacher_frames, device=teacher_tokens.device
                        ) if teacher_frames > 1 else torch.zeros(1, device=teacher_tokens.device)

                        if num_goals == 1:
                            teacher_features = teacher_tokens[:, :1]
                        elif teacher_frames == num_goals:
                            teacher_features = teacher_tokens
                        else:
                            target_positions = torch.arange(num_goals, device=teacher_tokens.device, dtype=teacher_positions.dtype)
                            right_idx = torch.searchsorted(teacher_positions, target_positions, right=False)
                            right_idx = torch.clamp(right_idx, max=teacher_frames - 1)
                            left_idx = torch.clamp(right_idx - 1, min=0)

                            left_pos = teacher_positions[left_idx]
                            right_pos = teacher_positions[right_idx]
                            denom = (right_pos - left_pos).clamp(min=1e-6)
                            alpha = (target_positions - left_pos) / denom
                            alpha = alpha.view(1, num_goals, 1, 1)

                            teacher_lower = teacher_tokens.index_select(1, left_idx)
                            teacher_upper = teacher_tokens.index_select(1, right_idx)

                            lerp_alpha = alpha.to(dtype=teacher_lower.dtype)
                            teacher_features = torch.lerp(teacher_lower, teacher_upper, lerp_alpha)

                            same_mask = (right_pos == left_pos).view(1, num_goals, 1, 1)
                            if same_mask.any():
                                teacher_features = torch.where(same_mask, teacher_lower, teacher_features)

                        # OPTIMIZATION: Keep teacher features in bfloat16 if using mixed precision
                        # Convert to same dtype as student features for consistency
                        teacher_features = teacher_features.contiguous()
                        if bfloat_enable:
                            teacher_features = teacher_features.to(dtype=torch.bfloat16)
                        else:
                            teacher_features = teacher_features.to(dtype=torch.float32)
            
            # Student training with autocast
            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    B, T = x.shape[:2]
                    x = x.flatten(0,1)
                    x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215)
                    x = x.unflatten(0, (B, T))
                
                num_goals = T - num_cond
                x_start = x[:, num_cond:].flatten(0, 1)
                x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
                y_actions = y.clone()  # Keep original actions for AC-REPA
                goal_actions = y_actions[:, :num_goals]

                if not logged_batch_shapes and rank == 0:
                    logger.info(
                        f"Initial batch: x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}, "
                        f"num_goals={num_goals}, goal_action_shape={tuple(goal_actions.shape)}"
                    )
                    logged_batch_shapes = True

                if goal_actions.shape[1] != num_goals:
                    logger.error(
                        f"Mismatch between num_goals ({num_goals}) and goal action length {goal_actions.shape[1]}"
                    )

                if num_goals <= 0 or goal_actions.shape[1] == 0:
                    logger.error(
                        f"AC-REPA received empty goal slice: num_goals={num_goals}, goal_action_shape={tuple(goal_actions.shape)}"
                    )

                y = y.flatten(0, 1)
                rel_t = rel_t.flatten(0, 1)
                
                t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)
                
                # If using AC-REPA, we need to extract student features
                if config.get('use_ac_repa', False) and ac_repa_loss_fn is not None:
                    # Use a custom diffusion training loss that extracts features
                    # We'll do a single forward pass and extract both diffusion loss and features
                    with torch.no_grad():
                        # Get a sample from the diffusion process for feature extraction
                        noise = torch.randn_like(x_start)
                        x_noisy = diffusion.q_sample(x_start, t, noise=noise)
                    
                    # Single forward pass with feature extraction
                    model_output, student_features = (model.module if is_distributed else model)(
                        x_noisy, t, y, x_cond, rel_t, return_features=True
                    )
                    
                    # Compute diffusion loss manually from model output
                    # When learn_sigma=True, model outputs [noise_pred, var_pred] with 2*C channels
                    # Extract only the noise prediction (first C channels)
                    if model_output.shape[1] == 2 * x_start.shape[1]:
                        # Model outputs both noise and variance - take only noise prediction
                        noise_pred = model_output[:, :x_start.shape[1]]
                    else:
                        # Model outputs only noise prediction
                        noise_pred = model_output
                    
                    target = noise  # Use noise as target since model predicts epsilon
                    diffusion_loss = torch.nn.functional.mse_loss(noise_pred, target)
                    
                    # Reshape student features to [B, T, N, D]
                    # student_features is [B*num_goals, num_patches, hidden_dim]
                    student_features = student_features.reshape(B, num_goals, -1, student_features.shape[-1])
                    
                    # Teacher features only contains goal frames, same as student
                    # No slicing needed - teacher_features is already [B, num_goals, N, D]
                    
                    # Compute AC-REPA loss
                    total_loss, ac_repa_loss_dict = ac_repa_loss_fn(
                        student_features,  # [B, num_goals, N, D] - from goal frames
                        teacher_features,  # [B, num_goals, N, D] - from same goal frames  
                        goal_actions,  # Use goal frame actions [B, num_goals, 3]
                        diffusion_loss
                    )
                    loss = total_loss
                    loss_dict = {"loss": diffusion_loss, **ac_repa_loss_dict}
                else:
                    # Standard diffusion-only training
                    model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
                    loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
                    loss = loss_dict["loss"].mean()

            # opt.zero_grad()
            opt.zero_grad(set_to_none=True)
            if not bfloat_enable:
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                if config.get('grad_clip_val', 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_val'])
                scaler.step(opt)
                scaler.update()
            
            update_ema(ema, model.module if is_distributed else model)

            # Log loss values:
            running_loss += loss.detach().item()
            
            # Track AC-REPA loss components if enabled
            if config.get('use_ac_repa', False) and 'diffusion_loss' in loss_dict:
                running_diffusion_loss += loss_dict['diffusion_loss']
                running_fa_loss += loss_dict['fa_loss']
                running_trd_loss += loss_dict['trd_loss']
            
            log_steps += 1
            train_steps += 1
            
            # Quick debug: exit after 3 steps (goes through all processes)
            if quick_debug and first_step is not None:
                steps_since_first = train_steps - first_step
                if steps_since_first >= 3:
                    logger.info("=" * 60)
                    logger.info(f"QUICK DEBUG MODE: Exiting after {steps_since_first} steps")
                    logger.info("All training processes completed successfully!")
                    logger.info("=" * 60)
                    # Break out of inner loop
                    break
            
            if train_steps % args.log_every == 0 or quick_debug:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Use batch size from config for samples/sec calculation
                samples_per_sec = world_size * config['batch_size'] * steps_per_sec
                
                # CRITICAL: Reduce loss history over all processes for accurate multi-GPU logging
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if is_distributed:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                else:
                    avg_loss = avg_loss.item()
                
                # CRITICAL: Reduce AC-REPA loss components across all GPUs
                avg_diffusion_loss = None
                avg_fa_loss = None
                avg_trd_loss = None
                if config.get('use_ac_repa', False) and running_diffusion_loss > 0:
                    # Create tensors for reduction
                    diffusion_loss_tensor = torch.tensor(running_diffusion_loss / log_steps, device=device)
                    fa_loss_tensor = torch.tensor(running_fa_loss / log_steps, device=device)
                    trd_loss_tensor = torch.tensor(running_trd_loss / log_steps, device=device)
                    
                    if is_distributed:
                        dist.all_reduce(diffusion_loss_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(fa_loss_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(trd_loss_tensor, op=dist.ReduceOp.SUM)
                        avg_diffusion_loss = diffusion_loss_tensor.item() / world_size
                        avg_fa_loss = fa_loss_tensor.item() / world_size
                        avg_trd_loss = trd_loss_tensor.item() / world_size
                    else:
                        avg_diffusion_loss = diffusion_loss_tensor.item()
                        avg_fa_loss = fa_loss_tensor.item()
                        avg_trd_loss = trd_loss_tensor.item()
                
                # Log to both logger and W&B
                log_message = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}"
                logger.info(log_message)
                
                # Log AC-REPA components to console (already reduced)
                if config.get('use_ac_repa', False) and avg_diffusion_loss is not None:
                    logger.info(f"  AC-REPA Components - Diffusion: {avg_diffusion_loss:.4f}, FA: {avg_fa_loss:.4f}, TRD: {avg_trd_loss:.4f}")
                
                # Log metrics to W&B (only rank 0 logs to avoid duplicate entries)
                if wandb_run and rank == 0:
                    wandb_metrics = {
                        "train/loss": avg_loss,
                        "train/steps_per_sec": steps_per_sec,
                        "train/samples_per_sec": samples_per_sec,
                        "train/epoch": epoch,
                        "train/learning_rate": opt.param_groups[0]['lr'],
                        "system/gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                    }
                    
                    # Add AC-REPA loss components (already reduced across GPUs)
                    if config.get('use_ac_repa', False) and avg_diffusion_loss is not None:
                        wandb_metrics["train/diffusion_loss"] = avg_diffusion_loss
                        wandb_metrics["train/fa_loss"] = avg_fa_loss
                        wandb_metrics["train/trd_loss"] = avg_trd_loss
                    
                    wandb_run.log(wandb_metrics, step=train_steps)
                
                # Reset monitoring variables:
                running_loss = 0
                running_diffusion_loss = 0
                running_fa_loss = 0
                running_trd_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint: (skip in quick_debug)
            if train_steps % args.ckpt_every == 0 and train_steps > 0 and not quick_debug:
                if rank == 0:
                    checkpoint = {
                        "model": (model.module if is_distributed else model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    if bfloat_enable:
                        checkpoint.update({"scaler": scaler.state_dict()})
                    checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
                    torch.save(checkpoint, checkpoint_path)
                    if train_steps % (10*args.ckpt_every) == 0 and train_steps > 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                        torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Log checkpoint to W&B (only if explicitly enabled)
                    if wandb_run and args.wandb_save_checkpoints:
                        # Save model artifact
                        artifact = wandb.Artifact(
                            name=f"model-checkpoint-{train_steps}",
                            type="model",
                            description=f"Model checkpoint at step {train_steps}"
                        )
                        artifact.add_file(checkpoint_path)
                        wandb_run.log_artifact(artifact)
                        logger.info(f"Uploaded checkpoint to W&B: model-checkpoint-{train_steps}")
                    elif wandb_run:
                        logger.info(f"Checkpoint saved locally only (W&B checkpoint saving disabled)")
            
            # Evaluation: skip during quick_debug (too slow)
            if train_steps % args.eval_every == 0 and train_steps > 0 and not quick_debug:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, rank, config["batch_size"], config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond, is_distributed, world_size)
                if is_distributed:
                    dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}")
                
                # Log evaluation metrics to W&B
                if wandb_run and rank == 0:
                    eval_metrics = {
                        "eval/perceptual_loss": sim_score.item() if torch.is_tensor(sim_score) else sim_score,
                        "eval/eval_time": eval_time,
                    }
                    wandb_run.log(eval_metrics, step=train_steps)
                    
                    # Log evaluation images as W&B artifacts (only if explicitly enabled)
                    if args.wandb_save_eval_images and os.path.exists(save_dir):
                        eval_artifact = wandb.Artifact(
                            name=f"eval-images-{train_steps}",
                            type="evaluation",
                            description=f"Evaluation images at step {train_steps}"
                        )
                        for img_file in glob(f"{save_dir}/*.png"):
                            eval_artifact.add_file(img_file)
                        if len(glob(f"{save_dir}/*.png")) > 0:
                            wandb_run.log_artifact(eval_artifact)
                            logger.info(f"Uploaded evaluation images to W&B: eval-images-{train_steps}")
                    elif os.path.exists(save_dir):
                        logger.info(f"Evaluation images saved locally only (W&B image artifact saving disabled)")
        
        # Break out of epoch loop if quick_debug completed
        if quick_debug and first_step is not None and (train_steps - first_step) >= 3:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    # Quick debug: run a quick validation at the end
    if quick_debug:
        logger.info("=" * 60)
        logger.info("QUICK DEBUG MODE: Running quick validation...")
        logger.info("=" * 60)
        eval_start_time = time()
        save_dir = os.path.join(experiment_dir, "quick_debug_eval")
        # Use smaller batch size for quick debug eval
        eval_batch_size = min(config["batch_size"], 4)
        sim_score = evaluate(ema, tokenizer, diffusion, test_dataset, rank, eval_batch_size, config["num_workers"], latent_size, device, save_dir, args.global_seed, bfloat_enable, num_cond, is_distributed, world_size)
        if is_distributed:
            dist.barrier()
        eval_end_time = time()
        eval_time = eval_end_time - eval_start_time
        logger.info(f"Quick Debug Eval - Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}")
        logger.info("=" * 60)

    logger.info("Done!")
    
    # Finish W&B run and cleanup
    if wandb_run:
        # Log final metrics
        if rank == 0:
            wandb_run.log({
                "training/final_epoch": epoch,
                "training/final_steps": train_steps,
                "training/status": "completed"
            })
        wandb.finish()
    
    # Clean up run_id file if we're the primary node
    if is_distributed and rank == 0:
        cleanup_run_id_file(config)
    
    cleanup()


@torch.no_grad
def evaluate(model, vae, diffusion, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond, is_distributed, world_size):
    if is_distributed:
        sampler = DistributedSampler(
            test_dataloaders,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize DreamSim model with distributed-safe loading
    if is_distributed:
        # Only rank 0 downloads the model to avoid race conditions
        if rank == 0:
            from dreamsim import dreamsim
            eval_model, _ = dreamsim(pretrained=True)
        
        # Wait for rank 0 to finish downloading
        dist.barrier()
        
        # Now all ranks can safely load the model
        if rank != 0:
            from dreamsim import dreamsim
            eval_model, _ = dreamsim(pretrained=True)
    else:
        from dreamsim import dreamsim
        eval_model, _ = dreamsim(pretrained=True)
    
    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    for x, y, rel_t in loader:
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            samples = model_forward_wrapper((model, diffusion, vae), x, y, num_timesteps=None, latent_size=latent_size, device=device, num_cond=num_cond, num_goals=num_goals, rel_t=rel_t)
            x_start_pixels = x[:, num_cond:].flatten(0, 1)
            x_cond_pixels = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
            samples = samples * 0.5 + 0.5
            x_start_pixels = x_start_pixels * 0.5 + 0.5
            x_cond_pixels = x_cond_pixels * 0.5 + 0.5
            res = eval_model(x_start_pixels, samples)
            score += res.sum()
            n_samples += len(res)
        break
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(samples.shape[0], 10)):
            _, ax = plt.subplots(1,3,dpi=256)
            ax[0].imshow((x_cond_pixels[i, -1].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[1].imshow((x_start_pixels[i].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            ax[2].imshow((samples[i].permute(1,2,0).cpu().float().numpy()*255).astype('uint8'))
            plt.savefig(f'{save_dir}/{i}.png')
            plt.close()

    if is_distributed:
        dist.all_reduce(score)
        dist.all_reduce(n_samples)
    sim_score = score/n_samples
    return sim_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    parser.add_argument("--single-gpu", type=int, default=1)
    
    # W&B arguments
    parser.add_argument("--wandb-project", type=str, default="nwm", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=[], help="W&B tags for the run")
    parser.add_argument("--wandb-disabled", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-save-checkpoints", action="store_true", help="Save model checkpoints to W&B as artifacts")
    parser.add_argument("--wandb-save-eval-images", action="store_true", help="Save evaluation images to W&B as artifacts")
    
    return parser

def init_wandb(args, config, rank, world_size, is_distributed):
    """
    Initialize Weights & Biases logging.
    
    Default behavior: ALWAYS FRESH (no resuming) unless --allow-resume is specified.
    For single GPU: Standard wandb.init()
    For distributed: Use shared mode with primary/worker node distinction
    """
    # Check if W&B is disabled via command line flag
    if args.wandb_disabled:
        print("W&B logging disabled via --wandb-disabled flag")
        return None, None
    
    # Check if quick_debug is enabled - disable W&B for quick debugging
    if config.get('quick_debug', False):
        if rank == 0:
            print("W&B logging disabled because quick_debug mode is enabled")
        return None, None
    
    # Enable wandb service for better reliability in distributed settings
    if hasattr(wandb, 'require'):
        wandb.require("service")
    
    # Prepare wandb config by merging args and config
    # Override config values with command line arguments if provided
    wandb_config = {
        **config,      # Configuration from YAML files
        **vars(args),  # Command line arguments (these take precedence)
    }
    
    # Use command line arguments for W&B settings, fallback to config file
    project = args.wandb_project or config.get('wandb_project', 'nwm-training')
    entity = args.wandb_entity or config.get('wandb_entity', None)
    tags = args.wandb_tags + config.get('wandb_tags', [])
    
    # Determine if we should create fresh runs or allow resuming (from config)
    allow_resume = config.get('allow_resume', False)  # Default: False (always start fresh)
    if not allow_resume:
        # Add timestamp to ensure unique run names (no resuming)
        import time
        timestamp = int(time.time())
        fresh_suffix = f"_{timestamp}"
        fresh_tag = "fresh-start"
        fresh_note_suffix = " - FRESH START"
        print("W&B: Creating FRESH run (no resuming)")
    else:
        fresh_suffix = ""
        fresh_tag = "resume-enabled"
        fresh_note_suffix = " - Resume Enabled"
        print("W&B: Resume mode enabled")
    
    if is_distributed:
        # Distributed training: Use shared mode
        # BEST PRACTICE: Use distributed broadcast for run_id sharing (most reliable)
        import os
        import torch.distributed as dist
        
        base_run_name = args.wandb_run_name or config.get('wandb_run_name') or f"{config['run_name']}_distributed"
        run_name = f"{base_run_name}{fresh_suffix}"
        
        # Check if run_id is pre-set via environment variable (e.g., from external launcher)
        run_id_from_env = os.environ.get('WANDB_RUN_ID', None)
        
        if rank == 0:
            # Primary node: Create new run
            try:
                run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    id=run_id_from_env,  # Use env var if provided, otherwise create new
                    config=wandb_config,
                    settings=wandb.Settings(
                        mode="shared",
                        x_primary=True,
                        x_label=f"rank_{rank}",
                        x_stats_gpu_device_ids=list(range(torch.cuda.device_count())),
                    ),
                    tags=["distributed", "training", fresh_tag] + tags,
                    notes=f"Distributed training with {world_size} processes{fresh_note_suffix}",
                )
                run_id = run.id
                
                # Validate run_id is not empty
                if not run_id or len(run_id.strip()) == 0:
                    raise ValueError(f"W&B returned empty run_id on rank {rank}")
                
                print(f"W&B initialized on primary node (rank {rank}) with run ID: {run_id}")
            except Exception as e:
                print(f"ERROR: W&B initialization failed on rank {rank}: {e}")
                print(f"Continuing without W&B logging on rank {rank}")
                run_id = None
                run = None
        
        # BEST PRACTICE: Use distributed broadcast to share run_id (more reliable than file)
        # Convert run_id to tensor for broadcasting (use CPU tensor for simplicity)
        max_len = 64  # wandb run IDs are typically short, use 64 chars max
        if rank == 0:
            # Encode run_id as bytes, then convert to tensor
            # If run_id is None (W&B init failed), use empty string
            run_id_str = run_id if run_id else ''
            run_id_bytes = run_id_str.encode('utf-8')
            # Pad to fixed size
            run_id_bytes = run_id_bytes[:max_len].ljust(max_len, b'\0')
            run_id_tensor = torch.tensor(list(run_id_bytes), dtype=torch.uint8)
        else:
            run_id_tensor = torch.zeros(max_len, dtype=torch.uint8)
        
        # Broadcast run_id from rank 0 to all ranks (CPU tensor)
        dist.broadcast(run_id_tensor, src=0)
        
        # Decode run_id on all ranks
        run_id_bytes = bytes(run_id_tensor.numpy()).rstrip(b'\0')
        run_id = run_id_bytes.decode('utf-8') if len(run_id_bytes) > 0 else None
        
        # Validate run_id
        if not run_id or len(run_id.strip()) == 0:
            if rank == 0:
                # Rank 0 already failed, return None
                return None, None
            else:
                print(f"ERROR: Invalid run_id received on rank {rank} after broadcast (rank 0 W&B init failed)")
                print(f"Continuing without W&B logging on rank {rank}")
                return None, None
        
        # All workers (including rank 0) now have run_id
        if rank == 0:
            return run, run_id
        else:
            # Worker nodes: Join existing run
            try:
                run = wandb.init(
                    project=project,
                    entity=entity,
                    id=run_id,
                    settings=wandb.Settings(
                        mode="shared",
                        x_primary=False,
                        x_label=f"rank_{rank}",
                        x_update_finish_state=False,
                    ),
                )
                print(f"W&B initialized on worker node (rank {rank}) with shared run ID: {run_id}")
                return run, run_id
            except Exception as e:
                print(f"ERROR: W&B initialization failed on worker rank {rank}: {e}")
                print(f"Continuing without W&B logging on rank {rank}")
                return None, None
    else:
        # Single GPU training: Standard initialization
        base_run_name = args.wandb_run_name or config.get('wandb_run_name') or f"{config['run_name']}_single_gpu"
        run_name = f"{base_run_name}{fresh_suffix}"
        
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=wandb_config,
            tags=["single-gpu", "training", fresh_tag] + tags,
            notes=f"Single GPU training{fresh_note_suffix}",
        )
        print(f"W&B initialized for single GPU training with run ID: {run.id}")
        return run, run.id

def save_run_id_for_workers(run_id, config):
    """Save run_id to a file for worker nodes to access"""
    run_id_file = f"/tmp/wandb_run_id_{config['run_name']}.txt"
    with open(run_id_file, 'w') as f:
        f.write(run_id)

def cleanup_run_id_file(config):
    """Clean up the run_id file"""
    run_id_file = f"/tmp/wandb_run_id_{config['run_name']}.txt"
    if os.path.exists(run_id_file):
        os.remove(run_id_file)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)