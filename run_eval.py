#!/usr/bin/env python3
"""
Unified evaluation script that runs both time and rollout evaluation
for a single checkpoint and logs everything to one W&B run.
"""

import argparse
import json
import os
import subprocess
import sys
import torch
import wandb
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Run full evaluation (time + rollout) for a checkpoint')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    
    # Optional arguments
    parser.add_argument('--datasets', type=str, default='recon',
                        help='Comma-separated list of datasets to evaluate')
    parser.add_argument('--output_dir', type=str, default='./evaluation_output',
                        help='Output directory for predictions and metrics')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader workers')
    parser.add_argument('--num_sec_eval', type=int, default=5,
                        help='Number of seconds for time evaluation')
    parser.add_argument('--input_fps', type=int, default=4,
                        help='Input FPS')
    parser.add_argument('--rollout_fps_values', type=str, default='1,4',
                        help='Comma-separated rollout FPS values')
    
    # W&B arguments
    parser.add_argument('--wandb_project', type=str, default='nwm',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (username or team)')
    parser.add_argument('--wandb_group', type=str, default='eval',
                        help='W&B group name')
    parser.add_argument('--wandb_tags', type=str, default='',
                        help='Comma-separated W&B tags')
    
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_path):
    """Extract epoch and training step from checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', 0)
        train_steps = ckpt.get('train_steps', 0)
        return epoch, train_steps
    except Exception as e:
        print(f"Warning: Could not load checkpoint metadata: {e}")
        return 0, 0


def run_command(cmd, description):
    """Run a subprocess command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with exit code {result.returncode}")
        return False
    
    print(f"✓ {description} completed successfully")
    return True


def load_metrics(json_path):
    """Load metrics from JSON file."""
    if not os.path.exists(json_path):
        print(f"Warning: Metrics file not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def log_metrics_to_wandb(wandb_run, metrics, prefix):
    """Log metrics to W&B with a prefix (e.g., 'time/', 'rollout_1fps/')."""
    if metrics is None:
        return
    
    # Flatten nested metrics and add prefix
    flat_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_metrics[f"{prefix}/{key}/{subkey}"] = subvalue
        else:
            flat_metrics[f"{prefix}/{key}"] = value
    
    wandb_run.log(flat_metrics)
    print(f"✓ Logged {len(flat_metrics)} metrics to W&B under '{prefix}/'")


def main():
    args = parse_args()
    
    # Extract checkpoint name and config name
    checkpoint_name = Path(args.checkpoint).stem  # e.g., 'latest' or 'checkpoint_epoch_10'
    config_name = Path(args.config).stem  # e.g., 'ac_repa_cdit_xl'
    
    # Load checkpoint metadata
    print(f"Loading checkpoint metadata from: {args.checkpoint}")
    epoch, train_steps = load_checkpoint_metadata(args.checkpoint)
    
    # Initialize W&B
    wandb_tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    wandb_tags.extend([config_name, checkpoint_name, 'evaluation'])
    
    run_name = f"eval_{config_name}_{checkpoint_name}_step{train_steps}"
    
    print(f"\nInitializing W&B run:")
    print(f"  Project: {args.wandb_project}")
    print(f"  Group: {args.wandb_group}")
    print(f"  Run name: {run_name}")
    print(f"  Tags: {wandb_tags}")
    
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        tags=wandb_tags,
        config={
            'checkpoint_path': args.checkpoint,
            'config_path': args.config,
            'epoch': epoch,
            'train_steps': train_steps,
            'datasets': args.datasets,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
        }
    )
    
    # Determine experiment output directory
    exp_output_dir = os.path.join(args.output_dir, f"{config_name}_{checkpoint_name}")
    
    datasets = args.datasets.split(',')
    success = True
    
    # =========================================================================
    # TIME EVALUATION
    # =========================================================================
    
    print("\n" + "="*80)
    print("STARTING TIME EVALUATION")
    print("="*80)
    
    # Step 1: Time inference
    time_infer_cmd = [
        'python', 'isolated_nwm_infer.py',
        '--exp', args.config,
        '--ckp_path', args.checkpoint,
        '--output_dir', args.output_dir,
        '--datasets', args.datasets,
        '--eval_type', 'time',
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--num_sec_eval', str(args.num_sec_eval),
        '--input_fps', str(args.input_fps),
        '--single-gpu', '1',
    ]
    
    if not run_command(time_infer_cmd, "Time evaluation inference"):
        success = False
    
    # Step 2: Time metrics computation (WITHOUT W&B logging)
    for dataset in datasets:
        dataset = dataset.strip()
        
        time_metrics_cmd = [
            'python', 'isolated_nwm_eval.py',
            '--datasets', dataset,
            '--gt_dir', os.path.join(args.output_dir, 'gt', dataset, 'time'),
            '--exp_dir', exp_output_dir,
            '--eval_types', 'time',
        ]
        
        if not run_command(time_metrics_cmd, f"Time metrics computation ({dataset})"):
            success = False
            continue
        
        # Step 3: Load and log time metrics to W&B
        time_metrics_path = os.path.join(exp_output_dir, f"{dataset}_time.json")
        time_metrics = load_metrics(time_metrics_path)
        log_metrics_to_wandb(wandb_run, time_metrics, f"time/{dataset}")
    
    # =========================================================================
    # ROLLOUT EVALUATION
    # =========================================================================
    
    print("\n" + "="*80)
    print("STARTING ROLLOUT EVALUATION")
    print("="*80)
    
    # Step 1: Rollout inference
    rollout_infer_cmd = [
        'python', 'isolated_nwm_infer.py',
        '--exp', args.config,
        '--ckp_path', args.checkpoint,
        '--output_dir', args.output_dir,
        '--datasets', args.datasets,
        '--eval_type', 'rollout',
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--rollout_fps_values', args.rollout_fps_values,
        '--input_fps', str(args.input_fps),
        '--single-gpu', '1',
    ]
    
    if not run_command(rollout_infer_cmd, "Rollout evaluation inference"):
        success = False
    
    # Step 2: Rollout metrics computation (WITHOUT W&B logging)
    rollout_fps_list = [fps.strip() for fps in args.rollout_fps_values.split(',')]
    
    for dataset in datasets:
        dataset = dataset.strip()
        
        for rollout_fps in rollout_fps_list:
            rollout_metrics_cmd = [
                'python', 'isolated_nwm_eval.py',
                '--datasets', dataset,
                '--gt_dir', os.path.join(args.output_dir, 'gt', dataset, f'rollout_{rollout_fps}fps'),
                '--exp_dir', exp_output_dir,
                '--eval_types', f'rollout_{rollout_fps}fps',
            ]
            
            if not run_command(rollout_metrics_cmd, f"Rollout metrics computation ({dataset}, {rollout_fps}fps)"):
                success = False
                continue
            
            # Step 3: Load and log rollout metrics to W&B
            rollout_metrics_path = os.path.join(exp_output_dir, f"{dataset}_rollout_{rollout_fps}fps.json")
            rollout_metrics = load_metrics(rollout_metrics_path)
            log_metrics_to_wandb(wandb_run, rollout_metrics, f"rollout_{rollout_fps}fps/{dataset}")
    
    # =========================================================================
    # FINALIZE
    # =========================================================================
    
    wandb.finish()
    
    print("\n" + "="*80)
    if success:
        print("✓ EVALUATION COMPLETED SUCCESSFULLY")
        print(f"✓ All metrics logged to W&B run: {run_name}")
    else:
        print("⚠ EVALUATION COMPLETED WITH ERRORS")
        print("⚠ Some metrics may be missing from W&B")
    print("="*80)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

