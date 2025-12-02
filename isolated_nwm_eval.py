# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import json

from PIL import Image
import cv2

# Eval
import lpips
from dreamsim import dreamsim
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
import distributed as dist

# W&B
import wandb


def get_loss_fn(loss_fn_type, secs, device):
    if loss_fn_type == 'lpips':
        general_lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
        def loss_fn(img0_paths, img1_paths):
            img0_list = []
            img1_list = []
            
            for img0_path, img1_path in zip(img0_paths, img1_paths):
                img0 = lpips.im2tensor(lpips.load_image(img0_path)).to(device) # RGB image from [-1,1]
                img1 = lpips.im2tensor(lpips.load_image(img1_path)).to(device)
                
                img0_list.append(img0)
                img1_list.append(img1)
                
            all_img0 = torch.cat(img0_list, dim=0)
            all_img1 = torch.cat(img1_list, dim=0)
            
            dist = general_lpips_loss_fn.forward(all_img0, all_img1)
            dist_avg = dist.mean()
            
            return dist_avg
    elif loss_fn_type == 'dreamsim':
        dreamsim_loss_fn, preprocess = dreamsim(pretrained=True, device=device)
        def loss_fn(img0_paths, img1_paths):
            img0_list = []
            img1_list = []
            
            for img0_path, img1_path in zip(img0_paths, img1_paths):
                img0 = preprocess(Image.open(img0_path)).to(device)
                img1 = preprocess(Image.open(img1_path)).to(device)
                
                img0_list.append(img0)
                img1_list.append(img1)
            
            all_img0 = torch.cat(img0_list, dim=0)
            all_img1 = torch.cat(img1_list, dim=0)
            
            dist = dreamsim_loss_fn(all_img0, all_img1)
            dist_mean = dist.mean()
            
            return dist_mean
    elif loss_fn_type == 'fid':
        fid_metrics = {}
        for sec in secs:
            fid_metrics[sec] = FrechetInceptionDistance(feature_dim=2048).to(device)
        
        return fid_metrics
    else:
        raise NotImplementedError
    
    return loss_fn


#################################################################################
#                    Temporal Artifact Metrics                                  #
#################################################################################

def compute_optical_flow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,  # flow (output)
        0.5,   # pyr_scale: pyramid scale factor
        3,     # levels: number of pyramid levels
        15,    # winsize: averaging window size
        3,     # iterations: number of iterations at each pyramid level
        5,     # poly_n: size of pixel neighborhood
        1.2,   # poly_sigma: standard deviation of Gaussian
        0     # flags
    )
    
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    return flow, flow_magnitude


def compute_temporal_smoothness(frame_paths, lpips_loss_fn):
    if len(frame_paths) < 2:
        return 0.0, 0.0
    
    lpips_distances = []
    for i in range(len(frame_paths) - 1):
        lpips_dist = lpips_loss_fn([frame_paths[i]], [frame_paths[i+1]])
        lpips_distances.append(lpips_dist.item())
    
    lpips_distances = np.array(lpips_distances)
    lpips_variance = np.var(lpips_distances)
    lpips_mean = np.mean(lpips_distances)
    
    return lpips_variance, lpips_mean


def compute_jerk_metric(flow_magnitudes):
    if len(flow_magnitudes) < 3:
        return 0.0, 0.0
    
    velocities = []
    for i in range(len(flow_magnitudes) - 1):
        velocity = flow_magnitudes[i+1] - flow_magnitudes[i]
        velocities.append(velocity)
    
    if len(velocities) < 2:
        return 0.0, 0.0
    
    accelerations = []
    for i in range(len(velocities) - 1):
        acceleration = velocities[i+1] - velocities[i]
        accelerations.append(acceleration)
    
    if len(accelerations) < 2:
        return 0.0, 0.0
    
    jerks = []
    for i in range(len(accelerations) - 1):
        jerk = accelerations[i+1] - accelerations[i]
        jerk_magnitude = np.abs(jerk)
        jerks.append(jerk_magnitude)
    
    if len(jerks) == 0:
        return 0.0, 0.0
    
    all_jerks = np.concatenate([j.flatten() for j in jerks])
    mean_jerk = np.mean(all_jerks)
    max_jerk = np.max(all_jerks)
    
    return mean_jerk, max_jerk


def evaluate_temporal_artifacts(args, dataset_name, eval_type, metric_logger, lpips_loss_fn, gt_dir, exp_dir, secs, rollout_fps):
    if eval_type == 'rollout':
        eval_name = f'rollout_{rollout_fps}fps'
        # For rollout, we evaluate sequences of frames
        max_frame_idx = int((secs[-1] * rollout_fps) - 1)
    elif eval_type == 'time':
        eval_name = 'time'
        # For time eval, we compare initial frame with predicted frames at different time steps
        max_frame_idx = int(secs[-1])
    else:
        return
    
    eps = os.listdir(gt_dir)
    
    for batch_start in tqdm(range(0, len(eps), args.batch_size), 
                           total=(len(eps) + args.batch_size - 1) // args.batch_size,
                           desc=f"Temporal artifacts ({eval_name})"):
        batch_eps = eps[batch_start:batch_start + args.batch_size]
        
        for ep in batch_eps:
            gt_ep_dir = os.path.join(gt_dir, ep)
            exp_ep_dir = os.path.join(exp_dir, ep)
            
            if not os.path.isdir(gt_ep_dir) or not os.path.isdir(exp_ep_dir):
                continue
            
            gt_frame_paths = []
            exp_frame_paths = []
            
            for frame_idx in range(max_frame_idx + 1):
                gt_frame_path = os.path.join(gt_ep_dir, f'{frame_idx}.png')
                exp_frame_path = os.path.join(exp_ep_dir, f'{frame_idx}.png')
                
                if os.path.exists(gt_frame_path) and os.path.exists(exp_frame_path):
                    gt_frame_paths.append(gt_frame_path)
                    exp_frame_paths.append(exp_frame_path)
            
            if len(gt_frame_paths) < 2:
                continue
            
            # 1. Optical Flow Consistency
            gt_flows = []
            exp_flows = []
            gt_flow_magnitudes = []
            exp_flow_magnitudes = []
            
            for i in range(len(gt_frame_paths) - 1):
                gt_img1 = np.array(Image.open(gt_frame_paths[i]).convert("RGB"))
                gt_img2 = np.array(Image.open(gt_frame_paths[i+1]).convert("RGB"))
                exp_img1 = np.array(Image.open(exp_frame_paths[i]).convert("RGB"))
                exp_img2 = np.array(Image.open(exp_frame_paths[i+1]).convert("RGB"))
                
                gt_flow, gt_flow_mag = compute_optical_flow(gt_img1, gt_img2)
                exp_flow, exp_flow_mag = compute_optical_flow(exp_img1, exp_img2)
                
                gt_flows.append(gt_flow)
                exp_flows.append(exp_flow)
                gt_flow_magnitudes.append(gt_flow_mag)
                exp_flow_magnitudes.append(exp_flow_mag)
            
            # Compute flow consistency (L2 distance between GT and predicted flows)
            if len(gt_flows) > 0:
                flow_errors = []
                for gt_flow, exp_flow in zip(gt_flows, exp_flows):
                    flow_error = np.mean(np.sqrt(np.sum((gt_flow - exp_flow)**2, axis=2)))
                    flow_errors.append(flow_error)
                
                mean_flow_error = np.mean(flow_errors)
                metric_logger.meters[f'{dataset_name}_{eval_name}_flow_error'].update(mean_flow_error, n=1)
            
            # 2. Temporal Smoothness (LPIPS variance)
            gt_lpips_var, gt_lpips_mean = compute_temporal_smoothness(gt_frame_paths, lpips_loss_fn)
            exp_lpips_var, exp_lpips_mean = compute_temporal_smoothness(exp_frame_paths, lpips_loss_fn)
            
            metric_logger.meters[f'{dataset_name}_{eval_name}_temporal_smoothness_var'].update(exp_lpips_var, n=1)
            metric_logger.meters[f'{dataset_name}_{eval_name}_temporal_smoothness_mean'].update(exp_lpips_mean, n=1)
            metric_logger.meters[f'{dataset_name}_{eval_name}_temporal_smoothness_var_gt'].update(gt_lpips_var, n=1)
            metric_logger.meters[f'{dataset_name}_{eval_name}_temporal_smoothness_mean_gt'].update(gt_lpips_mean, n=1)
            
            # 3. Jerk Metric (acceleration discontinuities)
            if len(gt_flow_magnitudes) >= 3:
                gt_mean_jerk, gt_max_jerk = compute_jerk_metric(gt_flow_magnitudes)
                exp_mean_jerk, exp_max_jerk = compute_jerk_metric(exp_flow_magnitudes)
                
                metric_logger.meters[f'{dataset_name}_{eval_name}_jerk_mean'].update(exp_mean_jerk, n=1)
                metric_logger.meters[f'{dataset_name}_{eval_name}_jerk_max'].update(exp_max_jerk, n=1)
                metric_logger.meters[f'{dataset_name}_{eval_name}_jerk_mean_gt'].update(gt_mean_jerk, n=1)
                metric_logger.meters[f'{dataset_name}_{eval_name}_jerk_max_gt'].update(gt_max_jerk, n=1)


def evaluate(args, dataset_name, eval_type, metric_logger, loss_fns, gt_dir, exp_dir, secs, rollout_fps):
    lpips_loss_fn, dreamsim_loss_fn, fid_loss_fn = loss_fns
    
    if eval_type == 'rollout':
        eval_name = f'rollout_{rollout_fps}fps'
        image_idxs = (secs * rollout_fps) - 1
    elif eval_type == 'time':
        eval_name = eval_type
        image_idxs = secs.copy()
        
    eps = os.listdir(gt_dir)
    
    for batch_start in tqdm(range(0, len(eps), args.batch_size), total=(len(eps) + args.batch_size - 1) // args.batch_size):
        batch_eps = eps[batch_start:batch_start + args.batch_size]
        
        gt_batch, exp_batch = {}, {}
        gt_paths_batch, exp_paths_batch = {}, {}
        for sec in secs:
            gt_batch[sec] = []
            exp_batch[sec] = []
            gt_paths_batch[sec] = []
            exp_paths_batch[sec] = []
        
        for ep in batch_eps:
            gt_ep_dir = os.path.join(gt_dir, ep)
            exp_ep_dir = os.path.join(exp_dir, ep)
        
            if not os.path.isdir(gt_ep_dir) and not os.path.isdir(exp_ep_dir):
                continue
        
            for sec, image_idx in zip(secs, image_idxs):
                gt_sec_img_path = os.path.join(gt_ep_dir, f'{image_idx}.png')
                gt_sec_img = transforms.ToTensor()(Image.open(gt_sec_img_path).convert("RGB")).unsqueeze(0)
                exp_sec_img_path = os.path.join(exp_ep_dir, f'{image_idx}.png')
                exp_sec_img = transforms.ToTensor()(Image.open(exp_sec_img_path).convert("RGB")).unsqueeze(0)
                
                gt_batch[sec].append(gt_sec_img)
                gt_paths_batch[sec].append(gt_sec_img_path)
                exp_batch[sec].append(exp_sec_img)
                exp_paths_batch[sec].append(exp_sec_img_path)
            
        for sec in secs:
            lpips_dists = lpips_loss_fn(gt_paths_batch[sec], exp_paths_batch[sec])
            dreamsim_dists = dreamsim_loss_fn(gt_paths_batch[sec], exp_paths_batch[sec])
            
            metric_logger.meters[f'{dataset_name}_{eval_name}_lpips_{sec}s'].update(lpips_dists, n=1)
            metric_logger.meters[f'{dataset_name}_{eval_name}_dreamsim_{sec}s'].update(dreamsim_dists, n=1)
            
            sec_gt_batch = torch.cat(gt_batch[sec], dim=0)
            sec_exp_batch = torch.cat(exp_batch[sec], dim=0)
            
            fid_loss_fn[sec].update(images=sec_gt_batch, is_real=True)
            fid_loss_fn[sec].update(images=sec_exp_batch, is_real=False)
            
    for sec in secs:
        metric_logger.meters[f'{dataset_name}_{eval_name}_fid_{sec}s'].update(fid_loss_fn[sec].compute().item(), n=1)
        
def save_metric_to_disk(metric_logger, log_p):
    metric_logger.synchronize_between_processes()
    log_stats = {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}
    with open(log_p, 'w') as json_file:
        json.dump(log_stats, json_file, indent=4)  # indent=4 adds indentation for readability            


def main(args):
    device = 'cuda'
    
    # Initialize W&B if enabled
    wandb_run = None
    if args.wandb_enabled:
        # Extract checkpoint name and config name for run naming
        checkpoint_name = os.path.basename(args.exp_dir)
        
        wandb_config = {
            'datasets': args.datasets,
            'eval_types': ','.join(args.eval_types),
            'num_sec_eval': args.num_sec_eval,
            'checkpoint': checkpoint_name,
        }
        
        # Add checkpoint metadata if provided
        if args.wandb_checkpoint_epoch is not None:
            wandb_config['checkpoint_epoch'] = args.wandb_checkpoint_epoch
        if args.wandb_checkpoint_step is not None:
            wandb_config['checkpoint_step'] = args.wandb_checkpoint_step
        if args.wandb_checkpoint_path:
            wandb_config['checkpoint_path'] = args.wandb_checkpoint_path
        
        # Create tags
        tags = ['evaluation']
        if args.wandb_tags:
            tags.extend(args.wandb_tags.split(','))
        
        # Create run name
        run_name = args.wandb_run_name or f"eval_{checkpoint_name}"
        
        # Initialize wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            tags=tags,
            config=wandb_config,
            reinit=True
        )
        print(f"W&B run initialized: {wandb_run.name}")
          
    # Loading Datasets
    dataset_names = args.datasets.split(',')
    
    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
    
    # These loss functions do not accumulate
    lpips_loss_fn = get_loss_fn('lpips', secs, device)
    dreamsim_loss_fn = get_loss_fn('dreamsim', secs, device)

    for dataset_name in dataset_names:
        gt_dataset_dir = os.path.join(args.gt_dir, dataset_name)
        exp_dataset_dir = os.path.join(args.exp_dir, dataset_name)
        
        if 'rollout' in args.eval_types:
            for rollout_fps in args.rollout_fps_values:
                try:
                    metric_logger = dist.MetricLogger(delimiter="  ")
                    print("Evaluating rollout", rollout_fps, dataset_name)
                    # Rollout (LPIPS, DreamSim, FID)
                    eval_name = f'rollout_{rollout_fps}fps'
                    gt_dataset_rollout_dir = os.path.join(gt_dataset_dir, eval_name)
                    exp_dataset_rollout_dir = os.path.join(exp_dataset_dir, eval_name)
                    rollout_fid_loss_fn = get_loss_fn('fid', secs, device)
                    rollout_loss_fns = (lpips_loss_fn, dreamsim_loss_fn, rollout_fid_loss_fn)
                    with torch.no_grad():
                        evaluate(args, dataset_name, 'rollout', metric_logger, rollout_loss_fns, gt_dataset_rollout_dir, exp_dataset_rollout_dir, secs, rollout_fps)
                        # Evaluate temporal artifacts for rollout
                        evaluate_temporal_artifacts(args, dataset_name, 'rollout', metric_logger, lpips_loss_fn, gt_dataset_rollout_dir, exp_dataset_rollout_dir, secs, rollout_fps)
                    output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                    save_metric_to_disk(metric_logger, output_fn)
                    
                    # Log to W&B if enabled
                    if wandb_run:
                        log_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                        wandb_metrics = {f"eval/{dataset_name}_{eval_name}_{k}": v for k, v in log_stats.items()}
                        wandb_run.log(wandb_metrics)
                        print(f"Logged {eval_name} metrics to W&B for {dataset_name}")
                except Exception as e:
                    print(e)

        if 'time' in args.eval_types:
            try:
                metric_logger = dist.MetricLogger(delimiter="  ")
                print("Evaluating time", dataset_name)
                eval_name = 'time'
                gt_dataset_time_dir = os.path.join(gt_dataset_dir, eval_name)
                exp_dataset_time_dir = os.path.join(exp_dataset_dir, eval_name)
                time_fid_loss_fn = get_loss_fn('fid', secs, device)
                time_loss_fns = (lpips_loss_fn, dreamsim_loss_fn, time_fid_loss_fn)
                with torch.no_grad():
                    evaluate(args, dataset_name, eval_name, metric_logger, time_loss_fns, gt_dataset_time_dir, exp_dataset_time_dir, secs, None)
                    # Evaluate temporal artifacts for time evaluation
                    evaluate_temporal_artifacts(args, dataset_name, 'time', metric_logger, lpips_loss_fn, gt_dataset_time_dir, exp_dataset_time_dir, secs, None)
                output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                save_metric_to_disk(metric_logger, output_fn)
                
                # Log to W&B if enabled
                if wandb_run:
                    log_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                    wandb_metrics = {f"eval/{dataset_name}_{eval_name}_{k}": v for k, v in log_stats.items()}
                    wandb_run.log(wandb_metrics)
                    print(f"Logged {eval_name} metrics to W&B for {dataset_name}")
            except Exception as e:
                print(e)
    
    # Finish W&B run
    if wandb_run:
        wandb.finish()
        print("W&B run finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--eval_types", type=str, default='time,rollout,rollout_video', help="evluations")
    parser.add_argument("--gt_dir", type=str, default=None, help="gt directory")
    parser.add_argument("--exp_dir", type=str, default=None, help="experiment directory")
    parser.add_argument("--num_sec_eval", type=int, default=5, help="experiment name")
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    
    parser.add_argument("--input_fps", type=int, default=4, help="experiment name")
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    
    # W&B arguments
    parser.add_argument("--wandb-enabled", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="nwm", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name")
    parser.add_argument("--wandb-tags", type=str, default=None, help="W&B tags (comma-separated)")
    parser.add_argument("--wandb-checkpoint-epoch", type=int, default=None, help="Checkpoint epoch")
    parser.add_argument("--wandb-checkpoint-step", type=int, default=None, help="Checkpoint training step")
    parser.add_argument("--wandb-checkpoint-path", type=str, default=None, help="Checkpoint file path")
    
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    
    args.eval_types = args.eval_types.split(',')
    
    main(args)