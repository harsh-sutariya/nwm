"""
Action Insensitivity Metrics for World Model Evaluation

This module implements metrics to evaluate how sensitive a world model is to action changes.
A good world model should be sensitive to actions - when actions change, predictions should change accordingly.

Based on recent research in world model evaluation (CoRL, NeurIPS, ICML 2024-2025):
- Action perturbation sensitivity analysis
- Counterfactual action evaluation
- Action-output correlation metrics
"""

import torch
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms
import lpips
from dreamsim import dreamsim
import cv2


def compute_action_perturbation_sensitivity(
    model, 
    initial_state, 
    actions, 
    perturbation_scale=0.1,
    num_perturbations=10,
    device='cuda'
):
    """
    Compute action perturbation sensitivity metric.
    
    Measures how much predictions change when actions are perturbed.
    Lower sensitivity = more action insensitive (bad).
    
    Args:
        model: World model that takes (state, action) -> next_state
        initial_state: Initial state tensor [B, ...]
        actions: Original action tensor [B, T, action_dim]
        perturbation_scale: Scale of action perturbations (relative to action norm)
        num_perturbations: Number of random perturbations to test
        device: Device to run on
        
    Returns:
        sensitivity_score: Mean sensitivity across perturbations (higher is better)
        sensitivity_std: Std dev of sensitivity
    """
    model.eval()
    with torch.no_grad():
        # Get baseline prediction
        baseline_pred = model(initial_state, actions)
        
        sensitivities = []
        action_norm = torch.norm(actions, dim=-1).mean()
        
        for _ in range(num_perturbations):
            # Generate random perturbation
            perturbation = torch.randn_like(actions) * perturbation_scale * action_norm
            perturbed_actions = actions + perturbation
            
            # Get prediction with perturbed actions
            perturbed_pred = model(initial_state, perturbed_actions)
            
            # Compute sensitivity: ||pred_change|| / ||action_change||
            pred_change = torch.norm(baseline_pred - perturbed_pred, dim=-1).mean()
            action_change = torch.norm(perturbation, dim=-1).mean()
            
            if action_change > 1e-6:
                sensitivity = pred_change / action_change
                sensitivities.append(sensitivity.item())
        
        if len(sensitivities) == 0:
            return 0.0, 0.0
        
        sensitivity_mean = np.mean(sensitivities)
        sensitivity_std = np.std(sensitivities)
        
        return sensitivity_mean, sensitivity_std


def compute_action_output_correlation(
    model,
    initial_state,
    action_sequence,
    device='cuda'
):
    """
    Compute correlation between action magnitude and prediction change.
    
    A good model should show high correlation - larger actions should lead to larger prediction changes.
    Low correlation = action insensitive (bad).
    
    Args:
        model: World model
        initial_state: Initial state tensor [B, ...]
        action_sequence: Sequence of actions [B, T, action_dim]
        device: Device to run on
        
    Returns:
        correlation: Pearson correlation coefficient (higher is better, range [-1, 1])
    """
    model.eval()
    with torch.no_grad():
        # Get predictions for each action in sequence
        predictions = []
        action_magnitudes = []
        
        for t in range(action_sequence.shape[1]):
            action = action_sequence[:, t:t+1, :]  # [B, 1, action_dim]
            pred = model(initial_state, action)
            
            # Use first timestep prediction for comparison
            if t == 0:
                baseline_pred = pred
            else:
                pred_change = torch.norm(pred - baseline_pred, dim=-1).mean().item()
                action_mag = torch.norm(action, dim=-1).mean().item()
                
                predictions.append(pred_change)
                action_magnitudes.append(action_mag)
        
        if len(predictions) < 2:
            return 0.0
        
        # Compute Pearson correlation
        correlation = np.corrcoef(action_magnitudes, predictions)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return correlation


def compute_counterfactual_action_divergence(
    model,
    initial_state,
    action1,
    action2,
    device='cuda'
):
    """
    Compute divergence between predictions with different actions from same initial state.
    
    Measures how different predictions are when actions differ.
    Low divergence = action insensitive (bad).
    
    Args:
        model: World model
        initial_state: Initial state tensor [B, ...]
        action1: First action sequence [B, T, action_dim]
        action2: Second action sequence [B, T, action_dim]
        device: Device to run on
        
    Returns:
        divergence: Normalized divergence score (higher is better)
    """
    model.eval()
    with torch.no_grad():
        pred1 = model(initial_state, action1)
        pred2 = model(initial_state, action2)
        
        # Compute prediction difference
        pred_diff = torch.norm(pred1 - pred2, dim=-1).mean()
        
        # Normalize by action difference
        action_diff = torch.norm(action1 - action2, dim=-1).mean()
        
        if action_diff > 1e-6:
            divergence = (pred_diff / action_diff).item()
        else:
            divergence = 0.0
        
        return divergence


def compute_action_gradient_norm(
    model,
    initial_state,
    actions,
    device='cuda'
):
    """
    Compute gradient norm of predictions with respect to actions.
    
    Measures how sensitive predictions are to infinitesimal action changes.
    Small gradient = action insensitive (bad).
    
    Args:
        model: World model
        initial_state: Initial state tensor [B, ...]
        actions: Action tensor [B, T, action_dim] (requires_grad=True)
        device: Device to run on
        
    Returns:
        gradient_norm: Mean gradient norm (higher is better)
    """
    model.eval()
    actions = actions.clone().detach().requires_grad_(True)
    
    pred = model(initial_state, actions)
    
    # Compute gradient w.r.t. actions
    grad_outputs = torch.ones_like(pred)
    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=actions,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False
    )[0]
    
    # Compute mean gradient norm
    gradient_norm = torch.norm(gradients, dim=-1).mean().item()
    
    return gradient_norm


def evaluate_action_insensitivity_from_predictions(
    gt_dir,
    exp_dir,
    dataset_name,
    eval_type,
    metric_logger,
    lpips_loss_fn,
    dreamsim_loss_fn,
    secs,
    rollout_fps=None,
    device='cuda'
):
    """
    Evaluate action insensitivity metrics from saved predictions.
    
    This version works with pre-computed predictions and action data files.
    Actions are expected to be saved as 'actions.npy' in each episode folder (id_*).
    
    Args:
        gt_dir: Directory with ground truth predictions
        exp_dir: Directory with model predictions
        dataset_name: Name of dataset
        eval_type: 'time' or 'rollout'
        metric_logger: Metric logger for tracking
        lpips_loss_fn: LPIPS loss function
        dreamsim_loss_fn: DreamSim loss function
        secs: Time steps to evaluate
        rollout_fps: FPS for rollout evaluation
        device: Device to run on
    """
    if eval_type == 'rollout':
        eval_name = f'rollout_{rollout_fps}fps'
        max_frame_idx = int((secs[-1] * rollout_fps) - 1)
    elif eval_type == 'time':
        eval_name = 'time'
        max_frame_idx = int(secs[-1])
    else:
        return
    
    # Get list of episode directories
    if not os.path.exists(exp_dir):
        return
    
    eps = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d)) and d.startswith('id_')]
    
    if len(eps) == 0:
        return
    
    # Track metrics
    action_correlations = []
    counterfactual_divergences = []
    
    for ep in tqdm(eps, desc=f"Action insensitivity ({eval_name})"):
        exp_ep_dir = os.path.join(exp_dir, ep)
        
        if not os.path.isdir(exp_ep_dir):
            continue
        
        # Load action data from episode folder
        action_file = os.path.join(exp_ep_dir, 'actions.npy')
        if not os.path.exists(action_file):
            continue
        
        try:
            actions = np.load(action_file)  # [T, action_dim] or [T] if 1D
            if actions.ndim == 1:
                # If 1D, assume it's action magnitude
                actions = actions.reshape(-1, 1)
        except Exception as e:
            continue
        
        # Load prediction paths
        pred_paths = []
        pred_indices = []
        
        if eval_type == 'rollout':
            # For rollout, frames are numbered 0, 1, 2, ...
            for frame_idx in range(max_frame_idx + 1):
                pred_path = os.path.join(exp_ep_dir, f'{frame_idx}.png')
                if os.path.exists(pred_path):
                    pred_paths.append(pred_path)
                    pred_indices.append(frame_idx)
        else:  # time
            # For time eval, frames are numbered by seconds (1, 2, 4, 8, ...)
            for sec in secs:
                pred_path = os.path.join(exp_ep_dir, f'{sec}.png')
                if os.path.exists(pred_path):
                    pred_paths.append(pred_path)
                    pred_indices.append(sec)
        
        if len(pred_paths) < 2:
            continue
        
        # Compute action-output correlation
        # Use action magnitude vs prediction change between consecutive frames
        action_magnitudes = []
        prediction_changes_lpips = []
        prediction_changes_dreamsim = []
        
        for i in range(len(pred_paths) - 1):
            # Map frame index to action index
            if eval_type == 'rollout':
                action_idx = pred_indices[i]
            else:  # time
                # For time eval, actions are cumulative, so use the action at the current timestep
                action_idx = i
            
            if action_idx >= len(actions):
                break
            
            # Action magnitude
            action_mag = np.linalg.norm(actions[action_idx])
            action_magnitudes.append(action_mag)
            
            # Prediction change (using LPIPS and DreamSim)
            pred_change_lpips = lpips_loss_fn([pred_paths[i]], [pred_paths[i+1]])
            prediction_changes_lpips.append(pred_change_lpips.item())
            
            pred_change_dreamsim = dreamsim_loss_fn([pred_paths[i]], [pred_paths[i+1]])
            prediction_changes_dreamsim.append(pred_change_dreamsim.item())
        
        if len(action_magnitudes) >= 2:
            # LPIPS correlation
            correlation_lpips = np.corrcoef(action_magnitudes, prediction_changes_lpips)[0, 1]
            if not np.isnan(correlation_lpips):
                action_correlations.append(correlation_lpips)
                metric_logger.meters[f'{dataset_name}_{eval_name}_action_correlation_lpips'].update(
                    correlation_lpips, n=1
                )
            
            # DreamSim correlation
            correlation_dreamsim = np.corrcoef(action_magnitudes, prediction_changes_dreamsim)[0, 1]
            if not np.isnan(correlation_dreamsim):
                metric_logger.meters[f'{dataset_name}_{eval_name}_action_correlation_dreamsim'].update(
                    correlation_dreamsim, n=1
                )
        
        # Compute counterfactual divergence
        # Compare predictions with small vs large actions from same initial state
        if len(actions) >= 2 and len(pred_paths) >= 2:
            # Find frames with smallest and largest actions
            action_norms = []
            for i, pred_idx in enumerate(pred_indices):
                if eval_type == 'rollout':
                    action_idx = pred_idx
                else:
                    action_idx = i
                if action_idx < len(actions):
                    action_norms.append((i, np.linalg.norm(actions[action_idx])))
            
            if len(action_norms) >= 2:
                action_norms_sorted = sorted(action_norms, key=lambda x: x[1])
                min_idx, min_norm = action_norms_sorted[0]
                max_idx, max_norm = action_norms_sorted[-1]
                
                if min_idx != max_idx and min_idx < len(pred_paths) and max_idx < len(pred_paths):
                    # Compare predictions from initial frame (index 0)
                    pred_min = lpips_loss_fn([pred_paths[0]], [pred_paths[min_idx]])
                    pred_max = lpips_loss_fn([pred_paths[0]], [pred_paths[max_idx]])
                    
                    # Normalize by action difference
                    action_diff = abs(max_norm - min_norm)
                    if action_diff > 1e-6:
                        divergence = abs(pred_max.item() - pred_min.item()) / action_diff
                        counterfactual_divergences.append(divergence)
                        metric_logger.meters[f'{dataset_name}_{eval_name}_counterfactual_divergence'].update(
                            divergence, n=1
                        )
    
    # Log summary statistics
    if len(action_correlations) > 0:
        mean_corr = np.mean(action_correlations)
        metric_logger.meters[f'{dataset_name}_{eval_name}_action_correlation_lpips_mean'].update(
            mean_corr, n=1
        )
    
    if len(counterfactual_divergences) > 0:
        mean_div = np.mean(counterfactual_divergences)
        metric_logger.meters[f'{dataset_name}_{eval_name}_counterfactual_divergence_mean'].update(
            mean_div, n=1
        )


def compute_action_insensitivity_score(
    action_correlation,
    counterfactual_divergence,
    action_perturbation_sensitivity=None
):
    """
    Compute a composite action insensitivity score.
    
    Lower score = more action insensitive (bad).
    Higher score = more action sensitive (good).
    
    Args:
        action_correlation: Action-output correlation (higher is better)
        counterfactual_divergence: Counterfactual divergence (higher is better)
        action_perturbation_sensitivity: Perturbation sensitivity (higher is better)
        
    Returns:
        insensitivity_score: Composite score (0-1, lower is better for "insensitivity")
        sensitivity_score: Composite score (0-1, higher is better for "sensitivity")
    """
    # Normalize metrics to [0, 1]
    # Correlation is already in [-1, 1], normalize to [0, 1]
    norm_correlation = (action_correlation + 1) / 2
    
    # Divergence and sensitivity need normalization (assuming reasonable ranges)
    # These should be tuned based on your specific use case
    norm_divergence = np.clip(counterfactual_divergence / 10.0, 0, 1)
    
    # Compute sensitivity score (higher is better)
    if action_perturbation_sensitivity is not None:
        norm_sensitivity = np.clip(action_perturbation_sensitivity / 5.0, 0, 1)
        sensitivity_score = (norm_correlation + norm_divergence + norm_sensitivity) / 3.0
    else:
        sensitivity_score = (norm_correlation + norm_divergence) / 2.0
    
    # Insensitivity score is the inverse
    insensitivity_score = 1.0 - sensitivity_score
    
    return insensitivity_score, sensitivity_score
