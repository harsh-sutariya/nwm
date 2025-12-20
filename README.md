# AC-REPA: Action-aware Representation Alignment For Navigation World Models

> **AC-REPA: Action-aware Representation Alignment For Navigation World Models**<br>
> Harsh Sutaria, Soham Chitnis, Shaswat Patel<br>
> New York University<br>
> {hs5580, sc11537, spp9399}@nyu.edu

## Abstract

Navigation world models (NWMs) are action-conditioned diffusion video predictors that roll out egocentric futures for planning. Despite progress with Conditional Diffusion Transformers (CDiTs), training remains costly and rollouts often exhibit poor temporal coherence (jittery motion, "teleporting" artifacts) and weak action sensitivity.

**AC-REPA** aligns internal denoising representations of an action-conditioned CDiT to a frozen video foundation encoder (VideoMAE-v2), explicitly conditioning the alignment on actions. It combines:
1. **Feature Alignment (FA)** on the student's spatio-temporal token grid
2. **Action-Conditioned Token Relation Distillation (AC-TRD)** that matches student/teacher Gram matrices while upweighting time–token pairs where commanded motion is high

## Key Results

On RECON dataset with ViT-Small backbone:
- ✅ **Lower motion jerk** (improved temporal smoothness)
- ✅ **Better semantic similarity at longer horizons** (DreamSim @ 8-16s)
- ✅ **Improved action sensitivity** (counterfactual divergence)
- ✅ **Reduced variance** in temporal smoothness

## Setup

```bash
git clone --recursive https://github.com/harsh-sutariya/nwm
cd nwm
```

### Requirements

```bash
mamba create -n nwm python=3.10
mamba activate nwm
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
mamba install ffmpeg
pip3 install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets opencv-python
```

## Data

Follow the data setup from [NoMaD](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling):
1. Download the [RECON dataset](https://sites.google.com/view/recon-robot)
2. Change preprocessing resolution from (160, 120) to (320, 240)
3. Run `process_recon.py` to save to `data/recon/`

Expected structure:
```
nwm/data
└── recon/
    ├── <trajectory_name>/
    │   ├── 0.jpg, 1.jpg, ..., T.jpg
    │   └── traj_data.pkl
    └── ...
```

## Training

### AC-REPA Training (Recommended)

```bash
# Single GPU
python train.py --config config/ac_repa_cdit_xl.yaml \
    --ckpt-every 5000 --eval-every 10000 --bfloat16 1 --epochs 300

# Multi-GPU with torchrun
torchrun --nproc-per-node=2 train.py \
    --config config/ac_repa_cdit_xl.yaml \
    --ckpt-every 5000 --eval-every 10000 --bfloat16 1 --epochs 300

### Baseline NWM Training

```bash
python train.py --config config/nwm_cdit_xl.yaml \
    --ckpt-every 5000 --eval-every 10000 --bfloat16 1 --epochs 300
```

### AC-REPA Configuration

Key parameters in `config/ac_repa_cdit_xl.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_ac_repa` | `True` | Enable AC-REPA training |
| `teacher_model` | `MCG-NJU/videomae-base-finetuned-kinetics` | Frozen video encoder |
| `lambda_fa` | `0.1` | Feature alignment loss weight |
| `lambda_trd` | `0.05` | Token relation distillation weight |
| `trd_gate_type` | `action` | Gate type: `action`, `motion`, or `none` |
| `use_sparse_trd` | `True` | Use sparse token selection |
| `sparse_ratio` | `0.25` | Ratio of tokens to keep |

## Evaluation

### Manual Evaluation

#### 1. Generate Ground Truth (one time)

```bash
export RESULTS_FOLDER=./evaluation_output

# Time prediction GT
python isolated_nwm_infer.py \
    --exp config/ac_repa_cdit_xl.yaml \
    --datasets recon \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1

# Rollout GT
python isolated_nwm_infer.py \
    --exp config/ac_repa_cdit_xl.yaml \
    --datasets recon \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4
```

#### 2. Generate Predictions

```bash
# Using run_eval.py (recommended - handles inference + evaluation)
python run_eval.py \
    --checkpoint /path/to/checkpoint.pth.tar \
    --config config/ac_repa_cdit_xl.yaml \
    --datasets recon \
    --output_dir ${RESULTS_FOLDER} \
    --wandb_project nwm
```

#### 3. Compute Metrics

```bash
python isolated_nwm_eval.py \
    --datasets recon \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/ac_repa_cdit_sl \
    --eval_types time,rollout
```

### Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Perceptual** | LPIPS | Perceptual similarity (lower = better) |
| | DreamSim | Semantic similarity (lower = better) |
| | FID | Fréchet Inception Distance (lower = better) |
| **Temporal** | Jerk Mean/Max | Motion smoothness (lower = better) |
| | Temporal Smoothness | Frame-to-frame LPIPS variance |
| | Flow Error | Optical flow consistency |
| **Action** | Counterfactual Divergence | Sensitivity to action changes (higher = better) |

## Acknowledgments

This work builds upon [Navigation World Models](https://github.com/facebookresearch/nwm) by Bar et al. We thank the authors for releasing their code and models.

We also acknowledge:
- [REPA](https://arxiv.org/abs/2410.06940) for representation alignment insights
- [VideoREPA](https://arxiv.org/abs/2505.23656) for video specific alignment strategies
- [VideoMAE-v2](https://arxiv.org/abs/2303.16727) for the pretrained video encoder