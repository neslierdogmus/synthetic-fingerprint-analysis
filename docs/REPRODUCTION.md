# Synthetic Fingerprint Data Augmentation — Reproduction Guide

This document describes how to reproduce the experiments from scratch on a new machine.

---

## 1. Hardware Requirements

- NVIDIA GPU with CUDA support (experiments were run on RTX PRO 6000 Blackwell, 96GB VRAM)
- For Blackwell (sm_120) architecture: PyTorch nightly with CUDA 12.8 is required
- Minimum 32GB RAM recommended
- ~2TB storage for datasets and outputs

---

## 2. Software Prerequisites

### 2.1 Ubuntu 22.04 or 24.04 LTS

### 2.2 NVIDIA Driver

Check if already installed:
```bash
nvidia-smi
```

If not installed:
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 2.3 Docker Engine

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

### 2.4 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
    https://nvidia.github.io/libnvidia-container/stable/deb/$(dpkg --print-architecture) /" \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 2.5 VS Code + Dev Containers extension

```bash
sudo snap install code --classic
```

Install the **Dev Containers** extension in VS Code.

---

## 3. Dataset Preparation

### 3.1 Real Datasets

Place the following datasets on the host machine:

| Dataset | Used for | Expected path |
|---------|----------|---------------|
| NIST SD4 | FC (Fingerprint Classification) | `/opt/data/FP_Real/NIST_SD4/` |
| FVC Berkay Topcu subset | FMD (Minutiae Detection) | `/opt/data/FP_Real/FVC_BERKAY_TOPCU/` |
| FOE Samples | FOE (Orientation Estimation) | `/opt/data/FP_Real/FOESamples/` |

NIST_SD4 structure:
```
NIST_SD4/png_txt/figs_0/ ... figs_7/
    f0001_01.png, f0001_01.txt, s0001_01.png, ...
```

FVC_BERKAY_TOPCU structure:
```
FVC_BERKAY_TOPCU/
    FM3_FVC2002DB1A_np/  1_1.tif, 1_1.npz, ...
    FM3_FVC2002DB3A_np/
    FM3_FVC2004DB1A_np/
    FM3_FVC2004DB3A_np/
```

FOESamples structure:
```
FOESamples/
    Bad/  00.bmp, 00.gt, 00.fg, ... 49.bmp
    Good/ 110.bmp, 110.gt, 110.fg, ... 119.bmp
```

### 3.2 Synthetic Datasets

Collect the following synthetic sources and arrange them in the standardised
`FP_Synth/` structure described in Section 3.3.

| ID | Dataset | Generator | Notes |
|----|---------|-----------|-------|
| S01 | FVC-SFINGE-pool | SFINGE | FVC2000–2006 DB4 A+B sets |
| S02 | Denkovarik | ANGULI | Bad/ and Good/ quality splits |
| S03 | ICB2018 GAN | GAN | Pre-generate 25,000 images |
| S04 | ICIP2021Clarkson | StyleGAN | ~50,048 images |
| S05 | PAMI2023 MSUPrintsGAN | GAN | 35,000 fingers × 15 impressions |
| S06 | SynFing | StyleGAN2 | 100,000 fingers × 2 impressions |
| S07 | VISAPP2023 SGRv1 | ANGULI+StyleGAN2-ada | 50,000 images |
| S08 | VISAPP2023 P2Pv1 | ANGULI+pix2pix | 500 subjects × 8 fingers × 10 imp |
| S09 | VISAPP2023 P2Pv2 | ANGULI+pix2pix | same structure as S08 |
| S10 | WACV2024 FPGAN | StyleGAN2 | Pre-generate 25,000 images |
| S11 | BAJECE2025 | SFINGE-like | 2500 fingers × 8 imp × 5 classes |

### 3.3 Synthetic Dataset Layout

Each source goes in its own flat folder under `/opt/data/FP_Synth/`
(`S01/`, `S02/`, …, `S11/`). The container mounts this at `/data/synthetic/`.
The synthetic labels are read separately from
`annotation_workspace/annotations/{S01..S11}/` relative to the repository root.
To set this up, unzip the released `annotations.zip` into
`annotation_workspace/` (`unzip annotations.zip -d annotation_workspace/`).

File naming convention: `{FP_id}--{impression_id}.ext`
The separator `--` is guaranteed unique; FP_id and impression_id may each
contain `_` internally.

---

## 4. Project Setup

### 4.1 Clone the repository

```bash
git clone <repo_url> /path/to/project
cd /path/to/project
```

### 4.2 Update dataset mount paths

Edit `.devcontainer/devcontainer.json` and set the correct host paths:

```json
"mounts": [
    "source=/opt/data/FP_Real,target=/data/real,type=bind,readonly",
    "source=/opt/data/FP_Synth,target=/data/synthetic,type=bind,readonly"
]
```

### 4.3 Open in Dev Container

Open the project in VS Code:
```bash
code /path/to/project
```

When prompted, click **Reopen in Container**. The container will build
automatically (this takes several minutes on first build as it downloads
the PyTorch nightly image and installs dependencies).

### 4.4 Verify GPU access

Inside the container:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

---

## 5. Generating Configs and Splits

All of the following commands are run **inside the container** from `/workspace`.

### 5.1 Generate experiment configs (702 YAML files)

```bash
python -m src.config.generator --out configs/
```

Expected output: `Generated 702 / skipped 0 / total 702`

### 5.2 Generate data splits

```bash
python -m src.data.split_generator --root . --data-root /data
```

Expected output:
```
Real splits:      3 generated / 0 skipped
Synthetic splits: 231 generated / 0 skipped
Total:            234 files written
```

### 5.3 Verify

```bash
# Config count
find configs -name "*.yaml" | grep experiments | wc -l
# Should be 702

# Split sizes
python -c "
import json
for task in ['fc', 'fmd', 'foe']:
    with open(f'splits/{task}/real_split.json') as f:
        s = json.load(f)
    print(f'{task}: train_val={len(s[chr(34)+\"train_val_ids\"+chr(34)])}, test={len(s[chr(34)+\"test_ids\"+chr(34)])}')
"
```

---

## 6. Running Experiments

### 6.1 Smoke test (verify setup before full runs)

```bash
python run_experiment.py \
    --config configs/fc/experiments/fc_M1_V0.yaml \
    --dev \
    --data-root /data
```

Expected: completes in ~5 minutes, prints `Experiment fc_M1_V0 complete.`

### 6.2 Baseline experiments (no synthetic data, 9 experiments)

Run these first — they establish the V0 baseline and take the longest
per experiment since they involve full HP tuning (24 configs × 4 folds).

```bash
nohup bash -c '
for exp in \
    configs/fc/experiments/fc_M1_V0.yaml \
    configs/fc/experiments/fc_M2_V0.yaml \
    configs/fc/experiments/fc_M3_V0.yaml \
    configs/fmd/experiments/fmd_M1_V0.yaml \
    configs/fmd/experiments/fmd_M2_V0.yaml \
    configs/fmd/experiments/fmd_M3_V0.yaml \
    configs/foe/experiments/foe_M1_V0.yaml \
    configs/foe/experiments/foe_M2_V0.yaml \
    configs/foe/experiments/foe_M3_V0.yaml; do
    python run_experiment.py --config $exp --data-root /data
done
' > baseline_runs.log 2>&1 &
echo $!
```

Monitor:
```bash
tail -f baseline_runs.log
python -m src.analysis.status_report
```

### 6.3 Non-baseline experiments (693 experiments)

After baselines complete, run the remaining experiments. These can be
batched or run selectively by task, model, or condition.

Example — all FC experiments:
```bash
nohup bash -c '
for exp in configs/fc/experiments/fc_*.yaml; do
    python run_experiment.py --config $exp --data-root /data
done
' > fc_runs.log 2>&1 &
```

Example — all experiments for a specific source (e.g. S11):
```bash
nohup bash -c '
for exp in configs/*/experiments/*_S11.yaml; do
    python run_experiment.py --config $exp --data-root /data
done
' > s11_runs.log 2>&1 &
```

### 6.4 Resuming after interruption

Every experiment is resumable. Simply rerun the same command —
the status.json file tracks which stages are complete and skips them:

```bash
python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml --data-root /data
```

### 6.5 Forcing a rerun from scratch

```bash
python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml \
    --data-root /data --force
```

**Warning:** `--force` deletes all outputs for that experiment and reruns
from scratch. Use with caution.

---

## 7. Monitoring Progress

```bash
# Overview of all experiments
python -m src.analysis.status_report

# Build results CSV from completed experiments
python -m src.analysis.aggregator --outputs outputs --data-root /data

# Generate tables and figures
python -c "
from src.analysis.tables import generate_tables
from src.analysis.plots import generate_plots
generate_tables()
generate_plots()
"
```

---

## 8. Experiment Structure

One experiment = 1 condition × 1 synthetic source × 1 task × 1 model

Total: **702 experiments**
- 9 baseline (V0, no synthetic) × 3 tasks × 3 models
- 693 non-baseline = 7 conditions × 11 sources × 3 tasks × 3 models

### Conditions

| ID | Type | Real | Synthetic | Total |
|----|------|------|-----------|-------|
| V0 | Volume baseline | N | 0 | N |
| V1 | Volume | N | N | 2N |
| V2 | Volume | N | 3N | 4N |
| V3 | Volume | N | 7N | 8N |
| R0 | Substitution baseline | N | 0 | N (=V0) |
| R1 | Substitution | 3N/4 | N/4 | N |
| R2 | Substitution | N/2 | N/2 | N |
| R3 | Substitution | N/4 | 3N/4 | N |
| R4 | Substitution | 0 | N | N |

### Models

| ID | Encoder | Initialisation | Role |
|----|---------|----------------|------|
| M1 | ResNet-50 | From scratch | Sensitive probe of synthetic data quality |
| M2 | ResNet-50 | ImageNet pretrained | Isolates pretraining effect |
| M3 | Swin-V2-T | ImageNet pretrained | Isolates architecture effect |

### Training runs per experiment

| Task | HP Phase 1 | HP Phase 2 | Final eval | Total |
|------|------------|------------|------------|-------|
| FC | 96 | 12 | 3 | 111 |
| FMD | 96 | 12 | 3 | 111 |
| FOE | 96 | 0 | 3 | 99 |

---

## 9. Output Structure

```
outputs/
└── {task}/
    └── {experiment_id}/
        ├── config.yaml        # Full resolved config snapshot
        ├── status.json        # Experiment progress tracker
        ├── hp_phase1/
        │   ├── fold_N_hp_M.npy    # Validation predictions
        │   └── fold_N_hp_M.json   # Training history
        ├── hp_phase2/             # FC and FMD only
        │   ├── fold_N_hp_M.npy
        │   └── fold_N_hp_M.json
        └── final_eval/
            ├── seed_0.npy         # Test predictions
            ├── seed_0.json        # Training history
            ├── seed_1.npy
            ├── seed_2.npy
            └── best_model.pth
```

---

## 10. Known Issues and Notes

- **Blackwell GPU (sm_120):** Requires PyTorch nightly with CUDA 12.8.
  The Dockerfile installs this automatically.
- **S01 shortfall:** Only ~4,440 images available. Results at V2/V3 not
  directly comparable to other sources (shortfall flagged in split JSON).
- **FOE dataset size:** N=48 produces high variance. All FOE conclusions
  require full variance reporting across seeds.
- ~~**S11 minutiae:** No type information (ridge ending vs bifurcation) in
  pre-computed labels. S11 minutiae are not used for FMD training.~~
  **Corrected 2026-08-05**: this was wrong. S11's `.mnt` files are fully
  typed (same format as every other source, confirmed by direct
  inspection of all 30,001 files) and are now used for FMD training like
  any other source.
- **Albumentations update warning:** Harmless. Set `NO_ALBUMENTATIONS_UPDATE=1`
  to suppress.
