# Experiment Design: Synthetic Data Augmentation for Fingerprint Analysis

---

## 1. Research Goal

Evaluate whether synthetic fingerprint data improves training performance across multiple fingerprint analysis tasks, and compare different synthetic data generation methods fairly and rigorously. The study is not optimising for a single benchmark score. Scientific rigour and publication strength take priority over maximising any single task's raw performance.

---

## 2. Tasks

| ID | Name | Input | Output | Output resolution |
|---|---|---|---|---|
| FC | Fingerprint Classification | Fingerprint image | Class label | — |
| FMD | Fingerprint Minutiae Detection | Fingerprint image | Minutiae map | W × H (full resolution) |
| FOE | Fingerprint Orientation Estimation | Fingerprint image | Orientation map | W/8 × H/8 (8×8 block grid) |

---

## 3. Real Datasets

| # | Name | Task | Resolution | Structure | Total images | Train (80%) | Test (20%) | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | NIST_SD4 | FC | 512×512 | 2000 fingers × 2 impressions | 4,000 | 3,200 | 800 | Identity-level split |
| 2 | FOESamples | FOE | 448×478 | (50B + 10G) × 1 | 60 | 48 | 12 | Stratified split, preserve 5:1 B/G ratio |
| 3 | FVCsubset | FMD | 388×374 | 400 fingers × 8 impressions | 3,200 | 2,560 | 640 | Finger-level split |

**N per task** (= real training pool size after 80/20 split):

| Task | N |
|---|---|
| FC | 3,200 |
| FOE | 48 |
| FMD | 2,560 |

---

## 4. Synthetic Datasets

| ID | Full Name | Resolution | Structure | Total images | Generator family | Notes |
|---|---|---|---|---|---|---|
| S01 | FVC-SFINGE-pool | 240–288×320–384 | ~4,440 total | ~4,440 | SFINGE | Pooled FVC2000–2006 DB4 A+B sets; cannot be extended |
| S02 | Denkovarik | 206×300 | 1000 fingers × (100B + 10G) | 110,000 | ANGULI | Quality labels available |
| S03 | ICB2018 | 512×512 | Generated on demand | Unlimited | GAN | Generator available |
| S04 | ICIP2021Clarkson | 512×512 | 50,047 × 1 | 50,047 | StyleGAN | |
| S05 | PAMI2023MSUPrintsGAN | 512×512 | 35,000 × 15 | 525,000 | GAN | Pending replacement by MSUGenPrint |
| S06 | SynFing | 512×512 | 100,000 × 2 | 200,000 | StyleGAN2 | |
| S07 | VISAPP2023SGRv1 | 512×512 | 50,000 × 1 | 50,000 | ANGULI + StyleGAN2-ada | |
| S08 | VISAPP2023P2Pv1 | 512×512 | (500 × 8) × 10 | 40,000 | ANGULI + pix2pix | |
| S09 | VISAPP2023P2Pv2 | 512×512 | (500 × 8) × 10 | 40,000 | ANGULI + pix2pix | |
| S10 | WACV2024 | TBD | Generated on demand | Unlimited | StyleGAN2 | Generator available |
| S11 | BAJECE2025 | 400×400 | 2,500 fingers × 8 imp × 5 classes | 100,000 | SFINGE-like | classes and native orientation labels at W/8 × H/8 (corrected 2026-07-06: was wrongly listed as 25,000 total, doesn't match README.md or the actual 100,000 files on disk) |

### 4.1 Synthetic Labels

Labels for synthetic images are produced per task as follows.

| Task | Sources |
|---|---|
| Fingerprint class label | NBIS PCASYS / Verifinger |
| Minutiea locations | NBIS MINDTCT / VeriFinger |
| Orientation map | NBIS MINDTCT / SNFOE (pyfing) |


> **Note:** For FOE, all synthetic orientation labels must be standardised to the W/8 × H/8 block format before training, matching the real ground truth annotation granularity. Loss is computed at W/8 × H/8 — predictions are never upsampled to full resolution before loss computation.

### 4.2 Subsampling Strategy

- **Multi-impression sources** (S02, S05, S06, S08, S09): sample at the finger/identity level. Sample ceil(target / impressions_per_identity) identities, then take all impressions up to the target count. This prevents within-identity leakage across folds.
- **Single-impression sources**: random image-level sampling.
- **Generated sources** (S03, S10): generate exactly the required number of images. No subsampling needed.
- **S01**: use all available images when target exceeds pool size. Note shortfall in results table.
- **FOE** FOE synthetic splits are stratified by quality (40B + 8G, matching the 80% training pool B/G composition)

### 4.3 Capacity Flags at V3 (largest synthetic demand)

| Source | FC needs 22,400 | FMD needs 17,920 |
|---|---|---|
| S01 (~4,440) | ✗ shortfall — use all available | ✗ shortfall — use all available |
| S11 (100,000) | ✓ | ✓ |
| All others | ✓ | ✓ |

---

## 5. Models

One framework applies uniformly across all three tasks. Three models per task:

| Model | Encoder | Initialisation | Role |
|---|---|---|---|
| M1 | ResNet-50 | From scratch | Sensitive probe of synthetic data quality; no prior knowledge |
| M2 | ResNet-50 | ImageNet pretrained | Isolates effect of pretraining; architecture held constant vs M1 |
| M3 | Swin-V2-T | ImageNet pretrained | Isolates effect of architecture; initialisation held constant vs M2 |

### 5.1 Per-Task Architecture

**FC:**

| Model | Encoder | Initialisation | Head |
|---|---|---|---|
| M1 | ResNet-50 | Scratch | Linear classifier |
| M2 | ResNet-50 | ImageNet pretrained | Linear classifier |
| M3 | Swin-V2-T | ImageNet pretrained | Linear classifier |

**FMD and FOE (identical encoder-decoder framework, different output resolution):**

| Model | Encoder | Initialisation | Decoder | FMD output | FOE output |
|---|---|---|---|---|---|
| M1 | ResNet-50 | Scratch | UNet-style, fixed design | W × H, 3 channel | W/8 × H/8, 128 channel |
| M2 | ResNet-50 | ImageNet pretrained | UNet-style, fixed design | W × H, 3 channel | W/8 × H/8, 128 channel |
| M3 | Swin-V2-T | ImageNet pretrained | UNet-style, fixed design | W × H, 3 channel | W/8 × H/8, 128 channel |

### 5.2 Decoder Design (FMD and FOE)

UNet-style decoder with skip connections from the corresponding encoder stage. Each upsampling stage: bilinear upsampling → conv-BN-ReLU → conv-BN-ReLU.

- **FMD:** 4 upsampling stages, reaches full input resolution (W × H)
- **FOE:** 2 upsampling stages, terminates at 1/8 input resolution (W/8 × H/8), matching the 8×8 block annotation format of the ground truth

For ResNet-50: skip connections from layer1–layer4. For Swin-V2-T: skip connections from the four hierarchical stages (natural spatial hierarchy). The decoder design is held constant across M1, M2, M3 — only encoder weights and initialisation differ.

---

## 6. Experimental Conditions

### 6.1 Volume Conditions (real fixed, synthetic added)

Total training size doubles at each step.

| Condition | Real | Synthetic | Total | Real:Synth ratio |
|---|---|---|---|---|
| V0 | N | 0 | N | 1:0 |
| V1 | N | N | 2N | 1:1 |
| V2 | N | 3N | 4N | 1:3 |
| V3 | N | 7N | 8N | 1:7 |

**Concrete counts:**

| Condition | FC Real | FC Synth | FC Total | FOE Real | FOE Synth | FOE Total | FMD Real | FMD Synth | FMD Total |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 3,200 | 0 | 3,200 | 48 | 0 | 48 | 2,560 | 0 | 2,560 |
| V1 | 3,200 | 3,200 | 6,400 | 48 | 48 | 96 | 2,560 | 2,560 | 5,120 |
| V2 | 3,200 | 9,600 | 12,800 | 48 | 144 | 192 | 2,560 | 7,680 | 10,240 |
| V3 | 3,200 | 22,400 | 25,600 | 48 | 336 | 384 | 2,560 | 17,920 | 20,480 |

### 6.2 Substitution Conditions (total fixed at N, real replaced by synthetic)

| Condition | Real | Synthetic | Total | Real:Synth ratio |
|---|---|---|---|---|
| R0 | N | 0 | N | 1:0 (identical to V0) |
| R1 | 3N/4 | N/4 | N | 3:1 |
| R2 | N/2 | N/2 | N | 1:1 |
| R3 | N/4 | 3N/4 | N | 1:3 |
| R4 | 0 | N | N | 0:1 |

**Concrete counts:**

| Condition | FC Real | FC Synth | FOE Real | FOE Synth | FMD Real | FMD Synth |
|---|---|---|---|---|---|---|
| R0 | 3,200 | 0 | 48 | 0 | 2,560 | 0 |
| R1 | 2,400 | 800 | 36 | 12 | 1,920 | 640 |
| R2 | 1,600 | 1,600 | 24 | 24 | 1,280 | 1,280 |
| R3 | 800 | 2,400 | 12 | 36 | 640 | 1,920 |
| R4 | 0 | 3,200 | 0 | 48 | 0 | 2,560 |

> **Note:** V0 and R0 are identical. They serve as the shared baseline anchor across all phases. R4 (zero real data) is a domain transfer experiment — the model never sees a real image during training. Frame it explicitly as such in the paper.

> **Note:** FOE substitution conditions involve very small absolute numbers (e.g. R3: 12 real images). Interpret with caution and report variance across seeds.

---

## 7. Training Protocol

### 7.1 Data Splits

- Test set: 20% of real data, fixed before any experiment begins. Never used during training or hyperparameter tuning.
- Training pool: remaining 80% of real data.
- Split strategy: identity/finger level for all datasets. FOE splits stratified to preserve 5:1 B/G quality ratio.
- All split indices saved and version-controlled before any training run begins.

### 7.2 Cross-Validation

4-fold CV on the training pool (80% real data). Validation set consists of real samples only, fixed size across all conditions. This ensures the validation oracle is identical across conditions. 4-fold cross-validation is used across all tasks to ensure that each FOE validation fold contains an integer number of Good-quality images under the 5:1 B/G stratification constraint; 5-fold would produce fractional Good-image counts per fold, making stratification inconsistent across folds.

### 7.3 Hyperparameter Tuning (per task × model × condition)

Two-stage procedure, run independently for each task/model/condition combination:

**Stage 1 — Subset search (cheap screening):**
- Run full HP grid on a reduced subset (~20–30% of training pool) using 4-fold CV
- Rank all 24 configurations by mean validation performance
- Identify top 3 configurations

**Stage 2 — Full CV rerun:**
- Run top 3 configurations on the full training pool using 4-fold CV
- Select single best configuration by mean validation performance

**HP grid (24 configurations total):**

| Hyperparameter | Values |
|---|---|
| Learning rate | 3e-5, 1e-4, 3e-4 |
| Weight decay | 0, 1e-5, 1e-4, 5e-4 |
| Cosine annealing | False, True |

> **Rationale:** Per-condition tuning avoids penalising synthetic conditions with a configuration optimised for real data alone. The fixed real-only validation set removes differential selection bias across conditions. The pre-specified grid, held constant across all conditions, prevents fishing.

> **Note:** For FMD, configurations are triplets including NMS threshold and radius, not plain HP configs.
> **Note:** For FOE, stage 1 and 2 are merged due to small training size: Full CV search will be done on the HP grid. 

### 7.4 Final Evaluation

- Train on full training pool (all non-test data) using the selected HP configuration
- Evaluate once on held-out test set
- Repeat with 3 independent random seeds
- Report mean ± std across seeds

### 7.5 Fixed-HP Secondary Experiment

After Stage 1 of HP tuning is complete across all conditions, identify the modal (most frequently selected) configuration across conditions. Use this as a fixed HP configuration (For FMD, the modal triplet (HP config, NMS threshold, NMS radius) is used in full — all three parameters are held fixed.) and rerun all conditions under a shared setting. Report alongside per-condition results.

This allows two complementary claims:
- Per-condition tuning: best achievable performance per condition under equal tuning budget
- Fixed HP: causal effect of synthetic data holding the learner constant

---
