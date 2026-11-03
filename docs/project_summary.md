# Research Project Summary: Synthetic Fingerprint Data Augmentation Study

## What This Is

A journal paper studying whether synthetic fingerprint data improves model training performance, and which synthetic generation methods are most effective. The study covers three fingerprint analysis tasks, uses a controlled factorial experimental design, and prioritises scientific rigour and reproducibility over maximising any single benchmark score.

---

## The Three Tasks

| ID | Task | Input | Output | Output resolution |
|---|---|---|---|---|
| FC | Fingerprint Classification | Image | 5-class label | — |
| FMD | Fingerprint Minutiae Detection | Image | Minutiae map | W × H, 3 channel |
| FOE | Fingerprint Orientation Estimation | Image | Orientation map | W/8 × H/8, 128 channels |

---

## Datasets

### Real (one per task)

| Task | Dataset | Total | Train | Test | Notes |
|---|---|---|---|---|---|
| FC | NIST_SD4 | 4,000 | 3,200 (N) | 800 | Identity-level split |
| FOE | FOESamples | 60 | 48 (N) | 12 | Stratified, 5:1 Bad/Good ratio. Manually labelled orientation maps |
| FMD | FVCsubset | 3,200 | 2,560 (N) | 640 | Finger-level split |

### Synthetic (11 sources)

| ID | Full name | Generator family | Total images | Notes |
|---|---|---|---|---|
| S01 | FVC-SFINGE-pool | SFINGE | ~4,440 | FC classes for all ~4,437 annotated images via the VeriFinger annotation pipeline (corrected 2026-07-06: previously only 180 images had labels via a separate legacy file, now superseded) |
| S02 | Denkovarik | ANGULI | 110,000 | Multi-impression (×11 with 10:1 bad/good ratio) |
| S03 | ICB2018 | GAN | Unlimited | Single impression, generator available |
| S04 | ICIP2021Clarkson | StyleGAN | 50,047 | Single impression |
| S05 | PAMI2023MSUPrintsGAN | GAN | 525,000 |Multi-impression (×15) |
| S06 | SynFing | StyleGAN2 | 200,000 | Multi-impression (×2) |
| S07 | VISAPP2023SGRv1 | ANGULI+StyleGAN2-ada | 50,000 | Single impression |
| S08 | VISAPP2023P2Pv1 | ANGULI+pix2pix | 40,000 | Multi-impression (×10) |
| S09 | VISAPP2023P2Pv2 | ANGULI+pix2pix | 40,000 | Multi-impression (×10) |
| S10 | WACV2024 | StyleGAN2 | Unlimited | Multi-impression, generator available |
| S11 | BAJECE2025 | SFINGE-like | 100,000 | 2,500 fingers × 8 impressions × 5 classes (corrected 2026-07-06: previously wrongly listed as 25,000/single-impression); FP classes and orientation labels at W/8×H/8 available natively |
(S12 - Pending access authorization for MSUGenPrint)

Multi-impression sources (S02, S05, S06, S08, S09) are sampled at identity level to prevent within-identity leakage across folds.

### Synthetic label generation (where needed)

| Task | Label source |
|---|---|
| FC | NBIS PCASYS / VeriFinger |
| FMD | NBIS MINDTCT / VeriFinger |
| FOE | NBIS MINDTCT / SNFOE (pyfing) |

Double annotation: synthetic images are labelled by two automated tools per task (see §4.1). Disagreement rates are reported per source as a label quality characterisation. A subset of experiments is rerun under the alternative labeller to assess sensitivity of conclusions to labelling choice. Full experiment suite uses the primary labeller only.

---

## Experimental Conditions

### Volume conditions (real fixed, synthetic added)

| Condition | Real | Synthetic | Total |
|---|---|---|---|
| V0 | N | 0 | N |
| V1 | N | N | 2N |
| V2 | N | 3N | 4N |
| V3 | N | 7N | 8N |

Each step doubles total training size.

### Substitution conditions (total fixed at N)

| Condition | Real | Synthetic |
|---|---|---|
| R0 | N | 0 (identical to V0) |
| R1 | 3N/4 | N/4 |
| R2 | N/2 | N/2 |
| R3 | N/4 | 3N/4 |
| R4 | 0 | N |

R4 is a domain transfer experiment — no real images seen during training.

**Total conditions:** 8 (V0/R0 shared baseline + V1–V3 + R1–R4)
**Non-baseline conditions:** 7, each tested with all 11 synthetic sources.

---

## Models

Three models per task, same framework across all tasks:

| Model | Encoder | Initialisation | Role |
|---|---|---|---|
| M1 | ResNet-50 | From scratch | Sensitive probe of synthetic data quality |
| M2 | ResNet-50 | ImageNet pretrained | Isolates effect of pretraining vs M1 |
| M3 | Swin-V2-T | ImageNet pretrained | Isolates effect of architecture vs M2 |

M1 vs M2: same architecture, different initialisation — clean ablation of pretraining effect.
M2 vs M3: same initialisation, different architecture — clean ablation of architecture effect.

### Architecture per task

**FC:** Encoder + linear classification head (5 classes).

**FMD:** Encoder + UNet-style decoder (4 upsampling stages) + 3-channel per-pixel output head. Output: W×H, 3 channels (background / ridge ending / bifurcation).

**FOE:** Encoder + UNet-style decoder (2 upsampling stages) + 128-channel per-pixel output head. Output: W/8×H/8, 128 channels (cyclic encoding). Decoder terminates at 1/8 resolution matching the ground truth annotation granularity.

Decoder design is identical across M1, M2, M3 for each dense task — only encoder weights and initialisation differ. Skip connections from encoder stages to decoder stages for both FMD and FOE.

---

## Training Protocol

### Splits
- 80/20 train/test split, fixed before any experiment begins
- 4-fold CV on the 80% training pool
- Validation set: real samples only, fixed size across all conditions
- All split indices saved and version-controlled before any training run

### Input normalisation
- Spatial: fixed target resolution per task (FC: 512×512, FMD: 388×374, FOE: 448×478)
- M2/M3 intensity: ImageNet mean/std; images converted to 3-channel if grayscale
- M1 intensity: per-dataset zero-mean unit-variance from training split

### Augmentation (paired image+label map transforms)

| Augmentation | FC | FMD | FOE | Label update |
|---|---|---|---|---|
| Horizontal flip | ✓ | ✓ | ✓ | FC: L → R and R → L, FOE: θ → π−θ |
| Rotation by α | ✓ | ✓ | ✓ | FOE: θ → (θ+α) mod π |
| Center crop | ✓ | ✓ | ✓ | FOE: crop scaled by 1/8 |
| Elastic deformation | ✓ | ✓ | — | Not applied to FOE |
| Brightness/contrast | ✓ | ✓ | ✓ | None |
| Gaussian blur | ✓ | ✓ | ✓ | None |
| Gaussian noise | ✓ | ✓ | ✓ | None |
| Gamma correction | ✓ | ✓ | ✓ | None |

FOE orientation maps stored as uint8 [0–255] (representing [0°, 180°)). After spatial transforms, pixel values updated and remapped as uint8. 128-bit cyclic encoding computed on-the-fly during training, never stored.

### Loss functions

| Task | Output | Loss | Notes |
|---|---|---|---|
| FC | 5-class softmax | Cross-entropy | Primary label as GT; multi-label tolerance at evaluation only |
| FMD | 3-channel per-pixel softmax | Weighted cross-entropy | Class weights needed — background dominates |
| FOE | 128-channel per-pixel sigmoid | Binary cross-entropy per bit | Cyclic encoding |

### FOE cyclic encoding (128-bit)

With B=256 bins, thresholds s_k = k×(180/256)°. Bit i is 1 if s_{(i+1) mod 256} ≤ θ < s_{(i+129) mod 256}, else 0. (circular wrap) Orientations 0° and 180° (physically identical) receive same codes, 0° and 179°, and 0° and 1° are maximally similar codes. Orientations 0° and 90° receive maximally dissimilar codes. Hamming distance proportional to circular angular distance.

FOE inference decoding: for each candidate class k (0–255), compute dot product between 128 predicted sigmoid values and encoding vector of k. Select class with highest dot product (soft decoding, no threshold needed).

### Training schedule

Step-based (max_steps per task, not epochs). Ensures comparable compute across dataset sizes. Batch size: largest power of 2 fitting in VRAM, minimum floor of 8.

### Hyperparameter tuning (per experiment)

HP grid: 24 configurations (LR ∈ {3e-5, 1e-4, 3e-4} × WD ∈ {0, 1e-5, 1e-4, 5e-4} × cosine annealing ∈ {False, True}).

| Task | HP Phase 1 | HP Phase 2 | Final eval |
|---|---|---|---|
| FC | 24 configs × 4 folds on subset | Top 3 × 4 folds on full CV | Best config × 3 seeds |
| FMD | 24 configs × 4 folds on subset | Top 3 × 4 folds on full CV | Best config × 3 seeds |
| FOE | 24 configs × 4 folds on full CV (merged) | Skipped | Best config × 3 seeds |

FOE uses merged single-stage because N=48 makes subset search unreliable.

NMS tuning for FMD (no retraining, sorted jointly with HP configs): for each fold, score_fold() sweeps detection threshold (7 values) × NMS radius (7 values) over the saved heatmaps, returning 49 F1 scalars. HPSelector accumulates these across all folds and HP configs, ranking all 1176 (HP, threshold, radius) triplets jointly. Constraint: NMS radius ≤ matching distance.

Validation set for HP ranking: real samples only, fixed across all conditions.

---

## Experiment Count

One experiment = 1 condition × 1 synthetic source × 1 task × 1 model

| Condition type | Count | Experiments |
|---|---|---|
| V0/R0 baseline (no synthetic) | 1 × 3 tasks × 3 models | 9 |
| V1–V3, R1–R4 (7 conditions × 11 sources) | 7 × 11 × 3 × 3 | 693 |
| **Total** | | **702** |

Execution order: V0/R0 baseline first (9 experiments), then remaining 693. Each experiment is fully self-contained and can be run independently.

### Training runs per experiment

| Task | HP Phase 1 | HP Phase 2 | Final eval | Total |
|---|---|---|---|---|
| FC | 96 | 12 | 3 | 111 |
| FMD | 96 | 12 | 3 | 111 |
| FOE | 96 | 0 | 3 | 99 |

**Note:** FMD Phase 1 also includes a post-hoc NMS parameter sweep over all saved heatmaps. This adds no training runs.

### Secondary fixed-HP experiment

After Stage 1 of HP tuning is complete, identify the modal HP config per task (or per-task-per-model depending on the variance between them) across conditions. Rerun all conditions under this fixed config (For FMD, the modal triplet (HP config, NMS threshold, NMS radius) is used in full — all three parameters are held fixed.). Allows two complementary claims: (1) per-condition tuning = best achievable performance under equal budget; (2) fixed HP = causal effect of synthetic data holding the learner constant.

---

## Evaluation

### Predictions saved (all as .npy)

| Task | What is saved | Notes |
|---|---|---|
| FC | Class probability vector per image | Softmax output, 5 values |
| FMD | Post-NMS minutiae point lists per image | (x, y, type, score); + raw heatmaps for 5 randomly selected images |
| FOE | Orientation map per image | W/8×H/8 float |

### Logging and training history

Experiment status is tracked with a json file. For all models trained, the history - training and validation losses, training and validation metrics (FC → accuracy; FMD → F1 at fixed matching distance; FOE → mean angular error- is saved.


### FMD minutiae matching

One-to-one bipartite matching (greedy, sort by distance, assign without conflict). TP = matched pairs, FP = unmatched predictions, FN = unmatched GT. Computed per image then averaged, and globally across all images.

### Metrics

**FC:** Top-1 accuracy (multi-label tolerant), per-class accuracy, confusion matrix.

**FMD:** Precision, recall, F1 at 3 fixed matching distances. Per-type (ridge ending / bifurcation). Per-quality (poor / fair / good). Per-image average and global aggregate.

**FOE:** Mean angular error per image then averaged, and global, accuracy at different error thresholds. Stratified by quality (B / G).

HP ranking metrics: FC → accuracy; FMD → F1 at fixed matching distance (after NMS sweep); FOE → mean angular error (negated).

---

## Software Architecture

**Stack:** Plain PyTorch, YAML configs, timm model zoo, torch.cuda.amp, pandas+matplotlib for analysis.

**One experiment = one self-contained run.** Launched manually via CLI. Resumes automatically from any interruption via status.json.

**Key modules:**
- Config system (inheritance + programmatic generation of 702 YAMLs)
- Split generator (run once, version-controlled)
- Dataset classes (Real + Synthetic + Combined)
- Model factory (M1/M2/M3 per task)
- Trainer (step-based, prediction saving, skip-if-exists)
- ExperimentRunner (HP Phase 1 [incl. NMS sweep for FMD] → HP Phase 2 [FC and FMD only] → Final eval)
- Evaluator (score_fold online + evaluate_full offline)
- Analysis layer (aggregator, tables, plots, status report)

**Environments:** 4080 machine for development and smoke tests (dev.yaml overrides: fewer steps, folds, HP configs, seeds). Pro 6000 for full runs.

---

## Key Design Decisions

**Why per-condition HP tuning (not fixed HP):** Fixed HP optimised for real data may penalise synthetic conditions that benefit from different regularisation. Per-condition tuning asks "what is the best this condition can achieve?" Fixed HP is run as a secondary experiment to support the causal claim.

**Why real-only validation set:** Ensures the validation oracle is identical across all conditions, removing differential selection bias from data-rich conditions.

**Why step-based training not epochs:** Ensures comparable training compute across very different dataset sizes (FOE N=48 vs FC N=3,200).

**Why scratch + pretrained models:** M1 (scratch) is a sensitive probe of synthetic data quality with no prior knowledge. M2 (same arch, pretrained) isolates the pretraining effect. M3 (different arch, pretrained) checks architecture robustness. Homogeneous framework across all three tasks.

**Why NMS tuning is separate from HP grid:** NMS is applied post-hoc to saved heatmaps with no retraining, allowing joint ranking of HP and NMS parameters without multiplying training compute.

**Why 128-bit cyclic encoding for FOE:** Respects circular topology of ridge orientation. Hamming distance between codes proportional to angular distance. Orientations 0° and 180° (physically identical) get maximally similar codes. BCE per bit is simple and effective.

**Why UNet decoder for both FMD and FOE:** Consistent framework across dense tasks. Skip connections allow fine spatial detail. FOE decoder truncated at 2 upsampling stages to match W/8×H/8 ground truth resolution — motivated by annotation granularity, not architectural preference.

**Why FOE has no post-processing parameter sweep:** Orientation field smoothing was necessary in classical gradient-based methods due to their purely local operation. Deep learning encoders aggregate multi-scale context through successive downsampling stages, producing spatially coherent orientation fields without explicit smoothing. No parametric post-processing step analogous to NMS is therefore needed for FOE.

---

## Rejected Ideas

**PyTorch Lightning:** Dropped. Training control flow is specialised enough that Lightning abstractions add overhead without simplifying the core loop.

**Study phases (Phase 1/2/3 sequential execution):** Dropped. Full factorial design runs all 11 sources × all 7 non-baseline conditions. Analytical groupings emerge from how results are presented, not from execution order.

**Subset search for FOE HP tuning:** Dropped for FOE only (N=48 makes subset search on ~8 images unreliable). FC and FMD retain two-stage procedure.

**AUC-based HP ranking for FMD:** Dropped in favour of F1 at a fixed matching distance — fixing both threshold and radius gives a single operating point, not a curve.

**Joint "HP + NMS" grid search (1176 triplets) training:** (retraining for each NMS parameter combination) Dropped. NMS is applied post-hoc to saved heatmaps, so training is not repeated per NMS configuration.

**Soft label targets for FC:** Primary label used as ground truth in training; multi-label tolerance handled only at evaluation.

**Elastic deformation for FOE:** Dropped — per-pixel orientation value correction after elastic deformation is not tractable.

**Many-to-one minutiae matching for FMD:** Dropped in favour of one-to-one bipartite matching, which correctly reflects that each physical minutia is a single event.

---

## Known Limitations (to state in paper)

1. **FOE label circularity for non-S11 sources:** Synthetic orientation labels derived from same tool family as related work. Mitigated by evaluation on manually labelled FOESamples.
2. **S01 capacity:** Hard ceiling at ~4,440 images. Results at V2/V3 not directly comparable to other sources.
3. **FOE dataset size:** N=48 produces high variance. All FOE conclusions require full variance reporting.
4. **Architecture coverage:** Two encoders is not universality — paper claims consistency under a controlled protocol, not universal architecture-agnosticism.

---
