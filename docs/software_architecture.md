# Software Architecture: Synthetic Fingerprint Augmentation Experiments

> Companion document to `experiment_design.md`

---

## 1. Design Principles

1. **Encapsulation** — each experiment is a fully self-contained unit. It can be launched, interrupted, and resumed independently.
2. **Resumability** — every expensive computation is saved to disk. Any stage can be skipped if its outputs already exist.
3. **Separation of concerns** — data, training, evaluation, and analysis are independent layers with clean interfaces.
4. **Config-driven** — all behaviour is determined by YAML files. No hardcoded paths, hyperparameters, or conditions.
5. **Environment portability** — the same codebase runs in dev mode (4080, smoke test) and full mode (Pro 6000) with a single flag.
6. **Reproducibility** — all splits, seeds, and configs are saved alongside outputs. A completed experiment is fully reproducible from its output folder alone.

---

## 2. Directory Structure

```
project_root/
│
├── datasets/                         # Raw data (read-only during experiments)
│   ├── real/
│   │   ├── fc/
│   │   │   └── NIST_SD4/
│   │   ├── fmd/
│   │   │   └── FVCsubset/
│   │   └── foe/
│   │       └── FOESamples/
│   └── synthetic/
│       ├── S01/                          # FVC-SFINGE-pool
│       ├── S02/                          # Denkovarik
│       ├── S03/                          # ICB2018
│       ├── S04/                          # ICIP2021Clarkson
│       ├── S05/                          # PAMI2023MSUPrintsGAN
│       ├── S06/                          # SynFing
│       ├── S07/                          # VISAPP2023SGRv1
│       ├── S08/                          # VISAPP2023P2Pv1
│       ├── S09/                          # VISAPP2023P2Pv2
│       ├── S10/                          # WACV2024
│       └── S11/                          # BAJECE2025
│
├── splits/                           # Pre-generated, version-controlled
│   ├── fc/
│   │   ├── real_split.json           # train/test indices for NIST_SD4
│   │   └── synthetic/
│   │       ├── S02_V1.json           # synthetic fp_ids for each source x condition
│   │       ├── S02_V2.json
│   │       └── ...
│   ├── fmd/
│   │   └── ...
│   └── foe/
│       └── ...
│
├── configs/
│   ├── global_defaults.yaml          # Shared defaults across all tasks
│   ├── fc/
│   │   ├── base.yaml                 # FC-specific defaults
│   │   ├── dev.yaml                  # Smoke test overrides
│   │   └── experiments/              # Generated, one per experiment
│   │       ├── fc_M1_V0.yaml
│   │       ├── fc_M2_V1_S01.yaml
│   │       └── ...
│   ├── fmd/
│   │   └── ...
│   └── foe/
│       └── ...
│
├── outputs/
│   ├── fc/
│   │   └── {experiment_id}/          # e.g. fc_M2_V1_S01
│   │       ├── config.yaml           # Full resolved config (snapshot)
│   │       ├── status.json           # Experiment state
│   │       ├── hp_phase1/
│   │       │   ├── fold_0_hp_0.npy   # Val predictions per fold per HP config
│   │       │   ├── fold_0_hp_0.json   # Training history per fold per HP config
│   │       │   └── ...
│   │       ├── hp_phase2/
│   │       │   ├── fold_0_hp_3.npy   # Val predictions per fold, top 3 configs only
│   │       │   ├── fold_0_hp_3.json   # Training history per fold, top 3 configs only
│   │       │   └── ...
│   │       └── final_eval/
│   │           ├── seed_0.npy        # Test predictions per seed
│   │           ├── seed_0.json     # Training history per seed
│   │           ├── seed_1.npy
│   │           ├── seed_1.json
│   │           ├── seed_2.npy
│   │           ├── seed_2.json
│   │           └── best_model.pth
│   ├── fmd/
│   │   └── {experiment_id}/
│   │       ├── config.yaml
│   │       ├── status.json
│   │       ├── hp_phase1/
│   │       │   ├── fold_0_hp_0.npy   # Raw heatmaps per fold per HP config
│   │       │   ├── fold_0_hp_0.json   # Training history per fold per HP config
│   │       │   ├── fold_0_hp_0_nms.json   # NMS results per fold per HP config
│   │       │   └── ...
│   │       ├── hp_phase2/
│   │       │   ├── fold_0_hp_3.npy   # Raw heatmaps per fold, top 3 configs only
│   │       │   ├── fold_0_hp_3.json   # Training history per fold, top 3 configs only
│   │       │   ├── fold_0_hp_0_nms.json   # NMS results per fold, top 3 configs only
│   │       │   └── ...
│   │       └── final_eval/
│   │           ├── seed_0.npy        # Post-NMS minutiae point list per seed
│   │           ├── seed_0.json     # Training history per seed
│   │           ├── seed_1.npy
│   │           ├── seed_1.json
│   │           ├── seed_2.npy
│   │           ├── seed_2.json
│   │           ├── representative_maps/   # Raw heatmaps, 5 images only
│   │           └── best_model.pth
│   └── foe/
│       └── {experiment_id}/
│           ├── config.yaml
│           ├── status.json
│           ├── hp_phase1/
│           │   ├── fold_0_hp_0.npy   # Orientation map predictions per fold
│           │   ├── fold_0_hp_0.json   # Training history per fold per HP config
│           │   └── ...
│           └── final_eval/
│               ├── seed_0.npy        # Orientation map predictions per seed
│               ├── seed_0.json     # Training history per seed
│               ├── seed_1.npy
│               ├── seed_1.json
│               ├── seed_2.npy
│               ├── seed_2.json
│               ├── representative_maps/   # Resulting orientation maps, 5 images only
│               └── best_model.pth
│
├── results/                          # Populated by analysis layer
│   └── master_results.csv
│
└── src/                              # Code
    │   └── fc/                       # FC task-specific code
    │   └── fmd/                      # FMD task-specific code
    │   └── foe/                      # FOE task-specific code
    ├── config/
    ├── data/
    ├── models/
    ├── training/
    ├── evaluation/
    ├── experiment/
    └── analysis/
```

---

## 3. Module Architecture

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────┐
│                        CLI / Launcher                    │
│              run_experiment.py  |  generate_configs.py   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   ExperimentRunner                       │
│      orchestrates HP tuning and final eval with resume  │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
        ┌──────────▼──────────┐  ┌────▼─────────────┐
        │      HP Tuning       │  │   Final Eval      │
        │  Phase1: subset CV   │  │  best config      │
        │  Phase2: full CV     │  │  × 3 seeds        │
        │  (FC, FMD)           │  │  (all tasks)      │
        │  Phase1 only: FOE    │  └──────────────────┘
        └──────────┬──────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                    Trainer                               │
│         train_one_fold / train_full (step-based)         │
└──────┬──────────────────────────┬───────────────────────┘
       │                          │
┌──────▼──────┐           ┌───────▼──────┐
│  DataModule │           │  ModelFactory│
│             │           │              │
│ RealDataset │           │  M1/M2/M3    │
│ SynthDataset│           │  per task    │
│ CombinedDS  │           └──────────────┘
└─────────────┘

       Trainer saves predictions to disk after each fold/seed
                         │
          ┌──────────────┴──────────────────┐
          │                                 │
┌─────────▼──────────────┐      ┌───────────▼───────────────┐
│  Evaluator.score_fold() │      │  Evaluator.evaluate_full() │
│  FC/FOE: 1 scalar       │      │  (offline — Analysis Layer)│
│  FMD Ph1: 49 scalars    │      │  all metrics, all tasks    │
│    via fmd_nms.sweep()  │      │                            │
│  FMD Ph2: 1 scalar      │      └───────────┬───────────────┘
└─────────┬───────────────┘                  │
          │                       ┌────────────▼──────────────┐
┌─────────▼──────────────┐        │       Analysis Layer       │
│     HPSelector          │        │  aggregator, tables, plots │
│  rank → top3 → best     │        └───────────────────────────┘
│  FC/FOE: by 1 scalar    │
│  FMD: by 1176 triplets  │
└─────────────────────────┘
```

---

### 3.2 Config System (`src/config/`)

```
src/config/
├── schema.py        # Pydantic models for config validation
├── loader.py        # YAML loading with inheritance resolution
└── generator.py     # Programmatic experiment config generation
```

**Config inheritance chain (resolved at load time):**

```
global_defaults.yaml
  └── fc/base.yaml  (or fmd/ or foe/)
        └── fc/dev.yaml  (if --dev flag)
              └── fc/experiments/fc_M2_V1_S02.yaml
```

Each level only specifies deltas. The loader resolves the full config and writes a snapshot to `outputs/{experiment_id}/config.yaml` before any training begins.

**Experiment config fields (minimal, rest inherited):**

```yaml
experiment_id: fc_M2_V1_S02
task: fc
model: M2
condition: V1
synthetic_source: S02
real_n: 3200
synthetic_n: 3200
split_file_real: splits/fc/real_split.json
split_file_synthetic: splits/fc/synthetic/S02_V1.json
```

**Config generator** (`generator.py`):

Takes an experiment matrix definition (all tasks × models × conditions × sources) and emits one YAML file per experiment. Run once before experiments start. Idempotent — skips already-existing configs.

```python
# Usage
python -m src.config.generator --matrix configs/experiment_matrix.yaml --out configs/
```

**Pydantic schema** validates every config at load time — catches missing fields, wrong types, and invalid combinations before any GPU time is spent.

---

### 3.3 Data Module (`src/data/`)

```
src/data/
├── base_dataset.py       # Abstract base class
├── real/
│   ├── nist_sd4.py       # Existing FC dataset class (adapted)
│   ├── fvc_subset.py     # Existing FMD dataset class (adapted)
│   └── foe_samples.py    # Existing FOE dataset class (adapted)
├── synthetic/
│   └── synthetic_dataset.py   # Generic synthetic dataset class
├── combined_dataset.py        # Concatenates real + synthetic
└── split_generator.py         # Generates and saves all splits
```

**Base interface (all datasets conform to this):**

```python
class FingerprintDataset(Dataset):
    def __init__(self, base_path: Path, fp_ids: list[str], transform=None):
        ...
    def __getitem__(self, idx) -> dict:
        # returns: {image, label, fp_id, quality (if available)}
        ...
```

**CombinedDataset:**

```python
class CombinedDataset(Dataset):
    def __init__(self, real_dataset, synthetic_dataset):
        # Concatenates real + synthetic
        # Tracks source (real / synthetic) per sample
        # No mixing of identities — real and synthetic are separate pools
        ...
```

**SplitGenerator** (`split_generator.py`):

Generates and saves all splits before any experiment runs. Outputs JSON files to `splits/`. Once generated, splits are never regenerated — the JSON files are the ground truth.

```python
# Split file format
{
  "task": "fc",
  "source": "Denkovarik",
  "condition": "V1",
  "real_n": 3200,
  "synthetic_n": 3200,
  "sampling_strategy": "identity_level",
  "train_ids": [...],
  "val_ids": [...],   # per fold: [[fold0_val], [fold1_val], ...]
  "test_ids": [...]
}
```

Run once:
```python
python -m src.data.split_generator --config configs/global_defaults.yaml
```

---

### 3.4 Model Factory (`src/models/`)

```
src/models/
├── factory.py              # build_model(task, model_id, config) -> nn.Module
├── encoders/
│   ├── resnet50.py         # Wrapper: scratch or pretrained
│   └── swin_v2_t.py        # Pretrained only
├── decoders/
│   └── unet_decoder.py     # Shared UNet-style decoder, configurable depth
└── heads/
    ├── classification_head.py    # FC linear head
    ├── fmd_head.py               # 3-channel full-res output
    └── foe_head.py               # 128-channel W/8 x H/8 output
```

**Factory interface:**

```python
def build_model(task: str, model_id: str, config: Config) -> nn.Module:
    # task: 'fc' | 'fmd' | 'foe'
    # model_id: 'M1' | 'M2' | 'M3'
    # Returns assembled model ready for training
    ...
```

**Model assembly:**

| Task | M1 | M2 | M3 |
|---|---|---|---|
| FC | ResNet50(scratch) + ClassHead(5) | ResNet50(pretrained) + ClassHead(5) | SwinV2T(pretrained) + ClassHead(5) |
| FMD | ResNet50(scratch) + UNetDecoder(4 stages) + FMDHead(3ch) | ResNet50(pretrained) + UNetDecoder(4 stages) + FMDHead(3ch) | SwinV2T(pretrained) + UNetDecoder(4 stages) + FMDHead(3ch) |
| FOE | ResNet50(scratch) + UNetDecoder(2 stages) + FOEHead(128ch) | ResNet50(pretrained) + UNetDecoder(2 stages) + FOEHead(128ch) | SwinV2T(pretrained) + UNetDecoder(2 stages) + FOEHead(128ch) |

The decoder depth (4 stages for FMD, 2 stages for FOE) is config-driven, not hardcoded. Head output: FC=5-class softmax, FMD=3-channel per-pixel softmax map (background/ridge-ending/bifurcation), FOE=128-channel per-pixel sigmoid map (cyclic encoding).

---

### 3.5 Trainer (`src/training/`)

```
src/training/
├── trainer.py              # Core training loop (plain PyTorch, step-based)
├── losses/
│   ├── fc_loss.py          # Cross-entropy, 5-class
│   ├── fmd_loss.py         # Weighted cross-entropy, 3-class per-pixel
│   └── foe_loss.py         # Binary cross-entropy per bit, 128-channel
├── transforms/
│   ├── common.py           # Augmentations via Albumentations
│   ├── foe_transforms.py   # FOE-specific: orientation value updates after flip/rotation
│   └── foe_crop.py         # FOE-specific: paired crop with 1/8 scaling for label map
└── utils.py                # LR schedulers, gradient clipping, mixed precision
```

**Trainer interface:**

```python
def train_one_fold(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    hp_config: HPConfig,
    device: str,
    output_path: Path,
    skip_if_exists: bool = True
) -> FoldResult:
    # Runs for max_steps gradient updates (not epochs)
    # Saves predictions to output_path at completion
    # Returns immediately if output already exists and skip_if_exists=True
    ...
```

**Step-based training:** Training runs for `max_steps` gradient updates, defined per task in the base config. This ensures comparable training compute across dataset sizes — particularly important for FOE (N=48) vs FC (N=3,200). Cosine annealing scheduler is expressed in total steps.

**CORRECTION (2026-07-06):** batch size is **not** auto-probed against
available VRAM despite the claim that used to be here. It's a fixed value
hardcoded per task in `configs/{task}/base.yaml` (`fc: 256, fmd: 128,
foe: 32`, with an fc/M3 override to 64 in `src/config/generator.py`) —
see `trainer.py`'s `batch_size = max(8, config.batch_size) if
config.dev_mode else config.batch_size`. Note `--dev` mode only *floors*
batch size at 8, it never reduces it below the configured value — so a
dev smoke test uses the exact same batch size as a full run, making it a
valid real VRAM check when moving to different GPU hardware (found while
migrating from a 96GB RTX Pro 6000 to a 64GB H100 cluster — if these fixed
sizes don't fit, it's a one-line edit to the relevant `base.yaml`, not a
code change).

**Loss functions:**

| Task | Loss | Notes |
|---|---|---|
| FC | Cross-entropy | Primary label as ground truth; multi-label tolerance at evaluation only |
| FMD | Weighted cross-entropy (3-class per-pixel) | Class weights required — background dominates label map heavily |
| FOE | Binary cross-entropy per bit (128-channel sigmoid) | Cyclic 128-bit encoding; each bit treated as independent binary classification |

**Augmentation implementation:** All augmentations applied via Albumentations as paired (image, label) transforms with shared random state. Image-only augmentations (brightness, contrast, blur, noise, gamma) applied to image only. Spatial augmentations (flip, rotation, crop, elastic deformation) applied to both.

FOE-specific exceptions:
- After horizontal flip: orientation values updated (negate sin component of encoding)
- After rotation by α: orientation values shifted by α before re-encoding
- Center crop: image cropped normally; orientation map cropped at coordinates scaled by 1/8
- Elastic deformation: not applied to FOE

**FC loss note:** Primary label used as ground truth in training loss. The second valid class (where present) is handled only at evaluation time.

**FMD class weight note:** Background class weight should be set inversely proportional to its frequency in the training labels, computed from the training split and saved as part of the experiment config.

---

### 3.6 Experiment Runner (`src/experiment/`)

```
src/experiment/
├── runner.py           # ExperimentRunner: orchestrates HP tuning and final evaluation
├── hp_phase1.py        # Subset CV grid search (FC, FMD) / full CV (FOE)
├── hp_phase2.py        # Full CV on top 3 configs/triplets (FC, FMD only)
├── fmd_nms.py          # NMS detection: applies (threshold, radius) to a heatmap, returns detected minutiae point list. No scoring.
├── final_eval.py       # Final evaluation across seeds
├── hp_selector.py      # Ranks scored configs/triplets, selects top-3 and best
└── status.py           # Reads/writes status.json
```

**Status file (`status.json`):**

```json
{
  "experiment_id": "fc_M2_V1_S02",
  "status": "hp_phase2_complete",
  "hp_phase1": {
    "status": "complete",
    "completed_folds": [0, 1, 2, 3],
    "top3_configs": [{"hp_config": 7}, {"hp_config": 12}, {"hp_config": 3}]
  },
  "hp_phase2": {
    "status": "complete",
    "completed_folds": [0, 1, 2, 3],
    "best_config_index": 7,
    "best_config": {"lr": 1e-4, "wd": 1e-5, "cosine": true}
  },
  "final_eval": {
    "status": "pending",
    "completed_seeds": []
  }
}
```

For FMD, top3_configs will also include the NMS parameters, eg. [{"hp_config": 7, "nms_threshold": 0.4, "nms_radius": 8}, {"hp_config": 12, "nms_threshold": 0.3, "nms_radius": 6}, {"hp_config": 3, "nms_threshold": 0.4, "nms_radius": 10}]. Similarly, best_config will include "nms_threshold" and "nms_radius" fields.


**ExperimentRunner logic (pseudocode):**

```python
def run(self):
    self.status = load_or_init_status()

    if self.status.hp_phase1 != 'complete':
        self.run_hp_phase1()           # trains all 24 configs × 4 folds, saves predictions
        self.select_top3()             # FC/FMD: top 3 configs or triplets; FOE: selects best directly (no phase 2)
        self.status.hp_phase1 = 'complete'
        save_status()

    if self.task != 'foe' and self.status.hp_phase2 != 'complete':
        self.run_hp_phase2()       # full CV on top 3 for FC/FMD
        self.select_best_config()
        self.status.hp_phase2 = 'complete'
        save_status()

    if self.status.final_eval != 'complete':
        self.run_final_eval()      # 3 seeds
        self.save_best_model()
        self.status.final_eval = 'complete'
        save_status()
```

Every write to `status.json` is atomic (write to temp file, then rename) to prevent corruption on interrupt.

---

### 3.7 Evaluator (`src/evaluation/`)

```
src/evaluation/
├── evaluator.py         # Dispatches to task-specific evaluator
├── fc_evaluator.py
├── fmd_evaluator.py
└── foe_evaluator.py
```

The evaluator has two distinct modes. It is never called *inside* the training loop — predictions are always saved first, then the evaluator reads from disk.

---

**Mode 1 — `score_fold()` (online, called by ExperimentRunner during HP tuning):**

Returns a ranking metric from saved fold predictions. The return type differs by task:

| Task | Input | `nms_params` | Returns | Ranking metric |
|---|---|---|---|---|
| FC | Saved class probability vectors | — | 1 scalar | Top-1 accuracy (multi-label tolerant) |
| FMD | Saved raw heatmaps | `None` (Phase 1) | 49 scalars | F1 at fixed matching distance, one per NMS combination |
| FMD | Saved raw heatmaps | Fixed `(threshold, radius)` (Phase 2) | 1 scalar | F1 at fixed matching distance |
| FOE | Saved orientation maps | — | 1 scalar | Mean angular error (negated) |

**FMD Phase 1 internal flow inside `score_fold()`:**

For each saved heatmap, `fmd_nms.sweep()` is called. `fmd_nms.sweep()` applies NMS at every one of the 49 (threshold, radius) combinations and returns 49 detected minutiae point lists — one per combination. No scoring occurs inside `fmd_nms`. `score_fold()` then computes F1 at a fixed matching distance for each of the 49 point lists against the ground truth, returning 49 scalars.

**HPSelector — accumulation and ranking:**

HPSelector accumulates `score_fold()` outputs across all folds for each HP config, then ranks:

- **FC / FOE Phase 1:** 24 configs × 1 scalar per fold → mean across folds → 24 scores → rank → select top 3 (FC) or best directly (FOE)
- **FMD Phase 1:** 24 configs × 49 scalars per fold → mean across folds per (HP, NMS) combination → 1176 triplet scores → rank jointly → select top 3 triplets
- **FC / FMD Phase 2:** 3 configs or triplets × 1 scalar per fold → mean across folds → 3 scores → rank → select best

Each selected triplet for FMD fully specifies: HP config index, NMS detection threshold, and NMS radius. The selected triplet is saved to `status.json` and `config.yaml` and used consistently in final evaluation.

> **NMS radius constraint:** NMS radius must be ≤ matching distance for all grid entries. Suppressing points at a radius larger than the matching distance would eliminate predictions that could have been valid distinct matches. Grid values must be pre-specified to satisfy this constraint.

> **Note:** Raw heatmaps are saved per fold during Phase 1 and Phase 2 for FMD. They are not saved during final evaluation — only post-NMS point lists are saved for the test set, except for 5 representative images kept for qualitative visualisation.

---

**FMD minutiae matching — one-to-one assignment:**

Predicted and GT minutiae are matched using one-to-one bipartite matching — each predicted minutia can be assigned to at most one GT minutia and vice versa. This reflects the physical reality that each ridge ending or bifurcation is a single event.

Matching procedure (greedy, sufficient for typical minutiae counts):
1. Compute all pairwise distances between predicted and GT minutiae
2. Sort candidate pairs by distance (ascending)
3. Assign greedily — accept a pair if neither point is already assigned and distance ≤ matching distance threshold
4. Unmatched predictions → false positives; unmatched GT minutiae → false negatives

```
TP = number of matched pairs
FP = |P| - TP   (unmatched predictions)
FN = |G| - TP   (unmatched GT minutiae)
Precision = TP / |P|
Recall    = TP / |G|
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Computed per image, then averaged across images, and also globally by aggregating TP/FP/FN across all images before computing P/R/F1.

---

**Mode 2 — `evaluate_full()` (offline, called by Analysis Layer after final evaluation):**

Computes all metrics from test predictions using the NMS parameters from the selected triplet. Called once per experiment after all seeds are complete.

**FC metrics** (from saved class probability vectors):
- Top-1 accuracy (multi-label tolerant)
- Per-class accuracy
- Confusion matrix

**FMD metrics** (from saved post-NMS minutiae point lists):
- Precision, Recall, F1 at 3 fixed matching distances
- Per-type breakdown (ridge ending, bifurcation)
- Per-quality breakdown (poor, fair, good)
- Computed per-image then averaged, and globally across all minutiae

**FOE metrics** (from saved orientation maps):
- Mean angular error per image, then averaged
- Mean angular error globally across all orientations
- Accuracy at different angular error thresholds
- Stratified by image quality (B/G)

---

**Call structure across stages:**

```
HP Phase 1 — FC (subset CV):
  for each hp_config (24):
    for each fold (4):
      train_one_fold()            → saves class probability vectors to disk
      score_fold()                → returns 1 scalar (top-1 accuracy)
  HPSelector.rank()               → ranks 24 configs, selects top 3

HP Phase 1 — FMD (subset CV):
  for each hp_config (24):
    for each fold (4):
      train_one_fold()            → saves raw heatmaps to disk
      score_fold(nms_params=None) → calls fmd_nms.sweep() on each heatmap
                                    fmd_nms.sweep() returns 49 minutiae lists
                                    score_fold() computes F1 for each
                                    → returns 49 scalars
  HPSelector.rank()               → ranks 1176 (HP, threshold, radius) triplets
                                    by mean F1 across folds, selects top 3 triplets

HP Phase 1 — FOE (full CV, merged with phase 2):
  for each hp_config (24):
    for each fold (4):
      train_one_fold()            → saves orientation maps to disk
      score_fold()                → returns 1 scalar (mean angular error, negated)
  HPSelector.rank()               → ranks 24 configs, selects best directly

HP Phase 2 — FC and FMD (full CV, top 3 only):
  for each top3_config_or_triplet (3):
    for each fold (4):
      train_one_fold()            → saves predictions to disk
      score_fold(nms_params)      → returns 1 scalar
                                    FC: top-1 accuracy
                                    FMD: F1 at triplet's fixed (threshold, radius)
  HPSelector.select_best()        → selects single best config (FC) or triplet (FMD)

Final Evaluation — all tasks:
  for each seed (3):
    train_full()                  → saves test predictions to disk
  # Evaluator.evaluate_full() called later by Analysis Layer
```

---

### 3.8 Analysis Layer (`src/analysis/`)

```
src/analysis/
├── aggregator.py        # Scans all output folders, builds master results table
├── tables.py            # Generates paper-ready LaTeX and CSV tables
├── plots.py             # Generates all paper figures
└── status_report.py     # Prints completion status across all experiments
```

**Aggregator** scans `outputs/{task}/{experiment_id}/status.json` for all experiments, collects stage3 predictions, computes metrics via the Evaluator, and builds a master DataFrame saved as `results/master_results.csv`.

**Status report** (run anytime to check progress):

```python
python -m src.analysis.status_report
```

Output example:
```
FC   M1   V0         complete   3/3 seeds
FC   M1   V1   S01   complete   3/3 seeds
FC   M2   V1   S02   running    2/3 seeds
FC   M3   V1   S03   pending    0/3 seeds
...
702 total | 487 complete | 12 running | 203 pending
```

**Plots generated:**

| Plot | X axis | Y axis | Series |
|---|---|---|---|
| Volume scaling curve | Synthetic volume (0N/1N/3N/7N) | Task metric | Per source (top 3) |
| Substitution curve | Real fraction (100%→0%) | Task metric | Per source (best) |
| Source comparison bar | Synthetic source | Task metric | Per model (M1/M2/M3) |
| M1 vs M2 vs M3 | Model | Task metric | Per condition |
| Label tier comparison | Label tier | Task metric | Per task |
| Confusion matrix | — | — | FC only |
| P/R curve | Recall | Precision | Per source, FMD only |
| FMD quality stratification | Quality (poor/fair/good) | P/R/F1 | Per source, FMD only |
| FOE quality stratification | Quality (B/G) | Mean angular error | Per source, FOE only |

---

### 3.9 CLI (`run_experiment.py`)

Single entry point for launching one experiment:

```bash
# Full run
python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml

# Dev/smoke test
python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml --dev

# Resume (default behaviour — skips completed stages automatically)
python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml

# Force rerun from stage 1 (ignores existing outputs)
python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml --force
```

The `--dev` flag loads `dev.yaml` overrides on top of the experiment config:
- Reduced subset size for HP Phase 1 (FC and FMD)
- 2 folds instead of 4
- 2 HP configs instead of 24
- 2 seeds instead of 3
- Fewer training steps
- Smaller batch size

---

## 4. Framework: Plain PyTorch

The training loop is implemented in plain PyTorch without a higher-level wrapper. The training control flow — step-based stopping, per-fold prediction saving, skip-if-exists resumability, task-specific loss functions, paired augmentation transforms — is specialised enough that Lightning or similar abstractions would add overhead without simplifying the core logic.

Plain PyTorch features used directly:
- `torch.cuda.amp` for mixed precision (automatic loss scaling)
- `torch.optim` with cosine annealing scheduler (`CosineAnnealingLR` expressed in steps)
- `torch.utils.data.DataLoader` with pin memory and persistent workers
- `torch.compile` (optional, for additional throughput on supported hardware)

---

## 5. Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Framework | PyTorch (plain) | Full control over step-based training loop and prediction saving |
| Augmentation | Albumentations | Native paired (image, label) transform support with shared random state |
| Config | YAML + Pydantic | Human-readable, validated, inheritance-capable |
| Experiment tracking | Custom status.json | Lightweight, no server needed, single-machine |
| Model zoo | timm | ResNet-50 and Swin-V2-T with versioned pretrained weights |
| Mixed precision | torch.cuda.amp | Native PyTorch, no wrapper needed |
| Data | Custom Dataset classes | Already started, clean interface |
| Analysis | pandas + matplotlib/seaborn | Sufficient for paper figures, no overhead |
| Version control | Git + GitHub | Already in use |
| Environment | conda or venv + requirements.txt | Reproducibility across machines |

---

## 6. Implementation Order

Build in this order — each layer is testable before the next is built:

1. **Config system** — schema, loader, generator. Validate all 702 experiment configs generate correctly.
2. **Split generator** — generate and save all splits. Verify identity-level sampling and stratification.
3. **Dataset classes** — adapt existing classes to base interface. Add SyntheticDataset and CombinedDataset. Unit test each.
4. **Model factory** — implement all M1/M2/M3 for all tasks. Verify output shapes for each task.
5. **Trainer** — single fold training loop with prediction saving and skip logic. Smoke test on dev config.
6. **Evaluator (`score_fold`)** — implement the ranking metric per task: single scalar for FC and FOE; 49 scalars (one per NMS combination) for FMD Phase 1, 1 scalar for FMD Phase 2.
7. **HP tuning stage** — full CV grid search with HPSelector. Verify top-3 selection and best config logic end to end.
8. **Final evaluation stage** — seed loop with prediction saving and best model checkpoint.
9. **ExperimentRunner** — assemble HP tuning and final eval with status tracking. Test resume after interrupt.
10. **Evaluator (`evaluate_full`)** — implement all remaining metrics per task. Can be built in parallel with steps 7–9.
11. **Analysis layer** — aggregator, tables, plots. Run after first real experiments complete.

---

## 7. Key Conventions

- **Experiment ID format:** `{task}_{model}_{condition}_{source}` e.g. `fc_M2_V1_S02`, `foe_M1_R3_S11`, `fmd_M3_V0` (no source suffix for V0/R0)
- **All paths relative to project root** — no absolute paths in any config or code
- **All randomness seeded** — PyTorch, NumPy, and Python random seeds set from config at the start of every run
- **Predictions saved as .npy** — consistent binary format across all tasks; visualisation handled by dedicated tools not raw file inspection
- **One output folder per experiment** — never shared across experiments
- **Config snapshot saved first** — before any training, the resolved config is written to the output folder

---

