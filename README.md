# Task-Dependent Utility of Synthetic Fingerprint Images

Code, experiment configurations, data splits, annotation pipeline and results for:

> N. Erdoğmuş, "Task-Dependent Utility of Synthetic Fingerprint Images,"
> *Journal of Electronic Imaging* (SPIE), 2026.

The study asks whether synthetic fingerprint images help train models for three
fingerprint analysis tasks, and which generation methods help most. Eleven
synthetic sources are compared under a controlled factorial design (702
experiments).

| Task | Description | Real dataset |
|---|---|---|
| FC | Fingerprint pattern classification (A, L, R, T, W) | NIST SD4 |
| FMD | Minutiae detection | FVC2002/2004 DB1A+DB3A subset |
| FOE | Orientation field estimation | FOESamples |

| Model | Encoder | Initialisation |
|---|---|---|
| M1 | ResNet-50 | From scratch |
| M2 | ResNet-50 | ImageNet pretrained |
| M3 | Swin-V2-T | ImageNet pretrained |

Conditions: volume (V0–V3, real data fixed and synthetic data added) and
substitution (R1–R4, total fixed and real data replaced by synthetic). See
[docs/experiment_design.md](docs/experiment_design.md).

---

## Data

**No fingerprint images are stored in this repository.**

### Released with this paper (Zenodo)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23926910.svg)](https://doi.org/10.5281/zenodo.23926910)

**DOI: [10.5281/zenodo.23926910](https://doi.org/10.5281/zenodo.23926910)**

| File | Contents |
|---|---|
| `annotations.zip` | Automatically generated FC / FMD / FOE labels for 294,459 synthetic images from 11 sources (`.txt`, `.mnt`, `.gt`), plus per-source manifests. License: CC BY-NC 4.0 |
| `S03_generated.zip` | 25,000 images generated with the [ICB 2018 fingerprint synthesis](https://github.com/prip-lab/fingerprint-synthesis) generator (Cao & Jain) |
| `S10_generated.zip` | 25,000 images generated with FPGAN-Control (WACV 2024) |

If you use the generated images, please also cite the original generator papers.

The training code reads labels from `annotation_workspace/annotations/`
relative to the repo root. Run `unzip annotations.zip -d annotation_workspace/`
from the repo root.

The label formats are documented in [annotation/LABEL_FORMAT.md](annotation/LABEL_FORMAT.md).
How the labels were produced and validated is described in
[annotation/ANNOTATIONS.md](annotation/ANNOTATIONS.md).

### Obtain independently

Get these from their original providers under their own licenses.

**Real datasets**

| Dataset | Task |
|---|---|
| NIST Special Database 4 | FC |
| FVC2002 DB1A, DB3A and FVC2004 DB1A, DB3A, with minutiae ground truth | FMD |
| FOESamples (FVC-onGoing FOE benchmark) | FOE |

**Synthetic datasets**

| ID | Dataset | Generator family |
|---|---|---|
| S01 | FVC2000–2006 DB4 A+B (SFinGe) | SFinGe |
| S02 | Denkovarik (Anguli) | Anguli |
| S04 | ICIP 2021 Clarkson | StyleGAN |
| S05 | PAMI 2023 MSU PrintsGAN | GAN |
| S06 | SynFing | StyleGAN2 |
| S07 | VISAPP 2023 SGRv1 | Anguli + StyleGAN2-ADA |
| S08 | VISAPP 2023 P2Pv1 | Anguli + pix2pix |
| S09 | VISAPP 2023 P2Pv2 | Anguli + pix2pix |
| S11 | BAJECE 2025 | SFinGe-like |

The expected directory layout is in [docs/REPRODUCTION.md](docs/REPRODUCTION.md) §3.
The container mounts real data at `/data/real/` and synthetic data at
`/data/synthetic/S01 … S11/`.

---

## Repository layout

```
├── run_experiment.py        # Entry point: runs one experiment end to end
├── src/                     # Data loading, models, training, evaluation, analysis
├── configs/                 # 702 experiment configs + global defaults
├── splits/                  # Exact real and synthetic splits used in the paper
├── results/
│   ├── master_results.csv           # All experiments (per-condition HP tuning)
│   ├── master_results_fixedhp.csv   # Fixed-hyperparameter control
│   ├── tables/                      # Paper tables (.csv and .tex)
│   └── figures/                     # Paper figures (.pdf and .png)
├── annotation/              # Synthetic-label pipeline
│   ├── src/                 # VeriFinger (FC, FMD) and pyfing SNFOE (FOE) labelling + validation
│   ├── container/           # Docker setup for the annotation pipeline
│   ├── manifests/           # Which images were annotated per source
│   └── validation_results/  # Label-quality evaluation against real ground truth
├── docs/                    # Experiment design, software architecture, full reproduction guide
└── .devcontainer/           # Training environment (PyTorch, CUDA 12.8)
```

---

## Quick start

Detailed steps are in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

1. Install Docker, the NVIDIA Container Toolkit and VS Code with Dev Containers.
2. Put the datasets on the host and edit the bind mounts in
   [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json).
3. Reopen the folder in the container, then run a smoke test:

   ```bash
   python3 run_experiment.py --config configs/fc/experiments/fc_M1_V0.yaml --dev --data-root /data
   ```

4. Run one full experiment:

   ```bash
   python3 run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml --data-root /data
   ```

5. Rebuild the results tables and figures from completed outputs:

   ```bash
   python3 -m src.analysis.aggregator --outputs outputs --data-root /data
   python3 -c "from src.analysis.tables import generate_tables; from src.analysis.plots import generate_plots; generate_tables(); generate_plots()"
   ```

The configs and splits in this repository are the exact ones used in the paper.
Regenerating them (REPRODUCTION.md §5) is optional.

### Annotation pipeline

FC and FMD labels were made with the **Neurotechnology VeriFinger SDK**, a
commercial product used under a trial license. The SDK is **not** included here.
To rerun [annotation/src/annotate_batch.py](annotation/src/annotate_batch.py),
get the SDK from Neurotechnology and follow
[annotation/container/README.md](annotation/container/README.md). FOE labels
use the open-source [pyfing](https://pypi.org/project/pyfing/) v0.6 (SNFOE).
You don't need the SDK to use the released labels.

---

## Citation

```bibtex
@article{erdogmus2026synthetic,
  author  = {Erdo{\u{g}}mu{\c{s}}, Nesli},
  title   = {Task-Dependent Utility of Synthetic Fingerprint Images},
  journal = {Journal of Electronic Imaging},
  year    = {2026},
  note    = {Volume, issue and DOI to be added}
}
```

Please also cite the original papers of every real and synthetic dataset you use.

## License

The code is under the MIT License (see [LICENSE](LICENSE)). The annotations on
Zenodo are under CC BY-NC 4.0.
