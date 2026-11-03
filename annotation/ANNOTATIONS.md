# Synthetic Fingerprint Data — Sources & Annotation Methodology

This directory documents the 11 synthetic fingerprint image sources
(`S01`–`S11`) used in the study, covering both how each source was
originally generated (this section) and how labels were subsequently
produced for them (see "Annotations — Methodology" below).

## 1. Synthetic Data Sources

Eleven synthetic sources were assembled, spanning five generator
families: classical procedural synthesis (SFINGE, ANGULI) and three
generations of learned image synthesis (GAN, StyleGAN, StyleGAN2),
including two hybrid procedural+learned pipelines (ANGULI+StyleGAN2-ada,
ANGULI+pix2pix). No single source was generated for this study; all were
obtained as existing datasets or generators and pooled under a common
sampling and annotation pipeline.

| ID | Full name | Generator family | Native resolution | Native structure | Total images available | Notes |
|---|---|---|---|---|---|---|
| S01 | FVC-SFINGE-pool | SFINGE | 240–288×320–384 | Pooled FVC2000–2006 DB4 A+B sets | ~4,440 | Fixed pool; cannot be extended. Hard capacity ceiling. |
| S02 | Denkovarik | ANGULI | 206×300 | 1,000 fingers × (100 bad + 10 good impressions) | 110,000 | Multi-impression; quality labels available at source |
| S03 | ICB2018 | GAN | 512×512 | Generated on demand | Unlimited | Generator available; single impression |
| S04 | ICIP2021Clarkson | StyleGAN | 512×512 | 50,047 × 1 impression | 50,047 | Single impression |
| S05 | PAMI2023MSUPrintsGAN | GAN | 512×512 | 35,000 fingers × 15 impressions | 525,000 | Multi-impression; pending replacement by MSUGenPrint (S12, access not yet authorized) |
| S06 | SynFing | StyleGAN2 | 512×512 | 100,000 fingers × 2 impressions | 200,000 | Multi-impression |
| S07 | VISAPP2023SGRv1 | ANGULI + StyleGAN2-ada | 512×512 | 50,000 × 1 impression | 50,000 | Hybrid procedural + learned pipeline; single impression |
| S08 | VISAPP2023P2Pv1 | ANGULI + pix2pix | 512×512 | (500 fingers × 8) × 10 impressions | 40,000 | Hybrid; multi-impression |
| S09 | VISAPP2023P2Pv2 | ANGULI + pix2pix | 512×512 | (500 fingers × 8) × 10 impressions | 40,000 | Hybrid; multi-impression, alternate variant of S08's pipeline |
| S10 | WACV2024 | StyleGAN2 | TBD | Generated on demand | Unlimited | Generator available; multi-impression |
| S11 | BAJECE2025 | SFINGE-like | 400×400 | 25,000 × 1 impression | 25,000 | Single impression; native class and orientation (W/8×H/8) labels available at source, independent of this project's annotation pipeline |

**Sampling for this study.** Sources are not used in full — each is
subsampled per experimental condition (see `experiment_design.md` §4.2
for the full strategy):

- **Multi-impression sources** (S02, S05, S06, S08, S09) are sampled at
  the finger/identity level — identities are drawn first, then all
  impressions up to the target count are kept together — to prevent an
  identity's impressions from leaking across train/validation folds.
- **Single-impression sources** are sampled at the image level.
- **Generator-based sources** (S03, S10) are generated on demand to the
  exact count required; no subsampling step is needed.
- **S01** is capacity-constrained: when a condition's target exceeds the
  ~4,440-image pool, all available images are used and the shortfall is
  reported explicitly rather than padded or extrapolated.

Separately, the `README.md`'s "Dataset scope and sampling" table below
records the specific counts actually drawn from each source for the
annotation pass described in this document (capped at ~30k per source
for annotation-pipeline validation purposes), which is independent of
the per-experiment-condition sampling above.

---

# Annotations — Methodology

This directory contains automatically generated labels for 11 synthetic
fingerprint image sources (`S01`–`S11`). Each image has three label files:

| Label | Extension | Content | Generated with |
|---|---|---|---|
| FC (pattern classification) | `.txt` | gender/class/history, 3 lines | Neurotechnology VeriFinger SDK |
| FMD (minutiae) | `.mnt` | minutia count + `x y type angle` per line | Neurotechnology VeriFinger SDK |
| FOE (orientation field) | `.gt` | raw uint8 block-orientation map | pyfing (SNFOE) |

Exact byte-level format for each file is specified in `LABEL_FORMAT.md`
(same repository). Labels are written in the same format used by common
real fingerprint datasets (e.g. NIST SD4 for FC, FVC for FMD), so
dataset-loading code written for real fingerprint data should be able to
read these synthetic annotations without modification.

## How annotations were generated

**FC and FMD** were extracted with the VeriFinger SDK, run per image:
grayscale-normalized, resolution set explicitly (see DPI note below,
since these generated images carry no reliable embedded resolution),
classified/templated with VeriFinger's pattern-classification and
large-template minutiae extraction. FC's 6-way pattern class enum was
mapped to the standard 5-class scheme `{A, L, R, T, W}` (arch, left loop,
right loop, tented arch, whorl); classes with no mapping under that scheme
were left unlabeled rather than guessed. `Gender` is always written as
`U` (unknown) and `History` self-references the same image/class, since
these are the correct values for synthetic images under LABEL_FORMAT.md's
convention (no real gender or paired-impression record exists).

**FOE** was extracted with `pyfing`'s SNFOE (Simple Network for
Fingerprint Orientation Estimation), an open-source learned orientation
estimator, run per image at the source's calibrated DPI.

**Per-source DPI**: synthetic generators typically don't model physical
scale, so resolution was estimated per source from median ridge spacing
(via a ridge-frequency estimator) and calibrated against two independent,
real, confirmed-500dpi reference datasets. Eight sources matched the
expected ~500dpi range and used that default; three sources (`S01`,
`S02`, `S11`) measured meaningfully lower and were processed at their own
measured resolution instead (~385/~375/~390dpi respectively).

## Validation

Most of these synthetic sources have no independent ground truth of their
own, so the identical extraction pipeline (same code, same configuration)
was instead validated against real, independently-labeled fingerprint
datasets before being applied here:

| Label | Real reference dataset | n images | Result |
|---|---|---|---|
| FC | NIST SD4 | 4,000 | 86.65% classification accuracy |
| FMD | FVC2002/2004, DB1A + DB3A | 3,143 | 67.2% recall / 54.8% precision (nearest-neighbor minutiae matching against ground truth) |
| FOE | FOESamples (good + low-quality scans) | 60 | 3.46° mean / 2.57° median orientation error |

These figures characterize the extraction pipeline's accuracy on real
fingerprints in general, not these specific synthetic images — they are
the best available proxy for expected annotation quality on sources
without their own ground truth, not a per-image guarantee.

**Known weak points**, consistent with typical automatic fingerprint
systems rather than specific to this pipeline: tented-arch (`T`) is the
most error-prone FC class, frequently confused with plain arch (`A`) on
real data; FMD recall/precision varies somewhat by source style.

### S11: directly validated against its own ground truth

`S11` is the one exception — it ships real per-image ground truth
alongside the images (class encoded in the filename; a `{stem}_o.txt`
block-orientation grid), so its shipped FC/FOE annotations here were
checked directly, not by proxy:

| Label | n | Result |
|---|---|---|
| FC | 30,001 | **87.69% accuracy** (per-class recall: A 98.5%, L 95.4%, R 97.4%, T 53.6%, W 92.9%) |
| FOE | 30,001 images / 49,080,409 blocks | **3.67° mean / 2.70° median error**, 95.6% within 10°, 99.2% within 20° |

Consistent with the real-data proxy numbers above, giving an independent
cross-check that those figures generalize to synthetic images too. `T`
(tented arch) is again the weakest FC class, as on real data — but the
confusion pattern differs: real NIST SD4 mostly confuses `T→A`, while
`S11` mostly confuses `T→L`/`T→R`, with `T→A` comparatively rare. Not yet
investigated whether that's a generator-specific rendering quirk.

## Dataset scope and sampling

| Source | Images annotated | Sampling |
|---|---|---|
| S01 | 4,437 | full source |
| S02 | 30,024 | capped at ~30k, identity-level (all impressions of each sampled identity kept together) |
| S03 | 25,000 | full source |
| S04 | 30,000 | capped at 30k, random per-image sampling |
| S05 | 30,003 | capped at ~30k, identity-level |
| S06 | 30,002 | capped at ~30k, identity-level |
| S07 | 30,000 | capped at 30k, random per-image sampling |
| S08 | 30,001 | capped at ~30k, identity-level |
| S09 | 30,001 | capped at ~30k, identity-level |
| S10 | 25,000 | full source |
| S11 | 30,001 | capped at 30k, random per-image sampling |

Sources where a single identity contributes many impressions were sampled
at the identity level, not as scattered individual images, so a capped
sample doesn't lose impression-coverage diversity.

## Attribution

- FC/FMD annotations were produced using the Neurotechnology VeriFinger
  SDK (commercial license). The SDK itself is not redistributed with this
  dataset — only its output labels.
- FOE annotations were produced using `pyfing` (open source), SNFOE
  method.
