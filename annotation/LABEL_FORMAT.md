# Annotation Label Format Specification

Synthetic labels were designed to match the real dataset label formats. In
practice this only turned out to be true for FC — see the correction notes
under FMD and FOE below, added after `SyntheticDataset` (the actual training-
time consumer of these files) was found not to read them at all. That gap and
its fix are documented in `/workspace/CONTEXT.md`.

---

## FC — Fingerprint Classification

**Real source:** NIST_SD4  `png_txt/figs_*/f{id}.txt`

**Format** — plain text, 3 lines:
```
Gender: M
Class: W
History: f0001_01.pct W a0591.pct
```

- Line 1: `Gender: <M|F>`  (write `Gender: U` for unknown — synthetics have no gender)
- Line 2: `Class: <class>` where class ∈ {A, L, R, T, W, S}
  - A = arch, L = left loop, R = right loop, T = tented arch, W = whorl, S = scar
- Line 3: `History: <stem> <class> <paired_stem>` — for synthetics with no paired
  impression, repeat the stem and class:
  `History: {stem} {class} {stem}`

**Actual output path (corrected 2026-07-06):**
`/workspace/annotation_workspace/annotations/{source_id}/{stem}.txt`

e.g. `/workspace/annotation_workspace/annotations/S04/0--1.txt`

This one genuinely is loaded "without modification" — `FCGroundTruth.from_file`
(`src/data/real/fc_dataset.py`) is format-agnostic to path and is reused as-is
by `SyntheticDataset` for every source except S11 (which encodes its class in
the filename instead).

---

## FMD — Fingerprint Minutiae Detection

**CORRECTION (2026-07-06):** the claim below that real FMD ground truth is
`.mnt` is wrong. The actual real dataset (`FMDRealDataset` /
`FMDGroundTruth.from_file` in `src/data/real/fmd_dataset.py`) reads a
completely different format: `.npz` (numpy archive with explicit
`size_x`/`size_y`/`finger_qual`/`T`/`X`/`Y`/`A`/`Q` arrays). The `.mnt` format
described here has no real-dataset equivalent — `SyntheticDataset` parses it
with its own dedicated reader (`_load_mnt` in
`src/data/synthetic/synthetic_dataset.py`), not `FMDGroundTruth`. The format
below is otherwise accurate for what's actually on disk.

**Real source (as originally, incorrectly, documented):** FVC_BERKAY_TOPCU —
minutiae stored as `.mnt` files (one per image) alongside the image files.
**Real source (actual):** FVC_BERKAY_TOPCU — `.npz` per image, unrelated format.

**Format** — plain text:
```
<count>
<x> <y> <type> <angle>
<x> <y> <type> <angle>
...
```

- Line 1: integer count of minutiae
- Each subsequent line: `x y type angle`
  - `x`, `y`: pixel coordinates (integers), origin top-left
  - `type`: `1` = ridge ending, `2` = bifurcation
  - `angle`: minutiae direction in degrees [0, 360), integer
- VeriFinger SDK returns angle in degrees — write it directly
- No width/height field — the consumer must get the image's own pixel
  dimensions separately to build a per-pixel label map.

**Actual output path (corrected 2026-07-06):**
`/workspace/annotation_workspace/annotations/{source_id}/{stem}.mnt`

e.g. `/workspace/annotation_workspace/annotations/S04/0--1.mnt`

**CORRECTION (2026-08-05):** the claim below that S11 is excluded from FMD
was wrong. S11's `.mnt` files (30,001 of them, one per image, confirmed by
direct inspection) are in this exact same format, fully typed, and load
correctly via the same parser as every other source — the exclusion just
meant every `fmd_*_S11` experiment silently trained on real data alone
(R1-R3/V1-V3) or crashed outright (R4, 0% real + 0% synthetic =
`num_samples=0`). `SyntheticDataset.FMD_EXCLUDED_SOURCES` is now an empty
set. ~~S11 is excluded from FMD regardless of this format — see README.md's
Known Issues (S11's own native minutiae carry no ridge-ending/bifurcation
type info). `SyntheticDataset.FMD_EXCLUDED_SOURCES` enforces this.~~

---

## FOE — Fingerprint Orientation Estimation

**CORRECTION (2026-07-06):** the claim below that `FOERealDataset` can load
this file is wrong. The real `.gt` format is a completely different, more
complex binary layout (8-byte skip + border/step/width/height int32 header +
interleaved orientation/weight byte pairs per cell — see
`annotation_src/foe_extract.py`'s docstring and `FOEGroundTruth.from_file` in
`src/data/real/foe_dataset.py`). The header-less raw-uint8 format described
below is only ever written and read by the synthetic-side code
(`annotate_batch.py`'s `write_foe`, `SyntheticDataset._load_synthetic_foe_gt`)
— real and synthetic `.gt` files are incompatible and must never be read with
each other's parser.

**Real source:** FOESamples — `{stem}.gt` files in Bad/ and Good/ subfolders
(different format, see correction above).

**Format** — binary, numpy uint8 array saved with `numpy.save` (`.npy` extension)
BUT named `.gt`, written as raw uint8 bytes (no numpy header).

Shape: `(H/8, W/8)` where H, W are the *source image's own native pixel
dimensions* (confirmed against `annotate_batch.py`'s `write_foe`: `h, w =
image.shape; hb, wb = h // 8, w // 8`) — this varies per source and even per
image, since synthetic sources have no consistent native resolution. One
uint8 value per 8×8 block, range [0, 255] representing orientation in
[0°, 180°):

```python
orientation = np.fromfile(gt_path, dtype=np.uint8).reshape(h // 8, w // 8)
```

pyfing SNFOE returns orientations in radians [0, π). Convert to uint8 bins:

```python
# ori_rad: float array shape (H/8, W/8), values in [0, pi)
ori_uint8 = (ori_rad / np.pi * 256).astype(np.uint8)  # [0, 255]
ori_uint8.tofile(output_path)   # raw bytes, no numpy header
```

**Consumer note:** real FOE labels always end up at a fixed `(69, 55)` grid
(decoupled from `config.input_size` — see `FOE_LABEL_GRID_SHAPE` in
`synthetic_dataset.py`), and real+synthetic samples are batched together by
the default collate, so any synthetic `.gt` read at its own native grid size
must be circularly resampled (via cos(2θ)/sin(2θ), not naive linear
interpolation — orientation wraps at 0/π) to `(69, 55)` before use.

**Actual output path (corrected 2026-07-06):**
`/workspace/annotation_workspace/annotations/{source_id}/{stem}.gt`

e.g. `/workspace/annotation_workspace/annotations/S04/0--1.gt`

**S11 is excluded from this path** — it ships its own native per-image
orientation file (`_o.txt`) alongside its images, read directly rather than
via the annotation pipeline's `.gt` output.

---

## Summary table

| Task | Extension | Format | Written by |
|------|-----------|--------|------------|
| FC   | `.txt`    | 3-line plain text | VeriFinger ctypes |
| FMD  | `.mnt`    | count + x y type angle lines | VeriFinger ctypes |
| FOE  | `.gt`     | raw uint8 bytes, shape H/8×W/8 | pyfing SNFOE |

---

## Skip-if-exists rule

The annotation runner checks whether all three sidecar files exist before
processing an image. If all three are present, the image is skipped. This makes
the pipeline safely resumable after interruption.

```python
fc_done  = Path(f"{stem}.txt").exists()
fmd_done = Path(f"{stem}.mnt").exists()
foe_done = Path(f"{stem}.gt").exists()
if fc_done and fmd_done and foe_done:
    continue
```
