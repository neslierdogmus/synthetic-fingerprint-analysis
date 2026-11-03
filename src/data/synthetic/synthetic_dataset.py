from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.base_dataset import FingerprintDataset
from src.data.real.fc_dataset import (
    FCGroundTruth,
    FC_CLASS_TO_IDX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _FLIP_MAP,
    _make_labels,
)
from src.data.real.foe_dataset import encode_orientation_map

# Root of the VeriFinger/pyfing-generated annotation files for all 11
# synthetic sources (see annotation_workspace/LABEL_FORMAT.md). Relative to
# cwd, matching every other path in this codebase (configs/, splits/,
# outputs/ are all cwd-relative) -- every entry point assumes cwd=/workspace.
ANNOTATIONS_ROOT = Path("annotation_workspace/annotations")

# S11 is NOT excluded from FMD. Earlier versions of this file excluded it
# based on a README.md "Known Issues" claim that S11's native minutiae lack
# ridge-ending/bifurcation type info -- confirmed incorrect: S11's labels
# come from a different source/methodology than the other 10 sources'
# auto-annotated .mnt files, but the resulting .mnt format is identical and
# fully typed (verified directly against annotation_workspace/annotations/S11:
# 30,001 .mnt files, one per image, same "count + x y type angle" structure
# as every other source, loadable by the same _load_mnt() below with no
# special-casing). The exclusion caused every fmd_*_S11 experiment to build
# an empty synthetic dataset -- crashing outright for R4 (0% real + 0%
# synthetic = num_samples=0) and silently degrading R1-R3/V1-V3 (real data
# alone, missing the intended synthetic augmentation). Kept as an empty set
# (not deleted) so a future source-specific exclusion has an obvious place
# to go, and so split_generator.py's shared import doesn't need touching.
FMD_EXCLUDED_SOURCES: set[str] = set()

# Real FOE labels are always exactly this (H_o, W_o) shape regardless of
# config.input_size -- confirmed empirically against FOERealDataset, since
# it pads/crops in grid units (border/step from the real .gt header) fully
# decoupled from the image's own Albumentations resize. Real and synthetic
# samples get batched together by the default collate, so any synthetic FOE
# label must be resampled to this exact shape to be stackable at all.
# FOELoss itself separately interpolates GT to the model's actual output
# resolution at loss time, so this shape only needs to be internally
# consistent for batching, not architecturally exact.
FOE_LABEL_GRID_SHAPE = (69, 55)


# ── S11 class parsing ──────────────────────────────────────────────────────────

_S11_CLASS_MAP: dict[str, str] = {
    "tented_arch": "T",
    "left_loop":   "L",
    "right_loop":  "R",
    "whorl":       "W",
    "arch":        "A",
}
# Match longest prefix first to avoid "arch" swallowing "tented_arch"
_S11_CLASSES_SORTED = sorted(_S11_CLASS_MAP.keys(), key=len, reverse=True)


def _parse_s11_class(fp_id: str) -> str | None:
    """Return FC class letter for an S11 FP_id, or None if unrecognised."""
    for cls in _S11_CLASSES_SORTED:
        if fp_id.startswith(cls + "_") or fp_id == cls:
            return _S11_CLASS_MAP[cls]
    return None


# ── Shared image helpers ───────────────────────────────────────────────────────

def _to_tensor(
    img: np.ndarray,
    model_type: str,
    normalisation_stats: dict | None,
) -> torch.Tensor:
    """Convert (H, W) uint8 numpy array to a normalised float32 tensor."""
    x = img.astype(np.float32) / 255.0

    if model_type == "pretrained":
        x3 = np.stack([x, x, x], axis=-1)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        x3 = (x3 - mean) / std
        return torch.from_numpy(x3.copy()).permute(2, 0, 1)  # [3, H, W]

    if normalisation_stats is not None:
        x = (x - normalisation_stats["mean"]) / normalisation_stats["std"]
    return torch.from_numpy(x.copy()).unsqueeze(0)  # [1, H, W]


def _was_hflipped(replay: dict) -> bool:
    """Recursively inspect an Albumentations ReplayCompose replay dict."""
    for t in replay.get("transforms", []):
        if "HorizontalFlip" in t.get("__class_fullname__", "") and t.get("applied"):
            return True
        if "transforms" in t and _was_hflipped(t):
            return True
    return False


# ── FC label loading ────────────────────────────────────────────────────────────

def _fc_label_tensor(class_str: str) -> torch.Tensor:
    """One-hot [5] float32 tensor for a single FC class letter."""
    label = torch.zeros(5, dtype=torch.float32)
    label[FC_CLASS_TO_IDX[class_str]] = 1.0
    return label


def _get_fc_ground_truth(
    stem: str, fp_id: str, source: str
) -> tuple[str, str, str] | None:
    """
    Return (fp_class, fp_class_2, gender) or None if unavailable.

    S11 encodes its class directly in the filename (native, no annotation
    file needed). Every other source's class comes from the annotation
    pipeline's VeriFinger-generated {stem}.txt (see LABEL_FORMAT.md) --
    same 3-line format as real NIST_SD4 ground truth, so FCGroundTruth.from_file
    is reused as-is.
    """
    if source == "S11":
        cls = _parse_s11_class(fp_id)
        return (cls, cls, "U") if cls is not None else None

    txt_path = ANNOTATIONS_ROOT / source / f"{stem}.txt"
    if not txt_path.exists():
        return None
    gt = FCGroundTruth.from_file(txt_path)
    return gt.fp_class, gt.fp_class_2, gt.gender


# ── FMD label loading ───────────────────────────────────────────────────────────

def _load_mnt(path: Path) -> list[tuple[int, int, int, float]]:
    """
    Parse an annotation-pipeline .mnt file (LABEL_FORMAT.md):
    first line is an integer count, each following line is "x y type angle".
    NOTE: this is NOT the same format as the real FVC_BERKAY_TOPCU .npz
    ground truth (FMDGroundTruth) -- those are unrelated, incompatible
    formats despite LABEL_FORMAT.md's claim that FMDRealDataset could load
    synthetic samples "without modification". It can't; this parser exists
    because of that gap.
    """
    lines = path.read_text().splitlines()
    count = int(lines[0])
    minutiae = []
    for line in lines[1:1 + count]:
        x_str, y_str, t_str, angle_str = line.split()
        minutiae.append((int(x_str), int(y_str), int(t_str), float(angle_str)))
    return minutiae


# Mirrors annotation_workspace/annotation_src/annotate_batch.py's
# DPI_OVERRIDES: VeriFinger was called with a non-standard capture DPI for
# these three sources, but normalises its own minutiae output internally to
# 500 dpi regardless -- the image file itself was never resized, so their
# .mnt (x, y) coordinates come out inflated by 500/dpi relative to the
# actual image pixels. Confirmed empirically against S11's own native
# ground truth (annotation_workspace/annotation_src/eval/s11_eval.py-style
# check): nearest-neighbour distance from these .mnt coordinates to the true
# minutiae positions is minimized exactly at scale 390/500 (~5px error),
# not at scale 1.0 (~38px error, with ~25% of points falling outside the
# image entirely and silently clipped by _build_fmd_label_map below). This
# produced a measurable, source-specific bug in every completed fmd_*_S01,
# fmd_*_S02, and fmd_*_S11 experiment: displaced training labels teach the
# model a smeared minutiae response (high recall, low precision vs. the
# other 8 unaffected sources).
MNT_DPI_OVERRIDES = {"S01": 385.0, "S02": 375.0, "S11": 390.0}


def _rescale_mnt(
    minutiae: list[tuple[int, int, int, float]], source: str
) -> list[tuple[int, int, int, float]]:
    dpi = MNT_DPI_OVERRIDES.get(source)
    if dpi is None:
        return minutiae
    scale = dpi / 500.0
    return [
        (round(x * scale), round(y * scale), t, angle)
        for x, y, t, angle in minutiae
    ]


def _build_fmd_label_map(
    h: int, w: int, minutiae: list[tuple[int, int, int, float]]
) -> np.ndarray:
    """
    Build a (H, W, 3) background/ridge-ending/bifurcation label map from
    parsed .mnt minutiae. Mirrors fmd_dataset._build_label_map, but that
    one reads dimensions off an FMDGroundTruth (.npz) object which has no
    synthetic equivalent -- the .mnt format carries no width/height, so the
    caller supplies the loaded image's own shape instead.
    """
    label = np.zeros((h, w, 3), dtype=np.float32)
    label[:, :, 0] = 1.0  # all background initially
    for x, y, t, _angle in minutiae:
        if 0 <= y < h and 0 <= x < w and t in (1, 2):
            label[y, x, 0] = 0.0
            label[y, x, t] = 1.0
    return label


def _minutiae_from_label(label: np.ndarray) -> dict:
    """Extract raw minutiae coordinates from a (H, W, 3) label map (same
    convention as fmd_dataset._minutiae_from_label, kept independent here
    since importing it would pull in the whole real-dataset module for one
    function that's trivial to duplicate)."""
    y1, x1 = np.where(label[:, :, 1] > 0.5)
    y2, x2 = np.where(label[:, :, 2] > 0.5)
    return {
        "x": np.concatenate([x1, x2]).astype(np.int32),
        "y": np.concatenate([y1, y2]).astype(np.int32),
        "type": np.concatenate([
            np.ones(len(x1), dtype=np.int32),
            np.full(len(x2), 2, dtype=np.int32),
        ]),
    }


# ── FOE label loading ───────────────────────────────────────────────────────────

def _load_synthetic_foe_gt(gt_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """
    Read a synthetic .gt file: header-less raw uint8, shape (img_h//8,
    img_w//8) -- confirmed against the actual writer (annotate_batch.py's
    write_foe: `h, w = image.shape; hb, wb = h // 8, w // 8`), NOT assumed
    from LABEL_FORMAT.md alone. This is a different, simpler binary layout
    than the real FOESamples .gt format (8-byte skip + border/step/w/h
    header + interleaved orientation/weight bytes) -- FOEGroundTruth.from_file
    (real) cannot read this file and vice versa.

    Returns orientation in radians, shape (img_h//8, img_w//8).
    """
    hb, wb = img_h // 8, img_w // 8
    raw = np.fromfile(str(gt_path), dtype=np.uint8)
    raw = raw.reshape(hb, wb)
    return raw.astype(np.float64) / 256.0 * np.pi


def _resize_orientation_circular(
    theta_rad: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """
    Resize an orientation map (radians, period pi) to target_shape via the
    cos(2*theta)/sin(2*theta) representation, then reconstruct via atan2.
    Orientation is circular (0 and pi-epsilon are adjacent) -- naive linear
    interpolation of raw angle values would corrupt everything near that
    wraparound, the same class of bug already hit once in this project
    (see CONTEXT.md's FOE ground-truth byte-layout finding).
    """
    cos2 = np.cos(2 * theta_rad).astype(np.float32)
    sin2 = np.sin(2 * theta_rad).astype(np.float32)
    th, tw = target_shape
    cos2_r = cv2.resize(cos2, (tw, th), interpolation=cv2.INTER_LINEAR)
    sin2_r = cv2.resize(sin2, (tw, th), interpolation=cv2.INTER_LINEAR)
    return (np.arctan2(sin2_r, cos2_r) / 2.0) % np.pi


def _get_foe_label_and_orientation(
    stem: str, source: str, source_dir: Path, img_h: int, img_w: int
) -> tuple[torch.Tensor, np.ndarray] | None:
    """
    Return (cyclic_encoded_label [128,69,55], orientation_rad [69,55]) or
    None if unavailable.

    S11 has its own native per-image orientation file (_o.txt): a
    border-inset grid (border=14px, step=8px -- FOESamples' convention,
    same as the real-data pipeline in src/data/real/foe_dataset.py, NOT the
    generic no-border synthetic .gt writer). It covers only the interior
    region starting at pixel (14,14), not the full image, so it must be
    placed back into the full (img_h//8, img_w//8) grid at its true offset
    before resampling -- otherwise the whole grid is stretched as if it
    spanned the full image, misregistering every orientation by ~14px.
    This placement (pad by border//step=1 cell top/left, remainder
    bottom/right) is the exact reprojection formula empirically validated
    in annotation_workspace/annotation_src/eval/s11_eval.py against S11's
    own shipped ground truth (~2 degree mean error using it, vs. a garbled
    ~45-90 degree error without it). Every other source (S01-S10) reads the
    annotation pipeline's .gt file at its own native (img_h//8, img_w//8)
    grid, then resamples to FOE_LABEL_GRID_SHAPE so it can batch with real
    samples; S11 now joins that same path once placed in the same
    coordinate frame.
    """
    S11_BORDER, S11_STEP = 14, 8

    if source == "S11":
        o_path = source_dir / f"{stem}_o.txt"
        if not o_path.exists():
            return None
        orientations_deg = np.loadtxt(str(o_path)).astype(np.float32)
        orientations_rad = (orientations_deg * (np.pi / 180.0)).astype(np.float64)
        hb_native, wb_native = orientations_rad.shape
        hb_full, wb_full = img_h // S11_STEP, img_w // S11_STEP
        pad_top = S11_BORDER // S11_STEP
        pad_left = S11_BORDER // S11_STEP
        pad_bottom = hb_full - hb_native - pad_top
        pad_right = wb_full - wb_native - pad_left
        full_grid = np.pad(
            orientations_rad,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
        )
        theta = _resize_orientation_circular(full_grid, FOE_LABEL_GRID_SHAPE)
        encoded = encode_orientation_map(theta)
        return torch.from_numpy(encoded), theta.astype(np.float32)

    gt_path = ANNOTATIONS_ROOT / source / f"{stem}.gt"
    if not gt_path.exists():
        return None
    theta_native = _load_synthetic_foe_gt(gt_path, img_h, img_w)
    theta = _resize_orientation_circular(theta_native, FOE_LABEL_GRID_SHAPE)
    encoded = encode_orientation_map(theta)
    return torch.from_numpy(encoded), theta.astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────

class SyntheticDataset(FingerprintDataset):
    """
    Generic dataset for all synthetic sources (S01-S11).

    Returns the same dict structure as the corresponding real dataset (image,
    label, filename, quality, plus task-specific keys), with label = None
    where ground truth is genuinely unavailable (annotation file missing).
    Labels come from the VeriFinger/pyfing-generated annotation files in
    ANNOTATIONS_ROOT (see annotation_workspace/LABEL_FORMAT.md), except S11
    which ships its own native labels directly alongside its images.
    """

    def __init__(
        self,
        data_root: str,
        source: str,
        filenames: list[str],
        task: str,
        transform=None,
        model_type: str = "scratch",
        normalisation_stats: dict | None = None,
    ):
        self.source_dir = Path(data_root) / source
        self.source = source
        self.filenames = list(filenames)
        self.task = task
        self.transform = transform
        self.model_type = model_type
        self.normalisation_stats = normalisation_stats

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        filename = self.filenames[index]
        stem = Path(filename).stem                    # e.g. "10000--Bad_1"
        fp_id = stem.split("--", 1)[0]               # e.g. "10000"

        # Load image (always grayscale — convert in _to_tensor for pretrained)
        img = np.array(
            Image.open(self.source_dir / filename).convert("L")
        )  # (H, W) uint8
        img_h, img_w = img.shape

        # Build task-specific label plus whatever extra keys the
        # corresponding real dataset returns, so combined real+synthetic
        # batches collate cleanly (every item in a batch must have the same
        # dict keys and matching tensor shapes per key).
        label = None
        label_multi = None
        quality = None
        minutiae = {"x": np.zeros(0, dtype=np.int32),
                    "y": np.zeros(0, dtype=np.int32),
                    "type": np.zeros(0, dtype=np.int32)}
        orientation = np.zeros(FOE_LABEL_GRID_SHAPE, dtype=np.float32)
        mask = np.ones(FOE_LABEL_GRID_SHAPE, dtype=np.float32)
        flipped = False

        if self.task == "fmd":
            # FMD's label is a per-pixel map built at the *native* image
            # size -- it must go through the exact same geometric transform
            # (crop/rotate/flip/resize) as the image, jointly, or the two
            # end up at different shapes (this was a real bug caught by the
            # dev smoke test: label stayed at native e.g. 512x512 while the
            # image was correctly resized to config.input_size).  Mirrors
            # FMDRealDataset's mask0/1/2 additional-targets pattern exactly.
            label_map = None
            if self.source not in FMD_EXCLUDED_SOURCES:
                mnt_path = ANNOTATIONS_ROOT / self.source / f"{stem}.mnt"
                if mnt_path.exists():
                    parsed = _load_mnt(mnt_path)
                    parsed = _rescale_mnt(parsed, self.source)
                    label_map = _build_fmd_label_map(img_h, img_w, parsed)

            if self.transform is not None:
                if label_map is not None:
                    result = self.transform(
                        image=img,
                        mask0=label_map[:, :, 0],
                        mask1=label_map[:, :, 1],
                        mask2=label_map[:, :, 2],
                    )
                    img = result["image"]
                    ch1, ch2 = result["mask1"], result["mask2"]
                    # Recompute background so padded/border-filled pixels
                    # are background, not unlabelled (same as FMDRealDataset).
                    ch0 = 1.0 - ch1 - ch2
                    label_map = np.stack([ch0, ch1, ch2], axis=2)
                else:
                    img = self.transform(image=img)["image"]

            if label_map is not None:
                minutiae = _minutiae_from_label(label_map)
                label = torch.from_numpy(label_map.copy()).permute(2, 0, 1)
                quality = float("nan")  # no per-image quality score available

        else:
            # FC/FOE: image-only geometric transform. FC's label is a
            # single class (flip only swaps L<->R, handled via replay
            # inspection). FOE's label grid is independent of the image's
            # own crop/resize (see FOE_LABEL_GRID_SHAPE note above).
            if self.transform is not None:
                result = self.transform(image=img)
                img = result["image"]
                replay = result.get("replay")
                if replay is not None:
                    flipped = _was_hflipped(replay)

            if self.task == "fc":
                gt = _get_fc_ground_truth(stem, fp_id, self.source)
                if gt is not None:
                    fp_class, fp_class_2, gender = gt
                    if flipped:
                        fp_class = _FLIP_MAP.get(fp_class, fp_class)
                        fp_class_2 = _FLIP_MAP.get(fp_class_2, fp_class_2)
                    label, label_multi = _make_labels(fp_class, fp_class_2)
                    quality = gender

            elif self.task == "foe":
                result_foe = _get_foe_label_and_orientation(
                    stem, self.source, self.source_dir, img_h, img_w
                )
                if result_foe is not None:
                    label, orientation_arr = result_foe
                    orientation = orientation_arr.astype(np.float32)
                    quality = "U"  # no per-image quality assessment available

        img_tensor = _to_tensor(img, self.model_type, self.normalisation_stats)

        item = {
            "image":    img_tensor,
            "label":    label,
            "filename": stem,
            "quality":  quality,
        }
        if self.task == "fc":
            item["label_multilabel"] = label_multi
        elif self.task == "fmd":
            item["minutiae"] = minutiae
        elif self.task == "foe":
            item["orientation"] = torch.from_numpy(orientation)
            item["mask"] = torch.from_numpy(mask)

        return item
