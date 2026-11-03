from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.base_dataset import FingerprintDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Ground truth ───────────────────────────────────────────────────────────────

def _read_int_bytes(fin, n: int) -> int:
    return int.from_bytes(fin.read(n), byteorder="little")


class FOEGroundTruth:
    def __init__(self, border, step, orientations, weights, mask):
        self.border = border
        self.step = step
        self.orientations = orientations  # H_o x W_o float, values in [0, π)
        self.weights = weights            # H_o x W_o uint8
        self.mask = np.asarray(mask, dtype=np.uint8)

    @classmethod
    def from_file(cls, gt_path, fg_path) -> "FOEGroundTruth":
        mask = np.loadtxt(fg_path, skiprows=1)
        with open(gt_path, "rb") as fin:
            fin.read(8)
            border = _read_int_bytes(fin, 4)
            border_y = _read_int_bytes(fin, 4)
            assert border == border_y
            step = _read_int_bytes(fin, 4)
            step_y = _read_int_bytes(fin, 4)
            assert step == step_y
            w = _read_int_bytes(fin, 4)
            h = _read_int_bytes(fin, 4)
            orientations = np.zeros((h, w), dtype=np.float64)
            weights = np.zeros((h, w), dtype=np.uint8)
            for r in range(h):
                for c in range(w):
                    orientations[r, c] = _read_int_bytes(fin, 1)
                    orientations[r, c] = orientations[r, c] / 256 * np.pi
                    weights[r, c] = _read_int_bytes(fin, 1)
        return cls(border, step, orientations, weights, mask)


# ── Orientation map size patches ───────────────────────────────────────────────

def _patch_orientation(gt: FOEGroundTruth, stem: str) -> FOEGroundTruth:
    """
    Fix two real FOE images whose orientation maps are 1 row/col short.
      stem "02": orientation map 1 column short → append zero column
      stem "07": orientation map 1 row short    → append zero row
    """
    if stem == "02":
        zero_col = np.zeros((gt.orientations.shape[0], 1), dtype=gt.orientations.dtype)
        gt.orientations = np.hstack([gt.orientations, zero_col])
        gt.weights = np.hstack([
            gt.weights,
            np.zeros((gt.weights.shape[0], 1), dtype=gt.weights.dtype),
        ])
        gt.mask = np.hstack([
            gt.mask,
            np.zeros((gt.mask.shape[0], 1), dtype=gt.mask.dtype),
        ])
    elif stem == "07":
        zero_row = np.zeros((1, gt.orientations.shape[1]), dtype=gt.orientations.dtype)
        gt.orientations = np.vstack([gt.orientations, zero_row])
        gt.weights = np.vstack([
            gt.weights,
            np.zeros((1, gt.weights.shape[1]), dtype=gt.weights.dtype),
        ])
        gt.mask = np.vstack([
            gt.mask,
            np.zeros((1, gt.mask.shape[1]), dtype=gt.mask.dtype),
        ])
    return gt


# ── Cyclic orientation encoding ────────────────────────────────────────────────

def encode_orientation_map(theta_rad: np.ndarray) -> np.ndarray:
    """
    Vectorised 128-bit cyclic encoding of a 2-D orientation map.

    B = 256 bins covering [0°, 180°).
    Bit i (0-indexed) = 1 iff the bin of θ is in
      {(i+1) % 256, (i+2) % 256, ..., (i+128) % 256}
    i.e.  ((bin_θ - (i+1)) % 256) < 128.

    Input : (H_o, W_o) float, values in [0, π)
    Output: (128, H_o, W_o) float32
    """
    B = 256
    theta_deg = (theta_rad * 180.0 / np.pi) % 180.0
    k = (theta_deg * B / 180.0).astype(np.int32) % B  # (H_o, W_o)

    i_vals = np.arange(128, dtype=np.int32)           # (128,)
    k_exp = k[np.newaxis, :, :]                        # (1, H_o, W_o)
    i_exp = (i_vals + 1)[:, np.newaxis, np.newaxis]   # (128, 1, 1)

    diff = (k_exp - i_exp) % B                         # (128, H_o, W_o)
    return (diff < 128).astype(np.float32)             # (128, H_o, W_o)


# ── Normalisation helpers ──────────────────────────────────────────────────────

def _masked_normalise(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    FOE masked normalisation (scratch models).
    Normalises only within the foreground mask.
    """
    mask_f = mask.astype(np.float32)
    n = mask_f.sum()
    if n == 0:
        return x
    x_mean = (x * mask_f).sum() / n
    x_centered = (x - x_mean) * mask_f
    x_var = (x_centered ** 2).sum() / n
    x_std = float(x_var ** 0.5)
    if x_std < 1e-8:
        return x_centered
    return x_centered / x_std


def _to_tensor(
    img: np.ndarray,
    model_type: str,
    normalisation_stats: dict | None,
    mask: np.ndarray | None = None,
) -> torch.Tensor:
    """Convert (H, W) uint8 numpy image to a normalised float32 tensor."""
    x = img.astype(np.float32) / 255.0  # [0, 1]

    if model_type == "pretrained":
        x3 = np.stack([x, x, x], axis=-1)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        x3 = (x3 - mean) / std
        return torch.from_numpy(x3.copy()).permute(2, 0, 1)  # [3, H, W]

    # scratch: masked normalisation
    if mask is not None:
        x = _masked_normalise(x, mask)
    elif normalisation_stats is not None:
        x = (x - normalisation_stats["mean"]) / normalisation_stats["std"]
    return torch.from_numpy(x.copy()).unsqueeze(0)  # [1, H, W]


# ── Filename helpers ───────────────────────────────────────────────────────────

def _parse_foe_filename(filename: str) -> tuple[str, str, str]:
    """
    Parse a FOE filename into (stem, quality_code, subfolder_name).

    Expected format: "Bad/00.bmp" or "Good/110.bmp".
      → stem="00", quality="B", subfolder="Bad"
      → stem="110", quality="G", subfolder="Good"
    """
    subfolder, name = filename.split("/", 1)
    stem = Path(name).stem
    quality = "B" if subfolder == "Bad" else "G"
    return stem, quality, subfolder


# ── Dataset ────────────────────────────────────────────────────────────────────

class FOERealDataset(FingerprintDataset):
    """
    FOE (Fingerprint Orientation Estimation) real dataset.

    Filenames are in "Bad/00.bmp" / "Good/110.bmp" format (subfolder/name).
    Quality is inferred from the subfolder: Bad → "B", Good → "G".
    Each stem has three sidecar files: .bmp (image), .gt (orientation map),
    .fg (foreground mask).

    The dataset returns a 128-channel cyclic-encoded orientation map as label.
    """

    def __init__(
        self,
        data_root: str,
        filenames: list[str],
        transform=None,
        model_type: str = "scratch",
        normalisation_stats: dict | None = None,
    ):
        self.data_root = Path(data_root)  # .../FOESamples
        self.filenames = list(filenames)
        self.transform = transform
        self.model_type = model_type
        self.normalisation_stats = normalisation_stats

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        filename = self.filenames[index]
        stem, quality, subfolder = _parse_foe_filename(filename)

        subdir = self.data_root / subfolder

        # Load image
        img = np.array(
            Image.open(subdir / f"{stem}.bmp").convert("L")
        )  # (H, W) uint8

        # Load ground truth; apply orientation map size patches
        gt = FOEGroundTruth.from_file(subdir / f"{stem}.gt", subdir / f"{stem}.fg")
        gt = _patch_orientation(gt, stem)

        orientations = gt.orientations.astype(np.float32)  # (H_o, W_o)
        fg_mask = gt.mask.astype(np.float32)               # (H_o, W_o) at orientation-map res

        # Pad to fixed intermediate size then crop to remove border,
        # producing a fixed image size and fixed orientation-map size for
        # every sample so DataLoader can collate batches.
        step = gt.step
        border = gt.border
        pad_h = 576 - img.shape[0]
        pad_w = 464 - img.shape[1]
        img = np.pad(img, ((0, pad_h), (0, pad_w)))
        fg_mask = np.pad(fg_mask, ((0, pad_h // step), (0, pad_w // step)))
        orientations = np.pad(orientations, ((0, pad_h // step), (0, pad_w // step)))
        omit = int(border - step / 2)
        img = img[omit:-omit, omit:-omit]

        # Apply augmentation (image only; orientation updated manually if flip)
        flipped = False
        if self.transform is not None:
            result = self.transform(image=img)
            img = result["image"]
            replay = result.get("replay")
            if replay is not None:
                flipped = _was_hflipped(replay)

        if flipped:
            orientations = np.fliplr(orientations)
            nonzero = orientations > 0
            orientations[nonzero] = np.pi - orientations[nonzero]
            orientations = np.clip(orientations, 0.0, np.pi - 1e-7)
            fg_mask = np.fliplr(fg_mask)

        # Resize fg mask to image resolution for masked normalisation
        img_h, img_w = img.shape
        mask_img_res = cv2.resize(fg_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Normalise image
        img_tensor = _to_tensor(
            img, self.model_type, self.normalisation_stats,
            mask=mask_img_res if self.model_type == "scratch" else None,
        )

        # Compute cyclic encoding: (128, H_o, W_o)
        label = encode_orientation_map(orientations)

        return {
            "image": img_tensor,
            "label": torch.from_numpy(label),
            "orientation": torch.from_numpy(orientations),
            "mask": torch.from_numpy(fg_mask),
            "filename": stem,
            "quality": quality,
        }


def _was_hflipped(replay: dict) -> bool:
    """Recursively inspect an Albumentations ReplayCompose replay dict."""
    for t in replay.get("transforms", []):
        if "HorizontalFlip" in t.get("__class_fullname__", "") and t.get("applied"):
            return True
        if "transforms" in t and _was_hflipped(t):
            return True
    return False
