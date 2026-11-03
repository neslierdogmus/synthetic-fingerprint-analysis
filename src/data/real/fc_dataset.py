from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.base_dataset import FingerprintDataset

FC_CLASS_LABELS = ("T", "A", "L", "R", "W")
FC_CLASS_TO_IDX = {c: i for i, c in enumerate(FC_CLASS_LABELS)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Horizontal flip swaps left loop and right loop
_FLIP_MAP = {"L": "R", "R": "L"}


# ── Ground truth ───────────────────────────────────────────────────────────────

class FCGroundTruth:
    def __init__(self, fp_class: str, fp_class_2: str, gender: str):
        self.fp_class = fp_class
        self.fp_class_2 = fp_class_2
        self.gender = gender

    @classmethod
    def from_file(cls, gt_path) -> "FCGroundTruth":
        with open(gt_path, "r") as f:
            lines = f.readlines()
        gender   = lines[0].split(": ", 1)[1].strip()    # "Gender: M" → "M"
        fp_class = lines[1].split(": ", 1)[1].strip()    # "Class: W"  → "W"
        if len(lines) > 2:
            tokens = lines[2].split()                     # "History: f0001_01.pct W a0591.pct"
            fp_class_2 = tokens[2] if len(tokens) > 2 else fp_class
        else:
            fp_class_2 = fp_class
        return cls(fp_class, fp_class_2, gender)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _figs_subfolder(filename: str) -> str:
    """Map filename like 'f0001_01.png' to its figs_N subfolder.
    Subject number NNNN → figs_{(NNNN - 1) // 250}.
    """
    stem = Path(filename).stem   # e.g. "f0001_01"
    nnnn = int(stem[1:5])        # subject number
    return f"figs_{(nnnn - 1) // 250}"


def _make_labels(
    fp_class: str, fp_class_2: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (label_onehot, label_multilabel) as [5] float32 tensors."""
    label = torch.zeros(5, dtype=torch.float32)
    label[FC_CLASS_TO_IDX[fp_class]] = 1.0
    multi = label.clone()
    # fp_class_2 from the history field may contain unknown codes (e.g. "TL");
    # fall back to fp_class in that case.
    if fp_class_2 != fp_class and fp_class_2 in FC_CLASS_TO_IDX:
        multi[FC_CLASS_TO_IDX[fp_class_2]] = 1.0
    return label, multi


def _was_hflipped(replay: dict) -> bool:
    """Recursively inspect an Albumentations ReplayCompose replay dict."""
    for t in replay.get("transforms", []):
        if "HorizontalFlip" in t.get("__class_fullname__", "") and t.get("applied"):
            return True
        if "transforms" in t and _was_hflipped(t):
            return True
    return False


def _to_tensor(
    img: np.ndarray,
    model_type: str,
    normalisation_stats: dict | None,
) -> torch.Tensor:
    """Convert (H, W) uint8 numpy array to a normalised float32 tensor."""
    x = img.astype(np.float32) / 255.0  # [0, 1]

    if model_type == "pretrained":
        x3 = np.stack([x, x, x], axis=-1)  # (H, W, 3)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        x3 = (x3 - mean) / std
        return torch.from_numpy(x3.copy()).permute(2, 0, 1)  # [3, H, W]

    # scratch: 1-channel
    if normalisation_stats is not None:
        x = (x - normalisation_stats["mean"]) / normalisation_stats["std"]
    return torch.from_numpy(x.copy()).unsqueeze(0)  # [1, H, W]


# ── Dataset ────────────────────────────────────────────────────────────────────

class FCRealDataset(FingerprintDataset):
    """
    NIST SD4 fingerprint classification dataset.

    Filenames come from a split JSON and are basenames like "f0001_01.png".
    The dataset resolves which figs_N subfolder each file lives in using
    the formula: figs_{(subject_number - 1) // 250}.
    """

    def __init__(
        self,
        data_root: str,
        filenames: list[str],
        transform=None,
        model_type: str = "scratch",
        normalisation_stats: dict | None = None,
    ):
        self.data_root = Path(data_root)  # .../NIST_SD4
        self.filenames = list(filenames)
        self.transform = transform
        self.model_type = model_type
        self.normalisation_stats = normalisation_stats

    def _img_path(self, filename: str) -> Path:
        return self.data_root / "png_txt" / _figs_subfolder(filename) / filename

    def _gt_path(self, filename: str) -> Path:
        stem = Path(filename).stem
        return self.data_root / "png_txt" / _figs_subfolder(filename) / f"{stem}.txt"

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        filename = self.filenames[index]
        stem = Path(filename).stem

        # Load grayscale image
        img = np.array(Image.open(self._img_path(filename)).convert("L"))  # (H, W)

        # Load ground truth
        gt = FCGroundTruth.from_file(self._gt_path(filename))
        fp_class = gt.fp_class
        fp_class_2 = gt.fp_class_2

        # Apply augmentation
        flipped = False
        if self.transform is not None:
            result = self.transform(image=img)
            img = result["image"]
            replay = result.get("replay")
            if replay is not None:
                flipped = _was_hflipped(replay)

        # Update class under horizontal flip: L ↔ R
        if flipped:
            fp_class = _FLIP_MAP.get(fp_class, fp_class)
            fp_class_2 = _FLIP_MAP.get(fp_class_2, fp_class_2)

        label, label_multi = _make_labels(fp_class, fp_class_2)
        img_tensor = _to_tensor(img, self.model_type, self.normalisation_stats)

        return {
            "image": img_tensor,
            "label": label,
            "label_multilabel": label_multi,
            "filename": stem,
            "quality": gt.gender,
        }
