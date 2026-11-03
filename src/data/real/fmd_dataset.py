from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from src.data.base_dataset import FingerprintDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Ground truth ───────────────────────────────────────────────────────────────

class FMDGroundTruth:
    def __init__(self, width, height, quality, lst_min_type, lst_min_posX,
                 lst_min_posY, lst_min_angle, lst_min_qual):
        self.width = width
        self.height = height
        self.quality = quality
        self.lst_min_type = lst_min_type
        self.lst_min_posX = lst_min_posX
        self.lst_min_posY = lst_min_posY
        self.lst_min_angle = lst_min_angle
        self.lst_min_qual = lst_min_qual

    @classmethod
    def from_file(cls, gt_path) -> "FMDGroundTruth":
        npz_file = np.load(gt_path)
        return cls(
            width=npz_file["size_x"],
            height=npz_file["size_y"],
            quality=npz_file["finger_qual"],
            lst_min_type=npz_file["T"],
            lst_min_posX=npz_file["X"],
            lst_min_posY=npz_file["Y"],
            lst_min_angle=npz_file["A"],
            lst_min_qual=npz_file["Q"],
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_label_map(gt: FMDGroundTruth) -> np.ndarray:
    """
    Build a 3-channel per-pixel label map (H x W x 3) float32.
      Channel 0: background  (1 where no minutia)
      Channel 1: ridge ending (type 1)
      Channel 2: bifurcation  (type 2)
    """
    h = int(gt.height)
    w = int(gt.width)
    label = np.zeros((h, w, 3), dtype=np.float32)
    label[:, :, 0] = 1.0  # all background initially

    for x, y, t in zip(gt.lst_min_posX, gt.lst_min_posY, gt.lst_min_type):
        xi, yi, ti = int(x), int(y), int(t)
        if 0 <= yi < h and 0 <= xi < w and ti in (1, 2):
            label[yi, xi, 0] = 0.0
            label[yi, xi, ti] = 1.0

    return label


def _minutiae_from_label(label: np.ndarray) -> dict:
    """Extract raw minutiae coordinates from a (H, W, 3) label map."""
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


def _to_tensor(
    img: np.ndarray,
    model_type: str,
    normalisation_stats: dict | None,
) -> torch.Tensor:
    """Convert (H, W) uint8 numpy array to a normalised float32 tensor."""
    x = img.astype(np.float32) / 255.0

    if model_type == "pretrained":
        x3 = np.stack([x, x, x], axis=-1)  # (H, W, 3)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        x3 = (x3 - mean) / std
        return torch.from_numpy(x3.copy()).permute(2, 0, 1)  # [3, H, W]

    if normalisation_stats is not None:
        x = (x - normalisation_stats["mean"]) / normalisation_stats["std"]
    return torch.from_numpy(x.copy()).unsqueeze(0)  # [1, H, W]


# ── Collate helper ────────────────────────────────────────────────────────────

def _fmd_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that keeps 'minutiae' as a list of dicts (variable length)."""
    result = {}
    for key in batch[0]:
        if key == "minutiae":
            result[key] = [item[key] for item in batch]
        else:
            result[key] = default_collate([item[key] for item in batch])
    return result


# ── Dataset ────────────────────────────────────────────────────────────────────

class FMDRealDataset(FingerprintDataset):
    collate_fn = staticmethod(_fmd_collate_fn)
    """
    FVC fingerprint minutiae detection dataset.

    Filenames from the split JSON are in the format "subfolder/basename.tif"
    e.g. "FM3_FVC2002DB1A_np/1_1.tif". The dataset resolves paths relative
    to data_root (/data/real/FVC_BERKAY_TOPCU).

    The paired Albumentations transform (if provided) must declare
    'mask' as an additional target (or use the built-in mask support)
    and receive the 3-channel label map as `mask` with shape (H, W, 3).
    """

    def __init__(
        self,
        data_root: str,
        filenames: list[str],
        transform=None,
        model_type: str = "scratch",
        normalisation_stats: dict | None = None,
    ):
        self.data_root = Path(data_root)  # .../FVC_BERKAY_TOPCU
        # Exclude corrupted file with invalid GT dimensions
        self.filenames = [f for f in filenames if f != "FM3_FVC2004DB3A_np/43_4.tif"]
        self.transform = transform
        self.model_type = model_type
        self.normalisation_stats = normalisation_stats

    def _img_path(self, rel: str) -> Path:
        return self.data_root / rel

    def _gt_path(self, rel: str) -> Path:
        return self.data_root / (Path(rel).with_suffix(".npz"))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        rel = self.filenames[index]          # e.g. "FM3_FVC2002DB1A_np/1_1.tif"
        stem = str(Path(rel).with_suffix(""))  # "FM3_FVC2002DB1A_np/1_1"

        # Load image; fall back to raw read for FVC2004 DB3A files that have
        # a non-standard 512-byte header followed by raw uint8 pixel data.
        img = cv2.imread(str(self._img_path(rel)), cv2.IMREAD_GRAYSCALE)  # (H, W) uint8
        if img is None:
            raw = open(str(self._img_path(rel)), "rb").read()
            n_pixels = len(raw) - 512
            # FVC2004 DB3A: 300 (W) × 480 (H) = 144000 px
            w = 300
            h = n_pixels // w
            img = np.frombuffer(raw[512:], dtype=np.uint8).reshape(h, w).copy()

        # Load ground truth and build label map
        gt = FMDGroundTruth.from_file(self._gt_path(rel))
        label_map = _build_label_map(gt)  # (H, W, 3) float32

        # Apply paired augmentation — split (H,W,3) label map into per-channel
        # masks so that Albumentations applies identical spatial transforms to
        # each channel.  Photometric transforms only affect the image key.
        if self.transform is not None:
            result = self.transform(
                image=img,
                mask0=label_map[:, :, 0],
                mask1=label_map[:, :, 1],
                mask2=label_map[:, :, 2],
            )
            img = result["image"]
            ch1 = result["mask1"]
            ch2 = result["mask2"]
            # Recompute background from the other two channels so that padded
            # and border-filled pixels (mask_value=0) are correctly classified
            # as background rather than unlabelled voids.
            ch0 = 1.0 - ch1 - ch2
            label_map = np.stack([ch0, ch1, ch2], axis=2)

        # Derive minutiae from (possibly augmented) label map
        minutiae = _minutiae_from_label(label_map)

        # Normalise image
        img_tensor = _to_tensor(img, self.model_type, self.normalisation_stats)

        # Label map: (H, W, 3) → [3, H, W]
        label_tensor = torch.from_numpy(label_map.copy()).permute(2, 0, 1)

        return {
            "image": img_tensor,
            "label": label_tensor,
            "filename": stem,
            "quality": float(gt.quality),
            "minutiae": minutiae,
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights from this dataset's label maps.
        weight_c = 1 / (freq_c * n_classes), normalised so weights sum to 3.
        """
        counts = np.zeros(3, dtype=np.float64)
        for rel in self.filenames:
            gt = FMDGroundTruth.from_file(self._gt_path(rel))
            label_map = _build_label_map(gt)
            counts += label_map.sum(axis=(0, 1))

        total = counts.sum()
        freq = counts / total
        weights = 1.0 / (freq * 3.0)
        weights = weights / weights.sum() * 3.0  # normalise so sum == 3
        return torch.from_numpy(weights.astype(np.float32))
