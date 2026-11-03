from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.real.foe_dataset import FOERealDataset

_ACC_THRESHOLDS_DEG = [5.0, 10.0, 15.0, 20.0, 30.0]


def angular_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Per-pixel angular error for orientations in [0, pi).
    Returns errors in [0, pi/2].
    """
    diff = np.abs(pred - gt)
    return np.minimum(diff, np.pi - diff)


def score_fold(pred_path: Path, val_dataset: FOERealDataset) -> float:
    """
    Negative mean angular error in degrees over masked pixels.
    Negated so that higher is better for HP selection.
    """
    preds = np.load(pred_path)  # [N, H_o, W_o]
    total_error = 0.0
    total_pixels = 0

    for i in range(len(preds)):
        item = val_dataset[i]
        gt_ori = item["orientation"].numpy()  # [H_gt, W_gt]
        mask   = item["mask"].numpy()         # [H_gt, W_gt]

        pred_i = preds[i]  # [H_out, W_out]

        if pred_i.shape != gt_ori.shape:
            pred_tensor = torch.from_numpy(pred_i).unsqueeze(0).unsqueeze(0).float()
            pred_tensor = F.interpolate(
                pred_tensor,
                size=gt_ori.shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_i = pred_tensor.squeeze().numpy()

        err = angular_error(pred_i, gt_ori)
        total_error  += float((err * mask).sum())
        total_pixels += int(mask.sum())

    mean_error_rad = total_error / (total_pixels + 1e-8)
    mean_error_deg = mean_error_rad * 180.0 / np.pi
    return -mean_error_deg  # negated: higher is better


def evaluate_full(pred_path: Path, test_dataset: FOERealDataset) -> dict:
    """
    Comprehensive FOE metrics:
      1. Mean angular error per image, then averaged
      2. Global mean angular error (all masked pixels)
      3. Accuracy at thresholds [5°, 10°, 15°, 20°, 30°]
         = fraction of masked pixels with error < threshold
      4. All metrics stratified by quality (B / G)
    """
    preds = np.load(pred_path)  # [N, H_o, W_o]

    per_image_errors: list[float] = []
    total_error = 0.0
    total_pixels = 0

    # Threshold accumulators: {thresh_deg: (correct_pixels, total_pixels)}
    thresh_correct  = {t: 0 for t in _ACC_THRESHOLDS_DEG}
    thresh_total    = {t: 0 for t in _ACC_THRESHOLDS_DEG}
    thresh_rad = {t: t * np.pi / 180.0 for t in _ACC_THRESHOLDS_DEG}

    # Quality stratification
    quality_accum: dict[str, dict] = {}

    for i in range(len(preds)):
        item = test_dataset[i]
        gt_ori  = item["orientation"].numpy()  # [H_gt, W_gt]
        mask    = item["mask"].numpy()         # [H_gt, W_gt]
        quality = item.get("quality", "unknown")

        pred_i = preds[i]
        if pred_i.shape != gt_ori.shape:
            pred_tensor = torch.from_numpy(pred_i).unsqueeze(0).unsqueeze(0).float()
            pred_tensor = F.interpolate(
                pred_tensor,
                size=gt_ori.shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_i = pred_tensor.squeeze().numpy()

        err = angular_error(pred_i, gt_ori)  # [H, W]
        masked_err = err * mask
        n_pixels = int(mask.sum())

        # Per-image mean error
        if n_pixels > 0:
            img_mean_rad = float(masked_err.sum()) / n_pixels
        else:
            img_mean_rad = 0.0
        per_image_errors.append(img_mean_rad * 180.0 / np.pi)

        total_error  += float(masked_err.sum())
        total_pixels += n_pixels

        # Threshold accuracy
        for t in _ACC_THRESHOLDS_DEG:
            thresh_correct[t] += int((err < thresh_rad[t])[mask > 0].sum())
            thresh_total[t]   += n_pixels

        # Per-quality
        if quality not in quality_accum:
            quality_accum[quality] = {
                "total_error": 0.0,
                "total_pixels": 0,
                "thresh_correct": {t: 0 for t in _ACC_THRESHOLDS_DEG},
            }
        qa = quality_accum[quality]
        qa["total_error"]  += float(masked_err.sum())
        qa["total_pixels"] += n_pixels
        for t in _ACC_THRESHOLDS_DEG:
            qa["thresh_correct"][t] += int((err < thresh_rad[t])[mask > 0].sum())

    # Assemble results
    global_mean_deg = (total_error / (total_pixels + 1e-8)) * 180.0 / np.pi
    acc_at_thresholds = {
        f"{int(t)}_deg": (thresh_correct[t] / (thresh_total[t] + 1e-8))
        for t in _ACC_THRESHOLDS_DEG
    }

    per_quality_results = {}
    for q, qa in quality_accum.items():
        n = qa["total_pixels"]
        mean_deg = (qa["total_error"] / (n + 1e-8)) * 180.0 / np.pi
        per_quality_results[q] = {
            "mean_angular_error_deg": float(mean_deg),
            "n_pixels": n,
            "acc_at_thresholds": {
                f"{int(t)}_deg": (qa["thresh_correct"][t] / (n + 1e-8))
                for t in _ACC_THRESHOLDS_DEG
            },
        }

    return {
        "mean_angular_error_per_image_deg": float(np.mean(per_image_errors)),
        "mean_angular_error_global_deg": float(global_mean_deg),
        "acc_at_thresholds": acc_at_thresholds,
        "per_quality": per_quality_results,
        "n_samples": len(preds),
    }
