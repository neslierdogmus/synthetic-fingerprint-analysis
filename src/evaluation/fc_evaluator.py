from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.real.fc_dataset import FCRealDataset

FC_CLASSES = ("T", "A", "L", "R", "W")


def score_fold(pred_path: Path, val_dataset: FCRealDataset) -> float:
    """
    Top-1 accuracy, multi-label tolerant.
    A prediction is correct if the argmax index has value 1.0 in
    label_multilabel (matches either primary or secondary GT class).
    """
    preds = np.load(pred_path)          # [N, 5]
    pred_classes = preds.argmax(axis=1) # [N]
    correct = 0
    for i, pred_class in enumerate(pred_classes):
        item = val_dataset[i]
        if item["label_multilabel"][pred_class] == 1.0:
            correct += 1
    return correct / len(pred_classes)


def evaluate_full(pred_path: Path, test_dataset: FCRealDataset) -> dict:
    """
    Compute all FC metrics:
      1. Top-1 accuracy (multi-label tolerant)
      2. Per-class accuracy (primary label only)
      3. Confusion matrix 5×5 (primary label only)
    """
    preds = np.load(pred_path)           # [N, 5]
    pred_classes = preds.argmax(axis=1)  # [N]

    gt_primary: list[int] = []
    gt_multilabel_list: list[np.ndarray] = []
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        gt_primary.append(int(item["label"].argmax().item()))
        gt_multilabel_list.append(item["label_multilabel"].numpy())

    gt_primary_arr = np.array(gt_primary)
    gt_multilabel_arr = np.array(gt_multilabel_list)  # [N, 5]

    # 1. Top-1 accuracy (multilabel tolerant)
    multilabel_correct = gt_multilabel_arr[
        np.arange(len(pred_classes)), pred_classes
    ].sum()
    top1_acc = float(multilabel_correct / len(pred_classes))

    # 2. Per-class accuracy (primary label only)
    per_class_acc: dict[str, float | None] = {}
    for c, name in enumerate(FC_CLASSES):
        mask = gt_primary_arr == c
        if mask.sum() == 0:
            per_class_acc[name] = None
        else:
            per_class_acc[name] = float((pred_classes[mask] == c).mean())

    # 3. Confusion matrix (primary label only)
    confusion = np.zeros((5, 5), dtype=int)
    for gt, pred in zip(gt_primary_arr, pred_classes):
        confusion[gt, pred] += 1

    return {
        "top1_accuracy": top1_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion.tolist(),
        "n_samples": len(pred_classes),
    }
