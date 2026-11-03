from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class FingerprintDataset(Dataset, ABC):
    """
    Abstract base class for all fingerprint datasets.

    __getitem__ must return a dict with at minimum:
        {
            "image":    torch.Tensor  # C x H x W, float32, normalised
            "label":    <task-specific>
            "filename": str           # original filename stem
        }
    """

    @abstractmethod
    def __getitem__(self, index) -> dict:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


def compute_normalisation_stats(dataset: FingerprintDataset) -> dict:
    """
    Compute per-dataset mean and std from a dataset (training split).
    Used for M1 (scratch) models only.
    Returns {"mean": float, "std": float}.
    """
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    for i in range(len(dataset)):
        item = dataset[i]
        img: torch.Tensor = item["image"]  # [C x H x W] float32, already in [0,1]
        pixel_sum += img.sum().item()
        pixel_sq_sum += (img ** 2).sum().item()
        pixel_count += img.numel()

    mean = pixel_sum / pixel_count
    std = ((pixel_sq_sum / pixel_count) - mean ** 2) ** 0.5
    return {"mean": mean, "std": std}
