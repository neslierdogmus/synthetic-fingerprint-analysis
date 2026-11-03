from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def normalise_scratch(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Per-dataset normalisation for M1 (scratch) models."""
    image = image.astype(np.float32) / 255.0
    return (image - mean) / (std + 1e-8)


def normalise_imagenet(image: np.ndarray) -> np.ndarray:
    """ImageNet normalisation for M2/M3 (pretrained) models."""
    image = image.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    return (image - mean) / std


def to_tensor_grayscale(image: np.ndarray) -> torch.Tensor:
    """H × W → 1 × H × W float32 tensor."""
    return torch.from_numpy(image).unsqueeze(0).float()


def to_tensor_rgb(image: np.ndarray) -> torch.Tensor:
    """H × W × 3 → 3 × H × W float32 tensor."""
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """H × W → H × W × 3 by channel replication."""
    return np.stack([image, image, image], axis=-1)
