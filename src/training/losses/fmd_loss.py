from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMDLoss(nn.Module):
    """
    Weighted cross-entropy loss for 3-class per-pixel minutiae detection.

    class_weights [3]: one weight per class (background, ridge-ending,
    bifurcation). If None, uses uniform weights.
    """

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # logits: [B, 3, H, W]  — raw, no softmax
        # labels: [B, 3, H, W]  — one-hot float32
        targets = labels.argmax(dim=1)   # [B, H, W] long
        return F.cross_entropy(logits, targets, weight=self.class_weights)
