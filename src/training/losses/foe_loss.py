from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FOELoss(nn.Module):
    """
    Binary cross-entropy per-bit loss for 128-channel orientation estimation.

    GT label is interpolated to the model's output resolution before loss
    computation (loss at network-native resolution).
    """

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # logits: [B, 128, H_out, W_out]  — raw, no sigmoid
        # labels: [B, 128, H_gt,  W_gt]   — binary float32

        # Align GT to model output shape if needed
        if logits.shape[-2:] != labels.shape[-2:]:
            labels = F.interpolate(
                labels,
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            labels = labels.clamp(0.0, 1.0)

        return F.binary_cross_entropy_with_logits(logits, labels)
