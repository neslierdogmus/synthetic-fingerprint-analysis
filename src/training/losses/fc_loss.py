from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLoss(nn.Module):
    """Cross-entropy loss for 5-class fingerprint classification."""

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # logits: [B, 5]  — raw, no softmax
        # labels: [B, 5]  — one-hot float32 (primary class only)
        targets = labels.argmax(dim=1)   # [B] long
        return F.cross_entropy(logits, targets)
