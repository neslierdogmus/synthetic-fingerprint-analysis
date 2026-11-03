from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    FC classification head: global average pooling + linear classifier.

    Operates on the deepest encoder feature map (features[-1]).
    Returns [B, num_classes] logits (no softmax — applied in loss).
    """

    def __init__(self, in_channels: int, num_classes: int = 5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]        # deepest feature map [B, C, H', W']
        x = self.pool(x)        # [B, C, 1, 1]
        x = x.flatten(1)        # [B, C]
        return self.fc(x)       # [B, num_classes]
