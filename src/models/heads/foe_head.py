from __future__ import annotations

import torch
import torch.nn as nn


class FOEHead(nn.Module):
    """
    FOE output head: 128-channel per-pixel logits at 1/8 input resolution.

    Input:  [B, in_channels, H/8, W/8]  (decoder output)
    Output: [B, 128, H/8, W/8]          (sigmoid applied in loss)
    """

    def __init__(self, in_channels: int, num_bits: int = 128):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_bits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
