from __future__ import annotations

import torch
import torch.nn as nn


class FMDHead(nn.Module):
    """
    FMD output head: upsample decoder output to target resolution
    and produce 3-channel per-pixel logits.

    Input:  [B, in_channels, H_dec, W_dec]  (decoder output)
    Output: [B, 3, H_target, W_target]       (softmax applied in loss)
    
    If target_size is specified, upsamples to that exact size.
    Otherwise, upsamples 2× (legacy behavior).
    """

    def __init__(self, in_channels: int, num_classes: int = 3,
                 target_size: tuple[int, int] | None = None):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(in_channels // 2, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample first to avoid large intermediate tensors
        if self.target_size is not None:
            x = nn.functional.interpolate(
                x, size=self.target_size, mode="bilinear", align_corners=False
            )
        else:
            # Legacy: 2× upsample
            x = nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
        x = self.conv(x)
        return self.classifier(x)
