from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ResNet50Encoder(nn.Module):
    """
    ResNet-50 encoder using timm features_only mode.

    Returns 4 skip-connection feature maps from shallowest to deepest:
      Stage 0 (layer1): [B, 256,  H/4,  W/4 ]
      Stage 1 (layer2): [B, 512,  H/8,  W/8 ]
      Stage 2 (layer3): [B, 1024, H/16, W/16]
      Stage 3 (layer4): [B, 2048, H/32, W/32]

    For 1-channel (scratch) input with pretrained=True, timm averages the
    first-layer weights across colour channels automatically via in_chans=1.
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 1):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            in_chans=in_channels,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)

    @property
    def skip_channels(self) -> list[int]:
        return [256, 512, 1024, 2048]

    @property
    def output_stride(self) -> int:
        return 32
