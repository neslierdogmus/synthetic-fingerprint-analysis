from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class UNetDecoder(nn.Module):
    """
    UNet-style decoder with skip connections from encoder stages.

    Each decode stage:
      bilinear upsample (×2) → concat skip → Conv-BN-ReLU-Conv-BN-ReLU

    Skip connections are consumed from deepest encoder stage to shallowest.
    The final decode stage (if n_stages == 4) has no skip connection because
    all encoder skip channels have been exhausted.

    Output channel schedule (base_channels=256):
      Stage 0: 512  (base_channels × 2)
      Stage 1: 256  (base_channels)
      Stage 2: 128  (base_channels // 2)
      Stage 3:  64  (base_channels // 4)

    Output spatial resolution (relative to input H × W):
      n_stages=2: H/8  × W/8   → passed to FOE head
      n_stages=4: H/2  × W/2   → passed to FMD head (head upsamples to H×W)
    """

    def __init__(
        self,
        encoder_skip_channels: list[int],
        n_stages: int,
        base_channels: int = 256,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.use_checkpoint = use_checkpoint

        # Encoder skip channels from shallowest to deepest, e.g. [256,512,1024,2048]
        # Bottleneck = deepest; available skips (excluding bottleneck) reversed
        bottleneck_c = encoder_skip_channels[-1]
        # [stage2, stage1, stage0] for 4-stage encoder
        available_skips: list[int] = list(reversed(encoder_skip_channels[:-1]))

        self.blocks = nn.ModuleList()
        in_c = bottleneck_c

        for i in range(n_stages):
            # Output channels halve each stage: 512, 256, 128, 64
            out_c: int = (base_channels * 2) >> i   # >> i = // 2^i

            if i < len(available_skips):
                conv_in = in_c + available_skips[i]
            else:
                conv_in = in_c

            self.blocks.append(_conv_block(conv_in, out_c))
            in_c = out_c

        self._out_channels = in_c   # output channels of the last stage

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: encoder output [stage0, stage1, stage2, stage3],
                      shallowest to deepest.
        Returns:
            Decoded feature map.
        """
        x = features[-1]                        # bottleneck (deepest)
        skips = list(reversed(features[:-1]))   # [stage2, stage1, stage0]

        for i, block in enumerate(self.blocks):
            if i < len(skips):
                # Upsample to skip's exact spatial size to handle encoder rounding
                x = F.interpolate(
                    x, size=skips[i].shape[2:], mode="bilinear", align_corners=False
                )
                x = torch.cat([x, skips[i]], dim=1)
            else:
                # Final stage: no skip, plain 2× upsample
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            
            # Use gradient checkpointing for memory efficiency
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return x


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )
