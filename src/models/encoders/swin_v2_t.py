from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinV2TEncoder(nn.Module):
    """
    Swin Transformer V2 Tiny encoder using timm features_only mode.

    Always pretrained, always 3-channel input (M3 only).

    The model is created with the padded input size (nearest multiple of
    patch_size × window_size = 32) so that window attention partitioning
    works for any input resolution.  The forward pass pads/unpadding is
    transparent — the caller uses FingerprintModel.output_size to crop
    the final prediction back to the original resolution.

    Returns 4 skip-connection feature maps from shallowest to deepest:
      Stage 0: [B, 96,  H'/4,  W'/4 ]
      Stage 1: [B, 192, H'/8,  W'/8 ]
      Stage 2: [B, 384, H'/16, W'/16]
      Stage 3: [B, 768, H'/32, W'/32]
    where H', W' are the padded dimensions.
    """

    # patch_size=4, window_size=8, 4 stages  → all feature-map dims must be
    # divisible by window_size at every stage after downsampling:
    #   stage0 needs H/4  % 8 == 0   → H % 32  == 0
    #   stage1 needs H/8  % 8 == 0   → H % 64  == 0
    #   stage2 needs H/16 % 8 == 0   → H % 128 == 0
    #   stage3 needs H/32 % 8 == 0   → H % 256 == 0   ← strictest
    _STRIDE = 256

    def __init__(self, in_channels: int = 3, img_size: int | tuple[int, int] = 256):
        super().__init__()
        H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
        padded_H = ((H + self._STRIDE - 1) // self._STRIDE) * self._STRIDE
        padded_W = ((W + self._STRIDE - 1) // self._STRIDE) * self._STRIDE
        self._pad_h = padded_H - H
        self._pad_w = padded_W - W

        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=(padded_H, padded_W),
        )
        # Enable gradient checkpointing to reduce memory with large batch sizes
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self._pad_h > 0 or self._pad_w > 0:
            x = F.pad(x, (0, self._pad_w, 0, self._pad_h))
        raw = self.backbone(x)
        result = []
        for expected_c, f in zip(self.skip_channels, raw):
            if f.ndim == 4 and f.shape[1] != expected_c:
                # timm Swin returns NHWC; convert to NCHW
                f = f.permute(0, 3, 1, 2).contiguous()
            result.append(f)
        return result

    @property
    def skip_channels(self) -> list[int]:
        return [96, 192, 384, 768]

    @property
    def output_stride(self) -> int:
        return 32
