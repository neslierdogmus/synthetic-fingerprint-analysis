from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.schema import ExperimentConfig
from src.models.encoders.resnet50 import ResNet50Encoder
from src.models.encoders.swin_v2_t import SwinV2TEncoder
from src.models.decoders.unet_decoder import UNetDecoder
from src.models.heads.classification_head import ClassificationHead
from src.models.heads.fmd_head import FMDHead
from src.models.heads.foe_head import FOEHead


# ── Model wrapper ──────────────────────────────────────────────────────────────

class FingerprintModel(nn.Module):
    """
    Unified wrapper: encoder → (optional decoder) → head.

    FC task:      encoder → ClassificationHead  (no decoder)
    FMD/FOE task: encoder → UNetDecoder → FMDHead / FOEHead

    output_size: if set, the spatial output is interpolated to (H, W) after
    the head.  Used for FMD to ensure the prediction exactly matches the
    label size despite encoder/decoder integer rounding.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None,
        head: nn.Module,
        output_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder          # None for FC task
        self.head = head
        self.output_size = output_size  # (H, W) for FMD, None otherwise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if self.decoder is not None:
            decoded = self.decoder(features)
            out = self.head(decoded)
        else:
            out = self.head(features)
        if self.output_size is not None and out.ndim == 4:
            out = F.interpolate(
                out, size=self.output_size, mode="bilinear", align_corners=False
            )
        return out


# ── Assembly table ─────────────────────────────────────────────────────────────
#
# | Task | M1                      | M2                       | M3                       |
# |------|-------------------------|--------------------------|--------------------------|
# | FC   | ResNet50(scratch, ch=1) | ResNet50(pretrained,ch=3)| SwinV2T(pretrained,ch=3) |
# | FMD  | ResNet50(scratch, ch=1) | ResNet50(pretrained,ch=3)| SwinV2T(pretrained,ch=3) |
# | FOE  | ResNet50(scratch, ch=1) | ResNet50(pretrained,ch=3)| SwinV2T(pretrained,ch=3) |
#
# Decoder:  FC → None; FMD → UNet(n_stages=4); FOE → UNet(n_stages=2)
# Head:     FC → ClassificationHead; FMD → FMDHead; FOE → FOEHead


def _build_encoder(model_id: str, input_size: tuple[int, int] | None = None) -> nn.Module:
    if model_id == "M1":
        return ResNet50Encoder(pretrained=False, in_channels=1)
    if model_id == "M2":
        return ResNet50Encoder(pretrained=True, in_channels=3)
    if model_id == "M3":
        img_size = tuple(input_size) if input_size is not None else 256
        return SwinV2TEncoder(in_channels=3, img_size=img_size)
    raise ValueError(f"Unknown model_id: {model_id!r}")


def build_model(config: ExperimentConfig) -> FingerprintModel:
    """
    Build a FingerprintModel from an ExperimentConfig.

    Args:
        config: validated experiment configuration.
    Returns:
        Assembled FingerprintModel ready for training.
    """
    encoder = _build_encoder(config.model_id, input_size=tuple(config.input_size))

    if config.task == "fc":
        decoder = None
        head = ClassificationHead(
            in_channels=encoder.skip_channels[-1],
            num_classes=config.num_classes,
        )
        output_size = None

    elif config.task == "fmd":
        # Enable checkpointing for Swin (M3) to handle large padded tensors
        use_checkpoint = (config.model_id == "M3")
        decoder = UNetDecoder(
            encoder_skip_channels=encoder.skip_channels,
            n_stages=config.n_decoder_stages,
            use_checkpoint=use_checkpoint,
        )
        # Pass target size to head so it upsamples to original size, not padded size
        target_size = tuple(config.input_size)
        head = FMDHead(in_channels=decoder.out_channels, target_size=target_size)
        # No final interpolation needed since head already produces correct size
        output_size = None

    elif config.task == "foe":
        decoder = UNetDecoder(
            encoder_skip_channels=encoder.skip_channels,
            n_stages=config.n_decoder_stages,
        )
        head = FOEHead(in_channels=decoder.out_channels)
        output_size = None  # FOE output is at 1/8 resolution; no fixed size

    else:
        raise ValueError(f"Unknown task: {config.task!r}")

    return FingerprintModel(encoder=encoder, decoder=decoder, head=head,
                            output_size=output_size)


# ── Shape verification ─────────────────────────────────────────────────────────

def verify_output_shapes(config: ExperimentConfig) -> dict:
    """
    Run a forward pass with a zero dummy batch and report output shapes.

    The expected FOE shape (H//8, W//8) is approximate — the actual spatial
    dims depend on encoder integer-rounding and may differ by ±1 pixel.
    """
    model = build_model(config)
    model.eval()

    H, W = config.input_size
    in_channels = 1 if config.model_id == "M1" else 3
    x = torch.zeros(2, in_channels, H, W)

    with torch.no_grad():
        out = model(x)

    expected_shapes: dict[str, tuple] = {
        "fc":  (2, 5),
        "fmd": (2, 3, H, W),
        "foe": (2, 128, H // 8, W // 8),
    }

    return {
        "expected": expected_shapes[config.task],
        "actual":   tuple(out.shape),
    }
