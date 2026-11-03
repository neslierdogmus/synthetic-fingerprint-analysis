from __future__ import annotations

import albumentations as A


def get_fc_transforms(input_size: tuple[int, int]) -> A.Compose:
    """
    Albumentations pipeline for FC (fingerprint classification).
    Applied to image only; horizontal-flip class-label swap is handled
    in FCRealDataset.__getitem__ via ReplayCompose replay inspection.
    """
    H, W = input_size
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        # Guard CenterCrop against sources whose native resolution is
        # smaller than (H, W) (S01, S02, S10, S11 are all < 512x512) --
        # without this, CenterCrop raises CropSizeError whenever its p=0.3
        # draw fires on one of them. No-op for sources already >= (H, W),
        # matching CenterCrop's existing (silent) no-op behavior there.
        A.PadIfNeeded(min_height=H, min_width=W, border_mode=0, value=0, p=1.0),
        A.CenterCrop(height=H, width=W, p=0.3),
        A.ElasticTransform(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.RandomGamma(p=0.3),
        A.Resize(height=H, width=W),
    ])


def get_fmd_transforms(input_size: tuple[int, int]) -> A.Compose:
    """
    Albumentations pipeline for FMD (minutiae detection).
    Geometric transforms are applied jointly to image and each mask channel
    via additional_targets.  Photometric transforms (brightness, blur, noise,
    gamma) operate on the image key only — Albumentations does not apply them
    to additional mask targets.
    """
    H, W = input_size
    return A.Compose([
        A.PadIfNeeded(min_height=H, min_width=W, border_mode=0, value=0, mask_value=0, p=1.0),
        A.CenterCrop(height=H, width=W, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=0, p=0.5),
        A.ElasticTransform(border_mode=0, p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.RandomGamma(p=0.3),
        A.Resize(height=H, width=W),
    ], additional_targets={"mask0": "mask", "mask1": "mask", "mask2": "mask"})


def get_foe_transforms(input_size: tuple[int, int]) -> A.Compose:
    """
    Albumentations pipeline for FOE (orientation estimation) — photometric
    augmentations only.  Spatial transforms are handled by FOESpatialTransform
    because they require custom orientation-value updates.
    """
    H, W = input_size
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.RandomGamma(p=0.3),
        A.Resize(height=H, width=W),
    ])
