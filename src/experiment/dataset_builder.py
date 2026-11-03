from __future__ import annotations

import json

import albumentations as A

from src.config.schema import ExperimentConfig
from src.data.base_dataset import FingerprintDataset
from src.data.real.fc_dataset import FCRealDataset
from src.data.real.fmd_dataset import FMDRealDataset
from src.data.real.foe_dataset import FOERealDataset
from src.data.synthetic.synthetic_dataset import SyntheticDataset
from src.training.transforms.common import (
    get_fc_transforms,
    get_fmd_transforms,
    get_foe_transforms,
)


def build_real_dataset(
    config: ExperimentConfig,
    filenames: list[str],
    transform=None,
    data_root: str = "/data",
) -> FingerprintDataset:
    """
    Build a real dataset for the given task and filenames.
    filenames: from split JSON (train_val_ids, test_ids, or fold train/val)
    """
    model_type = "scratch" if config.model_id == "M1" else "pretrained"

    if config.task == "fc":
        return FCRealDataset(
            data_root=f"{data_root}/real/NIST_SD4",
            filenames=filenames,
            transform=transform,
            model_type=model_type,
        )
    elif config.task == "fmd":
        return FMDRealDataset(
            data_root=f"{data_root}/real/FVC_BERKAY_TOPCU",
            filenames=filenames,
            transform=transform,
            model_type=model_type,
        )
    elif config.task == "foe":
        return FOERealDataset(
            data_root=f"{data_root}/real/FOESamples",
            filenames=filenames,
            transform=transform,
            model_type=model_type,
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")


def build_synthetic_dataset(
    config: ExperimentConfig,
    filenames: list[str],
    transform=None,
    data_root: str = "/data",
) -> SyntheticDataset | None:
    """
    Build synthetic dataset. Returns None if no synthetic data.
    """
    if not filenames:
        return None
    model_type = "scratch" if config.model_id == "M1" else "pretrained"
    return SyntheticDataset(
        data_root=f"{data_root}/synthetic",
        source=config.synthetic_source,
        filenames=filenames,
        task=config.task,
        transform=transform,
        model_type=model_type,
    )


def build_transform(config: ExperimentConfig, training: bool):
    """Build Albumentations transform for the given task."""
    if not training:
        H, W = config.input_size
        if config.task == "fmd":
            return A.Compose([
                A.PadIfNeeded(min_height=H, min_width=W, border_mode=0, value=0, mask_value=0, p=1.0),
                A.CenterCrop(height=H, width=W, p=1.0),
            ], additional_targets={"mask0": "mask", "mask1": "mask", "mask2": "mask"})
        return A.Compose([A.Resize(height=H, width=W)])

    H, W = config.input_size
    if config.task == "fc":
        return get_fc_transforms((H, W))
    elif config.task == "fmd":
        return get_fmd_transforms((H, W))
    elif config.task == "foe":
        return get_foe_transforms((H, W))
    else:
        raise ValueError(f"Unknown task: {config.task}")


def load_split(split_path: str) -> dict:
    with open(split_path) as f:
        return json.load(f)
