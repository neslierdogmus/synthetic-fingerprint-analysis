import json
import inspect
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

"""Common utility functions for configuration and object creation"""

from omegaconf import OmegaConf
import torch


_REPORTED_JSONL_RECORD_KEYS: Set[Tuple[str, str]] = set()


@dataclass(frozen=True)
class GroupedFoldSplit:
    train_groups: List[str]
    val_groups: List[str]


@dataclass(frozen=True)
class GroupedSplitDefinition:
    test_size: float
    num_folds: int
    seed: int
    test_groups: List[str]
    folds: List[GroupedFoldSplit]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data['folds'] = [asdict(fold) for fold in self.folds]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> 'GroupedSplitDefinition':
        fold_dicts = data.get('folds', [])
        folds = [GroupedFoldSplit(**fold_dict) for fold_dict in fold_dicts]
        return cls(
            test_size=float(data['test_size']),
            num_folds=int(data['num_folds']),
            seed=int(data['seed']),
            test_groups=list(data['test_groups']),
            folds=folds,
        )


def _ordered_unique(values: Sequence[str]) -> List[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _validate_group_labels(group_ids: Sequence[str], group_labels: Optional[Mapping[str, str]]) -> None:
    if group_labels is None:
        return

    missing_labels = sorted(set(group_ids) - set(group_labels))
    if missing_labels:
        missing_preview = ', '.join(missing_labels[:5])
        raise ValueError(f"Missing group labels for: {missing_preview}")


def _bucket_group_ids(group_ids: Sequence[str], group_labels: Optional[Mapping[str, str]]) -> Dict[str, List[str]]:
    if group_labels is None:
        return {'__all__': list(group_ids)}

    buckets: Dict[str, List[str]] = {}
    for group_id in group_ids:
        buckets.setdefault(group_labels[group_id], []).append(group_id)
    return buckets


def _test_group_count(bucket_size: int, test_size: float) -> int:
    if bucket_size < 2:
        return 0
    return min(int(round(bucket_size * test_size)), bucket_size - 1)


def create_grouped_holdout_split(group_ids: Sequence[str], test_size: float = 0.2, seed: int = 42, group_labels: Optional[Mapping[str, str]] = None) -> Dict[str, List[str]]:
    """Split unique group ids into train/validation and test partitions."""
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1.')

    unique_group_ids = _ordered_unique(group_ids)
    if len(unique_group_ids) < 2:
        raise ValueError('At least two groups are required to create a holdout split.')

    _validate_group_labels(unique_group_ids, group_labels)

    rng = random.Random(seed)
    train_val_groups: List[str] = []
    test_groups: List[str] = []
    buckets = _bucket_group_ids(unique_group_ids, group_labels)

    for bucket_group_ids in buckets.values():
        shuffled_group_ids = sorted(bucket_group_ids)
        rng.shuffle(shuffled_group_ids)
        num_test_groups = _test_group_count(len(shuffled_group_ids), test_size)
        test_groups.extend(shuffled_group_ids[:num_test_groups])
        train_val_groups.extend(shuffled_group_ids[num_test_groups:])

    if not test_groups:
        shuffled_group_ids = sorted(unique_group_ids)
        rng.shuffle(shuffled_group_ids)
        test_groups = [shuffled_group_ids[0]]
        train_val_groups = shuffled_group_ids[1:]

    if not train_val_groups:
        raise ValueError('The requested holdout split leaves no groups for training and validation.')

    return {
        'train_val_groups': sorted(train_val_groups),
        'test_groups': sorted(test_groups),
    }


def create_grouped_cv_splits(group_ids: Sequence[str], num_folds: int, seed: int = 42, group_labels: Optional[Mapping[str, str]] = None) -> List[GroupedFoldSplit]:
    """Create deterministic grouped cross-validation folds."""
    if num_folds < 2:
        raise ValueError('num_folds must be at least 2.')

    unique_group_ids = _ordered_unique(group_ids)
    if len(unique_group_ids) < num_folds:
        raise ValueError('The number of groups must be at least num_folds.')

    _validate_group_labels(unique_group_ids, group_labels)

    rng = random.Random(seed)
    buckets = _bucket_group_ids(unique_group_ids, group_labels)
    fold_group_ids = [[] for _ in range(num_folds)]

    for bucket_group_ids in buckets.values():
        shuffled_group_ids = sorted(bucket_group_ids)
        rng.shuffle(shuffled_group_ids)
        for index, group_id in enumerate(shuffled_group_ids):
            fold_group_ids[index % num_folds].append(group_id)

    all_group_ids = set(unique_group_ids)
    fold_splits = []
    for val_group_ids in fold_group_ids:
        if not val_group_ids:
            raise ValueError('At least one validation fold ended up empty. Reduce num_folds or add more groups.')
        val_group_ids = sorted(val_group_ids)
        train_group_ids = sorted(all_group_ids - set(val_group_ids))
        fold_splits.append(
            GroupedFoldSplit(
                train_groups=train_group_ids,
                val_groups=val_group_ids,
            )
        )

    return fold_splits


def create_grouped_split_definition(group_ids: Sequence[str], num_folds: int, test_size: float = 0.2, seed: int = 42, group_labels: Optional[Mapping[str, str]] = None) -> GroupedSplitDefinition:
    holdout_split = create_grouped_holdout_split(
        group_ids,
        test_size=test_size,
        seed=seed,
        group_labels=group_labels,
    )
    fold_splits = create_grouped_cv_splits(
        holdout_split['train_val_groups'],
        num_folds=num_folds,
        seed=seed,
        group_labels=group_labels,
    )
    return GroupedSplitDefinition(
        test_size=test_size,
        num_folds=num_folds,
        seed=seed,
        test_groups=holdout_split['test_groups'],
        folds=fold_splits,
    )


def save_grouped_split_definition(split_definition: GroupedSplitDefinition, file_path: str) -> None:
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file_handle:
        json.dump(split_definition.to_dict(), file_handle, indent=2, sort_keys=True)


def load_grouped_split_definition(file_path: str) -> GroupedSplitDefinition:
    with Path(file_path).open('r', encoding='utf-8') as file_handle:
        payload = json.load(file_handle)
    return GroupedSplitDefinition.from_dict(payload)


def safe_normalize_tensor(image: torch.Tensor, min_std: float = 1e-6) -> Tuple[torch.Tensor, float, float, bool]:
    mean_tensor = torch.mean(image)
    std_tensor = torch.std(image)
    mean = float(mean_tensor)
    std = float(std_tensor)
    requires_safe_std = not bool(torch.isfinite(std_tensor).item()) or std < min_std
    effective_std = image.new_tensor(min_std) if requires_safe_std else std_tensor
    normalized_image = (image - mean_tensor) / effective_std
    return normalized_image, mean, std, requires_safe_std


def append_unique_jsonl_record(report_path: Path | str, record_key: str, payload: Mapping[str, Any], warning_message: Optional[str] = None) -> bool:
    resolved_report_path = Path(report_path)
    dedupe_key = (str(resolved_report_path), record_key)
    if dedupe_key in _REPORTED_JSONL_RECORD_KEYS:
        return False

    _REPORTED_JSONL_RECORD_KEYS.add(dedupe_key)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_report_path.open('a', encoding='utf-8') as report_file:
        json.dump(dict(payload), report_file, sort_keys=True)
        report_file.write('\n')

    if warning_message is not None:
        print(warning_message, file=sys.stderr)

    return True


def groups_to_indices(sample_group_ids: Sequence[str], selected_group_ids: Sequence[str]) -> List[int]:
    selected_group_id_set = set(selected_group_ids)
    missing_groups = sorted(selected_group_id_set - set(sample_group_ids))
    if missing_groups:
        missing_preview = ', '.join(missing_groups[:5])
        raise ValueError(f"Requested groups are missing from the sample list: {missing_preview}")

    return [index for index, group_id in enumerate(sample_group_ids) if group_id in selected_group_id_set]


def materialize_grouped_split_definition(sample_group_ids: Sequence[str], split_definition: GroupedSplitDefinition) -> Dict[str, object]:
    return {
        'test_indices': groups_to_indices(sample_group_ids, split_definition.test_groups),
        'folds': [
            {
                'train_indices': groups_to_indices(sample_group_ids, fold.train_groups),
                'val_indices': groups_to_indices(sample_group_ids, fold.val_groups),
            }
            for fold in split_definition.folds
        ],
    }


def create_from_config(base_class, config_section, *args, **kwargs):
    """
    Generic function to create objects from config using spread unpacking.
    
    Args:
        base_class: The base module/class containing the factory (e.g., torch.optim, torch.nn)
        config_section: OmegaConf config section with 'name' field
        *args: Positional arguments to pass to the constructor
        **kwargs: Additional keyword arguments to pass to the constructor
    
    Returns:
        Instance of the created object
    """
    if isinstance(config_section, str):
        config_dict = {'name': config_section}
    elif isinstance(config_section, dict):
        config_dict = dict(config_section)
    else:
        config_dict = OmegaConf.to_container(config_section, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid configuration for {config_section}.")
    if 'name' not in config_dict:
        raise ValueError(f"{config_section} must have a 'name' field.")
    name = config_dict.pop('name')
    factory = getattr(base_class, name, None)
    if factory is None and isinstance(name, str):
        normalized_name = name.replace('_', '').lower()
        for candidate_name in dir(base_class):
            if candidate_name.replace('_', '').lower() == normalized_name:
                factory = getattr(base_class, candidate_name)
                break
    if factory is None:
        raise ValueError(f"Could not resolve '{name}' from {base_class}.")
    # Filter to ensure only string keys are passed
    config_dict = {k: v for k, v in config_dict.items() if isinstance(k, str)}
    return factory(*args, **config_dict, **kwargs)


def create_optimizer(model, optimizer_config):
    """Create optimizer from config"""
    if isinstance(optimizer_config, str):
        normalized_config = {'name': optimizer_config}
    elif isinstance(optimizer_config, dict):
        normalized_config = dict(optimizer_config)
    else:
        normalized_config = OmegaConf.to_container(optimizer_config, resolve=True)

    if 'learning_rate' in normalized_config and 'lr' not in normalized_config:
        normalized_config['lr'] = normalized_config.pop('learning_rate')

    optimizer_name = normalized_config['name']
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer_signature = inspect.signature(optimizer_class).parameters

    filtered_config = {'name': optimizer_name}
    for key, value in normalized_config.items():
        if key == 'name' or value is None:
            continue
        if key in optimizer_signature:
            filtered_config[key] = value

    return create_from_config(torch.optim, filtered_config, model.parameters())


def create_scheduler(optimizer, scheduler_config):
    """Create learning rate scheduler from config"""
    if isinstance(scheduler_config, str):
        normalized_config = {'name': scheduler_config}
    elif isinstance(scheduler_config, dict):
        normalized_config = dict(scheduler_config)
    else:
        normalized_config = OmegaConf.to_container(scheduler_config, resolve=True)

    scheduler_name = normalized_config['name']
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name, None)
    if scheduler_class is None and isinstance(scheduler_name, str):
        normalized_name = scheduler_name.replace('_', '').lower()
        for candidate_name in dir(torch.optim.lr_scheduler):
            if candidate_name.replace('_', '').lower() == normalized_name:
                scheduler_class = getattr(torch.optim.lr_scheduler, candidate_name)
                break
    if scheduler_class is None:
        raise ValueError(f"Could not resolve scheduler '{scheduler_name}'.")
    scheduler_signature = inspect.signature(scheduler_class).parameters

    filtered_config = {'name': scheduler_name}
    for key, value in normalized_config.items():
        if key == 'name' or value is None:
            continue
        if key in scheduler_signature:
            filtered_config[key] = value

    return create_from_config(torch.optim.lr_scheduler, filtered_config,
                              optimizer)


def create_loss_function(loss_config):
    """Create loss function from config"""
    return create_from_config(torch.nn, loss_config)


def create_kfold_splits(dataset, num_folds: int, shuffle: bool = True, random_state: int = 42):
    kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
    return list(kf.split(range(len(dataset))))


def create_dataloaders_from_splits(dataset, train_indices: List[int], val_indices: List[int], batch_size: int, num_workers: int = 1):
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader