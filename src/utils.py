import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


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


def _validate_group_labels(group_ids, group_labels):
    if group_labels is None:
        return
    missing_labels = sorted(set(group_ids) - set(group_labels))
    if missing_labels:
        missing_preview = ', '.join(missing_labels[:5])
        raise ValueError(f"Missing group labels for: {missing_preview}")


def _bucket_group_ids(group_ids, group_labels):
    if group_labels is None:
        return {'__all__': list(group_ids)}
    buckets = {}
    for group_id in group_ids:
        buckets.setdefault(group_labels[group_id], []).append(group_id)
    return buckets


def _test_group_count(bucket_size: int, test_size: float) -> int:
    if bucket_size < 2:
        return 0
    return min(int(round(bucket_size * test_size)), bucket_size - 1)


def create_grouped_holdout_split(group_ids, test_size=0.2, seed=42,
                                  group_labels=None):
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1.')
    unique_group_ids = _ordered_unique(group_ids)
    if len(unique_group_ids) < 2:
        raise ValueError('At least two groups are required.')
    _validate_group_labels(unique_group_ids, group_labels)
    rng = random.Random(seed)
    train_val_groups = []
    test_groups = []
    buckets = _bucket_group_ids(unique_group_ids, group_labels)
    for bucket_group_ids in buckets.values():
        shuffled = sorted(bucket_group_ids)
        rng.shuffle(shuffled)
        n_test = _test_group_count(len(shuffled), test_size)
        test_groups.extend(shuffled[:n_test])
        train_val_groups.extend(shuffled[n_test:])
    if not test_groups:
        shuffled = sorted(unique_group_ids)
        rng.shuffle(shuffled)
        test_groups = [shuffled[0]]
        train_val_groups = shuffled[1:]
    if not train_val_groups:
        raise ValueError('No groups left for training and validation.')
    return {
        'train_val_groups': sorted(train_val_groups),
        'test_groups': sorted(test_groups),
    }


def create_grouped_cv_splits(group_ids, num_folds, seed=42, group_labels=None):
    if num_folds < 2:
        raise ValueError('num_folds must be at least 2.')
    unique_group_ids = _ordered_unique(group_ids)
    if len(unique_group_ids) < num_folds:
        raise ValueError('Number of groups must be at least num_folds.')
    _validate_group_labels(unique_group_ids, group_labels)
    rng = random.Random(seed)
    buckets = _bucket_group_ids(unique_group_ids, group_labels)
    fold_group_ids = [[] for _ in range(num_folds)]
    for bucket_group_ids in buckets.values():
        shuffled = sorted(bucket_group_ids)
        rng.shuffle(shuffled)
        for i, gid in enumerate(shuffled):
            fold_group_ids[i % num_folds].append(gid)
    all_group_ids = set(unique_group_ids)
    fold_splits = []
    for val_group_ids in fold_group_ids:
        if not val_group_ids:
            raise ValueError('A validation fold ended up empty.')
        val_group_ids = sorted(val_group_ids)
        train_group_ids = sorted(all_group_ids - set(val_group_ids))
        fold_splits.append(GroupedFoldSplit(
            train_groups=train_group_ids,
            val_groups=val_group_ids,
        ))
    return fold_splits


def create_grouped_split_definition(group_ids, num_folds, test_size=0.2,
                                     seed=42, group_labels=None):
    holdout = create_grouped_holdout_split(group_ids, test_size=test_size,
                                           seed=seed, group_labels=group_labels)
    folds = create_grouped_cv_splits(holdout['train_val_groups'],
                                      num_folds=num_folds, seed=seed,
                                      group_labels=group_labels)
    return GroupedSplitDefinition(
        test_size=test_size,
        num_folds=num_folds,
        seed=seed,
        test_groups=holdout['test_groups'],
        folds=folds,
    )


def save_grouped_split_definition(split_definition, file_path):
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(split_definition.to_dict(), f, indent=2, sort_keys=True)


def load_grouped_split_definition(file_path):
    with Path(file_path).open('r', encoding='utf-8') as f:
        payload = json.load(f)
    return GroupedSplitDefinition.from_dict(payload)


def groups_to_indices(sample_group_ids, selected_group_ids):
    selected_set = set(selected_group_ids)
    missing = sorted(selected_set - set(sample_group_ids))
    if missing:
        raise ValueError(f"Requested groups missing: {', '.join(missing[:5])}")
    return [i for i, gid in enumerate(sample_group_ids)
            if gid in selected_set]


def materialize_grouped_split_definition(sample_group_ids, split_definition):
    return {
        'test_indices': groups_to_indices(sample_group_ids,
                                           split_definition.test_groups),
        'folds': [
            {
                'train_indices': groups_to_indices(sample_group_ids,
                                                    fold.train_groups),
                'val_indices': groups_to_indices(sample_group_ids,
                                                  fold.val_groups),
            }
            for fold in split_definition.folds
        ],
    }
