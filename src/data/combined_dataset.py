from __future__ import annotations

from torch.utils.data import ConcatDataset

from src.data.base_dataset import FingerprintDataset


class CombinedDataset(ConcatDataset):
    """
    Concatenates a real dataset and an optional synthetic dataset.
    Adds an "is_synthetic" flag to every returned sample.
    Real and synthetic samples remain in separate pools — identities are never mixed.
    """

    def __init__(
        self,
        real_dataset: FingerprintDataset,
        synthetic_dataset: FingerprintDataset | None = None,
    ):
        datasets = [real_dataset]
        if synthetic_dataset is not None:
            datasets.append(synthetic_dataset)
        super().__init__(datasets)
        self.real_n = len(real_dataset)
        self.synthetic_n = len(synthetic_dataset) if synthetic_dataset is not None else 0
        self.collate_fn = getattr(real_dataset, "collate_fn", None)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        item["is_synthetic"] = index >= self.real_n
        return item
