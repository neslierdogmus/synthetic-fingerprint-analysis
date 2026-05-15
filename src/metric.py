from abc import ABC, abstractmethod
from typing import Dict
import torch


class Metric(ABC):
    @abstractmethod
    def compute(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        pass


class AccuracyMetric(Metric):
    def compute(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds = preds.argmax(dim=1)
        targets = targets.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return {'accuracy': correct / total}


class AcceptableClassAccuracyMetric(Metric):
    def compute(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        predicted_classes = preds.argmax(dim=1)
        target_mask = targets > 0
        row_indices = torch.arange(targets.size(0), device=targets.device)
        correct = target_mask[row_indices, predicted_classes].sum().item()
        total = targets.size(0)
        return {'accuracy': correct / total}


class RMSEMetric(Metric):
    def compute(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        mse = torch.mean((preds - targets) ** 2)
        rmse = torch.sqrt(mse).item()
        return {'rmse': rmse}