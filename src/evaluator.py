from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch.utils.data import DataLoader
from .metric import Metric


class Evaluator(ABC):
    def __init__(self, model: torch.nn.Module, metric: Metric, device: torch.device):
        self.model = model
        self.metric = metric
        self.device = device

    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        pass


class StandardEvaluator(Evaluator):
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs, targets = batch['image'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs)
                all_targets.append(targets)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        return self.metric.compute(preds, targets)