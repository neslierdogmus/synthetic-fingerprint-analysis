from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from .metric import Metric


class Validator(ABC):
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, metric: Metric, device: torch.device, patience: Optional[int] = None, min_delta: Optional[float] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    @abstractmethod
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        pass

    def check_early_stopping(self, val_loss: float) -> bool:
        if self.patience is None or self.min_delta is None:
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


class StandardValidator(Validator):
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs, targets = batch['image'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                all_preds.append(outputs)
                all_targets.append(targets)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metrics = self.metric.compute(preds, targets)
        metrics['loss'] = total_loss / len(dataloader)
        return metrics