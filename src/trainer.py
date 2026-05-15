from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch.utils.data import DataLoader


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler

    @abstractmethod
    def train_epoch(self, dataloader: DataLoader) -> float:
        pass


class StandardTrainer(Trainer):
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs, targets = batch['image'].to(self.device), batch['label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        if self.scheduler:
            self.scheduler.step()
        return total_loss / len(dataloader)