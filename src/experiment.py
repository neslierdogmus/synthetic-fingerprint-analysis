from abc import ABC, abstractmethod
import json
from typing import Any, Dict, Optional
import torch
from omegaconf import OmegaConf
from pathlib import Path

from src.utils import (
    GroupedSplitDefinition,
    load_grouped_split_definition,
    save_grouped_split_definition,
)


class Experiment(ABC):
    def __init__(self, config_path: str, task: str, config_overrides: Optional[Dict[str, Any]] = None):
        loaded_config = OmegaConf.load(config_path)
        if config_overrides:
            loaded_config = OmegaConf.merge(
                loaded_config,
                OmegaConf.create(config_overrides),
            )

        self.config = loaded_config
        config_data = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(config_data, dict):
            raise ValueError('Experiment config must resolve to a mapping.')

        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        protocol_config = config_data.get('protocol', {})
        if not isinstance(protocol_config, dict):
            raise ValueError('The protocol config must resolve to a mapping.')

        self.seed = int(protocol_config.get('seed', 42))
        self.num_folds = int(protocol_config.get('num_folds', 5))
        self.test_size = float(protocol_config.get('test_size', 0.2))
        self.split_filename = str(protocol_config.get('split_filename', 'grouped_splits.json'))
        
        # Will be set in setup
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.trainer = None
        self.validator = None
        self.evaluator = None
        self.metric = None
        self.split_definition: Optional[GroupedSplitDefinition] = None
        self.test_results: Optional[Dict[str, Any]] = None
        
        # Paths
        self.output_dir = Path(config_data.get('output_dir', f'{task}/outputs'))
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.results_dir = self.output_dir / 'results'
        self.splits_dir = self.output_dir / 'splits'

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def create_split_definition(self) -> GroupedSplitDefinition:
        raise NotImplementedError

    @abstractmethod
    def get_dataloaders(self, fold_idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run_fold(self, fold_idx: int):
        raise NotImplementedError

    def run_test_evaluation(self) -> Optional[Dict[str, Any]]:
        return None

    def get_split_file_path(self) -> Path:
        return self.splits_dir / self.split_filename

    def load_or_create_split_definition(self, force_rebuild: bool = False) -> GroupedSplitDefinition:
        split_file_path = self.get_split_file_path()
        if split_file_path.exists() and not force_rebuild:
            self.split_definition = load_grouped_split_definition(str(split_file_path))
            return self.split_definition

        self.split_definition = self.create_split_definition()
        save_grouped_split_definition(self.split_definition, str(split_file_path))
        return self.split_definition

    def run_experiment(self, force_rebuild_splits: bool = False, results_filename: str = 'real_baseline.json') -> Dict[str, Any]:
        self.setup()
        split_definition = self.load_or_create_split_definition(force_rebuild=force_rebuild_splits)

        fold_results = []
        for fold_idx in range(len(split_definition.folds)):
            metrics = self.run_fold(fold_idx)
            fold_results.append(metrics)

        self.test_results = self.run_test_evaluation()

        results: Dict[str, Any] = {'folds': fold_results}
        if self.test_results is not None:
            results['test'] = self.test_results
        self.save_named_results(results, results_filename)
        return results

    def save_model(self, filename: str):
        if self.model is None:
            raise RuntimeError('Cannot save a model before it has been created.')
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoints_dir / filename)

    def save_training_checkpoint(
        self,
        filename: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        extra_state: Optional[Dict[str, Any]] = None,
    ):
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            payload['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            payload['scheduler_state_dict'] = scheduler.state_dict()
        if extra_state:
            payload['extra_state'] = dict(extra_state)
        torch.save(payload, self.checkpoints_dir / filename)

    def load_training_checkpoint(
        self,
        filename: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        checkpoint_path = self.checkpoints_dir / filename
        payload = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(payload['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in payload:
            scheduler.load_state_dict(payload['scheduler_state_dict'])

        return {
            'epoch': int(payload.get('epoch', 0)),
            'extra_state': payload.get('extra_state', {}),
        }

    def delete_checkpoint(self, filename: str):
        checkpoint_path = self.checkpoints_dir / filename
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def load_model(self, filename: str):
        if self.model is None:
            raise RuntimeError('Cannot load weights before the model has been created.')
        checkpoint_path = self.checkpoints_dir / filename
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def save_results(self, metrics: Dict[str, Any], fold_idx: int):
        results_path = self.results_dir / f'results_fold_{fold_idx}.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open('w', encoding='utf-8') as file_handle:
            json.dump(metrics, file_handle, indent=2, sort_keys=True)

    def save_named_results(self, metrics: Dict[str, Any], filename: str):
        results_path = self.results_dir / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open('w', encoding='utf-8') as file_handle:
            json.dump(metrics, file_handle, indent=2, sort_keys=True)