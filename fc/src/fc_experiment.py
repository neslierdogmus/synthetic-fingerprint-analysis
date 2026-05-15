import random
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_SRC_ROOT = Path(__file__).resolve().parent

for import_path in (PROJECT_ROOT, TASK_SRC_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from src.experiment import Experiment
from src.metric import AcceptableClassAccuracyMetric
from src.trainer import StandardTrainer
from src.validator import StandardValidator
from src.evaluator import StandardEvaluator
from src.utils import (
    create_grouped_split_definition,
    create_loss_function,
    create_optimizer,
    create_scheduler,
)

from fc_fp_dataset import FCFPDataset
from fc_model_factory import create_model


class FCExperiment(Experiment):
    def __init__(self, config_path: str, config_overrides=None):
        super().__init__(config_path, 'fc', config_overrides=config_overrides)
        self.fp_group_ids = [str(fp_id) for fp_id in range(int(self.config.dataset.img_num))]

    def setup(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def create_split_definition(self):
        return create_grouped_split_definition(
            self.fp_group_ids,
            num_folds=self.num_folds,
            test_size=self.test_size,
            seed=self.seed,
        )

    def _create_training_stack(self):
        model = create_model(self.config).to(self.device)
        optimizer = create_optimizer(model, self.config.train.optimizer)
        scheduler_config = self.config.train.get('scheduler', None)
        scheduler = create_scheduler(optimizer, scheduler_config) if scheduler_config else None
        loss_fn = create_loss_function(self.config.train.loss_function)
        metric = AcceptableClassAccuracyMetric()

        trainer = StandardTrainer(model, optimizer, loss_fn, self.device, scheduler)
        patience = self.config.train.get('early_stopping', {}).get('patience', None)
        min_delta = self.config.train.get('early_stopping', {}).get('min_delta', 0.001)
        validator = StandardValidator(model, loss_fn, metric, self.device, patience, min_delta)
        evaluator = StandardEvaluator(model, metric, self.device)
        return model, trainer, validator, evaluator

    def _build_loader(self, fp_ids, shuffle: bool):
        dataset = FCFPDataset(self.config.dataset.root, fp_ids)
        if shuffle:
            augmentation_config = self.config.train.get('augmentation', {})
            if augmentation_config.get('horizontal_flip', False):
                dataset.set_hflip()
            rotation_degrees = int(augmentation_config.get('rotation_degrees', 0))
            if rotation_degrees > 0:
                dataset.set_rotate(rotation_degrees)
            brightness = float(augmentation_config.get('brightness', 0.0))
            contrast = float(augmentation_config.get('contrast', 0.0))
            if brightness > 0 or contrast > 0:
                dataset.set_color_jitter(brightness=brightness,
                                         contrast=contrast)
            blur_config = augmentation_config.get('gaussian_blur', {})
            if blur_config.get('enabled', False):
                dataset.set_gaussian_blur(
                    kernel_size=blur_config.get('kernel_size', 3),
                    sigma=blur_config.get('sigma', (0.1, 1.0)),
                    probability=blur_config.get('probability', 0.0),
                )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.config.batch_size),
            shuffle=shuffle,
            num_workers=int(self.config.num_workers),
        )

    def get_fold_group_ids(self, fold_idx: int):
        if self.split_definition is None:
            raise RuntimeError('Split definition has not been loaded yet.')

        fold = self.split_definition.folds[fold_idx]
        return list(fold.train_groups), list(fold.val_groups)

    def build_loader_for_group_ids(self, group_ids, shuffle: bool):
        fp_ids = [int(group_id) for group_id in group_ids]
        return self._build_loader(fp_ids, shuffle)

    def get_dataloaders(self, fold_idx: int):
        train_group_ids, val_group_ids = self.get_fold_group_ids(fold_idx)
        train_loader = self.build_loader_for_group_ids(train_group_ids, shuffle=True)
        val_loader = self.build_loader_for_group_ids(val_group_ids, shuffle=False)
        return {
            'train': train_loader,
            'val': val_loader,
        }

    def _get_test_dataloaders(self):
        if self.split_definition is None:
            raise RuntimeError('Split definition has not been loaded yet.')

        test_group_ids = set(self.split_definition.test_groups)
        train_val_ids = [int(group_id) for group_id in self.fp_group_ids if group_id not in test_group_ids]
        test_ids = [int(group_id) for group_id in self.split_definition.test_groups]
        return {
            'train': self._build_loader(train_val_ids, shuffle=True),
            'test': self._build_loader(test_ids, shuffle=False),
        }

    def _run_training_loop(self, trainer, validator, train_loader, val_loader=None, split_label='Fold', split_idx=None):
        eval_interval = int(self.config.train.get('eval_interval', 10))
        num_epochs = int(self.config.train.num_epochs)
        resume_config = self.config.train.get('resume', {})
        resume_enabled = bool(resume_config.get('enabled', False))
        resume_checkpoint_name = None
        if split_idx is not None:
            resume_checkpoint_name = f'{split_label.lower()}_{split_idx}_resume.ckpt'
        elif split_label:
            resume_checkpoint_name = f"{split_label.lower().replace(' ', '_')}_resume.ckpt"

        start_epoch = 0
        if resume_enabled and resume_checkpoint_name is not None:
            checkpoint_path = self.checkpoints_dir / resume_checkpoint_name
            if checkpoint_path.exists():
                resume_state = self.load_training_checkpoint(
                    resume_checkpoint_name,
                    trainer.model,
                    trainer.optimizer,
                    trainer.scheduler,
                )
                start_epoch = int(resume_state.get('epoch', 0))
                extra_state = resume_state.get('extra_state', {})
                if isinstance(extra_state, dict):
                    validator.best_loss = float(extra_state.get('validator_best_loss', validator.best_loss))
                    validator.counter = int(extra_state.get('validator_counter', validator.counter))
                print(f"Resumed {split_label} from epoch {start_epoch}")

        final_metrics = None

        for epoch in range(start_epoch, num_epochs):
            train_loss = trainer.train_epoch(train_loader)

            if val_loader is None:
                if epoch % eval_interval == 0 or epoch == num_epochs - 1:
                    label = split_label if split_idx is None else f'{split_label} {split_idx}'
                    print(f"{label}, Epoch {epoch}: Train Loss {train_loss:.4f}")
                continue

            if epoch % eval_interval != 0 and epoch != num_epochs - 1:
                continue

            final_metrics = validator.validate(val_loader)
            label = split_label if split_idx is None else f'{split_label} {split_idx}'
            print(
                f"{label}, Epoch {epoch}: "
                f"Train Loss {train_loss:.4f}, "
                f"Val Loss {final_metrics['loss']:.4f}, "
                f"Accuracy {final_metrics['accuracy']:.4f}"
            )
            if validator.check_early_stopping(final_metrics['loss']):
                print(f"Early stopping at epoch {epoch}")
                if resume_enabled and resume_checkpoint_name is not None:
                    self.delete_checkpoint(resume_checkpoint_name)
                break

            if resume_enabled and resume_checkpoint_name is not None:
                self.save_training_checkpoint(
                    resume_checkpoint_name,
                    trainer.model,
                    trainer.optimizer,
                    trainer.scheduler,
                    epoch=epoch + 1,
                    extra_state={
                        'validator_best_loss': float(validator.best_loss),
                        'validator_counter': int(validator.counter),
                    },
                )

        if resume_enabled and resume_checkpoint_name is not None:
            self.delete_checkpoint(resume_checkpoint_name)

        return final_metrics

    def run_group_split(self, train_group_ids, val_group_ids, split_label='Fold', split_idx=None, checkpoint_name=None):
        model, trainer, validator, evaluator = self._create_training_stack()
        train_loader = self.build_loader_for_group_ids(train_group_ids, shuffle=True)
        val_loader = self.build_loader_for_group_ids(val_group_ids, shuffle=False)

        final_val_metrics = self._run_training_loop(
            trainer,
            validator,
            train_loader,
            val_loader,
            split_label=split_label,
            split_idx=split_idx,
        )

        final_metrics = evaluator.evaluate(val_loader)
        if final_val_metrics is not None:
            final_metrics['loss'] = final_val_metrics['loss']

        self.model = model
        if checkpoint_name is not None:
            self.save_model(checkpoint_name)
        print(f"{split_label} {split_idx} Final: {final_metrics}")
        return final_metrics

    def run_fold(self, fold_idx: int):
        train_group_ids, val_group_ids = self.get_fold_group_ids(fold_idx)
        return self.run_group_split(
            train_group_ids,
            val_group_ids,
            split_label='Fold',
            split_idx=fold_idx,
            checkpoint_name=f'fold_{fold_idx}.pt',
        )

    def run_test_evaluation(self):
        model, trainer, validator, evaluator = self._create_training_stack()
        dataloaders = self._get_test_dataloaders()

        self._run_training_loop(
            trainer,
            validator,
            dataloaders['train'],
            split_label='Test train',
        )

        test_metrics = evaluator.evaluate(dataloaders['test'])
        self.model = model
        self.save_model('test.pt')
        print(f"Held-out test Final: {test_metrics}")
        return test_metrics


def main(config_path: str):
    experiment = FCExperiment(config_path)
    print(f"Using device: {experiment.device}")
    experiment.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FC baseline experiment.')
    parser.add_argument(
        '--config',
        default=str(Path(__file__).resolve().parents[1] / 'configs' / 'base.yaml'),
        help='Path to experiment config YAML.',
    )
    args = parser.parse_args()
    main(args.config)