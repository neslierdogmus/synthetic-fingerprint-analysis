import random
import sys
from pathlib import Path

import numpy as np
from skimage.feature import peak_local_max
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_SRC_ROOT = Path(__file__).resolve().parent

for import_path in (PROJECT_ROOT, TASK_SRC_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from src.experiment import Experiment
from src.trainer import StandardTrainer
from src.utils import (
    create_grouped_split_definition,
    create_loss_function,
    create_optimizer,
    create_scheduler,
)

from fmd_fp_dataset import FMDFPImageDataset
from fmd_model_convnet import FMDConvNet


def extract_predicted_points(logits: torch.Tensor, min_distance: int) -> np.ndarray:
    heatmap = logits.detach().cpu().numpy()
    return peak_local_max(heatmap, min_distance=min_distance, exclude_border=False)


def extract_ground_truth_points(gt_minutiae_yx, item_idx: int) -> np.ndarray:
    gt_y, gt_x = gt_minutiae_yx
    gt_y = np.asarray(gt_y[item_idx])
    gt_x = np.asarray(gt_x[item_idx])
    valid_mask = np.logical_and(gt_y >= 0, gt_x >= 0)
    return np.stack([gt_y[valid_mask], gt_x[valid_mask]], axis=1)


def match_points(predicted_points: np.ndarray, gt_points: np.ndarray, radius: int) -> int:
    if len(predicted_points) == 0 or len(gt_points) == 0:
        return 0

    distances = np.zeros((len(predicted_points), len(gt_points)), dtype=float)
    for pred_idx, predicted_point in enumerate(predicted_points):
        for gt_idx, gt_point in enumerate(gt_points):
            distances[pred_idx, gt_idx] = np.linalg.norm(predicted_point - gt_point)

    matched_predictions = set()
    matched_ground_truth = set()
    true_positives = 0

    flat_indices = np.argsort(distances, axis=None)
    for flat_index in flat_indices:
        pred_idx, gt_idx = np.unravel_index(flat_index, distances.shape)
        if distances[pred_idx, gt_idx] > radius:
            break
        if pred_idx in matched_predictions or gt_idx in matched_ground_truth:
            continue
        matched_predictions.add(pred_idx)
        matched_ground_truth.add(gt_idx)
        true_positives += 1

    return true_positives


def precision_recall_f1(true_positives: int, num_predictions: int, num_ground_truth: int) -> dict:
    recall = true_positives / num_ground_truth if num_ground_truth > 0 else 0.0
    precision = true_positives / num_predictions if num_predictions > 0 else 0.0
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1_score,
    }


class FMDExperiment(Experiment):
    def __init__(self, config_path: str):
        super().__init__(config_path, 'fmd')
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
        model = FMDConvNet(int(self.config.model.get('size', 8))).to(self.device)
        optimizer = create_optimizer(model, self.config.train.optimizer)
        scheduler = create_scheduler(optimizer, self.config.train.scheduler)
        loss_fn = create_loss_function(self.config.train.loss_function)
        trainer = StandardTrainer(model, optimizer, loss_fn, self.device, scheduler)
        return model, trainer, loss_fn

    def _build_loader(self, fp_ids, shuffle: bool):
        dataset = FMDFPImageDataset(self.config.dataset.root, fp_ids, 1)
        if shuffle:
            dataset.set_hflip()
            dataset.set_rotate()
            dataset.set_resize()

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.config.batch_size),
            shuffle=shuffle,
            num_workers=int(self.config.num_workers),
        )

    def get_dataloaders(self, fold_idx: int):
        if self.split_definition is None:
            raise RuntimeError('Split definition has not been loaded yet.')

        fold = self.split_definition.folds[fold_idx]
        train_ids = [int(group_id) for group_id in fold.train_groups]
        val_ids = [int(group_id) for group_id in fold.val_groups]
        return {
            'train': self._build_loader(train_ids, shuffle=True),
            'val': self._build_loader(val_ids, shuffle=False),
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

    def _evaluate_loader(self, model, loss_fn, dataloader):
        radii = [int(radius) for radius in self.config.eval.radii]
        min_distance = int(self.config.eval.get('nms_min_distance', 3))
        image_level = {
            radius: {'recall': [], 'precision': [], 'f1': []}
            for radius in radii
        }
        pooled = {
            radius: {'tp': 0, 'num_predictions': 0, 'num_ground_truth': 0}
            for radius in radii
        }

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets, gt_minutiae_yx, _, _ = batch
                outputs = model(inputs.to(self.device))
                loss = loss_fn(outputs, targets.to(self.device))
                total_loss += loss.item()

                output_maps = outputs[:, 0, :, :]
                for item_idx in range(output_maps.shape[0]):
                    predicted_points = extract_predicted_points(output_maps[item_idx], min_distance)
                    gt_points = extract_ground_truth_points(gt_minutiae_yx, item_idx)
                    for radius in radii:
                        true_positives = match_points(predicted_points, gt_points, radius)
                        metrics = precision_recall_f1(
                            true_positives,
                            len(predicted_points),
                            len(gt_points),
                        )
                        image_level[radius]['recall'].append(metrics['recall'])
                        image_level[radius]['precision'].append(metrics['precision'])
                        image_level[radius]['f1'].append(metrics['f1'])
                        pooled[radius]['tp'] += true_positives
                        pooled[radius]['num_predictions'] += len(predicted_points)
                        pooled[radius]['num_ground_truth'] += len(gt_points)

        metrics = {'loss': total_loss / len(dataloader)}
        for radius in radii:
            pooled_metrics = precision_recall_f1(
                pooled[radius]['tp'],
                pooled[radius]['num_predictions'],
                pooled[radius]['num_ground_truth'],
            )
            metrics[f'radius_{radius}'] = {
                'image_level': {
                    'recall_mean': float(np.mean(image_level[radius]['recall'])) if image_level[radius]['recall'] else 0.0,
                    'precision_mean': float(np.mean(image_level[radius]['precision'])) if image_level[radius]['precision'] else 0.0,
                    'f1_mean': float(np.mean(image_level[radius]['f1'])) if image_level[radius]['f1'] else 0.0,
                },
                'pooled': pooled_metrics,
            }
        return metrics

    def _should_stop_early(self, current_loss: float, best_loss: float, counter: int):
        patience = self.config.train.get('early_stopping', {}).get('patience', None)
        min_delta = self.config.train.get('early_stopping', {}).get('min_delta', None)
        if patience is None or min_delta is None:
            return False, best_loss, counter

        if current_loss < best_loss - min_delta:
            return False, current_loss, 0

        counter += 1
        return counter >= patience, best_loss, counter

    def _run_training_loop(self, model, trainer, loss_fn, train_loader, val_loader=None, split_label='Fold', split_idx=None):
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
        best_loss = float('inf')
        patience_counter = 0

        if resume_enabled and resume_checkpoint_name is not None:
            checkpoint_path = self.checkpoints_dir / resume_checkpoint_name
            if checkpoint_path.exists():
                resume_state = self.load_training_checkpoint(
                    resume_checkpoint_name,
                    model,
                    trainer.optimizer,
                    trainer.scheduler,
                )
                start_epoch = int(resume_state.get('epoch', 0))
                extra_state = resume_state.get('extra_state', {})
                if isinstance(extra_state, dict):
                    best_loss = float(extra_state.get('best_loss', best_loss))
                    patience_counter = int(extra_state.get('patience_counter', patience_counter))
                print(f"Resumed {split_label} from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            train_loss = trainer.train_epoch(train_loader)
            label = split_label if split_idx is None else f'{split_label} {split_idx}'

            if val_loader is None:
                if epoch % eval_interval == 0 or epoch == num_epochs - 1:
                    print(f"{label}, Epoch {epoch}: Train Loss {train_loss:.4f}")
                continue

            if epoch % eval_interval != 0 and epoch != num_epochs - 1:
                continue

            final_metrics = self._evaluate_loader(model, loss_fn, val_loader)
            radius_key = f"radius_{int(self.config.eval.radii[-1])}"
            image_metrics = final_metrics[radius_key]['image_level']
            print(
                f"{label}, Epoch {epoch}: "
                f"Train Loss {train_loss:.4f}, "
                f"Val Loss {final_metrics['loss']:.4f}, "
                f"F1@{self.config.eval.radii[-1]} {image_metrics['f1_mean']:.4f}"
            )

            should_stop, best_loss, patience_counter = self._should_stop_early(
                final_metrics['loss'],
                best_loss,
                patience_counter,
            )
            if should_stop:
                print(f"Early stopping at epoch {epoch}")
                if resume_enabled and resume_checkpoint_name is not None:
                    self.delete_checkpoint(resume_checkpoint_name)
                break

            if resume_enabled and resume_checkpoint_name is not None:
                self.save_training_checkpoint(
                    resume_checkpoint_name,
                    model,
                    trainer.optimizer,
                    trainer.scheduler,
                    epoch=epoch + 1,
                    extra_state={
                        'best_loss': float(best_loss),
                        'patience_counter': int(patience_counter),
                    },
                )

        if resume_enabled and resume_checkpoint_name is not None:
            self.delete_checkpoint(resume_checkpoint_name)

        return final_metrics

    def run_fold(self, fold_idx: int):
        model, trainer, loss_fn = self._create_training_stack()
        dataloaders = self.get_dataloaders(fold_idx)
        final_metrics = self._run_training_loop(
            model,
            trainer,
            loss_fn,
            dataloaders['train'],
            dataloaders['val'],
            split_label='Fold',
            split_idx=fold_idx,
        )

        if final_metrics is None:
            final_metrics = self._evaluate_loader(model, loss_fn, dataloaders['val'])

        self.model = model
        self.save_model(f'fold_{fold_idx}.pt')
        print(f"Fold {fold_idx} Final: {final_metrics}")
        return final_metrics

    def run_test_evaluation(self):
        model, trainer, loss_fn = self._create_training_stack()
        dataloaders = self._get_test_dataloaders()
        self._run_training_loop(
            model,
            trainer,
            loss_fn,
            dataloaders['train'],
            split_label='Test train',
        )

        test_metrics = self._evaluate_loader(model, loss_fn, dataloaders['test'])
        self.model = model
        self.save_model('test.pt')
        print(f"Held-out test Final: {test_metrics}")
        return test_metrics


def main(config_path: str):
    experiment = FMDExperiment(config_path)
    print(f"Using device: {experiment.device}")
    experiment.run_experiment()


if __name__ == "__main__":
    main(str(Path(__file__).resolve().parents[1] / 'configs' / 'base.yaml'))
