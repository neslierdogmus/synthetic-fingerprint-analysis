from __future__ import annotations

from pathlib import Path

from src.config.schema import ExperimentConfig
from src.evaluation.fc_evaluator import (
    score_fold as fc_score_fold,
    evaluate_full as fc_evaluate_full,
)
from src.evaluation.fmd_evaluator import (
    score_fold as fmd_score_fold,
    evaluate_full as fmd_evaluate_full,
)
from src.evaluation.foe_evaluator import (
    score_fold as foe_score_fold,
    evaluate_full as foe_evaluate_full,
)


class Evaluator:
    """
    Dispatches to task-specific evaluators based on config.task.

    Two modes:
      score_fold()    — online, called by ExperimentRunner during HP tuning
      evaluate_full() — offline, called by analysis layer
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def score_fold(
        self,
        pred_path: Path,
        val_dataset,
        nms_params=None,   # FMD only
    ) -> float | dict:
        if self.config.task == "fc":
            return fc_score_fold(pred_path, val_dataset)
        elif self.config.task == "fmd":
            return fmd_score_fold(
                pred_path, val_dataset, nms_params,
                self.config.fmd_matching_distance,
                self.config.fmd_nms_thresholds,
                self.config.fmd_nms_radii,
            )
        elif self.config.task == "foe":
            return foe_score_fold(pred_path, val_dataset)
        else:
            raise ValueError(f"Unknown task: {self.config.task}")

    def evaluate_full(
        self,
        pred_path: Path,
        test_dataset,
        nms_params=None,   # FMD only
    ) -> dict:
        if self.config.task == "fc":
            return fc_evaluate_full(pred_path, test_dataset)
        elif self.config.task == "fmd":
            return fmd_evaluate_full(pred_path, test_dataset, nms_params)
        elif self.config.task == "foe":
            return foe_evaluate_full(pred_path, test_dataset)
        else:
            raise ValueError(f"Unknown task: {self.config.task}")
