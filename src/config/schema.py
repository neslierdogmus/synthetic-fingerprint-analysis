from __future__ import annotations

from pydantic import BaseModel, model_validator


class HPConfig(BaseModel):
    lr: float
    weight_decay: float
    cosine_annealing: bool


class FixedHPConfig(BaseModel):
    """
    Externally-provided HP choice for the Fixed-HP secondary experiment
    (experiment_design.md section 7.5): when set, ExperimentRunner skips HP
    Phase 1 and Phase 2 entirely and evaluates directly with this config,
    instead of searching hp_grid. hp_config indexes into hp_grid the same
    way a normal run's winning index would. nms_threshold/nms_radius are
    required for fmd (mirrors Phase 2's best_config triplet) and must be
    absent otherwise, since only fmd has an NMS-tuning stage to stand in for.
    """
    hp_config: int
    nms_threshold: float | None = None
    nms_radius: int | None = None


class ExperimentConfig(BaseModel):
    # Identity
    experiment_id: str
    task: str
    model_id: str
    condition: str
    synthetic_source: str | None = None

    # Data sizes
    real_n: int
    synthetic_n: int

    # Split files
    split_file_real: str
    split_file_synthetic: str | None = None

    # Training
    max_steps: int
    batch_size: int
    n_folds: int = 4
    n_seeds: int = 3
    hp_grid: list[HPConfig]
    fixed_hp: FixedHPConfig | None = None

    # Paths
    output_dir: str
    dev_mode: bool = False

    # Task-specific (optional, validated per task)
    input_size: list[int]
    num_classes: int | None = None
    n_decoder_stages: int | None = None
    fmd_nms_radii: list[int] | None = None
    fmd_nms_thresholds: list[float] | None = None
    fmd_matching_distance: int | None = None

    @model_validator(mode="after")
    def validate_task_fields(self) -> "ExperimentConfig":
        if self.task == "fc":
            if self.num_classes is None:
                raise ValueError("num_classes must be set when task == 'fc'")

        if self.task in ("fmd", "foe"):
            if self.n_decoder_stages is None:
                raise ValueError(
                    "n_decoder_stages must be set when task is 'fmd' or 'foe'"
                )

        if self.task == "fmd":
            if self.fmd_nms_radii is None:
                raise ValueError("fmd_nms_radii must be set when task == 'fmd'")
            if self.fmd_nms_thresholds is None:
                raise ValueError("fmd_nms_thresholds must be set when task == 'fmd'")
            if self.fmd_matching_distance is None:
                raise ValueError(
                    "fmd_matching_distance must be set when task == 'fmd'"
                )

        if self.fixed_hp is not None:
            if not (0 <= self.fixed_hp.hp_config < len(self.hp_grid)):
                raise ValueError(
                    f"fixed_hp.hp_config={self.fixed_hp.hp_config} out of range "
                    f"for hp_grid of length {len(self.hp_grid)}"
                )
            has_nms = (
                self.fixed_hp.nms_threshold is not None
                and self.fixed_hp.nms_radius is not None
            )
            if self.task == "fmd" and not has_nms:
                raise ValueError(
                    "fixed_hp.nms_threshold and nms_radius must both be set "
                    "when task == 'fmd'"
                )
            if self.task != "fmd" and has_nms:
                raise ValueError(
                    "fixed_hp.nms_threshold/nms_radius must be unset when "
                    "task != 'fmd' (only fmd has an NMS-tuning stage)"
                )

        baseline_conditions = {"V0", "R0"}
        if self.condition in baseline_conditions:
            if self.synthetic_source is not None:
                raise ValueError(
                    "synthetic_source must be None when condition is 'V0' or 'R0'"
                )
        else:
            if self.synthetic_source is None:
                raise ValueError(
                    "synthetic_source must be set when condition is not 'V0' or 'R0'"
                )

        return self
