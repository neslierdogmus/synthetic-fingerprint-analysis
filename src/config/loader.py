from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.config.schema import ExperimentConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on scalar conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_config(experiment_yaml_path: str, dev: bool = False) -> tuple[ExperimentConfig, dict]:
    """Resolve an experiment YAML to a validated config. No disk writes."""
    experiment_path = Path(experiment_yaml_path)

    # Locate project root by finding configs/ relative to experiment yaml.
    # Experiment yamls live at configs/{task}/experiments/{name}.yaml,
    # so project root is three levels up.
    project_root = experiment_path.parents[3]

    configs_dir = project_root / "configs"

    # 1. global defaults
    merged = _load_yaml(configs_dir / "global_defaults.yaml")

    # Determine task from experiment yaml (peek ahead)
    experiment_raw = _load_yaml(experiment_path)
    task = experiment_raw.get("task")
    if not task:
        raise ValueError(
            f"Experiment YAML {experiment_yaml_path} must specify 'task'."
        )

    # 2. task base
    merged = _deep_merge(merged, _load_yaml(configs_dir / task / "base.yaml"))

    # 3. task dev (only if dev=True)
    if dev:
        merged = _deep_merge(merged, _load_yaml(configs_dir / task / "dev.yaml"))

    # 4. experiment yaml
    merged = _deep_merge(merged, experiment_raw)

    # Override dev_mode flag if dev=True was passed explicitly
    if dev:
        merged["dev_mode"] = True

    # Validate
    config = ExperimentConfig(**merged)

    return config, merged


def write_config_snapshot(config: ExperimentConfig, merged: dict) -> None:
    """Write the resolved config snapshot into output_dir. Call this last —
    after any output_dir wipe (e.g. --force) — so the snapshot survives."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config.yaml"
    with snapshot_path.open("w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=True)


def load_config(experiment_yaml_path: str, dev: bool = False) -> ExperimentConfig:
    """Convenience wrapper: build the config and immediately write its snapshot.
    Callers that need to wipe output_dir first (e.g. --force) should use
    build_config() + write_config_snapshot() instead, in that order."""
    config, merged = build_config(experiment_yaml_path, dev=dev)
    write_config_snapshot(config, merged)
    return config
