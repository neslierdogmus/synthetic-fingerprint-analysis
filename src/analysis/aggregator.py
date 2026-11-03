from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.status_report import get_complete_experiment_ids, parse_experiment_id
from src.config.schema import ExperimentConfig
from src.evaluation.evaluator import Evaluator
from src.experiment.dataset_builder import (
    build_real_dataset,
    build_transform,
    load_split,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config_from_snapshot(config_path: Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a saved config.yaml snapshot."""
    with config_path.open() as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)


# ---------------------------------------------------------------------------
# Per-experiment evaluation
# ---------------------------------------------------------------------------

def _primary_metric(metrics: dict, task: str, fmd_dist: int | None) -> float:
    if task == "fc":
        return metrics["top1_accuracy"]
    if task == "fmd":
        return metrics["distance_results"][fmd_dist]["global"]["f1"]
    if task == "foe":
        return metrics["mean_angular_error_global_deg"]
    return 0.0


def _evaluate_experiment(
    exp_dir: Path,
    config: ExperimentConfig,
    status_data: dict,
    data_root: str,
) -> dict | None:
    """
    Run evaluate_full() on each seed's predictions.
    Returns one aggregated row dict, or None on failure.
    """
    final_eval_dir = exp_dir / "final_eval"
    seed_files = sorted(final_eval_dir.glob("seed_*.npy"))
    if not seed_files:
        return None

    # Build test dataset
    split = load_split(config.split_file_real)
    test_ids = split["test_ids"]
    test_transform = build_transform(config, training=False)
    test_ds = build_real_dataset(config, test_ids, test_transform, data_root)

    # FMD nms_params from saved best_config
    nms_params = None
    if config.task == "fmd":
        best = status_data.get("hp_phase2", {}).get("best_config") or {}
        t = best.get("nms_threshold")
        r = best.get("nms_radius")
        if t is not None and r is not None:
            nms_params = (float(t), int(r))

    evaluator = Evaluator(config)
    seed_metrics: list[dict] = []
    for sf in seed_files:
        m = evaluator.evaluate_full(sf, test_ds, nms_params)
        seed_metrics.append(m)

    if not seed_metrics:
        return None

    fmd_dist = config.fmd_matching_distance
    primary_values = [
        _primary_metric(m, config.task, fmd_dist) for m in seed_metrics
    ]

    # config.synthetic_n is the condition's theoretical target. For
    # capacity-constrained sources (e.g. S01 at V2/V3) the split file's
    # actual_count is the true number of synthetic samples used and can be
    # lower — report that instead, so the results table doesn't silently
    # overstate N for a shortfall source.
    synthetic_n_actual = config.synthetic_n
    if config.split_file_synthetic:
        synth_split = load_split(config.split_file_synthetic)
        synthetic_n_actual = synth_split.get("actual_count", config.synthetic_n)

    row: dict = {
        "experiment_id": config.experiment_id,
        "task": config.task,
        "model": config.model_id,
        "condition": config.condition,
        "source": config.synthetic_source or "-",
        "real_n": config.real_n,
        "synthetic_n": synthetic_n_actual,
        "n_seeds": len(seed_metrics),
        "mean_metric": float(np.mean(primary_values)),
        "std_metric": float(np.std(primary_values)) if len(primary_values) > 1 else 0.0,
    }

    # Per-seed primary metrics (pad to 3 with NaN)
    for i in range(3):
        row[f"seed_{i}_metric"] = (
            float(primary_values[i]) if i < len(primary_values) else float("nan")
        )

    # Task-specific JSON columns
    if config.task == "fc":
        row["per_class_accuracy_json"] = json.dumps(
            [m["per_class_accuracy"] for m in seed_metrics]
        )
        row["confusion_matrix_json"] = json.dumps(
            [m["confusion_matrix"] for m in seed_metrics]
        )
    elif config.task == "fmd":
        # Convert int keys to str for JSON serialisation
        def _str_keys(d: dict) -> dict:
            return {str(k): v for k, v in d.items()}

        row["precision_recall_json"] = json.dumps(
            [_str_keys(m["distance_results"]) for m in seed_metrics]
        )
        row["per_quality_json"] = json.dumps(
            [m["per_quality"] for m in seed_metrics]
        )
    elif config.task == "foe":
        row["per_quality_json"] = json.dumps(
            [m.get("per_quality") for m in seed_metrics]
        )

    return row


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------

def build_master_results(
    outputs_root: str = "outputs",
    data_root: str = "/data",
    results_path: str = "results/master_results.csv",
    force: bool = False,
) -> pd.DataFrame:
    """
    Scans all experiments with status=final_eval_complete.
    Runs evaluate_full() for each seed, aggregates metrics, saves CSV.
    Skips already-processed experiments unless force=True.
    """
    results_file = Path(results_path)

    if results_file.exists() and not force:
        existing_df = pd.read_csv(results_file)
        already_done: set[str] = set(existing_df["experiment_id"].tolist())
    else:
        existing_df = pd.DataFrame()
        already_done = set()

    root = Path(outputs_root)
    new_rows: list[dict] = []
    failed_ids: list[str] = []

    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        if task not in ("fc", "fmd", "foe"):
            continue

        for exp_dir in sorted(task_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            experiment_id = exp_dir.name

            status_path = exp_dir / "status.json"
            if not status_path.exists():
                continue
            try:
                with status_path.open() as f:
                    status_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if status_data.get("status") != "final_eval_complete":
                continue

            if experiment_id in already_done:
                print(f"  Skipping {experiment_id} (already in results)")
                continue

            config_path = exp_dir / "config.yaml"
            if not config_path.exists():
                print(f"  ERROR: no config.yaml for {experiment_id}, cannot aggregate")
                failed_ids.append(experiment_id)
                continue

            try:
                config = _load_config_from_snapshot(config_path)
            except Exception as exc:
                print(f"  ERROR: failed to load config for {experiment_id}: {exc}")
                failed_ids.append(experiment_id)
                continue

            print(f"  Processing {experiment_id}...")
            try:
                row = _evaluate_experiment(exp_dir, config, status_data, data_root)
                if row is not None:
                    new_rows.append(row)
                    print(f"    mean_metric={row['mean_metric']:.4f}")
                else:
                    print(f"  ERROR: no seed predictions found for {experiment_id}")
                    failed_ids.append(experiment_id)
            except Exception as exc:
                print(f"  ERROR: evaluation failed for {experiment_id}: {exc}")
                traceback.print_exc()
                failed_ids.append(experiment_id)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if not existing_df.empty
            else new_df
        )
    elif not existing_df.empty:
        combined = existing_df
    else:
        print("No completed experiments found.")
        return pd.DataFrame()

    results_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(results_file, index=False)
    print(f"Saved {len(combined)} rows to {results_file}")

    # Standing assert: aggregator row count must equal status_report's
    # final_eval_complete count. A mismatch means some experiment was
    # silently dropped (missing/corrupt config.yaml, failed evaluation,
    # missing predictions) instead of being reflected in the output.
    complete_ids = get_complete_experiment_ids(outputs_root)
    aggregated_ids = set(combined["experiment_id"].tolist()) if not combined.empty else set()
    missing = complete_ids - aggregated_ids
    if missing:
        raise RuntimeError(
            f"Aggregator row count ({len(aggregated_ids)}) does not match "
            f"status_report's final_eval_complete count ({len(complete_ids)}). "
            f"Missing from results: {sorted(missing)}. "
            f"Failed this run: {sorted(failed_ids) if failed_ids else 'none (stale from a prior run)'}"
        )

    return combined


# ---------------------------------------------------------------------------
# Modal HP config helper
# ---------------------------------------------------------------------------

def compute_modal_hp_config(
    outputs_root: str = "outputs",
    task: str = "fc",
    model: str = "M1",
) -> dict:
    """
    Finds the most frequently selected HP config (and, for fmd, NMS
    threshold/radius) across all completed experiments for the given
    (task, model) -- the basis for the Fixed-HP secondary experiment
    (experiment_design.md section 7.5). Each experiment's own final chosen
    config (hp_phase2.best_config -- post Phase 1 + Phase 2, what it actually
    used for final eval) is what gets counted, falling back to
    hp_phase1.top3_configs[0] only if phase 2 data is missing. For fmd, the
    key is the full (hp_config, nms_threshold, nms_radius) triplet, not just
    hp_config alone, since NMS params are as much a per-experiment tuning
    choice as the HP index and the Fixed-HP experiment needs to hold all
    three fixed together.
    """
    key_counts: Counter = Counter()
    root = Path(outputs_root)
    task_dir = root / task

    if not task_dir.exists():
        return {}

    def _key(entry: dict):
        hp_idx = entry.get("hp_config")
        if hp_idx is None:
            return None
        if task == "fmd":
            t, r = entry.get("nms_threshold"), entry.get("nms_radius")
            if t is None or r is None:
                return None
            return (hp_idx, t, r)
        return (hp_idx,)

    for exp_dir in sorted(task_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        experiment_id = exp_dir.name
        parts = experiment_id.split("_")
        if len(parts) < 2 or parts[0] != task or parts[1] != model:
            continue

        status_path = exp_dir / "status.json"
        if not status_path.exists():
            continue
        try:
            with status_path.open() as f:
                status_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Prefer best_config from hp_phase2
        best = status_data.get("hp_phase2", {}).get("best_config")
        if best and isinstance(best, dict):
            key = _key(best)
            if key is not None:
                key_counts[key] += 1
                continue

        # Fall back to top3_configs[0] from hp_phase1
        top3 = status_data.get("hp_phase1", {}).get("top3_configs", [])
        if top3:
            entry = top3[0]
            entry = entry if isinstance(entry, dict) else {"hp_config": int(entry)}
            key = _key(entry)
            if key is not None:
                key_counts[key] += 1

    if not key_counts:
        return {}

    modal_key, modal_count = key_counts.most_common(1)[0]
    result = {
        "hp_config": modal_key[0],
        "count": modal_count,
        "total": sum(key_counts.values()),
    }
    if task == "fmd":
        result["nms_threshold"] = modal_key[1]
        result["nms_radius"] = modal_key[2]
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build master results CSV from completed experiments."
    )
    parser.add_argument("--outputs", default="outputs", metavar="PATH")
    parser.add_argument("--data-root", default="/data", metavar="PATH")
    parser.add_argument("--results", default="results/master_results.csv",
                        metavar="PATH")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess all experiments, even if already in CSV.")
    parser.add_argument("--modal", action="store_true",
                        help="Print modal HP config instead of building results.")
    parser.add_argument("--task", default="fc")
    parser.add_argument("--model", default="M1")
    args = parser.parse_args()

    if args.modal:
        result = compute_modal_hp_config(args.outputs, args.task, args.model)
        print(f"Modal HP config for {args.task}/{args.model}: {result}")
    else:
        build_master_results(
            outputs_root=args.outputs,
            data_root=args.data_root,
            results_path=args.results,
            force=args.force,
        )
