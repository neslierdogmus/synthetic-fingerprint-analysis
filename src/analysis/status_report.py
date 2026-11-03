from __future__ import annotations

import json
import sys
from pathlib import Path


def parse_experiment_id(experiment_id: str) -> dict:
    """
    Parse experiment_id into components.
    Baseline:     {task}_{model}_V0       → source="-"
    Non-baseline: {task}_{model}_{cond}_{src}

    A trailing "_fixedhp" (the Fixed-HP secondary experiment's naming
    convention, experiment_design.md section 7.5) is stripped first so
    both forms parse the same as their primary-batch counterpart --
    otherwise a baseline like "fc_M1_V0_fixedhp" would wrongly read
    source="fixedhp" instead of "-" (4 parts instead of 3).
    """
    if experiment_id.endswith("_fixedhp"):
        experiment_id = experiment_id[: -len("_fixedhp")]
    parts = experiment_id.split("_")
    task = parts[0]
    model = parts[1]
    condition = parts[2]
    source = parts[3] if len(parts) > 3 else "-"
    return {"task": task, "model": model, "condition": condition, "source": source}


def _status_category(status_str: str) -> str:
    if status_str == "final_eval_complete":
        return "complete"
    if status_str in ("hp_phase2_complete", "hp_phase1_complete"):
        return "running"
    return "pending"


def get_complete_experiment_ids(outputs_root: str = "outputs") -> set[str]:
    """Scan outputs/{task}/{experiment_id}/status.json and return the set of
    experiment_ids whose status is final_eval_complete. Shared by the
    aggregator so its row count can be asserted against this count."""
    root = Path(outputs_root)
    complete_ids: set[str] = set()

    if not root.exists():
        return complete_ids

    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or task_dir.name not in ("fc", "fmd", "foe"):
            continue
        for exp_dir in sorted(task_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            status_path = exp_dir / "status.json"
            if not status_path.exists():
                continue
            try:
                with status_path.open() as f:
                    status_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if status_data.get("status") == "final_eval_complete":
                complete_ids.add(exp_dir.name)

    return complete_ids


def print_status_report(outputs_root: str = "outputs") -> None:
    """
    Scans outputs/{task}/{experiment_id}/status.json for all experiments.
    Prints a progress table:

    task  model  condition  source  status                seeds
    fc    M1     V0         -       final_eval_complete   3/3
    ...
    -------------------------------------------------------
    702 total | 487 complete | 12 running | 203 pending
    """
    root = Path(outputs_root)
    rows: list[dict] = []

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

            try:
                parsed = parse_experiment_id(experiment_id)
            except (IndexError, ValueError):
                continue

            status_path = exp_dir / "status.json"
            status_str = "pending"
            n_seeds = 0
            expected_seeds = 3

            if status_path.exists():
                try:
                    with status_path.open() as f:
                        status_data = json.load(f)
                    status_str = status_data.get("status", "pending")
                    n_seeds = len(
                        status_data.get("final_eval", {}).get("completed_seeds", [])
                    )
                except (json.JSONDecodeError, OSError):
                    pass

            config_path = exp_dir / "config.yaml"
            if config_path.exists():
                try:
                    import yaml  # noqa: PLC0415
                    with config_path.open() as f:
                        cfg_data = yaml.safe_load(f)
                    expected_seeds = cfg_data.get("n_seeds", 3)
                except Exception:
                    pass

            rows.append({
                **parsed,
                "status": status_str,
                "category": _status_category(status_str),
                "seeds": f"{n_seeds}/{expected_seeds}",
            })

    if not rows:
        print(f"No experiments found in {outputs_root}/")
        return

    # Column widths
    cw = {"task": 6, "model": 7, "condition": 11, "source": 8,
          "status": 24, "seeds": 6}

    def row_str(r: dict) -> str:
        return (
            f"{r['task']:<{cw['task']}}"
            f"{r['model']:<{cw['model']}}"
            f"{r['condition']:<{cw['condition']}}"
            f"{r['source']:<{cw['source']}}"
            f"{r['status']:<{cw['status']}}"
            f"{r['seeds']:<{cw['seeds']}}"
        )

    header = row_str({
        "task": "task", "model": "model", "condition": "condition",
        "source": "source", "status": "status", "seeds": "seeds",
    })
    sep = "-" * len(header)

    print(header)
    print(sep)
    for r in rows:
        print(row_str(r))
    print(sep)

    n_total = len(rows)
    n_complete = sum(1 for r in rows if r["category"] == "complete")
    n_running  = sum(1 for r in rows if r["category"] == "running")
    n_pending  = sum(1 for r in rows if r["category"] == "pending")
    print(
        f"{n_total} total | {n_complete} complete | "
        f"{n_running} running | {n_pending} pending"
    )


if __name__ == "__main__":
    outputs_root = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    print_status_report(outputs_root)
