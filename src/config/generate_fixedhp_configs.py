from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.config.generator import (
    TASKS,
    MODELS,
    SOURCES,
    CONDITIONS_NON_BASELINE,
    REAL_N,
    SYNTHETIC_N,
    REAL_N_NON_BASELINE,
    _write_yaml,
)

FIXEDHP_SUFFIX = "_fixedhp"


def generate_fixedhp_configs(
    modal_configs_path: str,
    output_dir: str = "configs",
    outputs_root: str = "outputs/fixed_hp",
) -> None:
    """
    Generate the Fixed-HP secondary experiment's 702 config YAMLs
    (experiment_design.md section 7.5): every primary-batch config (9
    baselines + 693 non-baseline), rerun with one shared/modal HP config per
    (task, model) instead of each condition's own tuned HP search. Mirrors
    generator.py's grid exactly (same real_n/synthetic_n/split files per
    condition) -- only experiment_id, output_dir, and the added fixed_hp
    block differ, so the fixed-HP vs per-condition-tuned contrast holds
    every other variable constant.

    modal_configs_path: JSON file mapping "{task}_{model}" -> a dict with
    "hp_config" (required) and, for fmd, "nms_threshold"/"nms_radius"
    (required) -- the exact shape compute_modal_hp_config() in
    src/analysis/aggregator.py returns per (task, model), computed from the
    now-complete primary batch's results.
    """
    with open(modal_configs_path) as f:
        modal_configs: dict[str, dict] = json.load(f)

    base = Path(output_dir)
    generated = 0
    skipped = 0
    missing_modal = []

    def _fixed_hp_block(task: str, model: str) -> dict | None:
        key = f"{task}_{model}"
        modal = modal_configs.get(key)
        if modal is None or "hp_config" not in modal:
            missing_modal.append(key)
            return None
        block = {"hp_config": modal["hp_config"]}
        if task == "fmd":
            if "nms_threshold" not in modal or "nms_radius" not in modal:
                missing_modal.append(key)
                return None
            block["nms_threshold"] = modal["nms_threshold"]
            block["nms_radius"] = modal["nms_radius"]
        return block

    # Baseline: task × model, condition=V0
    for task in TASKS:
        for model in MODELS:
            fixed_hp = _fixed_hp_block(task, model)
            if fixed_hp is None:
                continue

            base_id = f"{task}_{model}_V0"
            exp_id = f"{base_id}{FIXEDHP_SUFFIX}"
            out_path = base / task / "experiments" / f"{exp_id}.yaml"

            if out_path.exists():
                skipped += 1
                continue

            data = {
                "experiment_id": exp_id,
                "task": task,
                "model_id": model,
                "condition": "V0",
                "synthetic_source": None,
                "real_n": REAL_N[task],
                "synthetic_n": 0,
                "split_file_real": f"splits/{task}/real_split.json",
                "split_file_synthetic": None,
                "output_dir": f"{outputs_root}/{task}/{exp_id}",
                "fixed_hp": fixed_hp,
            }
            if task == "fc" and model == "M3":
                data["batch_size"] = 64
                data["max_steps"] = 5000
            _write_yaml(out_path, data)
            generated += 1

    # Non-baseline: condition × source × task × model
    for task in TASKS:
        for model in MODELS:
            fixed_hp = _fixed_hp_block(task, model)
            if fixed_hp is None:
                continue

            for cond in CONDITIONS_NON_BASELINE:
                for src in SOURCES:
                    base_id = f"{task}_{model}_{cond}_{src}"
                    exp_id = f"{base_id}{FIXEDHP_SUFFIX}"
                    out_path = base / task / "experiments" / f"{exp_id}.yaml"

                    if out_path.exists():
                        skipped += 1
                        continue

                    data = {
                        "experiment_id": exp_id,
                        "task": task,
                        "model_id": model,
                        "condition": cond,
                        "synthetic_source": src,
                        "real_n": REAL_N_NON_BASELINE[task][cond],
                        "synthetic_n": SYNTHETIC_N[task][cond],
                        "split_file_real": f"splits/{task}/real_split.json",
                        "split_file_synthetic": f"splits/{task}/synthetic/{src}_{cond}.json",
                        "output_dir": f"{outputs_root}/{task}/{exp_id}",
                        "fixed_hp": fixed_hp,
                    }
                    if task == "fc" and model == "M3":
                        data["batch_size"] = 64
                        data["max_steps"] = 5000
                    _write_yaml(out_path, data)
                    generated += 1

    print(f"Generated {generated} / skipped {skipped} / total expected 702")
    if missing_modal:
        print(
            f"WARNING: no modal config for {sorted(set(missing_modal))} -- "
            f"those (task,model) groups were skipped entirely. Check "
            f"{modal_configs_path}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Fixed-HP secondary experiment config YAMLs."
    )
    parser.add_argument(
        "--modal-configs", required=True,
        help="Path to a JSON file mapping '{task}_{model}' to a modal HP "
             "config dict (from compute_modal_hp_config()).",
    )
    parser.add_argument("--out", default="configs", help="Config root (default: configs/)")
    parser.add_argument(
        "--outputs-root", default="outputs/fixed_hp",
        help="Output root for these experiments (default: outputs/fixed_hp)",
    )
    args = parser.parse_args()
    generate_fixedhp_configs(args.modal_configs, args.out, args.outputs_root)


if __name__ == "__main__":
    main()
