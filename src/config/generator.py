from __future__ import annotations

import argparse
from pathlib import Path

import yaml

TASKS = ["fc", "fmd", "foe"]
MODELS = ["M1", "M2", "M3"]
SOURCES = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11"]
CONDITIONS_NON_BASELINE = ["V1", "V2", "V3", "R1", "R2", "R3", "R4"]

REAL_N = {
    "fc": 3200,
    "fmd": 2560,
    "foe": 48,
}

SYNTHETIC_N: dict[str, dict[str, int]] = {
    "fc":  {"V1": 3200,  "V2": 9600,  "V3": 22400, "R1": 800,  "R2": 1600, "R3": 2400, "R4": 3200},
    "fmd": {"V1": 2560,  "V2": 7680,  "V3": 17920, "R1": 640,  "R2": 1280, "R3": 1920, "R4": 2560},
    "foe": {"V1": 48,    "V2": 144,   "V3": 336,   "R1": 12,   "R2": 24,   "R3": 36,   "R4": 48},
}

# Real-side sample count per non-baseline condition. Volume conditions (V1-V3)
# keep the full real pool (real is additive with synthetic). Substitution
# conditions (R1-R4) replace an increasing fraction of real with synthetic,
# holding total N fixed — see experiment_design.md §6.2.
REAL_N_NON_BASELINE: dict[str, dict[str, int]] = {
    "fc":  {"V1": 3200, "V2": 3200, "V3": 3200, "R1": 2400, "R2": 1600, "R3": 800, "R4": 0},
    "fmd": {"V1": 2560, "V2": 2560, "V3": 2560, "R1": 1920, "R2": 1280, "R3": 640, "R4": 0},
    "foe": {"V1": 48,   "V2": 48,   "V3": 48,   "R1": 36,   "R2": 24,   "R3": 12,  "R4": 0},
}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def generate_configs(output_dir: str = "configs") -> None:
    base = Path(output_dir)
    generated = 0
    skipped = 0
    total_expected = 702  # 9 baseline + 693 non-baseline

    # Baseline: task × model, condition=V0
    for task in TASKS:
        for model in MODELS:
            exp_id = f"{task}_{model}_V0"
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
                "output_dir": f"outputs/{task}/{exp_id}",
            }
            if task == "fc" and model == "M3":
                data["batch_size"] = 64
                data["max_steps"] = 5000
            _write_yaml(out_path, data)
            generated += 1

    # Non-baseline: condition × source × task × model
    for task in TASKS:
        for model in MODELS:
            for cond in CONDITIONS_NON_BASELINE:
                for src in SOURCES:
                    exp_id = f"{task}_{model}_{cond}_{src}"
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
                        "output_dir": f"outputs/{task}/{exp_id}",
                    }
                    if task == "fc" and model == "M3":
                        data["batch_size"] = 64
                        data["max_steps"] = 5000
                    _write_yaml(out_path, data)
                    generated += 1

    print(
        f"Generated {generated} / skipped {skipped} / total {total_expected}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment config YAMLs.")
    parser.add_argument(
        "--out",
        default="configs",
        help="Root output directory for configs (default: configs/)",
    )
    args = parser.parse_args()
    generate_configs(output_dir=args.out)


if __name__ == "__main__":
    main()
