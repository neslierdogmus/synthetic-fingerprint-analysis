from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import torch

from src.config.loader import build_config, write_config_snapshot
from src.experiment.runner import ExperimentRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single fingerprint analysis experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run
  python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml

  # Dev/smoke test
  python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml --dev

  # Resume (default — skips completed stages automatically)
  python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml

  # Force rerun from scratch
  python run_experiment.py --config configs/fc/experiments/fc_M2_V1_S02.yaml --force
""",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable dev mode (fewer steps/folds/seeds/hp_configs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing outputs and rerun from scratch.",
    )
    parser.add_argument(
        "--data-root",
        default="/data",
        metavar="PATH",
        help="Path to dataset root directory (default: /data).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        metavar="DEVICE",
        help='CUDA device string (default: "cuda").',
    )
    return parser.parse_args()


def _load_split_counts(split_path: str | None) -> int:
    """Return the real_n or synthetic_n count from a split JSON, or 0."""
    if not split_path:
        return 0
    p = Path(split_path)
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text())
        return data.get("real_n", 0) or data.get("synthetic_n", 0)
    except Exception:
        return 0


def _print_summary(config, device: str, dev_mode: bool) -> None:
    real_n = _load_split_counts(config.split_file_real)
    synth_n = _load_split_counts(config.split_file_synthetic)

    lines = [
        "================================",
        f"Experiment: {config.experiment_id}",
        f"Task:       {config.task}",
        f"Model:      {config.model_id}",
        f"Condition:  {config.condition}",
        f"Source:     {config.synthetic_source or 'N/A'}",
        f"Real N:     {real_n}",
        f"Synthetic N:{synth_n}",
        f"Device:     {device}",
        f"Dev mode:   {dev_mode}",
        f"Output dir: {config.output_dir}",
        "================================",
    ]
    print("\n".join(lines))


def main() -> None:
    args = _parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Resolve config (no disk writes yet — output_dir may still need wiping below)
    try:
        config, merged_config = build_config(str(config_path), dev=args.dev)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    # Warn if CUDA requested but not available
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA is not available; running on CPU may be slow.",
            file=sys.stderr,
        )

    # Handle --force: confirm and delete existing outputs
    output_dir = Path(config.output_dir)
    if args.force and output_dir.exists():
        answer = input(
            f"This will delete all outputs for {config.experiment_id}. Continue? [y/N] "
        ).strip()
        if answer not in ("y", "Y"):
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(output_dir)

    # Write the config snapshot now that output_dir has been wiped (if --force),
    # so it isn't deleted along with the rest of the directory.
    write_config_snapshot(config, merged_config)

    # Print summary
    _print_summary(config, args.device, args.dev)

    # Run experiment
    try:
        runner = ExperimentRunner(config, device=args.device, data_root=args.data_root)
        runner.run()
        print(f"\nExperiment {config.experiment_id} complete.")
    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Progress saved — rerun without --force to resume.",
            file=sys.stderr,
        )
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
