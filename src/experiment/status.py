from __future__ import annotations

import json
from pathlib import Path


STATUS_STAGES = [
    "pending",
    "hp_phase1_complete",
    "hp_phase2_complete",
    "final_eval_complete",
]


class ExperimentStatus:
    """Reads and writes status.json for one experiment."""

    def __init__(self, output_dir: Path) -> None:
        self.path = Path(output_dir) / "status.json"
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with self.path.open() as f:
                return json.load(f)
        return {
            "status": "pending",
            "hp_phase1": {
                "status": "pending",
                "completed_folds": [],
                "top3_configs": [],
            },
            "hp_phase2": {
                "status": "pending",
                "completed_folds": [],
                "best_config": None,
            },
            "final_eval": {
                "status": "pending",
                "completed_seeds": [],
            },
        }

    def save(self) -> None:
        """Atomic write: write to temp file then rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(self.data, f, indent=2)
        tmp.rename(self.path)

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def hp_phase1_complete(self) -> bool:
        return self.data["hp_phase1"]["status"] == "complete"

    @property
    def hp_phase2_complete(self) -> bool:
        return self.data["hp_phase2"]["status"] == "complete"

    @property
    def final_eval_complete(self) -> bool:
        return self.data["final_eval"]["status"] == "complete"

    @property
    def completed_phase1_folds(self) -> list[int]:
        return self.data["hp_phase1"]["completed_folds"]

    @property
    def completed_phase2_folds(self) -> list[int]:
        return self.data["hp_phase2"]["completed_folds"]

    @property
    def completed_seeds(self) -> list[int]:
        return self.data["final_eval"]["completed_seeds"]

    @property
    def top3_configs(self) -> list[dict]:
        return self.data["hp_phase1"]["top3_configs"]

    @property
    def best_config(self) -> dict | None:
        return self.data["hp_phase2"]["best_config"]
