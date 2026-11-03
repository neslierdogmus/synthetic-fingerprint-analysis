from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn as nn

from src.config.generator import REAL_N as FULL_REAL_N
from src.config.schema import ExperimentConfig
from src.data.combined_dataset import CombinedDataset
from src.data.real.fmd_dataset import FMDRealDataset
from src.evaluation.evaluator import Evaluator
from src.experiment.dataset_builder import (
    build_real_dataset,
    build_synthetic_dataset,
    build_transform,
    load_split,
)
from src.experiment.hp_selector import HPSelector
from src.experiment.status import ExperimentStatus
from src.models.factory import build_model
from src.training.losses.fc_loss import FCLoss
from src.training.losses.fmd_loss import FMDLoss
from src.training.losses.foe_loss import FOELoss
from src.training.trainer import Trainer, _save_json, build_optimizer, build_scheduler


def _finger_key(img_id: str) -> str:
    p = Path(img_id)
    return str(p.parent / p.stem.rsplit("_", 1)[0])


# FMD class weights (background/ridge-ending/bifurcation) are computed once
# from the real V0/R0 training pool (2,560 images) and reused as a fixed
# constant across every FMD condition/source/model -- including R4 (0% real),
# which has no real data of its own to compute weights from. Explicit user
# decision (2026-07-06): do not recompute per-condition/per-experiment.
_FIXED_FMD_WEIGHTS_CACHE = Path("splits/fmd/real_class_weights.json")


def _get_fixed_fmd_class_weights(data_root: str) -> torch.Tensor:
    if _FIXED_FMD_WEIGHTS_CACHE.exists():
        with _FIXED_FMD_WEIGHTS_CACHE.open() as f:
            weights = json.load(f)
        return torch.tensor(weights, dtype=torch.float32)

    real_split = load_split("splits/fmd/real_split.json")
    full_real_ds = FMDRealDataset(
        data_root=f"{data_root}/real/FVC_BERKAY_TOPCU",
        filenames=real_split["train_val_ids"],
        transform=None,
        model_type="scratch",  # unused by get_class_weights()
    )
    weights = full_real_ds.get_class_weights()

    _FIXED_FMD_WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _FIXED_FMD_WEIGHTS_CACHE.open("w") as f:
        json.dump(weights.tolist(), f)

    return weights


# ── HP score persistence (score-then-delete) ────────────────────────────────
#
# HP Phase 1/2 heatmaps ((640, 3, 374, 388) float32 for FMD, ~1.1GB each,
# 108 per experiment) dominate storage -- ~27.6TB across the 693-experiment
# batch. Nothing downstream reads them once scored (verified: aggregator.py,
# tables.py, plots.py, status_report.py only ever touch status.json fields
# and final_eval/seed_*.npy). But resume previously keyed directly on the
# heatmap's existence (score_fold() re-run against the saved .npy on every
# resume) -- deleting it outright would make a resumed run see a missing
# file and silently retrain. So the score itself is now persisted into the
# fold/hp-config's history JSON (already written atomically by
# trainer.py's _save_json), and resume reads the score from there instead.

def _serialize_score(score):
    """JSON can't have tuple dict keys (FMD phase1's (threshold, radius) ->
    f1 grid) -- flatten to "threshold|radius" string keys. Scalars (FC/FOE,
    and FMD phase2 which scores at one fixed threshold/radius) pass through
    unchanged."""
    if isinstance(score, dict):
        return {f"{t}|{r}": f1 for (t, r), f1 in score.items()}
    return score


def _deserialize_score(serialized):
    if isinstance(serialized, dict):
        result = {}
        for k, f1 in serialized.items():
            t_str, r_str = k.split("|")
            result[(float(t_str), int(r_str))] = f1
        return result
    return serialized


def _write_score_and_maybe_delete_heatmap(
    hist_path: Path, pred_path: Path, history: dict, score, task: str
) -> None:
    """
    Persist `score` into the fold/hp-config's history JSON atomically,
    verify the write landed on disk, and only then (FMD only -- the
    storage-driving task; FC/FOE heatmaps are tiny so kept regardless)
    delete the raw heatmap .npy.

    Ordering is correctness-critical: a crash before this function is
    called leaves the heatmap intact (re-scorable without retraining, see
    the resume logic in _run_hp_phase1/_run_hp_phase2). A crash after the
    atomic rename below leaves the score durably on disk (resume reads it
    directly, never touches the heatmap again). The heatmap is never
    deleted until the score write is independently re-read back from disk
    and confirmed present -- there is no window where a crash could leave
    neither the heatmap nor the score recoverable.
    """
    history = dict(history)
    history["score"] = _serialize_score(score)
    _save_json(history, hist_path)

    with hist_path.open() as f:
        verify = json.load(f)
    if "score" not in verify:
        raise RuntimeError(
            f"Score write verification failed for {hist_path}: "
            "'score' key missing after write+reload -- refusing to delete "
            f"{pred_path}."
        )

    if task == "fmd":
        pred_path.unlink()


def _try_load_score_from_history(hist_path: Path):
    """Return the deserialized score if hist_path exists and already has
    one persisted (fast resume path, no heatmap needed), else None."""
    if not hist_path.exists():
        return None
    with hist_path.open() as f:
        history = json.load(f)
    if "score" not in history:
        return None
    return _deserialize_score(history["score"])


def _subsample_real_ids(ids: list[str], target_n: int, seed: int, task: str) -> list[str]:
    """
    Deterministically subsample real ids down to target_n. Used to realize
    the real-data reduction for substitution conditions (R1-R4), which hold
    total N fixed by replacing an increasing fraction of real data with
    synthetic (see experiment_design.md SS6.2).

    FC/FMD: grouped by finger/identity so no identity's images are split
    across the kept/dropped portion (mirrors the existing phase-1 subset
    logic). FOE: stratified by Bad/Good quality so the 5:1 ratio baked into
    real_split.json is preserved rather than skewed by a random cut.
    """
    if target_n >= len(ids):
        return list(ids)
    if target_n <= 0:
        return []

    if task == "foe":
        by_quality: dict[str, list[str]] = {}
        for x in ids:
            by_quality.setdefault(x.split("/")[0], []).append(x)
        rng = random.Random(seed)
        selected: list[str] = []
        for quality, group in sorted(by_quality.items()):
            shuffled = sorted(group)
            rng.shuffle(shuffled)
            n_q = round(target_n * len(group) / len(ids))
            selected.extend(shuffled[:n_q])
        return selected

    by_finger: dict[str, list[str]] = {}
    for x in ids:
        by_finger.setdefault(_finger_key(x), []).append(x)
    fingers = sorted(by_finger.keys())
    random.Random(seed).shuffle(fingers)
    selected = []
    for f in fingers:
        if len(selected) >= target_n:
            break
        selected.extend(by_finger[f])
    return selected


class ExperimentRunner:
    """
    Orchestrates the full experiment lifecycle:
      1. HP Phase 1: subset CV grid search (or full CV for FOE)
      2. HP Phase 2: full CV on top-3 configs (FC and FMD only)
      3. Final evaluation: train on full pool, evaluate on test set

    Resumable: checks status.json at each stage and skips completed work.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        device: str = "cuda",
        data_root: str = "/data",
    ) -> None:
        self.config = config
        self.device = device
        self.data_root = data_root
        self.output_dir = Path(config.output_dir)
        self.status = ExperimentStatus(self.output_dir)
        self.evaluator = Evaluator(config)

    def _reduce_real_ids(self, ids: list[str], seed: int) -> list[str]:
        """
        Apply the substitution-condition real-data reduction to a pool of real
        ids. No-op for V0-V3/R0 (config.real_n == the full task pool), reduces
        proportionally for R1-R4. See _subsample_real_ids for grouping rules.
        """
        fraction = self.config.real_n / FULL_REAL_N[self.config.task]
        if fraction >= 1.0:
            return ids
        target_n = round(len(ids) * fraction)
        return _subsample_real_ids(ids, target_n, seed=seed, task=self.config.task)

    def run(self) -> None:
        """Main entry point. Runs all stages with resume support."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.fixed_hp is not None:
            self._apply_fixed_hp()
        else:
            if not self.status.hp_phase1_complete:
                self._run_hp_phase1()

            if self.config.task != "foe" and not self.status.hp_phase2_complete:
                self._run_hp_phase2()

        if not self.status.final_eval_complete:
            self._run_final_eval()

    def _apply_fixed_hp(self) -> None:
        """
        Fixed-HP secondary experiment (experiment_design.md section 7.5):
        manufacture the same status.json contract HP Phase 1 + Phase 2 would
        produce, using the externally-provided config.fixed_hp instead of
        searching hp_grid -- so _run_final_eval needs no changes at all, since
        it only ever reads self.status.best_config regardless of how it got
        there. Idempotent, checked the same way the real phases are, so
        resume behaves identically (e.g. a wall-killed fixed-HP run just
        re-applies this cheaply before proceeding to wherever final eval left
        off).
        """
        entry: dict = {"hp_config": self.config.fixed_hp.hp_config}
        if self.config.task == "fmd":
            entry["nms_threshold"] = self.config.fixed_hp.nms_threshold
            entry["nms_radius"] = self.config.fixed_hp.nms_radius

        if not self.status.hp_phase1_complete:
            self.status.data["hp_phase1"]["top3_configs"] = [entry]
            self.status.data["hp_phase1"]["status"] = "complete"
            self.status.save()

        if not self.status.hp_phase2_complete:
            self.status.data["hp_phase2"]["best_config"] = entry
            self.status.data["hp_phase2"]["status"] = "complete"
            self.status.save()

    # ── HP Phase 1 ────────────────────────────────────────────────────────────

    def _run_hp_phase1(self) -> None:
        """
        Run HP grid search on a subset (FC/FMD) or full CV (FOE).
        Trains all n_hp_configs × n_folds combinations.
        Skips already-completed (fold, hp_idx) pairs.
        After all complete, calls HPSelector to select top-3 (or best for FOE).
        """
        real_split = load_split(self.config.split_file_real)
        synth_split = (
            load_split(self.config.split_file_synthetic)
            if self.config.split_file_synthetic
            else None
        )

        scores: dict = {}

        for fold_idx in range(self.config.n_folds):
            fold = real_split["cv_folds"][fold_idx]

            # Substitution conditions (R1-R4) reduce the real pool first;
            # no-op for V0-V3/R0 where config.real_n is the full task pool.
            train_ids_real = self._reduce_real_ids(fold["train"], seed=1000 + fold_idx)
            # Use subset for FC/FMD phase 1; FOE is small so use full fold
            if self.config.task != "foe":
                fingers = list(dict.fromkeys(_finger_key(x) for x in train_ids_real))
                random.Random(fold_idx).shuffle(fingers)  # deterministic cross-DB shuffle
                n_fingers = max(1, len(fingers) // 4)
                subset_fingers = set(fingers[:n_fingers])
                train_ids_real = [x for x in train_ids_real if _finger_key(x) in subset_fingers]

            val_ids_real = fold["val"]

            train_transform = build_transform(self.config, training=True)
            val_transform   = build_transform(self.config, training=False)

            real_train_ds = build_real_dataset(
                self.config, train_ids_real, train_transform, self.data_root
            )
            real_val_ds = build_real_dataset(
                self.config, val_ids_real, val_transform, self.data_root
            )

            synth_train_ds = None
            if synth_split and synth_split.get("train_ids"):
                synth_train_ids = synth_split["train_ids"]
                if self.config.task != "foe":
                    # ratio = what fraction of the intended (post R1-R4) real
                    # quota survived this phase's additional subset-to-1/4-
                    # of-fingers reduction, so synthetic gets scaled down to
                    # match -- keeping the subset's real:synthetic mix
                    # proportional to the full experiment's. R4 sets
                    # config.real_n = 0 by design (0% real, entirely
                    # synthetic); len(train_ids_real) is correctly 0 too, so
                    # the ratio is a real 0/0, not "no real data survived
                    # subsetting" -- must be treated as 1.0 (no reduction,
                    # full synthetic_n) rather than 0, or R4 collapses to a
                    # single synthetic training example for the entire
                    # phase-1 HP search.
                    if self.config.real_n > 0:
                        ratio = len(train_ids_real) / self.config.real_n
                    else:
                        ratio = 1.0
                    n_synth = max(1, int(self.config.synthetic_n * ratio))
                    synth_train_ids = synth_train_ids[:n_synth]
                synth_train_ds = build_synthetic_dataset(
                    self.config, synth_train_ids, train_transform, self.data_root
                )

            train_ds = (
                CombinedDataset(real_train_ds, synth_train_ds)
                if synth_train_ds
                else real_train_ds
            )

            for hp_idx, hp_config in enumerate(self.config.hp_grid):
                pred_path = (
                    self.output_dir / "hp_phase1"
                    / f"fold_{fold_idx}_hp_{hp_idx}.npy"
                )
                hist_path = pred_path.with_suffix(".json")

                print(f"[Phase 1] fold={fold_idx} hp={hp_idx}", flush=True)

                resumed_score = _try_load_score_from_history(hist_path)
                if resumed_score is not None:
                    # Fastest resume path: score already durable, no heatmap needed.
                    self._accumulate_score(scores, fold_idx, hp_idx, resumed_score)
                    print(f"  fold={fold_idx} hp={hp_idx} score={resumed_score} (resumed)", flush=True)
                    continue

                if pred_path.exists() and pred_path.stat().st_size > 0 and hist_path.exists():
                    # Heatmap + history survived (e.g. crash between training
                    # and the score-write step below) but score wasn't
                    # persisted yet -- re-score cheaply, no retraining.
                    score = self.evaluator.score_fold(pred_path, real_val_ds)
                    with hist_path.open() as f:
                        history = json.load(f)
                    _write_score_and_maybe_delete_heatmap(
                        hist_path, pred_path, history, score, self.config.task
                    )
                    self._accumulate_score(scores, fold_idx, hp_idx, score)
                    print(f"  fold={fold_idx} hp={hp_idx} score={score} (rescored from heatmap)", flush=True)
                    continue

                model = build_model(self.config).to(self.device)
                optimizer = build_optimizer(model, hp_config)
                scheduler = build_scheduler(
                    optimizer, hp_config, self.config.max_steps
                )
                loss_fn = self._build_loss(real_train_ds)
                trainer = Trainer(
                    model, optimizer, loss_fn, self.device, scheduler
                )

                pred_path.parent.mkdir(parents=True, exist_ok=True)
                history = trainer.train_one_fold(
                    train_ds, real_val_ds, self.config, hp_config,
                    pred_path, skip_if_exists=False,
                )

                score = self.evaluator.score_fold(pred_path, real_val_ds)
                _write_score_and_maybe_delete_heatmap(
                    hist_path, pred_path, history, score, self.config.task
                )
                self._accumulate_score(scores, fold_idx, hp_idx, score)
                print(f"  fold={fold_idx} hp={hp_idx} score={score}", flush=True)
                del model, optimizer, scheduler, trainer
                torch.cuda.empty_cache()

            if fold_idx not in self.status.completed_phase1_folds:
                self.status.data["hp_phase1"]["completed_folds"].append(fold_idx)
                self.status.save()

        self._select_top_configs(scores)
        self.status.data["hp_phase1"]["status"] = "complete"
        self.status.save()
        print(f"HP Phase 1 complete. Top configs: {self.status.top3_configs}")

    def _accumulate_score(
        self, scores: dict, fold_idx: int, hp_idx: int, score
    ) -> None:
        """Accumulate scores. Handles FMD (dict of 49 scalars) and others."""
        if self.config.task == "fmd" and isinstance(score, dict):
            for (t, r), f1 in score.items():
                scores[(fold_idx, hp_idx, t, r)] = f1
        else:
            scores[(fold_idx, hp_idx)] = score

    def _select_top_configs(self, scores: dict) -> None:
        """Run HPSelector and save results to status."""
        if self.config.task == "fmd":
            top3 = HPSelector.rank_fmd(scores, top_k=3)
            self.status.data["hp_phase1"]["top3_configs"] = top3
        elif self.config.task == "foe":
            best_idx = HPSelector.select_best_fc_foe(
                scores, len(self.config.hp_grid)
            )
            best = self.config.hp_grid[best_idx]
            self.status.data["hp_phase2"]["best_config"] = {
                "hp_config": best_idx,
                "lr": best.lr,
                "weight_decay": best.weight_decay,
                "cosine_annealing": best.cosine_annealing,
            }
            self.status.data["hp_phase2"]["status"] = "complete"
            self.status.data["hp_phase1"]["top3_configs"] = [best_idx]
        else:
            # FC
            top3_idxs = HPSelector.rank_fc_foe(
                scores, len(self.config.hp_grid), top_k=3
            )
            top3 = [{"hp_config": idx} for idx in top3_idxs]
            self.status.data["hp_phase1"]["top3_configs"] = top3

    # ── HP Phase 2 (FC and FMD only) ──────────────────────────────────────────

    def _run_hp_phase2(self) -> None:
        """
        Run full CV on top-3 configs. Selects single best config/triplet.
        """
        real_split = load_split(self.config.split_file_real)
        synth_split = (
            load_split(self.config.split_file_synthetic)
            if self.config.split_file_synthetic
            else None
        )

        top3 = self.status.top3_configs
        scores: dict = {}

        for fold_idx in range(self.config.n_folds):
            fold = real_split["cv_folds"][fold_idx]
            # Substitution conditions (R1-R4) reduce the real pool first;
            # no-op for V0-V3/R0 where config.real_n is the full task pool.
            train_ids_real = self._reduce_real_ids(fold["train"], seed=2000 + fold_idx)
            val_ids_real   = fold["val"]

            train_transform = build_transform(self.config, training=True)
            val_transform   = build_transform(self.config, training=False)

            real_train_ds = build_real_dataset(
                self.config, train_ids_real, train_transform, self.data_root
            )
            real_val_ds = build_real_dataset(
                self.config, val_ids_real, val_transform, self.data_root
            )

            synth_train_ds = None
            if synth_split and synth_split.get("train_ids"):
                synth_train_ds = build_synthetic_dataset(
                    self.config, synth_split["train_ids"],
                    train_transform, self.data_root
                )

            train_ds = (
                CombinedDataset(real_train_ds, synth_train_ds)
                if synth_train_ds
                else real_train_ds
            )

            for entry in top3:
                hp_idx = entry["hp_config"]
                hp_config = self.config.hp_grid[hp_idx]

                pred_path = (
                    self.output_dir / "hp_phase2"
                    / f"fold_{fold_idx}_hp_{hp_idx}.npy"
                )
                hist_path = pred_path.with_suffix(".json")

                nms_params = None
                if self.config.task == "fmd":
                    nms_params = (entry["nms_threshold"], entry["nms_radius"])

                print(f"[Phase 2] fold={fold_idx} hp={hp_idx}", flush=True)

                resumed_score = _try_load_score_from_history(hist_path)
                if resumed_score is not None:
                    self._accumulate_phase2_score(scores, fold_idx, entry, resumed_score)
                    print(f"  fold={fold_idx} hp={hp_idx} score={resumed_score} (resumed)", flush=True)
                    continue

                if pred_path.exists() and pred_path.stat().st_size > 0 and hist_path.exists():
                    score = self.evaluator.score_fold(
                        pred_path, real_val_ds, nms_params
                    )
                    with hist_path.open() as f:
                        history = json.load(f)
                    _write_score_and_maybe_delete_heatmap(
                        hist_path, pred_path, history, score, self.config.task
                    )
                    self._accumulate_phase2_score(scores, fold_idx, entry, score)
                    print(f"  fold={fold_idx} hp={hp_idx} score={score} (rescored from heatmap)", flush=True)
                    continue

                model = build_model(self.config).to(self.device)
                optimizer = build_optimizer(model, hp_config)
                scheduler = build_scheduler(
                    optimizer, hp_config, self.config.max_steps
                )
                loss_fn = self._build_loss(real_train_ds)
                trainer = Trainer(
                    model, optimizer, loss_fn, self.device, scheduler
                )

                pred_path.parent.mkdir(parents=True, exist_ok=True)
                history = trainer.train_one_fold(
                    train_ds, real_val_ds, self.config, hp_config,
                    pred_path, skip_if_exists=False,
                )

                score = self.evaluator.score_fold(
                    pred_path, real_val_ds, nms_params
                )
                _write_score_and_maybe_delete_heatmap(
                    hist_path, pred_path, history, score, self.config.task
                )
                self._accumulate_phase2_score(scores, fold_idx, entry, score)
                print(f"  fold={fold_idx} hp={hp_idx} score={score}", flush=True)
                del model, optimizer, scheduler, trainer
                torch.cuda.empty_cache()

            if fold_idx not in self.status.completed_phase2_folds:
                self.status.data["hp_phase2"]["completed_folds"].append(fold_idx)
                self.status.save()

        # Select best
        if self.config.task == "fmd":
            best = HPSelector.select_best_fmd(scores)
        else:
            best_idx = HPSelector.select_best_fc_foe(scores, len(top3))
            # best_idx is an hp_config index; find the matching top3 entry
            best_entry = next(e for e in top3 if e["hp_config"] == best_idx)
            hp_idx = best_entry["hp_config"]
            hp = self.config.hp_grid[hp_idx]
            best = {
                "hp_config": hp_idx,
                "lr": hp.lr,
                "weight_decay": hp.weight_decay,
                "cosine_annealing": hp.cosine_annealing,
            }

        self.status.data["hp_phase2"]["best_config"] = best
        self.status.data["hp_phase2"]["status"] = "complete"
        self.status.save()
        print(f"HP Phase 2 complete. Best config: {best}")

    def _accumulate_phase2_score(
        self, scores: dict, fold_idx: int, entry: dict, score
    ) -> None:
        hp_idx = entry["hp_config"]
        if self.config.task == "fmd":
            t = entry["nms_threshold"]
            r = entry["nms_radius"]
            scores[(fold_idx, hp_idx, t, r)] = score
        else:
            scores[(fold_idx, hp_idx)] = score

    # ── Final evaluation ──────────────────────────────────────────────────────

    def _run_final_eval(self) -> None:
        """
        Train on full training pool using best HP config.
        Repeat for n_seeds seeds. Save best model checkpoint.
        """
        real_split = load_split(self.config.split_file_real)
        synth_split = (
            load_split(self.config.split_file_synthetic)
            if self.config.split_file_synthetic
            else None
        )

        best = self.status.best_config
        hp_idx = best["hp_config"]
        hp_config = self.config.hp_grid[hp_idx]

        fmd_nms_params = None
        if self.config.task == "fmd":
            fmd_nms_params = (best["nms_threshold"], best["nms_radius"])

        # Substitution conditions (R1-R4) reduce the real pool first;
        # no-op for V0-V3/R0 where config.real_n is the full task pool.
        train_ids_real = self._reduce_real_ids(real_split["train_val_ids"], seed=3000)
        test_ids_real  = real_split["test_ids"]

        train_transform = build_transform(self.config, training=True)
        test_transform  = build_transform(self.config, training=False)

        real_train_ds = build_real_dataset(
            self.config, train_ids_real, train_transform, self.data_root
        )
        real_test_ds = build_real_dataset(
            self.config, test_ids_real, test_transform, self.data_root
        )

        synth_train_ds = None
        if synth_split and synth_split.get("train_ids"):
            synth_train_ds = build_synthetic_dataset(
                self.config, synth_split["train_ids"],
                train_transform, self.data_root
            )

        train_ds = (
            CombinedDataset(real_train_ds, synth_train_ds)
            if synth_train_ds
            else real_train_ds
        )

        best_val_score = -float("inf")
        best_seed = None

        for seed in range(self.config.n_seeds):
            if seed in self.status.completed_seeds:
                continue

            print(f"[Final Eval] seed={seed}", flush=True)

            pred_path = self.output_dir / "final_eval" / f"seed_{seed}.npy"
            pred_path.parent.mkdir(parents=True, exist_ok=True)

            model = build_model(self.config).to(self.device)
            optimizer = build_optimizer(model, hp_config)
            scheduler = build_scheduler(
                optimizer, hp_config, self.config.max_steps
            )
            loss_fn = self._build_loss(real_train_ds)
            trainer = Trainer(model, optimizer, loss_fn, self.device, scheduler)

            history = trainer.train_full(
                train_ds, self.config, hp_config, pred_path, seed,
                skip_if_exists=False,
                test_dataset=real_test_ds,
                fmd_nms_params=fmd_nms_params,
            )
            hist_path = pred_path.with_suffix(".json")
            with hist_path.open("w") as f:
                json.dump(history, f)

            # Track best model by final train loss (proxy — no val set here)
            final_loss = (
                history["train_loss"][-1]
                if history["train_loss"]
                else float("inf")
            )
            if -final_loss > best_val_score:
                best_val_score = -final_loss
                best_seed = seed
                torch.save(
                    model.state_dict(),
                    self.output_dir / "final_eval" / "best_model.pth",
                )

            del model, optimizer, scheduler, trainer
            torch.cuda.empty_cache()

            self.status.data["final_eval"]["completed_seeds"].append(seed)
            self.status.save()

        self.status.data["final_eval"]["status"] = "complete"
        self.status.data["status"] = "final_eval_complete"
        self.status.save()
        print(f"Final eval complete. Best seed: {best_seed}")

    # ── Loss helper ───────────────────────────────────────────────────────────

    def _build_loss(self, train_dataset) -> nn.Module:
        if self.config.task == "fc":
            return FCLoss()
        elif self.config.task == "fmd":
            # Fixed weights from the full real V0/R0 pool, not this
            # experiment's own (possibly reduced or empty, for R4) real
            # subset -- see _get_fixed_fmd_class_weights docstring/comment.
            weights = _get_fixed_fmd_class_weights(self.data_root)
            return FMDLoss(class_weights=weights.to(self.device))
        elif self.config.task == "foe":
            return FOELoss()
        else:
            raise ValueError(f"Unknown task: {self.config.task}")
