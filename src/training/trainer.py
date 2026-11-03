from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax as _softmax
from torch.utils.data import DataLoader, Dataset

from src.config.schema import ExperimentConfig, HPConfig
from src.evaluation.fmd_evaluator import apply_nms as _fmd_apply_nms


# ── Orientation decoding ───────────────────────────────────────────────────────

def _build_encoding_matrix() -> torch.Tensor:
    """
    Precompute [256, 128] float32 codebook matrix (must match
    src.data.real.foe_dataset.encode_orientation_map exactly).
    Row k: bit i = 1 iff bin k falls in arc [(i+1) mod 256, (i+129) mod 256).
    """
    B = 256
    k_vals = torch.arange(B, dtype=torch.int32)          # (256,)
    i_vals = torch.arange(128, dtype=torch.int32)         # (128,)
    # diff[k, i] = (k - (i+1)) % 256
    diff = (k_vals[:, None] - (i_vals[None, :] + 1)) % B  # (256, 128)
    return (diff < 128).float()                             # (256, 128)


_ENCODING_MATRIX: torch.Tensor | None = None


def _get_encoding_matrix(device: torch.device) -> torch.Tensor:
    global _ENCODING_MATRIX
    if _ENCODING_MATRIX is None:
        _ENCODING_MATRIX = _build_encoding_matrix()
    return _ENCODING_MATRIX.to(device)


def decode_foe_output(logits: torch.Tensor) -> np.ndarray:
    """
    Decode FOE raw logits to orientation angles in [0, π).

    Per src.data.real.foe_dataset.encode_orientation_map, codeword popcount
    is popcount(k) = min(k, 256-k) -- it is NOT constant across k. The
    codebook itself is a valid circular Hamming code (verified:
    hamming(codebook[a], codebook[b]) == circular_distance(a, b) for all
    a, b), so nearest-Hamming-distance decoding is exactly correct.

    The previous implementation instead took argmax of the raw dot product
    probs @ codebook.T. For binary-ish probs, dot(p, c_k) = popcount(k) -
    0.5*hamming(p, c_k) - 0.5*hamming-asymmetry-term -- in short, raw dot
    product is biased toward whichever k has the largest codeword popcount
    (k=128, popcount 128), regardless of the true input. That bias is what
    collapsed every decoded angle to exactly 90 deg. Subtracting half the
    candidate's own popcount before the argmax exactly cancels this bias
    and reduces to minimizing soft Hamming distance:
        cost(k) = sum(p) + popcount(k) - 2*dot(p, c_k)
    so argmin cost(k) == argmax [dot(p, c_k) - 0.5*popcount(k)].
    Verified exact (256/256 bins) round-trip recovery with this scoring.

    Args:
        logits: [B, 128, H, W] raw logits (no sigmoid applied)
    Returns:
        [B, H, W] float32 orientation angles in radians
    """
    probs = torch.sigmoid(logits)                          # [B, 128, H, W]
    B_sz, _, H, W = probs.shape
    enc = _get_encoding_matrix(probs.device)               # [256, 128]
    popcount = enc.sum(dim=1)                               # [256]

    # Dot product: probs reshaped to [B*H*W, 128], enc^T = [128, 256]
    flat = probs.permute(0, 2, 3, 1).reshape(-1, 128)     # [B*H*W, 128]
    scores = flat @ enc.T - 0.5 * popcount                 # [B*H*W, 256]
    best_k = scores.argmax(dim=1)                          # [B*H*W]

    # Convert bin index to angle
    angles = best_k.float() * (np.pi / 256.0)             # [B*H*W] in [0, π)
    return angles.reshape(B_sz, H, W).cpu().numpy()


# ── Validation metrics ─────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_val_metric(
    model: nn.Module,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: str,
) -> float:
    model.eval()
    task = config.task

    if task == "fc":
        correct = total = 0
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            if labels is None:
                continue
            if isinstance(labels, list):
                valid = [l for l in labels if l is not None]
                if not valid:
                    continue
                labels = torch.stack(valid)
                images = images[: len(valid)]
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            targets = labels.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(preds)
        return correct / total if total > 0 else 0.0

    elif task == "foe":
        total_err = total = 0
        for batch in val_loader:
            images = batch["image"].to(device)
            gt_orient = batch.get("orientation")
            if gt_orient is None:
                continue
            logits = model(images)
            pred_angles = decode_foe_output(logits)          # [B, H, W] rad
            gt_np = gt_orient.numpy()                        # [B, H_o, W_o]
            # Resize pred to GT orientation resolution if needed
            if pred_angles.shape[1:] != gt_np.shape[1:]:
                import cv2
                resized = []
                for pred in pred_angles:
                    r = cv2.resize(pred, (gt_np.shape[2], gt_np.shape[1]),
                                   interpolation=cv2.INTER_LINEAR)
                    resized.append(r)
                pred_angles = np.stack(resized, axis=0)
            diff = np.abs(pred_angles - gt_np)
            diff = np.minimum(diff, np.pi - diff)            # angular wrap
            total_err += diff.mean() * len(images)
            total += len(images)
        mean_rad = total_err / total if total > 0 else 0.0
        return float(np.degrees(mean_rad))

    elif task == "fmd":
        # Approximate: fraction of pixels where argmax matches
        correct = total = 0
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            if labels is None:
                continue
            if isinstance(labels, list):
                valid = [l for l in labels if l is not None]
                if not valid:
                    continue
                labels = torch.stack(valid)
                images = images[: len(valid)]
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            targets = labels.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += preds.numel()
        return correct / total if total > 0 else 0.0

    return 0.0


# ── Prediction saving ──────────────────────────────────────────────────────────

@torch.no_grad()
def _save_predictions(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: str,
    output_path: Path,
) -> None:
    model.eval()
    all_preds: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)

        if config.task == "fc":
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
        elif config.task == "fmd":
            all_preds.append(logits.cpu().numpy())           # [B, 3, H, W]
        elif config.task == "foe":
            angles = decode_foe_output(logits)               # [B, H, W]
            all_preds.append(angles)

    preds = np.concatenate(all_preds, axis=0)

    # Atomic write: temp file then rename
    # Use os.fdopen so np.save writes to the exact fd — avoids numpy
    # appending ".npy" to a path that doesn't already end in ".npy".
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, preds)
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@torch.no_grad()
def _save_fmd_final_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    output_path: Path,
    nms_params: tuple[float, int],
    n_representative: int = 5,
) -> None:
    """
    FMD final-eval predictions only: apply NMS with the already-selected
    best (threshold, radius) and save the resulting per-image minutiae
    point list instead of the raw dense heatmap -- shrinks final_eval from
    ~1.1GB/seed to a few KB (matches the documented design in
    software_architecture.md, which was never implemented until now).
    HP Phase 1/2 are unaffected and still save raw heatmaps via
    _save_predictions, since score_fold()'s NMS *sweep* needs the dense
    heatmap to search across all (threshold, radius) combinations --
    final eval already knows which single combination to use.

    A fixed first `n_representative` test images additionally keep their
    raw heatmap (saved separately) for qualitative visualization.
    """
    model.eval()
    t, r = nms_params
    point_lists: list[np.ndarray] = []

    rep_dir = output_path.parent / "representative_maps"
    seed_tag = output_path.stem  # e.g. "seed_0"

    img_idx = 0
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images).cpu().numpy()  # [B, 3, H, W]

        for i in range(logits.shape[0]):
            hm = logits[i]
            if img_idx < n_representative:
                rep_dir.mkdir(parents=True, exist_ok=True)
                rep_path = rep_dir / f"{seed_tag}_img{img_idx}.npy"
                rep_fd, rep_tmp = tempfile.mkstemp(dir=rep_dir, suffix=".npy.tmp")
                try:
                    with os.fdopen(rep_fd, "wb") as rf:
                        np.save(rf, hm)
                    os.replace(rep_tmp, rep_path)
                except Exception:
                    try:
                        os.unlink(rep_tmp)
                    except OSError:
                        pass
                    raise

            hm_softmax = _softmax(hm, axis=0)
            pred_points = _fmd_apply_nms(hm_softmax, t, r)
            arr = (
                np.array(pred_points, dtype=np.int32)
                if pred_points
                else np.zeros((0, 3), dtype=np.int32)
            )
            point_lists.append(arr)
            img_idx += 1

    points_obj = np.empty(len(point_lists), dtype=object)
    for i, arr in enumerate(point_lists):
        points_obj[i] = arr

    # Atomic write, same discipline as _save_predictions.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, points_obj, allow_pickle=True)
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── DataLoader helpers ─────────────────────────────────────────────────────────

def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 16,
) -> DataLoader:
    collate_fn = getattr(dataset, "collate_fn", None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def build_optimizer(model: nn.Module, hp_config: HPConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=hp_config.lr,
        weight_decay=hp_config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    hp_config: HPConfig,
    max_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if hp_config.cosine_annealing:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps
        )
    return None


# ── Seeding ────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Core training loop ─────────────────────────────────────────────────────────

def _run_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: ExperimentConfig,
    device: str,
    use_amp: bool,
) -> dict:
    max_steps = config.max_steps
    val_interval = max(1, max_steps // 10)
    log_interval = val_interval

    train_loss_history: list[float] = []
    val_metrics: list[float] = []
    val_steps: list[int] = []
    best_step = 0
    best_val = -float("inf") if config.task == "fc" else float("inf")

    step = 0
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            images = batch["image"].to(device)
            labels = batch["label"]

            # Filter samples without labels (synthetic without annotation)
            if isinstance(labels, list):
                valid_mask = [l is not None for l in labels]
                if not any(valid_mask):
                    continue
                valid_idx = [i for i, v in enumerate(valid_mask) if v]
                images = images[valid_idx]
                labels = torch.stack([labels[i] for i in valid_idx])
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            train_loss_history.append(loss.item())
            step += 1

            if step % log_interval == 0 or step == max_steps:
                print(f"  step {step}/{max_steps} | loss: {loss.item():.4f}", flush=True)

            # Validation checkpoint
            if val_loader is not None and step % val_interval == 0:
                metric = _compute_val_metric(model, val_loader, config, device)
                val_metrics.append(metric)
                val_steps.append(step)
                # Track best step (higher = better for FC/FMD pixel acc; lower for FOE)
                is_better = (
                    metric > best_val
                    if config.task in ("fc", "fmd")
                    else metric < best_val
                )
                if is_better:
                    best_val = metric
                    best_step = step
                model.train()

    return {
        "train_loss": train_loss_history,
        "val_metric": val_metrics,
        "val_steps":  val_steps,
        "best_step":  best_step,
    }


# ── Trainer class ──────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str,
        scheduler=None,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()

    def train_one_fold(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: ExperimentConfig,
        hp_config: HPConfig,
        output_path: Path,
        skip_if_exists: bool = True,
        fold_idx: int = 0,
    ) -> dict:
        """
        Train for max_steps using hp_config.
        Saves val predictions to output_path as .npy.
        Returns training history dict.
        """
        output_path = Path(output_path)
        if skip_if_exists and output_path.exists():
            history_path = output_path.with_suffix(".json")
            if history_path.exists():
                with open(history_path) as f:
                    return json.load(f)
            return {"train_loss": [], "val_metric": [], "val_steps": [],
                    "best_step": 0, "hp_config": hp_config.model_dump()}

        _seed_everything(fold_idx)

        batch_size = max(8, config.batch_size) if config.dev_mode else config.batch_size
        dw = {"num_workers": 0} if config.dev_mode else {}

        train_loader = build_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, **dw
        )
        val_loader = build_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, **dw
        )

        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.model.train()

        history = _run_steps(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            scaler=scaler,
            scheduler=self.scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=self.device,
            use_amp=self.use_amp,
        )
        history["hp_config"] = hp_config.model_dump()

        # Save predictions on val set
        _save_predictions(self.model, val_loader, config, self.device, output_path)

        # Save history alongside predictions (atomic)
        _save_json(history, output_path.with_suffix(".json"))

        return history

    def train_full(
        self,
        train_dataset: Dataset,
        config: ExperimentConfig,
        hp_config: HPConfig,
        output_path: Path,
        seed: int,
        skip_if_exists: bool = True,
        test_dataset: Dataset | None = None,
        fmd_nms_params: tuple[float, int] | None = None,
    ) -> dict:
        """
        Train on the full training pool (no validation split).
        Saves test predictions to output_path as .npy.
        """
        output_path = Path(output_path)
        if skip_if_exists and output_path.exists():
            history_path = output_path.with_suffix(".json")
            if history_path.exists():
                with open(history_path) as f:
                    return json.load(f)
            return {"train_loss": [], "val_metric": [], "val_steps": [],
                    "best_step": 0, "hp_config": hp_config.model_dump()}

        _seed_everything(seed)

        batch_size = max(8, config.batch_size) if config.dev_mode else config.batch_size
        dw = {"num_workers": 0} if config.dev_mode else {}

        train_loader = build_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, **dw
        )

        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.model.train()

        history = _run_steps(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            scaler=scaler,
            scheduler=self.scheduler,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            device=self.device,
            use_amp=self.use_amp,
        )
        history["hp_config"] = hp_config.model_dump()

        # Save test predictions if test dataset provided
        if test_dataset is not None:
            test_loader = build_dataloader(
                test_dataset, batch_size=batch_size, shuffle=False, **dw
            )
            if config.task == "fmd" and fmd_nms_params is not None:
                _save_fmd_final_predictions(
                    self.model, test_loader, self.device, output_path, fmd_nms_params
                )
            else:
                _save_predictions(self.model, test_loader, config, self.device, output_path)

        _save_json(history, output_path.with_suffix(".json"))

        return history


# ── JSON helper ────────────────────────────────────────────────────────────────

def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        os.close(fd)
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
