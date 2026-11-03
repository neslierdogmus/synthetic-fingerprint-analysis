from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.spatial import cKDTree

from src.data.real.fmd_dataset import FMDRealDataset


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def _extract_candidates(heatmap_3ch: np.ndarray, threshold: float) -> np.ndarray:
    """
    Extract candidate minutiae above *threshold*, sorted by score descending.

    Returns array of shape (N, 4): [score, y, x, type].
    Empty array of shape (0, 4) when no candidates exist.
    """
    score_map = heatmap_3ch[1] + heatmap_3ch[2]   # [H, W]
    type_map = heatmap_3ch[1:].argmax(axis=0) + 1  # [H, W], values 1 or 2

    ys, xs = np.where(score_map >= threshold)
    if len(ys) == 0:
        return np.empty((0, 4), dtype=np.float32)

    scores = score_map[ys, xs]
    types = type_map[ys, xs]
    order = np.argsort(-scores)
    return np.column_stack([scores[order], ys[order], xs[order], types[order]])


def _nms_single(
    candidates: np.ndarray,
    radius: int,
) -> list[tuple[int, int, int]]:
    """
    Greedy NMS with KD-tree radius suppression.

    candidates: array shape (N, 4) — [score, y, x, type], sorted by score descending.
    Returns list of (x, y, type).
    """
    if len(candidates) == 0:
        return []

    coords = candidates[:, 1:3]  # (y, x)
    tree = cKDTree(coords)
    suppressed: set[int] = set()
    result: list[tuple[int, int, int]] = []

    for i in range(len(candidates)):
        if i in suppressed:
            continue
        _, y, x, t = candidates[i]
        result.append((int(x), int(y), int(t)))
        neighbours = tree.query_ball_point([y, x], radius - 1e-9)
        suppressed.update(neighbours)

    return result


def apply_nms(
    heatmap_3ch: np.ndarray,
    threshold: float,
    radius: int,
) -> list[tuple[int, int, int]]:
    """
    Apply NMS to a 3-channel softmax heatmap.

    heatmap_3ch: [3, H, W] float32, softmax probabilities summing to 1 per pixel
    threshold:   minimum combined score (ch1 + ch2) to consider a point
    radius:      minimum Euclidean distance between accepted points

    Returns list of (x, y, type) where type 1=ridge ending, 2=bifurcation.
    """
    candidates = _extract_candidates(heatmap_3ch, threshold)
    return _nms_single(candidates, radius)


def sweep_nms(
    heatmap_3ch: np.ndarray,
    thresholds: list[float],
    radii: list[int],
) -> dict[tuple[float, int], list[tuple[int, int, int]]]:
    """
    Apply NMS at all (threshold, radius) combinations.
    Candidates are extracted once per threshold and reused across all radii.
    Returns dict keyed by (threshold, radius) → list of (x, y, type).
    """
    results: dict[tuple[float, int], list[tuple[int, int, int]]] = {}
    for t in thresholds:
        candidates = _extract_candidates(heatmap_3ch, t)
        for r in radii:
            results[(t, r)] = _nms_single(candidates, r)
    return results


# ---------------------------------------------------------------------------
# Minutiae matching
# ---------------------------------------------------------------------------

def match_minutiae(
    predicted: list[tuple[int, int, int]],
    ground_truth: list[tuple[int, int, int]],
    matching_distance: int,
) -> tuple[int, int, int]:
    """
    One-to-one greedy bipartite matching by Euclidean distance.
    Returns (TP, FP, FN).
    """
    if not predicted or not ground_truth:
        return 0, len(predicted), len(ground_truth)

    pred_xy = np.array([[p[0], p[1]] for p in predicted], dtype=float)
    gt_xy   = np.array([[g[0], g[1]] for g in ground_truth], dtype=float)

    # All pairwise distances: [|pred|, |gt|]
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]   # [P, G, 2]
    dists = np.sqrt((diff ** 2).sum(axis=2))           # [P, G]

    # Collect candidate pairs sorted by distance
    pi_idx, gi_idx = np.where(dists <= matching_distance)
    if len(pi_idx) == 0:
        return 0, len(predicted), len(ground_truth)

    pair_dists = dists[pi_idx, gi_idx]
    order = np.argsort(pair_dists)
    pi_idx = pi_idx[order]
    gi_idx = gi_idx[order]

    matched_pred: set[int] = set()
    matched_gt:   set[int] = set()
    tp = 0
    for pi, gi in zip(pi_idx, gi_idx):
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(int(pi))
            matched_gt.add(int(gi))
            tp += 1

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# score_fold
# ---------------------------------------------------------------------------

def score_fold(
    pred_path: Path,
    val_dataset: FMDRealDataset,
    nms_params=None,
    matching_distance: int = 12,
    thresholds: list[float] | None = None,
    radii: list[int] | None = None,
) -> dict | float:
    """
    Phase 1 (nms_params=None): sweep all NMS combinations,
        return {(t, r): f1_score}.
    Phase 2 (nms_params=(t, r)): single float F1 score.
    """
    heatmaps = np.load(pred_path)  # [N, 3, H, W]

    if nms_params is None:
        accum = {
            (t, r): {"tp": 0, "fp": 0, "fn": 0}
            for t in thresholds for r in radii
        }

        for i in range(len(heatmaps)):
            item = val_dataset[i]
            gt_x = item["minutiae"]["x"]
            gt_y = item["minutiae"]["y"]
            gt_t = item["minutiae"]["type"]
            gt_points = list(zip(gt_x, gt_y, gt_t))

            hm_softmax = softmax(heatmaps[i], axis=0)  # [3, H, W]
            sweep = sweep_nms(hm_softmax, thresholds, radii)

            for (t, r), pred_points in sweep.items():
                tp, fp, fn = match_minutiae(
                    pred_points, gt_points, matching_distance
                )
                accum[(t, r)]["tp"] += tp
                accum[(t, r)]["fp"] += fp
                accum[(t, r)]["fn"] += fn

        return {
            key: precision_recall_f1(v["tp"], v["fp"], v["fn"])["f1"]
            for key, v in accum.items()
        }

    else:
        t, r = nms_params
        tp_total = fp_total = fn_total = 0
        for i in range(len(heatmaps)):
            item = val_dataset[i]
            gt_points = list(zip(
                item["minutiae"]["x"],
                item["minutiae"]["y"],
                item["minutiae"]["type"],
            ))
            hm_softmax = softmax(heatmaps[i], axis=0)
            pred_points = apply_nms(hm_softmax, t, r)
            tp, fp, fn = match_minutiae(pred_points, gt_points, matching_distance)
            tp_total += tp
            fp_total += fp
            fn_total += fn
        return precision_recall_f1(tp_total, fp_total, fn_total)["f1"]


# ---------------------------------------------------------------------------
# evaluate_full
# ---------------------------------------------------------------------------

def evaluate_full(
    pred_path: Path,
    test_dataset: FMDRealDataset,
    nms_params: tuple[float, int],
    matching_distances: list[int] | None = None,
) -> dict:
    """
    Comprehensive FMD metrics:
      1. Precision, Recall, F1 at 3 matching distances (per-image avg + global)
      2. Per-type breakdown (ridge ending / bifurcation)
      3. Per-quality breakdown (poor / fair / good)

    pred_path holds per-image minutiae point lists (already NMS'd at the
    selected threshold/radius when saved -- see
    trainer._save_fmd_final_predictions), not raw heatmaps. nms_params is
    kept as an argument purely to be echoed back in the returned metadata
    (it's the same value that was used to produce the saved point lists),
    not to recompute NMS here.
    """
    if matching_distances is None:
        matching_distances = [8, 12, 16]

    point_lists = np.load(pred_path, allow_pickle=True)  # object array, one (k_i,3) int array per image
    t, r = nms_params

    # Structures: {dist: {global: {tp,fp,fn}, per_image: [metrics]}}
    dist_accum = {
        d: {"tp": 0, "fp": 0, "fn": 0, "per_image": []}
        for d in matching_distances
    }
    # Per-type: only count matched type-matched pairs
    type_accum = {
        1: {"tp": 0, "fp": 0, "fn": 0},
        2: {"tp": 0, "fp": 0, "fn": 0},
    }
    # Per-quality
    quality_accum: dict[str, dict] = {}

    primary_dist = matching_distances[1] if len(matching_distances) > 1 else matching_distances[0]

    for i in range(len(point_lists)):
        item = test_dataset[i]
        gt_points = list(zip(
            item["minutiae"]["x"],
            item["minutiae"]["y"],
            item["minutiae"]["type"],
        ))
        quality = item.get("quality", "unknown")

        pred_points = [tuple(p) for p in point_lists[i]]

        for d in matching_distances:
            tp, fp, fn = match_minutiae(pred_points, gt_points, d)
            dist_accum[d]["tp"] += tp
            dist_accum[d]["fp"] += fp
            dist_accum[d]["fn"] += fn
            dist_accum[d]["per_image"].append(
                precision_recall_f1(tp, fp, fn)
            )

        # Per-type breakdown at primary distance
        for mtype in (1, 2):
            pred_t = [p for p in pred_points if p[2] == mtype]
            gt_t   = [g for g in gt_points   if g[2] == mtype]
            tp_t, fp_t, fn_t = match_minutiae(pred_t, gt_t, primary_dist)
            type_accum[mtype]["tp"] += tp_t
            type_accum[mtype]["fp"] += fp_t
            type_accum[mtype]["fn"] += fn_t

        # Per-quality at primary distance
        if quality not in quality_accum:
            quality_accum[quality] = {"tp": 0, "fp": 0, "fn": 0}
        tp_q, fp_q, fn_q = match_minutiae(pred_points, gt_points, primary_dist)
        quality_accum[quality]["tp"] += tp_q
        quality_accum[quality]["fp"] += fp_q
        quality_accum[quality]["fn"] += fn_q

    # Assemble output
    distance_results = {}
    for d in matching_distances:
        global_metrics = precision_recall_f1(
            dist_accum[d]["tp"], dist_accum[d]["fp"], dist_accum[d]["fn"]
        )
        per_image = dist_accum[d]["per_image"]
        avg_metrics = {
            k: float(np.mean([m[k] for m in per_image]))
            for k in ("precision", "recall", "f1")
        }
        distance_results[d] = {"global": global_metrics, "per_image_avg": avg_metrics}

    per_type = {
        "ridge_ending":   precision_recall_f1(**type_accum[1]),
        "bifurcation":    precision_recall_f1(**type_accum[2]),
    }
    per_quality = {
        q: precision_recall_f1(v["tp"], v["fp"], v["fn"])
        for q, v in quality_accum.items()
    }

    return {
        "distance_results": distance_results,
        "per_type": per_type,
        "per_quality": per_quality,
        "n_samples": len(point_lists),
        "nms_params": {"threshold": t, "radius": r},
    }
