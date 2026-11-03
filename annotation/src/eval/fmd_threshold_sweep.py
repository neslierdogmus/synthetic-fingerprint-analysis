#!/usr/bin/env python3
"""One-off diagnostic: extract minutiae once per image for a given FVC
subset, cache raw detected+gt minutiae to disk, then report recall/
precision/F1 under several (distance, angle, type-aware) combinations
without re-running VeriFinger extraction for each combination.

Built to answer a specific follow-up question (type-agnostic F1 vs the
type-aware numbers already in fmd_eval_full.json) -- see CONTEXT.md /
conversation history, not a general-purpose tool.

Usage:
  python3 fmd_threshold_sweep.py --dir FM3_FVC2002DB1A_np --out /workspace/outputs/fmd_sweep_2002DB1A.json
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verifinger_extract as vf

FVC_ROOT = "/data/real/FVC_BERKAY_TOPCU"

THRESHOLD_PAIRS_DEG = [
    ("16px_pi/6", 16.0, 30.0),
    ("12px_pi/8", 12.0, 22.5),
    ("8px_pi/10", 8.0, 18.0),
    ("15px_30deg_default", 15.0, 30.0),
]


def match(detected, gt_x, gt_y, gt_t, gt_a, dist_thresh, angle_thresh, type_aware):
    used = set()
    matched = 0
    for i in range(len(gt_x)):
        best_j, best_d = None, dist_thresh + 1
        for j, (x, y, t, a) in enumerate(detected):
            if j in used:
                continue
            if type_aware and t != gt_t[i]:
                continue
            d = math.hypot(x - gt_x[i], y - gt_y[i])
            if d > dist_thresh:
                continue
            adiff = min(abs(a - gt_a[i]), 360 - abs(a - gt_a[i]))
            if adiff <= angle_thresh and d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
            matched += 1
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="FM3_FVC2002DB1A_np")
    ap.add_argument("--out", default="/workspace/outputs/fmd_sweep.json")
    ap.add_argument("--cache", default=None, help="path to cache raw detected+gt minutiae (skips re-extraction if present)")
    args = ap.parse_args()

    cache_path = args.cache or args.out.replace(".json", "_cache.json")

    if os.path.exists(cache_path):
        print(f"[sweep] loading cached raw minutiae from {cache_path}", flush=True)
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        pairs = []
        for npz_path in sorted(glob.glob(os.path.join(FVC_ROOT, args.dir, "*.npz"))):
            tif_path = npz_path[:-4] + ".tif"
            if os.path.exists(tif_path):
                pairs.append((tif_path, npz_path))
        print(f"[sweep] {len(pairs)} pairs in {args.dir}, extracting (no cache found)...", flush=True)
        vf.init()
        cache = []
        t0 = time.time()
        for idx, (tif_path, npz_path) in enumerate(pairs):
            try:
                detected, status = vf.extract_minutiae(tif_path)
            except RuntimeError as e:
                print(f"[sweep] SKIP {tif_path}: {e}", flush=True)
                detected, status = [], -1
            gt = np.load(npz_path)
            cache.append({
                "stem": os.path.basename(tif_path)[:-4],
                "status": status,
                "detected": detected,  # list of [x,y,type,angle]
                "gt_x": gt["X"].tolist(), "gt_y": gt["Y"].tolist(),
                "gt_t": gt["T"].tolist(), "gt_a": gt["A"].tolist(),
            })
            if (idx + 1) % 100 == 0 or (idx + 1) == len(pairs):
                elapsed = time.time() - t0
                print(f"[sweep] {idx+1}/{len(pairs)} extracted, {  (idx+1)/elapsed:.2f} img/s", flush=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"[sweep] wrote raw cache to {cache_path}", flush=True)

    sweep_results = {}
    for label, dist_thresh, angle_thresh in THRESHOLD_PAIRS_DEG:
        for type_aware in (True, False):
            key = f"{label}__type_{'aware' if type_aware else 'agnostic'}"
            total_gt = total_det = total_matched = 0
            for rec in cache:
                detected = rec["detected"]
                gt_x, gt_y, gt_t, gt_a = rec["gt_x"], rec["gt_y"], rec["gt_t"], rec["gt_a"]
                m = match(detected, gt_x, gt_y, gt_t, gt_a, dist_thresh, angle_thresh, type_aware)
                total_gt += len(gt_x)
                total_det += len(detected)
                total_matched += m
            recall = total_matched / total_gt if total_gt else 0.0
            precision = total_matched / total_det if total_det else 0.0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
            sweep_results[key] = {
                "dist_thresh_px": dist_thresh, "angle_thresh_deg": angle_thresh, "type_aware": type_aware,
                "recall": recall, "precision": precision, "f1": f1,
                "total_gt": total_gt, "total_detected": total_det, "total_matched": total_matched,
            }
            print(f"[sweep] {key}: recall={recall:.4f} precision={precision:.4f} F1={f1:.4f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[sweep] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
