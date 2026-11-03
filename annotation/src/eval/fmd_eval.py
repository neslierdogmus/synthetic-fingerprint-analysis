#!/usr/bin/env python3
"""FMD (minutiae) recall/precision vs FVC_BERKAY_TOPCU real ground truth.

Ground truth is per-image .npz (X, Y, T, A arrays), NOT a .mnt sidecar --
LABEL_FORMAT.md describes a .mnt real source but the actual mounted data
uses .npz (see CONTEXT.md). T already matches our 1=end/2=bifurcation
convention; A is already float degrees.

Matching is greedy nearest-neighbor per ground-truth minutia (same type,
within dist_thresh px and angle_thresh deg) -- the standard approach for
comparing two independent minutiae extractors, since exact-match isn't
meaningful (different algorithms find different subsets/order).

Usage:
  python3 fmd_eval.py --dev                                   # smoke test, 10 images
  python3 fmd_eval.py --dirs FM3_FVC2002DB3A_np,FM3_FVC2004DB1A_np,FM3_FVC2004DB3A_np \
      --out /workspace/outputs/fmd_eval.json
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
ALL_DIRS = ["FM3_FVC2002DB1A_np", "FM3_FVC2002DB3A_np", "FM3_FVC2004DB1A_np", "FM3_FVC2004DB3A_np"]


def match(detected, gt_x, gt_y, gt_t, gt_a, dist_thresh=15.0, angle_thresh=30.0):
    used = set()
    matched = 0
    for i in range(len(gt_x)):
        best_j, best_d = None, dist_thresh + 1
        for j, (x, y, t, a) in enumerate(detected):
            if j in used or t != gt_t[i]:
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
    ap.add_argument("--dev", action="store_true", help="smoke test: 10 images from FM3_FVC2002DB3A_np only")
    ap.add_argument("--dirs", default="FM3_FVC2002DB3A_np,FM3_FVC2004DB1A_np,FM3_FVC2004DB3A_np")
    ap.add_argument("--limit-per-dir", type=int, default=None)
    ap.add_argument("--out", default="/workspace/outputs/fmd_eval.json")
    args = ap.parse_args()

    dirs = ["FM3_FVC2002DB3A_np"] if args.dev else args.dirs.split(",")
    limit_per_dir = 10 if args.dev else args.limit_per_dir

    pairs = []
    for d in dirs:
        stems = sorted(glob.glob(os.path.join(FVC_ROOT, d, "*.npz")))
        if limit_per_dir:
            stems = stems[:limit_per_dir]
        for npz_path in stems:
            tif_path = npz_path[:-4] + ".tif"
            if os.path.exists(tif_path):
                pairs.append((tif_path, npz_path, d))

    print(f"[fmd_eval] {len(pairs)} image/label pairs across dirs={dirs}", flush=True)
    vf.init()

    per_image = []
    per_dir_recall = {d: [] for d in dirs}
    per_dir_precision = {d: [] for d in dirs}
    t0 = time.time()
    total_matched = total_gt = total_det = 0
    for idx, (tif_path, npz_path, d) in enumerate(pairs):
        try:
            detected, status = vf.extract_minutiae(tif_path)
        except RuntimeError as e:
            print(f"[fmd_eval] SKIP {tif_path}: {e}", flush=True)
            detected, status = [], -1
        gt = np.load(npz_path)
        gt_x, gt_y, gt_t, gt_a = gt["X"], gt["Y"], gt["T"], gt["A"]
        matched = match(detected, gt_x, gt_y, gt_t, gt_a)
        recall = matched / len(gt_x) if len(gt_x) else None
        precision = matched / len(detected) if detected else 0.0
        if recall is not None:
            per_dir_recall[d].append(recall)
            per_dir_precision[d].append(precision)
        total_matched += matched
        total_gt += len(gt_x)
        total_det += len(detected)
        per_image.append({
            "stem": os.path.basename(tif_path)[:-4],
            "dir": d,
            "status": status,
            "n_gt": int(len(gt_x)),
            "n_detected": len(detected),
            "matched": matched,
            "recall": recall,
            "precision": precision,
        })
        if (idx + 1) % 50 == 0 or (idx + 1) == len(pairs):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"[fmd_eval] {idx+1}/{len(pairs)} done, overall recall so far="
                  f"{total_matched/total_gt:.3f}, {rate:.2f} img/s, "
                  f"ETA {((len(pairs)-idx-1)/rate)/60:.1f} min", flush=True)

    overall_recall = total_matched / total_gt if total_gt else 0.0
    overall_precision = total_matched / total_det if total_det else 0.0
    per_dir_summary = {
        d: {
            "n_images": len(per_dir_recall[d]),
            "mean_recall": float(np.mean(per_dir_recall[d])) if per_dir_recall[d] else None,
            "mean_precision": float(np.mean(per_dir_precision[d])) if per_dir_precision[d] else None,
        }
        for d in dirs
    }
    summary = {
        "n_images": len(pairs),
        "overall_recall": overall_recall,
        "overall_precision": overall_precision,
        "total_gt_minutiae": total_gt,
        "total_detected_minutiae": total_det,
        "total_matched": total_matched,
        "per_dir": per_dir_summary,
    }
    print(f"[fmd_eval] DONE recall={overall_recall:.4f} precision={overall_precision:.4f}", flush=True)
    print(json.dumps(per_dir_summary, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": per_image}, f, indent=2)
    print(f"[fmd_eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
