#!/usr/bin/env python3
"""FOE (orientation) mean/median circular error vs FOESamples real ground truth.

Good/ (10 images) was already spot-checked in an earlier ad hoc session
(3.2 deg mean, see CONTEXT.md); Bad/ (50 images, presumably low-quality
scans) is new. Synth/ (~12,000, synthetic not real scans) is excluded by
default -- its scope for a "real data" accuracy analysis is unclear (see
CONTEXT.md), pass --dirs Synth explicitly if you want it included.

Usage:
  python3 foe_eval.py --dev              # smoke test, 3 images from Good/
  python3 foe_eval.py --dirs Good,Bad --out /workspace/outputs/foe_eval.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import foe_extract as fe

FOE_ROOT = "/data/real/FOESamples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="smoke test: 3 images from Good/ only")
    ap.add_argument("--dirs", default="Good,Bad")
    ap.add_argument("--limit-per-dir", type=int, default=None)
    ap.add_argument("--out", default="/workspace/outputs/foe_eval.json")
    args = ap.parse_args()

    dirs = ["Good"] if args.dev else args.dirs.split(",")
    limit_per_dir = 3 if args.dev else args.limit_per_dir

    import pyfing
    algo = pyfing.Snfoe()

    stems = []
    for d in dirs:
        bmps = sorted(glob.glob(os.path.join(FOE_ROOT, d, "*.bmp")))
        if limit_per_dir:
            bmps = bmps[:limit_per_dir]
        for bmp_path in bmps:
            stem = bmp_path[:-4]
            if os.path.exists(stem + ".gt") and os.path.exists(stem + ".fg"):
                stems.append((stem, d))

    print(f"[foe_eval] {len(stems)} images across dirs={dirs}", flush=True)

    per_image = []
    per_dir_errs = {d: [] for d in dirs}
    all_errs = []
    t0 = time.time()
    for idx, (stem, d) in enumerate(stems):
        errs = fe.evaluate_image(algo, stem + ".bmp", stem + ".gt", stem + ".fg")
        per_dir_errs[d].append(errs)
        all_errs.append(errs)
        per_image.append({
            "stem": os.path.basename(stem),
            "dir": d,
            "n_blocks": len(errs),
            "mean_error_deg": float(errs.mean()) if len(errs) else None,
            "median_error_deg": float(np.median(errs)) if len(errs) else None,
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == len(stems):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"[foe_eval] {idx+1}/{len(stems)} done, {rate:.2f} img/s, "
                  f"ETA {((len(stems)-idx-1)/rate)/60:.1f} min", flush=True)

    all_errs_flat = np.concatenate(all_errs) if all_errs else np.array([])
    per_dir_summary = {}
    for d in dirs:
        cat = np.concatenate(per_dir_errs[d]) if per_dir_errs[d] else np.array([])
        per_dir_summary[d] = {
            "n_images": len(per_dir_errs[d]),
            "n_blocks": int(len(cat)),
            "mean_error_deg": float(cat.mean()) if len(cat) else None,
            "median_error_deg": float(np.median(cat)) if len(cat) else None,
            "pct_within_10deg": float((cat <= 10).mean() * 100) if len(cat) else None,
            "pct_within_20deg": float((cat <= 20).mean() * 100) if len(cat) else None,
        }
    summary = {
        "n_images": len(stems),
        "n_blocks": int(len(all_errs_flat)),
        "mean_error_deg": float(all_errs_flat.mean()) if len(all_errs_flat) else None,
        "median_error_deg": float(np.median(all_errs_flat)) if len(all_errs_flat) else None,
        "pct_within_10deg": float((all_errs_flat <= 10).mean() * 100) if len(all_errs_flat) else None,
        "pct_within_20deg": float((all_errs_flat <= 20).mean() * 100) if len(all_errs_flat) else None,
        "per_dir": per_dir_summary,
    }
    print(f"[foe_eval] DONE mean={summary['mean_error_deg']} median={summary['median_error_deg']}", flush=True)
    print(json.dumps(per_dir_summary, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": per_image}, f, indent=2)
    print(f"[foe_eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
