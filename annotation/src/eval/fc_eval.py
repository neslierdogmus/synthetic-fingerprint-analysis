#!/usr/bin/env python3
"""FC (pattern classification) accuracy vs NIST_SD4 real ground truth.

Usage:
  python3 fc_eval.py --dev                 # smoke test, 20 images from figs_1
  python3 fc_eval.py --dirs figs_1,figs_2,figs_3,figs_4,figs_5,figs_6,figs_7 --out /workspace/outputs/fc_eval.json
  python3 fc_eval.py --dirs figs_0,figs_1,...,figs_7 --out ...   # full 4000, one consistent run

figs_0 was already spot-checked in an earlier ad hoc session (85.6%, see
CONTEXT.md) -- pass it explicitly if you want one clean run covering all
4000 pairs instead of stitching that old run in.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verifinger_extract as vf

NIST_ROOT = "/data/real/NIST_SD4/png_txt"
CLASSES = ["A", "L", "R", "T", "W", "S"]


def parse_gt(txt_path):
    with open(txt_path) as f:
        lines = f.read().splitlines()
    cls = None
    for l in lines:
        if l.startswith("Class:"):
            cls = l.split(":", 1)[1].strip()
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="smoke test: 20 images from figs_1 only")
    ap.add_argument("--dirs", default="figs_1,figs_2,figs_3,figs_4,figs_5,figs_6,figs_7")
    ap.add_argument("--limit", type=int, default=None, help="cap total images (after --dev)")
    ap.add_argument("--out", default="/workspace/outputs/fc_eval.json")
    args = ap.parse_args()

    dirs = ["figs_1"] if args.dev else args.dirs.split(",")
    limit = 20 if args.dev else args.limit

    pairs = []
    for d in dirs:
        for txt_path in sorted(glob.glob(os.path.join(NIST_ROOT, d, "*.txt"))):
            png_path = txt_path[:-4] + ".png"
            if os.path.exists(png_path):
                pairs.append((png_path, txt_path))
    if limit:
        pairs = pairs[:limit]

    print(f"[fc_eval] {len(pairs)} image/label pairs across dirs={dirs}", flush=True)
    vf.init()

    confusion = {gt: {p: 0 for p in CLASSES + ["NONE"]} for gt in CLASSES}
    unmapped_raw = {}
    results = []
    t0 = time.time()
    correct = 0
    total = 0
    for idx, (png_path, txt_path) in enumerate(pairs):
        gt_class = parse_gt(txt_path)
        if gt_class not in CLASSES:
            continue
        try:
            r = vf.classify(png_path)
        except RuntimeError as e:
            print(f"[fc_eval] SKIP {png_path}: {e}", flush=True)
            r = {"letter": None, "raw_class": None, "confidence": None, "status": -1}
        pred = r["letter"] or "NONE"
        confusion.setdefault(gt_class, {p: 0 for p in CLASSES + ["NONE"]})
        confusion[gt_class][pred] = confusion[gt_class].get(pred, 0) + 1
        if pred == "NONE" and r["raw_class"] is not None:
            unmapped_raw[r["raw_class"]] = unmapped_raw.get(r["raw_class"], 0) + 1
        total += 1
        if pred == gt_class:
            correct += 1
        results.append({
            "stem": os.path.basename(png_path)[:-4],
            "gt": gt_class,
            "pred": r["letter"],
            "raw_class": r["raw_class"],
            "confidence": r["confidence"],
            "status": r["status"],
        })
        if (idx + 1) % 100 == 0 or (idx + 1) == len(pairs):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"[fc_eval] {idx+1}/{len(pairs)} done, acc so far={correct/total:.3f}, "
                  f"{rate:.1f} img/s, ETA {((len(pairs)-idx-1)/rate)/60:.1f} min", flush=True)

    acc = correct / total if total else 0.0
    summary = {
        "n": total,
        "correct": correct,
        "accuracy": acc,
        "confusion_matrix": confusion,
        "unmapped_raw_pattern_classes": unmapped_raw,
        "dirs": dirs,
    }
    print(f"[fc_eval] DONE n={total} accuracy={acc:.4f}", flush=True)
    print(json.dumps(confusion, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"[fc_eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
