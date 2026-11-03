#!/usr/bin/env python3
"""FC + FOE accuracy for S11 against its own shipped ground truth.

S11 (unlike the other 10 synthetic sources) ships real per-image ground
truth alongside the images: the class is encoded in the filename itself
(`{class}_{id}--{impression}.png`, class in {arch, left_loop, right_loop,
tented_arch, whorl} -- an exact 5-way match to the FC {A,L,R,T,W} scheme),
and `{stem}_o.txt` is a plain-text 47x47 block orientation grid (degrees
0-179, 0 = background/singular-point sentinel -- confirmed by the zero
cells forming a clean fingerprint-shaped mask, not scattered noise).

This does NOT re-run VeriFinger/pyfing: it evaluates the annotations
already shipped in /workspace/annotations/S11 against this newly-noticed
ground truth, by decoding our own .gt output and reprojecting it onto the
ground truth's grid (border=14, step=8 -- confirmed empirically to match
FOESamples' convention: ceil((400-2*14)/8) = 47, matching the observed
grid size exactly, and validated by a spot-check giving a plausible ~2
degree mean error rather than a garbled ~45-90 degree one).
"""
import argparse
import glob
import json
import os

import numpy as np

SYNTH_DIR = "/data/synthetic/S11"
ANNOT_DIR = "/workspace/annotations/S11"
BORDER, STEP = 14, 8

CLASS_PREFIX_TO_LETTER = {
    "arch": "A",
    "left_loop": "L",
    "right_loop": "R",
    "tented_arch": "T",
    "whorl": "W",
}
CLASSES = ["A", "L", "R", "T", "W"]


def parse_stem_class(stem):
    for prefix, letter in CLASS_PREFIX_TO_LETTER.items():
        if stem.startswith(prefix + "_"):
            return letter
    return None


def parse_predicted_class(txt_path):
    with open(txt_path) as f:
        for line in f:
            if line.startswith("Class:"):
                return line.split(":", 1)[1].strip()
    return None


def foe_error_for_stem(stem):
    gt_path = os.path.join(SYNTH_DIR, stem + "_o.txt")
    pred_path = os.path.join(ANNOT_DIR, stem + ".gt")
    if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
        return None
    gt = np.loadtxt(gt_path, dtype=int)
    hb_gt, wb_gt = gt.shape
    raw = np.fromfile(pred_path, dtype=np.uint8)
    side = int(round(len(raw) ** 0.5))
    pred = raw.reshape(side, side)

    errs = []
    for jg in range(hb_gt):
        cy = BORDER + jg * STEP
        pj = cy // 8
        if pj >= pred.shape[0]:
            continue
        for ig in range(wb_gt):
            if gt[jg, ig] == 0:
                continue
            cx = BORDER + ig * STEP
            pi = cx // 8
            if pi >= pred.shape[1]:
                continue
            gt_deg = gt[jg, ig]
            pred_deg = pred[pj, pi] / 256.0 * 180.0
            diff = abs(gt_deg - pred_deg) % 180
            diff = min(diff, 180 - diff)
            errs.append(diff)
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="/workspace/outputs/s11_gt_eval.json")
    args = ap.parse_args()

    stems = sorted(
        os.path.basename(p)[:-4]
        for p in glob.glob(os.path.join(ANNOT_DIR, "*.txt"))
    )
    if args.limit:
        stems = stems[: args.limit]
    print(f"[s11_eval] {len(stems)} annotated stems to check", flush=True)

    confusion = {gt: {p: 0 for p in CLASSES} for gt in CLASSES}
    fc_total = fc_correct = 0
    fc_skipped = 0
    all_foe_errs = []
    foe_per_image = []
    foe_skipped = 0

    for idx, stem in enumerate(stems):
        gt_class = parse_stem_class(stem)
        pred_class = parse_predicted_class(os.path.join(ANNOT_DIR, stem + ".txt"))
        if gt_class is None or pred_class not in CLASSES:
            fc_skipped += 1
        else:
            fc_total += 1
            confusion[gt_class][pred_class] += 1
            if pred_class == gt_class:
                fc_correct += 1

        errs = foe_error_for_stem(stem)
        if errs is None or len(errs) == 0:
            foe_skipped += 1
        else:
            all_foe_errs.append(errs)
            foe_per_image.append({"stem": stem, "n_blocks": len(errs), "mean_deg": float(errs.mean())})

        if (idx + 1) % 5000 == 0 or (idx + 1) == len(stems):
            print(f"[s11_eval] {idx+1}/{len(stems)} done", flush=True)

    fc_acc = fc_correct / fc_total if fc_total else None
    all_foe_errs_flat = np.concatenate(all_foe_errs) if all_foe_errs else np.array([])

    summary = {
        "fc": {
            "n": fc_total,
            "correct": fc_correct,
            "accuracy": fc_acc,
            "skipped_unparseable": fc_skipped,
            "confusion_matrix": confusion,
        },
        "foe": {
            "n_images": len(all_foe_errs),
            "skipped": foe_skipped,
            "n_blocks": int(len(all_foe_errs_flat)),
            "mean_error_deg": float(all_foe_errs_flat.mean()) if len(all_foe_errs_flat) else None,
            "median_error_deg": float(np.median(all_foe_errs_flat)) if len(all_foe_errs_flat) else None,
            "pct_within_10deg": float((all_foe_errs_flat <= 10).mean() * 100) if len(all_foe_errs_flat) else None,
            "pct_within_20deg": float((all_foe_errs_flat <= 20).mean() * 100) if len(all_foe_errs_flat) else None,
        },
    }
    print(json.dumps({"fc": {**summary["fc"], "confusion_matrix": summary["fc"]["confusion_matrix"]},
                       "foe": summary["foe"]}, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "foe_per_image": foe_per_image}, f, indent=2)
    print(f"[s11_eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
