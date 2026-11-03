#!/usr/bin/env python3
"""Batch FC + FMD + FOE label extraction over /data/synthetic, writing to
/workspace/annotations/{source_id}/{stem}.{txt,mnt,gt} per LABEL_FORMAT.md.

Per CLAUDE.md: do NOT launch a full run (all 11 sources) without explicit
go-ahead -- use --dev first. Full runs must go through a detached tmux
session (see CLAUDE.md), not run in the foreground here.

Known open decisions / caveats (see CONTEXT.md for the full history):
  - Forced capture DPI defaults to 500 (NFR_RESOLUTION, VeriFinger's own
    default) for every source. Only actually verified true for S01
    (FVC2000/2002/2004/2006 DB4, confirmed via pixel dims). S02-S11 are
    synthetic and their true intended resolution hasn't been checked --
    override per-source with --dpi-overrides if that turns out to matter.
  - FMD angle uses BOTH documented conversions (RotationToDegrees, then
    (360-deg)%360 inversion) -- see verifinger_extract.py -- to match the
    real ground truth's angle convention validated this session, NOT the
    single conversion LABEL_FORMAT.md's text literally describes ("write
    it directly"). If FMDRealDataset's actual angle convention ever turns
    out to differ from the FVC_BERKAY_TOPCU .npz files used to validate
    this, this needs revisiting.
  - FC: images whose classification result has no {A,L,R,T,W,S} mapping
    (nfpcUnknown/nfpcAmputation/radial-ulnar, or a failed task) get NO
    .txt written for that stem (logged, not silently invented) -- this
    was flagged as an undecided fallback in CONTEXT.md and this script
    keeps it undecided rather than guessing; the image will simply be
    retried on the next run since not all 3 sidecars exist yet.
  - FOE foreground mask: no ground truth mask exists for synthetic
    images, so this uses pyfing's own Gmfs segmentation algorithm rather
    than a provided mask (differs from the FOESamples validation path in
    foe_extract.py, which used the real .fg mask).
  - History field always uses the LABEL_FORMAT.md "no paired impression"
    fallback (`{stem} {class} {stem}`) -- no attempt at same-subject
    impression pairing even for sources whose naming suggests it (S02,
    S05, etc. have multiple impressions per subject via the `--n` suffix).
"""
import argparse
import glob
import itertools
import os
import sys
import time

import cv2 as cv
import numpy as np
import pyfing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verifinger_extract as vf

SYNTH_ROOT = "/data/synthetic"
OUT_ROOT = "/workspace/annotations"

SOURCE_EXTENSIONS = {
    "S01": ["*.tif", "*.bmp"],
    "S02": ["*.jpg"],
    "S03": ["*.jpeg"],
    "S04": ["*.JPEG"],
    "S05": ["*.jpg"],
    "S06": ["*.jpg"],
    "S07": ["*.png"],
    "S08": ["*.png"],
    "S09": ["*.png"],
    "S10": ["*.png"],
    "S11": ["*.png"],
}
ALL_SOURCES = list(SOURCE_EXTENSIONS.keys())

DEFAULT_DPI = 500.0

# S01/S02/S11 measured well outside the natural spread seen on real,
# independently-confirmed 500dpi calibration data (see CONTEXT.md's
# "Per-source DPI check" section: dpi_probe.py's ridge-frequency estimator
# read 490-566 on known-true-500dpi real data, vs 374-389 for these three
# -- a real signal, not estimator noise, confirmed by calibrating against
# real data rather than assuming S01 was 500dpi from pixel-dim matching
# alone (that assumption was wrong, see CONTEXT.md). Values below are each
# source's own measured implied DPI, rounded to the nearest 5. No ground
# truth exists to verify these further (S01's own .txt sidecars are
# already known-untrustworthy) -- this is the best evidence-backed choice
# available, not a confirmed-correct one.
DPI_OVERRIDES = {
    "S01": 385.0,
    "S02": 375.0,
    "S11": 390.0,
}

# Sources with multiple impressions per identity: sample at the identity
# level (all impressions of a sampled identity, not scattered individual
# images) so a capped sample doesn't disproportionately lose identity
# diversity within its impression coverage. Identity = the stem up to the
# first "--", confirmed uniform per source via a real directory scan
# (2026-07-02): S02 10000 identities x 109-110 impr, S05 34993x15,
# S06 100000x2, S08/S09 4000x10 each. Everything else is sampled as plain
# random individual images.
MULTI_IMPRESSION_SOURCES = {"S02", "S05", "S06", "S08", "S09"}
SAMPLE_CAP = 30000
SAMPLE_SEED = 42
MANIFEST_DIR = os.path.join(OUT_ROOT, "_manifests")

_foe_algo = None
_foe_seg = None


def _foe_algorithms():
    global _foe_algo, _foe_seg
    if _foe_algo is None:
        _foe_algo = pyfing.Snfoe()
        _foe_seg = pyfing.Gmfs()
    return _foe_algo, _foe_seg


def iter_source_images(source, limit=None):
    """Yields (stem, path) for a source dir. Uses os.scandir + islice so a
    --dev run against a huge source (S02 has 1.09M files) doesn't pay for
    a full directory listing."""
    exts = tuple(e.lstrip("*") for e in SOURCE_EXTENSIONS[source])
    src_dir = os.path.join(SYNTH_ROOT, source)
    with os.scandir(src_dir) as it:
        matches = (e for e in it if e.is_file() and e.name.endswith(exts))
        if limit:
            matches = itertools.islice(matches, limit)
        for entry in matches:
            stem = os.path.splitext(entry.name)[0]
            yield stem, entry.path


def _all_stems(source):
    exts = tuple(e.lstrip("*") for e in SOURCE_EXTENSIONS[source])
    src_dir = os.path.join(SYNTH_ROOT, source)
    with os.scandir(src_dir) as it:
        return [os.path.splitext(e.name)[0] for e in it if e.is_file() and e.name.endswith(exts)]


def _build_sample(source, cap, seed):
    import random
    rng = random.Random(seed)
    if source in MULTI_IMPRESSION_SOURCES:
        stems = _all_stems(source)
        by_identity = {}
        for stem in stems:
            by_identity.setdefault(stem.split("--")[0], []).append(stem)
        identities = sorted(by_identity)
        if sum(len(v) for v in by_identity.values()) <= cap:
            return stems
        per_identity = round(sum(len(v) for v in by_identity.values()) / len(identities))
        n_ids = -(-cap // per_identity)  # ceil
        rng.shuffle(identities)
        chosen = []
        for ident in identities[:n_ids]:
            chosen.extend(by_identity[ident])
        return chosen
    else:
        stems = _all_stems(source)
        if len(stems) <= cap:
            return stems
        return rng.sample(stems, cap)


def sampled_stems(source, cap=SAMPLE_CAP, seed=SAMPLE_SEED):
    """Returns the fixed sample of stems for `source`, generating and
    caching a manifest on first call so re-running (e.g. after a restart
    mid-batch) doesn't redraw a different random sample -- resuming must
    see the same target set, or skip-if-exists bookkeeping is meaningless."""
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    manifest_path = os.path.join(MANIFEST_DIR, f"{source}.txt")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return [line.rstrip("\n") for line in f]
    stems = _build_sample(source, cap, seed)
    with open(manifest_path, "w") as f:
        f.write("\n".join(stems) + "\n")
    return stems


def iter_sampled_images(source, cap=SAMPLE_CAP, seed=SAMPLE_SEED, limit=None):
    """Like iter_source_images but over the fixed per-source sample
    instead of every file (see sampled_stems)."""
    exts = tuple(e.lstrip("*") for e in SOURCE_EXTENSIONS[source])
    src_dir = os.path.join(SYNTH_ROOT, source)
    stems = sampled_stems(source, cap, seed)
    if limit:
        stems = stems[:limit]
    for stem in stems:
        for ext in exts:
            path = os.path.join(src_dir, stem + ext)
            if os.path.exists(path):
                yield stem, path
                break


def outputs_exist(source, stem):
    base = os.path.join(OUT_ROOT, source, stem)
    return all(os.path.exists(base + ext) for ext in (".txt", ".mnt", ".gt"))


def write_fc(out_base, stem, letter):
    if letter is None:
        return False
    with open(out_base + ".txt", "w") as f:
        f.write("Gender: U\n")
        f.write(f"Class: {letter}\n")
        f.write(f"History: {stem} {letter} {stem}\n")
    return True


def write_fmd(out_base, minutiae):
    with open(out_base + ".mnt", "w") as f:
        f.write(f"{len(minutiae)}\n")
        for x, y, t, angle in minutiae:
            f.write(f"{x} {y} {t} {angle}\n")
    return True


def write_foe(out_base, image_path, dpi):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        return False
    h, w = image.shape
    algo, seg = _foe_algorithms()
    mask = seg.run(image)
    orientation, _strength = algo.run(image, mask, dpi=int(dpi))

    hb, wb = h // 8, w // 8
    ori_uint8 = np.zeros((hb, wb), dtype=np.uint8)
    for j in range(hb):
        for i in range(wb):
            block = orientation[j * 8:j * 8 + 8, i * 8:i * 8 + 8]
            cos2, sin2 = np.cos(2 * block).mean(), np.sin(2 * block).mean()
            ang = np.arctan2(sin2, cos2) / 2.0 % np.pi
            ori_uint8[j, i] = int(ang / np.pi * 256) & 0xFF
    ori_uint8.tofile(out_base + ".gt")
    return True


def process_image(source, stem, path, dpi):
    out_dir = os.path.join(OUT_ROOT, source)
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, stem)

    wrote = {"fc": False, "fmd": False, "foe": False}

    if not os.path.exists(out_base + ".txt"):
        r = vf.classify(path, dpi=dpi)
        wrote["fc"] = write_fc(out_base, stem, r["letter"])
        if r["letter"] is None:
            print(f"[annotate] {source}/{stem}: FC unmapped/failed (status={r['status']}, "
                  f"raw_class={r['raw_class']}), skipping .txt", flush=True)

    if not os.path.exists(out_base + ".mnt"):
        minutiae, status = vf.extract_minutiae(path, dpi=dpi)
        wrote["fmd"] = write_fmd(out_base, minutiae)

    if not os.path.exists(out_base + ".gt"):
        try:
            wrote["foe"] = write_foe(out_base, path, dpi)
        except Exception as e:
            print(f"[annotate] {source}/{stem}: FOE failed: {e}", flush=True)

    return wrote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="smoke test: 5 images from S01 only")
    ap.add_argument("--sources", default=",".join(ALL_SOURCES))
    ap.add_argument("--limit-per-source", type=int, default=None)
    ap.add_argument("--dpi", type=float, default=None,
                     help="force this DPI for every source, overriding DPI_OVERRIDES and the 500 default")
    ap.add_argument("--sampled", action="store_true",
                     help="use the capped per-source sample (SAMPLE_CAP, identity-aware for "
                          "MULTI_IMPRESSION_SOURCES) instead of every file. Builds/reuses a cached "
                          "manifest per source under _manifests/ so a restart resumes the same sample.")
    args = ap.parse_args()

    sources = ["S01"] if args.dev else args.sources.split(",")
    limit = 5 if args.dev else args.limit_per_source

    vf.init()

    total_processed = total_skipped = 0
    t0 = time.time()
    for source in sources:
        if source not in SOURCE_EXTENSIONS:
            print(f"[annotate] unknown source {source}, skipping", flush=True)
            continue
        source_dpi = args.dpi if args.dpi is not None else DPI_OVERRIDES.get(source, DEFAULT_DPI)
        image_iter = (iter_sampled_images(source, limit=limit) if args.sampled
                      else iter_source_images(source, limit))
        print(f"[annotate] === {source} (dpi={source_dpi}, sampled={args.sampled}) ===", flush=True)
        for idx, (stem, path) in enumerate(image_iter):
            if outputs_exist(source, stem):
                total_skipped += 1
                continue
            try:
                wrote = process_image(source, stem, path, source_dpi)
            except Exception as e:
                print(f"[annotate] {source}/{stem}: FAILED entirely: {e}", flush=True)
                continue
            total_processed += 1
            if total_processed % 50 == 0:
                elapsed = time.time() - t0
                print(f"[annotate] {total_processed} processed, {total_skipped} skipped "
                      f"(already done), {total_processed/elapsed:.2f} img/s", flush=True)

    print(f"[annotate] DONE: {total_processed} processed, {total_skipped} skipped, "
          f"{time.time()-t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
