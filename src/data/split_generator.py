"""Split generator for fingerprint analysis research project."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from src.data.synthetic.synthetic_dataset import (
    ANNOTATIONS_ROOT,
    FMD_EXCLUDED_SOURCES,
    _parse_s11_class,
)
from src.utils import create_grouped_split_definition

# ── Constants ──────────────────────────────────────────────────────────────────

TASKS = ["fc", "fmd", "foe"]
SOURCES = [
    "S01", "S02", "S03", "S04", "S05", "S06",
    "S07", "S08", "S09", "S10", "S11",
]
CONDITIONS_NON_BASELINE = ["V1", "V2", "V3", "R1", "R2", "R3", "R4"]

SYNTHETIC_N: dict[str, dict[str, int]] = {
    "fc":  {"V1": 3200,  "V2": 9600,  "V3": 22400,
            "R1": 800,   "R2": 1600,  "R3": 2400,  "R4": 3200},
    "fmd": {"V1": 2560,  "V2": 7680,  "V3": 17920,
            "R1": 640,   "R2": 1280,  "R3": 1920,  "R4": 2560},
    "foe": {"V1": 48,    "V2": 144,   "V3": 336,
            "R1": 12,    "R2": 24,    "R3": 36,    "R4": 48},
}

SYNTHETIC_SOURCES: dict[str, dict[str, int]] = {
    "S01": {"n_fingers": 480,    "total_images": 4440},
    "S02": {"n_fingers": 10000,  "total_images": 1099480},
    "S03": {"n_fingers": 25000,  "total_images": 25000},
    "S04": {"n_fingers": 50048,  "total_images": 50048},
    "S05": {"n_fingers": 34992,  "total_images": 524895},
    "S06": {"n_fingers": 100000, "total_images": 200000},
    "S07": {"n_fingers": 50000,  "total_images": 50000},
    "S08": {"n_fingers": 4000,   "total_images": 40000},
    "S09": {"n_fingers": 4000,   "total_images": 40000},
    "S10": {"n_fingers": 25000,  "total_images": 25000},
    "S11": {"n_fingers": 12500,  "total_images": 100000},
}

# S02 FOE stratified sampling: (n_bad, n_good) per condition
FOE_S02_COUNTS: dict[str, tuple[int, int]] = {
    "V1": (40, 8),
    "V2": (120, 24),
    "V3": (280, 56),
    "R1": (10, 2),
    "R2": (20, 4),
    "R3": (30, 6),
    "R4": (40, 8),
}

SEED = 42


# ── File scanning ──────────────────────────────────────────────────────────────

def _is_image_file(name: str, source: str) -> bool:
    """Return True if the file should be counted as an image for this source."""
    ext = Path(name).suffix.lower()
    if source == "S01":
        return ext in (".tif", ".bmp")
    if source == "S11":
        return ext == ".png"
    return True  # All other sources: every file in the folder is an image


def _scan_source_folder(
    source_dir: Path, source: str
) -> dict[str, list[str]]:
    """
    Scan source directory with os.scandir() for performance.
    Returns {fp_id: [filename, ...]} for all image files.
    FP_id = everything before '--' in the filename stem.
    """
    finger_to_files: dict[str, list[str]] = {}
    if not source_dir.exists():
        return finger_to_files
    for entry in os.scandir(source_dir):
        if not entry.is_file():
            continue
        name = entry.name
        if not _is_image_file(name, source):
            continue
        stem = Path(name).stem
        if "--" not in stem:
            continue
        fp_id = stem.split("--", 1)[0]
        finger_to_files.setdefault(fp_id, []).append(name)
    return finger_to_files


def _scan_annotated_stems(source: str, ext: str) -> set[str]:
    """
    Scan annotation_workspace/annotations/{source}/ once for every file with
    the given extension, returning their stems as a set. Used to constrain
    candidate images to those the annotation pipeline actually labelled --
    the pipeline covered a capped subset per source (e.g. ~30k of S02's 1.1M
    images), so sampling from the full raw pool without this filter silently
    picks mostly-unlabelled images (see CONTEXT.md's "split/annotation
    sampling mismatch" finding).
    """
    ann_dir = ANNOTATIONS_ROOT / source
    stems: set[str] = set()
    if not ann_dir.exists():
        return stems
    for entry in os.scandir(ann_dir):
        if entry.is_file() and entry.name.endswith(ext):
            stems.add(entry.name[: -len(ext)])
    return stems


def _filter_annotated(
    finger_to_files: dict[str, list[str]],
    source: str,
    task: str,
    source_dir: Path,
) -> dict[str, list[str]]:
    """
    Constrain a source's finger->files index to only files that would
    actually produce a non-None label for `task`, mirroring
    SyntheticDataset's label-availability logic exactly (synthetic_dataset.py)
    so a split never selects an image whose annotation doesn't exist.
    """
    if task == "fmd" and source in FMD_EXCLUDED_SOURCES:
        return {}

    if source == "S11":
        if task == "fc":
            # Native filename-encoded class -- no annotation file needed.
            return {
                fp_id: files for fp_id, files in finger_to_files.items()
                if _parse_s11_class(fp_id) is not None
            }
        if task == "foe":
            # Native per-image _o.txt alongside the raw image, not the
            # annotation-pipeline output.
            filtered: dict[str, list[str]] = {}
            for fp_id, files in finger_to_files.items():
                kept = [
                    f for f in files
                    if (source_dir / f"{Path(f).stem}_o.txt").exists()
                ]
                if kept:
                    filtered[fp_id] = kept
            return filtered
        # task == "fmd": S11 already excluded above; fall through to the
        # generic annotation-manifest check (unreachable in practice).

    ext = {"fc": ".txt", "fmd": ".mnt", "foe": ".gt"}[task]
    annotated_stems = _scan_annotated_stems(source, ext)
    filtered: dict[str, list[str]] = {}
    for fp_id, files in finger_to_files.items():
        kept = [f for f in files if Path(f).stem in annotated_stems]
        if kept:
            filtered[fp_id] = kept
    return filtered


def _get_quality_s02(filename: str) -> str:
    """Parse quality label ('B' or 'G') from an S02 filename.
    e.g. '10000--Bad_1.jpg' → 'B', '9999--Good_3.jpg' → 'G'.
    """
    imp_id = Path(filename).stem.split("--", 1)[1]  # "Bad_1" or "Good_3"
    return "B" if imp_id.split("_")[0].lower() == "bad" else "G"


# ── Sampling ───────────────────────────────────────────────────────────────────

def _groups_to_filenames(
    group_to_files: dict[str, list[str]], group_ids: list[str]
) -> list[str]:
    """Return filenames in group_ids order; files within each group are sorted."""
    result: list[str] = []
    for gid in group_ids:
        result.extend(sorted(group_to_files.get(gid, [])))
    return result


def _sample_standard(
    finger_to_files: dict[str, list[str]],
    target: int,
    seed: int = SEED,
) -> tuple[list[str], bool]:
    """
    Sample `target` files maximising finger diversity.
    Returns (selected_files, shortfall).

    Case A — target <= n_fingers:
        Sample target fingers randomly, 1 impression each.
    Case B — n_fingers < target <= total_images:
        Use all fingers (1 impression each), then distribute extras
        evenly via round-robin through sorted finger IDs.
    Case C — target > total_images:
        Return all available files; shortfall=True.
    """
    if not finger_to_files:
        return [], True

    n_fingers = len(finger_to_files)
    total_images = sum(len(v) for v in finger_to_files.values())
    rng = random.Random(seed)

    # Case C
    if target > total_images:
        all_files = sorted(f for files in finger_to_files.values() for f in files)
        return all_files, True

    fingers = sorted(finger_to_files.keys())
    rng.shuffle(fingers)

    # Case A
    if target <= n_fingers:
        result = [rng.choice(finger_to_files[fp_id]) for fp_id in fingers[:target]]
        return result, False

    # Case B
    selected: dict[str, list[str]] = {}
    for fp_id in fingers:
        chosen = rng.choice(finger_to_files[fp_id])
        selected[fp_id] = [chosen]

    remaining = target - n_fingers
    available_extra: dict[str, list[str]] = {
        fp_id: [f for f in finger_to_files[fp_id] if f != selected[fp_id][0]]
        for fp_id in fingers
    }

    # Round-robin through sorted finger IDs
    sorted_fingers = sorted(fingers)
    total_added = 0
    while total_added < remaining:
        any_added = False
        for fp_id in sorted_fingers:
            if total_added >= remaining:
                break
            extras = available_extra[fp_id]
            if extras:
                extra = rng.choice(extras)
                selected[fp_id].append(extra)
                extras.remove(extra)
                total_added += 1
                any_added = True
        if not any_added:
            break  # No more extras (should not happen in Case B)

    result = [f for fp_id in fingers for f in selected[fp_id]]
    return result, False


def _sample_s02_foe(
    finger_to_files: dict[str, list[str]],
    n_bad: int,
    n_good: int,
    seed: int = SEED,
) -> tuple[list[str], bool]:
    """
    Sample from S02 for FOE task with stratified Bad/Good quality.
    First n_bad fingers → 1 Bad impression each.
    Next n_good fingers → 1 Good impression each.
    Returns (selected_files, shortfall).
    """
    target = n_bad + n_good
    rng = random.Random(seed)
    all_fingers = sorted(finger_to_files.keys())
    rng.shuffle(all_fingers)

    bad_fingers = all_fingers[:n_bad]
    good_fingers = all_fingers[n_bad: n_bad + n_good]

    result: list[str] = []
    for fp_id in bad_fingers:
        bad_files = [f for f in finger_to_files[fp_id] if _get_quality_s02(f) == "B"]
        if bad_files:
            result.append(rng.choice(bad_files))
    for fp_id in good_fingers:
        good_files = [f for f in finger_to_files[fp_id] if _get_quality_s02(f) == "G"]
        if good_files:
            result.append(rng.choice(good_files))

    return result, len(result) < target


def _sample_synthetic(
    finger_to_files: dict[str, list[str]],
    target: int,
    source: str,
    task: str,
    condition: str,
    seed: int = SEED,
) -> tuple[list[str], bool]:
    """Dispatch to stratified (S02+FOE) or standard sampling."""
    if source == "S02" and task == "foe":
        n_bad, n_good = FOE_S02_COUNTS[condition]
        return _sample_s02_foe(finger_to_files, n_bad, n_good, seed)
    return _sample_standard(finger_to_files, target, seed)


# ── Real dataset scanners ──────────────────────────────────────────────────────

def _scan_real_fc(data_root: Path) -> dict[str, list[str]]:
    """Scan NIST_SD4/png_txt/figs_0/ .. figs_7/. Returns {group_id: [filenames]}.
    f0001_01.png and s0001_01.png → group "0001_01".
    """
    base_dir = data_root / "real" / "NIST_SD4" / "png_txt"
    group_to_files: dict[str, list[str]] = {}
    if not base_dir.exists():
        print(f"  WARNING: {base_dir} not found — skipping FC real split")
        return group_to_files
    for n in range(8):
        subdir = base_dir / f"figs_{n}"
        if not subdir.exists():
            continue
        for entry in os.scandir(subdir):
            if not entry.is_file():
                continue
            name = entry.name
            if not name.lower().endswith(".png"):
                continue
            stem = Path(name).stem  # e.g. "f0001_01"
            group_id = stem[1:]     # strip leading f/s → "0001_01"
            group_to_files.setdefault(group_id, []).append(name)
    return group_to_files


FMD_SUBFOLDERS = [
    "FM3_FVC2002DB1A_np",
    "FM3_FVC2002DB3A_np",
    "FM3_FVC2004DB1A_np",
    "FM3_FVC2004DB3A_np",
]


def _scan_real_fmd(data_root: Path) -> dict[str, list[str]]:
    """Scan FVC_BERKAY_TOPCU across 4 subfolders. Returns {group_id: [filenames]}.
    Finger 100 in FM3_FVC2002DB1A_np → group "FM3_FVC2002DB1A_np_0100".
    Filenames stored as '{subfolder}/{filename}' to avoid basename collisions.
    """
    base_dir = data_root / "real" / "FVC_BERKAY_TOPCU"
    group_to_files: dict[str, list[str]] = {}
    if not base_dir.exists():
        print(f"  WARNING: {base_dir} not found — skipping FMD real split")
        return group_to_files
    for subfolder in FMD_SUBFOLDERS:
        subdir = base_dir / subfolder
        if not subdir.exists():
            continue
        for entry in os.scandir(subdir):
            if not entry.is_file():
                continue
            name = entry.name
            if not name.lower().endswith(".tif"):
                continue
            stem = Path(name).stem          # e.g. "100_1"
            finger = stem.split("_")[0]     # "100"
            group_id = f"{subfolder}_{finger.zfill(4)}"  # "FM3_FVC2002DB1A_np_0100"
            rel = f"{subfolder}/{name}"     # preserve subfolder in stored filename
            group_to_files.setdefault(group_id, []).append(rel)
    return group_to_files


def _scan_real_foe(
    data_root: Path,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Scan FOESamples/Bad/ and /Good/. Returns (group_to_files, group_labels).
    Only .bmp files are counted as images.
    Filenames stored as 'Bad/filename.bmp' to preserve subfolder information.
    """
    base_dir = data_root / "real" / "FOESamples"
    group_to_files: dict[str, list[str]] = {}
    group_labels: dict[str, str] = {}

    if not base_dir.exists():
        print(f"  WARNING: {base_dir} not found — skipping FOE real split")
        return group_to_files, group_labels

    for subdir_name, quality_code in [("Bad", "B"), ("Good", "G")]:
        subdir = base_dir / subdir_name
        if not subdir.exists():
            continue
        for entry in os.scandir(subdir):
            if not entry.is_file():
                continue
            name = entry.name
            if not name.lower().endswith(".bmp"):
                continue
            stem = Path(name).stem
            rel = f"{subdir_name}/{name}"  # "Bad/00.bmp" or "Good/110.bmp"
            group_to_files[stem] = [rel]
            group_labels[stem] = quality_code

    return group_to_files, group_labels


# ── Real split writer ──────────────────────────────────────────────────────────

def _write_real_split_json(
    out_path: Path,
    task: str,
    group_to_files: dict[str, list[str]],
    group_labels: dict[str, str] | None = None,
) -> bool:
    """Build and write a real split JSON. Returns True if written, False if skipped."""
    if out_path.exists():
        return False

    unique_groups = sorted(group_to_files.keys())
    split_def = create_grouped_split_definition(
        group_ids=unique_groups,
        num_folds=4,
        test_size=0.2,
        seed=SEED,
        group_labels=group_labels,
    )

    test_group_set = set(split_def.test_groups)
    train_val_groups = [g for g in unique_groups if g not in test_group_set]

    all_ids = _groups_to_filenames(group_to_files, unique_groups)
    train_val_ids = _groups_to_filenames(group_to_files, train_val_groups)
    test_ids = _groups_to_filenames(group_to_files, split_def.test_groups)
    rng = random.Random(SEED)
    def _shuffled(groups: list[str]) -> list[str]:
        lst = list(groups)
        rng.shuffle(lst)
        return lst

    cv_folds = [
        {
            "train": _groups_to_filenames(group_to_files, _shuffled(fold.train_groups)),
            "val": _groups_to_filenames(group_to_files, _shuffled(fold.val_groups)),
        }
        for fold in split_def.folds
    ]

    payload = {
        "task": task,
        "source": "real",
        "condition": None,
        "real_n": len(train_val_ids),
        "synthetic_n": 0,
        "sampling_strategy": "finger_level",
        "random_state": SEED,
        "shortfall": False,
        "all_ids": all_ids,
        "train_val_ids": train_val_ids,
        "test_ids": test_ids,
        "cv_folds": cv_folds,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_splits(project_root: str = ".", data_root: str = "/data") -> None:
    root = Path(project_root)
    droot = Path(data_root)
    splits_dir = root / "splits"

    real_gen = real_skip = syn_gen = syn_skip = 0

    # ── Real splits ────────────────────────────────────────────────────────────
    print("Generating real splits...")

    for task, scanner in [
        ("fc",  lambda: (_scan_real_fc(droot), None)),
        ("fmd", lambda: (_scan_real_fmd(droot), None)),
        ("foe", lambda: _scan_real_foe(droot)),
    ]:
        group_to_files, group_labels = scanner()

        out_path = splits_dir / task / "real_split.json"
        if not group_to_files:
            real_skip += 1
            continue
        ok = _write_real_split_json(out_path, task, group_to_files, group_labels)
        if ok:
            real_gen += 1
            print(f"  Written:  {out_path}")
        else:
            real_skip += 1
            print(f"  Skipped:  {out_path}")

    # ── Synthetic splits ───────────────────────────────────────────────────────
    print("Generating synthetic splits...")

    # Cache finger index per source to avoid re-scanning for each task/condition
    source_cache: dict[str, dict[str, list[str]]] = {}
    # Cache annotation-filtered index per (source, task) -- filtering is
    # task-specific (different label file per task) but shared across all
    # 7 conditions for that (source, task) pair.
    filtered_cache: dict[tuple[str, str], dict[str, list[str]]] = {}

    for source in SOURCES:
        source_dir = droot / "synthetic" / source
        if source not in source_cache:
            print(f"  Scanning {source_dir} ...")
            source_cache[source] = _scan_source_folder(source_dir, source)
        finger_to_files_raw = source_cache[source]

        for task in TASKS:
            cache_key = (source, task)
            if cache_key not in filtered_cache:
                filtered_cache[cache_key] = _filter_annotated(
                    finger_to_files_raw, source, task, source_dir
                )
            finger_to_files = filtered_cache[cache_key]

            for condition in CONDITIONS_NON_BASELINE:
                out_path = (
                    splits_dir / task / "synthetic" / f"{source}_{condition}.json"
                )
                if out_path.exists():
                    syn_skip += 1
                    continue

                target = SYNTHETIC_N[task][condition]
                selected_files, shortfall = _sample_synthetic(
                    finger_to_files, target, source, task, condition, seed=SEED
                )
                actual_count = len(selected_files)

                payload: dict = {
                    "task": task,
                    "source": source,
                    "condition": condition,
                    "synthetic_n": target,
                    "sampling_strategy": "finger_level",
                    "random_state": SEED,
                    "shortfall": shortfall,
                    "actual_count": actual_count,
                    "train_ids": selected_files,
                }

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                syn_gen += 1

    total = real_gen + syn_gen
    print(f"\nReal splits:      {real_gen} generated / {real_skip} skipped")
    print(f"Synthetic splits: {syn_gen} generated / {syn_skip} skipped")
    print(f"Total:            {total} files written")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split JSON files."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root (splits/ and configs/ written here)",
    )
    parser.add_argument(
        "--data-root",
        default="/data",
        help="Dataset root (where real/ and synthetic/ live, default: /data)",
    )
    args = parser.parse_args()
    generate_splits(project_root=args.root, data_root=args.data_root)


if __name__ == "__main__":
    main()
