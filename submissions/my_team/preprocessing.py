"""
preprocessing.py
================

Builds a reproducible train/validation/test split for the 20-class dataset.

Strategy: augmentation-aware stratified 80/10/10 split.
---------------------------------------------------------------------------
The provided ``augmentations/`` folder contains manipulated versions of *specific*
clean training images (e.g. ``color_jitter/goldfish/00_goldfish_000043_color_jitter.jpg``
is derived from ``train/goldfish/00_goldfish_000043.jpg``).

The hidden test set's out-of-distribution half is made of manipulations of images
the model has NEVER seen during training. To make ``augmentations/`` an *honest*
local OOD proxy, we force the clean source images of every augmented sample out of
the TRAIN split, so the model never trains on them.

Three splits, stratified per class (800 / 100 / 100 per class):
  * train (80%) - used to fit the model.
  * val   (10%) - used often: tuning, early stopping, model selection.
  * test  (10%) - a LOCK-BOX. Touched ONCE, at the very end, to get an honest
    final estimate. Never tune against it, or it silently becomes a second val
    set and stops guarding against adaptive (experimenter) overfitting.
Splitting the held-out 20% into val+test costs zero training data.

Outputs
-------
1. ``split.json``               (next to this file) - the source of truth.
   Maps every image to "train" / "val" / "test". Used by train.py. The ~16k
   training images are never copied.
2. ``dataset/validation/<class>/`` - the val images are *copied* here so the
   provided ``evaluate.py`` (which reads ``dataset/validation/``) runs as-is.
3. ``dataset/test/<class>/``       - the lock-box test images, for a one-off
   final evaluation near submission time.

Run from anywhere:
    python submissions/my_team/preprocessing.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

# ── locate project root so labels.py is importable regardless of cwd ───────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labels import HF_INDEX_TO_NAME, HF_INDEX_TO_IDX, TARGET_HF_INDICES  # noqa: E402

# ── configuration ──────────────────────────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_ROOT / "train"
AUG_DIR = DATA_ROOT / "augmentations"
VALIDATION_DIR = DATA_ROOT / "validation"     # materialized for evaluate.py
TEST_DIR = DATA_ROOT / "test"                 # lock-box: touched once, at the very end
SPLIT_JSON = Path(__file__).resolve().parent / "split.json"

VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42
IMAGE_EXTS = (".jpg", ".jpeg", ".JPEG", ".png")
# ───────────────────────────────────────────────────────────────────────────────


def list_images(folder: Path) -> list[Path]:
    """All image files directly inside ``folder`` (sorted, deterministic)."""
    imgs: list[Path] = []
    for ext in IMAGE_EXTS:
        imgs.extend(folder.glob(f"*{ext}"))
    return sorted(set(imgs))


def augmentation_source_stems(class_name: str) -> set[str]:
    """
    Return the stems of clean train images that were used to create augmented
    samples for ``class_name``.

    An augmented file is named ``<source_stem>_<augtype>.<ext>`` where <augtype>
    matches the augmentation subfolder name (e.g. ``color_jitter``). We strip that
    suffix to recover the clean source stem (e.g. ``00_goldfish_000043``).
    """
    sources: set[str] = set()
    if not AUG_DIR.exists():
        return sources

    for aug_type_dir in sorted(p for p in AUG_DIR.iterdir() if p.is_dir()):
        suffix = f"_{aug_type_dir.name}"          # e.g. "_color_jitter"
        class_dir = aug_type_dir / class_name
        if not class_dir.exists():
            continue
        for aug_img in list_images(class_dir):
            stem = aug_img.stem                    # "00_goldfish_000043_color_jitter"
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]        # "00_goldfish_000043"
            sources.add(stem)
    return sources


def build_split() -> dict:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")

    rng = random.Random(SEED)

    train_entries: list[tuple[str, int]] = []
    val_entries: list[tuple[str, int]] = []
    test_entries: list[tuple[str, int]] = []
    per_class_stats: dict[str, dict] = {}

    for hf_idx in sorted(TARGET_HF_INDICES):
        class_name = HF_INDEX_TO_NAME[hf_idx]
        local_idx = HF_INDEX_TO_IDX[hf_idx]
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        images = list_images(class_dir)
        n_total = len(images)
        n_val = round(VAL_RATIO * n_total)
        n_test = round(TEST_RATIO * n_total)
        n_held = n_val + n_test                      # images kept out of train

        # Augmentation-source images must never appear in train; they go into the
        # held-out pool and are then distributed across val/test.
        forced_stems = augmentation_source_stems(class_name)
        forced = [p for p in images if p.stem in forced_stems]
        forced_set = set(forced)

        # Fill the rest of the held-out quota with a random stratified sample.
        pool = [p for p in images if p not in forced_set]
        rng.shuffle(pool)
        n_random_needed = max(0, n_held - len(forced))
        held = forced + pool[:n_random_needed]

        # Shuffle the held-out images, then split them into val and test so the
        # forced sources are spread across both.
        rng.shuffle(held)
        val_imgs = held[:n_val]
        test_imgs = held[n_val:n_val + n_test]

        held_set = set(val_imgs) | set(test_imgs)
        train_imgs = [p for p in images if p not in held_set]

        for p in train_imgs:
            train_entries.append((p.relative_to(DATA_ROOT).as_posix(), local_idx))
        for p in val_imgs:
            val_entries.append((p.relative_to(DATA_ROOT).as_posix(), local_idx))
        for p in test_imgs:
            test_entries.append((p.relative_to(DATA_ROOT).as_posix(), local_idx))

        per_class_stats[class_name] = {
            "total": n_total,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
            "forced_sources": len(forced),
        }

    split = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "data_root": "dataset",
        "classes": {HF_INDEX_TO_NAME[h]: HF_INDEX_TO_IDX[h] for h in sorted(TARGET_HF_INDICES)},
        "counts": {
            "train": len(train_entries),
            "val": len(val_entries),
            "test": len(test_entries),
            "total": len(train_entries) + len(val_entries) + len(test_entries),
        },
        "per_class": per_class_stats,
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }
    return split


def materialize_folder(split: dict, split_key: str, dest_dir: Path) -> None:
    """
    Copy the images of one split into ``dest_dir/<class>/`` so folder-based tools
    (e.g. the provided evaluate.py) run unchanged. Idempotent: rebuilt each run.
    """
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    idx_to_name = {v: k for k, v in split["classes"].items()}
    copied = 0
    for rel_path, local_idx in split[split_key]:
        src = DATA_ROOT / rel_path
        dst_dir = dest_dir / idx_to_name[local_idx]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        copied += 1
    print(f"Materialized {copied} {split_key} images -> {dest_dir}")


def main() -> None:
    print("Building augmentation-aware stratified split...")
    split = build_split()

    SPLIT_JSON.write_text(json.dumps(split, indent=2))
    print(f"Wrote split manifest -> {SPLIT_JSON}")

    materialize_folder(split, "val", VALIDATION_DIR)
    materialize_folder(split, "test", TEST_DIR)

    c = split["counts"]
    total_forced = sum(s["forced_sources"] for s in split["per_class"].values())
    print("\n--- Split summary ---")
    print(f"  classes        : {len(split['classes'])}")
    print(f"  total images   : {c['total']}")
    print(f"  train images   : {c['train']}  ({1 - split['val_ratio'] - split['test_ratio']:.0%})")
    print(f"  val images     : {c['val']}  ({split['val_ratio']:.0%})")
    print(f"  test images    : {c['test']}  ({split['test_ratio']:.0%})  [lock-box]")
    print(f"  forced sources : {total_forced}  (augmentation sources held out of train)")
    print(f"  seed           : {split['seed']}")


if __name__ == "__main__":
    main()
