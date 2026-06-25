"""
preview_augmentations.py
========================

Visual sanity check for the training augmentation pipeline.

Picks a few random clean training images, runs each through build_train_transform()
several times (the augmentations are random, so every pass differs), un-normalizes
the result back into a viewable image, and saves everything to ./aug_preview/ with
matching names so originals and variants sit side by side:

    aug_preview/00_goldfish_000043_original.jpg
    aug_preview/00_goldfish_000043_aug1.jpg
    aug_preview/00_goldfish_000043_aug2.jpg
    ...

Run (from anywhere):
    python submissions/my_team/preview_augmentations.py
    python submissions/my_team/preview_augmentations.py --num 8 --variants 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from augmentations import build_train_transform, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "dataset"
SPLIT_JSON = THIS_DIR / "split.json"
OUTPUT_DIR = PROJECT_ROOT / "aug_preview"

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
_to_pil = transforms.ToPILImage()


def unnormalize_to_pil(t: torch.Tensor) -> Image.Image:
    """Invert ImageNet normalization and convert a CHW tensor back to a PIL image."""
    img = (t * _STD + _MEAN).clamp(0.0, 1.0)
    return _to_pil(img)


def pick_random_images(num: int) -> list:
    """Random (path, stem) pairs from the clean training split."""
    split = json.loads(SPLIT_JSON.read_text())
    chosen = random.sample(split["train"], k=min(num, len(split["train"])))
    return [(DATA_ROOT / rel, Path(rel).stem) for rel, _ in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=6, help="number of source images")
    parser.add_argument("--variants", type=int, default=4, help="augmented versions per image")
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    OUTPUT_DIR.mkdir(exist_ok=True)
    transform = build_train_transform()

    images = pick_random_images(args.num)
    for path, stem in images:
        original = Image.open(path).convert("RGB")
        original.save(OUTPUT_DIR / f"{stem}_original.jpg")
        for i in range(1, args.variants + 1):
            aug_tensor = transform(original)            # random each call
            unnormalize_to_pil(aug_tensor).save(OUTPUT_DIR / f"{stem}_aug{i}.jpg")

    total = len(images) * (args.variants + 1)
    print(f"Saved {total} images ({len(images)} originals x {args.variants} variants) "
          f"-> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
