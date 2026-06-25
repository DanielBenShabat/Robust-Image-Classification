"""
eval_robustness.py
==================

Robustness dashboard for a trained model. Reports, side by side:

  * CLEAN accuracy   - on the held-out clean validation split (in-distribution).
                       Mirrors the test set's in-domain half.
  * OOD accuracy     - on the provided augmentations/ folder (held-out, UNSEEN).
                       Mirrors the test set's out-of-domain half. Broken down per
                       augmentation type so we can see where robustness is weak.
  * COMBINED         - 0.5*clean + 0.5*ood, mirroring the final 50/50 score.

The augmentations/ images are an HONEST generalization probe: their clean source
images were forced out of training (preprocessing.py), and the train pipeline
never uses rotation or ColorJitter. So a high OOD number here is evidence the
model generalizes to corruptions it never trained on - the real-test condition.

Run (from anywhere):
    python submissions/my_team/eval_robustness.py
    python submissions/my_team/eval_robustness.py --weights path/to/weights.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ModelArchitecture                       # noqa: E402
from augmentations import build_eval_transform            # noqa: E402
from labels import HF_INDEX_TO_NAME, HF_INDEX_TO_IDX, TARGET_HF_INDICES  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "dataset"
AUG_DIR = DATA_ROOT / "augmentations"
SPLIT_JSON = THIS_DIR / "split.json"
DEFAULT_WEIGHTS = THIS_DIR / "weights.joblib"
BATCH_SIZE = 64

NAME_TO_IDX = {HF_INDEX_TO_NAME[h]: HF_INDEX_TO_IDX[h] for h in TARGET_HF_INDICES}


class ImageListDataset(Dataset):
    """Loads (image, label) pairs from a list of (absolute_path, label) entries."""

    def __init__(self, entries: list, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        path, label = self.entries[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def clean_val_entries() -> list:
    """Clean validation images from the split manifest (absolute paths)."""
    split = json.loads(SPLIT_JSON.read_text())
    return [(DATA_ROOT / rel, label) for rel, label in split["val"]]


def ood_entries_by_type() -> dict:
    """{aug_type: [(path, label), ...]} for every augmentation folder."""
    out: dict = {}
    for aug_type_dir in sorted(p for p in AUG_DIR.iterdir() if p.is_dir()):
        entries = []
        for class_dir in sorted(p for p in aug_type_dir.iterdir() if p.is_dir()):
            label = NAME_TO_IDX[class_dir.name]
            for img in sorted(class_dir.glob("*.jpg")):
                entries.append((img, label))
        out[aug_type_dir.name] = entries
    return out


def load_model(weights_path: Path, device: torch.device) -> ModelArchitecture:
    model = ModelArchitecture(num_classes=20)
    model.load_state_dict(joblib.load(weights_path))
    return model.to(device).eval()


@torch.no_grad()
def accuracy(model, entries, transform, device) -> float:
    if not entries:
        return float("nan")
    loader = DataLoader(ImageListDataset(entries, transform),
                        batch_size=BATCH_SIZE, shuffle=False)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_eval_transform()
    model = load_model(args.weights, device)
    print(f"Loaded {args.weights.name} | device: {device}\n")

    # In-distribution.
    clean = clean_val_entries()
    clean_acc = accuracy(model, clean, transform, device)

    # Out-of-distribution (held-out augmentations), per type and overall.
    ood = ood_entries_by_type()
    print("--- Robustness report ---")
    print(f"  CLEAN  (val, {len(clean)} imgs){'':<6} acc: {clean_acc:.4f}")

    all_ood = []
    for aug_type, entries in ood.items():
        acc = accuracy(model, entries, transform, device)
        all_ood.extend(entries)
        print(f"  OOD    [{aug_type:<16}] ({len(entries):>4} imgs) acc: {acc:.4f}")

    ood_acc = accuracy(model, all_ood, transform, device)
    combined = 0.5 * clean_acc + 0.5 * ood_acc
    print(f"  OOD    (overall, {len(all_ood)} imgs){'':<3} acc: {ood_acc:.4f}")
    print(f"  ------------------------------------------")
    print(f"  COMBINED (0.5*clean + 0.5*ood){'':<6} {combined:.4f}")
    print(f"  GAP (clean - ood){'':<19} {clean_acc - ood_acc:+.4f}")


if __name__ == "__main__":
    main()
