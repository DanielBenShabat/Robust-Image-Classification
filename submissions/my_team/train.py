#!/usr/bin/env python3
"""
train.py
========

Trains ModelArchitecture (from model.py) on the clean training split defined by
``split.json`` (produced by preprocessing.py) and saves the learned weights to
``weights.joblib`` as required by the grader.

Phase 1 (baseline): standard classification only. No robustness augmentations yet.
Training/validation use the same deterministic transform as the grader:
    Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet-normalize

Run (from anywhere):
    python submissions/my_team/train.py                 # full baseline run
    python submissions/my_team/train.py --quick         # fast smoke test
    python submissions/my_team/train.py --epochs 10 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── paths (resolved from this file, so cwd does not matter) ────────────────────
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ModelArchitecture  # noqa: E402  (this file's dir is on sys.path)

DATA_ROOT = PROJECT_ROOT / "dataset"
SPLIT_JSON = THIS_DIR / "split.json"
OUTPUT = THIS_DIR / "weights.joblib"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── dataset that reads the split.json manifest ─────────────────────────────────
class ManifestDataset(Dataset):
    """Loads (image, label) pairs from a list of (relative_path, label) entries."""

    def __init__(self, entries: list, data_root: Path, transform=None):
        self.entries = entries
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        rel_path, label = self.entries[idx]
        image = Image.open(self.data_root / rel_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transform() -> transforms.Compose:
    """Deterministic preprocessing, identical to the grader's pipeline."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def subsample_per_class(entries: list, per_class: int) -> list:
    """Keep at most ``per_class`` entries per label (for fast smoke tests)."""
    buckets: dict = defaultdict(list)
    for e in entries:
        buckets[e[1]].append(e)
    out = []
    for label in sorted(buckets):
        out.extend(buckets[label][:per_class])
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="Train on a small subset for a fast pipeline smoke test.")
    args = parser.parse_args()

    if not SPLIT_JSON.exists():
        raise FileNotFoundError(
            f"{SPLIT_JSON} not found. Run preprocessing.py first.")

    split = json.loads(SPLIT_JSON.read_text())
    train_entries = split["train"]
    val_entries = split["val"]

    if args.quick:
        train_entries = subsample_per_class(train_entries, 40)
        val_entries = subsample_per_class(val_entries, 10)
        args.epochs = min(args.epochs, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transform()

    train_ds = ManifestDataset(train_entries, DATA_ROOT, transform)
    val_ds = ManifestDataset(val_entries, DATA_ROOT, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=False)

    print(f"Device: {device} | train: {len(train_ds)} | val: {len(val_ds)} "
          f"| epochs: {args.epochs} | batch: {args.batch_size}")

    model = ModelArchitecture(num_classes=20).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)
        scheduler.step()

        train_loss = running_loss / max(seen, 1)
        val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}/{args.epochs} | loss {train_loss:.4f} "
              f"| val_acc {val_acc:.4f} | {dt:.1f}s")

        if val_acc >= best_acc:
            best_acc = val_acc
            state_dict = model.cpu().state_dict()   # save best, on CPU
            joblib.dump(state_dict, OUTPUT)
            model.to(device)

    # Guarantee a weights file exists even if val_acc never improved.
    if not OUTPUT.exists():
        joblib.dump(model.cpu().state_dict(), OUTPUT)

    print(f"\nBest val_acc: {best_acc:.4f}")
    print(f"Saved weights -> {OUTPUT}")


if __name__ == "__main__":
    main()
