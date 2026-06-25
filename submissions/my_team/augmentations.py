"""
augmentations.py
================

Transforms for training and evaluation.

Design (Phase 2 - robustness):
------------------------------
Robustness to *unknown* corruptions comes from training on a broad, randomized,
composed *variety* of perturbations - not from matching the test's corruptions.
So the training pipeline applies many independent augmentations on-the-fly, each
with moderate probability, so the model rarely sees the same view twice and is
forced to rely on stable, semantic features.

Held-out generalization probe:
    The provided augmentations/ folder (color_jitter, random_rotation) is used as
    an *unseen* OOD probe. To keep that honest, the TRAIN pipeline deliberately
    excludes the two probe operations:
        * no rotation        (RandomAffine uses degrees=0)
        * no ColorJitter     (photometric variety comes from posterize/solarize/
                              equalize/autocontrast/grayscale instead)
    If robustness to the held-out rotation/color rises *without* training on them,
    that is evidence the approach generalizes to corruptions we never trained on.

The EVAL transform is identical to the grader's pipeline and must never change,
so local val/test numbers track the real evaluation.
"""

from __future__ import annotations

import torch
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianNoise:
    """Add zero-mean Gaussian noise to a [0,1] image tensor, then clamp to [0,1]."""

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)


def build_eval_transform() -> transforms.Compose:
    """Deterministic preprocessing, identical to the grader (val/test/inference)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transform() -> transforms.Compose:
    """
    Diverse, randomized, composed augmentation pipeline (train only).
    Excludes rotation and ColorJitter so augmentations/ stays an unseen probe.
    """
    return transforms.Compose([
        # ── geometric variety (position / scale / shape) - NO rotation ──────────
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.RandomPerspective(distortion_scale=0.3, p=1.0)], p=0.3),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10)], p=0.3),

        # ── photometric variety - NOT ColorJitter (different mechanisms) ────────
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPosterize(bits=4, p=0.2),
        transforms.RandomSolarize(threshold=128, p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomAutocontrast(p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),

        # ── quality degradations ────────────────────────────────────────────────
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),

        transforms.ToTensor(),
        transforms.RandomApply([GaussianNoise(std=0.05)], p=0.2),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),

        # ── occlusion (operates on the normalized tensor) ───────────────────────
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
