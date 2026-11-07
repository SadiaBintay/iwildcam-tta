"""Shared helpers: cache reader, normalisation, seeding."""
import json
import random
from pathlib import Path

import numpy as np
import torch

MEAN = (0.485, 0.456, 0.406)   # ImageNet stats (DINO uses the same)
STD = (0.229, 0.224, 0.225)
MIN_EVAL = 20                   # a camera needs at least n + MIN_EVAL images to be used


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prep(u8, dev, augment: bool = False):
    """uint8 images (B,H,W,3) (ndarray or tensor) -> normalised float (B,3,H,W) on dev.
    augment=True: random horizontal flip + brightness jitter (train only)."""
    x = torch.as_tensor(u8).to(dev, non_blocking=True).float()
    if augment:
        b = x.shape[0]
        flip = torch.rand(b, device=dev) < 0.5
        x = torch.where(flip.view(b, 1, 1, 1), x.flip(2), x)
        x = (x * (0.8 + 0.4 * torch.rand(b, 1, 1, 1, device=dev))).clamp(0, 255)
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1) * 255
    return (x - mean) / std


class Cache:
    """Reads one split of the cached subset: <split>_images.npy (uint8 N,224,224,3) + <split>_meta.npz."""

    def __init__(self, cache_dir, split, ram: bool = False):
        d = Path(cache_dir)
        m = np.load(d / f"{split}_meta.npz")
        self.split = split
        self.y = m["y"].astype(np.int64)
        self.loc = m["location"].astype(np.int64)
        self.images = np.load(d / f"{split}_images.npy", mmap_mode="r")
        if ram:
            self.images = np.array(self.images)

    def __len__(self):
        return len(self.y)

    def get(self, ids):
        return self.images[np.asarray(ids)]

    def camera_groups(self):
        return {int(c): np.where(self.loc == c)[0] for c in np.unique(self.loc)}


def num_classes(cache_dir) -> int:
    return len(json.load(open(Path(cache_dir) / "class_names.json")))


def class_names(cache_dir):
    return json.load(open(Path(cache_dir) / "class_names.json"))
