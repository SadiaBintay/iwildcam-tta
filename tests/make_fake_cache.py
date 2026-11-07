"""Tiny fake cache (random images) so the WHOLE pipeline can be smoke-tested in minutes.
  python tests/make_fake_cache.py fake_cache"""
import json, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
NC, S = 10, 224
json.dump([f"species_{i}" for i in range(NC)], open(out / "class_names.json", "w"))

def make(split, n_per_cam, cams):
    y, loc = [], []
    for c in cams:
        y += list(rng.integers(0, NC, n_per_cam)); loc += [c] * n_per_cam
    y, loc = np.array(y), np.array(loc)
    tint = rng.uniform(-30, 30, (int(loc.max()) + 1, 1, 1, 3))           # camera-specific colour shift
    base = np.array([[(c * 25) % 255, (c * 60) % 255, (c * 90) % 255] for c in range(NC)])[y][:, None, None, :]
    img = np.clip(base + tint[loc] + rng.normal(0, 25, (len(y), S, S, 3)), 0, 255).astype(np.uint8)
    np.save(out / f"{split}_images.npy", img)
    np.savez(out / f"{split}_meta.npz", y=y, location=loc, wilds_idx=np.arange(len(y)))

make("train", 40, [0, 1, 2, 3])
make("id_val", 30, [0, 1]); make("id_test", 30, [2, 3])
make("val", 70, [10, 11, 12]); make("test", 70, [20, 21, 22])
print("fake cache in", out)
