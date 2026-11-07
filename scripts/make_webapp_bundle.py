"""Assemble the deployable web app folder (code + checkpoints + class names + example photos).
  python scripts/make_webapp_bundle.py                 # uses ./data, writes ./build/webapp
Expected: data/ckpt/{resnet50,dino_vits16}.pt, data/cache/class_names.json
Optional (for the 'Load example camera' button): data/cache/test_images.npy + test_meta.npz"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "build" / "webapp"))
    ap.add_argument("--cameras", type=int, nargs="+", default=[58], help="unseen test cameras to include as examples")
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out)

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "webapp", out, ignore=shutil.ignore_patterns("__pycache__"))
    (out / "src").mkdir()
    for f in ["__init__.py", "common.py", "models.py", "adapt.py"]:
        shutil.copy(ROOT / "src" / f, out / "src" / f)
    (out / "ckpt").mkdir()
    for bb in ["resnet50", "dino_vits16"]:
        ck = data / "ckpt" / f"{bb}.pt"
        if not ck.exists():
            sys.exit(f"missing checkpoint: {ck}")
        shutil.copy(ck, out / "ckpt" / f"{bb}.pt")
    names = data / "cache" / "class_names.json"
    if not names.exists():
        sys.exit(f"missing {names}")
    shutil.copy(names, out / "class_names.json")
    class_names = json.load(open(names))

    meta, imgs = data / "cache" / "test_meta.npz", data / "cache" / "test_images.npy"
    if meta.exists() and imgs.exists():
        m = np.load(meta)
        y, loc = m["y"], m["location"]
        arr = np.load(imgs, mmap_mode="r")
        for cam in a.cameras:
            pool = np.where(loc == cam)[0]
            if len(pool) == 0:
                print(f"camera {cam} not in the test split - skipped")
                continue
            ids = np.random.default_rng(1).choice(pool, min(a.n, len(pool)), replace=False)
            d = out / "examples" / f"camera_{cam}"
            d.mkdir(parents=True)
            rows = []
            for i in ids:
                fn = f"cam{cam}_{int(i)}.jpg"
                Image.fromarray(np.asarray(arr[i])).save(d / fn, quality=92)
                rows.append([fn, class_names[int(y[i])]])
            with open(d / "true_labels.csv", "w", newline="") as f:
                csv.writer(f).writerows(rows)
            print(f"examples: {len(rows)} photos from camera {cam}")
    else:
        print("no test cache found: the app will work, but without example cameras")
    print("bundle ready:", out)


if __name__ == "__main__":
    main()
