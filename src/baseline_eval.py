"""No-adaptation baseline: seen cameras (id_test) vs unseen cameras (test)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Cache, get_device, num_classes
from .metrics import accuracy, macro_f1, predict
from .models import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--backbones", nargs="+", default=["resnet50", "dino_vits16"])
    a = ap.parse_args()
    dev = get_device()
    nc = num_classes(a.cache_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, rows = {}, []
    for bb in a.backbones:
        model = load_checkpoint(bb, Path(a.ckpt_dir) / f"{bb}.pt", nc, dev)
        res[bb] = {}
        for split in ("id_test", "test"):
            c = Cache(a.cache_dir, split)
            p = predict(model, c.images, dev)
            res[bb][split] = {"acc": accuracy(c.y, p), "macro_f1": macro_f1(c.y, p), "n": len(c)}
            if split == "test":
                for cam, ids in c.camera_groups().items():
                    rows.append({"backbone": bb, "camera": cam, "n": len(ids), "acc": accuracy(c.y[ids], p[ids]),
                                 "empty_frac": float((c.y[ids] == 0).mean())})
        print(bb, json.dumps(res[bb]))
    json.dump(res, open(out / "baseline.json", "w"), indent=1)
    pd.DataFrame(rows).to_csv(out / "baseline_per_camera.csv", index=False)


if __name__ == "__main__":
    main()
