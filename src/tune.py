"""Pick TENT learning rate / steps on the unseen-VALIDATION cameras (never on test).
  python -m src.tune --cache-dir CACHE --ckpt-dir ckpt --out results"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .common import Cache, get_device, num_classes
from .metrics import macro_f1
from .models import load_checkpoint
from .metrics import predict
from .run_adaptation import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--backbones", nargs="+", default=["resnet50", "dino_vits16"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--steps", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--skip-tent", action="store_true", help="only sweep the BN prior; keep tent_config_*.json untouched")
    ap.add_argument("--bn-priors", nargs="+", type=float, default=[0, 16, 64, 256, 1024])
    a = ap.parse_args()
    dev = get_device()
    nc = num_classes(a.cache_dir)
    cache = Cache(a.cache_dir, "val", ram=True)
    groups = cache.camera_groups()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for bb in a.backbones:
        model = load_checkpoint(bb, Path(a.ckpt_dir) / f"{bb}.pt", nc, dev)
        base_pred = predict(model, cache.images, dev)
        if not a.skip_tent:
            best, table = None, []
            for lr, st in itertools.product(a.lrs, a.steps):
                gains = []
                for s in a.seeds:
                    r = run_one(model, bb, cache, groups, base_pred, dev, "tent", a.n, s, "all", lr, st)
                    if r is not None:
                        gains.append(macro_f1(r["y"], r["pred"]) - macro_f1(r["y"], r["base"]))
                g = float(np.mean(gains)) if gains else float("-inf")
                table.append({"lr": lr, "steps": st, "delta_f1": g})
                print(bb, lr, st, f"dF1={g:+.4f}")
                if best is None or g > best["delta_f1"]:
                    best = {"lr": lr, "steps": st, "delta_f1": g}
            json.dump({"lr": best["lr"], "steps": best["steps"], "grid": table},
                      open(Path(a.out) / f"tent_config_{bb}.json", "w"), indent=1)
            print("best", bb, best)
        if bb == "resnet50":      # BN-statistics shrinkage towards the source statistics (extension)
            pt = []
            for pr in a.bn_priors:
                gains, accs = [], []
                for s_ in a.seeds:
                    r = run_one(model, bb, cache, groups, base_pred, dev, "bn", a.n, s_, "all", prior=pr)
                    if r is not None:
                        gains.append(macro_f1(r["y"], r["pred"]) - macro_f1(r["y"], r["base"]))
                        accs.append(float((r["pred"] == r["y"]).mean() - (r["base"] == r["y"]).mean()))
                row = {"prior": pr, "delta_f1": float(np.mean(gains)), "delta_acc": float(np.mean(accs))}
                pt.append(row)
                print("bn prior", row)
            bestp = max(pt, key=lambda r: r["delta_f1"])
            json.dump({"prior": bestp["prior"], "grid": pt}, open(Path(a.out) / "bn_prior_resnet50.json", "w"), indent=1)
            print("best bn prior", bestp)


if __name__ == "__main__":
    main()
