"""Per-camera test-time adaptation experiments.

For every unseen camera: draw n unlabeled images, adapt a COPY of the model, predict the REMAINING
images of that camera (never the adapt images), then reset. The no-adaptation prediction on the same
held-out images is stored next to it, so comparisons are paired.

  python -m src.run_adaptation --cache-dir CACHE --ckpt-dir ckpt --out results \
      --methods bn tent --sizes 16 32 64 128 --seeds 0 1 2 3 4 --oracle

Oracle ('all'): adapts on all images of the camera and evaluates on the same images (transductive, upper bound).
Labels are used only to evaluate, and (for --empty-frac ablation) to set the composition of the adapt batch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .adapt import adapt
from .common import MIN_EVAL, Cache, get_device, num_classes
from .metrics import accuracy, macro_f1, predict
from .models import load_checkpoint


def pick_adapt(y_cam, perm, n, empty_frac):
    """Indices (local to the camera) of the adapt batch. empty_frac=None -> random."""
    if empty_frac is None:
        return perm[:n]
    emp, non = perm[y_cam[perm] == 0], perm[y_cam[perm] != 0]
    k = int(round(empty_frac * n))
    if k > len(emp) or n - k > len(non):
        return None
    return np.concatenate([emp[:k], non[:n - k]])


def run_one(model, name, cache, groups, base_pred, dev, method, n, seed, scope="all", lr=1e-3, steps=10,
            mb=32, prior=0.0, empty_frac=None):
    cams, idxs, ys, preds, bases = [], [], [], [], []
    for cam, ids in groups.items():
        if n == "all":
            a_loc = e_loc = np.arange(len(ids))
        else:
            if len(ids) < n + MIN_EVAL:
                continue
            rng = np.random.default_rng([seed, cam])
            perm = rng.permutation(len(ids))
            a_loc = pick_adapt(cache.y[ids], perm, n, empty_frac)
            if a_loc is None:
                continue
            e_loc = perm[~np.isin(perm, a_loc)]
        imgs = torch.from_numpy(np.ascontiguousarray(cache.get(ids[a_loc]))).to(dev)   # unlabeled adapt batch
        gen = torch.Generator().manual_seed(int(seed) * 100003 + cam)
        m = adapt(model, name, method, imgs, dev, scope, lr, steps, mb, prior, gen)
        e_ids = ids[e_loc]
        p = predict(m, cache.get(e_ids), dev)
        if m is not model:
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        cams.append(np.full(len(e_ids), cam)); idxs.append(e_ids); ys.append(cache.y[e_ids])
        preds.append(p); bases.append(base_pred[e_ids])
    if not cams:
        return None
    return {k: np.concatenate(v) for k, v in
            dict(camera=cams, idx=idxs, y=ys, pred=preds, base=bases).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--backbones", nargs="+", default=["resnet50", "dino_vits16"])
    ap.add_argument("--methods", nargs="+", default=["bn", "tent"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--scope", default="all", choices=["all", "last"])
    ap.add_argument("--empty-frac", type=float, default=None)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--tent-lr", type=float, default=None)
    ap.add_argument("--tent-steps", type=int, default=None)
    ap.add_argument("--bn-prior", type=float, default=0.0)
    a = ap.parse_args()

    dev = get_device()
    nc = num_classes(a.cache_dir)
    cache = Cache(a.cache_dir, a.split, ram=True)
    groups = cache.camera_groups()
    print(f"{a.split}: {len(cache)} images, {len(groups)} cameras")
    sizes = list(a.sizes) + (["all"] if a.oracle else [])

    for bb in a.backbones:
        model = load_checkpoint(bb, Path(a.ckpt_dir) / f"{bb}.pt", nc, dev)
        for p in model.parameters():
            p.requires_grad = False
        cfg = {"lr": 1e-3, "steps": 10}
        cfg_path = Path(a.out) / f"tent_config_{bb}.json"
        if cfg_path.exists():
            cfg.update(json.load(open(cfg_path)))
        lr = a.tent_lr if a.tent_lr is not None else cfg["lr"]
        steps = a.tent_steps if a.tent_steps is not None else cfg["steps"]
        base_pred = predict(model, cache.images, dev)
        odir = Path(a.out) / bb
        odir.mkdir(parents=True, exist_ok=True)
        for method in a.methods:
            if method == "bn" and bb != "resnet50":
                print(f"[{bb}] skip bn (no BatchNorm in a ViT)")
                continue
            for n in sizes:
                for seed in (a.seeds[:3] if n == "all" else a.seeds):
                    tag = f"{a.split}__{method}__{a.scope}__n{n}__e{a.empty_frac}__s{seed}"
                    if a.bn_prior > 0:
                        tag += f"__p{a.bn_prior:g}"
                    f = odir / f"{tag}.npz"
                    if f.exists():
                        continue
                    t0 = time.time()
                    r = run_one(model, bb, cache, groups, base_pred, dev, method, n, seed, a.scope, lr, steps,
                                prior=a.bn_prior, empty_frac=a.empty_frac)
                    if r is None:
                        print("no usable camera for", tag)
                        continue
                    meta = dict(backbone=bb, split=a.split, method=method, scope=a.scope, n=str(n), seed=seed,
                                empty_frac=a.empty_frac, tent_lr=lr, tent_steps=steps, bn_prior=a.bn_prior)
                    np.savez_compressed(f, meta=json.dumps(meta), **r)
                    print(f"[{bb}] {tag}: acc {accuracy(r['y'], r['pred']):.3f} (base {accuracy(r['y'], r['base']):.3f})"
                          f"  F1 {macro_f1(r['y'], r['pred']):.3f} (base {macro_f1(r['y'], r['base']):.3f})"
                          f"  cams {len(np.unique(r['camera']))}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
