"""ERM fine-tuning of the last block(s) + classifier on source cameras.
  python -m src.train --backbone resnet50 --cache-dir CACHE --out ckpt
Model selection uses the unseen-camera validation split ('val'); 'test' is never touched here."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import Cache, get_device, num_classes, prep, seed_everything
from .metrics import accuracy, macro_f1, predict
from .models import build_model, freeze_all_but_last, head_module, set_train_mode


def evaluate(model, cache, dev):
    p = predict(model, cache.images, dev)
    return {"acc": accuracy(cache.y, p), "macro_f1": macro_f1(cache.y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["resnet50", "dino_vits16"])
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seed_everything(a.seed)
    dev = get_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nc = num_classes(a.cache_dir)
    tr = Cache(a.cache_dir, "train", ram=True)
    id_val = Cache(a.cache_dir, "id_val")
    val = Cache(a.cache_dir, "val")

    model = build_model(a.backbone, nc)
    freeze_all_but_last(model, a.backbone)
    model.to(dev)
    head_ids = {id(p) for p in head_module(model, a.backbone).parameters()}
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], "lr": a.lr_backbone},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": a.lr_head},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(tr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr_head],
                                                total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")

    log, best = [], -1.0
    for ep in range(a.epochs):
        t0, losses = time.time(), []
        perm = np.random.permutation(len(tr))
        for i in range(0, len(perm), a.bs):
            ids = np.sort(perm[i:i + a.bs])
            x = prep(tr.get(ids), dev, augment=True)
            y = torch.as_tensor(tr.y[ids]).to(dev)
            set_train_mode(model, a.backbone)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        r = {"epoch": ep + 1, "loss": float(np.mean(losses)), "id_val": evaluate(model, id_val, dev),
             "val_unseen": evaluate(model, val, dev), "sec": round(time.time() - t0)}
        log.append(r)
        print(json.dumps(r))
        if r["val_unseen"]["macro_f1"] > best:
            best = r["val_unseen"]["macro_f1"]
            torch.save(model.state_dict(), out / f"{a.backbone}.pt")
    json.dump(log, open(out / f"{a.backbone}_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
