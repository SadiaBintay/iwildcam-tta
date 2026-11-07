"""Test-time adaptation using ONLY unlabeled images. Every call works on a deep copy, so the
source model is untouched ("reset before the next camera")."""
import copy

import torch
import torch.nn as nn

from .common import prep
from .models import norm_modules


def reestimate_bn(model, name, imgs_u8, dev, scope="all", prior: float = 0.0, chunk: int = 128):
    """Replace BatchNorm running statistics by statistics of the unlabeled batch.
    prior>0 shrinks towards the source statistics: w = n / (n + prior)."""
    bns = norm_modules(model, name, scope)
    n = len(imgs_u8)
    saved = [(b.running_mean.clone(), b.running_var.clone()) for b in bns]
    model.eval()
    for b in bns:
        b.reset_running_stats()
        b.momentum = None            # cumulative average over chunks
        b.train()
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        for i in range(0, n, chunk):
            model(prep(imgs_u8[i:i + chunk], dev))
    w = n / (n + prior) if prior > 0 else 1.0
    for b, (m0, v0) in zip(bns, saved):
        b.running_mean.mul_(w).add_((1 - w) * m0)
        b.running_var.mul_(w).add_((1 - w) * v0)
        b.eval()
        b.momentum = 0.1


def entropy(logits):
    p = logits.softmax(1)
    return -(p * torch.log(p + 1e-8)).sum(1)


def tent(model, name, imgs_u8, dev, scope="all", lr=1e-3, steps=10, mb=32, prior=0.0, gen=None):
    """TENT-style entropy minimisation: only the affine parameters of the normalisation layers are trained."""
    norms = norm_modules(model, name, scope)
    for p in model.parameters():
        p.requires_grad = False
    params = []
    for m in norms:
        for p in m.parameters():
            p.requires_grad = True
            params.append(p)
    opt = torch.optim.Adam(params, lr=lr)
    model.eval()
    if name == "resnet50":
        for b in norms:
            b.train()                # use batch statistics while adapting
    n = len(imgs_u8)
    mb = min(mb, n)
    for _ in range(steps):
        ids = torch.randperm(n, generator=gen)[:mb]
        x = prep(imgs_u8[ids.to(imgs_u8.device)], dev)
        loss = entropy(model(x).float()).mean()       # fp32 on purpose (small grads)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    if name == "resnet50":           # predict with statistics from the adapt batch, never from the test batch
        reestimate_bn(model, name, imgs_u8, dev, scope, prior)
    for p in model.parameters():
        p.requires_grad = False


def adapt(model, name, method, imgs_u8, dev, scope="all", lr=1e-3, steps=10, mb=32, prior=0.0, gen=None):
    """Return an adapted copy of `model`. method in {'none','bn','tent'}. 'bn' is ResNet-only (ViT has no BatchNorm)."""
    if method == "none":
        return model
    if method == "bn" and name != "resnet50":
        raise ValueError("BN-statistics adaptation needs BatchNorm layers (ResNet-50 only)")
    m = copy.deepcopy(model)
    if method == "bn":
        reestimate_bn(m, name, imgs_u8, dev, scope, prior)
    elif method == "tent":
        tent(m, name, imgs_u8, dev, scope, lr, steps, mb, prior, gen)
    else:
        raise ValueError(method)
    return m.eval()
