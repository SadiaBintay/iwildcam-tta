"""Two backbones: supervised ResNet-50 and self-supervised DINO ViT-S/16 (both from timm)."""
import os

import timm
import torch
import torch.nn as nn

# name -> (timm tag, number of last blocks to unfreeze)
BACKBONES = {
    "resnet50": ("resnet50.tv_in1k", 1),             # layer4 + fc
    "dino_vits16": ("vit_small_patch16_224.dino", 2),  # last 2 blocks + norm + head
}


def build_model(name: str, n_classes: int, pretrained=None):
    if pretrained is None:
        pretrained = os.environ.get("NO_PRETRAINED", "0") != "1"   # NO_PRETRAINED=1 only for offline smoke tests
    tag, _ = BACKBONES[name]
    return timm.create_model(tag, pretrained=pretrained, num_classes=n_classes)


def trainable_modules(model, name):
    if name == "resnet50":
        return [model.layer4, model.fc]
    k = BACKBONES[name][1]
    return list(model.blocks[-k:]) + [model.norm, model.head]


def head_module(model, name):
    return model.fc if name == "resnet50" else model.head


def freeze_all_but_last(model, name):
    for p in model.parameters():
        p.requires_grad = False
    for m in trainable_modules(model, name):
        for p in m.parameters():
            p.requires_grad = True


def set_train_mode(model, name):
    """Frozen part stays in eval mode (BN stats fixed); only the trainable part is in train mode."""
    model.eval()
    for m in trainable_modules(model, name):
        m.train()


def norm_modules(model, name, scope: str = "all"):
    """Normalisation layers that test-time adaptation may change.
    ResNet -> BatchNorm2d, ViT -> LayerNorm. scope='last' = only the last stage / last 4 blocks."""
    if name == "resnet50":
        root = model.layer4 if scope == "last" else model
        return [m for m in root.modules() if isinstance(m, nn.BatchNorm2d)]
    blocks = model.blocks[-4:] if scope == "last" else model.blocks
    mods = [m for b in blocks for m in b.modules() if isinstance(m, nn.LayerNorm)]
    mods.append(model.norm)
    return mods


def load_checkpoint(name, path, n_classes, dev):
    model = build_model(name, n_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(dev).eval()
