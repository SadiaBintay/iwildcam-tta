"""Gradio web app: species predictions before / after test-time adaptation on photos from ONE camera.
Runs on CPU (Hugging Face Spaces free tier) or GPU if available."""
import csv
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.adapt import adapt
from src.common import prep
from src.models import load_checkpoint

ROOT = Path(__file__).resolve().parent
NAMES = json.load(open(ROOT / "class_names.json"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))

EX_DIR = ROOT / "examples"
EXAMPLES = {}                                    # label -> photo paths (one folder = one camera)
if EX_DIR.exists():
    for d in sorted(p for p in EX_DIR.iterdir() if p.is_dir()):
        fs = sorted(str(p) for p in d.glob("*.jpg"))
        if fs:
            EXAMPLES[f"{d.name} ({len(fs)} photos)"] = fs
    top = sorted(str(p) for p in EX_DIR.glob("*.jpg"))
    if top:
        EXAMPLES[f"example camera ({len(top)} photos)"] = top
TRUTH = {}
for tl in EX_DIR.rglob("true_labels.csv") if EX_DIR.exists() else []:
    with open(tl, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                TRUTH[row[0]] = row[1]

_models = {}


def get_model(bb):
    if bb not in _models:
        m = load_checkpoint(bb, ROOT / "ckpt" / f"{bb}.pt", len(NAMES), DEV)
        for p in m.parameters():
            p.requires_grad = False
        _models[bb] = m
    return _models[bb]


@torch.no_grad()
def probs(model, imgs):
    model.eval()
    out = [model(prep(imgs[i:i + 32], DEV)).float().softmax(1).cpu() for i in range(0, len(imgs), 32)]
    return torch.cat(out)


METHODS = {"Entropy minimisation (TENT)": "tent", "BN statistics update (ResNet-50 only)": "bn"}


def run(files, backbone, method, steps, lr):
    if not files:
        raise gr.Error("Upload photos from ONE camera first, or press 'Load example camera'.")
    paths = [f if isinstance(f, str) else f.name for f in files][:200]
    m = METHODS[method]
    if m == "bn" and backbone != "resnet50":
        raise gr.Error("The BN update needs BatchNorm layers: choose resnet50, or use TENT for the ViT.")
    imgs = np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)) for p in paths])
    model = get_model(backbone)
    pb = probs(model, imgs)
    am = adapt(model, backbone, m, torch.from_numpy(imgs).to(DEV), DEV, "all", lr=float(lr), steps=int(steps),
               gen=torch.Generator().manual_seed(0))
    pa = probs(am, imgs)

    rows, gallery, ok_b, ok_a, known = [], [], 0, 0, 0
    for i, p in enumerate(paths):
        b, a = int(pb[i].argmax()), int(pa[i].argmax())
        name = Path(p).name
        truth = TRUTH.get(name, "")
        if truth:
            known += 1
            ok_b += NAMES[b] == truth
            ok_a += NAMES[a] == truth
        rows.append([name, truth, NAMES[b], round(float(pb[i, b]), 3), NAMES[a], round(float(pa[i, a]), 3),
                     "yes" if a != b else ""])
        gallery.append((p, f"{NAMES[b]} -> {NAMES[a]}"))

    msg = [f"**{len(paths)} photos** treated as one camera."]
    if len(paths) < 16:
        msg.append("Warning: fewer than 16 photos - adaptation is unreliable, especially the BN update.")
    if known == len(paths) and known > 0:
        msg.append(f"Correct before adaptation: **{ok_b}/{known}** - after: **{ok_a}/{known}** (ground truth from the dataset).")
    return gallery, rows, " ".join(msg)


with gr.Blocks(title="Camera-adaptive species classifier") as demo:
    gr.Markdown(
        "## Camera Traps in New Places\n"
        "Upload **16 or more photos from one camera trap**. The model adapts using only these unlabeled photos "
        "(no labels), then predicts again. A fresh copy of the model is used on every click.\n\n"
        "*What our study found (27 unseen cameras, iWildCam subset): DINO ViT-S + TENT gives a small but consistent "
        "gain (about +1 to +2 accuracy points); the BN-statistics update on ResNet-50 hurts on unseen cameras. "
        "Adaptation can fail - the table lets you see it.*")
    files = gr.File(file_count="multiple", file_types=["image"], label="Photos from ONE camera")
    if EXAMPLES:
        with gr.Row():
            ex_dd = gr.Dropdown(list(EXAMPLES), value=list(EXAMPLES)[0], label="Example camera (unseen test cameras)")
            ex_btn = gr.Button("Load example photos")
        ex_btn.click(lambda k: EXAMPLES[k], ex_dd, files)
    with gr.Row():
        bb = gr.Dropdown(["dino_vits16", "resnet50"], value="dino_vits16", label="Backbone")
        meth = gr.Dropdown(list(METHODS), value="Entropy minimisation (TENT)", label="Adaptation method")
        steps = gr.Slider(1, 30, value=10, step=1, label="TENT steps")
        lr = gr.Number(value=1e-3, label="TENT learning rate")
    btn = gr.Button("Predict before / after", variant="primary")
    info = gr.Markdown()
    gal = gr.Gallery(label="before -> after", columns=4, height="auto")
    tab = gr.Dataframe(headers=["image", "true species", "before", "conf", "after", "conf", "changed"], wrap=True)
    btn.click(run, [files, bb, meth, steps, lr], [gal, tab, info])

if __name__ == "__main__":
    demo.launch()
