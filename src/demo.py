"""Gradio demo: upload several photos from ONE camera, see species predictions before / after adaptation.
  python -m src.demo --cache-dir CACHE --ckpt-dir ckpt --share
Note: adaptation treats the uploaded images as a single camera (that is the assumption of the study)."""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .adapt import adapt
from .common import class_names, get_device, num_classes
from .metrics import predict_logits
from .models import load_checkpoint


def locate(given, fname, what):
    """Use `given` if it contains `fname`; otherwise search the usual Kaggle / Colab / local places."""
    if (Path(given) / fname).exists():
        return str(given)
    roots = ["/kaggle/working", "/kaggle/input", "/content/drive/MyDrive", "/content", "."]
    for r in roots:
        hits = sorted(glob.glob(f"{r}/**/{fname}", recursive=True))
        if hits:
            found = str(Path(hits[0]).parent)
            print(f"[demo] {what}: '{given}' has no {fname}; using '{found}' instead")
            return found
    raise SystemExit(f"\nERROR: cannot find {fname} ({what}). '{given}' does not contain it.\n"
                     f"The {what} folder was not downloaded/copied to this machine. Run the download cell first "
                     f"(Drive links for the cache and ckpt folders), then run the demo again.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--results", default=None, help="accepted and ignored (older notebooks pass it)")
    a, _unknown = ap.parse_known_args()
    a.cache_dir = locate(a.cache_dir, "class_names.json", "cache")
    a.ckpt_dir = locate(a.ckpt_dir, "dino_vits16.pt", "checkpoint")
    import gradio as gr

    dev = get_device()
    nc = num_classes(a.cache_dir)
    names = class_names(a.cache_dir)
    models = {}

    def get_model(bb):
        if bb not in models:
            models[bb] = load_checkpoint(bb, Path(a.ckpt_dir) / f"{bb}.pt", nc, dev)
            for p in models[bb].parameters():
                p.requires_grad = False
        return models[bb]

    def run(files, bb, method, steps, lr):
        if not files:
            return [], []
        method = {"BN statistics update": "bn", "Entropy minimisation (TENT)": "tent"}[method]
        if method == "bn" and bb != "resnet50":
            raise gr.Error("BN update needs BatchNorm: choose resnet50, or use TENT for the ViT.")
        paths = [f if isinstance(f, str) else f.name for f in files]
        imgs = np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)) for p in paths])
        model = get_model(bb)
        pb = predict_logits(model, imgs, dev).softmax(1)
        am = adapt(model, bb, method, torch.from_numpy(imgs).to(dev), dev, "all", lr=lr, steps=int(steps),
                   gen=torch.Generator().manual_seed(0))
        pa = predict_logits(am, imgs, dev).softmax(1)
        rows, gallery = [], []
        for i, p in enumerate(paths):
            b, c = int(pb[i].argmax()), int(pa[i].argmax())
            rows.append([Path(p).name, names[b], round(float(pb[i, b]), 3), names[c], round(float(pa[i, c]), 3),
                         "yes" if b != c else ""])
            gallery.append((p, f"{names[b]} -> {names[c]}"))
        return gallery, rows

    with gr.Blocks(title="Camera-adaptive species classifier") as demo:
        gr.Markdown("### Upload a few photos from one camera trap and compare predictions before / after adaptation\n"
                    "Adaptation uses **no labels** - only the uploaded images (treated as one camera).")
        files = gr.File(file_count="multiple", file_types=["image"], label="Photos from one camera (16+ works best)")
        with gr.Row():
            bb = gr.Dropdown(["dino_vits16", "resnet50"], value="dino_vits16", label="Backbone")
            meth = gr.Dropdown(["Entropy minimisation (TENT)", "BN statistics update"],
                               value="Entropy minimisation (TENT)", label="Adaptation")
            steps = gr.Slider(1, 30, value=10, step=1, label="TENT steps")
            lr = gr.Number(value=1e-3, label="TENT learning rate")
        btn = gr.Button("Predict before / after")
        gal = gr.Gallery(label="before -> after", columns=4)
        tab = gr.Dataframe(headers=["image", "before", "conf", "after", "conf", "changed"])
        btn.click(run, [files, bb, meth, steps, lr], [gal, tab])
    demo.launch(share=a.share)


if __name__ == "__main__":
    main()
