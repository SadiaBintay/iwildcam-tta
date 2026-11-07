import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y, p) -> float:
    """WILDS-style macro F1: averaged over the classes that occur in the ground truth."""
    y, p = np.asarray(y), np.asarray(p)
    return float(f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0))


def accuracy(y, p) -> float:
    return float((np.asarray(y) == np.asarray(p)).mean())


def predict_logits(model, images, dev, bs: int = 128):
    import torch                      # lazy: lets analysis.py run without torch
    from .common import prep
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            x = prep(images[i:i + bs], dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                outs.append(model(x).float().cpu())
    return torch.cat(outs) if outs else torch.zeros(0, 1)


def predict(model, images, dev, bs: int = 128):
    return predict_logits(model, images, dev, bs).argmax(1).numpy()
