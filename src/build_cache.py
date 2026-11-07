"""Download iWildCam (WILDS), take a subset, resize to 224x224 and store as uint8 .npy + metadata.
Run once. Resumable: finished splits are skipped.

  python -m src.build_cache --root /content/data --cache-dir /content/drive/MyDrive/iwildcam_cache
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class _Imgs(torch.utils.data.Dataset):
    def __init__(self, ds, idx, size):
        self.ds, self.idx, self.size = ds, idx, size

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = self.ds.get_input(int(self.idx[i])).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def choose(split, y, loc, a, rng):
    ids = np.arange(len(y))
    if split == "train":                       # class-balanced cap
        out = []
        for c in np.unique(y):
            ci = ids[y == c]
            out.append(rng.choice(ci, min(len(ci), a.train_cap_per_class), replace=False))
        return np.sort(np.concatenate(out))
    if split in ("id_val", "id_test"):          # seen cameras: random subset
        return np.sort(rng.choice(ids, min(len(ids), a.seen_cap), replace=False))
    out = []                                    # unseen cameras (val / test): cap per camera, drop tiny cameras
    for c in np.unique(loc):
        ci = ids[loc == c]
        if len(ci) < a.min_camera_images:
            continue
        out.append(rng.choice(ci, min(len(ci), a.per_camera_cap), replace=False))
    return np.sort(np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="where WILDS downloads iWildCam (~11 GB)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "id_val", "id_test", "val", "test"])
    ap.add_argument("--train-cap-per-class", type=int, default=100)
    ap.add_argument("--seen-cap", type=int, default=3000)
    ap.add_argument("--per-camera-cap", type=int, default=400)
    ap.add_argument("--min-camera-images", type=int, default=150)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    from wilds import get_dataset
    out = Path(a.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = get_dataset(dataset="iwildcam", root_dir=a.root, download=not a.no_download)
    loc_col = ds.metadata_fields.index("location")
    y_all = ds.y_array.numpy()
    loc_all = ds.metadata_array[:, loc_col].numpy()
    split_all = ds.split_array

    names = [str(i) for i in range(ds.n_classes)]
    try:   # nicer species names if the dataset ships categories.csv
        import pandas as pd
        cat = pd.read_csv(Path(ds.data_dir) / "categories.csv")
        for _, r in cat.iterrows():
            if 0 <= int(r["y"]) < len(names):
                names[int(r["y"])] = str(r["name"])
    except Exception as e:
        print("class names fallback to ids:", e)
    json.dump(names, open(out / "class_names.json", "w"))

    for split in a.splits:
        if (out / f"{split}.done").exists():
            print("skip", split)
            continue
        rng = np.random.default_rng(a.seed)
        pool = np.where(split_all == ds.split_dict[split])[0]
        sel = pool[choose(split, y_all[pool], loc_all[pool], a, rng)]
        print(f"{split}: {len(sel)} images, {len(np.unique(loc_all[sel]))} cameras")
        arr = np.lib.format.open_memmap(out / f"{split}_images.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(sel), a.size, a.size, 3))
        dl = torch.utils.data.DataLoader(_Imgs(ds, sel, a.size), batch_size=64, num_workers=a.workers,
                                         collate_fn=lambda b: np.stack(b))
        k = 0
        for batch in tqdm(dl, desc=split):
            arr[k:k + len(batch)] = batch
            k += len(batch)
        arr.flush()
        np.savez(out / f"{split}_meta.npz", y=y_all[sel], location=loc_all[sel], wilds_idx=sel)
        (out / f"{split}.done").write_text("ok")


if __name__ == "__main__":
    main()
