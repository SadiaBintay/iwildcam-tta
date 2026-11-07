"""Offline test of the analysis step with synthetic results (no torch needed: a stub is injected).
  python tests/test_analysis_fake.py"""
import json, sys, tempfile, types
from pathlib import Path

_t = types.ModuleType("torch"); _t.Tensor = type("Tensor", (), {})
sys.modules.setdefault("torch", _t)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

tmp = Path(tempfile.mkdtemp())
cache, res = tmp / "cache", tmp / "results"
cache.mkdir(); res.mkdir()
rng = np.random.default_rng(0)
cams = np.repeat(np.arange(6), 200)
y = rng.integers(0, 10, len(cams))
np.save(cache / "test_images.npy", rng.integers(0, 255, (len(cams), 8, 8, 3), dtype=np.uint8))
np.savez(cache / "test_meta.npz", y=y, location=cams, wilds_idx=np.arange(len(cams)))
json.dump(["x"] * 10, open(cache / "class_names.json", "w"))
json.dump({"resnet50": {"id_test": {"acc": .8, "macro_f1": .5}, "test": {"acc": .6, "macro_f1": .3}}},
          open(res / "baseline.json", "w"))

def fake(bb, method, n, seed, scope="all", ef=None, gain=0.05, prior=0.0):
    r = np.random.default_rng(hash((bb, method, str(n), seed)) % 2**32)
    keep = []
    for c in range(6):
        ids = np.where(cams == c)[0]
        keep.append(ids if n == "all" else r.permutation(ids)[int(n):])
    idx = np.concatenate(keep)
    base = np.where(r.random(len(idx)) < .5, y[idx], r.integers(0, 10, len(idx)))
    pred = np.where(r.random(len(idx)) < gain, y[idx], base)
    d = res / bb; d.mkdir(exist_ok=True)
    meta = dict(backbone=bb, split="test", method=method, scope=scope, n=str(n), seed=seed, empty_frac=ef,
                tent_lr=1e-3, tent_steps=10, bn_prior=prior)
    np.savez_compressed(d / f"t_{method}_{scope}_{n}_{ef}_{seed}_{prior}.npz", meta=json.dumps(meta), camera=cams[idx],
                        idx=idx, y=y[idx], pred=pred, base=base)

for bb, methods in (("resnet50", ["bn", "tent"]), ("dino_vits16", ["tent"])):
    for m in methods:
        for n in (16, 32, 64, 128, "all"):
            for s in range(3):
                fake(bb, m, n, s)
        for s in range(3):
            fake(bb, m, 64, s, scope="last")
            for ef in (0.0, 0.5):
                fake(bb, m, 64, s, ef=ef)

for n in (16, 64):
    for s_ in range(3):
        fake("resnet50", "bn", n, s_, prior=64.0, gain=0.02)
from src.analysis import main
sys.argv = ["analysis", "--results", str(res), "--cache-dir", str(cache), "--focus-n", "64"]
main()
print("\nFILES:", sorted(p.name for p in res.glob("*") if p.is_file()))
