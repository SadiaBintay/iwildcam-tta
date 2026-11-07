"""Cross-platform end-to-end smoke test on FAKE data (Windows / Mac / Linux). CPU is fine (5-15 min).
  python tests/smoke.py"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, NO_PRETRAINED="1")          # no weight download needed for the test
tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
C, K, R = tmp / "cache", tmp / "ckpt", tmp / "results"


def run(*args):
    args = [str(x) for x in args]
    print(">>", " ".join(args), flush=True)
    subprocess.check_call([sys.executable, *args], cwd=ROOT, env=env)


run("tests/make_fake_cache.py", C)
for bb in ("resnet50", "dino_vits16"):
    run("-m", "src.train", "--backbone", bb, "--cache-dir", C, "--out", K, "--epochs", "1", "--bs", "32")
run("-m", "src.baseline_eval", "--cache-dir", C, "--ckpt-dir", K, "--out", R)
run("-m", "src.tune", "--cache-dir", C, "--ckpt-dir", K, "--out", R, "--lrs", "1e-3", "--steps", "3", "--n", "16", "--seeds", "0",
    "--bn-priors", "0", "256")
run("-m", "src.run_adaptation", "--cache-dir", C, "--ckpt-dir", K, "--out", R, "--methods", "bn", "tent", "--sizes", "16", "32",
    "--seeds", "0", "1", "--oracle")
run("-m", "src.run_adaptation", "--cache-dir", C, "--ckpt-dir", K, "--out", R, "--methods", "bn", "--sizes", "16", "--seeds", "0",
    "--bn-prior", "256")
run("-m", "src.analysis", "--results", R, "--cache-dir", C, "--focus-n", "32")
print("\nSMOKE TEST PASSED ->", R)
