# Camera Traps in New Places - how few unlabeled images are enough?

Code for the CSE-414 project: per-camera test-time adaptation of a species classifier on iWildCam (WILDS).
Two backbones (supervised ResNet-50, self-supervised DINO ViT-S/16), three settings (no adaptation, BN-statistics update, TENT entropy minimisation), n = 16/32/64/128 unlabeled images, oracle, ablations, per-camera analysis, Gradio demo.

## Quick start (Colab T4)
1. Unzip this folder into Google Drive (`MyDrive/iwildcam_adapt`).
2. Open `notebooks/run_on_colab.ipynb`, set runtime to **T4 GPU**, run the cells in order. Cell 1 is a 5-minute smoke test on fake data - run it first.

## Pipeline
| step | command | output |
|---|---|---|
| subset + cache | `python -m src.build_cache --root DATA --cache-dir CACHE` | 224px uint8 arrays, class-capped train set, unseen cameras with >=150 images |
| train (ERM) | `python -m src.train --backbone resnet50 --cache-dir CACHE` | `ckpt/<backbone>.pt` (selected on unseen-**val** cameras) |
| baseline | `python -m src.baseline_eval ...` | seen (`id_test`) vs unseen (`test`) acc / macro F1 |
| tune TENT | `python -m src.tune ...` | `results/tent_config_<backbone>.json` (chosen on **val**, not test) |
| adaptation | `python -m src.run_adaptation ... --oracle` | one `.npz` per (method, n, seed); resumable |
| analysis | `python -m src.analysis --results results --cache-dir CACHE` | `summary.md`, `summary.csv`, `per_camera.csv`, `fig_*.png` |
| demo | `python -m src.demo --cache-dir CACHE --share` | upload photos, see before/after |

## Design decisions (put these in the report)
* **Protocol.** For each unseen camera and each seed we draw *n* unlabeled images, adapt a **copy** of the model (no labels), predict the **remaining** images of that camera, then discard the copy. No-adaptation predictions are stored on the same held-out images, so every comparison is **paired**.
* **Metrics.** Macro F1 (WILDS style: over classes present in the ground truth) and accuracy, pooled over all unseen cameras; 95% t-intervals over seeds; a camera-level bootstrap for the per-camera accuracy change.
* **Oracle** adapts on *all* images of a camera and evaluates on the same images - transductive, an upper bound, not a fair deployable result.
* **BN update is ResNet-only.** A ViT has LayerNorm, no batch statistics, so the BN method is skipped for DINO ViT-S; TENT updates LayerNorm scale/shift there and BatchNorm scale/shift (plus re-estimated stats) in the ResNet.
* **TENT predictions never use test-batch statistics.** After the entropy steps, BN statistics are re-estimated from the adapt batch only, so each held-out image is predicted independently.
* **Empty-frame ablation** uses labels only to *compose* the adapt batch (an experimental manipulation); adaptation itself never sees labels.
* **Subset.** Train is capped per class, unseen cameras per camera, images are squashed to 224x224. Numbers are therefore **not comparable to the official WILDS leaderboard**; say so in the report.
* **Seeds.** Seeds randomise the adapt batch (and TENT minibatches). Each backbone is trained once (compute); say so as a limitation.
* Hyper-parameters (TENT lr/steps) are tuned on the unseen-validation cameras; the test cameras are used once for the final tables.

## Files
```
src/common.py        cache reader, normalisation
src/models.py        ResNet-50 / DINO ViT-S via timm, which layers are trainable, which norm layers adapt
src/adapt.py         BN-statistics update, TENT (all on a deep copy)
src/train.py         ERM training
src/run_adaptation.py  per-camera experiments (+ oracle, scope and empty-frame ablations)
src/analysis.py      tables and figures
src/demo.py          Gradio demo
tests/smoke.sh       end-to-end test on fake data
tests/test_analysis_fake.py  offline test of the analysis step
```
