# START HERE (Google Colab)

## A. Setup (once)
1. Unzip this zip. Upload the folder `iwildcam_adapt` to Google Drive: `MyDrive/iwildcam_adapt`.
   (So that `MyDrive/iwildcam_adapt/src/train.py` exists. If it ends up one folder deeper, Cell 0 detects that.)
2. Open `notebooks/run_on_colab.ipynb` in Colab.
3. Runtime -> Change runtime type -> **T4 GPU** (needed for training and for a fast demo; with no GPU use the fast path below).

## B. Fast path (your cache, checkpoints and results are already in Drive) - about 10 minutes
Run only: **Cell 0 -> Cell 8 (analysis) -> Cell 9 (demo)**.
Cell 8 regenerates `results/summary.md` and the figures. Cell 9 prints a public Gradio link.
In the demo choose **Backbone = dino_vits16**, **Adaptation = Entropy minimisation (TENT)**, upload 16 or more photos of ONE camera.

## C. Full path (from scratch, many hours)
Run Cell 0 to 9 in order. Cell 1 is a 5-minute smoke test on fake data; run it first.
Cell 2 (download + cache) takes 1 to 3 hours; Cell 3 (training) and Cell 6 (adaptation) take the most GPU time. All cells are resumable.

## D. What is in this zip
| path | content |
|---|---|
| `src/` | all code (data cache, training, adaptation, analysis, demo) |
| `notebooks/run_on_colab.ipynb` | the Colab notebook |
| `results_reported/summary.md` | the REAL results we obtained on the 27 unseen test cameras |
| `REPORT.md` | report draft with the real numbers (fill the [brackets], add your names) |
| `presentation/CameraTraps.pptx` | 20-slide presentation |
| `webapp/` | Hugging Face Space version of the demo |
| `tests/` | smoke test on fake data |
| `README.md` | design decisions and commands |

## E. Honest summary of the results (say this to your supervisor)
* ViT-S (DINO) + TENT: small but consistent gain. 16 unlabeled images give +1.2 points accuracy, the oracle +1.7. About 70% of the 27 cameras improve. Macro F1 changes by under 0.01.
* ResNet-50 + BN update or TENT: accuracy collapses (0.567 to about 0.12), 0 of 27 cameras improve. The cause is not proven; a BatchNorm-statistics hypothesis is in the report.
* H1 holds for the ViT, not for the ResNet. H2 is only partly supported.
* The data is a subsample at 224x224, so numbers are not comparable to the official WILDS leaderboard.

## F. Common problems
* `No module named 'src'` -> run Cell 0 again (after every runtime restart).
* `Cannot connect to GPU backend` -> free quota used up; Runtime -> Change runtime type -> None to use CPU (fast path only), or try later / use Kaggle (`KAGGLE_README.md`).
* The demo link stops working when the Cell 9 stops. Run Cell 9 again for a new link.
* Demo with ResNet + BN collapses on purpose: that is the finding. Use dino_vits16 + TENT for a normal demo.

## G. "HTTP Error 404" in Cell 2, then `class_names.json not found` in Cell 3
* Cause: Cell 2 tries to download iWildCam from CodaLab (the official host), which now returns 404. The cache is then not built, so training has nothing to read.
* You do NOT need Cell 2 or 3 if your Drive already has `iwildcam_cache`, `ckpt` and `results` from the earlier run. Search Drive (web) for `class_names.json`. Check you are logged in with the same Google account as before.
* If you really need to rebuild: run Cell 2a (mirror download), then Cell 2, 3, 4, 5, 6, 7, 8. Needs a GPU and several hours.
* The zip cannot contain the cache or checkpoints (several GB); they live only in your Drive.
