# Running the project in VS Code (your own computer, no GPU needed)

## What runs locally (CPU) and what does not
| task | locally? |
|---|---|
| Tables + figures (`src.analysis`) | yes, seconds |
| Demo web app (before/after adaptation) | yes, ~1-2 min per click |
| Smoke test on fake data | yes, 5-15 min |
| Build web-app bundle, upload to Hugging Face | yes |
| Download iWildCam (11 GB), train, full adaptation experiments | **no** - use Colab with GPU (already done; results are on your Drive) |

## 1. One-time setup
1. Install **Python 3.11** (3.10-3.12 work; tick "Add Python to PATH" on Windows) and **VS Code** with the *Python* extension.
2. VS Code: **File -> Open Folder** -> this project folder (the one containing `src/`, `webapp/`, `README_LOCAL.md`).
3. Open the terminal: **Terminal -> New Terminal**, then:

   Windows (PowerShell):
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   (if it says scripts are disabled: `Set-ExecutionPolicy -Scope Process Bypass` and activate again)

   Mac / Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install packages (the prompt should show `(.venv)`):
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements-local.txt
   ```
5. Tell VS Code which Python to use: **Ctrl+Shift+P -> "Python: Select Interpreter" -> `.venv`**.

## 2. Check that everything works (fake data, no downloads)
```
python tests/smoke.py
```
It ends with `SMOKE TEST PASSED`. If it fails, copy the last 30 lines of the error.

## 3. Put your real files in `data/` (see `data/README.txt`)
Download from Google Drive: `ckpt/*.pt`, the `results/` folder and `iwildcam_cache/class_names.json`
(the 1.4 GB `test_images.npy` + `test_meta.npz` are optional).

```
data/ckpt/resnet50.pt   data/ckpt/dino_vits16.pt
data/results/...        data/cache/class_names.json
```

## 4. Run things
Always run from the project folder, with `(.venv)` active.

**Tables and figures** (output in `data/results/`: `summary.md`, `fig_*.png`):
```
python -m src.analysis --results data/results --focus-n 64
```
Add `--cache-dir data/cache` only if you downloaded the optional test cache (adds brightness / empty-frame plots).

**Demo web app** (opens http://127.0.0.1:7860):
```
python -m src.demo --cache-dir data/cache --ckpt-dir data/ckpt
```
Upload 16+ photos from ONE camera, choose `dino_vits16` + TENT, press *Predict before / after*.

**Deployable web app + Hugging Face link** (permanent, free):
```
python scripts/make_webapp_bundle.py          # builds build/webapp  (test it: python build/webapp/app.py)
python scripts/push_to_hf.py                  # asks for your Hugging Face WRITE token
```

## 5. Colab-only steps (need a GPU; already done)
`src.build_cache`, `src.train`, `src.baseline_eval`, `src.tune`, `src.run_adaptation` - see `notebooks/run_on_colab.ipynb`.
They can also run locally on CPU, but expect many hours.

## Common problems
* `No module named 'src'` -> you are not in the project folder; `cd` into it (commands start with `python -m src...`).
* `No module named 'torch'` -> the venv is not active, or step 4 was skipped.
* `FileNotFoundError ... class_names.json` -> step 3, file is missing in `data/cache/`.
* Demo is slow -> normal on CPU. ViT + TENT is slower than ResNet; lower "TENT steps" to 5.
