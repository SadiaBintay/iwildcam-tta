#!/usr/bin/env bash
# End-to-end smoke test on fake data (CPU is fine, GPU faster). Pretrained weights are skipped so it also works offline.
set -e
cd "$(dirname "$0")/.."
export NO_PRETRAINED=1
C=/tmp/fake_cache; K=/tmp/fake_ckpt; R=/tmp/fake_results
rm -rf $C $K $R
python tests/make_fake_cache.py $C
for B in resnet50 dino_vits16; do python -m src.train --backbone $B --cache-dir $C --out $K --epochs 1 --bs 32; done
python -m src.baseline_eval --cache-dir $C --ckpt-dir $K --out $R
python -m src.tune --cache-dir $C --ckpt-dir $K --out $R --lrs 1e-3 --steps 3 --n 16 --seeds 0
python -m src.run_adaptation --cache-dir $C --ckpt-dir $K --out $R --methods bn tent --sizes 16 32 --seeds 0 1 --oracle
python -m src.run_adaptation --cache-dir $C --ckpt-dir $K --out $R --methods bn tent --sizes 32 --seeds 0 1 --scope last
python -m src.run_adaptation --cache-dir $C --ckpt-dir $K --out $R --methods tent --sizes 32 --seeds 0 1 --empty-frac 0.1
python -m src.analysis --results $R --cache-dir $C --focus-n 32
echo "SMOKE TEST PASSED -> $R"
