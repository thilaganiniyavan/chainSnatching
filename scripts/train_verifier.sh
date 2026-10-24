#!/usr/bin/env bash
# Fine-tune the learned verifier on an NVIDIA GPU under Linux, evaluate it, pick the model to deploy
# (snatchguard/verifier/finetune.py; the same steps as scripts/windows/train_verifier.ps1).
#   bash scripts/train_verifier.sh ~/verifier_data --quick     # ~10-minute check
#   bash scripts/train_verifier.sh ~/verifier_data             # the real run
#   bash scripts/train_verifier.sh ~/verifier_data --batch 8   # GPUs with 4-6 GB
#   SKIP_SWIN=1 bash scripts/train_verifier.sh ~/verifier_data # R(2+1)D-18 only
# Creates .venv on first use (PyTorch with CUDA from pip). Results: verifier_ft_results.zip in the repo folder;
# the model to use live: models/verifier/verifier_ft.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:?usage: bash scripts/train_verifier.sh <verifier_data folder> [--quick] [finetune.py run options]}
shift
[ ! -f "$DATA/index.json" ] && [ -f "$DATA/verifier_data/index.json" ] && DATA="$DATA/verifier_data"   # unzipped into a folder of the same name
[ -f "$DATA/index.json" ] || { echo "$DATA has no index.json: point at the unzipped verifier_data folder"; exit 1; }
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install torch torchvision          # the default Linux wheels include CUDA
    .venv/bin/pip install -e ".[learned]"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "nvidia-smi not found: is the NVIDIA driver installed?"
.venv/bin/python -c "import torch, sys; print('CUDA:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "PyTorch cannot see the GPU: update the NVIDIA driver, then rm -rf .venv and run again"; exit 1; }
SFX=""
EXTRA=()
for a in "$@"; do
    if [ "$a" = "--quick" ]; then SFX=_quick; EXTRA=(--steps 20); fi
done
DEPLOY=models/verifier/verifier_ft.pt
[ -n "$SFX" ] && DEPLOY=runtime/verifier_ft_quick/deploy/verifier_ft.pt
RUNS=()
.venv/bin/python -m snatchguard.verifier.finetune run --data "$DATA" --arch r2plus1d_18 --out "runtime/verifier_ft$SFX" \
    --protocols external curve s1-cv final "${EXTRA[@]}" "$@"
RUNS+=("runtime/verifier_ft$SFX")
if [ -z "${SKIP_SWIN:-}" ]; then
    if .venv/bin/python -m snatchguard.verifier.finetune run --data "$DATA" --arch swin3d_t \
        --out "runtime/verifier_ft_swin$SFX" --protocols external s1-cv final "${EXTRA[@]}" "$@"; then
        RUNS+=("runtime/verifier_ft_swin$SFX")
    else
        echo "swin3d_t failed: continuing with the runs that finished"
    fi
fi
.venv/bin/python -m snatchguard.verifier.finetune select --out "$DEPLOY" --runs "${RUNS[@]}"
if [ -n "$SFX" ]; then
    echo "Quick check passed. Now run the same command without --quick."
    exit 0
fi
# results to send back: reports, logs and test scores (a few MB); the trained models stay here for live use
.venv/bin/python - "${RUNS[@]}" <<'EOF'
import json, sys, zipfile
from pathlib import Path
with zipfile.ZipFile("verifier_ft_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for run in sys.argv[1:]:
        arch = json.loads((Path(run) / "report.json").read_text()).get("arch", "r2plus1d_18")
        for f in Path(run).iterdir():
            if f.is_file() and f.suffix not in (".pt", ".tmp"):
                z.write(f, f"{arch}/{f.name}")
    z.write("models/verifier/selection.json", "selection.json")
EOF
echo "Done. The model to use live is $DEPLOY (see models/verifier/selection.json)."
echo "Send back: $(pwd)/verifier_ft_results.zip"
