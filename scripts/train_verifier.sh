#!/usr/bin/env bash
# Fine-tune the learned verifier on an NVIDIA GPU under Linux (snatchguard/verifier/finetune.py).
#   bash scripts/train_verifier.sh ~/verifier_data --quick     # 5-minute check
#   bash scripts/train_verifier.sh ~/verifier_data             # the real run
#   bash scripts/train_verifier.sh ~/verifier_data --batch 8   # GPUs with 4-6 GB
# Creates .venv on first use (PyTorch with CUDA from pip). Results: verifier_ft_results.zip in the repo folder.
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
OUT=runtime/verifier_ft
EXTRA=()
for a in "$@"; do
    if [ "$a" = "--quick" ]; then OUT=runtime/verifier_ft_quick; EXTRA=(--steps 20); fi
done
.venv/bin/python -m snatchguard.verifier.finetune run --data "$DATA" --out "$OUT" "${EXTRA[@]}" "$@"
if [ "$OUT" = runtime/verifier_ft ]; then
    .venv/bin/python -c "import shutil; shutil.make_archive('verifier_ft_results', 'zip', '$OUT')"
    echo "Done. Send back: $(pwd)/verifier_ft_results.zip"
else
    echo "Quick check passed. Now run the same command without --quick."
fi
