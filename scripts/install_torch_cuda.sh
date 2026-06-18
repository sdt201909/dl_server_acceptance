#!/usr/bin/env bash
set -euo pipefail

CUDA_FLAVOR="${1:-cu128}"
PYTHON_BIN="${PYTHON:-python3}"

case "$CUDA_FLAVOR" in
  cu118|cu121|cu124|cu126|cu128)
    ;;
  *)
    echo "ERROR: unsupported CUDA wheel flavor: $CUDA_FLAVOR" >&2
    echo "Use the PyTorch selector for the right value, for example cu128." >&2
    exit 2
    ;;
esac

echo "Installing PyTorch for CUDA wheel flavor: $CUDA_FLAVOR"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install numpy
"$PYTHON_BIN" -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_FLAVOR}"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("ERROR: PyTorch installed but CUDA is not available.", file=sys.stderr)
    raise SystemExit(1)
PY

