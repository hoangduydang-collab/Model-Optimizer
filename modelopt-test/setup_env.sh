#!/usr/bin/env bash
# Create the ModelOpt + TensorRT-LLM venv for the w4a8_awq feasibility test.
#
# Default venv: <repo>/.venv  (gitignored, project-local — does not touch venvs/main)
#
# Usage:
#   bash setup_env.sh
#   RECREATE_VENV=1 bash setup_env.sh   # wipe and recreate .venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

# Override VENV from env.sh — keep this test's env inside the repo checkout.
MODELOPT_VENV="${MODELOPT_VENV:-${MODEL_OPT_REPO}/.venv}"

echo "=== modelopt-test setup ==="
echo "HOME=$HOME"
echo "MODELOPT_VENV=$MODELOPT_VENV"
echo "MODEL_OPT_REPO=$MODEL_OPT_REPO"

if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
  echo "Recreating venv (--clear)..."
  "$UV" venv --clear --python 3.12 "$MODELOPT_VENV"
elif [[ -f "${MODELOPT_VENV}/bin/activate" ]]; then
  echo "Reusing existing venv at $MODELOPT_VENV"
else
  echo "Creating new venv..."
  "$UV" venv --python 3.12 "$MODELOPT_VENV"
fi

# shellcheck disable=SC1091
source "${MODELOPT_VENV}/bin/activate"

echo "=== installing Model Optimizer (editable) ==="
"$UV" pip install -e "${MODEL_OPT_REPO}[hf]"

echo "=== installing hf_ptq example requirements ==="
# flash-attn's setup imports torch but does not declare it as a build dependency.
# modelopt[hf] installs torch above; use --no-build-isolation so the build sees it.
"$UV" pip install compressed-tensors fire transformers_stream_generator zstandard
if [[ "${SKIP_FLASH_ATTN:-0}" == "1" ]]; then
  echo "SKIP_FLASH_ATTN=1: not installing flash-attn"
elif python -c "import flash_attn" 2>/dev/null; then
  echo "flash-attn already installed"
else
  echo "=== building flash-attn (--no-build-isolation) ==="
  "$UV" pip install 'flash-attn>=2.6.0' --no-build-isolation
fi

echo "=== installing mpi4py (required by modelopt.deploy.llm) ==="
"$UV" pip install mpi4py

echo "=== installing TensorRT-LLM (NVIDIA PyPI) ==="
"$UV" pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

echo "=== installing CUDA runtime libs for tensorrt-llm wheels ==="
"$UV" pip install nvidia-cublas nvidia-cudnn --extra-index-url https://pypi.nvidia.com

echo "=== re-pin local Model Optimizer (tensorrt-llm may replace PyPI modelopt) ==="
"$UV" pip install -e "${MODEL_OPT_REPO}[hf]"

echo "=== smoke import ==="
# shellcheck disable=SC1091
source "${MODELOPT_VENV}/bin/activate"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
python - <<'PY'
import modelopt
import tensorrt_llm
from modelopt.deploy.llm import LLM  # noqa: F401

print("modelopt:", getattr(modelopt, "__version__", "unknown"))
print("tensorrt_llm:", tensorrt_llm.__version__)
print("import OK")
PY

echo "=== applying TRT-LLM Qwen MoE W4A8_CUSTOM patches ==="
python - <<'PY'
import site
from pathlib import Path

# Remove legacy runtime import-hook bootstrap if present.
legacy = Path(site.getsitepackages()[0]) / "modelopt_trtllm_w4a8_custom.pth"
if legacy.exists():
    legacy.unlink()
    print(f"Removed legacy hook: {legacy}")

from modelopt.deploy.trtllm_qwen_moe_patch import apply_trtllm_qwen_moe_patches

patched = apply_trtllm_qwen_moe_patches()
print(patched or "already patched")
PY

echo "=== done ==="
echo "Activate with: source ${MODELOPT_VENV}/bin/activate"
