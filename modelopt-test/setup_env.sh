#!/usr/bin/env bash
# Create the ModelOpt + TensorRT-LLM venv for the w4a8_awq feasibility test.
#
# Default venv: <repo>/.venv  (gitignored, project-local — does not touch venvs/main)
#
# Usage:
#   bash setup_env.sh                    # deploy path (TRT-LLM + patches; no hf_ptq)
#   INSTALL_HF_PTQ=1 bash setup_env.sh   # also install hf_ptq extras + flash-attn
#   RECREATE_VENV=1 bash setup_env.sh    # wipe and recreate .venv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_uv_pip.sh"

# Override VENV from env.sh — keep this test's env inside the repo checkout.
MODELOPT_VENV="${MODELOPT_VENV:-${MODEL_OPT_REPO}/.venv}"

echo "=== modelopt-test setup ==="
echo "HOME=$HOME"
echo "MODELOPT_VENV=$MODELOPT_VENV"
echo "MODEL_OPT_REPO=$MODEL_OPT_REPO"
echo "INSTALL_HF_PTQ=${INSTALL_HF_PTQ:-0}"

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

echo "=== removing legacy TRT-LLM import-hook .pth (if any) ==="
python - <<'PY'
import site
from pathlib import Path

legacy = Path(site.getsitepackages()[0]) / "modelopt_trtllm_w4a8_custom.pth"
if legacy.exists():
    legacy.unlink()
    print(f"Removed legacy hook: {legacy}")
else:
    print("No legacy .pth found")
PY

echo "=== installing Model Optimizer (editable) ==="
"$UV" pip install -e "${MODEL_OPT_REPO}[hf]"

echo "=== installing mpi4py (required by modelopt.deploy.llm) ==="
"$UV" pip install mpi4py

echo "=== installing TensorRT-LLM (NVIDIA PyPI) ==="
TENSORRT_LLM_SPEC="${TENSORRT_LLM_SPEC:-tensorrt-llm==1.2.1}"
echo "TENSORRT_LLM_SPEC=$TENSORRT_LLM_SPEC"
"$UV" pip install "${TENSORRT_LLM_SPEC}" --extra-index-url https://pypi.nvidia.com

echo "=== installing CUDA runtime libs for tensorrt-llm wheels ==="
"$UV" pip install nvidia-cublas nvidia-cudnn --extra-index-url https://pypi.nvidia.com

echo "=== re-pin local Model Optimizer (tensorrt-llm may replace PyPI modelopt) ==="
_reinstall_editable_modelopt "${MODEL_OPT_REPO}"

echo "=== pin transformers (fused Qwen3MoeExperts requires >= 5.0) ==="
_pin_transformers_for_moe

if [[ "${INSTALL_HF_PTQ:-0}" == "1" ]]; then
  echo "=== installing hf_ptq example requirements (optional quant path) ==="
  _uv_pip_install_hf_ptq_extras

  if [[ "${SKIP_FLASH_ATTN:-0}" == "1" ]]; then
    echo "SKIP_FLASH_ATTN=1: not installing flash-attn"
  elif python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already installed"
  else
    echo "=== building flash-attn (--no-build-isolation; may fail without matching CUDA) ==="
    if ! "$UV" pip install 'flash-attn>=2.6.0' --no-build-isolation; then
      echo "WARN: flash-attn install failed. hf_ptq may use sdpa/eager attention instead." >&2
      echo "      Or set SKIP_FLASH_ATTN=1 if flash-attn is not needed." >&2
    fi
  fi
else
  echo "=== skipping hf_ptq requirements (deploy-only setup) ==="
  echo "    For Gate A quantization: INSTALL_HF_PTQ=1 bash setup_env.sh"
fi

echo "=== smoke import ==="
# shellcheck disable=SC1091
source "${MODELOPT_VENV}/bin/activate"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
python - <<'PY'
import modelopt
import transformers
import tensorrt_llm
from modelopt.deploy.llm import LLM  # noqa: F401
import torch

print("modelopt:", getattr(modelopt, "__version__", "unknown"))
print("transformers:", transformers.__version__)
print("tensorrt_llm:", tensorrt_llm.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("import OK")
PY

echo "=== applying TRT-LLM patches (transformers 5.x + Qwen MoE W4A8_CUSTOM) ==="
python - <<'PY'
from modelopt.deploy.trtllm_transformers5_patch import apply_trtllm_transformers5_compat_patch
from modelopt.deploy.trtllm_qwen_moe_patch import apply_trtllm_qwen_moe_patches

t5 = apply_trtllm_transformers5_compat_patch()
w4 = apply_trtllm_qwen_moe_patches()
print("transformers5:", t5 or "already applied")
print("W4A8_CUSTOM:", w4 or "already applied")
PY

echo "=== done ==="
echo "Activate with: source ${MODELOPT_VENV}/bin/activate"
echo "Deploy:       source ${SCRIPT_DIR}/_env.sh && python ${SCRIPT_DIR}/deploy_trtllm.py ..."
