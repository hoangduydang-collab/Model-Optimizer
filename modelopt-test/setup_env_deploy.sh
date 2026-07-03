#!/usr/bin/env bash
# Gate C: TensorRT-LLM deploy venv (tensorrt-llm 1.2.1 + transformers 4.57.3).
#
# Usage:
#   bash setup_env_deploy.sh
#   RECREATE_VENV=1 bash setup_env_deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_uv_pip.sh"

MODELOPT_VENV="${MODELOPT_VENV:-${MODEL_OPT_REPO}/.venv-deploy}"
TENSORRT_LLM_SPEC="${TENSORRT_LLM_SPEC:-tensorrt-llm==1.2.1}"

echo "=== modelopt-test setup (DEPLOY) ==="
echo "HOME=$HOME"
echo "MODELOPT_VENV=$MODELOPT_VENV"
echo "MODEL_OPT_REPO=$MODEL_OPT_REPO"
echo "TENSORRT_LLM_SPEC=$TENSORRT_LLM_SPEC"

_create_modelopt_venv "$MODELOPT_VENV"
# shellcheck disable=SC1091
source "${MODELOPT_VENV}/bin/activate"

_remove_legacy_trtllm_pth

echo "=== installing Model Optimizer (editable) ==="
_reinstall_editable_modelopt "${MODEL_OPT_REPO}"

echo "=== installing mpi4py (required by modelopt.deploy.llm) ==="
"$UV" pip install mpi4py

echo "=== installing TensorRT-LLM (CUDA 12 line) ==="
"$UV" pip install "${TENSORRT_LLM_SPEC}" --extra-index-url https://pypi.nvidia.com

echo "=== installing CUDA runtime libs for tensorrt-llm wheels ==="
"$UV" pip install nvidia-cublas nvidia-cudnn --extra-index-url https://pypi.nvidia.com

echo "=== re-pin editable Model Optimizer (tensorrt-llm may replace PyPI modelopt) ==="
_reinstall_editable_modelopt "${MODEL_OPT_REPO}"

echo "=== pin transformers (TRT-LLM 1.2.x expects 4.57.3) ==="
_pin_transformers_for_deploy

echo "=== applying TRT-LLM W4A8_CUSTOM Qwen MoE patches ==="
export PYTHONPATH="${MODEL_OPT_REPO}"
python - <<'PY'
from modelopt.deploy.trtllm_qwen_moe_patch import apply_trtllm_qwen_moe_patches

patched = apply_trtllm_qwen_moe_patches()
print("W4A8_CUSTOM:", patched or "already applied")
PY

echo "=== smoke test (deploy) ==="
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env_deploy.sh"
python - <<'PY'
import modelopt
import transformers
import tensorrt_llm
from modelopt.deploy.llm import LLM  # noqa: F401
import torch

print("modelopt:", getattr(modelopt, "__version__", "unknown"))
print("modelopt.__file__:", modelopt.__file__)
print("transformers:", transformers.__version__)
print("tensorrt_llm:", tensorrt_llm.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
assert transformers.__version__.startswith("4."), "deploy env must use transformers 4.x"
print("deploy env OK")
PY

echo "=== done (deploy) ==="
echo "Activate: source ${SCRIPT_DIR}/_env_deploy.sh"
echo "Deploy:   python ${SCRIPT_DIR}/deploy_trtllm.py --checkpoint_dir ... --tp 2"
