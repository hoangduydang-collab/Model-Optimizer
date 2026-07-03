#!/usr/bin/env bash
# Controlled deploy-env upgrade for Qwen3 W4A8_AWQ on CUDA 12.x (H100).
#
# Root-cause fix for Bug #7: keep TRT-LLM 1.2.1 (CUDA-12 wheel line), pin
# transformers >= 5.0 for fused Qwen3MoeExperts, patch TRT-LLM in-place for
# transformers 5 import compat (NVIDIA upstream pattern).
#
# TRT-LLM 1.3.x requires CUDA 13 / torch cu130 — do NOT install on Polaris CUDA 12.4.
#
# Usage:
#   bash upgrade_deploy_env.sh              # upgrade packages in existing .venv
#   RECREATE_VENV=1 bash upgrade_deploy_env.sh   # wipe + full setup_env.sh first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

# Polaris CUDA 12.x: stay on TRT-LLM 1.2.x. Override only on CUDA 13 nodes.
TENSORRT_LLM_SPEC="${TENSORRT_LLM_SPEC:-tensorrt-llm==1.2.1}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=5.0,<5.13}"

if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
  echo "=== RECREATE_VENV=1: full setup_env.sh ==="
  INSTALL_HF_PTQ="${INSTALL_HF_PTQ:-1}" RECREATE_VENV=1 bash "${SCRIPT_DIR}/setup_env.sh"
  exit 0
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

echo "=== upgrade_deploy_env.sh ==="
echo "host=$(hostname) date=$(date -Is)"
echo "TENSORRT_LLM_SPEC=$TENSORRT_LLM_SPEC"
echo "TRANSFORMERS_SPEC=$TRANSFORMERS_SPEC"

echo "=== 1/5 re-pin editable Model Optimizer ==="
"$UV" pip uninstall -y nvidia-modelopt 2>/dev/null || true
"$UV" pip install -e "${MODEL_OPT_REPO}[hf]"

echo "=== 2/5 (re)install TensorRT-LLM (CUDA 12 line) ==="
"$UV" pip install "${TENSORRT_LLM_SPEC}" --extra-index-url https://pypi.nvidia.com

echo "=== 3/5 pin transformers for fused Qwen3 MoE ==="
NO_UPGRADE=(
  --no-upgrade-package torch
  --no-upgrade-package triton
  --no-upgrade-package cuda-toolkit
  --no-upgrade-package nvidia-cublas
  --no-upgrade-package nvidia-cuda-runtime
  --no-upgrade-package nvidia-cuda-nvrtc
  --no-upgrade-package nvidia-nccl-cu13
)
"$UV" pip install "${NO_UPGRADE[@]}" "${TRANSFORMERS_SPEC}"

echo "=== 4/5 re-pin editable Model Optimizer (after TRT-LLM) ==="
"$UV" pip uninstall -y nvidia-modelopt 2>/dev/null || true
"$UV" pip install -e "${MODEL_OPT_REPO}[hf]"

echo "=== 5/5 apply TRT-LLM patches ==="
python - <<'PY'
from modelopt.deploy.trtllm_transformers5_patch import apply_trtllm_transformers5_compat_patch
from modelopt.deploy.trtllm_qwen_moe_patch import apply_trtllm_qwen_moe_patches

t5 = apply_trtllm_transformers5_compat_patch()
w4 = apply_trtllm_qwen_moe_patches()
print("transformers5 patch:", t5 or "already applied")
print("W4A8_CUSTOM patch:", w4 or "already applied")
PY

echo "=== smoke test (no runtime shim) ==="
python - <<'PY'
import transformers
import tensorrt_llm
from modelopt.deploy.llm import LLM  # noqa: F401

print("transformers:", transformers.__version__)
print("tensorrt_llm:", tensorrt_llm.__version__)
print("modelopt import OK")
PY

echo "=== done ==="
echo "Deploy: source ${SCRIPT_DIR}/_env.sh && python ${SCRIPT_DIR}/deploy_trtllm.py ..."
