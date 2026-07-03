#!/usr/bin/env bash
# Gate A: ModelOpt quant/export venv (transformers 5.x, fused Qwen3MoeExperts).
#
# Usage:
#   bash setup_env_quant.sh
#   RECREATE_VENV=1 bash setup_env_quant.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_uv_pip.sh"

MODELOPT_VENV="${MODELOPT_VENV_OVERRIDE:-${MODEL_OPT_REPO}/.venv-quant}"

echo "=== modelopt-test setup (QUANT) ==="
echo "HOME=$HOME"
echo "MODELOPT_VENV=$MODELOPT_VENV"
echo "MODEL_OPT_REPO=$MODEL_OPT_REPO"

_create_modelopt_venv "$MODELOPT_VENV"
# shellcheck disable=SC1091
source "${MODELOPT_VENV}/bin/activate"

echo "=== installing Model Optimizer (editable) ==="
_reinstall_editable_modelopt "${MODEL_OPT_REPO}"

echo "=== pin transformers (fused Qwen3MoeExperts requires >= 5.0) ==="
_pin_transformers_for_moe

echo "=== installing hf_ptq example requirements ==="
_uv_pip_install_hf_ptq_extras
_install_flash_attn_optional

echo "=== smoke test (quant) ==="
python - <<'PY'
import modelopt
import transformers

print("modelopt:", getattr(modelopt, "__version__", "unknown"))
print("modelopt.__file__:", modelopt.__file__)
print("transformers:", transformers.__version__)
assert int(transformers.__version__.split(".")[0]) >= 5, "need transformers 5.x"
print("quant env OK")
PY

echo "=== done (quant) ==="
echo "Activate:  source ${MODELOPT_VENV}/bin/activate"
echo "Or:        source ${SCRIPT_DIR}/_env_quant.sh"
echo "Quantize:  bash ${SCRIPT_DIR}/run_quant.sh"
