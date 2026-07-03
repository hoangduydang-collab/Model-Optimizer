#!/usr/bin/env bash
# Create both modelopt-test venvs (recommended for the team).
#
#   .venv-quant  — Gate A: calib, export, restore (transformers 5.x)
#   .venv-deploy — Gate C: TensorRT-LLM deploy (transformers 4.57.3)
#
# Usage:
#   bash setup_env.sh
#   RECREATE_VENV=1 bash setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== modelopt-test dual-venv setup ==="
echo "Creating quant + deploy environments (see setup_env_quant.sh / setup_env_deploy.sh)"

bash "${SCRIPT_DIR}/setup_env_quant.sh"
echo ""
bash "${SCRIPT_DIR}/setup_env_deploy.sh"

echo ""
echo "=== dual-venv setup complete ==="
echo "Gate A:  source ${SCRIPT_DIR}/_env_quant.sh && bash ${SCRIPT_DIR}/run_quant.sh"
echo "Gate C:  source ${SCRIPT_DIR}/_env_deploy.sh && python ${SCRIPT_DIR}/deploy_trtllm.py ..."
