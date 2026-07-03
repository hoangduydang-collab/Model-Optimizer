#!/usr/bin/env bash
# Gate C: TensorRT-LLM deploy (always uses .venv-deploy/bin/python).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env_deploy.sh"

exec "${MODELOPT_VENV}/bin/python" "${SCRIPT_DIR}/deploy_trtllm.py" "$@"
