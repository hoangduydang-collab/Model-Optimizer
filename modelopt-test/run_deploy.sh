#!/usr/bin/env bash
# Gate C: TensorRT-LLM deploy (always uses .venv-deploy/bin/python).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_VENV="${MODEL_OPT_REPO}/.venv-deploy"

if [[ ! -x "${DEPLOY_VENV}/bin/python" ]]; then
  echo "ERROR: deploy venv not found: ${DEPLOY_VENV}" >&2
  echo "Run: bash ${SCRIPT_DIR}/setup_env_deploy.sh" >&2
  exit 1
fi

unset MODELOPT_VENV
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_cache_helpers.sh"
_clear_stale_flashinfer_cache "${MODEL_OPT_REPO}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env_deploy.sh"

exec "${DEPLOY_VENV}/bin/python" "${SCRIPT_DIR}/deploy_trtllm.py" "$@"
