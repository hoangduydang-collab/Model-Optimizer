#!/usr/bin/env bash
# Shared cluster env + venv activation for modelopt-test scripts.
#
# Venv lives at Model-Optimizer/.venv (gitignored) unless MODELOPT_VENV is set.
# Intentionally overrides VENV from env.sh (which may point at venvs/main).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

MODELOPT_VENV="${MODELOPT_VENV:-${MODEL_OPT_REPO}/.venv}"

if [[ ! -f "${MODELOPT_VENV}/bin/activate" ]]; then
  echo "ERROR: venv not found at ${MODELOPT_VENV}" >&2
  echo "Run: bash ${SCRIPT_DIR}/setup_env.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${MODELOPT_VENV}/bin/activate"
