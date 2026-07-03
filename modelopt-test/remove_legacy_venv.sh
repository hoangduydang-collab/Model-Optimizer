#!/usr/bin/env bash
# Remove deprecated single .venv from Model-Optimizer (safe after dual-venv setup).
#
# Gate A  -> .venv-quant
# Gate C  -> .venv-deploy
#
# Usage:
#   bash remove_legacy_venv.sh
#   DRY_RUN=1 bash remove_legacy_venv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LEGACY_VENV="${MODEL_OPT_REPO}/.venv"

if [[ ! -d "${LEGACY_VENV}" ]]; then
  echo "No legacy .venv at ${LEGACY_VENV} (nothing to do)"
  exit 0
fi

if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" == "${LEGACY_VENV}" ]]; then
  if type deactivate &>/dev/null; then
    deactivate || true
  fi
fi

size="$(du -sh "${LEGACY_VENV}" 2>/dev/null | cut -f1 || echo unknown)"
echo "Legacy .venv found: ${LEGACY_VENV} (${size})"
echo "Dual venvs in use: ${MODEL_OPT_REPO}/.venv-quant, ${MODEL_OPT_REPO}/.venv-deploy"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: would rm -rf ${LEGACY_VENV}"
  exit 0
fi

rm -rf "${LEGACY_VENV}"
echo "Removed ${LEGACY_VENV}"
echo "Use: source modelopt-test/_env_quant.sh  (Gate A)"
echo "     bash modelopt-test/run_deploy.sh ... (Gate C)"
