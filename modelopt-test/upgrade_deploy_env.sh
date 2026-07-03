#!/usr/bin/env bash
# Rebuild or refresh the **deploy** venv only (.venv-deploy).
#
# Quant/export uses .venv-quant (transformers 5.x) — see setup_env_quant.sh.
#
# Usage:
#   bash upgrade_deploy_env.sh
#   RECREATE_VENV=1 bash upgrade_deploy_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/setup_env_deploy.sh"
else
  echo "Refreshing deploy venv (set RECREATE_VENV=1 to wipe and recreate)"
  bash "${SCRIPT_DIR}/setup_env_deploy.sh"
fi
