#!/usr/bin/env bash
# Start Qwen3 w4a8_awq full re-calib in a detached tmux session (survives SSH disconnect).
#
# Usage:
#   bash modelopt-test/start_quant_tmux.sh
#   WIPE_EXPORT_PATH=1 bash modelopt-test/start_quant_tmux.sh
#   tmux attach -t qwen3-w4a8-quant

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION="${QUANT_TMUX_SESSION:-qwen3-w4a8-quant}"
LOG_DIR="${WORK_ROOT:-/mnt/nfs/hoangduy}/logs"
LOG_FILE="${LOG_DIR}/qwen3-w4a8-quant-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${SESSION}" >&2
  echo "  tmux attach -t ${SESSION}" >&2
  echo "  tmux kill-session -t ${SESSION}   # to replace" >&2
  exit 1
fi

WIPE_ENV=""
if [[ "${WIPE_EXPORT_PATH:-0}" == "1" ]]; then
  WIPE_ENV="export WIPE_EXPORT_PATH=1"
fi

tmux new-session -d -s "${SESSION}" bash -lc "
  set -euo pipefail
  cd '${MODEL_OPT_REPO}'
  source modelopt-test/_env_quant.sh
  ${WIPE_ENV}
  echo 'Log: ${LOG_FILE}'
  echo 'Started: '\$(date -Is)
  exec bash modelopt-test/run_quant_qwen3_w4a8.sh 2>&1 | tee '${LOG_FILE}'
"

echo "Started tmux session: ${SESSION}"
echo "  attach:  tmux attach -t ${SESSION}"
echo "  detach:  Ctrl-b then d"
echo "  log:     ${LOG_FILE}"
echo "  tail:    tail -f ${LOG_FILE}"
