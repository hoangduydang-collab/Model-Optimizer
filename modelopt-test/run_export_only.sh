#!/usr/bin/env bash
# Retry HF export only — restores .modelopt_calib_checkpoint.pth saved after calibration.
# Use when quantize succeeded but export_hf_checkpoint failed (no ~1.5h re-calib).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
HF_PTQ_DIR="${MODEL_OPT_REPO}/examples/hf_ptq"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
QFORMAT="${QFORMAT:-w4a8_awq}"
EXPORT_PATH="${EXPORT_PATH:-/mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq}"
CALIB_CKPT="${EXPORT_PATH}/.modelopt_calib_checkpoint.pth"

if [[ ! -f "$CALIB_CKPT" ]]; then
  echo "ERROR: calibration checkpoint not found: $CALIB_CKPT" >&2
  echo "Run bash run_quant.sh first (saves checkpoint after calib, before export)." >&2
  exit 1
fi

echo "=== Gate A export-only retry ==="
echo "host=$(hostname) date=$(date -Is)"
echo "model=$MODEL qformat=$QFORMAT export_path=$EXPORT_PATH"
echo "calib_ckpt=$CALIB_CKPT ($(du -h "$CALIB_CKPT" | cut -f1))"

cd "$HF_PTQ_DIR"
export TORCH_COMPILE_DISABLE=1

python hf_ptq.py \
  --pyt_ckpt_path "$MODEL" \
  --qformat "$QFORMAT" \
  --export_path "$EXPORT_PATH" \
  --export_only \
  --trust_remote_code \
  --skip_generate \
  "$@"

echo "=== Export-only DONE ==="
echo "checkpoint: $EXPORT_PATH"

if [[ "$QFORMAT" == "w4a8_awq" ]]; then
  echo "Preparing checkpoint for TRT-LLM W4A8_CUSTOM MoE..."
  python "${SCRIPT_DIR}/trtllm_w4a8_moe_custom.py" "$EXPORT_PATH"
fi

echo "Next: python ${SCRIPT_DIR}/inspect_ckpt.py $EXPORT_PATH"
