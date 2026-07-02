#!/usr/bin/env bash
# Gate A: ModelOpt PTQ + unified HF export via hf_ptq.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
HF_PTQ_DIR="${MODEL_OPT_REPO}/examples/hf_ptq"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
QFORMAT="${QFORMAT:-w4a8_awq}"
# hf_ptq default (cnn_nemotron_v2_mix) needs gated Nemotron v2 on HF Hub.
# cnn_dailymail is public and supported by ModelOpt out of the box.
CALIB_DATASET="${CALIB_DATASET:-cnn_dailymail}"
CALIB_SIZE="${CALIB_SIZE:-512}"
CALIB_SEQ="${CALIB_SEQ:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXPORT_PATH="${EXPORT_PATH:-/mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq}"

echo "=== Gate A: ModelOpt quantize ==="
echo "host=$(hostname) date=$(date -Is)"
echo "repo=$MODEL_OPT_REPO"
echo "model=$MODEL qformat=$QFORMAT dataset=$CALIB_DATASET calib_size=$CALIB_SIZE export_path=$EXPORT_PATH"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

cd "$HF_PTQ_DIR"
export TORCH_COMPILE_DISABLE=1

python hf_ptq.py \
  --pyt_ckpt_path "$MODEL" \
  --qformat "$QFORMAT" \
  --dataset "$CALIB_DATASET" \
  --calib_size "$CALIB_SIZE" \
  --calib_seq "$CALIB_SEQ" \
  --batch_size "$BATCH_SIZE" \
  --export_path "$EXPORT_PATH" \
  --trust_remote_code \
  --skip_generate \
  "$@"

echo "=== Gate A DONE ==="
echo "checkpoint: $EXPORT_PATH"
echo "calib cache: $EXPORT_PATH/.modelopt_calib_checkpoint.pth (for export-only retry)"
echo "Next: python ${SCRIPT_DIR}/inspect_ckpt.py $EXPORT_PATH"
echo "Export-only retry: bash ${SCRIPT_DIR}/run_export_only.sh"
