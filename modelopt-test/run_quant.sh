#!/usr/bin/env bash
# Gate A: ModelOpt PTQ + unified HF export via hf_ptq.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
HF_PTQ_DIR="${MODEL_OPT_REPO}/examples/hf_ptq"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"
# shellcheck disable=SC1091
source /mnt/nfs/hoangduy/venvs/modelopt/bin/activate

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
QFORMAT="${QFORMAT:-w4a8_awq}"
CALIB_SIZE="${CALIB_SIZE:-512}"
CALIB_SEQ="${CALIB_SEQ:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXPORT_PATH="${EXPORT_PATH:-/mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq}"

echo "=== Gate A: ModelOpt quantize ==="
echo "host=$(hostname) date=$(date -Is)"
echo "repo=$MODEL_OPT_REPO"
echo "model=$MODEL qformat=$QFORMAT calib_size=$CALIB_SIZE export_path=$EXPORT_PATH"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

cd "$HF_PTQ_DIR"
export TORCH_COMPILE_DISABLE=1

python hf_ptq.py \
  --pyt_ckpt_path "$MODEL" \
  --qformat "$QFORMAT" \
  --calib_size "$CALIB_SIZE" \
  --calib_seq "$CALIB_SEQ" \
  --batch_size "$BATCH_SIZE" \
  --export_path "$EXPORT_PATH" \
  --trust_remote_code \
  --skip_generate \
  "$@"

echo "=== Gate A DONE ==="
echo "checkpoint: $EXPORT_PATH"
echo "Next: python ${SCRIPT_DIR}/inspect_ckpt.py $EXPORT_PATH"
