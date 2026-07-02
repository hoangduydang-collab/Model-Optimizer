#!/usr/bin/env bash
# Triage control: int4_awq (W4A16) — TRT-LLM matrix lists this for Qwen3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QFORMAT=int4_awq
export EXPORT_PATH="${EXPORT_PATH:-/mnt/nfs/hoangduy/artifacts/modelopt_qwen3_int4_awq}"

echo "=== Control quantize: int4_awq -> $EXPORT_PATH ==="
bash "${SCRIPT_DIR}/run_quant.sh"

echo ""
echo "=== Control deploy (paste output back) ==="
source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"
source /mnt/nfs/hoangduy/venvs/modelopt/bin/activate

python "${SCRIPT_DIR}/deploy_trtllm.py" \
  --checkpoint_dir "$EXPORT_PATH" \
  --tp 2 \
  --prompt "The capital of France is"
