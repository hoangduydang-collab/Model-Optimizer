#!/usr/bin/env bash
# Full re-calibration + export: Qwen3-30B-A3B w4a8_awq (Gate A).
#
# MoE: routes 100% of experts during calibration (moe_calib_experts_ratio=1.0).
# Calib: diverse public mix (news + web + chat + wiki); 1024 samples @ 2048 seq.
#
# Usage:
#   bash modelopt-test/run_quant_qwen3_w4a8.sh
#   WIPE_EXPORT_PATH=1 bash modelopt-test/run_quant_qwen3_w4a8.sh   # delete prior artifact first
#   sbatch modelopt-test/slurm/quant.sbatch   # uses this script when QUANT_SCRIPT is unset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
export QFORMAT="${QFORMAT:-w4a8_awq}"

# All experts see calibration tokens (required for fused Qwen3MoeExperts AWQ).
export MOE_CALIB_EXPERTS_RATIO="${MOE_CALIB_EXPERTS_RATIO:-1.0}"

# Public datasets only (no gated Nemotron). Samples split evenly across members.
export CALIB_DATASET="${CALIB_DATASET:-cnn_dailymail,pile,magpie,wikitext}"
export CALIB_SIZE="${CALIB_SIZE:-1024}"
export CALIB_SEQ="${CALIB_SEQ:-2048}"
export BATCH_SIZE="${BATCH_SIZE:-1}"

# New artifact dir — keep v1 until Gate C passes on v2.
export EXPORT_PATH="${EXPORT_PATH:-/mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq_v2}"

if [[ "${MOE_CALIB_EXPERTS_RATIO}" != "1.0" && "${MOE_CALIB_EXPERTS_RATIO}" != "1" ]]; then
  echo "WARN: MOE_CALIB_EXPERTS_RATIO=${MOE_CALIB_EXPERTS_RATIO} (expected 1.0 for full expert coverage)" >&2
fi

if [[ -e "${EXPORT_PATH}" ]]; then
  if [[ "${WIPE_EXPORT_PATH:-0}" == "1" ]]; then
    echo "WIPE_EXPORT_PATH=1: removing ${EXPORT_PATH}"
    rm -rf "${EXPORT_PATH}"
  else
    echo "ERROR: export path already exists: ${EXPORT_PATH}" >&2
    echo "Set WIPE_EXPORT_PATH=1 to remove it, or EXPORT_PATH=... for a new directory." >&2
    exit 1
  fi
fi

echo "=== Qwen3 MoE w4a8_awq full re-calib ==="
echo "moe_calib_experts_ratio=${MOE_CALIB_EXPERTS_RATIO}"
echo "dataset=${CALIB_DATASET} calib_size=${CALIB_SIZE} calib_seq=${CALIB_SEQ}"
echo "export_path=${EXPORT_PATH}"

bash "${SCRIPT_DIR}/run_quant.sh" "$@"

echo ""
echo "=== Next steps ==="
echo "  source modelopt-test/_env_quant.sh"
echo "  python modelopt-test/inspect_ckpt.py ${EXPORT_PATH}"
echo "  bash modelopt-test/run_deploy.sh --checkpoint_dir ${EXPORT_PATH} --tp 2"
