#!/bin/bash
#SBATCH --job-name=modelopt-deploy
#SBATCH --partition=compute
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/modelopt-deploy-%j.out
#
# Gate C: TensorRT-LLM load + sample generate (tp=2)
#
#   cd /mnt/nfs/hoangduy/projects/Model-Optimizer
#   sbatch modelopt-test/slurm/deploy.sbatch

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${TEST_DIR}/_env_deploy.sh"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/nfs/hoangduy/artifacts/modelopt_qwen3_w4a8_awq}"
TP="${TP:-2}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

bash "${TEST_DIR}/run_deploy.sh" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --tp "$TP" \
  --prompt "The capital of France is"

echo "DEPLOY JOB DONE (rc=$?)"
