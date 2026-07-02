#!/bin/bash
#SBATCH --job-name=modelopt-quant
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/modelopt-quant-%j.out
#
# Gate A: ModelOpt w4a8_awq on Qwen3-30B-A3B
#
#   cd /mnt/nfs/hoangduy/projects/Model-Optimizer
#   sbatch modelopt-test/slurm/quant.sbatch

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${TEST_DIR}/run_quant.sh"
echo "QUANT JOB DONE (rc=$?)"
