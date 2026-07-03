#!/usr/bin/env bash
# Activate Gate C TensorRT-LLM deploy venv (transformers 4.57.3).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODELOPT_PROFILE=deploy
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
