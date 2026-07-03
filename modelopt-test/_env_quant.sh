#!/usr/bin/env bash
# Activate Gate A quant/export venv (transformers 5.x).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset MODELOPT_VENV
export MODELOPT_PROFILE=quant
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"
