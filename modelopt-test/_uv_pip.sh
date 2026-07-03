#!/usr/bin/env bash
# Shared uv/pip helpers for modelopt-test scripts (Polaris cluster).
# Cluster uv may not support ``uv pip install --no-upgrade-package``; fall back to
# ``python -m pip`` inside the active venv for pins that must not bump torch.

# shellcheck shell=bash

_reinstall_editable_modelopt() {
  local repo="${1:?MODEL_OPT_REPO required}"
  "$UV" pip uninstall -y nvidia-modelopt 2>/dev/null || true
  "$UV" pip install -e "${repo}[hf]"
}

_pin_transformers_for_moe() {
  local spec="${TRANSFORMERS_SPEC:-transformers>=5.0,<5.13}"
  echo "TRANSFORMERS_SPEC=$spec"
  if "$UV" pip install --help 2>&1 | grep -q 'no-upgrade-package'; then
    "$UV" pip install \
      --no-upgrade-package torch \
      --no-upgrade-package triton \
      --no-upgrade-package cuda-toolkit \
      --no-upgrade-package nvidia-cublas \
      --no-upgrade-package nvidia-cuda-runtime \
      --no-upgrade-package nvidia-cuda-nvrtc \
      --no-upgrade-package nvidia-nccl-cu13 \
      "${spec}"
  else
    echo "Note: uv lacks --no-upgrade-package; using uv pip install (torch should stay pinned)"
    "$UV" pip install "${spec}"
  fi
}

_uv_pip_install_hf_ptq_extras() {
  local packages=(compressed-tensors fire transformers_stream_generator zstandard)
  if "$UV" pip install --help 2>&1 | grep -q 'no-upgrade-package'; then
    "$UV" pip install \
      --no-upgrade-package torch \
      --no-upgrade-package triton \
      --no-upgrade-package cuda-toolkit \
      --no-upgrade-package nvidia-cublas \
      --no-upgrade-package nvidia-cuda-runtime \
      --no-upgrade-package nvidia-cuda-nvrtc \
      --no-upgrade-package nvidia-nccl-cu13 \
      "${packages[@]}"
  else
    echo "Note: uv lacks --no-upgrade-package; using uv pip install for hf_ptq extras"
    "$UV" pip install "${packages[@]}"
  fi
}
