#!/usr/bin/env bash
# Shared uv/pip helpers for modelopt-test scripts (Polaris cluster).

# shellcheck shell=bash

_create_modelopt_venv() {
  local venv_path="${1:?venv path required}"
  if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
    echo "Recreating venv (--clear): $venv_path"
    "$UV" venv --clear --python 3.12 "$venv_path"
  elif [[ -f "${venv_path}/bin/activate" ]]; then
    echo "Reusing venv: $venv_path"
  else
    echo "Creating venv: $venv_path"
    "$UV" venv --python 3.12 "$venv_path"
  fi
}

_reinstall_editable_modelopt() {
  local repo="${1:?MODEL_OPT_REPO required}"
  while "$UV" pip uninstall -y nvidia-modelopt 2>/dev/null | grep -q 'Uninstalled'; do
    :
  done
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
    echo "Note: uv lacks --no-upgrade-package; using uv pip install"
    "$UV" pip install "${spec}"
  fi
}

_pin_transformers_for_deploy() {
  local spec="${TRANSFORMERS_SPEC:-transformers==4.57.3}"
  echo "TRANSFORMERS_SPEC=$spec (TRT-LLM 1.2.x pin)"
  "$UV" pip install "${spec}"
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
    "$UV" pip install "${packages[@]}"
  fi
}

_install_flash_attn_optional() {
  if [[ "${SKIP_FLASH_ATTN:-0}" == "1" ]]; then
    echo "SKIP_FLASH_ATTN=1: not installing flash-attn"
  elif python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already installed"
  else
    echo "=== building flash-attn (--no-build-isolation; may fail without matching CUDA) ==="
    if ! "$UV" pip install 'flash-attn>=2.6.0' --no-build-isolation; then
      echo "WARN: flash-attn install failed. hf_ptq may use sdpa/eager attention instead." >&2
    fi
  fi
}

_remove_legacy_trtllm_pth() {
  python - <<'PY'
import site
from pathlib import Path

legacy = Path(site.getsitepackages()[0]) / "modelopt_trtllm_w4a8_custom.pth"
if legacy.exists():
    legacy.unlink()
    print(f"Removed legacy hook: {legacy}")
else:
    print("No legacy .pth found")
PY
}
