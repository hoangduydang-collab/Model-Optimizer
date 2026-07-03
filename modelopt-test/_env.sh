#!/usr/bin/env bash
# Shared cluster env + venv activation for modelopt-test scripts.
#
# Venv lives at Model-Optimizer/.venv (gitignored) unless MODELOPT_VENV is set.
# Intentionally overrides VENV from env.sh (which may point at venvs/main).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

MODELOPT_VENV="${MODELOPT_VENV:-${MODEL_OPT_REPO}/.venv}"

if [[ ! -f "${MODELOPT_VENV}/bin/activate" ]]; then
  echo "ERROR: venv not found at ${MODELOPT_VENV}" >&2
  echo "Run: bash ${SCRIPT_DIR}/setup_env.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${MODELOPT_VENV}/bin/activate"

_export_nvidia_wheel_libs() {
  local py="${MODELOPT_VENV}/bin/python"
  [[ -x "$py" ]] || return 0
  local lib_paths
  lib_paths="$("$py" - <<'PY'
import site
from pathlib import Path

dirs: list[str] = []
for root in site.getsitepackages():
    nvidia = Path(root) / "nvidia"
    if not nvidia.is_dir():
        continue
    for child in nvidia.iterdir():
        lib = child / "lib"
        if lib.is_dir():
            dirs.append(str(lib))
if dirs:
    print(":".join(dirs))
PY
)"
  if [[ -n "$lib_paths" ]]; then
    export LD_LIBRARY_PATH="${lib_paths}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

_export_nvidia_wheel_libs
