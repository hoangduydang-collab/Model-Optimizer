#!/usr/bin/env bash
# Shared cluster env + venv activation for modelopt-test scripts.
#
# Dual venv (recommended):
#   MODELOPT_PROFILE=quant  -> .venv-quant  (Gate A: calib/export)
#   MODELOPT_PROFILE=deploy -> .venv-deploy (Gate C: TRT-LLM)
#
# Convenience wrappers: _env_quant.sh, _env_deploy.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_REPO="${MODEL_OPT_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

case "${MODELOPT_PROFILE:-deploy}" in
  quant)
    MODELOPT_VENV="${MODELOPT_VENV:-${MODEL_OPT_REPO}/.venv-quant}"
    SETUP_HINT="bash ${SCRIPT_DIR}/setup_env_quant.sh"
    ;;
  deploy)
    MODELOPT_VENV="${MODELOPT_VENV:-${MODEL_OPT_REPO}/.venv-deploy}"
    SETUP_HINT="bash ${SCRIPT_DIR}/setup_env_deploy.sh"
    ;;
  *)
    echo "ERROR: unknown MODELOPT_PROFILE=${MODELOPT_PROFILE} (use quant|deploy)" >&2
    exit 1
    ;;
esac

export MODELOPT_VENV MODELOPT_PROFILE

if [[ ! -f "${MODELOPT_VENV}/bin/activate" ]]; then
  echo "ERROR: venv not found: ${MODELOPT_VENV} (profile=${MODELOPT_PROFILE})" >&2
  echo "Run: ${SETUP_HINT}" >&2
  exit 1
fi

# Switching profiles in an interactive shell can leave stale PATH / PYTHONPATH entries.
if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${MODELOPT_VENV}" ]]; then
  if type deactivate &>/dev/null; then
    deactivate || true
  fi
  unset VIRTUAL_ENV
fi

_strip_other_modelopt_venvs_from_path() {
  local entry cleaned=""
  IFS=':' read -ra _path_parts <<< "${PATH}"
  for entry in "${_path_parts[@]}"; do
    if [[ "$entry" =~ /Model-Optimizer/\.venv(-quant|-deploy)?/bin$ || "$entry" == "${MODEL_OPT_REPO}/.venv/bin" ]] \
      && [[ "$entry" != "${MODELOPT_VENV}/bin" ]]; then
      continue
    fi
    cleaned="${cleaned:+$cleaned:}$entry"
  done
  export PATH="$cleaned"
}

_strip_other_modelopt_venvs_from_path

# shellcheck disable=SC1091
source "${MODELOPT_VENV}/bin/activate"

# Editable modelopt only — do not append a session PYTHONPATH that may list another venv.
export PYTHONPATH="${MODEL_OPT_REPO}"

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

_export_openmpi_localhost() {
  export OMPI_MCA_oob="${OMPI_MCA_oob:-tcp}"
  export OMPI_MCA_oob_tcp_if_include="${OMPI_MCA_oob_tcp_if_include:-lo}"
  export OMPI_MCA_oob_tcp_peer_retries="${OMPI_MCA_oob_tcp_peer_retries:-60}"
  export OMPI_MCA_oob_tcp_connect_sleep="${OMPI_MCA_oob_tcp_connect_sleep:-10}"
  export OMPI_MCA_btl="${OMPI_MCA_btl:-self,vader,tcp}"
  export OMPI_MCA_btl_tcp_if_include="${OMPI_MCA_btl_tcp_if_include:-lo}"
  export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
}

_export_nvidia_wheel_libs

if [[ "${MODELOPT_PROFILE}" == "deploy" ]]; then
  _export_openmpi_localhost
fi
