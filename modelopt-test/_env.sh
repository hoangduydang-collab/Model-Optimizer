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
  legacy)
    MODELOPT_VENV="${MODELOPT_VENV:-${MODEL_OPT_REPO}/.venv}"
    SETUP_HINT="bash ${SCRIPT_DIR}/setup_env.sh  # migrate to dual venv"
    ;;
  *)
    echo "ERROR: unknown MODELOPT_PROFILE=${MODELOPT_PROFILE} (use quant|deploy|legacy)" >&2
    exit 1
    ;;
esac

export MODELOPT_VENV MODELOPT_PROFILE

if [[ ! -f "${MODELOPT_VENV}/bin/activate" ]]; then
  echo "ERROR: venv not found: ${MODELOPT_VENV} (profile=${MODELOPT_PROFILE})" >&2
  echo "Run: ${SETUP_HINT}" >&2
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
