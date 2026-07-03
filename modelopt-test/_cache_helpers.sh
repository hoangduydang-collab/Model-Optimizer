#!/usr/bin/env bash
# Helpers to clear JIT caches that pin absolute paths to deleted venvs.

# shellcheck shell=bash

_clear_stale_flashinfer_cache() {
  local model_opt_repo="${1:?MODEL_OPT_REPO required}"
  local cache_root="${FLASHINFER_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/flashinfer}"
  local legacy_marker="${model_opt_repo}/.venv/"

  if [[ ! -d "${cache_root}" ]]; then
    return 0
  fi

  if grep -rqF "${legacy_marker}" "${cache_root}" 2>/dev/null; then
    echo "Clearing stale flashinfer JIT cache: ${cache_root}"
    echo "  (build files still referenced deleted ${legacy_marker})"
    rm -rf "${cache_root}"
    return 0
  fi

  return 0
}
