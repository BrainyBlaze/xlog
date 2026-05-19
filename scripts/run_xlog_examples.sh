#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

ensure_cuda_loader_compat() {
  # Some WSL environments expose only libcuda.so.1; cudarc needs libcuda.so/libnvcuda.so.
  if ldconfig -p 2>/dev/null | grep -qE 'libcuda\.so([^.]|$)'; then
    return
  fi

  local wsl_cuda="/usr/lib/wsl/lib/libcuda.so.1"
  if [[ ! -f "$wsl_cuda" ]]; then
    return
  fi

  local shim_dir="/tmp/xlog-cuda-shim"
  mkdir -p "$shim_dir"
  ln -sf "$wsl_cuda" "${shim_dir}/libcuda.so"
  ln -sf "$wsl_cuda" "${shim_dir}/libnvcuda.so"
  export LD_LIBRARY_PATH="${shim_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "INFO: Added CUDA loader shim at ${shim_dir}"
}

ensure_cuda_loader_compat

cargo_profile_args=()
if [[ "${XLOG_EXAMPLES_RELEASE:-0}" == "1" ]]; then
  cargo_profile_args+=(--release)
fi

cargo build "${cargo_profile_args[@]}" -p xlog-integration --bin xlog_run

POSITIVE=$(find examples/xlog -type f -name "*.xlog" -not -path "examples/xlog/90-negative-tests/*" | sort)
NEGATIVE=$(find examples/xlog/90-negative-tests -type f -name "*.xlog" | sort)

for file in $POSITIVE; do
  echo "[POSITIVE] $file"
  cargo run -q "${cargo_profile_args[@]}" -p xlog-integration --bin xlog_run -- "$file"
  echo "OK: $file"
  echo
 done

for file in $NEGATIVE; do
  echo "[NEGATIVE] $file"
  if cargo run -q "${cargo_profile_args[@]}" -p xlog-integration --bin xlog_run -- "$file"; then
    echo "ERROR: negative example unexpectedly succeeded: $file" >&2
    exit 1
  else
    echo "OK (failed as expected): $file"
  fi
  echo
 done
