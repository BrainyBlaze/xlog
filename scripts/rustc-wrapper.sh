#!/usr/bin/env bash
set -euo pipefail

# Cargo invokes a rustc wrapper as: <wrapper> <rustc> <args...>
real_rustc="${1:?missing rustc path}"
shift

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
shim_dir="$repo_root/tools/exdev-shim"
shim_so="$shim_dir/libexdev_shim.so"

if [[ ! -f "$shim_so" ]]; then
  "$shim_dir/build.sh" >/dev/null
fi

# Prepend so our shim's symbols win.
if [[ -n "${LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="$shim_so $LD_PRELOAD"
else
  export LD_PRELOAD="$shim_so"
fi

exec "$real_rustc" "$@"

