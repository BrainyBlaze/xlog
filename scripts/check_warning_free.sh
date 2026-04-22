#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

warning_target_root="${CARGO_WARNING_TARGET_DIR:-target/warning-check}"
rm -rf "$warning_target_root"

run_warning_clean_build() {
  local label="$1"
  shift

  echo "==> ${label}"
  CARGO_TARGET_DIR="${warning_target_root}/${label}" \
  CARGO_INCREMENTAL=0 \
  RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }-Dwarnings" \
    cargo build --locked "$@"
}

run_warning_clean_build "xlog-cli-host-io" -p xlog-cli --features host-io
run_warning_clean_build "pyxlog" -p pyxlog
