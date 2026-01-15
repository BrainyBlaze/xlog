#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

cargo build -p xlog-logic --example xlog_run

POSITIVE=$(find examples/xlog -type f -name "*.xlog" -not -path "examples/xlog/90-negative-tests/*" | sort)
NEGATIVE=$(find examples/xlog/90-negative-tests -type f -name "*.xlog" | sort)

for file in $POSITIVE; do
  echo "[POSITIVE] $file"
  cargo run -q -p xlog-logic --example xlog_run -- "$file"
  echo "OK: $file"
  echo
 done

for file in $NEGATIVE; do
  echo "[NEGATIVE] $file"
  if cargo run -q -p xlog-logic --example xlog_run -- "$file"; then
    echo "ERROR: negative example unexpectedly succeeded: $file" >&2
    exit 1
  else
    echo "OK (failed as expected): $file"
  fi
  echo
 done
