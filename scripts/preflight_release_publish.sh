#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

cd "$repo_root"

cmd=(
  cargo
  publish
  -p xlog-cuda
  --dry-run
  --allow-dirty
)

printf '+'
for arg in "${cmd[@]}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${cmd[@]}"
