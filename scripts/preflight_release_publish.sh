#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

cd "$repo_root"

if command -v release-plz >/dev/null 2>&1 \
  && [[ -n "${GITHUB_TOKEN:-}" ]] \
  && [[ -n "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  cmd=(
    release-plz
    release
    --dry-run
    --git-token "$GITHUB_TOKEN"
    --token "$CARGO_REGISTRY_TOKEN"
    --config release-plz.toml
    -o json
  )
else
  cmd=(
    cargo
    publish
    -p xlog-cuda
    --dry-run
    --allow-dirty
  )
fi

printf '+'
for arg in "${cmd[@]}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${cmd[@]}"
