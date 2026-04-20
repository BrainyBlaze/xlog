#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/preflight_release_publish.sh [--dry-run]

Validate the release publish surface that can be checked before crates.io knows
about a new interdependent workspace version wave.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$repo_root"

publishable_crates=(
  xlog-core
  xlog-ir
  xlog-cuda
  xlog-stats
  xlog-runtime
  xlog-logic
  xlog-solve
  xlog-gpu
  xlog-prob
  xlog-cli
)

print_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$dry_run" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_quiet_cmd() {
  print_cmd "$@"
  if [[ "$dry_run" == "1" ]]; then
    return 0
  fi
  "$@" >/dev/null
}

run_cmd python3 scripts/validate_package_metadata.py

for crate in "${publishable_crates[@]}"; do
  run_quiet_cmd cargo package --locked --allow-dirty --no-verify --list -p "$crate"
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run complete."
else
  echo "Release publish preflight checks completed."
fi
