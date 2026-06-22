#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/validate_release_gpu.sh [--mode smoke|release] [--dry-run]

Run the canonical manual release-validation flow on a supported Linux x86_64
CUDA machine. GitHub Actions do not run this script; maintainers run it
manually before dispatching the publish workflow.
EOF
}

mode="smoke"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "--mode requires smoke or release" >&2
        exit 2
      fi
      mode="$2"
      shift 2
      ;;
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

case "$mode" in
  smoke|release)
    ;;
  *)
    echo "unsupported mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

run_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'

  if [[ "$dry_run" == "1" ]]; then
    return 0
  fi

  "$@"
}

require_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    return 0
  fi
  echo "required command not found: $1" >&2
  exit 1
}

if [[ "$dry_run" != "1" ]]; then
  require_cmd cargo
  require_cmd python3
  require_cmd maturin
fi

wheel_dir="${TMPDIR:-/tmp}/xlog-wheel-validation"
bundle_dir="${TMPDIR:-/tmp}/xlog-cli-validation"

rm -rf "$wheel_dir" "$bundle_dir"
mkdir -p "$wheel_dir" "$bundle_dir"

cd "$repo_root"

# Hard gate: this script certifies GPU behavior. The test harness turns
# CUDA-init failures into loud panics under XLOG_REQUIRE_CUDA=1 (see
# crates/xlog-cuda-tests/src/harness/provider.rs and the require_cuda_guard
# tests), so a CPU-only machine can never satisfy this gate via the
# skip-on-missing-device paths.
export XLOG_REQUIRE_CUDA=1
if [[ "$dry_run" != "1" ]]; then
  require_cmd nvidia-smi
  if ! nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
    echo "FATAL: no CUDA GPU visible to nvidia-smi; refusing to run the GPU release gate" >&2
    exit 1
  fi
fi

run_cmd python3 scripts/xlog_doctor.py --workflow release
run_cmd bash scripts/preflight_release_publish.sh
run_cmd cargo build --workspace --locked --release --exclude pyxlog
run_cmd cargo build --locked --release -p xlog-cli --features host-io
run_cmd bash scripts/stage_pyxlog_kernels.sh
run_cmd maturin build -m crates/pyxlog/Cargo.toml --release --compatibility linux --out "$wheel_dir"
run_cmd bash scripts/package_cli_release.sh --output "$bundle_dir"

if [[ "$mode" == "smoke" ]]; then
  run_cmd cargo test -p xlog-cuda-tests --test quick_smoke --release -- --nocapture
else
  run_cmd cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture
fi

run_cmd ./target/release/xlog run examples/xlog/00-basics/01_tc_reachability.xlog

run_cmd python3 -c 'import pathlib, sys, tarfile; dist = pathlib.Path(sys.argv[1]); archives = sorted(dist.glob("xlog-v*.tar.gz")); assert archives, "no CLI release archive built"; names = tarfile.open(archives[0], "r:gz").getnames(); assert any(name.endswith("/xlog") for name in names), "CLI archive is missing xlog"; assert any("/kernels/" in name for name in names), "CLI archive is missing staged kernels"; print(f"validated CLI archive layout: {archives[0]}")' "$bundle_dir"
run_cmd python3 -c 'import pathlib, sys, zipfile; dist = pathlib.Path(sys.argv[1]); wheels = sorted(dist.glob("pyxlog-*.whl")); assert wheels, "no pyxlog wheel built"; names = zipfile.ZipFile(wheels[0]).namelist(); assert any(name.startswith("pyxlog/kernels/") for name in names), "wheel is missing staged kernels"; print(f"validated pyxlog wheel layout: {wheels[0]}")' "$wheel_dir"

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run complete."
else
  echo "GPU release validation complete."
fi
