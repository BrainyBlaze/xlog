#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/validate_release_gpu.sh [--mode smoke|release] [--dry-run]

Run the canonical manual release-validation flow on a supported Linux x86_64
CUDA machine. GitHub Actions do not run this script; maintainers run it
manually before dispatching the publish workflow.

XLOG_PINNED_CORPUS_ROOT must name the exact clean pinned corpus checkout.
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

run_exact_rust_gate() {
  local label="$1"
  local expected_passed="$2"
  shift 2
  print_cmd "$@"

  if [[ "$dry_run" == "1" ]]; then
    return 0
  fi

  local log_file
  local cargo_status
  local tee_status
  local parser_status=0
  local -a pipeline_status
  log_file="$(mktemp "${TMPDIR:-/tmp}/xlog-resident-gate.XXXXXX")"

  set +e
  "$@" 2>&1 | tee "$log_file"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  cargo_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"

  python3 - "$log_file" "$cargo_status" "$tee_status" "$expected_passed" "$label" <<'PY' || parser_status=$?
import pathlib
import re
import sys

output = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
cargo_status = int(sys.argv[2])
tee_status = int(sys.argv[3])
expected_passed = int(sys.argv[4])
label = sys.argv[5]
summary_lines = re.findall(r"(?im)^\s*test result:.*$", output)
summary = None
if len(summary_lines) == 1:
    summary = re.search(
        r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; "
        r"(\d+) ignored; (\d+) measured; (\d+) filtered out",
        summary_lines[0],
    )
problems = []
if cargo_status != 0:
    problems.append(f"cargo test exited with status {cargo_status}")
if tee_status != 0:
    problems.append(f"tee exited with status {tee_status}")
if len(summary_lines) != 1:
    problems.append(
        f"expected one Rust test summary, found {len(summary_lines)}"
    )
elif summary is None:
    problems.append("Rust test summary did not match the expected format")
else:
    status, passed, failed, ignored, measured, _filtered = summary.groups()
    actual = tuple(map(int, (passed, failed, ignored, measured)))
    expected = (expected_passed, 0, 0, 0)
    if status != "ok" or actual != expected:
        problems.append(
            "expected "
            f"{expected_passed} passed; 0 failed; 0 ignored; 0 measured, "
            f"got {passed} passed; {failed} failed; {ignored} ignored; "
            f"{measured} measured with status {status}"
        )
skipped = len(re.findall(r"(?im)^\s*Skipping test:", output))
if skipped:
    problems.append(f"{skipped} tests skipped CUDA execution")
if problems:
    raise SystemExit(f"{label}: " + "; ".join(problems))
print(
    f"{label}: exact Rust summary {expected_passed} passed; "
    "0 failed; 0 ignored; 0 measured"
)
PY

  rm -f "$log_file"
  return "$parser_status"
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
python_install_dir="${TMPDIR:-/tmp}/xlog-python-wheel-site"

rm -rf "$wheel_dir" "$bundle_dir" "$python_install_dir"
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

if [[ -z "${XLOG_PINNED_CORPUS_ROOT:-}" ]]; then
  if [[ "$dry_run" == "1" ]]; then
    export XLOG_PINNED_CORPUS_ROOT=/path/to/clean-pinned-corpus
  else
    echo "FATAL: XLOG_PINNED_CORPUS_ROOT must name the clean pinned corpus checkout" >&2
    exit 1
  fi
elif [[ "$dry_run" != "1" && ! -d "$XLOG_PINNED_CORPUS_ROOT" ]]; then
  echo "FATAL: XLOG_PINNED_CORPUS_ROOT is not a directory: $XLOG_PINNED_CORPUS_ROOT" >&2
  exit 1
fi
export XLOG_PINNED_CORPUS_ROOT

run_cmd python3 scripts/xlog_doctor.py --workflow release
run_cmd bash scripts/preflight_release_publish.sh
run_cmd cargo build --workspace --locked --release --exclude pyxlog
run_cmd cargo build --locked --release -p xlog-cli --features host-io
run_cmd cargo test --locked --release \
  -p xlog-cuda \
  --lib \
  -- \
  --nocapture \
  --test-threads=1
run_cmd cargo test --locked --release \
  -p xlog-cli \
  --features host-io \
  --test prob_cli_tests \
  -- \
  --nocapture \
  --test-threads=1
run_cmd cargo test --locked --release \
  -p xlog-prob \
  --features host-io \
  --test epistemic_prob_gpu_accepted_evidence \
  --test epistemic_prob_production_reuse \
  -- \
  --nocapture \
  --test-threads=1
run_exact_rust_gate "resident graph runtime module" 22 \
  cargo test --locked --release -p xlog-runtime --lib \
  --features resident-graph-tests \
  executor::resident_graph_tests -- \
  --nocapture --test-threads=1
run_exact_rust_gate "pinned corpus prepare-only" 1 \
  cargo test --locked --release -p xlog-gpu --lib \
  logic::tests::pinned_corpus_prepares_resident_graph_without_launching_it -- \
  --ignored --exact --nocapture --test-threads=1
run_exact_rust_gate "pinned corpus production launch" 1 \
  cargo test --locked --release -p xlog-gpu --lib \
  logic::tests::pinned_corpus_certifies_and_runs_through_the_resident_production_path -- \
  --ignored --exact --nocapture --test-threads=1
run_exact_rust_gate "disconnected-rule scaling acceptance" 1 \
  cargo test --locked --release -p xlog-gpu --lib \
  logic::tests::resident_disconnected_four_thousand_rule_scaling_acceptance -- \
  --ignored --exact --nocapture --test-threads=1
run_exact_rust_gate "resident semantic acceptance matrix" 1 \
  cargo test --locked --release -p xlog-gpu --lib \
  logic::tests::resident_semantic_acceptance_matrix -- \
  --ignored --exact --nocapture --test-threads=1
run_cmd bash scripts/stage_pyxlog_kernels.sh
run_cmd python3 scripts/validate_reproducible_pyxlog_wheel.py --out-dir "$wheel_dir"
run_cmd python3 -m pip install --target "$python_install_dir" --no-deps "$wheel_dir"/pyxlog-*.whl
run_cmd env \
  PYTHONPATH="$python_install_dir" \
  PYTHONNOUSERSITE=1 \
  XLOG_PYTHON_INSTALL_ROOT="$python_install_dir" \
  python3 -c 'import os, pathlib, pyxlog, pytest, torch; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; install_root = pathlib.Path(os.environ["XLOG_PYTHON_INSTALL_ROOT"]).resolve(); package_path = pathlib.Path(pyxlog.__file__).resolve(); native_path = pathlib.Path(pyxlog._native.__file__).resolve(); package_path.relative_to(install_root); native_path.relative_to(install_root); print(f"validated wheel import: package={package_path} native={native_path} torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} pytest={pytest.__version__}")'
run_cmd env \
  PYTHONPATH="$python_install_dir" \
  PYTHONNOUSERSITE=1 \
  python3 -m pytest -q \
  python/tests/test_logic_relation_provenance.py \
  python/tests/test_relation_provenance_contract.py \
  python/tests/test_relation_provenance_public_api.py \
  python/tests/test_relation_callbacks_runtime.py
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
