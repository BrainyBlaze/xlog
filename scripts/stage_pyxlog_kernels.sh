#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/stage_pyxlog_kernels.sh [--to <dir>]

Stage the generated xlog-cuda kernel artifacts into the pyxlog Python package
tree so maturin can include them in built wheels.
EOF
}

dest_dir="crates/pyxlog/python/pyxlog/kernels"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to)
      if [[ $# -lt 2 ]]; then
        echo "--to requires a directory" >&2
        exit 2
      fi
      dest_dir="$2"
      shift 2
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

resolve_target_dir() {
  cargo metadata --locked --format-version 1 --no-deps \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
}

resolve_kernel_out_dir_from_dep_info() {
  local target_dir="$1"

  python3 - "$target_dir" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path


target_dir = Path(sys.argv[1]).resolve()
deps_dir = target_dir / "release" / "deps"
if not deps_dir.is_dir():
    sys.exit(1)

matches: list[str] = []
for dep_info in sorted(deps_dir.glob("xlog_cuda-*.d")):
    for line in dep_info.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "# env-dep:OUT_DIR="
        if not line.startswith(marker):
            continue

        candidate = Path(line[len(marker) :])
        if candidate.is_dir() and any(candidate.glob("*.portable.ptx")):
            matches.append(str(candidate))
        break

unique_matches = sorted(dict.fromkeys(matches))
if len(unique_matches) != 1:
    sys.exit(1)

print(unique_matches[0])
PY
}

resolve_kernel_out_dir_from_build_tree() {
  local target_dir="$1"

  python3 - "$target_dir" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path


target_dir = Path(sys.argv[1]).resolve()
build_dir = target_dir / "release" / "build"
if not build_dir.is_dir():
    sys.exit(1)

candidates = []
for out_dir in build_dir.glob("xlog-cuda-*/out"):
    if not out_dir.is_dir():
        continue
    if not any(out_dir.glob("*.portable.ptx")):
        continue
    try:
        mtime = out_dir.stat().st_mtime_ns
    except FileNotFoundError:
        continue
    candidates.append((mtime, out_dir))

if not candidates:
    sys.exit(1)

candidates.sort(key=lambda item: item[0], reverse=True)
print(candidates[0][1])
PY
}

build_kernels_release() {
  # We only need the CUDA kernel artifacts that xlog-cuda's build.rs emits
  # into OUT_DIR (*.portable.ptx). Build xlog-cuda directly instead of pyxlog:
  # building pyxlog with `--features host-io` (no extension-module) makes pyo3
  # link libpython of the container's default interpreter (python3.10), which
  # breaks every wheel-matrix leg whose interpreter is not 3.10
  # ("rust-lld: error: unable to find library -lpython3.10"). xlog-cuda has no
  # pyo3 dependency, so it produces the same kernels with no libpython linkage.
  cargo build -p xlog-cuda --release --locked
}

cd "$repo_root"

build_kernels_release
target_dir="$(resolve_target_dir)"
kernel_out_dir="$(resolve_kernel_out_dir_from_dep_info "$target_dir" || true)"
if [[ -z "$kernel_out_dir" ]]; then
  kernel_out_dir="$(resolve_kernel_out_dir_from_build_tree "$target_dir" || true)"
fi
if [[ -z "$kernel_out_dir" ]]; then
  kernel_out_dir="$(resolve_kernel_out_dir_from_build_tree "$target_dir")"
fi

python3 "$repo_root/scripts/stage_kernels.py" \
  --from-out-dir "$kernel_out_dir" \
  --to "$repo_root/$dest_dir"

printf 'staged pyxlog kernels from %s into %s\n' "$kernel_out_dir" "$repo_root/$dest_dir"
