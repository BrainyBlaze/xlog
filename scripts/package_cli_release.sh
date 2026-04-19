#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/package_cli_release.sh --output <dir>

Build the xlog CLI in release mode, stage CUDA kernels beside the binary,
and write a .tar.gz bundle suitable for a GitHub release.
EOF
}

output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      if [[ $# -lt 2 ]]; then
        echo "--output requires a directory" >&2
        exit 2
      fi
      output_dir="$2"
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

if [[ -z "$output_dir" ]]; then
  echo "--output is required" >&2
  usage >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

resolve_target_dir() {
  if [[ -n "${XLOG_PACKAGE_TARGET_DIR:-}" ]]; then
    printf '%s\n' "$XLOG_PACKAGE_TARGET_DIR"
    return
  fi

  cargo metadata --format-version 1 --no-deps \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
}

resolve_version() {
  if [[ -n "${XLOG_PACKAGE_VERSION:-}" ]]; then
    printf '%s\n' "$XLOG_PACKAGE_VERSION"
    return
  fi

  cargo metadata --format-version 1 --no-deps \
    | python3 -c 'import json, sys; data=json.load(sys.stdin); print(next(pkg["version"] for pkg in data["packages"] if pkg["name"] == "xlog-cli"))'
}

resolve_host_triple() {
  if [[ -n "${XLOG_PACKAGE_HOST_TRIPLE:-}" ]]; then
    printf '%s\n' "$XLOG_PACKAGE_HOST_TRIPLE"
    return
  fi

  rustc -vV | awk '/^host:/ {print $2}'
}

resolve_binary_path() {
  local target_dir="$1"
  if [[ -n "${XLOG_PACKAGE_BINARY_PATH:-}" ]]; then
    printf '%s\n' "$XLOG_PACKAGE_BINARY_PATH"
    return
  fi

  printf '%s\n' "$target_dir/release/xlog"
}

build_release_and_capture_kernel_out_dir() {
  cargo build -p xlog-cli --release --bin xlog --message-format=json-render-diagnostics \
    | python3 -c '
import json
import sys

kernel_out_dir = None

for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")
    if not line:
        continue
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        print(line, file=sys.stderr)
        continue

    rendered = message.get("message", {}).get("rendered")
    if rendered:
        print(rendered, end="", file=sys.stderr)

    if message.get("reason") != "build-script-executed":
        continue

    package_id = message.get("package_id", "")
    out_dir = message.get("out_dir")
    if "xlog-cuda" in package_id and out_dir and kernel_out_dir is None:
        kernel_out_dir = out_dir

if not kernel_out_dir:
    print("unable to capture xlog-cuda OUT_DIR from cargo build output", file=sys.stderr)
    sys.exit(1)

print(kernel_out_dir)
'
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
    print(f"release deps directory not found: {deps_dir}", file=sys.stderr)
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
if not unique_matches:
    print(
        f"unable to locate xlog-cuda OUT_DIR from dep-info under {deps_dir}",
        file=sys.stderr,
    )
    sys.exit(1)

if len(unique_matches) > 1:
    print(
        "multiple xlog-cuda OUT_DIR values found in release dep-info; "
        "set XLOG_PACKAGE_KERNEL_OUT_DIR explicitly or rebuild from a clean target directory",
        file=sys.stderr,
    )
    for match in unique_matches:
        print(f"  - {match}", file=sys.stderr)
    sys.exit(1)

print(unique_matches[0])
PY
}

resolve_kernel_out_dir() {
  local target_dir="$1"
  if [[ -n "${XLOG_PACKAGE_KERNEL_OUT_DIR:-}" ]]; then
    printf '%s\n' "$XLOG_PACKAGE_KERNEL_OUT_DIR"
    return
  fi

  resolve_kernel_out_dir_from_dep_info "$target_dir"
}

cd "$repo_root"

target_dir="$(resolve_target_dir)"
version="$(resolve_version)"
host_triple="$(resolve_host_triple)"
release_name="xlog-v${version}-${host_triple}"
stage_root="${output_dir%/}/$release_name"
tarball_path="${output_dir%/}/${release_name}.tar.gz"

mkdir -p "$output_dir"
rm -rf "$stage_root"
rm -f "$tarball_path"

if [[ "${XLOG_PACKAGE_SKIP_BUILD:-0}" != "1" ]]; then
  kernel_out_dir="$(build_release_and_capture_kernel_out_dir)"
fi

binary_path="$(resolve_binary_path "$target_dir")"
if [[ -z "${kernel_out_dir:-}" ]]; then
  kernel_out_dir="$(resolve_kernel_out_dir "$target_dir")"
fi

if [[ ! -f "$binary_path" ]]; then
  echo "release binary not found: $binary_path" >&2
  exit 1
fi

mkdir -p "$stage_root/kernels"
cp "$binary_path" "$stage_root/xlog"
chmod 755 "$stage_root/xlog"

python3 "$repo_root/scripts/stage_kernels.py" \
  --from-out-dir "$kernel_out_dir" \
  --to "$stage_root/kernels"

cp "$repo_root/README.md" "$stage_root/README.md"
cp "$repo_root"/LICENSE* "$stage_root/"

tar -C "$output_dir" -czf "$tarball_path" "$release_name"

printf 'staged release directory: %s\n' "$stage_root"
printf 'release tarball: %s\n' "$tarball_path"
