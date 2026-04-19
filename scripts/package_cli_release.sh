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

resolve_kernel_out_dir() {
  local target_dir="$1"
  if [[ -n "${XLOG_PACKAGE_KERNEL_OUT_DIR:-}" ]]; then
    printf '%s\n' "$XLOG_PACKAGE_KERNEL_OUT_DIR"
    return
  fi

  local build_root="$target_dir/release/build"
  if [[ ! -d "$build_root" ]]; then
    echo "release build root not found: $build_root" >&2
    exit 1
  fi

  local found
  found="$(
    find "$build_root" -type f -name '*.portable.ptx' -printf '%T@ %h\n' \
      | sort -nr \
      | head -n1 \
      | cut -d' ' -f2-
  )"
  if [[ -z "$found" ]]; then
    echo "unable to locate xlog-cuda OUT_DIR under $build_root" >&2
    exit 1
  fi

  printf '%s\n' "$found"
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
  cargo build -p xlog-cli --release --bin xlog
fi

binary_path="$(resolve_binary_path "$target_dir")"
kernel_out_dir="$(resolve_kernel_out_dir "$target_dir")"

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
