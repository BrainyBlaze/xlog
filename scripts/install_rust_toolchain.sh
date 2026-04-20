#!/usr/bin/env bash
set -euo pipefail

toolchain="stable"
profile="${RUSTUP_PROFILE:-minimal}"
components=()
targets=()

usage() {
  cat <<'EOF' >&2
Usage: install_rust_toolchain.sh [--toolchain stable] [--component clippy] [--target x86_64-unknown-linux-gnu]
EOF
}

while (($# > 0)); do
  case "$1" in
    --toolchain)
      toolchain="${2:?missing value for --toolchain}"
      shift 2
      ;;
    --component)
      components+=("${2:?missing value for --component}")
      shift 2
      ;;
    --target)
      targets+=("${2:?missing value for --target}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

retry_curl() {
  curl \
    --proto '=https' \
    --tlsv1.2 \
    --location \
    --silent \
    --show-error \
    --fail \
    --retry 10 \
    --retry-delay 3 \
    --retry-all-errors \
    "$@"
}

export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export RUSTUP_MAX_RETRIES="${RUSTUP_MAX_RETRIES:-10}"
export PATH="$CARGO_HOME/bin:$PATH"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "CARGO_HOME=$CARGO_HOME"
    echo "RUSTUP_HOME=$RUSTUP_HOME"
  } >>"$GITHUB_ENV"
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$CARGO_HOME/bin" >>"$GITHUB_PATH"
fi

if ! command -v rustup >/dev/null 2>&1; then
  retry_curl https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile "$profile"
fi

rustup set profile "$profile"
rustup toolchain install "$toolchain"
rustup default "$toolchain"

for component in "${components[@]}"; do
  rustup component add --toolchain "$toolchain" "$component"
done

for target in "${targets[@]}"; do
  rustup target add --toolchain "$toolchain" "$target"
done

rustc --version
cargo --version
