#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cc -shared -fPIC -O2 -Wall -Wextra -o "$root_dir/libexdev_shim.so" "$root_dir/exdev_shim.c" -ldl

echo "built $root_dir/libexdev_shim.so"

