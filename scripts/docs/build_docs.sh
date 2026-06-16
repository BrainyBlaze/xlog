#!/usr/bin/env bash
set -euo pipefail

build_rust=1
build_cuda=1

for arg in "$@"; do
  case "$arg" in
    --no-rust)
      build_rust=0
      ;;
    --no-cuda)
      build_cuda=0
      ;;
    *)
      echo "unknown docs build option: $arg" >&2
      exit 64
      ;;
  esac
done

mkdir -p .docs/generated docs/api/generated

cp ROADMAP.md docs/ROADMAP.md

python3 scripts/docs/gen_pyxlog_api.py \
  --output docs/api/generated/pyxlog-api.md

if ((build_rust)); then
  cargo doc --workspace --no-deps --locked
  cargo_target_dir="$(
    cargo metadata --locked --no-deps --format-version=1 \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
  )"
  rustdoc_dir="${cargo_target_dir}/doc"
  rm -rf docs/api/generated/rust
  mkdir -p docs/api/generated/rust
  cp -R "${rustdoc_dir}/." docs/api/generated/rust/
  python3 - <<'PY'
from pathlib import Path

root = Path("docs/api/generated/rust")
crates = sorted(
    path.name for path in root.iterdir()
    if path.is_dir() and (path / "index.html").exists()
)
items = "\n".join(
    f'<li><a href="{crate}/index.html">{crate}</a></li>' for crate in crates
)
Path("docs/api/generated/rust/index.html").write_text(
    f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>XLOG Rust API</title></head>
<body><main><h1>XLOG Rust API</h1><ul>{items}</ul></main></body>
</html>
""",
    encoding="utf-8",
)
PY
else
  mkdir -p docs/api/generated/rust
  cat > docs/api/generated/rust/index.html <<'HTML'
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Rust API placeholder</title></head>
<body><main><h1>Rust API placeholder</h1><p>Run make docs to generate rustdoc output.</p></main></body>
</html>
HTML
fi

if ((build_cuda)); then
  if ! command -v doxygen >/dev/null 2>&1; then
    echo "doxygen is required for full CUDA docs. Install doxygen or run with --no-cuda." >&2
    exit 127
  fi
  doxygen Doxyfile.docs
else
  mkdir -p docs/api/generated/cuda/html
  cat > docs/api/generated/cuda/html/index.html <<'HTML'
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>CUDA API placeholder</title></head>
<body><main><h1>CUDA API placeholder</h1><p>Run make docs to generate Doxygen output.</p></main></body>
</html>
HTML
fi

python3 -m mkdocs build --strict
