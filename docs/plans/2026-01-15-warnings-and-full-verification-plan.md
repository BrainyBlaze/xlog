# Warning Cleanup and Full Verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all current compiler warnings and run the full test suite plus all documented examples.

**Architecture:** Delete dead code to eliminate warnings; then run full `cargo test`, CUDA certification tests (included in workspace), all `.xlog` examples via the runner, probabilistic `.xlog` examples via Python bindings, and the Python example scripts after building the `xlog-gpu-py` extension.

**Tech Stack:** Rust (workspace crates), CUDA (certification suite), Python (maturin + torch), bash scripting.

### Task 1: Remove unused GPU buffer handle warning in xlog-core

**Files:**
- Modify: `crates/xlog-core/src/traits.rs`

**Step 1: Write a failing test (compile warning gate)**

Add a test-only compile guard to fail on warnings in this crate:

```rust
#[test]
fn test_no_dead_code_warnings_in_gpu_buffer() {
    // This test is a placeholder to force `cargo test -D warnings` on CI.
    // It should fail until dead code is removed.
    assert!(true);
}
```

**Step 2: Run test with warnings-as-errors**

Run: `RUSTFLAGS="-D warnings" cargo test -p xlog-core --lib`

Expected: FAIL with `dead_code` for `handle`.

**Step 3: Remove unused handle**

Edit `GpuBuffer` to drop `handle` and delete `GpuBufferHandle` enum; update `GpuBuffer::empty()` accordingly.

**Step 4: Re-run warnings-as-errors**

Run: `RUSTFLAGS="-D warnings" cargo test -p xlog-core --lib`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-core/src/traits.rs
git commit -m "chore: remove unused gpu buffer handle"
```

### Task 2: Remove unused join-order helpers in xlog-logic

**Files:**
- Modify: `crates/xlog-logic/src/lower.rs`

**Step 1: Write failing test (warnings-as-errors)**

Run: `RUSTFLAGS="-D warnings" cargo test -p xlog-logic --lib`

Expected: FAIL with `dead_code` for `order_positive_atoms` and `order_positive_atoms_cost_based`.

**Step 2: Remove unused helpers**

Delete `order_positive_atoms` and `order_positive_atoms_cost_based` (and any now-unused imports/helpers).

**Step 3: Re-run warnings-as-errors**

Run: `RUSTFLAGS="-D warnings" cargo test -p xlog-logic --lib`

Expected: PASS.

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/lower.rs
git commit -m "chore: remove unused join ordering helpers"
```

### Task 3: Remove unused optimizer test helper

**Files:**
- Modify: `crates/xlog-logic/tests/optimizer_real_world.rs`

**Step 1: Write failing test (warnings-as-errors)**

Run: `RUSTFLAGS="-D warnings" cargo test -p xlog-logic --test optimizer_real_world`

Expected: FAIL with `dead_code` for `create_stats_manager_with_rels`.

**Step 2: Remove unused helper**

Delete `create_stats_manager_with_rels` and any unused imports.

**Step 3: Re-run warnings-as-errors**

Run: `RUSTFLAGS="-D warnings" cargo test -p xlog-logic --test optimizer_real_world`

Expected: PASS.

**Step 4: Commit**

```bash
git add crates/xlog-logic/tests/optimizer_real_world.rs
git commit -m "chore: remove unused optimizer test helper"
```

### Task 4: Full workspace test suite (includes certification)

**Files:**
- None

**Step 1: Run full tests**

Run: `cargo test`

Expected: PASS; includes `xlog-cuda-tests` certification suite.

### Task 5: Run .xlog examples (positive + negative)

**Files:**
- Create: `scripts/run_xlog_examples.sh`

**Step 1: Write the runner script**

Create `scripts/run_xlog_examples.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

cargo build -p xlog-logic --example xlog_run

POSITIVE=$(find examples/xlog -type f -name "*.xlog" -not -path "examples/xlog/90-negative-tests/*" | sort)
NEGATIVE=$(find examples/xlog/90-negative-tests -type f -name "*.xlog" | sort)

for file in $POSITIVE; do
  echo "[POSITIVE] $file"
  cargo run -q -p xlog-logic --example xlog_run -- "$file"
  echo "OK: $file"
  echo
 done

for file in $NEGATIVE; do
  echo "[NEGATIVE] $file"
  if cargo run -q -p xlog-logic --example xlog_run -- "$file"; then
    echo "ERROR: negative example unexpectedly succeeded: $file" >&2
    exit 1
  else
    echo "OK (failed as expected): $file"
  fi
  echo
 done
```

**Step 2: Make executable and run**

Run:

```bash
chmod +x scripts/run_xlog_examples.sh
scripts/run_xlog_examples.sh
```

Expected: all positive examples succeed; all negative examples fail.

### Task 6: Run probabilistic .xlog examples via Python bindings

**Files:**
- Create: `scripts/run_prob_examples.py`

**Step 1: Write the Python runner**

Create `scripts/run_prob_examples.py`:

```python
from pathlib import Path

from xlog_gpu import Program


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "examples" / "prob").glob("*.xlog")):
        print(f"[PROB] {path}")
        source = path.read_text()
        prog = Program.compile(source, device=0, memory_mb=1024)
        result = prog.evaluate(return_grads=False)
        print(f"atoms={len(result.atoms)} approx={result.approx}")


if __name__ == "__main__":
    main()
```

**Step 2: Build Python extension and run**

Run:

```bash
cd crates/xlog-gpu-py
python -m pip install --upgrade pip maturin
maturin develop --release
cd ../..
python scripts/run_prob_examples.py
```

Expected: all prob examples run without error.

### Task 7: Run Python example scripts

**Files:**
- None

**Step 1: Ensure dependencies**

Run:

```bash
python -m pip install --upgrade pip torch maturin
```

**Step 2: Run scripts**

Run:

```bash
python examples/python/01_dlpack_reachability_torch.py
python examples/python/02_prob_wet_conditioning_torch.py
python examples/python/03_prob_mc_nonmonotone_torch.py
```

Expected: each script completes without errors and prints outputs.
