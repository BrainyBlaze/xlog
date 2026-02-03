# SAT/CDCL Block-Parallel BCP Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the GPU CDCL verifier scale to real equivalence workloads by implementing one-block-per-instance CDCL with block-parallel unit propagation (BCP), while preserving GPU-only verification (zero DTOH), determinism, and proof checking.

**Architecture:** Keep CDCL control flow on a single “control thread” (lane 0) for determinism, but parallelize the hot propagation path: for each propagated literal, process watched clauses in fixed-size chunks using the whole block, then commit updates deterministically. Parallelize O(n) initialization loops (variable state, watch arrays) across the block. Keep memory arenas fixed-capacity; overflow remains a hard error.

**Tech Stack:** CUDA C (PTX embedded), Rust (`xlog-solve`, `xlog-cuda`, `xlog-prob`), `cudarc` (Driver API), WSL CUDA runtime (`LD_LIBRARY_PATH=/usr/lib/wsl/lib`).

---

## Baseline (Evidence)

**Repro:** The Python MNIST-addition training test stalls in GPU equivalence verification at:
`[xlog-prob] equivalence: solve_expect_unsat q1`.

Run (from worktree `gpu-native-io-mc-device`):

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_LAUNCH_BLOCKING=1 .venv/bin/python -m pytest -q \
  python/tests/test_minimal_example.py::TestAdditionQueryTraining::test_addition_query_forward_backward -s
```

**Root cause:** `kernels/sat.cu:sat_cdcl_solve` effectively runs single-threaded (`threadIdx.x != 0` returns). This cannot meet the design verifier target (docs/design/2026-01-22-gpu-native-compilation-design.md §7) for large equivalence CNFs.

---

## Task 1: Update CDCL Kernel Launch to Use a Full Block

**Files:**
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs`
- Modify: `kernels/sat.cu`

**Step 1: Change `sat_cdcl_solve` launch config to `block_dim=(256,1,1)`**

- In `crates/xlog-solve/src/gpu_cdcl.rs`, update the `LaunchConfig` used to launch `sat_cdcl_solve`.
- Keep `grid_dim=(1,1,1)` (one block per instance).

**Step 2: Remove the early return for nonzero `threadIdx.x`**

- In `kernels/sat.cu`, remove/replace:
  - `if (threadIdx.x != 0) return;`
- Keep `if (blockIdx.x != 0) return;`.

**Verify:**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo test -p xlog-solve --test gpu_cdcl_tests -- --nocapture
```

Expected: PASS.

---

## Task 2: Parallelize Solver Initialization (Deterministic)

**Files:**
- Modify: `kernels/sat.cu`

**Step 1: Parallelize variable array initialization**

- Replace the single-thread `for (v=0..nv)` init with strided per-thread init:
  - `for (v = threadIdx.x; v <= nv; v += blockDim.x) { ... }`
- Ensure `v=0` is handled consistently (index 0 is reserved but arrays include it).
- Add `__syncthreads()` after init.

**Step 2: Parallelize watch array initialization**

- Parallelize:
  - `watch_head[0..2*nv)` init to `-1`
  - `watch_next/watch_prev[0..2*max_total_clauses)` init to `-1`
  - `trail_lim[0..nv]` init to `0`
- Add `__syncthreads()` after each phase where later code relies on the writes.

**Verify:** same as Task 1.

---

## Task 3: Implement Block-Parallel Watched-Literal Propagation (BCP)

**Files:**
- Modify: `kernels/sat.cu`

**Core requirement:** All threads in the block participate in clause inspection; only thread 0 mutates global solver state (assignments/trail/reasons, watch-list pointer updates) in a deterministic commit order.

**Step 1: Introduce a block-cooperative propagate function**

- Replace `sat_propagate(...)` with a cooperative version that assumes an active block.
- Use shared memory to broadcast the current propagated literal `p` and to stage a chunk of watch nodes:
  - `__shared__ int32_t sh_p;`
  - `__shared__ uint32_t sh_chunk_n;`
  - `__shared__ int32_t sh_nodes[CHUNK];` (CHUNK=256)

**Step 2: Chunked traversal of the watch linked-list**

- Thread 0 walks `watch_head[fals_idx]` and fills `sh_nodes[]` with up to CHUNK node ids, without modifying the list.
- Broadcast `sh_chunk_n`, then all threads `t < sh_chunk_n` process one node.

**Step 3: Per-node parallel clause inspection (read-only)**

Each worker thread computes an action for its node:
- `SKIP` (other watch satisfied)
- `MOVE` (new watched literal found; record `new_idx` + new watch position `k`)
- `UNIT` (no move, other lit unassigned; record implied lit + clause id)
- `CONFLICT` (no move, other lit false; record clause id)
- `DELETE` (learned clause deleted; record node for removal)

Store results in shared arrays (one entry per node in the chunk). No global writes.

**Step 4: Deterministic commit (thread 0)**

- Thread 0 applies actions in a deterministic order (e.g., increasing `i` in `sh_nodes[]` for the chunk).
- For `MOVE`: call existing `sat_watch_remove` / `sat_watch_insert_head` and update `watch0_pos/watch1_pos`.
- For `UNIT`: call `sat_enqueue` (only thread 0); if enqueue fails, set conflict deterministically.
- For `CONFLICT`: choose the minimum clause id conflict encountered in this chunk (deterministic).
- For `DELETE`: remove the node from the watch list.
- After commit, `__syncthreads()` and continue with the next chunk.

**Step 5: Repeat for each `p` in the propagation queue**

- Thread 0 updates `qhead`/`trail_len`/`assigned_count`.
- Broadcast termination when `qhead == trail_len`.

**Verify:**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo test -p xlog-solve --test gpu_cdcl_tests -- --nocapture
```

Expected: PASS.

---

## Task 4: Regenerate `kernels/sat.ptx` with NVRTC 12.8 (WSL-Compatible)

**Files:**
- Modify: `kernels/sat.ptx`

**Step 1: Compile `kernels/sat.cu` via NVRTC 12.8 and write `kernels/sat.ptx`**

- Use the known-good NVRTC shared lib:
  - `/home/dev/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12`
- Compile options:
  - `--std=c++17`
  - `--gpu-architecture=compute_70`
- Strip the trailing NUL byte from the PTX buffer before writing to disk.

**Step 2: Verify `.ptx` is newer than `.cu`**

```bash
python3 - <<'PY'
from pathlib import Path
cu=Path('kernels/sat.cu'); ptx=Path('kernels/sat.ptx')
print('cu mtime', cu.stat().st_mtime, 'ptx mtime', ptx.stat().st_mtime)
assert ptx.stat().st_mtime >= cu.stat().st_mtime
PY
```

---

## Task 5: End-to-End: Rebuild Python Extension + Repro

**Step 1: Rebuild `pyxlog`**

```bash
maturin develop -m crates/pyxlog/Cargo.toml --pip-path .venv/bin/pip -F host-io
```

**Step 2: Rerun baseline repro**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_LAUNCH_BLOCKING=1 .venv/bin/python -m pytest -q \
  python/tests/test_minimal_example.py::TestAdditionQueryTraining::test_addition_query_forward_backward -s
```

Expected: completes (does not stall at `solve_expect_unsat q1`).

---

## Task 6: Production Cleanup (No Debug Spam / No DTOH in Production Paths)

**Files (likely):**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs`
- Modify: `crates/xlog-prob/src/compilation/mod.rs`
- Modify: `crates/xlog-prob/src/compilation/validation.rs`

**Step 1: Remove debug `eprintln!` instrumentation**

**Step 2: Remove any device→host reads from production compilation/verification modules**

**Verify:**

```bash
rg -n \"dtoh|copy_to_host|eprintln!\\(\\[xlog-prob\\]\" crates/xlog-prob/src/compilation crates/xlog-solve/src | cat
cargo test -p xlog-prob --tests
```

Expected: no matches in production modules; tests pass.

