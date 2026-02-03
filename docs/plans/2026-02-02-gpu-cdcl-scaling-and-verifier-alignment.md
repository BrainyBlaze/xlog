# GPU CDCL Scaling + Verifier Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the GPU-native equivalence verifier complete and production-grade on large real workloads (e.g. the MNIST addition training query) while preserving strict “zero host reads” contracts and determinism.

**Architecture:** Improve the GPU CDCL solver’s scaling by eliminating O(num_vars) decision scans via a deterministic decision heap, keep all verifier paths GPU-only (no device->host transfers), and remove any debug/temporary host reads. Keep `cudarc` as the CUDA driver API integration as specified by the design docs.

**Tech Stack:** Rust (xlog-prob/xlog-solve/xlog-cuda), CUDA C kernels embedded as PTX (loaded via `cudarc`), WSL CUDA runtime (`LD_LIBRARY_PATH=/usr/lib/wsl/lib`).

---

## Baseline Repro (Evidence)

**Symptom:** Python training test hangs inside GPU equivalence verification at the first UNSAT query (`q1`), indicating the GPU CDCL solver does not complete in a reasonable time on the large equivalence CNF.

Run (from worktree `gpu-native-io-mc-device`):

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_LAUNCH_BLOCKING=1 .venv/bin/python -m pytest -q \
  python/tests/test_minimal_example.py::TestAdditionQueryTraining::test_addition_query_forward_backward -s
```

Expected (pre-fix): stalls/hangs after the debug stage marker:
`[xlog-prob] equivalence: solve_expect_unsat q1`

---

## Task 1: Finish Deterministic Decision Heap in `sat_cdcl_solve`

**Files:**
- Modify: `kernels/sat.cu`

**Step 1: Add solver-owned heap buffers to the kernel signature**

Add:
- `decision_heap: uint32_t*` (len = `var_cap + 1`, used indices `0..num_vars`)
- `decision_heap_pos: uint32_t*` (len = `var_cap + 1`, `SAT_HEAP_NONE` sentinel)

**Step 2: Initialize the heap deterministically**

- Fill heap with vars `1..=num_vars` (tie-breaker: smaller var id wins).
- Heapify in a deterministic order.

**Step 3: Maintain heap correctness**

- On assignment (`sat_enqueue`): remove `v` from heap.
- On unassignment (`sat_unassign_lit` / `sat_backtrack`): insert `v` back into heap.
- On VSIDS bump: if `v` is currently unassigned (present in heap), sift it upward.
- On activity rescale: rebuild heap (heapify) so ordering remains correct.

**Step 4: Replace decision selection**

- Replace decision scans with `heap[0]` (after dropping any stale root defensively).
- If heap is empty and `assigned_count == num_vars`, return SAT.

**Step 5: Remove unused decision-scan helpers**

- Delete `sat_pick_branch_var` and `sat_maybe_rescale_activities` if no longer used.

**Verify (build-only):**

```bash
cargo test -p xlog-solve --no-run
```

Expected: compile succeeds once Task 2 regenerates PTX and Task 3 updates the Rust launch signature.

---

## Task 2: Regenerate Embedded PTX with NVRTC 12.8 (WSL-Compatible)

**Files:**
- Modify: `kernels/sat.ptx`

**Step 1: Compile `kernels/sat.cu` to PTX using NVRTC 12.8**

- Use the NVRTC shared library available on the machine (12.8).
- IMPORTANT: strip the trailing NUL byte when writing the PTX file (write `ptx_size - 1` bytes).

**Step 2: Ensure build scripts do not attempt to invoke `nvcc`**

- Confirm `kernels/sat.ptx` is newer than `kernels/sat.cu`.

**Verify:**

```bash
cargo test -p xlog-cuda --lib
```

Expected: PASS and PTX symbol validation still succeeds.

---

## Task 3: Update Rust CDCL Orchestrator for New Kernel Signature

**Files:**
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs`

**Step 1: Allocate and retain heap buffers for the solver run**

Allocate:
- `decision_heap: TrackedCudaSlice<u32>` (len = `var_cap + 1`)
- `decision_heap_pos: TrackedCudaSlice<u32>` (len = `var_cap + 1`)

Keep them alive by storing them inside `GpuCdclRun`.

**Step 2: Pass heap buffers in the `sat_cdcl_solve` parameter list**

Insert the two device pointers at the exact position matching `kernels/sat.cu`.

**Verify:**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo test -p xlog-solve --test gpu_cdcl_tests -- --nocapture
```

Expected: PASS (or test skips if CUDA unavailable).

---

## Task 4: Remove Debug/Temporary Host Reads and Debug Spam (Verifier Paths)

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs`
- Modify: `crates/xlog-prob/src/compilation/mod.rs`
- Modify: `crates/xlog-prob/src/compilation/validation.rs`

**Step 1: Remove any `dtoh_*` calls and debug-only host reads**

The verifier and compiler production paths must be zero device->host transfers.

**Step 2: Remove debug stage prints and temporary instrumentation**

No debug markers should remain in production paths.

**Verify:**

```bash
rg -n "dtoh|copy_to_host" crates/xlog-prob/src/compilation crates/xlog-solve/src | cat
cargo test -p xlog-prob --tests
```

Expected: no matches in verifier/compiler production modules; tests pass.

---

## Task 5: End-to-End Verification (Docs + Code + Tests)

**Step 1: Run Rust workspace tests**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo test --workspace
```

**Step 2: Rebuild the Python extension in the worktree**

```bash
maturin develop -m crates/pyxlog/Cargo.toml --pip-path .venv/bin/pip -F host-io
```

**Step 3: Rerun the baseline repro**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_LAUNCH_BLOCKING=1 .venv/bin/python -m pytest -q \
  python/tests/test_minimal_example.py::TestAdditionQueryTraining::test_addition_query_forward_backward -s
```

Expected (post-fix): completes successfully and does not stall in `solve_expect_unsat q1`.

