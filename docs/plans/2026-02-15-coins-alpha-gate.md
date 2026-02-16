# Coins Alpha Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the coins boolean-query pipeline crash (CUDA_ERROR_LAUNCH_FAILED) and produce passing gate evidence for v0.4.0-alpha release.

**Architecture:** The coins program (`win() :- coin(1,X,heads), coin(2,Y,heads).`) exercises the multi-network boolean query path. The pipeline is: `forward_backward_complex_tensor` → `compile_circuit_for_template` → `generate_template_ast` → `ExactDdnnfProgram::compile_from_program` → `compile_provenance_with_gpu` → D4 GPU kernels. The crash occurs in the D4 compilation step — a GPU kernel launch fails on the CNF derived from the template AST. The fix requires diagnosing whether the crash is in the template AST generation, the PIR→CNF encoding, or the D4 kernel parameters, then fixing the root cause.

**Tech Stack:** Rust (pyxlog crate, xlog-prob crate, xlog-cuda crate), CUDA PTX kernels, Python + pytest, PyO3

**Worktree:** `/home/dev/projects/xlog/.worktrees/v0.4.0-alpha-integrated/`

**Current State:**
- Commit `f2a84aa5` is HEAD. User's commits already fixed: output_var=None for constants, cache determinism (template_compile_count), example test guards
- `kernels/weights.ptx` has uncommitted recompilation (CUDA 12.8, ISA 8.7) — needed for Task 6 symbols
- All other PTX files: CUDA 12.9, ISA 8.7, sm_70 — compatible with driver 573.71
- `test_coins_compile_and_register` PASSES
- `test_coins_forward_backward_expected_true` CRASHES with `CUDA_ERROR_LAUNCH_FAILED` in D4 compilation

**Issue Board Mapping:**
| Issue | Plan Task |
|-------|-----------|
| Infra: weights.ptx recompilation | Task 0 (isolated) |
| A1: Constant-bound neural output crash | Tasks 1-3 |
| A2: Template preserves constraints | Tasks 1-3 |
| A3: Cache determinism | Task 4 |
| A4: Example/gate tests CUDA-safe | Task 5 |
| A5: Coins gate evidence artifact | Task 6 |

---

### Task 0: Commit weights.ptx recompilation (infra, separate from crash fix)

**Goal:** Land the PTX recompilation as its own isolated commit so it doesn't contaminate the crash-fix bisect/blame history.

**Files:**
- Commit: `kernels/weights.ptx` (already modified in working tree)

**Step 1: Verify the recompiled PTX has the required symbols**

```bash
cd /home/dev/projects/xlog/.worktrees/v0.4.0-alpha-integrated
grep -c 'weights_force_var_true\|weights_restore_var_true' kernels/weights.ptx
```

Expected: 2+ matches (at least one `.visible .entry` each).

**Step 2: Verify ISA/target compatibility**

```bash
head -6 kernels/weights.ptx
```

Expected: `.version 8.7`, `.target sm_70`, CUDA 12.8.

**Step 3: Commit**

```bash
git add kernels/weights.ptx
git commit -m "chore: recompile weights.ptx with CUDA 12.8 (ISA 8.7, sm_70)

Restores Task 6 symbols: weights_force_var_true, weights_restore_var_true.
Matches driver 573.71 (CUDA 12.8).
Isolated from coins crash fix to keep bisect clean."
```

---

### Task 1: Diagnose the CUDA_ERROR_LAUNCH_FAILED crash

**Goal:** Determine exactly where and why the D4 compilation crashes for the coins boolean query.

**Files:**
- Modify: `crates/xlog-prob/src/exact.rs:805-996` (compile_provenance_with_gpu)
- Modify: `crates/xlog-prob/src/compilation/mod.rs:120-180` (compile_gpu_d4_and_verify_cached)

**Context for the implementer:**

The crash stack trace is:
```
xlog_prob::compilation::gpu_cache::GpuCircuitCache::new
xlog_prob::exact::ExactDdnnfProgram::compile_provenance_with_gpu
xlog_prob::exact::ExactDdnnfProgram::compile_from_program
pyxlog::CompiledProgram::compile_circuit_for_template
pyxlog::CompiledProgram::forward_backward_complex_tensor
```

The template AST for coins generates this program:
```prolog
% Annotated disjunctions (group 0 = net1, group 1 = net2)
0.4999999::coin(1, 0, heads); 0.4999999::coin(1, 0, tails).
0.4999999::coin(2, 1, heads); 0.4999999::coin(2, 1, tails).
% Rule
win() :- coin(1, X, heads), coin(2, Y, heads).
% Query
?- win().
```

This should produce a trivial CNF (4 random variables, simple AND formula). CUDA_ERROR_LAUNCH_FAILED means a kernel crashed at runtime — usually illegal memory access from bad parameters.

**Step 1: Add diagnostic logging to compile_provenance_with_gpu**

In `crates/xlog-prob/src/exact.rs`, add `eprintln!` after each major step in `compile_provenance_with_gpu` (lines 805-996) to narrow down exactly which step crashes:

```rust
// After line 882 (GpuPirGraph::from_host)
eprintln!("[diag] PIR uploaded to GPU");

// After line 883 (GpuPirRoots::from_host)
eprintln!("[diag] PIR roots uploaded: {} roots", roots.len());

// After line 884 (encode_cnf_gpu)
eprintln!("[diag] CNF encoded: var_cap={}, clause_cap={}", encoding.cnf.var_cap, encoding.cnf.clause_cap);

// After line 934 (build_weights_gpu)
eprintln!("[diag] Weights built on GPU");

// After line 947 (default_cache_config)
eprintln!("[diag] Cache config: node_cap={}, edge_cap={}, level_cap={}, var_cap={}",
    cache_config.node_cap, cache_config.edge_cap, cache_config.level_cap, cache_config.var_cap);

// After line 949 (GpuCircuitCache::new)
eprintln!("[diag] GpuCircuitCache created");

// After line 957 (compile_gpu_d4_and_verify_cached)
eprintln!("[diag] D4 compiled successfully");
```

Also add the same in `compile_gpu_d4_and_verify_cached` in `crates/xlog-prob/src/compilation/mod.rs` (around line 120-180):

```rust
// After hash_cnf_gpu (line ~136)
eprintln!("[diag] CNF hash computed");

// After cache.lookup_or_insert_device (line ~146)
eprintln!("[diag] Cache lookup complete");

// Before compile_gpu_d4_gated (line ~153)
eprintln!("[diag] Starting D4 compilation...");

// After compile_gpu_d4_gated (line ~153)
eprintln!("[diag] D4 compilation complete");
```

**Step 2: Rebuild and run the crashing test**

```bash
cd /home/dev/projects/xlog/.worktrees/v0.4.0-alpha-integrated
cargo build -p pyxlog --features host-io --release
```

Verify cargo succeeded (exit code 0) before proceeding. Then run only the crashing test to see where the diagnostic output stops:

```bash
CUDA_LAUNCH_BLOCKING=1 python -m pytest python/tests/test_coins.py::test_coins_forward_backward_expected_true -xvs 2>&1
```

Setting `CUDA_LAUNCH_BLOCKING=1` forces synchronous kernel launches, which makes the crash location precise.

**Expected output:** The diagnostic prints will show exactly which step crashes. The last `[diag]` message printed before the crash reveals the failing operation.

**Step 3: If crash is in D4 compilation, check CNF validity**

If the diagnostic shows the crash is after "CNF encoded" but during "D4 compilation", add a host-side CNF dump before the D4 call in `compile_provenance_with_gpu`:

The `GpuCnf` struct (xlog-solve/src/gpu_cnf.rs:13) has `var_cap: u32`, `clause_cap: u32`, `lit_cap: u32` as host fields, and `num_vars`, `num_clauses`, `num_lits` as device-resident `TrackedCudaSlice<u32>`. To read actual counts, copy from device:

```rust
// After encode_cnf_gpu, before D4 compilation
{
    let mut num_vars_host = [0u32; 1];
    let mut num_clauses_host = [0u32; 1];
    let mut num_lits_host = [0u32; 1];
    provider.device().inner().dtoh_sync_copy_into(&encoding.cnf.num_vars, &mut num_vars_host).ok();
    provider.device().inner().dtoh_sync_copy_into(&encoding.cnf.num_clauses, &mut num_clauses_host).ok();
    provider.device().inner().dtoh_sync_copy_into(&encoding.cnf.num_lits, &mut num_lits_host).ok();
    eprintln!("[diag] CNF actual: num_vars={}, num_clauses={}, num_lits={} (caps: var={}, clause={}, lit={})",
        num_vars_host[0], num_clauses_host[0], num_lits_host[0],
        encoding.cnf.var_cap, encoding.cnf.clause_cap, encoding.cnf.lit_cap);
}
```

Check if any cap is 0 or if `num_vars > var_cap` — either could cause kernel OOB access.

**Step 4: Commit diagnostic changes**

```bash
git add crates/xlog-prob/src/exact.rs crates/xlog-prob/src/compilation/mod.rs
git commit -m "diag: add eprintln tracing to compile_provenance_with_gpu pipeline"
```

**Decision Point:** Based on diagnostic output, proceed to Task 2A, 2B, or 2C.

---

### Task 2A: Fix — Template AST grounding issue

**Applies if:** Diagnostic shows crash at or before `encode_cnf_gpu`, or CNF has unexpected dimensions (var_cap=0, clause_cap=0, etc.)

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:1879-1998` (generate_template_ast)
- Test: `python/tests/test_coins.py`

**Root cause hypothesis:** The template AST might produce a program that `extract_from_program` cannot properly ground, resulting in empty or malformed provenance, which then produces a degenerate CNF that crashes kernel allocation.

**Step 1: Add template AST debug dump**

In `compile_circuit_for_template` (lib.rs:2178), after generating the template AST, serialize and print it:

```rust
#[cfg(feature = "host-io")]
{
    let (expanded_ast, target_domain) =
        self.generate_template_ast(signature, query_pred, query_arity)?;

    // Debug dump
    eprintln!("[template] rules: {}", expanded_ast.rules.len());
    eprintln!("[template] ADs: {}", expanded_ast.annotated_disjunctions.len());
    eprintln!("[template] queries: {}", expanded_ast.prob_queries.len());
    for (i, ad) in expanded_ast.annotated_disjunctions.iter().enumerate() {
        eprintln!("[template] AD[{}]: {} choices", i, ad.choices.len());
        for (j, choice) in ad.choices.iter().enumerate() {
            eprintln!("[template]   choice[{}]: {}({}) p={}", j, choice.atom.predicate,
                choice.atom.terms.len(), choice.prob);
        }
    }
    for rule in &expanded_ast.rules {
        eprintln!("[template] rule head: {}({})", rule.head.predicate, rule.head.terms.len());
    }
    for q in &expanded_ast.prob_queries {
        eprintln!("[template] query: {}({})", q.atom.predicate, q.atom.terms.len());
    }

    let program = ExactDdnnfProgram::compile_from_program(&expanded_ast, self._gpu_config)
    // ... rest unchanged
}
```

**Step 2: Run and analyze output**

```bash
CUDA_LAUNCH_BLOCKING=1 python -m pytest python/tests/test_coins.py::test_coins_forward_backward_expected_true -xvs 2>&1
```

Verify the template AST has:
- 2 annotated disjunctions, each with 2 choices
- 1 rule (`win/0`)
- 1 query (`win/0`)

If any of these are wrong, fix `generate_template_ast`.

**Step 3: If template looks correct, check PIR construction**

The issue might be in how `extract_from_program` (in xlog-prob) handles the template program. Check:

```bash
grep -n "fn extract_from_program" crates/xlog-prob/src/*.rs crates/xlog-prob/src/**/*.rs
```

Look for how it handles the body variables X, Y in the rule when they appear in the annotated disjunctions.

**Step 4: Apply the fix and commit**

Fix depends on what the diagnostic reveals. After fixing:

```bash
cargo build -p pyxlog --features host-io --release
CUDA_LAUNCH_BLOCKING=1 python -m pytest python/tests/test_coins.py::test_coins_forward_backward_expected_true -xvs
git add -A && git commit -m "fix: correct template AST grounding for boolean queries"
```

---

### Task 2B: Fix — D4 kernel parameter overflow

**Applies if:** Diagnostic shows CNF has valid dimensions but crash is during `compile_gpu_d4_gated` kernel launch.

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs` (compile_gpu_d4_with_gate)
- Modify: `crates/xlog-prob/src/exact.rs` (default_compile_config)
- Test: `python/tests/test_coins.py`

**Root cause hypothesis:** The D4 compilation's buffer allocation or kernel launch parameters might be incompatible with the small CNF (4 vars, few clauses). Edge cases: zero-length allocations, grid_dim=0, or buffer sizes computed from zero multiplied caps.

**Step 1: Check the compile_config values**

Add logging for actual config values passed to D4:

```rust
eprintln!("[diag] GpuCompileConfig: frontier_depth={}, max_frontier_items={}, max_depth={}, smooth_node_cap={}, smooth_edge_cap={}",
    compile_config.frontier_depth,
    compile_config.max_frontier_items,
    compile_config.max_depth,
    compile_config.smooth_node_cap,
    compile_config.smooth_edge_cap);
```

**Step 2: Check for zero-size buffer allocations**

In `compile_gpu_d4_with_gate` (gpu_d4.rs), search for any `.alloc::<T>(0)` or allocations that might compute to 0 for a small CNF. The key values from `default_compile_config` are:

- `var_cap = cnf.var_cap.max(1)` — safe
- `smooth_node_cap = per_item_nodes * frontier_cap_factor` where `per_item_nodes = (var_cap * 5).max(1024)` — for var_cap=4, this is 1024. `frontier_cap_factor` comes from `min(2^6, max_frontier_items)`. Should be ≥8.

This suggests buffer sizes are fine. But check for edge cases in the D4 kernel launch dimensions — if `grid_dim` is computed from clause_cap and it's 0, that would crash.

**Step 3: If found, fix the allocation/launch parameters**

Add `.max(1)` guards to any dimension that could be 0.

**Step 4: Test and commit**

```bash
cargo build -p pyxlog --features host-io --release
CUDA_LAUNCH_BLOCKING=1 python -m pytest python/tests/test_coins.py::test_coins_forward_backward_expected_true -xvs
git add -A && git commit -m "fix: guard D4 kernel launch parameters for small CNFs"
```

---

### Task 2C: Fix — GpuCircuitCache buffer issue

**Applies if:** Diagnostic shows crash is during `GpuCircuitCache::new` itself (after "Cache config" but before "GpuCircuitCache created").

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs:230-469` (GpuCircuitCache::new)
- Test: `python/tests/test_coins.py`

**Root cause hypothesis:** `GpuCircuitCache::new` allocates many GPU buffers based on `GpuCircuitCacheConfig`. If any capacity is 0, or if `memset_zeros` on a 0-length buffer fails, the subsequent kernel launches on those buffers will crash.

**Step 1: Add bounds checks to GpuCircuitCache::new**

Check that all capacities are > 0 before allocating:

```rust
if config.node_cap == 0 || config.edge_cap == 0 || config.level_cap == 0 || config.var_cap == 0 {
    return Err(XlogError::Compilation(format!(
        "GpuCircuitCache requires non-zero capacities: node={}, edge={}, level={}, var={}",
        config.node_cap, config.edge_cap, config.level_cap, config.var_cap
    )));
}
```

**Step 2: Fix zero-capacity handling**

If the error triggers, trace back to `default_cache_config` and fix the capacity computation. For a small CNF (var_cap=4), the values should be:
- `node_cap = smooth_node_cap = (4*5).max(1024) * frontier_cap_factor` = at least 8192
- `edge_cap = smooth_edge_cap = node_cap * 2` = at least 16384
- `level_cap = smooth_node_cap` = at least 8192
- `var_cap = cnf.var_cap` = 4

These all look safe. But verify at runtime.

**Step 3: Test and commit**

```bash
cargo build -p pyxlog --features host-io --release
CUDA_LAUNCH_BLOCKING=1 python -m pytest python/tests/test_coins.py::test_coins_forward_backward_expected_true -xvs
git add -A && git commit -m "fix: guard GpuCircuitCache against zero-capacity allocations"
```

---

### Task 3: Validate all coins tests pass

**Files:**
- Test: `python/tests/test_coins.py`

**Prerequisite:** Task 2A, 2B, or 2C completed (crash fixed).

**Step 1: Remove diagnostic logging**

Remove all `eprintln!("[diag]..."` and `eprintln!("[template]..."` lines added in Tasks 1-2.

```bash
cd /home/dev/projects/xlog/.worktrees/v0.4.0-alpha-integrated
grep -n 'eprintln!\("\[diag\]' crates/xlog-prob/src/exact.rs crates/xlog-prob/src/compilation/mod.rs
grep -n 'eprintln!\("\[template\]' crates/pyxlog/src/lib.rs
```

Remove each line. Then rebuild:

```bash
cargo build -p pyxlog --features host-io --release
```

**Step 2: Run all 7 coins tests**

```bash
python -m pytest python/tests/test_coins.py -xvs 2>&1
```

Expected: All 7 pass:
- `test_coins_compile_and_register` ✅
- `test_coins_forward_backward_expected_true` ✅
- `test_coins_forward_backward_expected_false` ✅
- `test_coins_training_reduces_loss` ✅
- `test_coins_cache_reuse` ✅
- `test_coins_batching_per_network` ✅
- `test_coins_direct_query_with_constant` ✅

**Step 3: Run circuit cache tests too**

```bash
python -m pytest python/tests/test_circuit_cache.py -xvs 2>&1
```

Expected: All pass with deterministic template_compile_count assertions.

**Step 4: Commit clean state**

```bash
git add crates/xlog-prob/src/exact.rs crates/xlog-prob/src/compilation/mod.rs crates/pyxlog/src/lib.rs
git commit -m "fix: resolve CUDA_ERROR_LAUNCH_FAILED in coins boolean query pipeline

Fixes A1 (constant-bound neural output crash) and A2 (template constraint preservation).
All 7 coins tests and 4 circuit cache tests pass."
```

---

### Task 4: Verify cache determinism (A3)

**Files:**
- Test: `python/tests/test_circuit_cache.py`
- Test: `python/tests/test_coins.py`

**Context:** User's commit `1b657fc8` already replaced flaky timing assertions with deterministic `template_compile_count()` and `template_cache_size()` checks. This task verifies those invariants hold.

**Step 1: Run cache tests 3 times to confirm determinism**

```bash
for i in 1 2 3; do
    echo "=== Run $i ==="
    python -m pytest python/tests/test_circuit_cache.py -v
    echo "Exit code: $?"
done
```

Expected: All pass consistently. No flaky failures.

**Step 2: Verify coins cache_reuse test**

```bash
python -m pytest python/tests/test_coins.py::test_coins_cache_reuse -xvs 2>&1
```

Expected: Pass. Two `forward_backward("win()")` calls return valid losses.

**Step 3: Commit verification note (no code changes needed)**

No code changes needed if all pass. A3 is satisfied by user's existing commit.

---

### Task 5: Verify example/gate tests are CUDA-safe (A4)

**Files:**
- Test: `python/tests/test_example_02_coins.py`
- Test: `python/tests/test_example_03_mnist_multidigit.py`
- Test: `python/tests/test_example_04_hwf.py`
- Test: `python/tests/test_example_05_poker.py`
- Test: `python/tests/test_example_05_poker_filename_parsing.py`
- Test: `python/tests/test_example_05_poker_model_focus.py`
- Test: `python/tests/test_example_06_clutrr.py`
- Test: `python/tests/test_neural_accuracy_gates.py`

**Context:** User's commit `f2a84aa5` added CUDA skip guards to example tests. This task verifies those guards work and no test crashes due to missing CUDA or sparse worktree files.

**Step 1: Run example and gate tests with verbose output to verify skip behavior**

```bash
python -m pytest python/tests/ -v --ignore=python/tests/test_circuit_cache.py --ignore=python/tests/test_coins.py
echo "Exit code: $?"
```

Expected: Example tests either PASS or are SKIPPED with clear messages (not CRASH or ERROR).

**Step 3: If any tests crash, add CUDA/file guards**

```python
import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
```

And for missing worktree files:
```python
import os
if not os.path.exists("path/to/required/file"):
    pytest.skip("Required file not in worktree", allow_module_level=True)
```

**Step 4: Commit if changes needed**

```bash
git add python/tests/
git commit -m "test: add CUDA/sparse-worktree guards to example tests"
```

---

### Task 6: Produce coins gate evidence artifact (A5)

**Files:**
- Create: `docs/evidence/2026-02-15-coins-gate.txt`

**Prerequisite:** Tasks 0, 3-5 completed (weights.ptx already committed in Task 0).

**Step 1: Create evidence directory**

```bash
mkdir -p docs/evidence
```

**Step 2: Generate system info header**

```bash
cat > docs/evidence/2026-02-15-coins-gate.txt <<HEADER
=== Coins Alpha Gate Evidence ===
Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Commit: $(git rev-parse HEAD)
Branch: $(git branch --show-current)
GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)
CUDA Toolkit: $(/usr/local/cuda-12.8/bin/nvcc --version 2>/dev/null | tail -1 || echo "N/A")
PTX ISA: 8.7 (all kernels)
Target: sm_70
===

HEADER
```

**Step 3: Run the full gate evidence suite and capture output**

```bash
python -m pytest python/tests/test_coins.py python/tests/test_circuit_cache.py -v 2>&1; echo "EXIT_CODE=$?"
```

**Important:** Do NOT pipe pytest through `tee` without `set -o pipefail` — it hides failures. Instead, run pytest directly, check exit code, then append output:

```bash
python -m pytest python/tests/test_coins.py python/tests/test_circuit_cache.py -v >> docs/evidence/2026-02-15-coins-gate.txt 2>&1
PYTEST_EXIT=$?
echo "pytest exit code: $PYTEST_EXIT" >> docs/evidence/2026-02-15-coins-gate.txt
if [ $PYTEST_EXIT -ne 0 ]; then
    echo "EVIDENCE GENERATION FAILED: pytest returned $PYTEST_EXIT"
    exit 1
fi
```

**Step 4: Verify evidence shows all tests passing**

```bash
grep -c "PASSED" docs/evidence/2026-02-15-coins-gate.txt
grep -c "FAILED" docs/evidence/2026-02-15-coins-gate.txt
```

Expected: 11 PASSED (7 coins + 4 circuit cache), 0 FAILED.

**Step 5: Commit evidence**

```bash
git add docs/evidence/
git commit -m "evidence: coins alpha gate - all tests pass

A5: Captured coins gate evidence artifact with system info.
All 7 coins tests + 4 circuit cache tests pass."
```

---

### Task 7: Final full test suite run

**Step 1: Run all Python tests**

```bash
python -m pytest python/tests/ -v
echo "Exit code: $?"
```

**Step 2: Document results**

Count PASSED, FAILED, SKIPPED, ERROR. Any failures should be known pre-existing issues (not related to coins pipeline).

**Step 3: Final commit with status summary**

```bash
git add -A
git commit -m "chore: finalize coins alpha gate

All coins and circuit cache tests pass.
Template compilation, cache determinism, and CUDA guards verified."
```

---

## Execution Order

0. **Task 0** — Commit weights.ptx separately (infra, isolated from crash fix)
1. **Task 1** — Diagnose crash (add eprintln, run with CUDA_LAUNCH_BLOCKING=1)
2. **Task 2A/2B/2C** — Fix root cause (only one path applies based on diagnosis)
3. **Task 3** — Remove diagnostics, validate all 7 coins tests + 4 cache tests
4. **Task 4** — Verify cache determinism (3 consecutive runs)
5. **Task 5** — Verify example test guards
6. **Task 6** — Produce evidence artifact (weights.ptx already committed in Task 0)
7. **Task 7** — Full suite run + final commit

## Definition of Done

- [ ] All 7 `test_coins.py` tests pass
- [ ] All 4 `test_circuit_cache.py` tests pass (test_cache_hit_same_structure, test_multiple_cache_hits, test_correctness_cached_vs_fresh, test_training_loop_speedup)
- [ ] Cache tests pass 3 consecutive times (no flakiness)
- [ ] Example tests skip gracefully (no crashes)
- [ ] `weights.ptx` committed with weights_force_var_true/weights_restore_var_true symbols (Task 0, separate from crash fix)
- [ ] Evidence artifact in `docs/evidence/` with system info + test output
- [ ] No diagnostic eprintln left in production code
- [ ] All verification commands check exit codes (no piped commands hiding failures)
