# Wave 3: Executor Decomposition + Targeted Python ILP Validation

**Date**: 2026-03-10
**Status**: Approved
**Depends on**: Wave 2 (provider turbofish signatures)

## Overview

Split `crates/xlog-runtime/src/executor.rs` (4,337 lines, 61 methods) into focused
submodules. Extract `DeltaRelationTracker` and `JoinIndexCache` as standalone internal types.
Deduplicate the expression evaluation paths. This wave requires targeted Python ILP gates
because pyxlog directly constructs and drives Executor in the ILP training path.

## Constraints

- Green at wave boundary: workspace + 206/206 + pyxlog check + targeted Python ILP gates
- The Executor public facade is wider than `execute()` — see Section 3 for full surface
- xlog-prob::mc directly calls `execute_node`, `execute_recursive_scc`, and `execute_non_recursive_scc` — these are NOT purely internal
- The expression dedup targets are production execution methods, not `#[cfg(test)]` helpers
- Include `execute_fixpoint()` in the recursive split
- Keep DeltaRelationTracker with owned String keys — no lifetime optimization in this wave
- Sequential after Wave 2 (recommended over parallel worktree)

## 1. File Split: Distributed impl Blocks

```
crates/xlog-runtime/src/
├── executor.rs          → executor/mod.rs
├──                        executor/node_dispatch.rs
├──                        executor/recursive.rs
├──                        executor/expression.rs
├──                        executor/rewrite.rs
├──                        executor/join_cache.rs
└──                        executor/delta.rs
```

### Module Responsibilities

| Module | Content | LOC est. |
|--------|---------|----------|
| `mod.rs` | Executor struct, fields, `execute_plan()` entry point, relation store access, profiler integration, public accessors (set_profiling, execution_stats, store, store_mut, put_relation, register_relation, reset_for_mc, reset_for_mc_relations, reset_for_ilp, ilp_registry_mut, ilp_last_result) | ~500 |
| `node_dispatch.rs` | `execute_node()` match dispatch, per-node handlers (scan, project, filter, distinct, limit), `execute_tensor_masked_join()` (moved as-is, 178 lines) | ~800 |
| `recursive.rs` | `execute_recursive_scc()`, `execute_non_recursive_scc()`, `execute_fixpoint()`, fixpoint iteration using DeltaRelationTracker | ~500 |
| `expression.rs` | Production expression evaluation: `execute_filter`, `eval_predicate_mask_gpu`, `compare_buffers_mask`, `evaluate_arith_expr`, `const_to_bytes_and_type`, mask operations | ~400 |
| `rewrite.rs` | `rewrite_scan_nth_impl()`, `apply_deltas_and_recompute()`, tree rewriting helpers | ~500 |
| `join_cache.rs` | JoinIndexCache extracted as standalone `pub(crate)` struct | ~100 |
| `delta.rs` | DeltaRelationTracker: manages delta relation lifecycle during recursive evaluation | ~200 |

**Total**: ~3,000 LOC (vs 4,337 current — ~31% reduction)

## 2. Public Facade

The public API is wider than `execute_plan()`. Cross-crate callers depend on a significant
surface. All methods listed below must remain `pub` after the split.

### Orchestration & state management

| Method | Location | Called by |
|--------|----------|----------|
| `new()` / `new_with_config()` | executor.rs:189, executor.rs:199 | xlog-gpu/logic.rs, xlog-prob, pyxlog |
| `execute_plan()` | executor.rs (main entry) | xlog-gpu/logic.rs, xlog-prob (exact, wfs) |
| `set_profiling()` / `execution_stats()` | executor.rs | xlog-gpu/logic.rs |
| `store()` / `store_mut()` | executor.rs | xlog-prob (mc, exact, wfs) |
| `put_relation()` / `register_relation()` | executor.rs | xlog-prob, pyxlog |
| `reset_for_mc()` / `reset_for_mc_relations()` | executor.rs | xlog-prob/mc |
| `reset_for_ilp()` | executor.rs | pyxlog ILP path |
| `ilp_registry_mut()` / `ilp_last_result()` | executor.rs | pyxlog ILP path |

### Execution methods called directly by xlog-prob::mc (NOT purely internal)

| Method | Location | Called by | Target module |
|--------|----------|----------|---------------|
| `execute_recursive_scc()` | executor.rs | mc.rs:1722 | → recursive.rs |
| `execute_non_recursive_scc()` | executor.rs | mc.rs (nearby) | → recursive.rs |
| `execute_node()` | executor.rs | mc.rs:1783 | → node_dispatch.rs |

### Additional public methods used cross-crate

| Method | Location | Called by | Target module |
|--------|----------|----------|---------------|
| `apply_deltas_and_recompute()` | executor.rs:435 | xlog-prob | → rewrite.rs |
| `execute_stratum()` | executor.rs:1577 | xlog-prob | → recursive.rs |
| `execute_filter()` | executor.rs:1609 | xlog-prob, internal | → expression.rs |

All of these must remain `pub` (not `pub(crate)`) since xlog-prob and pyxlog are separate
crates. The file split changes where the source lives, not the API surface.

## 3. Expression Evaluation Dedup

The `#[cfg(test)]` helpers `evaluate_expr_as_f64` and `evaluate_expr_as_i64` are NOT the
production extraction targets. The real production methods to extract into `expression.rs`:

| Method | Lines (approx) | Content |
|--------|----------------|---------|
| `execute_filter` | ~80 | Filter execution with predicate evaluation |
| `eval_predicate_mask_gpu` | ~100 | GPU predicate mask generation |
| `compare_buffers_mask` | ~60 | Buffer comparison mask |
| `evaluate_arith_expr` | ~80 | Arithmetic expression evaluation |
| `const_to_bytes_and_type` | ~40 | Constant serialization |

Where these share structure across scalar types, parameterize by `ScalarType` (already in
xlog-core) for dispatch — the executor delegates GPU work to the provider, so it doesn't
need `GpuScalar`.

## 4. DeltaRelationTracker Extraction

New type in `delta.rs`:

```rust
pub(crate) struct DeltaRelationTracker {
    deltas: HashMap<String, CudaBuffer>,  // owned String keys, no lifetime optimization
    generation: usize,
}

impl DeltaRelationTracker {
    pub(crate) fn new() -> Self { ... }
    pub(crate) fn update(&mut self, name: &str, new_delta: CudaBuffer) { ... }
    pub(crate) fn get(&self, name: &str) -> Option<&CudaBuffer> { ... }
    pub(crate) fn is_empty(&self) -> bool { ... }  // fixpoint convergence check
    pub(crate) fn advance_generation(&mut self) { ... }
    pub(crate) fn clear(&mut self) { ... }
}
```

This makes `execute_recursive_scc()` read as: create tracker → loop { execute rules →
update tracker → check fixpoint } → merge finals.

## 5. JoinIndexCache Extraction

Currently ~50 lines of ad-hoc LRU embedded in Executor fields + inline logic. Extract to
`join_cache.rs` as a `pub(crate)` struct with `get_or_build()` / `invalidate()` /
`evict_lru()` API.

Not reusable outside xlog-runtime (GPU-buffer-specific), so `pub(crate)` is correct.

## 6. What Wave 3 Does NOT Do

| Deferred item | Why |
|---------------|-----|
| RIR visitor trait | Cross-crate abstraction, not a local refactor. Revisit in Wave 5 (5c.10). |
| `execute_tensor_masked_join()` decomposition | Single-purpose, 178 lines. Moves to `node_dispatch.rs` as-is. |
| Lifetime-based clone reduction in DeltaRelationTracker | Keep owned String keys first; optimize later once the recursive split is stable. |

## 7. Ride-Along Improvements

| Ride-along | Scope |
|------------|-------|
| **Visibility** | Internal helpers become `pub(crate)`. Methods used by xlog-prob::mc stay `pub`. ~25 visibility tightens. |
| **Error context** | Replace `XlogError::Execution(format!(...))` with `XlogError::execution_ctx(...)` as functions relocate. |
| **Unwrap fixes** | Opportunistic — most executor.rs unwrap/expect calls are below the test module boundary (executor.rs:3066+). Production-path unwraps are few; fix any encountered during the move but do not frame this as a significant remediation target. |

## 8. Gate

| Gate | Required | Rationale |
|------|----------|-----------|
| `cargo test --workspace --all-targets --exclude pyxlog --release` | Yes | Rust workspace |
| `cargo test -p xlog-cuda-tests --test certification_suite --release` | Yes | 206/206 |
| `cargo check -p pyxlog` | Yes | Compile gate |
| `.venv/bin/python -m pytest python/tests/test_ilp_sparse.py -v --timeout=120` | Yes | Targeted ILP sparse gate |
| `.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600` | Yes | 20/20 ILP reliability gate |

Not the full Python matrix — that's Wave 4. Wave 3 requires targeted Python ILP gates
because pyxlog directly constructs and drives Executor in the ILP training path
(lib.rs:4349, lib.rs:5174).

## 9. Parallelizability with Wave 2

**Recommendation: Sequential.** Wave 2 changes provider method signatures (turbofish), and
executor.rs calls those methods. Running Wave 3 in a parallel worktree would require a
mechanical rebase (~40 turbofish additions) — feasible but adds risk for small time savings.

## 10. Diff Profile (estimated)

| Change type | Files | Lines added | Lines removed |
|-------------|-------|-------------|---------------|
| New executor submodules (6 files) | 6 | ~3,000 | — |
| Delete executor.rs (replaced by executor/) | 1 | — | ~4,337 |
| DeltaRelationTracker (new type) | 1 | ~200 | — |
| JoinIndexCache extraction | 1 | ~100 | — |
| Ride-along (visibility, error ctx, unwraps) | within above | ~50 | ~50 |
| **Net** | ~8 files | ~3,350 | ~4,387 |

**Net reduction: ~1,040 lines**

## 11. Risks

| Risk | Mitigation |
|------|-----------|
| DeltaRelationTracker changes fixpoint convergence order | Unit test: same convergence for known recursive programs. ILP reliability gate (20/20) catches regressions. |
| xlog-prob::mc calls executor internals that move files | Methods stay `pub` with identical signatures. Only the source file changes. |
| `rewrite_scan_nth_impl()` is recursive and fragile | Move as-is first, verify with existing rewrite tests. Don't restructure the recursion. |
| pyxlog ILP path accesses Executor beyond execute_plan() | Full public facade documented in Section 2. Verify actual call sites at lib.rs:4349 and lib.rs:5174 during implementation. |
