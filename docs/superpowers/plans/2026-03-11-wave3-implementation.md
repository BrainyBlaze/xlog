# Wave 3: Executor Decomposition Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `crates/xlog-runtime/src/executor.rs` (4,337 lines) into 7 focused submodules, extract `JoinIndexCache` and `DeltaRelationTracker` as standalone types, and apply ride-along improvements.

**Architecture:** Distributed `impl Executor` blocks across submodules (same pattern as Wave 2 provider split). `JoinIndexCache` and `DeltaRelationTracker` become `pub(crate)` standalone types. All public method signatures remain identical. `#[cfg(test)]` tests stay in `mod.rs`.

**Tech Stack:** Rust, cudarc, xlog-core/xlog-ir/xlog-cuda/xlog-stats crate dependencies

---

## File Structure

After Wave 3, the executor module layout is:

```
crates/xlog-runtime/src/
├── executor/mod.rs           # Executor struct + fields, new(), accessors, reset methods,
│                             # execute_plan(), store helpers, buffer helpers, tests
├── executor/node_dispatch.rs # execute_node() match dispatch + per-node handlers
├── executor/recursive.rs     # execute_recursive_scc(), execute_non_recursive_scc(),
│                             # execute_stratum_impl(), execute_fixpoint()
├── executor/expression.rs    # execute_filter(), eval_predicate_mask_gpu(),
│                             # compare_buffers_mask(), evaluate_arith_expr(),
│                             # const_to_bytes_and_type(), mask ops, wrap_single_column
├── executor/rewrite.rs       # apply_deltas_and_recompute(), rewrite_scan_nth(),
│                             # rewrite_scan_nth_impl(), collect_scan_rels(),
│                             # contains_non_monotonic_ops()
├── executor/join_cache.rs    # JoinIndexCache, JoinIndexKey, CachedJoinIndex (extracted)
└── executor/delta.rs         # DeltaRelationTracker (new extraction)
```

**What stays in mod.rs (~1,800 lines):**
- Executor struct definition + all field accessors (lines 166–377)
- `execute_plan()` (lines 392–429) — entry point, stays with struct
- `store_put()` / `store_remove()` / `get_rel_name()` — private helpers that touch multiple fields
- `get_or_create_rel_name()`, `create_empty_buffer()`, `clone_buffer()`, `clone_device_row_count()`, `buffer_row_count()` — shared utility methods
- `RelationDelta` struct + impl (lines 135–145) — used only in rewrite.rs + tests
- `execute_stratum()` stub (lines 1577–1583) — kept near `execute_plan()` for discoverability (spec §2 suggests recursive.rs, but it's a 6-line error stub with no SCC logic)
- `execute_scan()` (lines 1590–1606) — simple store lookup
- `#[cfg(test)] mod tests` (lines 3065–4337) — entire test block
- `expr_may_be_float()` — used by both expression.rs and test-only helpers

**Spec deviations (intentional):**
- `execute_project()` / `project_schema()` → expression.rs (spec §1 lists "project" under node_dispatch.rs, but `execute_project` delegates to `evaluate_arith_expr` for computed columns — co-locating with expression evaluation is more cohesive)
- `execute_stratum()` stub → mod.rs (spec §2 maps to recursive.rs, but the stub contains no SCC logic — it returns an error directing callers to `execute_plan()`)
- Visibility tightens: plan achieves ~12 of spec's ~25 target; remaining are `pub` → `pub(crate)` demotions deferred to Wave 5 per spec §2 note

**What moves:**
- `node_dispatch.rs` (~190 lines): `execute_node()` match
- `recursive.rs` (~690 lines): `execute_recursive_scc()`, `execute_non_recursive_scc()`, `execute_stratum_impl()`, `execute_fixpoint()`
- `expression.rs` (~530 lines): filter/predicate/arith/mask methods
- `rewrite.rs` (~230 lines): delta recompute + tree rewriting
- `join_cache.rs` (~120 lines): extracted JoinIndexCache type
- `delta.rs` (~50 lines): new DeltaRelationTracker wrapper

## Chunk 1: Structural Scaffolding + JoinIndexCache Extraction

### Task 1: Rename executor.rs → executor/mod.rs

**Files:**
- Rename: `crates/xlog-runtime/src/executor.rs` → `crates/xlog-runtime/src/executor/mod.rs`

- [ ] **Step 1: Create executor directory and rename file**

```bash
cd crates/xlog-runtime/src
mkdir -p executor
git mv executor.rs executor/mod.rs
```

- [ ] **Step 2: Verify build**

Run: `cargo check -p xlog-runtime`
Expected: PASS (Rust treats `module.rs` and `module/mod.rs` identically)

- [ ] **Step 3: Verify workspace compiles**

Run: `cargo check --workspace --exclude pyxlog`
Expected: PASS — all `use xlog_runtime::executor::*` paths still resolve

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-runtime/src/executor/mod.rs
git commit -m "refactor(xlog-runtime): rename executor.rs to executor/mod.rs for Wave 3 split"
```

---

### Task 2: Extract JoinIndexCache to join_cache.rs

**Files:**
- Create: `crates/xlog-runtime/src/executor/join_cache.rs`
- Modify: `crates/xlog-runtime/src/executor/mod.rs`

The JoinIndexCache type (lines 26–133) is a self-contained LRU cache with no Executor field access. It can be extracted as-is.

- [ ] **Step 1: Create join_cache.rs with extracted types**

Create `crates/xlog-runtime/src/executor/join_cache.rs` containing:
- `JoinIndexKey` struct (from mod.rs lines 26–31)
- `CachedJoinIndex` struct (from mod.rs lines 33–37)
- `JoinIndexCache` struct + impl (from mod.rs lines 39–133)

All three types become `pub(crate)`. Add the required imports:

```rust
use std::collections::HashMap;
use xlog_core::RelId;
use xlog_cuda::JoinIndexV2;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct JoinIndexKey {
    pub(crate) rel: RelId,
    pub(crate) version: u64,
    pub(crate) key_cols: Vec<usize>,
}

struct CachedJoinIndex {
    index: JoinIndexV2,
    bytes: u64,
    last_used: u64,
}

pub(crate) struct JoinIndexCache {
    entries: HashMap<JoinIndexKey, CachedJoinIndex>,
    clock: u64,
    total_bytes: u64,
    pub(crate) max_bytes: u64,
}
```

Keep all impl methods identical. Change visibility of methods to `pub(crate)`.

- [ ] **Step 2: Update mod.rs — remove extracted types, add module declaration**

At top of mod.rs, add:
```rust
mod join_cache;
use join_cache::JoinIndexCache;
```

Remove the `JoinIndexKey`, `CachedJoinIndex`, and `JoinIndexCache` struct definitions and `impl JoinIndexCache` block (lines 26–133) from mod.rs.

**Note**: `JoinIndexKey` is not imported here — it is only needed by `execute_join()` which moves to `node_dispatch.rs` in Task 5. That file imports it directly via `use super::join_cache::JoinIndexKey;`.

- [ ] **Step 3: Verify build**

Run: `cargo check -p xlog-runtime`
Expected: PASS

- [ ] **Step 4: Run tests**

Run: `cargo test -p xlog-runtime --release`
Expected: PASS — JoinIndexCache is used internally, no API change

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-runtime/src/executor/join_cache.rs crates/xlog-runtime/src/executor/mod.rs
git commit -m "refactor(xlog-runtime): extract JoinIndexCache to executor/join_cache.rs"
```

---

### Task 3: Create DeltaRelationTracker in delta.rs

**Files:**
- Create: `crates/xlog-runtime/src/executor/delta.rs`
- Modify: `crates/xlog-runtime/src/executor/mod.rs`

This extracts the ad-hoc `delta_rel_by_pred: HashMap<String, (RelId, String)>` pattern used in `execute_recursive_scc()` into a named type. The type wraps the HashMap and provides named operations. This task only creates the type — Task 6 (recursive.rs) will migrate `execute_recursive_scc()` to use it.

- [ ] **Step 1: Create delta.rs**

Create `crates/xlog-runtime/src/executor/delta.rs`:

```rust
use std::collections::HashMap;
use xlog_core::RelId;

/// Tracks delta relation name mappings during semi-naive fixpoint iteration.
///
/// Each recursive predicate gets a synthetic delta relation (with a unique RelId
/// and store name). This tracker maps predicate names to their delta identifiers.
pub(crate) struct DeltaRelationTracker {
    /// Maps predicate name → (delta RelId, delta store name)
    entries: HashMap<String, (RelId, String)>,
}

impl DeltaRelationTracker {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a delta relation for a recursive predicate.
    pub(crate) fn insert(&mut self, pred: String, rel_id: RelId, store_name: String) {
        self.entries.insert(pred, (rel_id, store_name));
    }

    /// Look up the delta (RelId, store_name) for a predicate.
    pub(crate) fn get(&self, pred: &str) -> Option<&(RelId, String)> {
        self.entries.get(pred)
    }

    /// Iterate over all (pred_name, (rel_id, store_name)) entries.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, &(RelId, String))> {
        self.entries.iter()
    }

    /// Consume the tracker, yielding owned entries for cleanup.
    pub(crate) fn into_inner(self) -> HashMap<String, (RelId, String)> {
        self.entries
    }
}
```

- [ ] **Step 2: Add module declaration in mod.rs**

Add to mod.rs module declarations (near `mod join_cache;`):
```rust
mod delta;
```

No `use` import yet — Task 6 will add it when recursive.rs adopts the type.

- [ ] **Step 3: Verify build**

Run: `cargo check -p xlog-runtime`
Expected: PASS (dead_code warning on DeltaRelationTracker expected until Task 6)

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-runtime/src/executor/delta.rs crates/xlog-runtime/src/executor/mod.rs
git commit -m "refactor(xlog-runtime): add DeltaRelationTracker in executor/delta.rs"
```

---

## Chunk 2: Expression + Node Dispatch Extraction

### Task 4: Extract expression evaluation methods to expression.rs

**Files:**
- Create: `crates/xlog-runtime/src/executor/expression.rs`
- Modify: `crates/xlog-runtime/src/executor/mod.rs`

Move these methods from mod.rs into a new `impl Executor` block in expression.rs:

| Method | Current line | Visibility |
|--------|-------------|------------|
| `execute_filter` | 1609 | `pub` (stays pub — spec §2 notes it has no verified cross-crate production callers but keeps pub for now) |
| `eval_predicate_mask_gpu` | 1618 | `fn` → `pub(crate) fn` (called from node_dispatch.rs conditional arm) |
| `compare_buffers_mask` | 1708 | `fn` (private to Executor) |
| `mask_and` | 1791 | `fn` |
| `mask_or` | 1816 | `fn` |
| `mask_not` | 1841 | `fn` |
| `mask_filled` | 1861 | `fn` |
| `evaluate_arith_expr` | 2313 | `fn` → `pub(crate) fn` (called from node_dispatch.rs via execute_project) |
| `const_to_bytes_and_type` | 2410 | `fn` → `pub(crate) fn` (called from node_dispatch via evaluate_arith_expr) |
| `wrap_single_column` | 2272 | `fn` → `pub(crate) fn` (called from expression methods + node_dispatch) |
| `execute_project` | 2430 | `fn` → `pub(crate) fn` (called from node_dispatch.rs) |
| `project_schema` | 2467 | `fn` → `pub(crate) fn` (called from execute_project) |

**Note**: `expr_may_be_float()` (line 1984) stays in mod.rs because it is used by both expression.rs production code AND the `#[cfg(test)]` helpers (`evaluate_expr_as_f64`, `evaluate_expr_as_i64`) that remain in mod.rs. Making it `pub(crate)` in mod.rs lets expression.rs call `Self::expr_may_be_float()`.

The `#[cfg(test)]` functions `evaluate_predicate`, `evaluate_expr_as_f64`, `evaluate_expr_as_i64` (lines 1892–2270) stay in mod.rs — they are test-only CPU-side evaluators, not production GPU code.

- [ ] **Step 1: Create expression.rs**

Create `crates/xlog-runtime/src/executor/expression.rs` with:

```rust
//! Expression evaluation methods for the Executor.
//!
//! Production GPU-accelerated filter, predicate mask, arithmetic expression,
//! and mask operation methods.

use cudarc::driver::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{arith_kernels, filter_kernels, ARITH_MODULE, FILTER_MODULE};
use xlog_cuda::CudaBuffer;
use xlog_ir::{CompareOp, ConstValue, Expr, ProjectExpr};

use super::Executor;

impl Executor {
    // Paste the following methods here (cut from mod.rs):
    // - execute_filter (pub fn)
    // - eval_predicate_mask_gpu (pub(crate) fn)
    // - compare_buffers_mask (fn)
    // - mask_and (fn)
    // - mask_or (fn)
    // - mask_not (fn)
    // - mask_filled (fn)
    // - wrap_single_column (pub(crate) fn)
    // - evaluate_arith_expr (pub(crate) fn)
    // - const_to_bytes_and_type (pub(crate) fn)
    // - execute_project (pub(crate) fn)
    // - project_schema (pub(crate) fn)
}
```

Copy each method body verbatim from mod.rs. Apply these visibility changes:
- `eval_predicate_mask_gpu`: `fn` → `pub(crate) fn`
- `wrap_single_column`: `fn` → `pub(crate) fn`
- `evaluate_arith_expr`: `fn` → `pub(crate) fn`
- `const_to_bytes_and_type`: `fn` → `pub(crate) fn`
- `execute_project`: `fn` → `pub(crate) fn`
- `project_schema`: `fn` → `pub(crate) fn`

Apply ride-along error context improvements as methods are relocated:
- Replace `XlogError::Execution(format!("..."))` with `XlogError::execution_ctx(op, detail, &source)` where the original pattern is `format!("Failed to X: {}", e)` or similar single-source patterns.

- [ ] **Step 2: Update mod.rs**

Add module declaration:
```rust
mod expression;
```

Remove the moved methods from the `impl Executor` block in mod.rs. Keep `expr_may_be_float()` in mod.rs but change its visibility to `pub(crate)`:

```rust
pub(crate) fn expr_may_be_float(expr: &Expr, schema: &Schema) -> bool {
```

- [ ] **Step 3: Verify build**

Run: `cargo check -p xlog-runtime`
Expected: PASS

- [ ] **Step 4: Run tests**

Run: `cargo test -p xlog-runtime --release`
Expected: PASS

- [ ] **Step 5: Run workspace check**

Run: `cargo check --workspace --exclude pyxlog`
Expected: PASS — no cross-crate signature changes

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-runtime/src/executor/expression.rs crates/xlog-runtime/src/executor/mod.rs
git commit -m "refactor(xlog-runtime): extract expression evaluation to executor/expression.rs"
```

---

### Task 5: Extract node dispatch to node_dispatch.rs

**Files:**
- Create: `crates/xlog-runtime/src/executor/node_dispatch.rs`
- Modify: `crates/xlog-runtime/src/executor/mod.rs`

Move these methods from mod.rs:

| Method | Current line | Visibility |
|--------|-------------|------------|
| `execute_node` | 694 | `pub` (cross-crate: mc.rs:1783) |
| `execute_join` | 2494 | `fn` |
| `execute_groupby` | 2640 | `fn` |
| `execute_union` | 2658 | `fn` |
| `execute_distinct` | 2679 | `fn` |
| `execute_diff` | 2686 | `fn` |
| `execute_tensor_masked_join` | 2796 | `fn` |

**Note on `execute_node` cross-module calls**: `execute_node()` calls methods that now live in other submodules:
- `execute_filter()` → expression.rs (pub(crate))
- `execute_project()` → expression.rs (pub(crate))
- `execute_fixpoint()` → recursive.rs (to be extracted in Task 6)

These work because all submodules use `impl Executor` blocks — method calls resolve across files automatically.

- [ ] **Step 1: Create node_dispatch.rs**

Create `crates/xlog-runtime/src/executor/node_dispatch.rs` with:

```rust
//! RIR node dispatch and per-node execution handlers.

use std::borrow::Cow;
use std::collections::HashMap;

use xlog_core::{AggOp, RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::{CudaBuffer, JoinType as CudaJoinType};
use xlog_ir::{JoinType, ProjectExpr, RirNode};

use crate::ilp_registry::{read_device_row_count, IlpMask, IlpTagEntry, IlpTaggedResult};

use super::join_cache::JoinIndexKey;
use super::Executor;

impl Executor {
    // Paste the following methods here (cut from mod.rs):
    // - execute_node (pub fn) — the big match dispatch
    // - execute_join (fn) — includes estimate_join_index_bytes nested fn
    // - execute_groupby (fn)
    // - execute_union (fn)
    // - execute_distinct (fn)
    // - execute_diff (fn)
    // - execute_tensor_masked_join (fn)
}
```

Copy each method body verbatim. The `execute_join` method accesses `self.join_index_cache` and `self.stats` — this works because it's still `impl Executor`.

Apply ride-along error context improvements where `format!("Failed to X: {}", e)` patterns appear.

- [ ] **Step 2: Update mod.rs**

Add module declaration:
```rust
mod node_dispatch;
```

Remove the moved methods from the `impl Executor` block in mod.rs.

Also remove the `MAX_FIXPOINT_ITERATIONS` constant from mod.rs — it moves to recursive.rs in Task 6. (If it causes a compile error here before Task 6, keep it temporarily and move in Task 6.)

- [ ] **Step 3: Verify build**

Run: `cargo check -p xlog-runtime`
Expected: PASS

- [ ] **Step 4: Run tests**

Run: `cargo test -p xlog-runtime --release`
Expected: PASS

- [ ] **Step 5: Run workspace check**

Run: `cargo check --workspace --exclude pyxlog`
Expected: PASS — `execute_node` is pub, signature unchanged

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-runtime/src/executor/node_dispatch.rs crates/xlog-runtime/src/executor/mod.rs
git commit -m "refactor(xlog-runtime): extract node dispatch to executor/node_dispatch.rs"
```

---

## Chunk 3: Recursive + Rewrite Extraction

### Task 6: Extract recursive SCC execution to recursive.rs

**Files:**
- Create: `crates/xlog-runtime/src/executor/recursive.rs`
- Modify: `crates/xlog-runtime/src/executor/mod.rs`

Move these methods from mod.rs:

| Method | Current line | Visibility |
|--------|-------------|------------|
| `execute_recursive_scc` | 960 | `pub` (cross-crate: mc.rs:1722/1757) |
| `execute_non_recursive_scc` | 658 | `pub` (cross-crate: mc.rs:1724/1759) |
| `execute_stratum_impl` | 888 | `fn` |
| `execute_fixpoint` | 2722 | `fn` |
| `MAX_FIXPOINT_ITERATIONS` | 2691 | const |

This task also migrates `execute_recursive_scc()` to use `DeltaRelationTracker` from delta.rs.

- [ ] **Step 1: Create recursive.rs**

Create `crates/xlog-runtime/src/executor/recursive.rs` with:

```rust
//! Recursive SCC execution using semi-naive fixpoint iteration.

use std::collections::{HashMap, HashSet};

use xlog_core::{RelId, Result, Schema, XlogError};
use xlog_cuda::CudaBuffer;
use xlog_ir::{ExecutionPlan, RirNode, Stratum};

use super::delta::DeltaRelationTracker;
use super::Executor;

impl Executor {
    /// Maximum iterations for fixpoint computation to prevent infinite loops
    const MAX_FIXPOINT_ITERATIONS: usize = 1000;

    // Paste the following methods here (cut from mod.rs):
    // - execute_non_recursive_scc (pub fn)
    // - execute_stratum_impl (fn)
    // - execute_recursive_scc (pub fn) — migrate to use DeltaRelationTracker
    // - execute_fixpoint (fn)
}
```

**DeltaRelationTracker migration in `execute_recursive_scc()`:**

Replace the current ad-hoc pattern:
```rust
let mut delta_rel_by_pred: HashMap<String, (RelId, String)> = HashMap::new();
// ... inserts ...
delta_rel_by_pred.get(&pred_name)
// ... cleanup loop ...
for (_pred, (rel_id, delta_name)) in delta_rel_by_pred { ... }
```

With:
```rust
let mut delta_tracker = DeltaRelationTracker::new();
// ... inserts via delta_tracker.insert(pred, rel_id, name) ...
delta_tracker.get(&pred_name)
// ... cleanup via delta_tracker.into_inner() ...
for (_pred, (rel_id, delta_name)) in delta_tracker.into_inner() { ... }
```

This is a mechanical replacement — same logic, named type.

- [ ] **Step 2: Update mod.rs**

Add to module declarations:
```rust
mod recursive;
```

Add the `use` import for DeltaRelationTracker (needed by recursive.rs via `super::delta`):
```rust
// delta.rs is already declared (Task 3), no additional declaration needed
```

Remove the moved methods and `MAX_FIXPOINT_ITERATIONS` const from the `impl Executor` block in mod.rs.

- [ ] **Step 3: Verify build**

Run: `cargo check -p xlog-runtime`
Expected: PASS — delta.rs dead_code warning should be gone now

- [ ] **Step 4: Run tests**

Run: `cargo test -p xlog-runtime --release`
Expected: PASS

- [ ] **Step 5: Run workspace check**

Run: `cargo check --workspace --exclude pyxlog`
Expected: PASS — `execute_recursive_scc` and `execute_non_recursive_scc` are pub, signatures unchanged

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-runtime/src/executor/recursive.rs crates/xlog-runtime/src/executor/mod.rs
git commit -m "refactor(xlog-runtime): extract recursive SCC execution to executor/recursive.rs"
```

---

### Task 7: Extract rewrite/delta recompute to rewrite.rs

**Files:**
- Create: `crates/xlog-runtime/src/executor/rewrite.rs`
- Modify: `crates/xlog-runtime/src/executor/mod.rs`

Move these methods from mod.rs:

| Method | Current line | Visibility |
|--------|-------------|------------|
| `apply_deltas_and_recompute` | 435 | `pub` (no verified cross-crate callers — candidate for pub(crate) in Wave 5) |
| `collect_scan_rels` | 1336 | `fn` → `pub(crate) fn` (also called from recursive.rs) |
| `rewrite_scan_nth` | 1369 | `fn` → `pub(crate) fn` (also called from recursive.rs) |
| `rewrite_scan_nth_impl` | 1381 | `fn` |

**Note**: `contains_non_monotonic_ops` is a nested `fn` inside `apply_deltas_and_recompute` (line 533). It moves along with its parent — keep it as a nested fn.

**Note**: `collect_scan_rels` and `rewrite_scan_nth` are called from `execute_recursive_scc()` (now in recursive.rs). They need to be `pub(crate)` so recursive.rs can call `Self::collect_scan_rels()` and `Self::rewrite_scan_nth()`.

- [ ] **Step 1: Create rewrite.rs**

Create `crates/xlog-runtime/src/executor/rewrite.rs` with:

```rust
//! Tree rewriting and incremental delta recomputation.

use std::collections::{HashMap, HashSet};

use xlog_core::{RelId, Result, Schema, XlogError};
use xlog_cuda::CudaBuffer;
use xlog_ir::{ExecutionPlan, JoinType, RirNode, Stratum};

use super::Executor;
pub(crate) use super::RelationDelta;

impl Executor {
    // Paste the following methods here (cut from mod.rs):
    // - apply_deltas_and_recompute (pub fn)
    //   (includes nested fn contains_non_monotonic_ops)
    // - collect_scan_rels (pub(crate) fn)
    // - rewrite_scan_nth (pub(crate) fn)
    // - rewrite_scan_nth_impl (fn)
}
```

Change `collect_scan_rels` and `rewrite_scan_nth` visibility from `fn` to `pub(crate) fn`.

- [ ] **Step 2: Update mod.rs**

Add module declaration:
```rust
mod rewrite;
```

Remove the moved methods from the `impl Executor` block in mod.rs.

- [ ] **Step 3: Verify build**

Run: `cargo check -p xlog-runtime`
Expected: PASS

- [ ] **Step 4: Run tests**

Run: `cargo test -p xlog-runtime --release`
Expected: PASS

- [ ] **Step 5: Run workspace check**

Run: `cargo check --workspace --exclude pyxlog`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-runtime/src/executor/rewrite.rs crates/xlog-runtime/src/executor/mod.rs
git commit -m "refactor(xlog-runtime): extract rewrite/delta recompute to executor/rewrite.rs"
```

---

## Chunk 4: Verification + Gate

### Task 8: Full Verification Gate

**Files:**
- No file changes — verification only

- [ ] **Step 1: Workspace test**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: All tests pass

- [ ] **Step 2: CUDA certification**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`
Expected: 206/206

- [ ] **Step 3: pyxlog compile gate**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 4: Python ILP sparse gate**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse.py -v --timeout=120`
Expected: All tests pass

- [ ] **Step 5: Python ILP reliability gate**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600`
Expected: 20/20

- [ ] **Step 6: Verify mod.rs line count**

Run: `wc -l crates/xlog-runtime/src/executor/mod.rs`
Expected: ~2,600–2,800 lines (down from 4,337 — the test block alone is ~1,270 lines)

Check submodule line counts:
```bash
wc -l crates/xlog-runtime/src/executor/*.rs
```

- [ ] **Step 7: Verify no regressions in cross-crate callers**

Run:
```bash
cargo check -p xlog-prob
cargo check -p xlog-gpu
```
Expected: Both pass — all `execute_node`, `execute_recursive_scc`, `execute_non_recursive_scc` still resolve

### Task 9: Snapshot worktree for rollback reference

- [ ] **Step 1: Create pre-Wave-3 snapshot**

This is done at the start of execution, not the end. The executor creates it automatically when using subagent-driven-development.

---

## Appendix: Import Reference

### expression.rs imports
```rust
use cudarc::driver::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{arith_kernels, filter_kernels, ARITH_MODULE, FILTER_MODULE};
use xlog_cuda::CudaBuffer;
use xlog_ir::{CompareOp, ConstValue, Expr, ProjectExpr};
use super::Executor;
```

### node_dispatch.rs imports
```rust
use std::borrow::Cow;
use std::collections::HashMap;
use xlog_core::{AggOp, RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::{CudaBuffer, JoinType as CudaJoinType};
use xlog_ir::{JoinType, ProjectExpr, RirNode};
use crate::ilp_registry::{read_device_row_count, IlpMask, IlpTagEntry, IlpTaggedResult};
use super::join_cache::JoinIndexKey;
use super::Executor;
```

### recursive.rs imports
```rust
use std::collections::{HashMap, HashSet};
use xlog_core::{RelId, Result, Schema, XlogError};
use xlog_cuda::CudaBuffer;
use xlog_ir::{ExecutionPlan, RirNode, Stratum};
use super::delta::DeltaRelationTracker;
use super::Executor;
```

### rewrite.rs imports
```rust
use std::collections::{HashMap, HashSet};
use xlog_core::{RelId, Result, Schema, XlogError};
use xlog_cuda::CudaBuffer;
use xlog_ir::{ExecutionPlan, JoinType, RirNode, Stratum};
use super::Executor;
pub(crate) use super::RelationDelta;
```

### mod.rs (remaining) imports
```rust
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use xlog_core::{AggOp, RelId, Result, RuntimeConfig, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_ir::{Expr, Stratum};
use xlog_stats::{StatsManager, StatsSnapshot};

use crate::ilp_registry::{IlpRegistry, IlpTaggedResult};
use crate::profiler::{ExecutionStats, Profiler};
use crate::RelationStore;
```

Note: `std::borrow::Cow`, `cudarc::driver::{LaunchAsync, LaunchConfig}`, `xlog_cuda::JoinType`, `xlog_ir::{CompareOp, ConstValue, ExecutionPlan, JoinType, ProjectExpr, RirNode}`, `xlog_cuda::provider::*` imports move out of mod.rs into the submodules that use them. The `IlpMask`, `IlpTagEntry`, `read_device_row_count` imports also move to node_dispatch.rs.

## Appendix: Method Location Summary

| Method | Before | After | Visibility change |
|--------|--------|-------|-------------------|
| `new()` | mod.rs:194 | mod.rs | — |
| `new_with_config()` | mod.rs:199 | mod.rs | — |
| `set_profiling()` | mod.rs:220 | mod.rs | — |
| `is_profiling()` | mod.rs:229 | mod.rs | — |
| `execution_stats()` | mod.rs:236 | mod.rs | — |
| `store()` | mod.rs:241 | mod.rs | — |
| `store_mut()` | mod.rs:246 | mod.rs | — |
| `ilp_registry_mut()` | mod.rs:251 | mod.rs | — |
| `ilp_last_result()` | mod.rs:256 | mod.rs | — |
| `put_relation()` | mod.rs:261 | mod.rs | — |
| `stats()` | mod.rs:266 | mod.rs | — |
| `reset_for_mc()` | mod.rs:273 | mod.rs | — |
| `reset_for_mc_relations()` | mod.rs:290 | mod.rs | — |
| `reset_for_ilp()` | mod.rs:319 | mod.rs | — |
| `stats_mut()` | mod.rs:329 | mod.rs | — |
| `stats_snapshot()` | mod.rs:336 | mod.rs | — |
| `store_put()` | mod.rs:346 | mod.rs | — |
| `store_remove()` | mod.rs:353 | mod.rs | — |
| `register_relation()` | mod.rs:368 | mod.rs | — |
| `get_rel_name()` | mod.rs:375 | mod.rs | — |
| `execute_plan()` | mod.rs:392 | mod.rs | — |
| `execute_stratum()` | mod.rs:1577 | mod.rs | — |
| `execute_scan()` | mod.rs:1590 | mod.rs | — |
| `expr_may_be_float()` | mod.rs:1984 | mod.rs | `fn` → `pub(crate) fn` |
| `get_or_create_rel_name()` | mod.rs:2984 | mod.rs | — |
| `create_empty_buffer()` | mod.rs:2996 | mod.rs | — |
| `clone_buffer()` | mod.rs:3001 | mod.rs | — |
| `clone_device_row_count()` | mod.rs:3040 | mod.rs | — |
| `buffer_row_count()` | mod.rs:3050 | mod.rs | — |
| `evaluate_predicate()` | mod.rs:1892 | mod.rs (test-only) | — |
| `evaluate_expr_as_f64()` | mod.rs:2008 | mod.rs (test-only) | — |
| `evaluate_expr_as_i64()` | mod.rs:2130 | mod.rs (test-only) | — |
| `execute_filter()` | mod.rs:1609 | expression.rs | — |
| `eval_predicate_mask_gpu()` | mod.rs:1618 | expression.rs | `fn` → `pub(crate) fn` |
| `compare_buffers_mask()` | mod.rs:1708 | expression.rs | — |
| `mask_and()` | mod.rs:1791 | expression.rs | — |
| `mask_or()` | mod.rs:1816 | expression.rs | — |
| `mask_not()` | mod.rs:1841 | expression.rs | — |
| `mask_filled()` | mod.rs:1861 | expression.rs | — |
| `wrap_single_column()` | mod.rs:2272 | expression.rs | `fn` → `pub(crate) fn` |
| `evaluate_arith_expr()` | mod.rs:2313 | expression.rs | `fn` → `pub(crate) fn` |
| `const_to_bytes_and_type()` | mod.rs:2410 | expression.rs | `fn` → `pub(crate) fn` |
| `execute_project()` | mod.rs:2430 | expression.rs | `fn` → `pub(crate) fn` |
| `project_schema()` | mod.rs:2467 | expression.rs | `fn` → `pub(crate) fn` |
| `execute_node()` | mod.rs:694 | node_dispatch.rs | — |
| `execute_join()` | mod.rs:2494 | node_dispatch.rs | — |
| `execute_groupby()` | mod.rs:2640 | node_dispatch.rs | — |
| `execute_union()` | mod.rs:2658 | node_dispatch.rs | — |
| `execute_distinct()` | mod.rs:2679 | node_dispatch.rs | — |
| `execute_diff()` | mod.rs:2686 | node_dispatch.rs | — |
| `execute_tensor_masked_join()` | mod.rs:2796 | node_dispatch.rs | — |
| `execute_recursive_scc()` | mod.rs:960 | recursive.rs | — |
| `execute_non_recursive_scc()` | mod.rs:658 | recursive.rs | — |
| `execute_stratum_impl()` | mod.rs:888 | recursive.rs | — |
| `execute_fixpoint()` | mod.rs:2722 | recursive.rs | — |
| `apply_deltas_and_recompute()` | mod.rs:435 | rewrite.rs | — |
| `collect_scan_rels()` | mod.rs:1336 | rewrite.rs | `fn` → `pub(crate) fn` |
| `rewrite_scan_nth()` | mod.rs:1369 | rewrite.rs | `fn` → `pub(crate) fn` |
| `rewrite_scan_nth_impl()` | mod.rs:1381 | rewrite.rs | — |
