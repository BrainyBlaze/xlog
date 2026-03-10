# xlog Codebase Refactoring Design

**Date:** 2026-03-10
**Scope:** Workspace-wide structural refactoring across 12 Rust crates + Python bindings
**Approach:** Risk-ordered waves (Critical → High → Medium → Low)

## Problem

Six parallel analyses of the xlog codebase identified 31 improvements across four severity tiers. Three god modules exceed 4,000 lines each. Six error types across 3 crates share zero `From` conversions. Roughly 1,000 lines of duplicated code span test harnesses, type-specialized download functions, and cross-crate helpers.

## Approach

Work proceeds in five waves ordered by risk. Each wave is independently shippable. No wave changes runtime behavior — the test suite must remain green throughout.

---

## Wave 1: Safety & Foundations

Two changes that must land before any structural work.

### C1 — Unify the error system

The workspace has 6 error types across 3 crates (`XlogError`, `NeuralError`, `TensorSourceError`, `ModuleError`, `FunctionError`, `TypeError`) with zero `From` implementations. Every cross-crate call requires manual `.map_err()`.

**Changes:**
- Add 5 `From<CrateError> for XlogError` implementations: 2 typed (`#[from]`) for neural errors in `xlog-core`, 3 string-wrapping in `xlog-logic` (to avoid circular dependency)
- Remove the `Result<T>` type alias from `xlog-neural` (collides with `xlog-core::Result<T>`)
- Convert 5–15 `.map_err()` call sites to use `?` directly (most of the 1,094 `.map_err()` calls convert cudarc errors and are unaffected)

### C2 — Lock down visibility

Fewer than 60 uses of `pub(crate)` exist in the workspace (concentrated in `xlog-solve` and `xlog-prob`). Most crates have zero. Roughly 40% of public types serve no external consumer.

**Changes:**
- Audit every `pub` item; convert internal-only types to `pub(crate)`
- Target ~200 visibility changes
- Must complete before Wave 2 module splits, which move types across module boundaries

---

## Wave 2: God Module Splits

Three files exceed 4,000 lines. Split the two largest into focused submodules; the third (`executor.rs`, 4,337 LOC) will shrink below 4K after H5 removes profiling boilerplate.

### H1 — Split `provider.rs` (12,809 LOC, 309 functions)

Target structure for `xlog-cuda/src/provider/`:

| Module | Responsibility | ~LOC |
|--------|---------------|------|
| `mod.rs` | `CudaKernelProvider` struct, lifecycle | 1,500 |
| `joins.rs` | hash_join, merge_join, nested_loop | 2,000 |
| `set_ops.rs` | union, difference, intersect, dedup | 1,500 |
| `groupby.rs` | group_by, aggregate, count | 1,500 |
| `filters.rs` | filter, project, select | 1,500 |
| `memory.rs` | buffer management, upload, download | 2,000 |
| `sampling.rs` | MC sampling kernels | 1,500 |

### H2 — Genericize type-specialized functions (part of H1)

Eight `download_column_*` functions (u32, u64, i32, i64, f32, f64, bool, u8) share 85% identical code. Replace with one generic:

```rust
fn download_column<T: DeviceRepr>(&self, buf: &CudaBuffer, col: usize) -> Result<Vec<T>>
```

Seven `create_buffer_from_*_slice` functions get the same treatment. Net reduction: ~500 LOC.

### H3 — Split `pyxlog/lib.rs` (6,202 LOC, 140 functions)

Target structure for `pyxlog/src/`:

| Module | Responsibility | ~LOC |
|--------|---------------|------|
| `lib.rs` | PyO3 module registration, re-exports | 500 |
| `bindings.rs` | Core query/program bindings | 1,500 |
| `neural.rs` | Neural-symbolic bindings | 1,000 |
| `ilp.rs` | ILP/dILP bindings | 1,500 |
| `types.rs` | Wrapper types, conversions, FFI helpers | 1,500 |

### H4 — Remove unnecessary Cargo dependencies

- `xlog-stats/Cargo.toml`: remove `xlog-cuda` (zero usage)
- `xlog-logic/Cargo.toml`: remove `xlog-runtime` (zero usage)

### H5 — Extract profiling wrapper from `execute_node`

The 11-arm match in `executor.rs` repeats 5–6 identical profiling lines in 8 arms (~45 LOC of boilerplate). Extract a `profiled_execute` helper to consolidate.

---

## Wave 3: API Coherence

Public surface cleanup. Deferred until after module splits to avoid merge conflicts in 12K-line files.

### H6 — Naming standardization

| Pattern | Convention |
|---------|-----------|
| GPU→CPU data transfer | `read_*` (replaces download/fetch) |
| CPU→GPU data transfer | `write_*` (replaces upload/set) |
| Cheap accessor | `get_*` |
| Expensive computation | `compute_*` or `evaluate_*` |
| Simple setter | `set_*` |
| Multi-field configuration | `configure_*` |

### H7 — Remove `Result<T>` collision

Already addressed in C1, listed here for tracking completeness.

### H8 — Test harness consolidation

Create `xlog-cuda/tests/common/mod.rs` with shared helpers:
- `setup_provider()` — replace 21 duplicate copies
- `device_row_count()` — replace 13 duplicate copies
- `make_schema()` — replace 3 duplicate copies

Net reduction: ~300 LOC across 21 test files.

---

## Wave 4: Deduplication & Cleanup

Medium-priority items. Each is a focused PR.

### M1 — WFS entry point consolidation

Five public entry points exist; only `evaluate_wfs_rules()` is used in production. Deprecate the other four with `#[deprecated]` for one release, then remove.

### M2 — Relocate schema compatibility check

`schemas_type_compatible` is currently a private method on `CudaKernelProvider` in `provider.rs`. Move it to `xlog-core` as a free function so that the executor and lowerer can also use it without depending on the CUDA provider.

### M3 — mc.rs clone reduction

The 111 `.clone()` calls in `mc.rs` are an optimization target. Profile to identify which clones copy large buffers, then replace with borrows or `Arc` where data is read-only.

### M4 — Python `_commit_rule()` dedup

Extract the duplicated function from `holdout.py` and `promoter.py` into a shared `ilp/_utils.py` module.

### M5 — Split `_run_single_attempt()` (340 LOC)

Break `trainer.py:247-586` into four functions:
- `_prepare_attempt()` — setup and initialization
- `_run_training_loop()` — the step loop
- `_evaluate_candidate()` — scoring
- `_finalize_attempt()` — cleanup and result packaging

### M6 — Fix silent exception swallowing

Replace bare `except Exception: continue` patterns in:
- `promoter.py:370`
- `holdout.py:84`
- `holdout.py:155`

Fix: catch only expected exception types; log unexpected exceptions.

### M7 — Trait abstractions (design only)

Design `DeviceBuffer`, `Registry`, and `StorageBackend` traits. Defer implementation to v0.7+ after module splits stabilize.

### M8 — Row count helper dedup

Three locations define row-count helpers (`executor.rs`, `provider.rs`, test files). Consolidate into a single canonical location in `xlog-cuda`.

### M9 — Frontier type dedup in `gpu_d4.rs`

Merge duplicate frontier type definitions.

### M10 — Join planning logic dedup in `lower.rs`

Extract shared join-planning logic into a reusable function.

### M11 — Python configurable constants

Move hardcoded values to `IlpConfig` dataclass fields:
- Adam learning rate (currently 0.1)
- Mining interval (currently 20)
- LOO threshold (currently 20)

Current values become defaults — no behavioral change.

### M12 — Kernel launch macro for `gpu_cache.rs`

Replace 55+ instances of `func.clone().launch(...)` with a `launch_kernel!` macro that handles clone + error mapping.

---

## Wave 5: Polish

Low-priority items to address opportunistically when touching nearby code:

| ID | Item |
|----|------|
| L1 | Review 8 `allow(dead_code)` suppressions (all currently justified) |
| L2 | Add type hints to internal Python functions in `ilp/` |
| L3 | Add docstrings to Python backend classes |
| L4 | Improve test assertion specificity |
| L5 | Document feature flags in `Cargo.toml` files |
| L6 | Namespace CUDA kernel constants to avoid collision |
| L7 | Add field-level documentation to public Config structs |
| L8 | Clean up module re-exports to reduce namespace pollution |
| L9 | Replace schema iteration with Iterator trait |

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Largest file | 12,809 LOC | ~2,000 LOC |
| Duplicate code | ~1,000 LOC | ~100 LOC |
| Error `From` impls | 0 | ~15 |
| `pub(crate)` usage | ~53 | ~200+ |
| Test helper copies | 21 | 1 |
| Public WFS entry points | 5 | 1–2 |
| God modules (>4K LOC) | 3 | 1 (executor.rs, borderline at ~4.3K) |

## Risk Mitigations

- **Each wave ships independently.** Wave 1 can land without Wave 2.
- **No behavioral changes.** Every refactoring is structural; the full test suite must pass.
- **Module splits preserve blame.** Use `git mv` where possible.
- **Deprecation before removal.** Legacy WFS entry points get `#[deprecated]` for one release cycle.
- **Feature branch per wave.** Enables parallel review.

## Out of Scope

- GPU kernel optimization (v0.7+ — eval phase consumes 72% of MC runtime)
- Multi-GPU architecture (v0.7+)
- Trait abstraction implementation (M7 — design only until splits stabilize)
- Python binding API redesign (split the file, preserve the API)
