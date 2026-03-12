# Wave 5: Probabilistic Backend Decomposition + Remaining Coherence/Polish

**Date**: 2026-03-10
**Status**: Approved
**Depends on**: Waves 1–4

## Overview

Wave 5 has three sub-waves, each a separate commit. Sub-waves 5a and 5b decompose the two
largest remaining domain backend files. Sub-wave 5c is the final coherence sweep. The full
documented release matrix is the gate at the 5c boundary.

| Sub-wave | Focus | Commit boundary |
|----------|-------|-----------------|
| 5a | `gpu_d4.rs` decomposition (3,669 lines) | Own commit, workspace + 206/206 |
| 5b | `mc.rs` decomposition (3,399 lines) | Own commit, workspace + 206/206 + targeted MC supplemental |
| 5c | Remaining coherence, naming, exports, test harness, config | Own commit, full release matrix |

## Sub-wave 5a: gpu_d4.rs Decomposition

### File

`crates/xlog-prob/src/compilation/gpu_d4.rs` — 3,669 lines, 39 functions.

### Strategy

Structural split only. The public surface is function-based (not an object model). Preserve
the existing shape: external entry points in `compilation/mod.rs`, lower-level functions
distributed across gpu_d4 submodules.

Current public surface (function-based, not object model):

Entry points in compilation/mod.rs (orchestrate the full pipeline):
- `compile_gpu_d4_and_verify` (mod.rs:117 — compile + verify, no caching)
- `compile_gpu_d4_and_verify_cached` (mod.rs:145 — compile + cache + verify + smooth; used by exact.rs, exact_gpu.rs)

Public functions in gpu_d4.rs:
- `validate_cnf_gpu` (gpu_d4.rs:76)
- `compute_free_var_mask_gpu` (gpu_d4.rs:123)
- `compute_free_var_mask_gpu_gated` (gpu_d4.rs:132)
- `build_frontier_bitset` (gpu_d4.rs:464)
- `build_frontier_dense` (gpu_d4.rs:639)
- `compile_gpu_d4` (gpu_d4.rs:836)
- `compile_gpu_d4_gated` (gpu_d4.rs:847)

Public structs in gpu_d4.rs:
- `GpuCompileConfig` (gpu_d4.rs:49 — re-exported via compilation/mod.rs:30)
- `GpuFrontierBitset` (gpu_d4.rs:395)
- `GpuFrontierDense` (gpu_d4.rs:431)

Internal helpers:
- `exclusive_scan_u32_inplace` (gpu_d4.rs:319, pub(crate) — used by xlog-prob/gpu.rs)

```
crates/xlog-prob/src/compilation/
├── gpu_d4.rs          → gpu_d4/mod.rs       (entry points, config, validation, scan utility)
├──                      gpu_d4/frontier.rs   (frontier types, expansion, node processing)
├──                      gpu_d4/build.rs      (circuit construction, XGCF output)
```

**Implementation note (smoothing.rs removed):** Code analysis during implementation revealed
that smoothing logic lives in `GpuXgcf::smooth_random_vars_device` (compilation/mod.rs:337),
not in gpu_d4.rs. The spec's `smoothing.rs` submodule would have been empty. The split is
3 modules instead of 4, which is a better factoring of the actual code.

### Module Responsibilities

| Module | Content | LOC est. |
|--------|---------|----------|
| `mod.rs` | Config, validation, free-var mask, `exclusive_scan_u32_inplace`, thin compile wrappers | ~450 |
| `frontier.rs` | `D4WorkItem`, `GpuFrontierBitset`, `GpuFrontierDense`, `build_frontier_*`, 4 tests | ~1,480 |
| `build.rs` | `compile_gpu_d4_with_gate`, `alloc_component_scratch`, 5 tests | ~1,850 |

**Total**: ~3,780 LOC (vs 3,669 — slight increase due to `pub(super)` field promotions and
test redistribution)

### Visibility Adaptation

Frontier types (`GpuFrontierBitset`, `GpuFrontierDense`, `D4WorkItem`) were tightened from
`pub` to `pub(crate)` since they are only used within xlog-prob. Their struct fields are
`pub(super)` for cross-submodule access within gpu_d4/. The `pub use` re-exports in
gpu_d4/mod.rs and compilation/mod.rs were removed accordingly.

### Unwrap/Expect Policy

The 301 unwrap/expect calls are predominantly in `mod tests` (starting at gpu_d4.rs:1248).
Sub-wave 5a is framed as a structural split first, with opportunistic invariant cleanup
second:

- **Test code**: `.unwrap()` is acceptable, leave as-is
- **Production code**: Where bare `.unwrap()` exists, add descriptive `.expect("invariant: ...")`
  messages during the move. Convert to `Result` only where the unwrap guards a user-facing
  condition (malformed input, resource exhaustion).
- **Count and document** final unwrap/expect tally in wave completion notes

### 5a Gate

- `cargo test --workspace --all-targets --exclude pyxlog --release` — green
- `cargo test -p xlog-cuda-tests --test certification_suite --release` — 206/206

## Sub-wave 5b: mc.rs Decomposition

### File

`crates/xlog-prob/src/mc.rs` — 3,399 lines.

### Strategy

Split the Monte Carlo inference engine into focused submodules. Preserve the real `McProgram`
public surface.

Current public surface on `McProgram`:
- `evaluate` (mc.rs:322)
- `evaluate_cpu` (mc.rs:394)
- `evaluate_gpu` (mc.rs:549)
- `evaluate_gpu_device` (mc.rs:553)
- `evaluate_gpu_device_with_provider` (mc.rs:558)

```
crates/xlog-prob/src/
├── mc.rs              → mc/mod.rs
├──                      mc/sampling.rs
├──                      mc/evidence.rs
├──                      mc/buffers.rs
├──                      mc/results.rs
```

### Module Responsibilities

| Module | Content | LOC est. |
|--------|---------|----------|
| `mod.rs` | `McProgram` struct, `McEvalConfig`, public `evaluate*` methods, sampling method selection | ~600 |
| `sampling.rs` | Inner sampling loop, per-sample store execution, convergence checking, iteration management | ~900 |
| `evidence.rs` | `EvidenceForcing` struct, `compile_evidence_forcing()`, `resolve_sampling_method()`, clamping setup | ~400 |
| `buffers.rs` | `build_sample_buffers()`, GPU buffer allocation for samples, buffer reuse/cleanup | ~500 |
| `results.rs` | Result aggregation, probability computation from sample counts, `McResult` construction, metadata assembly | ~400 |

**Total**: ~2,800 LOC (vs 3,399 — ~18% reduction)

### Key Coupling

xlog-prob::mc depends on a broad Executor surface:

| Executor method | mc.rs call site |
|----------------|----------------|
| `set_profiling()` | mc.rs:861 |
| `register_relation()` | mc.rs:865 |
| `put_relation()` | mc.rs:869 |
| `reset_for_mc_relations()` | mc.rs:942 |
| `store()` | mc.rs:974, mc.rs:978 |
| `execute_recursive_scc()` | mc.rs:1722 |
| `execute_non_recursive_scc()` | mc.rs (nearby) |
| `execute_node()` | mc.rs:1783 |

By Wave 5, these live across `executor/mod.rs`, `executor/recursive.rs`, and
`executor/node_dispatch.rs` with stable signatures from Wave 3. The mc.rs decomposition
doesn't change which executor methods are called — only where within mc/ the calls live.

### Unwrap/Expect in mc.rs

Only two live `expect(...)` sites found (mc.rs:2713, mc.rs:3285). Sub-wave 5b is primarily
structural, not an unwrap-remediation wave.

### 5b Gate

- `cargo test --workspace --all-targets --exclude pyxlog --release` — green
- `cargo test -p xlog-cuda-tests --test certification_suite --release` — 206/206
- Targeted MC tests as supplemental sub-wave coverage (not a documented release-gate label)

The full documented release matrix runs at the 5c boundary.

## Sub-wave 5c: Remaining Coherence & Polish

### 5c.1 Naming Standardization (N2, N6)

After Waves 2–4, the provider API uses `download_column<T>` and `create_buffer_from_slice<T>`.
Remaining naming inconsistencies:

- Standardize mixed `fetch_*` / `get_*` to `get_*` where appropriate (Rust convention)
- Only rename internal (non-frozen) APIs
- Each rename is a crate-internal find-replace with compile verification
- Estimated ~10–15 renames across the workspace

### 5c.2 Config Coherence (N1, ST4)

12 Config structs exist across 5 crates (xlog-core: 1, xlog-logic: 1, xlog-neural: 1,
xlog-solve: 2, xlog-prob: 7). Full unification is over-engineering. Realistic improvements:

- Add `Default` impls to all Config structs that lack them
- Add `#[non_exhaustive]` to public Config structs **where Rust allows it**
- Add `///` doc comments explaining when/why to customize
- Do NOT create a hierarchical config tree or builder pattern

**Implementation note (`#[non_exhaustive]` constraint):** Rust's `#[non_exhaustive]` blocks ALL
struct literal construction from outside the defining crate, even with `..Default::default()`.
Only 3 of 13 config structs could receive it (those never constructed via struct literals from
external crates): `GpuCompileConfig`, `McEvalConfig`, `WfsConfig`. The remaining 10 use struct
literal construction in pyxlog or test code across crate boundaries, making `#[non_exhaustive]`
a breaking change. This is a Rust language constraint, not an implementation shortfall.

### 5c.3 Test Harness Consolidation (A5)

24 copies of `setup_provider()` across test files (22 in xlog-cuda/tests/, 1 in
xlog-prob/tests/, 1 in xlog-runtime/src/relation.rs). Each crate keeps one canonical copy
in a `tests/common/mod.rs` or `src/test_support.rs` — less DRY but avoids new test-utility
dependency edges between crates. Comment points to pattern origin. Saves ~300 lines
(24 copies → ~5 copies).

**Implementation note (relation.rs exclusion):** The `xlog-runtime/src/relation.rs` copy uses
a different pattern — it's an inline `#[cfg(test)] mod tests` block within a source file, not
a standalone integration test file. Consolidating it would require either making the shared
helper available to `#[cfg(test)]` modules inside `src/` (a different dependency shape) or
moving the tests to a separate integration test file. Left as-is; the 22 xlog-cuda + 1
xlog-prob copies are consolidated into 2 canonical `tests/common/mod.rs` helpers.

### 5c.4 xlog-prob Export Cleanup (ST3)

Add top-level re-exports to `xlog-prob/src/lib.rs`:

```rust
// Primary entry points — these are what external callers actually use
pub use compilation::{compile_gpu_d4_and_verify, compile_gpu_d4_and_verify_cached};
pub use compilation::gpu_d4::GpuCompileConfig;
pub use exact::{ExactDdnnfProgram, ExactResult};
pub use mc::{McProgram, McEvalConfig, McResult};
// WFS: see 5c.5 for consolidation — re-export the primary free functions
```

Existing deep imports continue to work. These add convenience paths. The lower-level
`compile_gpu_d4` and `compile_gpu_d4_gated` are NOT re-exported — they are internal to
the compilation module, called only by the `_and_verify*` orchestrators.

### 5c.5 WFS Entry Point Consolidation (N5)

Currently 5 public WFS entry points as free functions:
- wfs.rs:519
- wfs.rs:586
- wfs.rs:595
- wfs.rs:632
- wfs.rs:637

There is no `WfsEngine` or `evaluate_device` — the surface is function-based. Audit and
reduce to 1–2 primary entry points:
- Keep: primary evaluation entry
- Keep: device-variant if distinct
- Deprecate or make `pub(crate)`: internal helpers that leaked to public API

### 5c.6 Unwrap/Expect Policy Documentation (E6)

After opportunistic fixes in Waves 2–5a/5b, document the final policy:

- **Production code**: `Result<T>` for all fallible operations. `.expect("invariant: ...")` for
  internal invariants that indicate compiler/system bugs.
- **Test code**: `.unwrap()` is acceptable.
- **Build scripts**: `.expect()` with descriptive messages (fix the 8 bare unwraps in
  build.rs).

Produce a final count of remaining unwrap/expect in production paths as a wave artifact.

### 5c.7 Remaining Visibility Tightening

After Waves 2–4 ride-alongs, audit remaining `pub` items that should be `pub(crate)`:
- xlog-prob internal helpers (compilation submodules, WFS internals)
- xlog-solve internal solver state
- xlog-logic internal AST manipulation helpers
- Estimated ~40–60 additional tightens

### 5c.8 Clone Reduction Audit

Profile-guided, not blind removal:
- xlog-prob: sample store handling
- xlog-logic: AST manipulation during lowering
- Schema cloning patterns surviving the provider split
- Document clones deliberately kept with brief comments

### 5c.9 Stale Comment Cleanup

- Remove/update comments referencing pre-refactoring file structure
- Update comments referencing old function names (e.g., `download_column_u32`)
- Remove comments referencing specific line numbers in pre-split files

### 5c.10 RIR Visitor Trait — Revisit Decision

Deferred from Wave 3. After the executor split, assess:
- **If** 3+ distinct RirNode traversal patterns exist across executor/, xlog-prob, and
  xlog-logic → introduce the trait
- **If** only executor uses match dispatch → leave as-is (match arms are explicit and
  greppable)
- Document the decision either way

## 5c Gate (Wave 5 Final Gate)

### Documented release matrix (from v0.4.0-beta-release-design.md)

| # | Gate | Command | Required |
|---|------|---------|----------|
| 1 | Rust workspace (v0.4.0-beta-release-design.md:50) | `cargo test --workspace --all-targets --exclude pyxlog --release` | Yes |
| 2 | CUDA certification (v0.4.0-beta-release-design.md:51) | `cargo test -p xlog-cuda-tests --test certification_suite --release` | Yes (206/206) |
| 3 | Non-slow batch (v0.4.0-beta-release-design.md:52) | | Yes |
| 4 | ILP reliability (v0.4.0-beta-release-design.md:53) | | Yes |
| 5 | ILP sparse (v0.4.0-beta-release-design.md:54) | | Yes |
| 6 | GA reliability (v0.4.0-beta-release-design.md:55) | | Yes |
| 7 | ILP performance (v0.4.0-beta-release-design.md:56) | | Yes |

### Additional build gates (not in documented matrix, required for Wave 5)

| Gate | Command | Required |
|------|---------|----------|
| pyxlog compile | `cargo check -p pyxlog` | Yes |
| Python wheel build | `maturin develop --release -m crates/pyxlog/Cargo.toml` | Yes |

Wave 5 is the final wave — it must pass the full documented matrix plus additional build
gates before the refactoring is considered complete.

### Known Testing Patterns

**Rust early-return skips** (e.g., `run_cli_tests.rs`): Rust's built-in test framework has no
`skip`/`skip_unless` mechanism like pytest. Hardware-dependent tests use early `return` with
`eprintln!` skip messages. This is the standard Rust pattern and is not a Wave 5 regression.
The test's skip conditions (CUDA unavailable, insufficient GPU memory) are hardware-dependent
and cannot be "gamed" in code. Python tests use `pytest.skip()` and `@pytest.mark.slow` which
provide proper deselection semantics.

**Pre-existing slow tests**: `test_non_monotone_simple_cycle` (50K MC samples),
`test_ilp_showcase_all_stages_converge` (subprocess, up to 600s), and
`test_recursive_reach_runs_without_crash` (7×150 ILP steps with recursive candidates) are
marked `@pytest.mark.slow` and excluded from the non-slow batch gate. These are
WSL2-batch-context timing issues, not Wave 5 regressions.

## Diff Profile (estimated)

| Sub-wave | Files changed | Lines added | Lines removed | Net |
|----------|--------------|-------------|---------------|-----|
| 5a: gpu_d4 split | ~5 | ~3,300 | ~3,669 | -369 |
| 5b: mc split | ~6 | ~2,800 | ~3,399 | -599 |
| 5c: coherence/polish | ~25–30 | ~400 | ~600 | -200 |
| **Wave 5 total** | ~35 | ~6,500 | ~7,668 | **-1,168** |

## Risks

| Risk | Mitigation |
|------|-----------|
| gpu_d4 split changes compilation output | D4 compilation is deterministic. Existing exact inference tests validate. 206/206 cert. |
| mc.rs split changes sampling behavior | MC is stochastic but seeded. Reliability gate catches convergence regressions. Evidence clamping tests validate forcing. |
| 5c touches many files | Each 5c item is independently small and testable. Review as focused diffs within the commit. |
| RIR visitor decision deferred too long | Explicit revisit point in 5c.10. Document decision either way. |
| Naming renames break external examples | Only rename internal APIs. Frozen surfaces untouched. Examples updated as part of rename. |

## Overall Refactoring Summary

| Wave | Focus | LOC reduction | Key gate |
|------|-------|--------------|----------|
| 1 | Dependency cleanup + error/type seams | +290 (new seams) | Workspace + 206/206 |
| 2 | Provider decomposition + GpuScalar | -5,990 | Workspace + 206/206 + pyxlog check |
| 3 | Executor decomposition | -1,040 | Workspace + 206/206 + ILP reliability/sparse |
| 4 | Pyxlog FFI extraction | -1,320 | Full Python release matrix |
| 5 | Prob backends + coherence/polish | -1,168 | Full release matrix |
| **Total** | | **~-9,228 net lines** | |

Current state: main is documented as v0.5.0 (ROADMAP.md:4). This refactoring targets
structural improvements without changing the documented version or product surface.
