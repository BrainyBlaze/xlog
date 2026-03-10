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
- `validate_cnf_gpu` (compilation/mod.rs:117)
- `compile_gpu_d4*` entry points (gpu_d4.rs:76, gpu_d4.rs:464, gpu_d4.rs:639)
- `compute_free_var_mask_gpu*` (gpu_d4.rs)
- Internal helpers exposed from gpu_d4.rs (audit during split for pub(crate) candidates)

```
crates/xlog-prob/src/compilation/
├── gpu_d4.rs          → gpu_d4/mod.rs       (entry points, config, result types)
├──                      gpu_d4/frontier.rs   (frontier expansion, node processing)
├──                      gpu_d4/smoothing.rs  (circuit smoothing passes)
├──                      gpu_d4/build.rs      (circuit construction, XGCF output)
```

### Module Responsibilities

| Module | Content | LOC est. |
|--------|---------|----------|
| `mod.rs` | Public entry points (`compile_gpu_d4*`, `compute_free_var_mask_gpu*`), `GpuCompileConfig`, result types, orchestration | ~600 |
| `frontier.rs` | `build_frontier_*`, frontier expansion loop, node state tracking, BFS exploration, cache interaction | ~1,200 |
| `smoothing.rs` | Post-compilation smoothing passes, determinism normalization | ~800 |
| `build.rs` | XGCF circuit layout construction, levelized DAG building, output finalization | ~700 |

**Total**: ~3,300 LOC (vs 3,669 — ~10% reduction)

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

xlog-prob::mc directly calls `Executor::execute_node`, `execute_recursive_scc`, and
`execute_non_recursive_scc` (mc.rs:1722, mc.rs:1783). By Wave 5, these live in
`executor/node_dispatch.rs` and `executor/recursive.rs` with stable signatures. The mc.rs
decomposition doesn't change which executor methods are called — only where within mc/ the
calls live.

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
- Add `#[non_exhaustive]` to public Config structs
- Add `///` doc comments explaining when/why to customize
- Do NOT create a hierarchical config tree or builder pattern

### 5c.3 Test Harness Consolidation (A5)

21 copies of `setup_provider()` across test files. Each crate keeps one canonical copy in a
`tests/common/mod.rs` or `src/test_support.rs` — less DRY but avoids new test-utility
dependency edges between crates. Comment points to pattern origin. Saves ~250 lines
(21 copies → ~5 copies).

### 5c.4 xlog-prob Export Cleanup (ST3)

Add top-level re-exports to `xlog-prob/src/lib.rs`:

```rust
// gpu_d4 is function-based — re-export the actual entry point functions
pub use compilation::gpu_d4::{compile_gpu_d4, compile_gpu_d4_with_config};
pub use exact::{ExactDdnnfProgram, ExactResult};
pub use mc::{McProgram, McEvalConfig, McResult};
// WFS: see 5c.5 for consolidation — re-export the primary free functions
```

Existing deep imports continue to work. These add convenience paths. The exact function
names to re-export should be verified against the post-5a split surface.

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

| Gate | Required |
|------|----------|
| Non-slow batch (v0.4.0-beta-release-design.md:52) | Yes |
| ILP reliability (v0.4.0-beta-release-design.md:53) | Yes |
| ILP sparse (v0.4.0-beta-release-design.md:54) | Yes |
| GA reliability (v0.4.0-beta-release-design.md:55) | Yes |
| ILP performance (v0.4.0-beta-release-design.md:56) | Yes |

### Supplemental build gates (not part of documented matrix, but required)

| Gate | Command | Required |
|------|---------|----------|
| Rust workspace | `cargo test --workspace --all-targets --exclude pyxlog --release` | Yes |
| CUDA certification | `cargo test -p xlog-cuda-tests --test certification_suite --release` | Yes (206/206) |
| pyxlog compile | `cargo check -p pyxlog` | Yes |
| Python wheel build | `maturin develop --release -m crates/pyxlog/Cargo.toml` | Yes |

Wave 5 is the final wave — it must pass the full documented matrix plus supplemental build
gates before the refactoring is considered complete.

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
