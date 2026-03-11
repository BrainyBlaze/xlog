# Align Negation Docs, CUDA Packaging, and Runtime Iteration Limits Design

**Goal:** Align documentation negation syntax with the parser, ensure CUDA PTX packaging is consistent with runtime loading, and honor runtime-configured SCC iteration limits.

**Architecture:**
- Documentation updates are limited to a single validation report plan file, replacing `\+` with `not` to match the existing grammar.
- CUDA packaging compiles `circuit` and `mc_sample` kernels in the crate build script so Cargo builds always produce the PTX that the runtime includes and loads.
- Runtime configuration becomes an explicit field on the executor, with `RuntimeConfig.max_iterations` driving recursive SCC fixpoint bounds.

**Tech Stack:** Rust (`xlog-runtime`, `xlog-cuda`), CUDA PTX build via `nvcc`, Markdown documentation.

## Scope

### In Scope
- Update `docs/plans/2026-01-11-full-system-validation-report.md` to use `not` instead of `\+`.
- Add `circuit` and `mc_sample` to `crates/xlog-cuda/build.rs` kernel list.
- Add `Executor::new_with_config` and store `RuntimeConfig` on the executor; use the configured `max_iterations` for recursive SCC evaluation.

### Out of Scope
- Grammar or parser changes for additional negation syntax.
- Changes to CUDA kernel implementations or runtime loading logic.
- Any behavior changes beyond respecting the configured iteration limit.

## Components and Data Flow

### Documentation
Replace the three `\+` negation occurrences in the validation report plan with `not` to match the current grammar and examples elsewhere in the spec.

### CUDA Packaging
Extend the `build.rs` kernel list to include `circuit` and `mc_sample`. This ensures `nvcc` builds `circuit.ptx` and `mc_sample.ptx` during Cargo builds, which are then embedded by `include_str!` in `crates/xlog-cuda/src/provider/mod.rs`. The runtime continues to load these modules and resolve entry points as before.

### Runtime Configuration
Introduce `Executor::new_with_config(provider, config)` storing the config in the executor. `Executor::new(provider)` delegates to `RuntimeConfig::default()` to preserve existing behavior. The recursive SCC loop uses `self.config.max_iterations` (cast to `usize`) and reports the configured limit in errors.

## Error Handling
- SCC iteration cap exceed continues to return `XlogError::Execution` (no partial results). Error messages include the configured limit.
- CUDA build failures remain explicit via `nvcc` error propagation.

## Testing
- Add a unit test in `xlog-runtime` that constructs a minimal recursive plan (or uses a tiny plan from existing IR helpers) and asserts that a small `max_iterations` triggers the expected error.
- Keep existing PTX embedding tests; no changes required beyond ensuring the PTX files build.
- Documentation change requires no tests.

## Rollout Notes
- No API breaks: existing callers can keep using `Executor::new`.
- CI/Cargo builds gain determinism by always compiling all PTX needed by the runtime.
