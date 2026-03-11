# Wave 1: Dependency Cleanup + Error/Type Seams

**Date**: 2026-03-10
**Status**: Approved
**Scope**: Seam creation only — no call-site migration, no file splits, no visibility sweep

## Overview

Wave 1 creates the foundational seams that Waves 2–5 consume. It removes false dependency
cycles, introduces canonical error conversion impls, adds error-context helpers, and defines
the `GpuScalar` type seam for later generic migration. All changes are additive or
subtractive at the manifest level — no existing function signatures change.

## Constraints

- Green at wave boundary: `cargo test --workspace --all-targets --exclude pyxlog --release` + 206/206 CUDA cert
- No pyxlog or Python gates (Wave 1 does not touch pyxlog)
- No call-site migration — old functions remain, new seams are available
- No file splits
- No visibility sweep
- Preserve zero data-plane D2H / GPU-residency contracts
- Preserve determinism/reproducibility

## 1. Dependency Graph Cleanup

### Removals

| Crate | Remove from `[dependencies]` | Reason |
|-------|------------------------------|--------|
| `xlog-logic/Cargo.toml` | `xlog-runtime` | Unused in production code; only test/example |
| `xlog-stats/Cargo.toml` | `xlog-cuda` | Zero production usage |

If either dependency is used in `#[cfg(test)]` blocks, `tests/` integration tests, or
`examples/`, move to `[dev-dependencies]` instead of removing entirely.

**Note**: `xlog-runtime/Cargo.toml` already keeps `xlog-logic` in `[dev-dependencies]` only —
no production entry exists there to remove.

### New Edge

| New edge | Justification |
|----------|--------------|
| `xlog-neural → xlog-core` | Enables `From<NeuralError> for XlogError` impl. xlog-core is a leaf crate. No cycle risk. |

**Result<T> alias collision (resolved)**: xlog-neural previously defined `pub type Result<T> = ...`
which collided with `xlog_core::Result<T>`. Wave 1 renamed it to `NeuralResult<T>` (src/lib.rs:75).

### Post-Cleanup Tier Map

```
Tier 0: xlog-core (leaf, no internal deps)

Tier 1: xlog-ir → core
        xlog-cuda → core
        xlog-stats → core
        xlog-neural → core (new edge)

Tier 2: xlog-runtime → core, ir, cuda, stats
        xlog-logic → core, ir, stats
        xlog-solve → core, cuda

Tier 3: xlog-gpu → core, cuda, ir, logic, runtime
        xlog-prob → core, cuda, logic, runtime, solve, ir

Tier 4: pyxlog → 8 crates (integration hub)
        xlog-cli → core, cuda, logic, gpu, prob
        xlog-cuda-tests → cuda, core, solve
```

## 2. Error Conversion Seams

### From Impls

Each `From` impl must live in a crate that defines either the source or target type
(orphan rule). `XlogError` is defined in xlog-core; all source error types are defined
in their respective crates.

| From impl | Crate | File | Orphan rule basis |
|-----------|-------|------|-------------------|
| `From<NeuralError> for XlogError` | xlog-neural | `src/lib.rs` | xlog-neural defines NeuralError |
| `From<TensorSourceError> for XlogError` | xlog-neural | `src/lib.rs` | xlog-neural defines TensorSourceError |
| `From<FunctionError> for XlogError` | xlog-logic | `src/function.rs` | xlog-logic defines FunctionError |
| `From<TypeError> for XlogError` | xlog-logic | `src/function.rs` | xlog-logic defines TypeError |
| `From<ModuleError> for XlogError` | xlog-logic | `src/module.rs` | xlog-logic defines ModuleError |

**DriverError (cudarc) → XlogError**: This `From` impl CANNOT live in xlog-cuda because
neither `DriverError` (cudarc) nor `XlogError` (xlog-core) is defined there — the orphan
rule forbids it. Instead, xlog-cuda uses a local helper function:

```rust
// In xlog-cuda/src/error_helpers.rs (new, ~10 lines)
pub(crate) fn driver_err(e: cudarc::driver::DriverError) -> XlogError {
    XlogError::Kernel(format!("CUDA driver error: {e}"))
}
```

This helper is adopted incrementally in Wave 2 as provider methods are relocated.

**Not in Wave 1**: `From<NeuralError> for PyErr` and `From<XlogError> for PyErr` — these
belong in Wave 4 (pyxlog FFI boundary). Due to the orphan rule, pyxlog cannot use `From`
for these; it will use local helper functions or a wrapper trait instead.

### Error Context Helpers

New methods on `XlogError` in `xlog-core/src/error.rs`:

```rust
impl XlogError {
    pub fn kernel_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Kernel(format!("{op}: {detail}: {source}"))
    }

    pub fn execution_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Execution(format!("{op}: {detail}: {source}"))
    }

    pub fn compilation_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Compilation(format!("{op}: {detail}: {source}"))
    }
}
```

These are available in Wave 1, adopted incrementally as ride-alongs in Waves 2–5.

## 3. GpuScalar Type Seam

New file: `crates/xlog-cuda/src/type_seam.rs` (~80 lines)

Defines an XLOG-internal marker trait for Rust scalar types that round-trip through GPU
column storage. Builds on `cudarc::driver::DeviceRepr` (already in broad use across
provider/, gpu_d4.rs, gpu_cdcl.rs) without replacing it.

```rust
/// Marker: a Rust scalar type that can round-trip through GPU column storage.
/// Requires cudarc::DeviceRepr + known byte width + little-endian serialization.
///
/// Post-Wave-2: generic download_column::<T>() and create_buffer_from_slice::<T>()
/// now replace the old type-specialized families. Trait is pub + sealed.
pub trait GpuScalar: sealed::Sealed + cudarc::driver::DeviceRepr + Copy + Send + 'static {
    const BYTE_WIDTH: usize;
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn to_le_bytes_into(self, buf: &mut [u8]);  // fixed-width, zero-allocation
}
```

Normal trait, normal impls (not `unsafe` — the unsafe contract is already carried by
`cudarc::driver::DeviceRepr`). Implementations for 8 types: u8, u32, u64, i32, i64, f32,
f64, bool.

**Bool encoding**: The bool impl must pin its encoding to match current behavior. Note that
D2H decoding (in `provider/transfer.rs`) treats any nonzero byte as `true` (not just `0x01`), while
H2D encoding should canonicalize to `0x00`/`0x01`. Document both the canonical write
encoding (`0x00`=false, `0x01`=true) and the lenient read semantics (`0x00`=false,
nonzero=true) in the trait impl to prevent future encoding drift.

**Visibility**: `pub` + sealed — external crates can name the bound (required by
`private_bounds` for turbofish calls like `download_column::<u32>()`) but cannot implement it.

## 4. Unit Tests

| Test | Location | Validates |
|------|----------|-----------|
| `test_neural_error_into_xlog` | `xlog-neural/tests/test_error_conversion.rs` | `From<NeuralError> for XlogError` preserves message |
| `test_tensor_source_error_into_xlog` | `xlog-neural/tests/test_error_conversion.rs` | `From<TensorSourceError> for XlogError` |
| `test_function_error_into_xlog` | `xlog-logic/src/function.rs` | `From<FunctionError> for XlogError` |
| `test_type_error_into_xlog` | `xlog-logic/src/function.rs` | `From<TypeError> for XlogError` |
| `test_module_error_into_xlog` | `xlog-logic/src/module.rs` | `From<ModuleError> for XlogError` |
| `test_driver_err_helper` | `xlog-cuda/src/error_helpers.rs` | `driver_err()` helper produces correct XlogError variant |
| `test_error_context_helpers` | `xlog-core/src/error.rs` | `kernel_ctx`, `execution_ctx`, `compilation_ctx` formatting |
| `test_gpu_scalar_roundtrip_{type}` | `xlog-cuda/src/type_seam.rs` | `GpuScalar` le-bytes roundtrip for each of 8 types |

**Note on xlog-neural test location**: xlog-neural disables library tests in Cargo.toml
(`test = false, doctest = false` for PyO3 compatibility). The error conversion tests must
live in a separate integration test file (`tests/test_error_conversion.rs`) which is not
affected by the `test = false` setting. This test file must NOT have `required-features`
that would exclude it from the default workspace test run — it must remain auto-discovered
by `cargo test --workspace`. Verify by checking that the existing xlog-neural Cargo.toml
`[[test]]` entries (e.g., Cargo.toml:33) use `required-features = ["python-tests"]` only
for PyO3-dependent tests, and ensure the new error conversion test does NOT follow that
pattern.

## 5. Gate

- `cargo test --workspace --all-targets --exclude pyxlog --release` — green
- `cargo test -p xlog-cuda-tests --test certification_suite --release` — 206/206
- No pyxlog or Python gates

## 6. Diff Profile (estimated)

| Change type | Files | Lines |
|-------------|-------|-------|
| Cargo.toml edits (remove deps + add xlog-core to xlog-neural) | 3 | ~6 |
| New `From` impls + CUDA helper | 2 `From` impls (neural, logic) + 1 helper fn (cuda) | ~60 |
| Error context helpers | 1 (xlog-core/error.rs) | ~25 |
| GpuScalar trait + impls | 1 (new: xlog-cuda/type_seam.rs) | ~80 |
| Unit tests | 5 files | ~120 |
| **Total** | ~10 files | ~290 lines |

## 7. What Wave 1 Does NOT Do

- No `download_column_u32` → `download_column::<u32>` rewrites (Wave 2)
- No `.map_err()` migrations (Waves 2–4, ride-along)
- No visibility changes (ride-along with each wave's hotspot)
- No file splits (Waves 2–4)
- No `From<NeuralError> for PyErr` (Wave 4)
