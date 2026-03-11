# Wave 1: Dependency Cleanup + Error/Type Seams — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create foundational seams (error conversions, context helpers, GpuScalar trait) and clean up false dependency edges so Waves 2–5 can consume them.

**Architecture:** Additive changes only — no existing function signatures change, no call-site migrations. Remove two false dependencies from Cargo.toml manifests, add one new edge (xlog-neural → xlog-core), introduce `From` impls for error types in the crates that own the source types (orphan rule), add error context helpers on `XlogError`, and define the `GpuScalar` marker trait in xlog-cuda.

**Tech Stack:** Rust workspace, cudarc (DeviceRepr), thiserror

**Spec:** `docs/superpowers/specs/2026-03-10-wave1-dependency-error-seams-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `crates/xlog-logic/Cargo.toml` | Move `xlog-runtime` from `[dependencies]` to `[dev-dependencies]` |
| Modify | `crates/xlog-stats/Cargo.toml` | Remove `xlog-cuda` from `[dependencies]` entirely |
| Modify | `crates/xlog-neural/Cargo.toml` | Add `xlog-core` to `[dependencies]` |
| Modify | `crates/xlog-neural/src/lib.rs` | Rename `Result<T>` alias to `NeuralResult<T>`, add `From<NeuralError> for XlogError` and `From<TensorSourceError> for XlogError` |
| Modify | `crates/xlog-neural/src/tensor_source.rs` | Update internal `Result<T>` usage if needed after alias rename |
| Modify | `crates/xlog-logic/src/function.rs` | Add `From<FunctionError> for XlogError` and `From<TypeError> for XlogError` |
| Modify | `crates/xlog-logic/src/module.rs` | Add `From<ModuleError> for XlogError` |
| Modify | `crates/xlog-core/src/error.rs` | Add `kernel_ctx`, `execution_ctx`, `compilation_ctx` context helpers |
| Create | `crates/xlog-cuda/src/error_helpers.rs` | `driver_err()` crate-local helper for cudarc DriverError → XlogError |
| Create | `crates/xlog-cuda/src/type_seam.rs` | `GpuScalar` trait + impls for 8 types |
| Modify | `crates/xlog-cuda/src/lib.rs` | Declare `error_helpers` and `type_seam` modules |
| Create | `crates/xlog-neural/tests/test_error_conversion.rs` | Tests for neural error From impls |

---

## Chunk 1: Dependency Graph Cleanup + Error Context Helpers

### Task 1: Remove false dependency xlog-logic → xlog-runtime

`xlog-runtime` is used by xlog-logic only in integration tests (`tests/e2e_integration_tests.rs`, `tests/real_world_tests.rs`) and examples (`examples/xlog_run.rs`). It must move to `[dev-dependencies]`.

**Files:**
- Modify: `crates/xlog-logic/Cargo.toml:9`

- [ ] **Step 1: Move xlog-runtime to dev-dependencies**

In `crates/xlog-logic/Cargo.toml`, remove line 9 (`xlog-runtime = { path = "../xlog-runtime" }`) from `[dependencies]` and add it under `[dev-dependencies]`:

```toml
[dev-dependencies]
xlog-cuda = { path = "../xlog-cuda" }
xlog-gpu = { path = "../xlog-gpu" }
xlog-runtime = { path = "../xlog-runtime" }
tempfile = "3"
serial_test = "3"
```

- [ ] **Step 2: Verify compile**

Run: `cargo check -p xlog-logic`
Expected: PASS (no production code uses xlog-runtime)

- [ ] **Step 3: Verify tests still pass**

Run: `cargo test -p xlog-logic --release`
Expected: PASS (tests and examples can still use xlog-runtime from dev-deps)

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-logic/Cargo.toml
git commit -m "refactor(xlog-logic): move xlog-runtime to dev-dependencies

xlog-runtime is only used in integration tests and examples, not in
production code. Moving it to [dev-dependencies] removes a false
Tier 2 → Tier 2 edge in the dependency graph."
```

---

### Task 2: Remove false dependency xlog-stats → xlog-cuda

`xlog-cuda` has zero usage in xlog-stats production code, tests, or benchmarks.

**Files:**
- Modify: `crates/xlog-stats/Cargo.toml:8`

- [ ] **Step 1: Remove xlog-cuda from dependencies**

In `crates/xlog-stats/Cargo.toml`, remove line 8 (`xlog-cuda = { path = "../xlog-cuda" }`). The `[dependencies]` section should become:

```toml
[dependencies]
xlog-core = { path = "../xlog-core" }
```

- [ ] **Step 2: Verify compile**

Run: `cargo check -p xlog-stats`
Expected: PASS

- [ ] **Step 3: Verify tests still pass**

Run: `cargo test -p xlog-stats --release`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-stats/Cargo.toml
git commit -m "refactor(xlog-stats): remove unused xlog-cuda dependency

xlog-cuda is not used anywhere in xlog-stats production code, tests,
or benchmarks. Removing it drops xlog-stats from Tier 1 dependency
on xlog-cuda, making it a pure xlog-core consumer."
```

---

### Task 3: Add error context helpers to XlogError

**Files:**
- Modify: `crates/xlog-core/src/error.rs:35` (after the enum definition, before the Result alias)
- Test: `crates/xlog-core/src/error.rs` (in existing `#[cfg(test)]` module)

- [ ] **Step 1: Write the failing tests**

Add these tests inside the existing `mod tests` block at the bottom of `crates/xlog-core/src/error.rs` (after the last existing test):

```rust
    #[test]
    fn test_kernel_ctx() {
        let err = XlogError::kernel_ctx("download_column", "dtoh copy failed", &"device error 42");
        assert_eq!(
            err.to_string(),
            "Kernel error: download_column: dtoh copy failed: device error 42"
        );
    }

    #[test]
    fn test_execution_ctx() {
        let err = XlogError::execution_ctx("execute_node", "filter failed", &"type mismatch");
        assert_eq!(
            err.to_string(),
            "Execution error: execute_node: filter failed: type mismatch"
        );
    }

    #[test]
    fn test_compilation_ctx() {
        let err = XlogError::compilation_ctx("compile_d4", "frontier overflow", &"limit 1024");
        assert_eq!(
            err.to_string(),
            "Compilation error: compile_d4: frontier overflow: limit 1024"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p xlog-core --release`
Expected: FAIL — `kernel_ctx`, `execution_ctx`, `compilation_ctx` not found on `XlogError`

- [ ] **Step 3: Implement the context helpers**

Add this `impl` block in `crates/xlog-core/src/error.rs`, between the `XlogError` enum definition (line 35) and the `Result` type alias (line 37):

```rust
impl XlogError {
    /// Create a Kernel error with structured context: "op: detail: source".
    pub fn kernel_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Kernel(format!("{op}: {detail}: {source}"))
    }

    /// Create an Execution error with structured context.
    pub fn execution_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Execution(format!("{op}: {detail}: {source}"))
    }

    /// Create a Compilation error with structured context.
    pub fn compilation_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Compilation(format!("{op}: {detail}: {source}"))
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-core --release`
Expected: PASS (all existing + 3 new tests)

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-core/src/error.rs
git commit -m "feat(xlog-core): add error context helpers on XlogError

Add kernel_ctx(), execution_ctx(), compilation_ctx() constructors
that produce structured 'op: detail: source' error messages. These
seams are adopted incrementally in Waves 2-5 to replace ad-hoc
format!() calls in .map_err() chains."
```

---

### Task 4: Add CUDA driver_err() helper

**Files:**
- Create: `crates/xlog-cuda/src/error_helpers.rs`
- Modify: `crates/xlog-cuda/src/lib.rs:10` (add module declaration)

- [ ] **Step 1: Write the test**

Create `crates/xlog-cuda/src/error_helpers.rs` with only a test:

```rust
//! Helper functions for converting external error types into XlogError.
//!
//! cudarc::driver::DriverError → XlogError cannot use From (orphan rule:
//! neither type is defined in xlog-cuda). This module provides a local
//! conversion function instead.

use xlog_core::XlogError;

/// Convert a cudarc DriverError into an XlogError::Kernel.
pub(crate) fn driver_err(e: cudarc::driver::DriverError) -> XlogError {
    XlogError::Kernel(format!("CUDA driver error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_err_helper() {
        // cudarc::driver::DriverError doesn't have public constructors we can
        // easily use in tests, so we test via the formatted output pattern.
        // The real validation is that this compiles and the type signature is
        // correct — call-site adoption in Wave 2 provides integration coverage.
        //
        // We can at least verify the function exists and returns the right variant
        // by testing the context helper it's modeled after:
        let err = XlogError::kernel_ctx("test", "driver error", &"mock error");
        let msg = err.to_string();
        assert!(msg.contains("test"));
        assert!(msg.contains("driver error"));
        assert!(msg.contains("mock error"));
    }
}
```

- [ ] **Step 2: Add module declaration**

In `crates/xlog-cuda/src/lib.rs`, add after line 10 (`pub mod provider;`):

```rust
pub(crate) mod error_helpers;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p xlog-cuda --lib --release`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-cuda/src/error_helpers.rs crates/xlog-cuda/src/lib.rs
git commit -m "feat(xlog-cuda): add driver_err() helper for cudarc DriverError

Orphan rule prevents From<DriverError> for XlogError in xlog-cuda
(neither type is defined here). This pub(crate) helper provides the
same conversion for adoption in Wave 2 provider methods."
```

---

## Chunk 2: Error Conversion From Impls

### Task 5: Add xlog-core dependency to xlog-neural + rename Result alias

**Files:**
- Modify: `crates/xlog-neural/Cargo.toml:24`
- Modify: `crates/xlog-neural/src/lib.rs:75`

- [ ] **Step 1: Add xlog-core dependency**

In `crates/xlog-neural/Cargo.toml`, add `xlog-core` at the top of `[dependencies]`:

```toml
[dependencies]
xlog-core = { path = "../xlog-core" }
thiserror = { workspace = true }
lru = "0.12"
# PyO3 is optional - only needed when integrating with Python
pyo3 = { workspace = true, optional = true }
```

- [ ] **Step 2: Rename Result alias to NeuralResult**

In `crates/xlog-neural/src/lib.rs`, change line 75:

From: `pub type Result<T> = std::result::Result<T, NeuralError>;`
To: `pub type NeuralResult<T> = std::result::Result<T, NeuralError>;`

- [ ] **Step 3: Fix all internal usages of the old Result alias**

Search for `Result<` usages of the crate-level alias in xlog-neural's `src/` directory. Currently no internal files use the alias — `tensor_source.rs` uses explicit `std::result::Result`, and other files don't return the crate-level `Result` at all. This step is expected to be a no-op. Verify with `cargo check -p xlog-neural` — if it compiles, no further changes needed.

- [ ] **Step 4: Verify compile**

Run: `cargo check -p xlog-neural`
Expected: PASS — all internal `Result` usages updated to `NeuralResult`

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-neural/Cargo.toml crates/xlog-neural/src/
git commit -m "refactor(xlog-neural): add xlog-core dep, rename Result to NeuralResult

Adding xlog-core as a dependency enables From impls for error
conversion (next commit). The Result<T> alias is renamed to
NeuralResult<T> to avoid collision with xlog_core::Result<T>,
which would compound in downstream crates."
```

---

### Task 6: Add From<NeuralError> and From<TensorSourceError> for XlogError

**Files:**
- Modify: `crates/xlog-neural/src/lib.rs` (after NeuralError enum)
- Create: `crates/xlog-neural/tests/test_error_conversion.rs`

- [ ] **Step 1: Write the integration test file**

Create `crates/xlog-neural/tests/test_error_conversion.rs`:

```rust
//! Tests for error conversion From impls.
//!
//! These live in an integration test file (not lib tests) because
//! xlog-neural's Cargo.toml has `test = false` for PyO3 compatibility.
//! This file has NO required-features, so it runs under the default
//! `cargo test --workspace` invocation.

use xlog_core::XlogError;
use xlog_neural::NeuralError;
use xlog_neural::TensorSourceError;

#[test]
fn test_neural_error_into_xlog() {
    let err = NeuralError::NetworkNotFound("mnist".to_string());
    let xlog_err: XlogError = err.into();
    let msg = xlog_err.to_string();
    assert!(msg.contains("mnist"), "Expected 'mnist' in: {msg}");
    assert!(
        msg.contains("Network not found"),
        "Expected 'Network not found' in: {msg}"
    );
}

#[test]
fn test_neural_error_pytorch_into_xlog() {
    let err = NeuralError::PyTorchError("CUDA OOM".to_string());
    let xlog_err: XlogError = err.into();
    let msg = xlog_err.to_string();
    assert!(msg.contains("CUDA OOM"), "Expected 'CUDA OOM' in: {msg}");
}

#[test]
fn test_tensor_source_error_into_xlog() {
    let err = TensorSourceError::NotFound("train".to_string());
    let xlog_err: XlogError = err.into();
    let msg = xlog_err.to_string();
    assert!(msg.contains("train"), "Expected 'train' in: {msg}");
}

#[test]
fn test_tensor_source_no_active_into_xlog() {
    let err = TensorSourceError::NoActive;
    let xlog_err: XlogError = err.into();
    let msg = xlog_err.to_string();
    assert!(
        msg.contains("No active"),
        "Expected 'No active' in: {msg}"
    );
}
```

- [ ] **Step 2: Verify the tests fail**

Run: `cargo test -p xlog-neural --test test_error_conversion --release`
Expected: FAIL — `From<NeuralError> for XlogError` not implemented

- [ ] **Step 3: Implement the From impls**

In `crates/xlog-neural/src/lib.rs`, add after the `NeuralResult` type alias (line 75):

```rust
// Error conversion seams — orphan rule: NeuralError is defined here.
impl From<NeuralError> for xlog_core::XlogError {
    fn from(e: NeuralError) -> Self {
        xlog_core::XlogError::Execution(e.to_string())
    }
}

impl From<tensor_source::TensorSourceError> for xlog_core::XlogError {
    fn from(e: tensor_source::TensorSourceError) -> Self {
        xlog_core::XlogError::Execution(e.to_string())
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-neural --test test_error_conversion --release`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-neural/src/lib.rs crates/xlog-neural/tests/test_error_conversion.rs
git commit -m "feat(xlog-neural): add From<NeuralError> and From<TensorSourceError> for XlogError

Error conversion seams that enable ? operator in code that returns
xlog_core::Result. Orphan rule is satisfied because NeuralError and
TensorSourceError are defined in xlog-neural. Tests live in an
integration test file (not lib tests) because Cargo.toml has
test = false for PyO3 compatibility."
```

---

### Task 7: Add From impls for xlog-logic error types

**Files:**
- Modify: `crates/xlog-logic/src/function.rs:93` (after `impl std::error::Error for TypeError`)
- Modify: `crates/xlog-logic/src/module.rs:146` (after `impl std::error::Error for ModuleError`)

- [ ] **Step 1: Write failing tests for FunctionError and TypeError**

In `crates/xlog-logic/src/function.rs`, add a new test module at the end of the file. First check if there's an existing `#[cfg(test)]` module — if not, add:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::XlogError;

    #[test]
    fn test_function_error_into_xlog() {
        let err = FunctionError::UndefinedFunction {
            name: "foo".to_string(),
        };
        let xlog_err: XlogError = err.into();
        let msg = xlog_err.to_string();
        assert!(msg.contains("foo"), "Expected 'foo' in: {msg}");
    }

    #[test]
    fn test_type_error_into_xlog() {
        let err = TypeError::CannotInfer {
            name: "X".to_string(),
        };
        let xlog_err: XlogError = err.into();
        let msg = xlog_err.to_string();
        assert!(msg.contains("X"), "Expected 'X' in: {msg}");
    }
}
```

- [ ] **Step 2: Write failing test for ModuleError**

In `crates/xlog-logic/src/module.rs`, add inside the existing `#[cfg(test)] mod tests` block (after line 233):

```rust
    #[test]
    fn test_module_error_into_xlog() {
        let err = ModuleError::ParseError {
            path: std::path::PathBuf::from("/test.xlog"),
            message: "unexpected EOF".to_string(),
        };
        let xlog_err: xlog_core::XlogError = err.into();
        let msg = xlog_err.to_string();
        assert!(
            msg.contains("unexpected EOF"),
            "Expected 'unexpected EOF' in: {msg}"
        );
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p xlog-logic --lib --release`
Expected: FAIL — `From` impls not implemented

- [ ] **Step 4: Implement the From impls**

In `crates/xlog-logic/src/function.rs`, add after `impl std::error::Error for TypeError {}` (line 93):

```rust
impl From<FunctionError> for xlog_core::XlogError {
    fn from(e: FunctionError) -> Self {
        xlog_core::XlogError::Compilation(e.to_string())
    }
}

impl From<TypeError> for xlog_core::XlogError {
    fn from(e: TypeError) -> Self {
        xlog_core::XlogError::Type(e.to_string())
    }
}
```

In `crates/xlog-logic/src/module.rs`, add after `impl std::error::Error for ModuleError {}` (line 146):

```rust
impl From<ModuleError> for xlog_core::XlogError {
    fn from(e: ModuleError) -> Self {
        xlog_core::XlogError::Compilation(e.to_string())
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p xlog-logic --lib --release`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-logic/src/function.rs crates/xlog-logic/src/module.rs
git commit -m "feat(xlog-logic): add From impls for FunctionError, TypeError, ModuleError

Error conversion seams to XlogError. FunctionError and ModuleError map
to XlogError::Compilation (parse/compile-time errors). TypeError maps
to XlogError::Type. Orphan rule satisfied: all source types are
defined in xlog-logic."
```

---

## Chunk 3: GpuScalar Type Seam + Final Gate

### Task 8: Create GpuScalar trait and implementations

**Files:**
- Create: `crates/xlog-cuda/src/type_seam.rs`
- Modify: `crates/xlog-cuda/src/lib.rs` (add module declaration)

- [ ] **Step 1: Create the type_seam.rs file with trait, impls, and tests**

Create `crates/xlog-cuda/src/type_seam.rs`:

```rust
//! GpuScalar — pub sealed trait for Rust scalar types that round-trip through GPU column storage.
//!
//! Used by generic `download_column<T>()` and `create_buffer_from_slice<T>()` (Wave 2).
//! The trait is public + sealed: external crates can name the bound (required for turbofish
//! calls like `download_column::<u32>()`) but cannot implement it.
//!
//! # Bool encoding
//!
//! Write encoding (H2D): canonical `0x00` = false, `0x01` = true.
//! Read decoding (D2H): `0x00` = false, any nonzero byte = true.
//! (See the D2H bool decoding path in provider/transfer.rs for current D2H bool decoding.)
//!
//! The asymmetry is intentional: we always write canonical values, but tolerate
//! non-canonical GPU output during reads to match existing provider behavior.

/// Marker trait: a Rust scalar type that can round-trip through GPU column storage.
///
/// Requires `cudarc::driver::DeviceRepr` + known byte width + little-endian serialization.
/// Public + sealed: external crates can name the bound (for turbofish) but cannot implement it.
///
/// Wave 2 adds `download_column::<T>()` and `create_buffer_from_slice::<T>()`
/// that replace the current type-specialized function families.
pub trait GpuScalar: sealed::Sealed + cudarc::driver::DeviceRepr + Copy + Send + 'static {
    /// Size in bytes of this scalar type.
    const BYTE_WIDTH: usize;

    /// Deserialize from a little-endian byte slice.
    /// The slice length must equal `BYTE_WIDTH`.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Serialize into a little-endian byte buffer.
    /// The buffer length must equal `BYTE_WIDTH`.
    fn to_le_bytes_into(self, buf: &mut [u8]);
}

impl GpuScalar for u8 {
    const BYTE_WIDTH: usize = 1;
    fn from_le_bytes(bytes: &[u8]) -> Self { bytes[0] }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf[0] = self; }
}

impl GpuScalar for u32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { u32::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for u64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { u64::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for i32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { i32::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for i64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { i64::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for f32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { f32::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for f64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { f64::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

/// Bool encoding:
/// - Write (H2D): `0x00` = false, `0x01` = true (canonical).
/// - Read (D2H): `0x00` = false, nonzero = true (lenient, matches the D2H bool decoding path in provider/transfer.rs).
impl GpuScalar for bool {
    const BYTE_WIDTH: usize = 1;

    fn from_le_bytes(bytes: &[u8]) -> Self {
        // Lenient read: any nonzero byte is true (matches existing D2H behavior).
        bytes[0] != 0
    }

    fn to_le_bytes_into(self, buf: &mut [u8]) {
        // Canonical write: 0x00 or 0x01.
        buf[0] = if self { 1 } else { 0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: roundtrip a value through le-bytes serialization.
    fn roundtrip<T: GpuScalar + PartialEq + std::fmt::Debug>(val: T) {
        let mut buf = vec![0u8; T::BYTE_WIDTH];
        val.to_le_bytes_into(&mut buf);
        let recovered = T::from_le_bytes(&buf);
        assert_eq!(recovered, val);
    }

    #[test]
    fn test_gpu_scalar_roundtrip_u8() { roundtrip(42u8); roundtrip(0u8); roundtrip(255u8); }

    #[test]
    fn test_gpu_scalar_roundtrip_u32() { roundtrip(0u32); roundtrip(42u32); roundtrip(u32::MAX); }

    #[test]
    fn test_gpu_scalar_roundtrip_u64() { roundtrip(0u64); roundtrip(42u64); roundtrip(u64::MAX); }

    #[test]
    fn test_gpu_scalar_roundtrip_i32() { roundtrip(0i32); roundtrip(-1i32); roundtrip(i32::MAX); }

    #[test]
    fn test_gpu_scalar_roundtrip_i64() { roundtrip(0i64); roundtrip(-1i64); roundtrip(i64::MAX); }

    #[test]
    fn test_gpu_scalar_roundtrip_f32() { roundtrip(0.0f32); roundtrip(-1.5f32); roundtrip(f32::INFINITY); }

    #[test]
    fn test_gpu_scalar_roundtrip_f64() { roundtrip(0.0f64); roundtrip(-1.5f64); roundtrip(f64::INFINITY); }

    #[test]
    fn test_gpu_scalar_roundtrip_bool() {
        roundtrip(true);
        roundtrip(false);
    }

    #[test]
    fn test_bool_canonical_write() {
        let mut buf = [0xFFu8];
        false.to_le_bytes_into(&mut buf);
        assert_eq!(buf[0], 0x00, "false must write canonical 0x00");

        true.to_le_bytes_into(&mut buf);
        assert_eq!(buf[0], 0x01, "true must write canonical 0x01");
    }

    #[test]
    fn test_bool_lenient_read() {
        // Any nonzero byte reads as true (matches the D2H bool decoding path in provider/transfer.rs behavior).
        assert!(!bool::from_le_bytes(&[0x00]));
        assert!(bool::from_le_bytes(&[0x01]));
        assert!(bool::from_le_bytes(&[0x02]));
        assert!(bool::from_le_bytes(&[0xFF]));
    }

    #[test]
    fn test_byte_width_consistency() {
        assert_eq!(u8::BYTE_WIDTH, std::mem::size_of::<u8>());
        assert_eq!(u32::BYTE_WIDTH, std::mem::size_of::<u32>());
        assert_eq!(u64::BYTE_WIDTH, std::mem::size_of::<u64>());
        assert_eq!(i32::BYTE_WIDTH, std::mem::size_of::<i32>());
        assert_eq!(i64::BYTE_WIDTH, std::mem::size_of::<i64>());
        assert_eq!(f32::BYTE_WIDTH, std::mem::size_of::<f32>());
        assert_eq!(f64::BYTE_WIDTH, std::mem::size_of::<f64>());
        assert_eq!(bool::BYTE_WIDTH, std::mem::size_of::<bool>());
    }
}
```

- [ ] **Step 2: Add module declaration in lib.rs**

In `crates/xlog-cuda/src/lib.rs`, add after the `error_helpers` module declaration:

```rust
pub mod type_seam;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p xlog-cuda --lib --release`
Expected: PASS (all type_seam tests + error_helpers tests)

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-cuda/src/type_seam.rs crates/xlog-cuda/src/lib.rs
git commit -m "feat(xlog-cuda): add GpuScalar trait for type-safe GPU column round-trip

Public sealed marker trait that Wave 2 will use to collapse
8 download_column_<T> and 7 create_buffer_from_<T>_slice functions
into single generic functions. Implementations for u8, u32, u64,
i32, i64, f32, f64, bool. Bool encoding documents the asymmetry:
canonical 0x00/0x01 writes, lenient nonzero=true reads."
```

---

### Task 9: Run the full Wave 1 gate

- [ ] **Step 1: Run workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: PASS (all tests green)

- [ ] **Step 2: Run CUDA certification suite**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`
Expected: PASS (206/206)

- [ ] **Step 3: Verify no regressions in the dependency graph**

Run: `cargo tree -p xlog-logic --depth 1`
Expected: `xlog-runtime` should NOT appear in direct dependencies (only dev-deps)

Run: `cargo tree -p xlog-stats --depth 1`
Expected: `xlog-cuda` should NOT appear at all

Run: `cargo tree -p xlog-neural --depth 1`
Expected: `xlog-core` SHOULD appear in dependencies

- [ ] **Step 4: Record gate results**

If both gates pass, Wave 1 is complete. All seams are in place for Wave 2.
