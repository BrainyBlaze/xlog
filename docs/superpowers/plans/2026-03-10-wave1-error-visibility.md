# Wave 1: Error Unification & Visibility Lockdown

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the workspace error system with `From` bridges and remove accidental public API surface before structural module splits in Wave 2.

**Architecture:** Two strategies based on the dependency graph:
- **xlog-neural** errors (cycle-free): Add `NeuralError`/`TensorSourceError` as typed variants in `XlogError` with `#[from]`, since `xlog-neural` has zero xlog dependencies.
- **xlog-logic** errors (would create cycle): Add `Module(String)`/`Function(String)`/`LogicType(String)` string-wrapping variants in `XlogError`. Implement `From<ModuleError/FunctionError/TypeError> for XlogError` in `xlog-logic` using `.to_string()`. This avoids the `xlog-core → xlog-logic` cycle while still enabling `?` propagation.

Remove the `Result<T>` alias collision in `xlog-neural`. Audit and restrict visibility across all crates.

**Tech Stack:** Rust (thiserror), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-10-codebase-refactoring-design.md` (C1, C2)

---

## Chunk 1: Error System Unification (C1)

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `crates/xlog-core/src/error.rs` | Add 5 new `XlogError` string-wrapping variants |
| Modify | `crates/xlog-core/Cargo.toml` | Add dependency on `xlog-neural` only (cycle-free) |
| Modify | `crates/xlog-logic/src/module.rs` | Add `impl From<ModuleError> for XlogError` |
| Modify | `crates/xlog-logic/src/function.rs` | Add `impl From<FunctionError/TypeError> for XlogError` |
| Modify | `crates/xlog-neural/src/lib.rs:75` | Remove `Result<T>` alias |
| Create | `crates/xlog-core/tests/test_error_conversions.rs` | Verify From impls compile and convert correctly |

### Task 1: Add XlogError variants and From impls for neural errors (cycle-free)

`xlog-neural` has zero xlog dependencies, so `xlog-core` can safely depend on it.

**Files:**
- Modify: `crates/xlog-core/src/error.rs:7-35`
- Modify: `crates/xlog-core/Cargo.toml`
- Create: `crates/xlog-core/tests/test_error_conversions.rs` (mkdir -p the tests/ dir first)

**Context:** The neural error types are:
- `xlog-neural::NeuralError` (6 variants, thiserror) — `crates/xlog-neural/src/lib.rs:48-72`
- `xlog-neural::TensorSourceError` (3 variants, thiserror) — `crates/xlog-neural/src/tensor_source.rs:47-59`

- [ ] **Step 1: Write the failing test**

```bash
mkdir -p crates/xlog-core/tests
```

Create `crates/xlog-core/tests/test_error_conversions.rs`:

```rust
use xlog_core::XlogError;
use xlog_neural::{NeuralError, TensorSourceError};

#[test]
fn neural_error_converts_to_xlog_error() {
    let err = NeuralError::NetworkNotFound("test".into());
    let xlog_err: XlogError = err.into();
    assert!(matches!(xlog_err, XlogError::Neural(_)));
}

#[test]
fn tensor_source_error_converts_to_xlog_error() {
    let err = TensorSourceError::NoActive;
    let xlog_err: XlogError = err.into();
    assert!(matches!(xlog_err, XlogError::TensorSource(_)));
}

#[test]
fn question_mark_propagation_neural() {
    fn inner() -> Result<(), XlogError> {
        let result: Result<(), NeuralError> = Err(NeuralError::InvalidConfig("bad".into()));
        result?; // This should compile with From impl
        Ok(())
    }
    assert!(inner().is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-core --test test_error_conversions --release 2>&1 | head -30`

Expected: Compilation failure — `XlogError::Neural` variant doesn't exist.

- [ ] **Step 3: Add xlog-neural dependency to xlog-core**

In `crates/xlog-core/Cargo.toml`, add under `[dependencies]`:

```toml
xlog-neural = { path = "../xlog-neural" }
```

**Do NOT add xlog-logic** — it depends on xlog-core, so adding it here would create a cycle.

- [ ] **Step 4: Add Neural and TensorSource variants with #[from] to XlogError**

In `crates/xlog-core/src/error.rs`, add to the enum:

```rust
use xlog_neural::{NeuralError, TensorSourceError};

#[derive(Debug, thiserror::Error)]
pub enum XlogError {
    // ... existing 8 variants unchanged ...

    #[error("Neural error: {0}")]
    Neural(#[from] NeuralError),

    #[error("Tensor source error: {0}")]
    TensorSource(#[from] TensorSourceError),
}
```

The `#[from]` attribute auto-generates the `From` impl.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p xlog-core --test test_error_conversions --release`

Expected: 3/3 PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-core/src/error.rs crates/xlog-core/Cargo.toml crates/xlog-core/tests/test_error_conversions.rs
git commit -m "feat(core): add From<NeuralError/TensorSourceError> for XlogError"
```

### Task 1b: Add string-wrapping variants and From impls for logic errors (cycle-safe)

`xlog-logic` depends on `xlog-core`, so we cannot add `xlog-logic` as a dependency of `xlog-core`. Instead:
1. Add `Module(String)`, `Function(String)`, `LogicType(String)` variants to `XlogError` in `xlog-core` (string-wrapping, no dependency needed).
2. Implement `From<ModuleError/FunctionError/TypeError> for XlogError` in `xlog-logic` (where those types are local).

This loses structured error info but enables `?` propagation, which is the primary goal.

**Files:**
- Modify: `crates/xlog-core/src/error.rs`
- Modify: `crates/xlog-logic/src/module.rs:146+`
- Modify: `crates/xlog-logic/src/function.rs:60+, 93+`

**Prerequisites:** `ModuleError`, `FunctionError`, and `TypeError` must implement `std::error::Error`. Verify:
- `crates/xlog-logic/src/function.rs:60` — `impl std::error::Error for FunctionError {}`
- `crates/xlog-logic/src/function.rs:93` — `impl std::error::Error for TypeError {}`
- `crates/xlog-logic/src/module.rs:146` — `impl std::error::Error for ModuleError {}`

- [ ] **Step 1: Add string-wrapping variants to XlogError**

In `crates/xlog-core/src/error.rs`, add to the enum (NO dependency on xlog-logic needed):

```rust
#[derive(Debug, thiserror::Error)]
pub enum XlogError {
    // ... existing variants + Neural + TensorSource from Task 1 ...

    #[error("Module error: {0}")]
    Module(String),

    #[error("Function error: {0}")]
    Function(String),

    #[error("Logic type error: {0}")]
    LogicType(String),
}
```

**NOTE:** The existing `Type(String)` variant is for string-based type errors from other sources. `LogicType(String)` wraps the structured `TypeError` via `.to_string()`. Keep both.

- [ ] **Step 2: Add From impls in xlog-logic**

In `crates/xlog-logic/src/module.rs`, after the existing `impl std::error::Error for ModuleError {}`:

```rust
impl From<ModuleError> for xlog_core::XlogError {
    fn from(e: ModuleError) -> Self {
        xlog_core::XlogError::Module(e.to_string())
    }
}
```

In `crates/xlog-logic/src/function.rs`, after `impl std::error::Error for FunctionError {}`:

```rust
impl From<FunctionError> for xlog_core::XlogError {
    fn from(e: FunctionError) -> Self {
        xlog_core::XlogError::Function(e.to_string())
    }
}
```

After `impl std::error::Error for TypeError {}`:

```rust
impl From<TypeError> for xlog_core::XlogError {
    fn from(e: TypeError) -> Self {
        xlog_core::XlogError::LogicType(e.to_string())
    }
}
```

- [ ] **Step 3: Write tests for logic error conversion**

Add to `crates/xlog-core/tests/test_error_conversions.rs` (the test crate can depend on both xlog-core and xlog-logic since it's a test binary, not a library):

```rust
use xlog_logic::module::ModuleError;
use xlog_logic::function::{FunctionError, TypeError};

#[test]
fn module_error_converts_to_xlog_error() {
    let err = ModuleError::CircularImport { cycle: vec![] };
    let xlog_err: XlogError = err.into();
    assert!(matches!(xlog_err, XlogError::Module(_)));
}

#[test]
fn function_error_converts_to_xlog_error() {
    let err = FunctionError::UndefinedFunction { name: "test".into() };
    let xlog_err: XlogError = err.into();
    assert!(matches!(xlog_err, XlogError::Function(_)));
}

#[test]
fn type_error_converts_to_xlog_error() {
    let err = TypeError::CannotInfer { name: "X".into() };
    let xlog_err: XlogError = err.into();
    assert!(matches!(xlog_err, XlogError::LogicType(_)));
}

#[test]
fn question_mark_propagation_logic() {
    fn inner() -> Result<(), XlogError> {
        let result: Result<(), ModuleError> = Err(ModuleError::CircularImport { cycle: vec![] });
        result?;
        Ok(())
    }
    assert!(inner().is_err());
}
```

Note: `xlog-core/Cargo.toml` needs `xlog-logic` as a `[dev-dependencies]` for the test binary:

```toml
[dev-dependencies]
xlog-logic = { path = "../xlog-logic" }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-core --test test_error_conversions --release`

Expected: 7/7 PASS.

- [ ] **Step 5: Run full workspace test**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-core/src/error.rs crates/xlog-core/Cargo.toml crates/xlog-logic/src/module.rs crates/xlog-logic/src/function.rs crates/xlog-core/tests/test_error_conversions.rs
git commit -m "feat(core,logic): add From impls bridging logic error types to XlogError (string-wrapping to avoid cycle)"
```

### Task 2: Remove the `Result<T>` alias collision

**Files:**
- Modify: `crates/xlog-neural/src/lib.rs:75`
- Modify: All files in `xlog-neural` that use the local `Result<T>`

**Context:** `xlog-neural/src/lib.rs:75` defines `pub type Result<T> = std::result::Result<T, NeuralError>;`. This collides with `xlog_core::Result<T>`. Any crate importing both gets ambiguity.

- [ ] **Step 1: Find all usages of the neural Result alias**

Run: `grep -rn 'Result<' crates/xlog-neural/src/ | grep -v 'std::result'`

This shows every function signature using the unqualified `Result<T>` within xlog-neural. Each must be changed to `std::result::Result<T, NeuralError>` or `Result<T, NeuralError>`.

- [ ] **Step 2: Remove the alias and fix all usages**

In `crates/xlog-neural/src/lib.rs:75`, delete:
```rust
pub type Result<T> = std::result::Result<T, NeuralError>;
```

Then in every file that used the alias, replace `Result<T>` with `Result<T, NeuralError>`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p xlog-neural --release`

Expected: Clean compilation.

- [ ] **Step 4: Run workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20`

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-neural/
git commit -m "fix(neural): remove Result<T> alias that collides with xlog-core"
```

### Task 3: Convert high-value `.map_err()` sites to `?`

**Context:** There are 1,094 `.map_err()` calls across the workspace. The vast majority convert `cudarc::DriverError` to strings — those are NOT affected by our new `From` impls. Focus only on sites that convert between xlog error types.

**Realistic scope:** The neural `From` impls (typed, `#[from]`) enable direct `?` on any `Result<_, NeuralError>` or `Result<_, TensorSourceError>`. The logic `From` impls (string-wrapping) also enable `?` but lose the structured error. Estimate: 5–15 convertible sites (most `.map_err()` calls are cudarc-to-string, not xlog-to-xlog).

**Files:**
- Search: All crates for `.map_err()` calls involving `NeuralError`, `ModuleError`, `FunctionError`, `TypeError`, `TensorSourceError`

- [ ] **Step 1: Identify convertible .map_err() sites**

Run these searches:
```bash
grep -rn 'map_err.*NeuralError\|map_err.*ModuleError\|map_err.*FunctionError\|map_err.*TypeError\|map_err.*TensorSourceError' crates/
```

Only convert sites where the source error type exactly matches one of our 5 `From` targets.

**Do NOT convert** sites that format error messages with additional context (e.g., `format!("Failed to load network {}: {}", name, e)`) — that context is valuable for debugging.

**For string-wrapped logic errors:** Converting `map_err(|e| XlogError::Execution(format!("Module: {}", e)))` to bare `?` will now produce `XlogError::Module(e.to_string())` instead. The error message changes from "Module: ..." to whatever `ModuleError::fmt()` produces. Verify the resulting message is still useful.

- [ ] **Step 2: Convert identified sites**

For each site, change:
```rust
// Before
something.map_err(|e| XlogError::Execution(format!("Neural: {}", e)))?;

// After (only if error type is NeuralError)
something?;
```

- [ ] **Step 3: Run workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20`

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "refactor: replace .map_err() with ? where From impls now handle conversion"
```

---

## Chunk 2: Dependency Cleanup & Visibility Lockdown (C2 + H4)

### Task 4: Remove unnecessary Cargo dependencies (H4)

Move this first — it's zero-risk and cleans the dependency graph before visibility audit.

**Files:**
- Modify: `crates/xlog-stats/Cargo.toml` — remove `xlog-cuda` from `[dependencies]`
- Modify: `crates/xlog-logic/Cargo.toml` — remove `xlog-runtime` from `[dependencies]`

**Note:** `xlog-logic` also has `xlog-cuda` and `xlog-gpu` in `[dev-dependencies]` — leave those alone (they're for tests).

- [ ] **Step 1: Verify zero usage**

```bash
grep -r 'xlog_cuda' crates/xlog-stats/src/
grep -r 'xlog_runtime' crates/xlog-logic/src/
```

Expected: No matches (confirming zero usage in library code).

- [ ] **Step 2: Remove the dependencies**

In `crates/xlog-stats/Cargo.toml`, remove the `xlog-cuda` line from `[dependencies]`.
In `crates/xlog-logic/Cargo.toml`, remove the `xlog-runtime` line from `[dependencies]`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check --workspace --release`

Expected: Clean compilation.

- [ ] **Step 4: Commit**

```bash
git add crates/xlog-stats/Cargo.toml crates/xlog-logic/Cargo.toml
git commit -m "chore: remove unused xlog-cuda and xlog-runtime dependencies"
```

### Task 5: Audit and restrict visibility in xlog-core

**Files:**
- Modify: `crates/xlog-core/src/*.rs`

**Approach:** For each `pub` item, check if it's used outside the crate. If not, change to `pub(crate)`. Start with the smallest crate to establish the pattern.

- [ ] **Step 1: List all pub items in xlog-core**

Run: `grep -n '^pub ' crates/xlog-core/src/*.rs`

- [ ] **Step 2: For each pub item, check external usage**

For each `pub struct`, `pub fn`, `pub enum`, `pub type`:
```bash
grep -r 'ItemName' crates/ --include='*.rs' | grep -v 'xlog-core'
```

If no external usage → change to `pub(crate)`.

If used only via re-export from `lib.rs` → keep `pub` on the item but consider if the re-export is needed.

- [ ] **Step 3: Apply visibility changes**

Change identified internal-only items from `pub` to `pub(crate)`.

- [ ] **Step 4: Verify compilation**

Run: `cargo check --workspace --release`

Expected: Clean. If anything breaks, an external crate was using the item — revert that specific change.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-core/
git commit -m "refactor(core): restrict internal types to pub(crate)"
```

### Task 6: Audit and restrict visibility in xlog-neural

Same process as Task 4 for `crates/xlog-neural/src/*.rs`.

- [ ] **Step 1–4:** Same as Task 4 but for xlog-neural.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-neural/
git commit -m "refactor(neural): restrict internal types to pub(crate)"
```

### Task 7: Audit and restrict visibility in xlog-logic

Same process for `crates/xlog-logic/src/*.rs`.

- [ ] **Step 1–4:** Same as Task 4 but for xlog-logic.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-logic/
git commit -m "refactor(logic): restrict internal types to pub(crate)"
```

### Task 8: Audit and restrict visibility in xlog-runtime

Same process for `crates/xlog-runtime/src/*.rs`.

- [ ] **Step 1–4:** Same as Task 4 but for xlog-runtime.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-runtime/
git commit -m "refactor(runtime): restrict internal types to pub(crate)"
```

### Task 9: Audit and restrict visibility in xlog-cuda

**IMPORTANT:** This is the largest crate (provider.rs has 12,809 lines). Focus on:
- Struct fields that should be private
- Helper functions that are only used within provider.rs
- Types in memory.rs, device.rs that are internal

Same audit process as Task 4.

- [ ] **Step 1–4:** Same as Task 4 but for xlog-cuda.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-cuda/
git commit -m "refactor(cuda): restrict internal types to pub(crate)"
```

### Task 10: Audit and restrict visibility in remaining crates

Apply the same process to: xlog-ir, xlog-prob, xlog-solve, xlog-gpu, xlog-stats.

- [ ] **Step 1–4:** Same audit for each crate.

- [ ] **Step 5: Commit per crate**

```bash
git commit -m "refactor(<crate>): restrict internal types to pub(crate)"
```

### Task 11: Final validation

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`

Expected: All pass.

- [ ] **Step 2: CUDA certification**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`

Expected: 206/206 PASS.

- [ ] **Step 3: Python smoke test**

```bash
cd /home/dev/projects/xlog && .venv/bin/python -c "import pyxlog; print('OK')"
```

Expected: OK.
