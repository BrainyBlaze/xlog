# Wave 1: Error Unification & Visibility Lockdown

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the workspace error system with `From` bridges and remove accidental public API surface before structural module splits in Wave 2.

**Architecture:** Add `From<CrateError> for XlogError` implementations in `xlog-core/src/error.rs` for all 5 downstream error types. Remove the `Result<T>` alias collision in `xlog-neural`. Audit and restrict visibility across all crates.

**Tech Stack:** Rust (thiserror), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-10-codebase-refactoring-design.md` (C1, C2)

---

## Chunk 1: Error System Unification (C1)

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `crates/xlog-core/src/error.rs` | Add 5 `From` impls, add 2 new XlogError variants |
| Modify | `crates/xlog-core/Cargo.toml` | Add dependencies on xlog-neural, xlog-logic |
| Modify | `crates/xlog-neural/src/lib.rs:75` | Remove `Result<T>` alias |
| Test | `crates/xlog-core/tests/test_error_conversions.rs` | New: verify From impls compile and convert correctly |

### Task 1: Add XlogError variants for downstream error types

The current `XlogError` enum has generic `String` variants. We need variants that wrap the actual error types for proper `From` conversion.

**Files:**
- Modify: `crates/xlog-core/src/error.rs:7-35`
- Modify: `crates/xlog-core/Cargo.toml`

**Important context:** The error types are:
- `xlog-neural::NeuralError` (6 variants, thiserror) — `crates/xlog-neural/src/lib.rs:48-72`
- `xlog-neural::TensorSourceError` (3 variants, thiserror) — `crates/xlog-neural/src/tensor_source.rs:47-59`
- `xlog-logic::ModuleError` (6 variants, custom Display) — `crates/xlog-logic/src/module.rs:45-65`
- `xlog-logic::FunctionError` (5 variants, custom Display) — `crates/xlog-logic/src/function.rs:9-20`
- `xlog-logic::TypeError` (2 variants, custom Display) — `crates/xlog-logic/src/function.rs:64-73`

- [ ] **Step 1: Write the failing test**

Create `crates/xlog-core/tests/test_error_conversions.rs`:

```rust
use xlog_core::XlogError;
use xlog_neural::{NeuralError, TensorSourceError};
use xlog_logic::module::ModuleError;
use xlog_logic::function::{FunctionError, TypeError};

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
fn module_error_converts_to_xlog_error() {
    let err = ModuleError::CircularImport {
        cycle: vec![],
    };
    let xlog_err: XlogError = err.into();
    assert!(matches!(xlog_err, XlogError::Module(_)));
}

#[test]
fn function_error_converts_to_xlog_error() {
    let err = FunctionError::UndefinedFunction {
        name: "test".into(),
    };
    let xlog_err: XlogError = err.into();
    assert!(matches!(xlog_err, XlogError::Function(_)));
}

#[test]
fn type_error_converts_to_xlog_error() {
    let err = TypeError::CannotInfer {
        name: "X".into(),
    };
    let xlog_err: XlogError = err.into();
    // TypeError maps to the existing XlogError::Type variant via From
    assert!(matches!(xlog_err, XlogError::LogicType(_)));
}

#[test]
fn question_mark_propagation_works() {
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

Expected: Compilation failure — `XlogError::Neural` variant doesn't exist, no `From` impl.

- [ ] **Step 3: Add xlog-neural and xlog-logic dependencies to xlog-core**

In `crates/xlog-core/Cargo.toml`, add under `[dependencies]`:

```toml
xlog-neural = { path = "../xlog-neural" }
xlog-logic = { path = "../xlog-logic" }
```

**IMPORTANT:** Check this doesn't create a circular dependency. `xlog-neural` and `xlog-logic` must NOT depend on `xlog-core`. Verify:
- `crates/xlog-neural/Cargo.toml` — check for `xlog-core` dependency
- `crates/xlog-logic/Cargo.toml` — check for `xlog-core` dependency

If either depends on xlog-core, we have a cycle. In that case, use a different approach: define the `From` impls in the downstream crate instead (orphan rule allows `impl From<LocalType> for ForeignType` when `LocalType` is local). Document the decision.

- [ ] **Step 4: Add new variants and From impls to XlogError**

In `crates/xlog-core/src/error.rs`, add 5 new variants to the enum and 5 `From` impls:

```rust
use xlog_neural::{NeuralError, TensorSourceError};
use xlog_logic::module::ModuleError;
use xlog_logic::function::{FunctionError, TypeError};

#[derive(Debug, thiserror::Error)]
pub enum XlogError {
    // ... existing variants unchanged ...

    #[error("Neural error: {0}")]
    Neural(#[from] NeuralError),

    #[error("Tensor source error: {0}")]
    TensorSource(#[from] TensorSourceError),

    #[error("Module error: {0}")]
    Module(#[from] ModuleError),

    #[error("Function error: {0}")]
    Function(#[from] FunctionError),

    #[error("Logic type error: {0}")]
    LogicType(#[from] TypeError),
}
```

The `#[from]` attribute on thiserror auto-generates the `From` impl.

**NOTE:** The existing `Type(String)` variant must NOT conflict with the new `LogicType(TypeError)`. Keep both — `Type(String)` is for string-based type errors from other sources; `LogicType` wraps the structured TypeError enum.

**NOTE:** `ModuleError`, `FunctionError`, and `TypeError` use custom `Display`/`Error` impls, not thiserror. The `#[from]` attribute requires `std::error::Error` to be implemented — verify that all three have `impl std::error::Error`. Check:
- `crates/xlog-logic/src/function.rs:60` — should have `impl std::error::Error for FunctionError {}`
- `crates/xlog-logic/src/function.rs:93` — should have `impl std::error::Error for TypeError {}`
- `crates/xlog-logic/src/module.rs:146` — should have `impl std::error::Error for ModuleError {}`

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p xlog-core --test test_error_conversions --release`

Expected: 6/6 PASS

- [ ] **Step 6: Run full workspace test to check no regressions**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20`

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/xlog-core/src/error.rs crates/xlog-core/Cargo.toml crates/xlog-core/tests/test_error_conversions.rs
git commit -m "feat(core): add From impls bridging all error types to XlogError"
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

**Context:** There are 1,094 `.map_err()` calls across the workspace. Most convert `cudarc::DriverError` to strings — those are NOT affected by our new `From` impls. Focus only on sites that convert between xlog error types.

**Files:**
- Search: All crates for `.map_err(|e| XlogError::...(...Neural...))` patterns
- Search: All crates for `.map_err(|e| XlogError::...(...Module...))` patterns

- [ ] **Step 1: Identify convertible .map_err() sites**

Run these searches:
```bash
grep -rn 'map_err.*NeuralError\|map_err.*ModuleError\|map_err.*FunctionError\|map_err.*TypeError\|map_err.*TensorSourceError' crates/
```

Only convert sites where the source error type exactly matches one of our 5 `From` targets. Do NOT convert sites that format error messages with additional context (those `.map_err()` calls add valuable information).

- [ ] **Step 2: Convert identified sites**

For each site, change:
```rust
// Before
something.map_err(|e| XlogError::Execution(format!("Neural: {}", e)))?;

// After (only if error type is NeuralError)
something?;
```

**CAUTION:** Only convert if the `.map_err()` is purely wrapping the error without adding context. If it adds context like `format!("Failed to load network {}: {}", name, e)`, keep the `.map_err()` — that context is valuable for debugging.

- [ ] **Step 3: Run workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -20`

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "refactor: replace .map_err() with ? where From impls now handle conversion"
```

---

## Chunk 2: Visibility Lockdown (C2)

### Task 4: Audit and restrict visibility in xlog-core

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

### Task 5: Audit and restrict visibility in xlog-neural

Same process as Task 4 for `crates/xlog-neural/src/*.rs`.

- [ ] **Step 1–4:** Same as Task 4 but for xlog-neural.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-neural/
git commit -m "refactor(neural): restrict internal types to pub(crate)"
```

### Task 6: Audit and restrict visibility in xlog-logic

Same process for `crates/xlog-logic/src/*.rs`.

- [ ] **Step 1–4:** Same as Task 4 but for xlog-logic.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-logic/
git commit -m "refactor(logic): restrict internal types to pub(crate)"
```

### Task 7: Audit and restrict visibility in xlog-runtime

Same process for `crates/xlog-runtime/src/*.rs`.

- [ ] **Step 1–4:** Same as Task 4 but for xlog-runtime.

- [ ] **Step 5: Commit**

```bash
git add crates/xlog-runtime/
git commit -m "refactor(runtime): restrict internal types to pub(crate)"
```

### Task 8: Audit and restrict visibility in xlog-cuda

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

### Task 9: Audit and restrict visibility in remaining crates

Apply the same process to: xlog-ir, xlog-prob, xlog-solve, xlog-gpu, xlog-stats.

- [ ] **Step 1–4:** Same audit for each crate.

- [ ] **Step 5: Commit per crate**

```bash
git commit -m "refactor(<crate>): restrict internal types to pub(crate)"
```

### Task 10: Remove unnecessary Cargo dependencies (H4)

**Files:**
- Modify: `crates/xlog-stats/Cargo.toml` — remove `xlog-cuda`
- Modify: `crates/xlog-logic/Cargo.toml` — remove `xlog-runtime`

- [ ] **Step 1: Verify zero usage**

```bash
grep -r 'xlog_cuda' crates/xlog-stats/src/
grep -r 'xlog_runtime' crates/xlog-logic/src/
```

Expected: No matches (confirming zero usage).

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
