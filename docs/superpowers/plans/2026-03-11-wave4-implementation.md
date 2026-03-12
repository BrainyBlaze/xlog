# Wave 4: Pyxlog FFI/Module Extraction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `crates/pyxlog/src/lib.rs` (6,202 lines) into 7 focused submodules, consolidate ILP impl blocks, extract GPU loss computation, collapse f32/f64 forward-backward duplication, and add local error-mapping helpers.

**Architecture:** `#[pyclass]` struct definitions stay in `lib.rs`; `#[pymethods]` impl blocks move to submodules using Rust's distributed-impl pattern. GPU loss computation (574 lines) decomposes into `ilp_gpu.rs` helper functions. Error mapping via local helper functions in `types.rs` replaces ~184 scattered `.map_err(|e| PyRuntimeError::new_err(...))` calls.

**Tech Stack:** Rust, PyO3, CUDA (via xlog-cuda), maturin

---

## File Structure

After Wave 4, `crates/pyxlog/src/` will contain:

```
crates/pyxlog/src/
├── lib.rs               (~400 LOC) — #[pymodule], #[pyclass] struct defs, registration, imports
├── types.rs             (~250 LOC) — error mapping helpers, scalar_type_name, value conversions
├── program.rs           (~1,200 LOC) — CompiledProgram #[pymethods] + private impl
├── logic.rs             (~600 LOC) — Program, LogicProgram, CompiledLogicProgram, result types
├── ilp.rs               (~1,000 LOC) — CompiledIlpProgram #[pymethods] (consolidated), IlpProgramFactory
├── ilp_gpu.rs           (~700 LOC) — GPU loss/grad computation helpers
├── training.rs          (~400 LOC) — TrainingHistory, EpochStats, train_model, train_model_tensor
├── neural.rs            (~500 LOC) — network/embedding registration, forward pass, tensor sources
├── neural_registry.rs   (329 LOC) — existing, unchanged
```

Key principles:
- `#[pyclass]` struct definitions with fields MUST stay in `lib.rs` (PyO3 requirement for `#[pymodule]` registration)
- `#[pymethods]` blocks can be distributed across submodules
- Plain `impl` blocks (private helpers) also move to submodules
- Module-level functions (`train_model`, etc.) move to their thematic module
- All modules are `pub(crate)` or private — pyxlog is a `cdylib`, no Rust consumers

---

## Chunk 1: Foundation (Tasks 1–3)

### Task 1: Create types.rs — Error mapping helpers + utility functions

**Files:**
- Create: `crates/pyxlog/src/types.rs`
- Modify: `crates/pyxlog/src/lib.rs`

- [ ] **Step 1: Create `types.rs` with error helpers and utility functions**

Create `crates/pyxlog/src/types.rs` containing:

```rust
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use xlog_core::ScalarType;

/// Convert XlogError to PyErr (PyRuntimeError).
///
/// Cannot use From impl (orphan rule: From, XlogError, PyErr all foreign).
pub(crate) fn xlog_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Convert neural/value errors to PyErr (PyValueError).
pub(crate) fn val_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Format error with context prefix (for GPU operations).
pub(crate) fn gpu_err(context: &str, e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("{}: {}", context, e))
}

pub(crate) fn scalar_type_name(typ: &ScalarType) -> String {
    match *typ {
        ScalarType::U32 => "u32".to_string(),
        ScalarType::U64 => "u64".to_string(),
        ScalarType::I32 => "i32".to_string(),
        ScalarType::I64 => "i64".to_string(),
        ScalarType::F32 => "f32".to_string(),
        ScalarType::F64 => "f64".to_string(),
        ScalarType::Bool => "bool".to_string(),
        ScalarType::Symbol => "symbol".to_string(),
    }
}

#[cfg(not(feature = "host-io"))]
pub(crate) fn host_io_disabled_pyerr() -> PyErr {
    PyRuntimeError::new_err(
        "Host output is disabled (feature \"host-io\" is OFF). \
         Use device-resident APIs (DLPack) or rebuild with --features host-io.",
    )
}

/// Epsilon value for numerical stability in log computations.
pub(crate) const NLL_EPSILON: f64 = 1e-38;

/// Compute negative log-likelihood loss from probability.
#[inline]
pub(crate) fn nll_loss_value(probability: f64) -> f64 {
    -(probability.max(NLL_EPSILON)).ln()
}

/// Create a PyTorch tensor from a scalar f64 value.
pub(crate) fn create_torch_tensor(py: pyo3::Python<'_>, value: f64) -> pyo3::PyResult<pyo3::PyObject> {
    let torch = py.import_bound("torch")?;
    let tensor = torch.call_method1("tensor", (value,))?;
    Ok(tensor.into())
}
```

- [ ] **Step 2: Add `mod types;` to lib.rs and verify compilation**

Add `mod types;` after the existing `mod neural_registry;` line in lib.rs. Do NOT yet change any callers.

Run: `cargo check -p pyxlog`
Expected: PASS (types.rs compiles, no callers changed yet so existing code still works)

- [ ] **Step 3: Commit**

```bash
git add crates/pyxlog/src/types.rs crates/pyxlog/src/lib.rs
git commit -m "refactor(pyxlog): add types.rs with error mapping helpers"
```

### Task 2: Migrate lib.rs callers to types.rs helpers

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`

This task replaces the ~184 scattered `.map_err(|e| PyRuntimeError::new_err(e.to_string()))` calls with the helpers from types.rs.

- [ ] **Step 1: Replace error mapping patterns across lib.rs**

Apply these systematic replacements throughout lib.rs:

1. **Simple runtime errors**: `.map_err(|e| PyRuntimeError::new_err(e.to_string()))` → `.map_err(types::xlog_err)`
2. **Value errors**: `.map_err(|e| PyValueError::new_err(e.to_string()))` → `.map_err(types::val_err)`
3. **Prefixed GPU errors**: `.map_err(|e| PyRuntimeError::new_err(format!("context: {}", e)))` → `.map_err(|e| types::gpu_err("context", e))`
4. **Direct function references**: Replace `scalar_type_name(...)` calls with `types::scalar_type_name(...)`, `nll_loss_value(...)` with `types::nll_loss_value(...)`, `create_torch_tensor(...)` with `types::create_torch_tensor(...)`, `host_io_disabled_pyerr()` with `types::host_io_disabled_pyerr()`
5. Remove the now-dead original function definitions from lib.rs: `scalar_type_name`, `host_io_disabled_pyerr`, `nll_loss_value`, `create_torch_tensor`, and the `NLL_EPSILON` constant

Keep `use pyo3::exceptions::{PyRuntimeError, PyValueError};` in lib.rs for now — some direct `PyValueError::new_err(...)` calls with inline messages won't use the helper (they're not wrapping another error).

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS — all callers now use types.rs helpers

- [ ] **Step 3: Commit**

```bash
git add crates/pyxlog/src/lib.rs crates/pyxlog/src/types.rs
git commit -m "refactor(pyxlog): migrate error mapping to types.rs helpers (~184 call sites)"
```

### Task 3: Create training.rs — Extract training infrastructure

**Files:**
- Create: `crates/pyxlog/src/training.rs`
- Modify: `crates/pyxlog/src/lib.rs`

- [ ] **Step 1: Create `training.rs`**

Move from lib.rs to training.rs:
- `TrainingHistory` plain `impl` block (lib.rs:3819–3837): `new()`, `add_epoch()`, `add_batch()`
- `train_model` function (lib.rs:3857–3924) — module-level `#[pyfunction]`
- `train_model_tensor` function (lib.rs:3932–4005) — module-level `#[pyfunction]`

The `#[pyclass]` struct definitions for `EpochStats` and `TrainingHistory` STAY in lib.rs.

The training.rs file needs these imports:
```rust
use pyo3::prelude::*;
use super::CompiledProgram;
use super::types;

// For train_model / train_model_tensor:
pub(crate) use super::TrainingHistory;  // Re-export for internal use
```

For the `#[pyfunction]` functions, they need to be `pub(crate)` and registered in lib.rs's `#[pymodule]` via:
```rust
// In lib.rs pymodule function:
m.add_function(wrap_pyfunction!(training::train_model, m)?)?;
m.add_function(wrap_pyfunction!(training::train_model_tensor, m)?)?;
```

Note: `wrap_pyfunction!` requires the function to be `pub` with `#[pyfunction]` attribute. Mark them `pub` in training.rs and adjust the `#[pymodule]` registration path.

- [ ] **Step 2: Add `mod training;` to lib.rs, update #[pymodule], remove moved code from lib.rs**

Add `mod training;` to lib.rs. Remove the moved function bodies from lib.rs. Update the `#[pymodule]` function to use `training::train_model` and `training::train_model_tensor`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/pyxlog/src/training.rs crates/pyxlog/src/lib.rs
git commit -m "refactor(pyxlog): extract training infrastructure to training.rs"
```

---

## Chunk 2: Logic + ILP extraction (Tasks 4–6)

### Task 4: Create logic.rs — Extract logic program classes

**Files:**
- Create: `crates/pyxlog/src/logic.rs`
- Modify: `crates/pyxlog/src/lib.rs`

- [ ] **Step 1: Create `logic.rs`**

Move from lib.rs to logic.rs:
- `Program` `#[pymethods]` impl block (lib.rs:454–541) — the `compile` staticmethod
- `LogicProgram` `#[pymethods]` impl block (lib.rs:3572–3596)
- `CompiledLogicProgram` `#[pymethods]` impl block (lib.rs:3604–3652)
- `CompiledLogicProgram` plain `impl` block (lib.rs:3654–3692) — `pack_logic_result`

The `#[pyclass]` struct definitions for ALL of these STAY in lib.rs:
- `Program` (lib.rs:451–452)
- `LogicProgram` (lib.rs:3569–3570)
- `CompiledLogicProgram` (lib.rs:3598–3602)
- `LogicQueryResult` (lib.rs:3694–3706)
- `LogicEvalResult` (lib.rs:3708–3712)
- `McDeviceEvalResult` (lib.rs:3714–3738)
- `EvalResult` (lib.rs:3740–3780)

The logic.rs needs these imports:
```rust
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};
use pyo3::exceptions::PyValueError;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, DlpackManagedTensor, GpuMemoryManager};
use xlog_logic::ast::ProbEngine;
use xlog_neural::{NetworkRegistry, TensorSourceRegistry};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::mc::McProgram;

use super::{
    types, CompiledProgram, CompiledLogicProgram, CompiledProbProgram,
    LogicEvalResult, LogicQueryResult, Program, LogicProgram,
    dlpack_capsule_from_tensor, dlpack_from_py, provider_from_config,
    parse_prob_engine_override,
};
use super::neural_registry::NeuralPredicateRegistry;
```

Note: `provider_from_config` and `parse_prob_engine_override` are module-level functions that stay in lib.rs (or could move to types.rs). They're used by logic.rs and other modules.

- [ ] **Step 2: Add `mod logic;` to lib.rs, remove moved code**

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/pyxlog/src/logic.rs crates/pyxlog/src/lib.rs
git commit -m "refactor(pyxlog): extract logic program classes to logic.rs"
```

### Task 5: Create ilp.rs — Extract ILP program (consolidated impl blocks)

**Files:**
- Create: `crates/pyxlog/src/ilp.rs`
- Modify: `crates/pyxlog/src/lib.rs`

- [ ] **Step 1: Create `ilp.rs`**

Move from lib.rs to ilp.rs — **consolidating both impl blocks into one file**:
- `IlpProgramFactory` `#[pymethods]` impl block (lib.rs:4302–4384)
- `CompiledIlpProgram` `#[pymethods]` impl block (lib.rs:4412–5675) — the main block with `compute_ilp_loss_grad_gpu`, `set_rule_mask`, `evaluate`, etc.
- `CompiledIlpProgram` plain `impl` block (lib.rs:5677–6174) — the GPU export helpers (`build_zero_loss_grad`, `export_loss_grad_device_f32/f64`, `build_loss_grad_empty_coo`)

Also move these ILP-related helper functions:
- `push_term_bytes` (lib.rs:4011–4078)
- `pack_i64_columns_typed` (lib.rs:4083–4176)
- `load_facts_into_store` (lib.rs:4178–4224)
- `TmjMeta` struct + `walk_tmj` + `extract_tmj_meta` + `extract_tmj_meta_for_mask` (lib.rs:4227–4283)
- `strip_learnable_declarations` + `extract_learnable_declarations` (lib.rs:4285–4297)

The `#[pyclass]` struct definitions STAY in lib.rs:
- `IlpProgramFactory` (lib.rs:4299–4300)
- `CompiledIlpProgram` (lib.rs:4386–4410)

The two impl blocks merge into a single `#[pymethods] impl CompiledIlpProgram { ... }` block plus a plain `impl CompiledIlpProgram { ... }` block for the private GPU helpers.

**Stale comment cleanup**: Remove the outdated host-merging comment at lib.rs:4689–4693 during extraction. The chunked path has been GPU-only since v0.5.0-phase1.

**Ride-along — unwrap fixes**: During extraction, audit GPU compute paths for production-path `.unwrap()` calls. Replace with proper error propagation (`.ok_or_else(|| ...)` or `.map_err(...)`).

- [ ] **Step 2: Add `mod ilp;` to lib.rs, remove moved code**

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/pyxlog/src/ilp.rs crates/pyxlog/src/lib.rs
git commit -m "refactor(pyxlog): extract ILP program to ilp.rs, consolidate impl blocks"
```

### Task 6: Create ilp_gpu.rs — Extract GPU loss/grad computation

**Files:**
- Create: `crates/pyxlog/src/ilp_gpu.rs`
- Modify: `crates/pyxlog/src/ilp.rs`

- [ ] **Step 1: Create `ilp_gpu.rs` with decomposed GPU computation**

Extract the body of `compute_ilp_loss_grad_gpu` (574 lines, lib.rs:4459–5037) from ilp.rs into focused helper functions in ilp_gpu.rs:

```rust
use std::collections::HashMap;
use pyo3::prelude::*;
use cudarc::driver::DeviceSlice;
use xlog_core::ScalarType;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use super::types;

/// Task descriptor for COO fill (one per query×candidate pair).
pub(crate) struct CooTask {
    pub(crate) d_mask: cudarc::driver::CudaSlice<u8>,
    pub(crate) fact_indices_idx: usize,
    pub(crate) cidx: u32,
    pub(crate) num_query: u32,
}

/// Build COO arrays via single-pass device fill (non-chunked path).
pub(crate) fn build_coo_single(
    provider: &CudaKernelProvider,
    tasks: &[CooTask],
    fact_indices_buffers: &[cudarc::driver::CudaSlice<u32>],
    num_facts: u32,
    num_cands: u32,
    upper_bound: u32,
) -> PyResult<(cudarc::driver::CudaSlice<u32>, cudarc::driver::CudaSlice<u32>, u32)> {
    // ... non-chunked COO fill logic (lib.rs:4694–4765)
}

/// Build COO arrays via two-pass GPU-only bounded-memory merge (chunked path).
pub(crate) fn build_coo_chunked(
    provider: &CudaKernelProvider,
    tasks: &[CooTask],
    fact_indices_buffers: &[cudarc::driver::CudaSlice<u32>],
    num_facts: u32,
    num_cands: u32,
    coo_chunk_budget: u64,
) -> PyResult<(cudarc::driver::CudaSlice<u32>, cudarc::driver::CudaSlice<u32>, u32)> {
    // ... chunked COO merge logic (lib.rs:4767–4918)
}

/// Sort COO by fact index and build CSR row offsets on device.
pub(crate) fn sort_and_build_csr(
    provider: &CudaKernelProvider,
    d_coo_facts: &mut cudarc::driver::CudaSlice<u32>,
    d_coo_cands: &mut cudarc::driver::CudaSlice<u32>,
    actual_nnz: u32,
    num_facts: u32,
) -> PyResult<cudarc::driver::CudaSlice<u32>> {
    // ... Phase D sort + CSR build (lib.rs:4921–4971)
}

/// Run forward (credit gather + NLL loss) and backward (gradient scatter)
/// CUDA kernels, then reduce and export via DLPack.
pub(crate) fn forward_backward_reduce(
    provider: &CudaKernelProvider,
    py: Python<'_>,
    d_row_offsets: &cudarc::driver::CudaSlice<u32>,
    d_coo_cands: &cudarc::driver::CudaSlice<u32>,
    cand_col: &xlog_cuda::CudaColumn,
    d_is_positive: &cudarc::driver::CudaSlice<u8>,
    num_facts: u32,
    num_cands: u32,
    is_f64: bool,
    // export closures or direct provider ref for DLPack export
) -> PyResult<(PyObject, PyObject)> {
    // ... Phase E forward + backward + reduction (lib.rs:4973–5036)
}
```

The original `compute_ilp_loss_grad_gpu` in ilp.rs becomes a ~40-line dispatcher:
```rust
pub fn compute_ilp_loss_grad_gpu<'py>(&self, py: Python<'py>, ...) -> PyResult<(PyObject, PyObject)> {
    // Phase A: validation + DLPack import (unchanged, ~40 lines)
    // Phase B: task collection (unchanged, ~100 lines — stays inline)
    // Dispatch to ilp_gpu helpers for Phases C–E
    let (d_coo_facts, d_coo_cands, actual_nnz) = if !needs_chunking {
        ilp_gpu::build_coo_single(...)
    } else {
        ilp_gpu::build_coo_chunked(...)
    };
    let d_row_offsets = ilp_gpu::sort_and_build_csr(...)?;
    ilp_gpu::forward_backward_reduce(...)
}
```

- [ ] **Step 2: Add `mod ilp_gpu;` to lib.rs, update ilp.rs to use ilp_gpu helpers**

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/pyxlog/src/ilp_gpu.rs crates/pyxlog/src/ilp.rs crates/pyxlog/src/lib.rs
git commit -m "refactor(pyxlog): extract GPU loss/grad computation to ilp_gpu.rs"
```

---

## Chunk 3: Neural + Program extraction (Tasks 7–8)

### Task 7: Create neural.rs — Extract neural network registration and forward pass

**Files:**
- Create: `crates/pyxlog/src/neural.rs`
- Modify: `crates/pyxlog/src/lib.rs` (or program.rs if already split)

- [ ] **Step 1: Create `neural.rs`**

Move the neural-specific `#[pymethods]` from the `CompiledProgram` impl block in lib.rs:
- `register_network` (lib.rs:853–892)
- `register_embedding` (lib.rs:913–1023)
- `forward_embedding` (lib.rs:1035–1088)
- `set_train_mode` (lib.rs:1175)
- `add_tensor_source` (lib.rs:1203–1228)
- `set_active_tensor_source` (lib.rs:1236–1242)
- `active_tensor_source` (lib.rs:1243–1245)
- `active_tensor_source_size` (lib.rs:1248–1254)
- `tensor_source_names` (lib.rs:1255–1257)
- `has_tensor_source` (lib.rs:1260–1262)

Also move the private neural forward-backward helpers from the plain `impl CompiledProgram` block:
- `forward_backward_tensor_internal` (lib.rs:1659–1681)
- `forward_backward_direct_tensor` (lib.rs:1962–2024)
- `forward_backward_complex_tensor` (lib.rs:2038–~2300)
- `forward_backward_batch_complex_tensor` (lib.rs:2305–~2450)
- Helper methods used only by neural code: `get_input_tensor`, `get_label_index`, `try_parse_direct_neural_query`, `term_to_input_idx`, `term_to_label`, `parse_query_atom`, `get_or_build_query_signature`

These are distributed `impl CompiledProgram` blocks — a `#[pymethods]` block for the public PyO3 methods and a plain `impl` block for private helpers.

- [ ] **Step 2: Collapse f32/f64 forward-backward duplication**

Per spec Section 4, create a generic internal helper in neural.rs:

```rust
/// Single implementation for both f32 and f64 forward-backward paths.
/// The #[pymethods] wrappers call this with the appropriate ScalarType.
fn forward_backward_typed(
    provider: &CudaKernelProvider,
    circuit: &GpuCircuit,
    weights: &CudaBuffer,
    scalar_type: ScalarType,
    batch_size: Option<usize>,
) -> Result<ForwardBackwardResult> {
    // Single implementation, kernel names selected by scalar_type
    ...
}
```

Identify the f32/f64 duplication in the forward-backward methods (e.g. `forward_backward_complex_tensor` builds `schema_f32`/`schema_f64` and imports via DLPack with different schemas). Consolidate into a single code path parameterized by `ScalarType`, following the same pattern as Wave 2's GpuScalar migration.

- [ ] **Step 3: Add `mod neural;` to lib.rs, remove moved code**

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/pyxlog/src/neural.rs crates/pyxlog/src/lib.rs
git commit -m "refactor(pyxlog): extract neural registration and forward pass to neural.rs"
```

### Task 8: Create program.rs — Extract remaining CompiledProgram methods

**Files:**
- Create: `crates/pyxlog/src/program.rs`
- Modify: `crates/pyxlog/src/lib.rs`

- [ ] **Step 1: Create `program.rs`**

Move all remaining `CompiledProgram` `#[pymethods]` and plain `impl` blocks from lib.rs:
- The `#[pymethods]` block (lib.rs:654–1656): `evaluate`, `evaluate_device`, and all other public methods not moved to neural.rs
- The plain `impl CompiledProgram` blocks not moved to neural.rs: `parse_sampling_method` (lib.rs:640–651), `train_epoch_internal` (lib.rs:1684+), pack result helpers, etc.
- `CompiledProbProgram` enum + impl (lib.rs:591–604) — internal type
- `CachedCircuit` struct (lib.rs:547–556) — internal type
- `QuerySignature` enum + impl (lib.rs:572–589) — internal type
- `InputSource` enum (lib.rs:558–562) — internal type
- `NeuralGroup` struct (lib.rs:564–570) — internal type

After this step, lib.rs should contain ONLY:
- Imports
- `mod` declarations
- DLPack/Arrow capsule functions (FFI boundary — can stay in lib.rs or move to types.rs)
- `#[pyclass]` struct definitions with fields
- `provider_from_config` and `parse_prob_engine_override` helper functions
- The `#[pymodule]` function

- [ ] **Step 2: Add `mod program;` to lib.rs, remove moved code**

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/pyxlog/src/program.rs crates/pyxlog/src/lib.rs
git commit -m "refactor(pyxlog): extract CompiledProgram methods to program.rs"
```

---

## Chunk 4: Deduplication + Validation (Tasks 9–11)

### Task 9: Collapse f32/f64 forward-backward duplication in ilp_gpu.rs

**Files:**
- Modify: `crates/pyxlog/src/ilp_gpu.rs`
- Modify: `crates/pyxlog/src/ilp.rs`

- [ ] **Step 1: Add typed dispatch helper**

In `ilp_gpu.rs`, the `forward_backward_reduce` function has two near-identical branches for f32 and f64. The key difference is which provider method is called (`ilp_credit_forward_f32_launch` vs `ilp_credit_forward_f64_launch`, etc.) and the epsilon type.

Create a helper that takes `is_f64: bool` and dispatches, reducing the f32/f64 duplication within the forward/backward/reduce phase. The GpuScalar trait from Wave 2 may be usable here — check if the ILP kernel launch methods are generic or if they remain type-specialized.

If the ILP provider methods are NOT generic (they're `_f32_launch` / `_f64_launch` suffixed), the duplication reduction is limited to the export helpers (`export_loss_grad_device_f32/f64` → single generic `export_loss_grad_device<T>`). This is still valuable as the export helpers have significant overlap.

Also collapse `build_zero_loss_grad` — it has identical f32/f64 branches differing only in alloc type.

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/pyxlog/src/ilp_gpu.rs crates/pyxlog/src/ilp.rs
git commit -m "refactor(pyxlog): collapse f32/f64 duplication in ILP GPU helpers"
```

### Task 10: Clean up lib.rs — Final slimming

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`

- [ ] **Step 1: Review and slim lib.rs**

After Tasks 1–9, verify lib.rs contains only:
1. Crate-level imports
2. `mod` declarations (types, training, logic, ilp, ilp_gpu, neural, program, neural_registry)
3. DLPack/Arrow FFI capsule functions (these are unsafe FFI and stay in lib.rs)
4. `#[pyclass]` struct definitions with fields
5. `provider_from_config` and `parse_prob_engine_override` (shared utility)
6. The `#[pymodule]` function

Remove any unused imports. Verify no production logic remains beyond the above.

**Ride-along — visibility audit**: Audit `#[pyo3(get)]` fields on all `#[pyclass]` structs for correctness (~10 changes). Ensure fields that should not be publicly writable from Python are `get`-only (no `set`). This is a small audit — fields are already private in Rust.

Target: ~400 lines for lib.rs.

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p pyxlog`
Expected: PASS

- [ ] **Step 3: Commit (if any changes)**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "refactor(pyxlog): final lib.rs cleanup (~400 LOC)"
```

### Task 11: Full gate validation

**Files:** None (validation only)

This is the first wave with the FULL documented release matrix. All 7 suites plus pyxlog-specific build gates.

- [ ] **Step 1: pyxlog compile check**

Run: `cargo check -p pyxlog`
Expected: PASS — no compilation errors

- [ ] **Step 2: Python wheel build**

Run: `maturin develop --release -m crates/pyxlog/Cargo.toml`
Expected: PASS — wheel builds and installs into .venv

- [ ] **Step 3: Rust workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: PASS

- [ ] **Step 4: CUDA certification suite**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`
Expected: 206/206 PASS

- [ ] **Step 5: Python non-slow batch (Gate 3)**

Run: `.venv/bin/python -m pytest python/tests/ --timeout=120 -k "not slow and not test_second_epoch and not test_non_monotone" -v`
Expected: ALL PASS — this is the critical gate for Wave 4 since we restructured FFI boundaries

- [ ] **Step 6: ILP sparse tests (Gate 5)**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_sparse.py -x -v`
Expected: PASS

- [ ] **Step 7: ILP reliability tests (Gate 4)**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -x -v`
Expected: 20/20 PASS

- [ ] **Step 8: GA reliability (Gate 6)**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_ga_reliability.py -v --timeout=600`
Expected: PASS

- [ ] **Step 9: ILP performance (Gate 7)**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_performance.py -v --timeout=600`
Expected: PASS

- [ ] **Step 10: Verify line counts**

Run: `wc -l crates/pyxlog/src/*.rs`
Expected:
- lib.rs: ~400 (down from 6,202)
- types.rs: ~250
- program.rs: ~1,200
- logic.rs: ~600
- ilp.rs: ~1,000
- ilp_gpu.rs: ~700
- training.rs: ~400
- neural.rs: ~500
- neural_registry.rs: 329 (unchanged)
- **Total**: ~5,380 (vs 6,531 current — ~18% net reduction)

- [ ] **Step 11: Verify Python API surface**

Run:
```bash
.venv/bin/python -c "
from pyxlog import Program, CompiledProgram, LogicProgram, CompiledLogicProgram
from pyxlog import LogicQueryResult, LogicEvalResult, McDeviceEvalResult, EvalResult
from pyxlog import EpochStats, TrainingHistory, train_model, train_model_tensor
from pyxlog import IlpProgramFactory, CompiledIlpProgram
print('All imports OK')
"
```
Expected: "All imports OK" — frozen API surface preserved

- [ ] **Step 12: Commit validation results (if any fixes needed)**

If any gate fails, fix and re-run. Final commit only if fixes were needed.
