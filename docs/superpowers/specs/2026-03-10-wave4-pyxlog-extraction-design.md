# Wave 4: Pyxlog FFI/Module Extraction + Full Python Gates

**Date**: 2026-03-10
**Status**: Approved
**Depends on**: Waves 2–3 (cleaner provider + executor APIs)

## Overview

Split `crates/pyxlog/src/lib.rs` (6,202 lines) into focused submodules. Consolidate the two
non-contiguous `CompiledIlpProgram` impl blocks. Extract GPU loss/gradient computation.
Collapse f32/f64 forward-backward duplication. Add local PyO3 error-mapping helpers. This is
the first wave that triggers the full documented Python release matrix.

## Constraints

- Green at wave boundary: full documented release matrix
- Frozen Python API surface — see Section 2 for accurate inventory
- Internal Rust restructuring only — no Python-visible changes
- Cannot use `From<XlogError> for PyErr` (orphan rule) — use local helpers instead
- Preserve zero data-plane D2H with metadata-only untracked scalar reads allowed
- Clean up stale host-merging comment (former lib.rs:4689) during extraction
- CompiledProgram and CompiledIlpProgram fields are already private — visibility ride-along is small
- pyxlog is a `cdylib` — no Rust crate imports from it

## 1. File Split

```
crates/pyxlog/src/
├── lib.rs               → lib.rs          (slimmed: #[pymodule] + #[pyclass] struct defs + registration)
├──                        program.rs       (CompiledProgram #[pymethods])
├──                        logic.rs         (LogicProgram, Program, CompiledLogicProgram, result types)
├──                        ilp.rs           (CompiledIlpProgram, IlpProgramFactory #[pymethods] — consolidated)
├──                        ilp_gpu.rs       (GPU loss/grad computation helpers)
├──                        training.rs      (TrainingHistory, EpochStats)
├──                        neural.rs        (network/embedding registration, forward pass)
├──                        types.rs         (type conversions, scalar helpers, error mapping)
```

### Module Responsibilities

| Module | Content | LOC est. |
|--------|---------|----------|
| `lib.rs` | `#[pymodule]` function (lib.rs:6176), `#[pyclass]` struct definitions with fields, module registration, submodule imports | ~400 |
| `program.rs` | `CompiledProgram` `#[pymethods]`: all method blocks currently starting at lib.rs:654 — evaluate, evaluate_device, tensor-source management, training helpers, forward/backward paths, cache controls, query-probability helpers | ~1,200 |
| `logic.rs` | `LogicProgram` (lib.rs:3570), `Program` (lib.rs:451), `CompiledLogicProgram` (lib.rs:3599), `LogicQueryResult` (lib.rs:3694), `LogicEvalResult` (lib.rs:3709), `McDeviceEvalResult` (lib.rs:3714), `EvalResult` (lib.rs:3740) — all #[pymethods] for these types | ~600 |
| `ilp.rs` | `CompiledIlpProgram` `#[pymethods]`: consolidated from 2 non-contiguous impl blocks (lib.rs:4412 and lib.rs:5677). `IlpProgramFactory` (lib.rs:4299) with `compile`. Runtime/control surface, training loop orchestration. Delegates GPU compute to `ilp_gpu.rs`. | ~1,000 |
| `ilp_gpu.rs` | GPU loss/gradient computation extracted from ILP: chunked/non-chunked paths, COO merge, sparse reduction, evidence clamping setup | ~700 |
| `training.rs` | `TrainingHistory` (lib.rs:3802), `EpochStats` (lib.rs:3787) — recording, metrics access | ~400 |
| `neural.rs` | Network registration, embedding registration/forward, tensor source management, generic f32/f64 forward-backward helper | ~500 |
| `types.rs` | `scalar_type_name()`, error mapping helpers, value conversion utilities | ~250 |

**Total**: ~5,050 LOC (vs 6,202 current — ~19% reduction)

## 2. Frozen Python API Surface (Accurate Inventory)

The following `#[pyclass]` and `#[pyfunction]` exports must remain identical from Python:

| PyClass / Export | Defined at | Target module | Key methods |
|-----------------|------------|---------------|-------------|
| `Program` | lib.rs:451 | logic.rs | Constructor, compile |
| `CompiledProgram` | lib.rs:654+ | program.rs | Full method surface (evaluate, evaluate_device, tensor-source, training, forward/backward, cache, query-probability) |
| `LogicProgram` | lib.rs:3570 | logic.rs | Constructor, compile |
| `CompiledLogicProgram` | lib.rs:3599 | logic.rs | Evaluation methods |
| `LogicQueryResult` | lib.rs:3694 | logic.rs | Result access |
| `LogicEvalResult` | lib.rs:3709 | logic.rs | Result access |
| `McDeviceEvalResult` | lib.rs:3714 | logic.rs | Device result access |
| `EvalResult` | lib.rs:3740 | logic.rs | Result access |
| `EpochStats` | lib.rs:3787 | training.rs | Training metrics |
| `TrainingHistory` | lib.rs:3802 | training.rs | Training recording, `add_batch` (lib.rs:3834) |
| `IlpProgramFactory` | lib.rs:4299 | ilp.rs | `compile` (lib.rs:4299) |
| `CompiledIlpProgram` | lib.rs:4412+ | ilp.rs | Full runtime/control surface across both impl blocks |
| Module functions | lib.rs:6176 | lib.rs | All `#[pyfunction]` exports in `#[pymodule]` (stays in lib.rs for registration) |

Experimental feature-gated surfaces (e.g., Arrow device import/export) can move faster.

## 3. Decomposing compute_ilp_loss_grad_gpu() (574 lines)

Extract into `ilp_gpu.rs` as focused helper functions:

| Extracted function | LOC est. | Responsibility |
|-------------------|----------|----------------|
| `compute_loss_grad_chunked()` | ~200 | Chunked GPU loss path with COO chunk iteration |
| `compute_loss_grad_single()` | ~120 | Non-chunked single-pass GPU loss path |
| `merge_coo_chunks()` | ~100 | Two-pass GPU-only chunk merge (count NNZ → fill COO) |
| `apply_sparse_reduction()` | ~80 | CSR histogram + reduce_sum for gradient accumulation |
| `apply_evidence_clamping()` | ~60 | Evidence forcing setup for clamped mode |

The original method becomes a ~40-line dispatcher in `ilp.rs` that calls these helpers.

**GPU-residency contract**: Extracted functions preserve zero data-plane D2H. The existing
`dtoh_scalar_untracked` for metadata-only reads (e.g., total_nnz scalar at lib.rs:4841)
stays as-is — metadata reads are excluded from transfer accounting per CHANGELOG.md:76 and
ROADMAP.md:479.

**Stale comment cleanup**: The host-merging comment at lib.rs:4689 is outdated (the chunked
path has been GPU-only since v0.5.0-phase1). Remove during extraction.

## 4. Collapsing f32/f64 Forward-Backward Duplication

Internal generic helper in `neural.rs`:

```rust
// Pure Rust internal helper, not a #[pymethods]
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

The `#[pymethods]` wrappers call this with the appropriate `ScalarType`. Python users see
no change.

## 5. PyO3 Error Mapping

**Cannot use `From` impls** (orphan rule: `From`, `PyErr`, `XlogError`, `NeuralError` are
all foreign to pyxlog).

**Instead**: Local helper functions in `types.rs`:

```rust
pub(crate) fn xlog_err_to_py(e: XlogError) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

pub(crate) fn neural_err_to_py(e: NeuralError) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
```

Or alternatively, a local extension trait:

```rust
pub(crate) trait IntoPyErr {
    fn into_py_err(self) -> pyo3::PyErr;
}

impl IntoPyErr for XlogError {
    fn into_py_err(self) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(self.to_string())
    }
}
```

Either approach replaces the scattered `.map_err(|e| PyRuntimeError::new_err(e.to_string()))`
calls with `.map_err(xlog_err_to_py)` or `.map_err(IntoPyErr::into_py_err)`.

**Error type mapping** (preserves existing conventions):
- `XlogError` → `PyRuntimeError`
- `NeuralError` → `PyValueError`

## 6. Consolidating CompiledIlpProgram Impl Blocks

The 2 non-contiguous `impl CompiledIlpProgram` blocks at lib.rs:4412 and lib.rs:5677 merge
into a single `impl` block in `ilp.rs`. The split was an artifact of incremental development.

## 7. Ride-Along Improvements

| Ride-along | Scope |
|------------|-------|
| **Error mapping cleanup** | Replace scattered `.map_err(|e| PyRuntimeError::new_err(...))` with the helper from types.rs. ~80–100 removals. |
| **Visibility** | Smaller than typical — fields are already private. Audit `#[pyo3(get)]` fields for correctness. ~10 changes. |
| **Unwrap fixes** | Fix production-path unwraps in GPU compute paths encountered during extraction. |
| **Stale comments** | Remove host-merging comment at former lib.rs:4689. Update any comments referencing pre-split line numbers. |

## 8. Gate

### Documented release matrix (from v0.4.0-beta-release-design.md)

| Gate | Required |
|------|----------|
| Non-slow batch | Per v0.4.0-beta-release-design.md:52 | Yes |
| ILP reliability | Per v0.4.0-beta-release-design.md:53 | Yes |
| ILP sparse | Per v0.4.0-beta-release-design.md:54 | Yes |
| GA reliability | Per v0.4.0-beta-release-design.md:55 | Yes |
| ILP performance | Per v0.4.0-beta-release-design.md:56 | Yes |

### Supplemental build gates (not part of documented matrix, but required for Wave 4)

| Gate | Command | Required |
|------|---------|----------|
| Rust workspace | `cargo test --workspace --all-targets --exclude pyxlog --release` | Yes |
| CUDA certification | `cargo test -p xlog-cuda-tests --test certification_suite --release` | Yes (206/206) |
| pyxlog compile | `cargo check -p pyxlog` | Yes |
| Python wheel build | `maturin develop --release -m crates/pyxlog/Cargo.toml` | Yes |

This is the first wave with the full documented release matrix. Wave 4 restructures the FFI
boundary, so the wheel build gate is essential even though it's not part of the formal matrix.

## 9. Call-Site Impact

**Internal only.** All changes within the pyxlog crate. No other crate's source changes.
pyxlog is a `cdylib` — no Rust consumer imports from it. Python API verified by gate.

## 10. Diff Profile (estimated)

| Change type | Files | Lines added | Lines removed |
|-------------|-------|-------------|---------------|
| New submodules (7 files) | 7 | ~4,650 | — |
| Slim lib.rs | 1 | ~400 | ~6,202 |
| Error mapping helpers | 1 (types.rs) | ~30 | — |
| .map_err() cleanup | within above | — | ~200 |
| ilp_gpu.rs extraction | 1 (included above) | ~700 | — |
| **Net** | ~8 files | ~5,080 | ~6,402 |

**Net reduction: ~1,320 lines**

## 11. Risks

| Risk | Mitigation |
|------|-----------|
| PyO3 `#[pymethods]` across modules | Well-documented PyO3 pattern. `#[pyclass]` in lib.rs, `#[pymethods]` in submodules. Verified by maturin build + full test matrix. |
| GPU loss decomposition changes numerical behavior | Identical computation order preserved. No algorithm changes. Full Python matrix validates. |
| f32/f64 generic helper type mismatch | `ScalarType` dispatch is runtime-checked. Existing tests cover both precision paths. |
| Missing `#[pymethods]` registration | `maturin develop` + Python import + full test matrix catches missing registrations immediately. |
