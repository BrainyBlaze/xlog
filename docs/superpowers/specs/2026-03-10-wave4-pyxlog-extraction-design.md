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
| `logic.rs` | `Program` #[pymethods] (lib.rs:454), `LogicProgram` #[pymethods] (lib.rs:3572), `CompiledLogicProgram` #[pymethods] (lib.rs:3604). Getter-only #[pyclass] data types (no #[pymethods]): `LogicQueryResult` (lib.rs:3694), `LogicEvalResult` (lib.rs:3709), `McDeviceEvalResult` (lib.rs:3714), `EvalResult` (lib.rs:3740). | ~600 |
| `ilp.rs` | `CompiledIlpProgram` `#[pymethods]` (lib.rs:4412–5675, single block). `IlpProgramFactory` #[pymethods] (lib.rs:4302) with `compile`. The second impl block at lib.rs:5677 is a **plain Rust impl** (NOT #[pymethods]) containing private GPU export helpers (`build_zero_loss_grad`, `export_loss_grad_device_f32/f64`). Both blocks consolidate into `ilp.rs`. Delegates GPU compute to `ilp_gpu.rs`. | ~1,000 |
| `ilp_gpu.rs` | GPU loss/gradient computation extracted from ILP: chunked/non-chunked paths, COO merge, sparse reduction, evidence clamping setup | ~700 |
| `training.rs` | Getter-only #[pyclass] data types: `EpochStats` (lib.rs:3789), `TrainingHistory` (lib.rs:3804). TrainingHistory has a plain `impl` (lib.rs:3819, NOT #[pymethods]) with private helpers `new()`, `add_epoch()`, `add_batch()`. | ~400 |
| `neural.rs` | Network registration, embedding registration/forward, tensor source management, generic f32/f64 forward-backward helper | ~500 |
| `types.rs` | `scalar_type_name()`, error mapping helpers, value conversion utilities | ~250 |

**Total**: ~5,050 LOC (vs 6,202 current — ~19% reduction)

## 2. Frozen Python API Surface (Accurate Inventory)

The frozen surface includes both native `#[pyclass]`/`#[pyfunction]` exports AND the
pure-Python package surface.

### Native Rust exports (#[pyclass] / #[pyfunction])

| PyClass / Export | Defined at | Target module | Notes |
|-----------------|------------|---------------|-------|
| `Program` | lib.rs:451 | logic.rs | Constructor, compile |
| `CompiledProgram` | lib.rs:654+ | program.rs | Full method surface (evaluate, evaluate_device, tensor-source, training, forward/backward, cache, query-probability) |
| `LogicProgram` | lib.rs:3570 | logic.rs | Constructor, compile |
| `CompiledLogicProgram` | lib.rs:3599 | logic.rs | Evaluation methods |
| `LogicQueryResult` | lib.rs:3694 | logic.rs | Getter-only data class |
| `LogicEvalResult` | lib.rs:3709 | logic.rs | Getter-only data class |
| `McDeviceEvalResult` | lib.rs:3714 | logic.rs | Getter-only data class |
| `EvalResult` | lib.rs:3740 | logic.rs | Getter-only data class |
| `EpochStats` | lib.rs:3787 | training.rs | Getter-only data class |
| `TrainingHistory` | lib.rs:3802 | training.rs | Training recording. Note: `add_batch` (lib.rs:3819) is a private Rust helper, NOT part of the Python surface. |
| `IlpProgramFactory` | lib.rs:4299 | ilp.rs | `compile` (lib.rs:4299) |
| `CompiledIlpProgram` | lib.rs:4412+ | ilp.rs | Full runtime/control surface across both impl blocks |
| Module functions | lib.rs:6176 | lib.rs | All `#[pyfunction]` exports in `#[pymodule]` (stays in lib.rs for registration) |

### Pure-Python package surface

| File | Content | Constraint |
|------|---------|-----------|
| `python/pyxlog/__init__.py` | Package imports, public API re-exports | Frozen — user-facing import paths |
| `python/pyxlog/ilp/__init__.py` | ILP subpackage imports | Frozen — user-facing import paths |

Changes to these files require the same care as changes to `#[pymethods]` — they define
what Python users can `import`. Verify after refactoring that all existing `from pyxlog import X`
and `from pyxlog.ilp import Y` paths still resolve.

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

| # | Gate | Ref |
|---|------|-----|
| 1 | Rust workspace (`cargo test --workspace --all-targets --exclude pyxlog --release`) | v0.4.0-beta-release-design.md:50 |
| 2 | CUDA certification 206/206 (`cargo test -p xlog-cuda-tests --test certification_suite --release`) | v0.4.0-beta-release-design.md:51 |
| 3 | Non-slow batch | v0.4.0-beta-release-design.md:52 |
| 4 | ILP reliability | v0.4.0-beta-release-design.md:53 |
| 5 | ILP sparse | v0.4.0-beta-release-design.md:54 |
| 6 | GA reliability | v0.4.0-beta-release-design.md:55 |
| 7 | ILP performance | v0.4.0-beta-release-design.md:56 |

### Additional build gates (not in documented matrix, required for Wave 4)

| Gate | Command | Rationale |
|------|---------|-----------|
| pyxlog compile | `cargo check -p pyxlog` | Catch Rust compilation errors before building wheel |
| Python wheel build | `maturin develop --release -m crates/pyxlog/Cargo.toml` | Wave 4 restructures FFI boundary; wheel must build |

This is the first wave with the full documented release matrix (all 7 suites).

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
