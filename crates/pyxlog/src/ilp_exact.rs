//! pyxlog bridge for the M8 Phase 1 native exact-induction engine.
//!
//! Deliberately kept separate from `crates/pyxlog/src/ilp.rs` (which is
//! already large) so the Phase 1 binding seam has its own testable surface.
//!
//! Task 2 (this file): defines the pyo3 method signature on
//! `CompiledIlpProgram` and raises `PyNotImplementedError`. This shifts the
//! native-backend failure from the Python dispatcher to the Rust seam so
//! Task 3 can flip it green by wiring to `xlog_induce::induce_exact()`.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

use super::CompiledIlpProgram;

#[pymethods]
impl CompiledIlpProgram {
    /// Native backend entrypoint for `pyxlog.ilp.induce_exact(backend="native")`.
    ///
    /// The Python wrapper in `crates/pyxlog/python/pyxlog/ilp/exact_induce.py`
    /// dispatches here when `backend="native"` and translates the returned
    /// structure back into the `ExactInductionResult` / `ScoredCandidate`
    /// dataclass surface.
    #[pyo3(signature = (
        head_relation,
        candidate_relations,
        positive_arg0,
        positive_arg1,
        negative_arg0 = None,
        negative_arg1 = None,
        k_per_topology = 2,
        deterministic = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn induce_exact_native<'py>(
        &mut self,
        _py: Python<'py>,
        head_relation: String,
        candidate_relations: Vec<String>,
        positive_arg0: &Bound<'py, PyAny>,
        positive_arg1: &Bound<'py, PyAny>,
        negative_arg0: Option<&Bound<'py, PyAny>>,
        negative_arg1: Option<&Bound<'py, PyAny>>,
        k_per_topology: u32,
        deterministic: bool,
    ) -> PyResult<PyObject> {
        // Silence unused warnings while keeping the full signature stable
        // for Task 3 to wire without breaking the pyxlog Python wrapper.
        let _ = (
            head_relation,
            candidate_relations,
            positive_arg0,
            positive_arg1,
            negative_arg0,
            negative_arg1,
            k_per_topology,
            deterministic,
        );
        Err(PyNotImplementedError::new_err(
            "induce_exact_native: xlog-induce engine not implemented yet (M8 Phase 1 Task 3)",
        ))
    }
}
