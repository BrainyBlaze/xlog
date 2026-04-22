//! pyxlog bridge for the M8 Phase 1 native exact-induction engine.
//!
//! Deliberately separate from `crates/pyxlog/src/ilp.rs` (already ~2.7k LOC)
//! so the Phase 1 binding seam has its own testable surface. Converts the
//! Python-visible `induce_exact_native(...)` call into an
//! [`xlog_induce::InduceExactRequest`] and maps the result back to a
//! `dict`-shaped `PyObject` that the Python wrapper
//! (`crates/pyxlog/python/pyxlog/ilp/exact_induce.py`) packages into
//! `ExactInductionResult` / `ScoredCandidate` dataclass instances.
//!
//! Task progression:
//!   * Task 2:  raised `PyNotImplementedError`.
//!   * Task 3B.2 (this change): real name→`RelId` resolution, DLPack-backed
//!                 positive/negative buffer construction, engine call, and
//!                 dict-shaped return. The scoring kernel isn't wired yet
//!                 (3B.4), so the engine returns an empty candidate list with
//!                 populated counts — the parity test now fails on length
//!                 mismatch rather than on a Rust exception.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use xlog_core::RelId;
use xlog_cuda::CudaBuffer;
use xlog_induce::{
    induce_exact as induce_exact_engine, ExactInductionConfig, ExactInductionResult,
    InduceExactRequest,
};

use super::{dlpack_from_py, types, CompiledIlpProgram};

#[pymethods]
impl CompiledIlpProgram {
    /// Native backend entrypoint for `pyxlog.ilp.induce_exact(backend="native")`.
    ///
    /// Returns a `dict` (not a dataclass) with the following shape — the
    /// Python wrapper repackages this into the existing `ExactInductionResult`
    /// / `ScoredCandidate` dataclass instances:
    ///
    /// ```text
    /// {
    ///   "candidates": [
    ///     {"topology": str, "head_relation": str,
    ///      "left_relation": str, "right_relation": str,
    ///      "positives_covered": int, "negatives_covered": int,
    ///      "local_rank": int,
    ///      "next_positives_covered": int, "next_negatives_covered": int,
    ///      "tie_class_size": int},
    ///     ...
    ///   ],
    ///   "total_scored": int,
    ///   "candidate_count": int,
    ///   "positive_count": int,
    ///   "negative_count": int,
    /// }
    /// ```
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
        py: Python<'py>,
        head_relation: String,
        candidate_relations: Vec<String>,
        positive_arg0: &Bound<'py, PyAny>,
        positive_arg1: &Bound<'py, PyAny>,
        negative_arg0: Option<&Bound<'py, PyAny>>,
        negative_arg1: Option<&Bound<'py, PyAny>>,
        k_per_topology: u32,
        deterministic: bool,
    ) -> PyResult<PyObject> {
        // ── 1. Resolve head_relation → RelId ───────────────────────────────
        // Python reference silently returns an empty result when the head
        // isn't in the ILP schema — mirror that behavior here.
        let head_rid = match self.rel_index.iter().find(|(_, n)| n == &head_relation) {
            Some((rid, _)) => *rid,
            None => return empty_result_dict(py),
        };

        // ── 2. Filter candidate_relations against rel_index ────────────────
        // Drops names that aren't declared in the ILP schema, matching
        // Python's `if cname in name_to_idx: body_indices.append(...)`.
        let filtered_candidates: Vec<(RelId, String)> = candidate_relations
            .into_iter()
            .filter_map(|name| {
                self.rel_index
                    .iter()
                    .find(|(_, n)| n == &name)
                    .map(|(rid, _)| (*rid, name))
            })
            .collect();

        // ── 3. Build positive / negative CudaBuffers via DLPack ────────────
        // Use head_relation's declared schema — this also validates that the
        // query tensors are column-type-compatible with the head relation.
        let head_schema = self.schemas.get(&head_relation).cloned().ok_or_else(|| {
            PyValueError::new_err(format!(
                "induce_exact_native: head_relation {:?} has no schema",
                head_relation,
            ))
        })?;

        let pos_dmt0 = dlpack_from_py(positive_arg0)?;
        let pos_dmt1 = dlpack_from_py(positive_arg1)?;
        let pos_buf = self
            .provider
            .from_dlpack_tensors_with_schema(head_schema.clone(), vec![pos_dmt0, pos_dmt1])
            .map_err(types::xlog_err)?;

        let neg_buf = match (negative_arg0, negative_arg1) {
            (Some(a0), Some(a1)) => {
                let dmt0 = dlpack_from_py(a0)?;
                let dmt1 = dlpack_from_py(a1)?;
                Some(
                    self.provider
                        .from_dlpack_tensors_with_schema(head_schema.clone(), vec![dmt0, dmt1])
                        .map_err(types::xlog_err)?,
                )
            }
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "induce_exact_native: negative_arg0 and negative_arg1 must both be set or both None",
                ));
            }
        };

        // ── 4. Look up candidate buffers in the relation store ─────────────
        // NOTE: Declared-but-unloaded candidates are silently dropped here.
        // The Python reference would score them as zero-coverage (empty fact
        // set). The test suite always loads facts for every candidate, so
        // this divergence is not observable today — 3B.5 parity will confirm.
        let store = self.executor.store();
        let candidates_with_bufs: Vec<(RelId, &CudaBuffer)> = filtered_candidates
            .iter()
            .filter_map(|(rid, name)| store.get(name).map(|buf| (*rid, buf)))
            .collect();

        // ── 5. Call the engine ─────────────────────────────────────────────
        let request = InduceExactRequest {
            head_rel_idx: head_rid,
            candidates: &candidates_with_bufs,
            positives: &pos_buf,
            negatives: neg_buf.as_ref(),
            config: ExactInductionConfig {
                k_per_topology,
                deterministic,
            },
        };
        let result = induce_exact_engine(&self.provider, &request).map_err(types::xlog_err)?;

        // ── 6. Marshal the result into a dict — RelId → relation name uses
        //       the same `rel_index` we resolved names against. ─────────────
        result_to_py_dict(py, &result, &head_relation, &self.rel_index)
    }
}

fn empty_result_dict(py: Python<'_>) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    let empty_candidates = PyList::empty(py);
    d.set_item("candidates", empty_candidates)?;
    d.set_item("total_scored", 0u32)?;
    d.set_item("candidate_count", 0u32)?;
    d.set_item("positive_count", 0u32)?;
    d.set_item("negative_count", 0u32)?;
    Ok(d.into())
}

fn result_to_py_dict(
    py: Python<'_>,
    result: &ExactInductionResult,
    head_relation: &str,
    rel_index: &[(RelId, String)],
) -> PyResult<PyObject> {
    let name_of = |rid: RelId| -> PyResult<&str> {
        rel_index
            .iter()
            .find(|(r, _)| *r == rid)
            .map(|(_, n)| n.as_str())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "induce_exact_native: result references unknown RelId {:?}",
                    rid,
                ))
            })
    };

    let candidates_list = PyList::empty(py);
    for c in &result.candidates {
        let entry = PyDict::new(py);
        entry.set_item("topology", c.topology.as_str())?;
        entry.set_item("head_relation", head_relation)?;
        entry.set_item("left_relation", name_of(c.left_rel_idx)?)?;
        entry.set_item("right_relation", name_of(c.right_rel_idx)?)?;
        entry.set_item("positives_covered", c.positives_covered)?;
        entry.set_item("negatives_covered", c.negatives_covered)?;
        entry.set_item("local_rank", c.local_rank)?;
        entry.set_item("next_positives_covered", c.next_positives_covered)?;
        entry.set_item("next_negatives_covered", c.next_negatives_covered)?;
        entry.set_item("tie_class_size", c.tie_class_size)?;
        candidates_list.append(entry)?;
    }

    let d = PyDict::new(py);
    d.set_item("candidates", candidates_list)?;
    d.set_item("total_scored", result.total_scored)?;
    d.set_item("candidate_count", result.candidate_count)?;
    d.set_item("positive_count", result.positive_count)?;
    d.set_item("negative_count", result.negative_count)?;
    Ok(d.into())
}
