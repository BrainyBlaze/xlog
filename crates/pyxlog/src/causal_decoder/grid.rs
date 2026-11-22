use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use xlog_runtime::causal_decoder::grid::{GridAction, GridMemory, GridQuery};

/// Immutable native state of the observed episode at one frame.
#[pyclass(name = "NativeGridMemory", module = "pyxlog._native", frozen)]
pub(crate) struct PyGridMemory {
    pub(super) inner: GridMemory,
}

#[pymethods]
impl PyGridMemory {
    /// Actual number of numerical positions consumed by an action query.
    #[getter]
    fn sequence_rows(&self) -> usize {
        self.inner.sequence_rows()
    }

    /// Accumulated successful substitutions across the frames leading to this memory.
    #[getter]
    fn disabled_calls(&self) -> usize {
        self.inner.disabled_calls
    }
}

pub(super) type Action = BTreeMap<String, usize>;

pub(super) fn action(mut fields: Action) -> PyResult<GridAction> {
    let id = fields
        .remove("id")
        .ok_or_else(|| PyValueError::new_err("missing grid action id"))?;
    let click = match (fields.remove("x"), fields.remove("y")) {
        (None, None) => None,
        (Some(x), Some(y)) => Some((x, y)),
        _ => return Err(PyValueError::new_err("a click requires x and y")),
    };
    if !fields.is_empty() {
        return Err(PyValueError::new_err("unknown grid action field"));
    }
    Ok(GridAction { id, click })
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(super) struct Query {
    available_actions: Vec<usize>,
    action: Action,
    target: (usize, usize),
    attention_patch: (usize, usize),
}

impl Query {
    pub(super) fn into_query(self) -> PyResult<GridQuery> {
        Ok(GridQuery {
            available_actions: self.available_actions,
            action: action(self.action)?,
            target: self.target,
            attention_patch: self.attention_patch,
        })
    }
}
