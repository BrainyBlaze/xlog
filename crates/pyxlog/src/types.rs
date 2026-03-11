use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::PyAnyMethods;
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
