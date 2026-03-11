// DLPack / Arrow C Device Array #[pyfunction] bindings.
//
// Contains the higher-level pyfunction wrappers that call the FFI primitives
// defined in lib.rs (dlpack_from_py, dlpack_capsule_from_tensor, etc.).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "arrow-device-import")]
use pyo3::types::PySequence;

use xlog_prob::exact::GpuConfig;

#[cfg(feature = "arrow-device-import")]
use super::{arrow_device_capsule_from_device_array, arrow_device_from_py};
use super::{dlpack_capsule_from_tensor, dlpack_from_py, provider_from_config, types};

/// Export one (struct) Arrow C Device array from one or more DLPack columns (zero-copy).
///
/// EXPERIMENTAL: Requires building `pyxlog` with `--features arrow-device-import`.
#[cfg(feature = "arrow-device-import")]
#[pyfunction]
#[pyo3(signature = (dlpack_columns, device=0, memory_mb=32768))]
pub(crate) fn export_arrow_device(
    py: Python<'_>,
    dlpack_columns: &Bound<'_, PyAny>,
    device: usize,
    memory_mb: u64,
) -> PyResult<PyObject> {
    if memory_mb == 0 {
        return Err(PyValueError::new_err("memory_mb must be > 0"));
    }

    let config = GpuConfig {
        device_ordinal: device,
        memory_bytes: memory_mb * 1024 * 1024,
    };
    let provider = provider_from_config(config).map_err(types::xlog_err)?;

    let tensors = match dlpack_columns.downcast::<PySequence>() {
        Ok(seq) => {
            let mut out = Vec::with_capacity(seq.len()? as usize);
            for item in seq.iter()? {
                out.push(dlpack_from_py(&item?)?);
            }
            out
        }
        Err(_) => vec![dlpack_from_py(dlpack_columns)?],
    };

    let buffer = provider
        .from_dlpack_tensors(tensors)
        .map_err(types::xlog_err)?;
    let device_array = provider
        .to_arrow_device_record_batch(buffer)
        .map_err(types::xlog_err)?;
    arrow_device_capsule_from_device_array(py, device_array)
}

/// Import an Arrow C Device array (device-resident) into DLPack columns (zero-copy).
///
/// EXPERIMENTAL: Requires building `pyxlog` with `--features arrow-device-import`.
#[cfg(feature = "arrow-device-import")]
#[pyfunction]
#[pyo3(signature = (device_array, device=0, memory_mb=32768))]
pub(crate) fn import_arrow_device(
    py: Python<'_>,
    device_array: &Bound<'_, PyAny>,
    device: usize,
    memory_mb: u64,
) -> PyResult<(Vec<PyObject>, Vec<String>, usize)> {
    if memory_mb == 0 {
        return Err(PyValueError::new_err("memory_mb must be > 0"));
    }

    let config = GpuConfig {
        device_ordinal: device,
        memory_bytes: memory_mb * 1024 * 1024,
    };
    let provider = provider_from_config(config).map_err(types::xlog_err)?;

    let device_array = arrow_device_from_py(device_array)?;

    let buffer = provider
        .from_arrow_device_record_batch(device_array)
        .map_err(types::xlog_err)?;

    let names: Vec<String> = buffer
        .schema
        .columns
        .iter()
        .map(|(n, _)| n.clone())
        .collect();
    let num_rows = buffer.num_rows() as usize;
    let arity = buffer.arity();

    let table = provider.to_dlpack_table(buffer);
    let mut tensors: Vec<PyObject> = Vec::with_capacity(arity);
    for col_idx in 0..arity {
        let tensor = table.column(col_idx).map_err(types::xlog_err)?;
        tensors.push(dlpack_capsule_from_tensor(py, tensor)?);
    }

    Ok((tensors, names, num_rows))
}

#[pyfunction]
pub(crate) fn dlpack_roundtrip(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
    device: usize,
    memory_mb: u64,
) -> PyResult<PyObject> {
    if memory_mb == 0 {
        return Err(PyValueError::new_err("memory_mb must be > 0"));
    }
    let config = GpuConfig {
        device_ordinal: device,
        memory_bytes: memory_mb * 1024 * 1024,
    };
    let provider = provider_from_config(config).map_err(types::xlog_err)?;
    let managed = dlpack_from_py(tensor)?;
    let buffer = provider
        .from_dlpack_tensors(vec![managed])
        .map_err(types::xlog_err)?;
    let out = provider
        .to_dlpack_table(buffer)
        .column(0)
        .map_err(types::xlog_err)?;
    dlpack_capsule_from_tensor(py, out)
}
