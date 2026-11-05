//! Public native decoder ownership; no external model owns the numerical work.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use xlog_runtime::causal_decoder::{NativeCausalModel, NativeLogits};

use super::{dlpack_capsule_from_tensor, types::xlog_err};

/// Native BF16 causal decoder imported from a local SafeTensors checkpoint.
#[pyclass(name = "NativeCausalModel", module = "pyxlog._native")]
pub(crate) struct PyNativeCausalModel {
    inner: NativeCausalModel,
}

#[pymethods]
impl PyNativeCausalModel {
    /// Import config.json and model.safetensors or its indexed shards.
    /// The explicit byte budget covers weights, activations and cuBLAS workspace.
    #[staticmethod]
    #[pyo3(signature = (directory, *, device, memory_budget_bytes, max_context))]
    fn from_checkpoint(
        py: Python<'_>,
        directory: PathBuf,
        device: usize,
        memory_budget_bytes: u64,
        max_context: usize,
    ) -> PyResult<Self> {
        py.detach(move || {
            NativeCausalModel::from_checkpoint(&directory, device, memory_budget_bytes, max_context)
        })
        .map(|inner| Self { inner })
        .map_err(xlog_err)
    }

    /// Use original token IDs and a zero-based prediction position. This computes
    /// next-token logits at that position, with full-prefix causal attention.
    #[pyo3(signature = (token_ids, *, prediction_position))]
    fn forward(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        prediction_position: usize,
    ) -> PyResult<PyNativeLogits> {
        py.detach(|| self.inner.forward(&token_ids, prediction_position))
            .map(|inner| PyNativeLogits { inner })
            .map_err(xlog_err)
    }

    /// Fork a candidate; the source model and its retained outputs stay unchanged.
    #[pyo3(signature = (*, down_projection_layer))]
    fn fork_trainable(&self, py: Python<'_>, down_projection_layer: usize) -> PyResult<Self> {
        py.detach(|| self.inner.fork_trainable(down_projection_layer))
            .map(|inner| Self { inner })
            .map_err(xlog_err)
    }

    /// One binary NLL minibatch update. Returns (mean_loss, gradient_norm, step).
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (token_ids, *, prediction_positions, outcomes, negative_token_id, positive_token_id))]
    fn fit_binary(
        &mut self,
        py: Python<'_>,
        token_ids: Vec<Vec<u32>>,
        prediction_positions: Vec<usize>,
        outcomes: Vec<bool>,
        negative_token_id: u32,
        positive_token_id: u32,
    ) -> PyResult<(f32, f32, u64)> {
        py.detach(|| {
            self.inner.fit_binary(
                &token_ids,
                &prediction_positions,
                &outcomes,
                negative_token_id,
                positive_token_id,
            )
        })
        .map(|result| (result.mean_loss, result.gradient_norm, result.step))
        .map_err(xlog_err)
    }

    /// One complete-vocabulary CE update. Returns (mean_loss, gradient_norm, step).
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (token_ids, *, prediction_positions, target_ids))]
    fn fit_tokens(
        &mut self,
        py: Python<'_>,
        token_ids: Vec<Vec<u32>>,
        prediction_positions: Vec<usize>,
        target_ids: Vec<u32>,
    ) -> PyResult<(f32, f32, u64)> {
        py.detach(|| {
            self.inner
                .fit_tokens(&token_ids, &prediction_positions, &target_ids)
        })
        .map(|result| (result.mean_loss, result.gradient_norm, result.step))
        .map_err(xlog_err)
    }

    #[getter]
    fn checkpoint_identity(&self) -> String {
        self.inner
            .checkpoint_identity()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[getter]
    fn max_context(&self) -> usize {
        self.inner.max_context()
    }

    #[getter]
    fn vocabulary_size(&self) -> usize {
        self.inner.vocabulary_size()
    }
}

/// Retained native FP32 logits for one prediction position.
#[pyclass(name = "NativeLogits", module = "pyxlog._native")]
pub(crate) struct PyNativeLogits {
    inner: NativeLogits,
}

#[pymethods]
impl PyNativeLogits {
    /// Read specified token IDs to the host, in exactly the requested order.
    #[cfg(feature = "host-io")]
    fn read(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.read(&token_ids)).map_err(xlog_err)
    }

    #[getter]
    fn prediction_position(&self) -> usize {
        self.inner.prediction_position
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (1, self.inner.vocabulary_size)
    }

    fn __dlpack_device__(&self) -> (i32, usize) {
        (2, self.inner.device)
    }

    /// The legacy export synchronizes recorded producer writes before transfer.
    #[pyo3(signature = (stream=None))]
    fn __dlpack__(&self, py: Python<'_>, stream: Option<i64>) -> PyResult<Py<PyAny>> {
        if stream.is_some_and(|value| value == 0 || value < -1) {
            return Err(PyValueError::new_err("invalid DLPack CUDA stream"));
        }
        let tensor = py.detach(|| self.inner.export()).map_err(xlog_err)?;
        dlpack_capsule_from_tensor(py, tensor)
    }
}
