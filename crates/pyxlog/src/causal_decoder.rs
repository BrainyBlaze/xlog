//! Public native decoder ownership; no external model owns the numerical work.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "host-io")]
use pyo3::types::{PyBytes, PyDict};
use xlog_runtime::causal_decoder::{NativeCausalModel, NativeLogits};

use super::{dlpack_capsule_from_tensor, types::xlog_err};

mod acquisition;
pub(crate) use acquisition::PyNativeAcquisitionMask;

/// Native BF16 causal decoder imported from a local SafeTensors checkpoint.
#[pyclass(name = "NativeCausalModel", module = "pyxlog._native")]
pub(crate) struct PyNativeCausalModel {
    inner: NativeCausalModel,
}

#[pymethods]
impl PyNativeCausalModel {
    /// Save a cold native state and application bytes; returns its SHA256 identity.
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (path, *, application_state))]
    fn save_snapshot(
        &self,
        py: Python<'_>,
        path: PathBuf,
        application_state: Vec<u8>,
    ) -> PyResult<String> {
        py.detach(|| self.inner.save_snapshot(&path, application_state))
            .map(|identity| identity.iter().map(|byte| format!("{byte:02x}")).collect())
            .map_err(xlog_err)
    }

    /// Restore a self-contained state and its application bytes after digest validation.
    #[cfg(feature = "host-io")]
    #[staticmethod]
    #[pyo3(signature = (path, *, snapshot_identity, device, memory_budget_bytes))]
    fn from_snapshot(
        py: Python<'_>,
        path: PathBuf,
        snapshot_identity: &str,
        device: usize,
        memory_budget_bytes: u64,
    ) -> PyResult<(Self, Py<PyBytes>)> {
        if snapshot_identity.len() != 64
            || !snapshot_identity
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(PyValueError::new_err(
                "snapshot_identity must be a 64-digit SHA256",
            ));
        }
        let mut identity = [0u8; 32];
        for (id, byte) in identity.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&snapshot_identity[id * 2..id * 2 + 2], 16)
                .map_err(|_| PyValueError::new_err("invalid snapshot SHA256"))?;
        }
        let (inner, application_state) = py
            .detach(|| {
                NativeCausalModel::from_snapshot(&path, identity, device, memory_budget_bytes)
            })
            .map_err(xlog_err)?;
        Ok((
            Self { inner },
            PyBytes::new(py, &application_state).unbind(),
        ))
    }

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
    #[pyo3(signature = (token_ids, *, prediction_position, history_span=None))]
    fn forward(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        prediction_position: usize,
        history_span: Option<(usize, usize)>,
    ) -> PyResult<PyNativeLogits> {
        py.detach(|| {
            self.inner
                .forward(&token_ids, prediction_position, history_span)
        })
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

    /// Exact continuation retaining definitions, trainable roles and all Adam states.
    fn fork_candidate(&self) -> Self {
        Self {
            inner: self.inner.fork_candidate(),
        }
    }

    /// Contextual wire schema and ABI from this model's actual function registry.
    #[pyo3(signature = (*, insertion_layer=17, language_version=1, raw_token_ids=None))]
    fn acquisition_language(
        &self,
        insertion_layer: usize,
        language_version: usize,
        raw_token_ids: Option<Vec<u32>>,
    ) -> PyResult<String> {
        let context = if matches!(language_version, 12..=27) {
            self.inner
                .acquisition_context_with_output_dependencies(insertion_layer)
        } else {
            self.inner.acquisition_context(insertion_layer)
        }
        .map_err(xlog_err)?;
        acquisition::language(context, language_version, raw_token_ids)
    }

    /// Construct an acquired candidate from actual canonical definitions and FP32 arrays.
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (definition_json, parameter_values, *, insertion_layer=17, train_down_projection=true, source_encoding="funcdef-json-v1", require_output_dependencies=false))]
    fn fork_acquired(
        &self,
        py: Python<'_>,
        definition_json: String,
        parameter_values: std::collections::BTreeMap<String, Vec<f32>>,
        insertion_layer: usize,
        train_down_projection: bool,
        source_encoding: &str,
        require_output_dependencies: bool,
    ) -> PyResult<Self> {
        py.detach(|| match source_encoding {
            "funcdef-json-v1" => self
                .inner
                .fork_acquired_with_dependencies(
                    &definition_json,
                    parameter_values,
                    insertion_layer,
                    train_down_projection,
                    require_output_dependencies,
                )
                .map_err(xlog_err),
            acquisition::NUMERICAL_ROWS_ENCODING
            | acquisition::CONTEXT_ENTRY_ROWS_ENCODING
            | acquisition::INTENT_ROWS_ENCODING
            | acquisition::COMPACT_INDICES_ROWS_ENCODING
            | acquisition::COMPACT_METADATA_ROWS_ENCODING
            | acquisition::SELF_DELIMITED_ROWS_ENCODING
            | acquisition::NAMED_BODY_ROWS_ENCODING
            | acquisition::NUMERICAL_EXPRESSIONS_ENCODING => {
                let expressions = source_encoding == acquisition::NUMERICAL_EXPRESSIONS_ENCODING;
                let named_body =
                    source_encoding == acquisition::NAMED_BODY_ROWS_ENCODING || expressions;
                let self_delimited =
                    source_encoding == acquisition::SELF_DELIMITED_ROWS_ENCODING || named_body;
                let compact_metadata = source_encoding
                    == acquisition::COMPACT_METADATA_ROWS_ENCODING
                    || self_delimited;
                let compact_indices = source_encoding == acquisition::COMPACT_INDICES_ROWS_ENCODING
                    || compact_metadata;
                let context = if compact_indices {
                    self.inner
                        .acquisition_context_with_output_dependencies(insertion_layer)
                } else {
                    self.inner.acquisition_context(insertion_layer)
                }
                .map_err(xlog_err)?;
                let (entrypoint, definitions) = acquisition::decode_numerical_rows(
                    definition_json.as_bytes(),
                    &context,
                    source_encoding != acquisition::NUMERICAL_ROWS_ENCODING,
                    source_encoding == acquisition::INTENT_ROWS_ENCODING || compact_indices,
                    compact_indices,
                    compact_metadata,
                    self_delimited,
                    named_body,
                    expressions,
                )?;
                self.inner
                    .fork_acquired_definitions_with_dependencies(
                        entrypoint,
                        definitions,
                        parameter_values,
                        insertion_layer,
                        train_down_projection,
                        require_output_dependencies,
                    )
                    .map_err(xlog_err)
            }
            _ => Err(PyValueError::new_err(
                "unsupported acquired source encoding",
            )),
        })
        .map(|inner| Self { inner })
    }

    /// Versioned JSON with actual insertion, parameter and optimizer roles.
    #[getter]
    fn adaptation_manifest(&self) -> PyResult<String> {
        self.inner.adaptation_manifest().map_err(xlog_err)
    }

    /// Canonical JSON definitions available to subsequent acquisition proposals.
    #[getter]
    fn function_definitions(&self) -> PyResult<String> {
        self.inner.function_definitions().map_err(xlog_err)
    }

    /// One binary NLL minibatch update. Returns (mean_loss, gradient_norm, step).
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (token_ids, *, prediction_positions, outcomes, negative_token_id, positive_token_id, history_spans=None))]
    fn fit_binary(
        &mut self,
        py: Python<'_>,
        token_ids: Vec<Vec<u32>>,
        prediction_positions: Vec<usize>,
        outcomes: Vec<bool>,
        negative_token_id: u32,
        positive_token_id: u32,
        history_spans: Option<Vec<(usize, usize)>>,
    ) -> PyResult<(f32, f32, u64)> {
        py.detach(|| {
            self.inner.fit_binary(
                &token_ids,
                &prediction_positions,
                &outcomes,
                negative_token_id,
                positive_token_id,
                history_spans.as_deref(),
            )
        })
        .map(|result| (result.mean_loss, result.gradient_norm, result.step))
        .map_err(xlog_err)
    }

    /// One complete-vocabulary CE update. Returns (mean_loss, gradient_norm, step).
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (token_ids, *, prediction_positions, target_ids, history_spans=None))]
    fn fit_tokens(
        &mut self,
        py: Python<'_>,
        token_ids: Vec<Vec<u32>>,
        prediction_positions: Vec<usize>,
        target_ids: Vec<u32>,
        history_spans: Option<Vec<(usize, usize)>>,
    ) -> PyResult<(f32, f32, u64)> {
        py.detach(|| {
            self.inner.fit_tokens(
                &token_ids,
                &prediction_positions,
                &target_ids,
                history_spans.as_deref(),
            )
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
    fn optimizer_steps(&self) -> std::collections::BTreeMap<String, u64> {
        self.inner.optimizer_steps()
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

#[cfg(feature = "host-io")]
fn selection_record(
    py: Python<'_>,
    result: xlog_cuda::tensor::LogitSelection,
) -> PyResult<Py<PyDict>> {
    let selected = PyDict::new(py);
    selected.set_item("raw_argmax", result.raw_argmax)?;
    selected.set_item("raw_score", result.raw_score)?;
    selected.set_item("selected", result.selected)?;
    selected.set_item("selected_score", result.selected_score)?;
    selected.set_item(
        "logits_sha256",
        result
            .logits_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
    )?;
    Ok(selected.unbind())
}

#[pymethods]
impl PyNativeLogits {
    /// Read specified token IDs to the host, in exactly the requested order.
    #[cfg(feature = "host-io")]
    fn read(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.read(&token_ids)).map_err(xlog_err)
    }

    /// Select on CUDA without materializing the vocabulary as Python floats.
    /// Hashing still reads the original full row through tracked host I/O.
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (allowed_mask=None))]
    fn select_greedy(&self, py: Python<'_>, allowed_mask: Option<Vec<u8>>) -> PyResult<Py<PyDict>> {
        let result = py
            .detach(|| self.inner.select_greedy(allowed_mask.as_deref()))
            .map_err(xlog_err)?;
        selection_record(py, result)
    }

    /// Draw on CUDA at temperature one using an explicit Philox seed/address.
    #[cfg(feature = "host-io")]
    #[pyo3(signature = (allowed_mask=None, *, seed, sequence, position))]
    fn select_categorical(
        &self,
        py: Python<'_>,
        allowed_mask: Option<Vec<u8>>,
        seed: u64,
        sequence: u64,
        position: u32,
    ) -> PyResult<Py<PyDict>> {
        let result = py
            .detach(|| {
                self.inner
                    .select_categorical(allowed_mask.as_deref(), seed, sequence, position)
            })
            .map_err(xlog_err)?;
        selection_record(py, result)
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
