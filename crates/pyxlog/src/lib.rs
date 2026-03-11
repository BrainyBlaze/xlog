use std::collections::{HashMap, HashSet};
use std::os::raw::{c_char, c_void};
use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;

use ::xlog_gpu::logic as gpu_logic;
use xlog_core::{MemoryBudget, ScalarType, Schema};
#[cfg(feature = "arrow-device-import")]
use xlog_cuda::{ArrowDeviceArray, ArrowDeviceArrayOwned};
use xlog_cuda::{CudaDevice, CudaKernelProvider, DlpackManagedTensor, GpuMemoryManager};
use xlog_logic::ast::ProbEngine;
use xlog_neural::{NetworkRegistry, TensorSourceRegistry};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
#[cfg(feature = "host-io")]
use xlog_prob::exact::{ExactResultWithGrads, QueryProbability};
use xlog_prob::mc::{McEvalConfig, McProgram, McSamplingMethod};
use xlog_prob::neural_fast_path::GpuWeightSlots;

use std::collections::HashMap as StdHashMap;

use xlog_core::RelId;
use xlog_logic::ast::Program as AstProgram;
use xlog_runtime::Executor;
use xlog_ir::ExecutionPlan;

mod neural_registry;
use neural_registry::{NeuralPredicateInfo, NeuralPredicateRegistry};
mod types;
mod training;
mod logic;
mod ilp;
mod ilp_gpu;
mod neural;

const DLPACK_CAPSULE_NAME: &[u8] = b"dltensor\0";
const USED_DLPACK_CAPSULE_NAME: &[u8] = b"used_dltensor\0";

#[cfg(feature = "arrow-device-import")]
const ARROW_DEVICE_ARRAY_CAPSULE_NAME: &[u8] = b"arrow_device_array\0";
#[cfg(feature = "arrow-device-import")]
const USED_ARROW_DEVICE_ARRAY_CAPSULE_NAME: &[u8] = b"used_arrow_device_array\0";

unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if capsule.is_null() {
        return;
    }

    let valid =
        pyo3::ffi::PyCapsule_IsValid(capsule, DLPACK_CAPSULE_NAME.as_ptr() as *const c_char);
    if valid == 0 {
        return;
    }

    let ptr =
        pyo3::ffi::PyCapsule_GetPointer(capsule, DLPACK_CAPSULE_NAME.as_ptr() as *const c_char);
    if ptr.is_null() {
        pyo3::ffi::PyErr_Clear();
        return;
    }

    let managed = ptr as *mut xlog_cuda::DLManagedTensor;
    drop(DlpackManagedTensor::from_raw(managed));
}

pub(crate) fn dlpack_capsule_from_tensor(py: Python<'_>, tensor: DlpackManagedTensor) -> PyResult<PyObject> {
    let raw = tensor.into_raw();
    let ptr = raw as *mut c_void;
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            ptr,
            DLPACK_CAPSULE_NAME.as_ptr() as *const c_char,
            Some(dlpack_capsule_destructor),
        )
    };
    if capsule.is_null() {
        unsafe {
            drop(DlpackManagedTensor::from_raw(raw));
        }
        return Err(PyRuntimeError::new_err("Failed to create DLPack capsule"));
    }
    let obj: Py<PyAny> = unsafe { Py::from_owned_ptr(py, capsule) };
    Ok(obj.into_py(py))
}

#[cfg(feature = "arrow-device-import")]
unsafe extern "C" fn arrow_device_array_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if capsule.is_null() {
        return;
    }

    let valid = pyo3::ffi::PyCapsule_IsValid(
        capsule,
        ARROW_DEVICE_ARRAY_CAPSULE_NAME.as_ptr() as *const c_char,
    );
    if valid == 0 {
        return;
    }

    let ptr = pyo3::ffi::PyCapsule_GetPointer(
        capsule,
        ARROW_DEVICE_ARRAY_CAPSULE_NAME.as_ptr() as *const c_char,
    );
    if ptr.is_null() {
        pyo3::ffi::PyErr_Clear();
        return;
    }

    drop(ArrowDeviceArrayOwned::from_raw(
        ptr as *mut ArrowDeviceArray,
    ));
}

#[cfg(feature = "arrow-device-import")]
fn arrow_device_capsule_from_device_array(
    py: Python<'_>,
    device_array: ArrowDeviceArrayOwned,
) -> PyResult<PyObject> {
    let raw = device_array.into_raw();
    let ptr = raw as *mut c_void;
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            ptr,
            ARROW_DEVICE_ARRAY_CAPSULE_NAME.as_ptr() as *const c_char,
            Some(arrow_device_array_capsule_destructor),
        )
    };
    if capsule.is_null() {
        unsafe {
            drop(ArrowDeviceArrayOwned::from_raw(raw));
        }
        return Err(PyRuntimeError::new_err(
            "Failed to create Arrow device array capsule",
        ));
    }
    let obj: Py<PyAny> = unsafe { Py::from_owned_ptr(py, capsule) };
    Ok(obj.into_py(py))
}

#[cfg(feature = "arrow-device-import")]
fn arrow_device_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ArrowDeviceArrayOwned> {
    if unsafe {
        pyo3::ffi::PyCapsule_IsValid(
            obj.as_ptr(),
            ARROW_DEVICE_ARRAY_CAPSULE_NAME.as_ptr() as *const c_char,
        )
    } == 0
    {
        return Err(PyValueError::new_err(
            "Expected an Arrow device array capsule (arrow_device_array)",
        ));
    }

    let ptr = unsafe {
        pyo3::ffi::PyCapsule_GetPointer(
            obj.as_ptr(),
            ARROW_DEVICE_ARRAY_CAPSULE_NAME.as_ptr() as *const c_char,
        )
    };
    if ptr.is_null() {
        return Err(PyRuntimeError::new_err(
            "Failed to get Arrow device array pointer",
        ));
    }

    // Mark consumed so the capsule destructor doesn't free the pointer we now own.
    let rc = unsafe {
        pyo3::ffi::PyCapsule_SetName(
            obj.as_ptr(),
            USED_ARROW_DEVICE_ARRAY_CAPSULE_NAME.as_ptr() as *const c_char,
        )
    };
    if rc != 0 {
        return Err(PyRuntimeError::new_err(
            "Failed to mark Arrow device array capsule as consumed",
        ));
    }

    Ok(unsafe { ArrowDeviceArrayOwned::from_raw(ptr as *mut ArrowDeviceArray) })
}

#[cfg(feature = "host-io")]
fn atom_to_string(atom: &xlog_prob::provenance::GroundAtom) -> String {
    use xlog_prob::provenance::Value;

    if atom.args.is_empty() {
        return format!("{}()", atom.predicate);
    }

    let mut s = String::new();
    s.push_str(&atom.predicate);
    s.push('(');
    for (i, arg) in atom.args.iter().enumerate() {
        if i != 0 {
            s.push_str(", ");
        }
        match arg {
            Value::I64(v) => s.push_str(&v.to_string()),
            Value::F64(bits) => s.push_str(&f64::from_bits(*bits).to_string()),
            Value::Symbol(sym) => s.push_str(&format!("sym#{}", sym)),
            Value::String(v) => s.push_str(v),
        }
    }
    s.push(')');
    s
}

pub(crate) fn provider_from_config(config: GpuConfig) -> xlog_core::Result<CudaKernelProvider> {
    let device = Arc::new(CudaDevice::new(config.device_ordinal)?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(config.memory_bytes),
    ));
    CudaKernelProvider::new(device, memory)
}

pub(crate) fn parse_prob_engine_override(s: &str) -> PyResult<ProbEngine> {
    let v = s.trim().to_ascii_lowercase();
    match v.as_str() {
        "exact_ddnnf" | "exact" | "ddnnf" => Ok(ProbEngine::ExactDdnnf),
        "mc" => Ok(ProbEngine::Mc),
        other => Err(PyValueError::new_err(format!(
            "Unknown prob_engine '{}'; expected 'exact_ddnnf' or 'mc'",
            other
        ))),
    }
}

pub(crate) fn dlpack_from_py(obj: &Bound<'_, PyAny>) -> PyResult<DlpackManagedTensor> {
    let py = obj.py();

    let capsule_obj: Bound<'_, PyAny> = if unsafe {
        pyo3::ffi::PyCapsule_IsValid(obj.as_ptr(), DLPACK_CAPSULE_NAME.as_ptr() as *const c_char)
    } != 0
    {
        obj.clone()
    } else if obj.hasattr("__dlpack__")? {
        match obj.call_method0("__dlpack__") {
            Ok(v) => v,
            Err(err) => {
                if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    obj.call_method1("__dlpack__", (py.None(),))?
                } else {
                    return Err(err);
                }
            }
        }
    } else {
        return Err(PyValueError::new_err(
            "Expected a DLPack capsule or an object with __dlpack__",
        ));
    };

    if unsafe {
        pyo3::ffi::PyCapsule_IsValid(
            capsule_obj.as_ptr(),
            DLPACK_CAPSULE_NAME.as_ptr() as *const c_char,
        )
    } == 0
    {
        return Err(PyValueError::new_err("Invalid DLPack capsule"));
    }

    let ptr = unsafe {
        pyo3::ffi::PyCapsule_GetPointer(
            capsule_obj.as_ptr(),
            DLPACK_CAPSULE_NAME.as_ptr() as *const c_char,
        )
    };
    if ptr.is_null() {
        return Err(PyRuntimeError::new_err("Failed to get DLPack pointer"));
    }

    let rc = unsafe {
        pyo3::ffi::PyCapsule_SetName(
            capsule_obj.as_ptr(),
            USED_DLPACK_CAPSULE_NAME.as_ptr() as *const c_char,
        )
    };
    if rc != 0 {
        return Err(PyRuntimeError::new_err(
            "Failed to mark DLPack capsule as consumed",
        ));
    }

    Ok(unsafe { DlpackManagedTensor::from_raw(ptr as *mut xlog_cuda::DLManagedTensor) })
}

/// Export one (struct) Arrow C Device array from one or more DLPack columns (zero-copy).
///
/// EXPERIMENTAL: Requires building `pyxlog` with `--features arrow-device-import`.
#[cfg(feature = "arrow-device-import")]
#[pyfunction]
#[pyo3(signature = (dlpack_columns, device=0, memory_mb=32768))]
fn export_arrow_device(
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
    let provider =
        provider_from_config(config).map_err(types::xlog_err)?;

    let tensors: Vec<DlpackManagedTensor> = match dlpack_columns.downcast::<PySequence>() {
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
fn import_arrow_device(
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
    let provider =
        provider_from_config(config).map_err(types::xlog_err)?;

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
        let tensor = table
            .column(col_idx)
            .map_err(types::xlog_err)?;
        tensors.push(dlpack_capsule_from_tensor(py, tensor)?);
    }

    Ok((tensors, names, num_rows))
}

#[pyfunction]
fn dlpack_roundtrip(
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
    let provider =
        provider_from_config(config).map_err(types::xlog_err)?;
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

#[pyclass]
pub struct Program;


/// A cached circuit for a specific query template.
///
/// The circuit structure is immutable - only weights change between queries.
/// Weight slots map network outputs to circuit variables.
pub(crate) struct CachedCircuit {
    /// The compiled program containing the GPU circuit
    program: ExactDdnnfProgram,

    /// Device-resident mapping from neural output slots to CNF variable ids.
    slots: GpuWeightSlots,

    /// Ordered target domain for Targeted signatures. Empty for Boolean.
    target_domain: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum InputSource {
    QueryArg(usize),
    ImplicitSlot(usize),
}

#[derive(Debug, Clone)]
pub(crate) struct NeuralGroup {
    info: NeuralPredicateInfo,
    input_source: InputSource,
    #[cfg(feature = "host-io")]
    output_var: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) enum QuerySignature {
    Boolean {
        groups: Vec<NeuralGroup>,
    },
    Targeted {
        target_position: usize,
        groups: Vec<NeuralGroup>,
    },
}

impl QuerySignature {
    fn groups(&self) -> &[NeuralGroup] {
        match self {
            QuerySignature::Boolean { groups } | QuerySignature::Targeted { groups, .. } => groups,
        }
    }
}

pub(crate) enum CompiledProbProgram {
    Exact(ExactDdnnfProgram),
    Mc(McProgram),
}

impl CompiledProbProgram {
    #[cfg(feature = "host-io")]
    fn num_vars(&self) -> usize {
        match self {
            Self::Exact(p) => p.num_vars(),
            Self::Mc(p) => p.num_vars(),
        }
    }
}

#[pyclass]
pub struct CompiledProgram {
    pub(crate) program: CompiledProbProgram,
    pub(crate) output_provider: Arc<CudaKernelProvider>,
    /// Registry for neural networks
    pub(crate) network_registry: NetworkRegistry,
    /// Registry for neural predicate metadata (predicate -> network/labels)
    pub(crate) neural_registry: NeuralPredicateRegistry,
    /// Names of neural networks declared in the program (from nn() declarations)
    pub(crate) declared_networks: HashSet<String>,
    /// Map from network name to form: true = embedding, false = classification
    pub(crate) declared_network_forms: HashMap<String, bool>,
    /// Registry for tensor data sources (images, embeddings, etc.)
    pub(crate) tensor_sources: TensorSourceRegistry,
    /// Original program source (for dynamic query compilation)
    pub(crate) _source: String,
    /// Parsed program AST (for signature analysis)
    pub(crate) ast: xlog_logic::ast::Program,
    /// GPU configuration
    pub(crate) _gpu_config: GpuConfig,
    /// Probabilistic inference engine
    pub(crate) _prob_engine: ProbEngine,
    /// Cache of analyzed query signatures.
    pub(crate) query_signature_cache: StdHashMap<String, QuerySignature>,
    /// Cache of compiled circuits by template signature
    pub(crate) circuit_cache: StdHashMap<String, CachedCircuit>,
    /// Number of times the template compilation path executed.
    pub(crate) template_compile_count: usize,
    /// When true, batch queries sharing the same circuit template in training.
    pub(crate) batch_queries: bool,
    /// Latest circuit compilation profile (populated on cache miss when profiling).
    pub(crate) last_compile_profile: Option<xlog_prob::compilation::CircuitCompileProfile>,
}

impl CompiledProgram {
    fn parse_sampling_method(s: Option<String>) -> PyResult<Option<McSamplingMethod>> {
        match s.as_deref() {
            None => Ok(None),
            Some("rejection") => Ok(Some(McSamplingMethod::Rejection)),
            Some("evidence_clamping") => Ok(Some(McSamplingMethod::EvidenceClamping)),
            Some(other) => Err(PyValueError::new_err(format!(
                "Unknown sampling_method '{}'. Use 'rejection' or 'evidence_clamping'.",
                other
            ))),
        }
    }
}

#[pymethods]
impl CompiledProgram {
    #[pyo3(signature = (return_grads=false, samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024, sampling_method=None))]
    pub fn evaluate(
        &self,
        _py: Python<'_>,
        return_grads: bool,
        samples: Option<usize>,
        seed: Option<u64>,
        confidence: f64,
        max_nonmonotone_iterations: usize,
        sampling_method: Option<String>,
    ) -> PyResult<EvalResult> {
        match &self.program {
            CompiledProbProgram::Exact(_program) => {
                if samples.is_some() || seed.is_some() {
                    return Err(PyValueError::new_err(
                        "samples/seed are only supported for prob_engine='mc'",
                    ));
                }
                #[cfg(feature = "host-io")]
                {
                    if return_grads {
                        let result = _program
                            .evaluate_gpu_with_grads()
                            .map_err(types::xlog_err)?;
                        self.pack_result_with_grads(_py, result)
                    } else {
                        let result = _program
                            .evaluate()
                            .map_err(types::xlog_err)?;
                        self.pack_result_probs(_py, result.query_probs)
                    }
                }
                #[cfg(not(feature = "host-io"))]
                {
                    let _ = return_grads;
                    Err(types::host_io_disabled_pyerr())
                }
            }
            CompiledProbProgram::Mc(_program) => {
                if return_grads {
                    return Err(PyValueError::new_err(
                        "MC inference does not support gradients (return_grads must be false)",
                    ));
                }

                let cfg = McEvalConfig {
                    samples: samples.unwrap_or(10000),
                    seed: seed.unwrap_or(0),
                    confidence,
                    max_nonmonotone_iterations,
                    sampling_method: Self::parse_sampling_method(sampling_method)?,
                };
                #[cfg(feature = "host-io")]
                {
                    let result = _program
                        .evaluate(cfg)
                        .map_err(types::xlog_err)?;
                    self.pack_result_mc(_py, result)
                }
                #[cfg(not(feature = "host-io"))]
                {
                    let _ = cfg;
                    Err(types::host_io_disabled_pyerr())
                }
            }
        }
    }

    /// Evaluate Monte Carlo programs and return device-only result counts via DLPack.
    ///
    /// This is the primary GPU-native API surface for MC inference. It never performs
    /// device->host reads for result data (only returns device buffers).
    #[pyo3(signature = (samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024, sampling_method=None))]
    pub fn evaluate_device(
        &self,
        py: Python<'_>,
        samples: Option<usize>,
        seed: Option<u64>,
        confidence: f64,
        max_nonmonotone_iterations: usize,
        sampling_method: Option<String>,
    ) -> PyResult<McDeviceEvalResult> {
        let (
            query_counts,
            evidence_count,
            total_samples,
            seed,
            confidence,
            nonmonotone_sccs,
            nonmonotone_cycles,
            nonmonotone_iteration_limit_hits,
            sampling_method_val,
        ) = match &self.program {
            CompiledProbProgram::Mc(program) => {
                let cfg = McEvalConfig {
                    samples: samples.unwrap_or(10000),
                    seed: seed.unwrap_or(0),
                    confidence,
                    max_nonmonotone_iterations,
                    sampling_method: Self::parse_sampling_method(sampling_method)?,
                };

                let result = program
                    .evaluate_gpu_device_with_provider(cfg, self.output_provider.clone())
                    .map_err(types::xlog_err)?;

                (
                    result.query_counts,
                    result.evidence_count,
                    result.total_samples,
                    result.seed,
                    result.confidence,
                    result.nonmonotone_sccs,
                    result.nonmonotone_cycles,
                    result.nonmonotone_iteration_limit_hits,
                    result.sampling_method,
                )
            }
            _ => {
                return Err(PyValueError::new_err(
                    "evaluate_device is only supported for prob_engine='mc'",
                ))
            }
        };

        // PyTorch does not support unsigned 32-bit types. Export as i32 (bitwise identical for
        // counts < 2^31) for maximum DLPack consumer compatibility.
        let schema_i32 = Schema::new(vec![("col0".to_string(), ScalarType::I32)]);

        let make_count_tensor =
            |counts: xlog_cuda::memory::TrackedCudaSlice<u32>, rows: u64| -> PyResult<PyObject> {
                let rows_u32 = u32::try_from(rows).map_err(|_| {
                    PyValueError::new_err(format!("Row count {} exceeds u32::MAX", rows))
                })?;

                let mut d_num_rows = self
                    .output_provider
                    .memory()
                    .alloc::<u32>(1)
                    .map_err(types::xlog_err)?;
                self.output_provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
                    .map_err(types::xlog_err)?;

                let buffer = xlog_cuda::CudaBuffer::from_columns(
                    vec![counts.into_bytes().into()],
                    rows,
                    d_num_rows,
                    schema_i32.clone(),
                );
                let tensor = self
                    .output_provider
                    .to_dlpack_table(buffer)
                    .column(0)
                    .map_err(types::xlog_err)?;
                dlpack_capsule_from_tensor(py, tensor)
            };

        let query_rows = u64::try_from(query_counts.len())
            .map_err(|_| PyValueError::new_err("query_counts length overflow"))?;
        let query_counts_capsule = make_count_tensor(query_counts, query_rows)?;
        let evidence_count_capsule = make_count_tensor(evidence_count, 1)?;

        Ok(McDeviceEvalResult {
            query_counts: query_counts_capsule,
            evidence_count: evidence_count_capsule,
            total_samples,
            seed,
            confidence,
            nonmonotone_semantics: xlog_prob::mc::NONMONOTONE_SEMANTICS.to_string(),
            nonmonotone_sccs,
            nonmonotone_cycles,
            nonmonotone_iteration_limit_hits,
            sampling_method: match sampling_method_val {
                McSamplingMethod::Rejection => "rejection".to_string(),
                McSamplingMethod::EvidenceClamping => "evidence_clamping".to_string(),
            },
        })
    }

    // =========================================================================
    // NLL Loss Functions
    // =========================================================================

    /// Compute negative log-likelihood loss for a single query.
    ///
    /// NLL loss = -log(P(query))
    ///
    /// This is the fundamental training objective for neural-symbolic programs.
    /// Lower loss means higher probability of the query being true.
    ///
    /// # Arguments
    /// * `query` - Query atom as string, e.g., "digit(0, 5)" or "path(1, 3)"
    ///
    /// # Returns
    /// The NLL loss value (always non-negative, 0 for certain facts)
    fn nll_loss(&self, query: &str) -> PyResult<f64> {
        let prob = self.evaluate_query_probability(query)?;
        Ok(types::nll_loss_value(prob))
    }

    /// Compute sum of NLL losses for a batch of queries.
    ///
    /// Batch loss = Σ -log(P(query_i))
    ///
    /// More efficient than calling nll_loss repeatedly as all queries
    /// are compiled and evaluated together.
    ///
    /// # Arguments
    /// * `queries` - List of query atoms as strings
    ///
    /// # Returns
    /// Sum of individual NLL losses (0.0 for empty batch)
    fn nll_loss_batch(&self, queries: Vec<String>) -> PyResult<f64> {
        if queries.is_empty() {
            return Ok(0.0);
        }

        let probs = self.evaluate_query_probabilities(&queries)?;
        Ok(probs.iter().map(|&p| types::nll_loss_value(p)).sum())
    }

    /// Compute mean NLL loss for a batch of queries.
    ///
    /// Mean loss = (1/n) Σ -log(P(query_i))
    ///
    /// Useful for comparing loss across batches of different sizes.
    ///
    /// # Arguments
    /// * `queries` - List of query atoms as strings (must be non-empty)
    ///
    /// # Returns
    /// Mean of individual NLL losses
    ///
    /// # Errors
    /// Returns error if queries is empty
    fn nll_loss_mean(&self, queries: Vec<String>) -> PyResult<f64> {
        if queries.is_empty() {
            return Err(PyValueError::new_err(
                "Cannot compute mean NLL loss for empty query batch",
            ));
        }

        let probs = self.evaluate_query_probabilities(&queries)?;
        let sum: f64 = probs.iter().map(|&p| types::nll_loss_value(p)).sum();
        Ok(sum / probs.len() as f64)
    }

    /// Compute NLL loss and return as PyTorch tensor.
    ///
    /// Returns a scalar tensor that can participate in autograd.
    /// Use this when you need gradients to flow back through the loss.
    ///
    /// # Arguments
    /// * `query` - Query atom as string
    ///
    /// # Returns
    /// PyTorch scalar tensor containing the loss value
    fn nll_loss_tensor(&self, py: Python<'_>, query: &str) -> PyResult<PyObject> {
        let loss = self.nll_loss(query)?;
        types::create_torch_tensor(py, loss)
    }

    /// Compute batch NLL loss and return as PyTorch tensor.
    ///
    /// # Arguments
    /// * `queries` - List of query atoms as strings
    ///
    /// # Returns
    /// PyTorch scalar tensor containing the sum of losses
    fn nll_loss_batch_tensor(&self, py: Python<'_>, queries: Vec<String>) -> PyResult<PyObject> {
        let loss = self.nll_loss_batch(queries)?;
        types::create_torch_tensor(py, loss)
    }

    // =========================================================================
    // Backward Pass / Training Methods
    // =========================================================================

    /// Zero gradients for all registered networks.
    ///
    /// This should be called at the start of each training iteration
    /// to clear accumulated gradients from previous iterations.
    fn zero_grad(&self, py: Python<'_>) -> PyResult<()> {
        for name in self.network_registry.names() {
            if let Some(handle) = self.network_registry.get(name) {
                if let Some(ref optimizer) = handle.optimizer() {
                    optimizer.call_method0(py, "zero_grad")?;
                }
            }
        }
        Ok(())
    }

    /// Perform optimizer step for all registered networks.
    ///
    /// This applies the accumulated gradients to update network parameters.
    /// Should be called after forward_backward().
    fn optimizer_step(&self, py: Python<'_>) -> PyResult<()> {
        for name in self.network_registry.names() {
            if let Some(handle) = self.network_registry.get(name) {
                if let Some(ref optimizer) = handle.optimizer() {
                    optimizer.call_method0(py, "step")?;
                }
            }
        }
        Ok(())
    }

    /// Clip gradient norms for all registered networks.
    ///
    /// Uses `torch.nn.utils.clip_grad_norm_`.
    fn clip_grad_norms(&self, py: Python<'_>, max_norm: f64) -> PyResult<()> {
        let clip_fn = py
            .import("torch.nn.utils")?
            .getattr("clip_grad_norm_")?;
        for name in self.network_registry.names() {
            if let Some(handle) = self.network_registry.get(name) {
                if let Some(ref module) = handle.module() {
                    let params = module.call_method0(py, "parameters")?;
                    clip_fn.call1((params, max_norm))?;
                }
            }
        }
        Ok(())
    }

    /// Step the learning rate scheduler.
    ///
    /// If `network_name` is provided, steps only that network's scheduler.
    /// If `None` (default), steps all registered schedulers.
    #[pyo3(signature = (network_name=None))]
    fn scheduler_step(
        &self,
        py: Python<'_>,
        network_name: Option<&str>,
    ) -> PyResult<()> {
        match network_name {
            Some(name) => {
                let handle = self
                    .network_registry
                    .get(name)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "No network registered with name '{name}'"
                        ))
                    })?;
                if let Some(ref scheduler) = handle.scheduler() {
                    scheduler.call_method0(py, "step")?;
                }
            }
            None => {
                for name in self.network_registry.names() {
                    if let Some(handle) = self.network_registry.get(name) {
                        if let Some(ref scheduler) = handle.scheduler() {
                            scheduler.call_method0(py, "step")?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Get the current learning rate for a registered network.
    ///
    /// Reads `optimizer.param_groups[0]['lr']`.
    ///
    /// # Arguments
    /// * `network_name` - Name used in register_network()
    fn get_lr(&self, py: Python<'_>, network_name: &str) -> PyResult<f64> {
        let handle = self
            .network_registry
            .get(network_name)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No network registered with name '{network_name}'"
                ))
            })?;
        let optimizer = handle
            .optimizer()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Network '{network_name}' has no optimizer"
                ))
            })?;
        let param_groups = optimizer.getattr(py, "param_groups")?;
        let group0 = param_groups.call_method1(py, "__getitem__", (0i32,))?;
        let lr = group0.call_method1(py, "__getitem__", ("lr",))?;
        lr.extract(py)
    }

    /// Set the learning rate for a registered network.
    ///
    /// Writes to all `optimizer.param_groups[i]['lr']`.
    ///
    /// # Arguments
    /// * `network_name` - Name used in register_network()
    /// * `lr` - New learning rate value
    fn set_lr(&self, py: Python<'_>, network_name: &str, lr: f64) -> PyResult<()> {
        let handle = self
            .network_registry
            .get(network_name)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No network registered with name '{network_name}'"
                ))
            })?;
        let optimizer = handle
            .optimizer()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Network '{network_name}' has no optimizer"
                ))
            })?;
        let param_groups = optimizer.getattr(py, "param_groups")?;
        let num_groups: usize = param_groups.call_method0(py, "__len__")?.extract(py)?;
        for i in 0..num_groups {
            let group = param_groups.call_method1(py, "__getitem__", (i as i32,))?;
            group.call_method(py, "__setitem__", ("lr", lr), None)?;
        }
        Ok(())
    }

    // =========================================================================
    // Training Methods
    // =========================================================================

    /// Train for one epoch over the given queries.
    ///
    /// This method:
    /// 1. Processes queries in batches
    /// 2. For each batch: zero_grad, forward_backward for each query, optimizer_step
    /// 3. Returns statistics for the epoch
    ///
    /// # Arguments
    /// * `queries` - List of query strings to train on
    /// * `batch_size` - Number of queries per batch (default: 32)
    ///
    /// # Returns
    /// EpochStats with avg_loss, num_batches, total_queries
    #[pyo3(signature = (queries, batch_size=32, max_grad_norm=None))]
    fn train_epoch(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        batch_size: usize,
        max_grad_norm: Option<f64>,
    ) -> PyResult<EpochStats> {
        let mut history = TrainingHistory::new();
        self.train_epoch_internal(py, &queries, batch_size, usize::MAX, max_grad_norm, &mut history)
    }

    /// Evaluate mean NLL loss over queries without updating parameters.
    ///
    /// Useful for validation/test set evaluation.
    ///
    /// # Arguments
    /// * `queries` - List of query strings to evaluate
    ///
    /// # Returns
    /// Mean NLL loss over all queries
    fn evaluate_loss(&self, queries: Vec<String>) -> PyResult<f64> {
        if queries.is_empty() {
            return Ok(0.0);
        }

        let probs = self.evaluate_query_probabilities(&queries)?;
        let total_loss: f64 = probs.iter().map(|&p| types::nll_loss_value(p)).sum();
        Ok(total_loss / queries.len() as f64)
    }

    /// Train for one epoch with GPU-native loss accumulation (no per-query .item()).
    #[pyo3(signature = (queries, batch_size=32, max_grad_norm=None))]
    fn train_epoch_tensor(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        batch_size: usize,
        max_grad_norm: Option<f64>,
    ) -> PyResult<EpochStats> {
        let mut history = TrainingHistory::new();
        self.train_epoch_tensor_internal(py, &queries, batch_size, usize::MAX, max_grad_norm, &mut history)
    }

    /// Return warmup profiling data as a Python dict (or None if profiling disabled).
    ///
    /// When XLOG_WARMUP_PROFILE=1, returns a dict with:
    ///   - "ptx": PTX load timing breakdown
    ///   - "circuit": circuit compilation timing breakdown
    /// Returns None if profiling is not enabled or no data is available.
    fn warmup_breakdown(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let ptx_profile = self.output_provider.ptx_load_profile();
        let circuit_profile = self.last_compile_profile.as_ref();

        // Return None if neither profile is available.
        if ptx_profile.is_none() && circuit_profile.is_none() {
            return Ok(None);
        }

        let result = PyDict::new_bound(py);

        if let Some(ptx) = ptx_profile {
            let ptx_dict = PyDict::new_bound(py);
            ptx_dict.set_item("total_sec", ptx.total_sec)?;
            ptx_dict.set_item("cubin_loaded", ptx.cubin_loaded)?;
            ptx_dict.set_item("ptx_fallback", ptx.ptx_fallback)?;
            let per_module = PyDict::new_bound(py);
            for (name, sec) in &ptx.per_module_sec {
                per_module.set_item(name, *sec)?;
            }
            ptx_dict.set_item("per_module_sec", per_module)?;
            result.set_item("ptx", ptx_dict)?;
        }

        if let Some(circuit) = circuit_profile {
            let circuit_dict = PyDict::new_bound(py);
            circuit_dict.set_item("gpu_cache_hit", circuit.gpu_cache_hit)?;
            circuit_dict.set_item("disk_cache_hit", circuit.disk_cache_hit)?;
            circuit_dict.set_item("d4_compile_sec", circuit.d4_compile_sec)?;
            circuit_dict.set_item("verify_sec", circuit.verify_sec)?;
            circuit_dict.set_item("smooth_sec", circuit.smooth_sec)?;
            circuit_dict.set_item("cache_store_sec", circuit.cache_store_sec)?;
            circuit_dict.set_item("free_var_mask_sec", circuit.free_var_mask_sec)?;
            circuit_dict.set_item("cnf_hash_sec", circuit.cnf_hash_sec)?;
            result.set_item("circuit", circuit_dict)?;
        }

        Ok(Some(result.into()))
    }

    /// Clear the circuit template cache, forcing recompilation on next query.
    /// Used for cache ablation benchmarks.
    fn clear_circuit_cache(&mut self) {
        self.circuit_cache.clear();
    }
}

impl CompiledProgram {
    /// Evaluate probability of a single query by compiling a temporary program.
    fn evaluate_query_probability(&self, query: &str) -> PyResult<f64> {
        let probs = self.evaluate_query_probabilities(&[query.to_string()])?;
        probs
            .into_iter()
            .next()
            .ok_or_else(|| PyRuntimeError::new_err("Query evaluation returned no results"))
    }

    /// Evaluate probabilities for multiple queries by compiling a temporary program.
    fn evaluate_query_probabilities(&self, queries: &[String]) -> PyResult<Vec<f64>> {
        #[cfg(not(feature = "host-io"))]
        {
            let _ = queries;
            return Err(types::host_io_disabled_pyerr());
        }

        #[cfg(feature = "host-io")]
        {
            // Build source with queries appended
            let mut source_with_queries = self._source.clone();
            for query in queries {
                source_with_queries.push_str(&format!("\nquery({}).", query));
            }

            // Compile and evaluate the temporary program
            let result: Vec<QueryProbability> = match self._prob_engine {
                ProbEngine::ExactDdnnf => {
                    let program = ExactDdnnfProgram::compile_source_with_gpu(
                        &source_with_queries,
                        self._gpu_config,
                    )
                    .map_err(|e| types::gpu_err("Query compilation error", e))?;

                    program
                        .evaluate()
                        .map_err(|e| types::gpu_err("Query evaluation error", e))?
                        .query_probs
                }
                ProbEngine::Mc => {
                    let program =
                        McProgram::compile_source_with_gpu(&source_with_queries, self._gpu_config)
                            .map_err(|e| types::gpu_err("Query compilation error", e))?;

                    let cfg = McEvalConfig::default();
                    program
                        .evaluate(cfg)
                        .map_err(|e| types::gpu_err("Query evaluation error", e))?
                        .query_estimates
                        .into_iter()
                        .map(|e| QueryProbability {
                            atom: e.atom,
                            prob: e.prob,
                            log_prob: e.log_prob,
                        })
                        .collect()
                }
            };

            // Extract probabilities in query order
            // The results should be in the same order as queries were added
            let probs: Vec<f64> = result.iter().map(|qp| qp.prob).collect();

            if probs.len() != queries.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "Expected {} query results, got {}",
                    queries.len(),
                    probs.len()
                )));
            }

            Ok(probs)
        }
    }

    #[cfg(feature = "host-io")]
    fn pack_result_probs(
        &self,
        py: Python<'_>,
        query_probs: Vec<QueryProbability>,
    ) -> PyResult<EvalResult> {
        let mut atoms: Vec<String> = Vec::with_capacity(query_probs.len());
        let mut probs: Vec<f64> = Vec::with_capacity(query_probs.len());
        let mut log_probs: Vec<f64> = Vec::with_capacity(query_probs.len());

        for q in query_probs {
            atoms.push(atom_to_string(&q.atom));
            probs.push(q.prob);
            log_probs.push(q.log_prob);
        }

        let schema = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);
        let prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&probs, schema.clone())
            .map_err(types::xlog_err)?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&log_probs, schema)
            .map_err(types::xlog_err)?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;

        Ok(EvalResult {
            atoms,
            prob: dlpack_capsule_from_tensor(py, prob_tensor)?,
            log_prob: dlpack_capsule_from_tensor(py, log_prob_tensor)?,
            num_vars: self.program.num_vars(),
            grad_true: None,
            grad_false: None,
            approx: false,
            stderr: None,
            ci_low: None,
            ci_high: None,
            samples: None,
            evidence_samples: None,
            seed: None,
            confidence: None,
            nonmonotone_semantics: None,
            nonmonotone_sccs: None,
            nonmonotone_cycles: None,
            nonmonotone_iteration_limit_hits: None,
            sampling_method: None,
        })
    }

    #[cfg(feature = "host-io")]
    fn pack_result_with_grads(
        &self,
        py: Python<'_>,
        result: ExactResultWithGrads,
    ) -> PyResult<EvalResult> {
        let mut atoms: Vec<String> = Vec::with_capacity(result.query_grads.len());
        let mut probs: Vec<f64> = Vec::with_capacity(result.query_grads.len());
        let mut log_probs: Vec<f64> = Vec::with_capacity(result.query_grads.len());

        let mut grad_true_caps: Vec<PyObject> = Vec::with_capacity(result.query_grads.len());
        let mut grad_false_caps: Vec<PyObject> = Vec::with_capacity(result.query_grads.len());

        let schema = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);

        let num_vars = self.program.num_vars();
        for q in result.query_grads {
            atoms.push(atom_to_string(&q.atom));
            probs.push(q.prob);
            log_probs.push(q.log_prob);

            let grad_true_buf = self
                .output_provider
                .create_buffer_from_slice::<f64>(&q.grad_true, schema.clone())
                .map_err(types::xlog_err)?;
            let grad_false_buf = self
                .output_provider
                .create_buffer_from_slice::<f64>(&q.grad_false, schema.clone())
                .map_err(types::xlog_err)?;

            let grad_true_tensor = self
                .output_provider
                .to_dlpack_table(grad_true_buf)
                .column(0)
                .map_err(types::xlog_err)?;
            let grad_false_tensor = self
                .output_provider
                .to_dlpack_table(grad_false_buf)
                .column(0)
                .map_err(types::xlog_err)?;

            grad_true_caps.push(dlpack_capsule_from_tensor(py, grad_true_tensor)?);
            grad_false_caps.push(dlpack_capsule_from_tensor(py, grad_false_tensor)?);
        }

        let prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&probs, schema.clone())
            .map_err(types::xlog_err)?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&log_probs, schema)
            .map_err(types::xlog_err)?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;

        Ok(EvalResult {
            atoms,
            prob: dlpack_capsule_from_tensor(py, prob_tensor)?,
            log_prob: dlpack_capsule_from_tensor(py, log_prob_tensor)?,
            num_vars,
            grad_true: Some(grad_true_caps),
            grad_false: Some(grad_false_caps),
            approx: false,
            stderr: None,
            ci_low: None,
            ci_high: None,
            samples: None,
            evidence_samples: None,
            seed: None,
            confidence: None,
            nonmonotone_semantics: None,
            nonmonotone_sccs: None,
            nonmonotone_cycles: None,
            nonmonotone_iteration_limit_hits: None,
            sampling_method: None,
        })
    }

    #[cfg(feature = "host-io")]
    fn pack_result_mc(
        &self,
        py: Python<'_>,
        result: xlog_prob::mc::McResult,
    ) -> PyResult<EvalResult> {
        let mut atoms: Vec<String> = Vec::with_capacity(result.query_estimates.len());
        let mut probs: Vec<f64> = Vec::with_capacity(result.query_estimates.len());
        let mut log_probs: Vec<f64> = Vec::with_capacity(result.query_estimates.len());
        let mut stderrs: Vec<f64> = Vec::with_capacity(result.query_estimates.len());
        let mut ci_lows: Vec<f64> = Vec::with_capacity(result.query_estimates.len());
        let mut ci_highs: Vec<f64> = Vec::with_capacity(result.query_estimates.len());

        for q in &result.query_estimates {
            atoms.push(atom_to_string(&q.atom));
            probs.push(q.prob);
            log_probs.push(q.log_prob);
            stderrs.push(q.stderr);
            ci_lows.push(q.ci_low);
            ci_highs.push(q.ci_high);
        }

        let schema = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);
        let prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&probs, schema.clone())
            .map_err(types::xlog_err)?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&log_probs, schema.clone())
            .map_err(types::xlog_err)?;
        let stderr_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&stderrs, schema.clone())
            .map_err(types::xlog_err)?;
        let ci_low_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&ci_lows, schema.clone())
            .map_err(types::xlog_err)?;
        let ci_high_buf = self
            .output_provider
            .create_buffer_from_slice::<f64>(&ci_highs, schema)
            .map_err(types::xlog_err)?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let stderr_tensor = self
            .output_provider
            .to_dlpack_table(stderr_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let ci_low_tensor = self
            .output_provider
            .to_dlpack_table(ci_low_buf)
            .column(0)
            .map_err(types::xlog_err)?;
        let ci_high_tensor = self
            .output_provider
            .to_dlpack_table(ci_high_buf)
            .column(0)
            .map_err(types::xlog_err)?;

        Ok(EvalResult {
            atoms,
            prob: dlpack_capsule_from_tensor(py, prob_tensor)?,
            log_prob: dlpack_capsule_from_tensor(py, log_prob_tensor)?,
            num_vars: self.program.num_vars(),
            grad_true: None,
            grad_false: None,
            approx: true,
            stderr: Some(dlpack_capsule_from_tensor(py, stderr_tensor)?),
            ci_low: Some(dlpack_capsule_from_tensor(py, ci_low_tensor)?),
            ci_high: Some(dlpack_capsule_from_tensor(py, ci_high_tensor)?),
            samples: Some(result.total_samples),
            evidence_samples: Some(result.evidence_samples),
            seed: Some(result.seed),
            confidence: Some(result.confidence),
            nonmonotone_semantics: Some(xlog_prob::mc::NONMONOTONE_SEMANTICS.to_string()),
            nonmonotone_sccs: Some(result.nonmonotone_sccs),
            nonmonotone_cycles: Some(result.nonmonotone_cycles),
            nonmonotone_iteration_limit_hits: Some(result.nonmonotone_iteration_limit_hits),
            sampling_method: Some(match result.sampling_method {
                McSamplingMethod::Rejection => "rejection".to_string(),
                McSamplingMethod::EvidenceClamping => "evidence_clamping".to_string(),
            }),
        })
    }
}

#[pyclass]
pub struct LogicProgram;

#[pyclass]
pub struct CompiledLogicProgram {
    pub(crate) program: gpu_logic::LogicProgram,
    pub(crate) provider: Arc<CudaKernelProvider>,
}

#[pyclass]
pub struct LogicQueryResult {
    #[pyo3(get)]
    pub relation_name: String,
    #[pyo3(get)]
    pub columns: Vec<String>,
    #[pyo3(get)]
    pub tensors: Vec<PyObject>,
    #[pyo3(get)]
    pub num_rows: usize,
    #[pyo3(get)]
    pub is_true: bool,
}

#[pyclass]
pub struct LogicEvalResult {
    #[pyo3(get)]
    pub queries: Vec<Py<LogicQueryResult>>,
}

#[pyclass]
pub struct McDeviceEvalResult {
    /// Per-query satisfying-sample counts. DLPack int32 tensor on CUDA.
    #[pyo3(get)]
    pub query_counts: PyObject,
    /// Evidence satisfying-sample count. DLPack int32 tensor with shape [1] on CUDA.
    #[pyo3(get)]
    pub evidence_count: PyObject,
    #[pyo3(get)]
    pub total_samples: usize,
    #[pyo3(get)]
    pub seed: u64,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub nonmonotone_semantics: String,
    #[pyo3(get)]
    pub nonmonotone_sccs: usize,
    #[pyo3(get)]
    pub nonmonotone_cycles: usize,
    #[pyo3(get)]
    pub nonmonotone_iteration_limit_hits: usize,
    #[pyo3(get)]
    pub sampling_method: String,
}

#[pyclass]
pub struct EvalResult {
    #[pyo3(get)]
    pub atoms: Vec<String>,
    #[pyo3(get)]
    pub prob: PyObject,
    #[pyo3(get)]
    pub log_prob: PyObject,
    #[pyo3(get)]
    pub num_vars: usize,
    #[pyo3(get)]
    pub grad_true: Option<Vec<PyObject>>,
    #[pyo3(get)]
    pub grad_false: Option<Vec<PyObject>>,
    #[pyo3(get)]
    pub approx: bool,
    #[pyo3(get)]
    pub stderr: Option<PyObject>,
    #[pyo3(get)]
    pub ci_low: Option<PyObject>,
    #[pyo3(get)]
    pub ci_high: Option<PyObject>,
    #[pyo3(get)]
    pub samples: Option<usize>,
    #[pyo3(get)]
    pub evidence_samples: Option<usize>,
    #[pyo3(get)]
    pub seed: Option<u64>,
    #[pyo3(get)]
    pub confidence: Option<f64>,
    #[pyo3(get)]
    pub nonmonotone_semantics: Option<String>,
    #[pyo3(get)]
    pub nonmonotone_sccs: Option<usize>,
    #[pyo3(get)]
    pub nonmonotone_cycles: Option<usize>,
    #[pyo3(get)]
    pub nonmonotone_iteration_limit_hits: Option<usize>,
    #[pyo3(get)]
    pub sampling_method: Option<String>,
}

// =========================================================================
// Training Infrastructure
// =========================================================================

/// Statistics for a single training epoch.
#[pyclass]
#[derive(Clone)]
pub struct EpochStats {
    /// Average loss across all batches in the epoch
    #[pyo3(get)]
    pub avg_loss: f64,
    /// Number of batches processed
    #[pyo3(get)]
    pub num_batches: usize,
    /// Total number of queries processed
    #[pyo3(get)]
    pub total_queries: usize,
}

/// Training history tracking loss over epochs and batches.
#[pyclass]
#[derive(Clone)]
pub struct TrainingHistory {
    /// Loss at the end of each epoch
    #[pyo3(get)]
    pub epoch_losses: Vec<f64>,
    /// Wall-clock time (seconds) for each epoch
    #[pyo3(get)]
    pub epoch_times: Vec<f64>,
    /// Loss for each batch across all epochs
    #[pyo3(get)]
    pub batch_losses: Vec<f64>,
    /// True if training was stopped early due to validation loss plateau.
    #[pyo3(get)]
    pub stopped_early: bool,
}

#[pyclass]
pub struct IlpProgramFactory;

#[pyclass]
pub struct CompiledIlpProgram {
    pub(crate) base_source: String,
    pub(crate) _learnable_source: String,
    pub(crate) ast: AstProgram,
    pub(crate) executor: Executor,
    pub(crate) provider: Arc<CudaKernelProvider>,
    pub(crate) plan: ExecutionPlan,
    pub(crate) rel_index: Vec<(RelId, String)>,
    pub(crate) schemas: HashMap<String, Schema>,
    pub(crate) left_keys: Vec<usize>,
    pub(crate) right_keys: Vec<usize>,
    pub(crate) head_projection: Vec<usize>,
    pub(crate) compiled_schema_size: usize,
    pub(crate) head_rel_name: String,
    pub(crate) max_active_rules: usize,
    pub(crate) candidate_map: Option<HashMap<(u32, u32, u32), u32>>,
    /// Maximum bytes for per-chunk temp allocations (masks, prefix sums,
    /// chunk-local COO scratch). The final merged COO buffer is exact-NNZ
    /// sized and may exceed this budget. Default: 16 MB.
    pub(crate) coo_chunk_budget: u64,
    /// When true, raise instead of falling back to chunked COO path.
    /// Use in zero-D2H benchmarks and CI gates. Default: false.
    pub(crate) strict_zero_dtoh: bool,
}


#[pymodule]
#[pyo3(name = "_native")]
fn pyxlog(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Program>()?;
    m.add_class::<CompiledProgram>()?;
    m.add_class::<LogicProgram>()?;
    m.add_class::<CompiledLogicProgram>()?;
    m.add_class::<LogicQueryResult>()?;
    m.add_class::<LogicEvalResult>()?;
    m.add_class::<McDeviceEvalResult>()?;
    m.add_class::<EvalResult>()?;
    // Training infrastructure
    m.add_class::<EpochStats>()?;
    m.add_class::<TrainingHistory>()?;
    // ILP bindings
    m.add_class::<IlpProgramFactory>()?;
    m.add_class::<CompiledIlpProgram>()?;
    m.add_function(wrap_pyfunction!(training::train_model, m)?)?;
    m.add_function(wrap_pyfunction!(training::train_model_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(dlpack_roundtrip, m)?)?;
    #[cfg(feature = "arrow-device-import")]
    m.add_function(wrap_pyfunction!(export_arrow_device, m)?)?;
    #[cfg(feature = "arrow-device-import")]
    m.add_function(wrap_pyfunction!(import_arrow_device, m)?)?;
    Ok(())
}
