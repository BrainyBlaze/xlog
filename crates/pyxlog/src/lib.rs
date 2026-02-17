use std::collections::{HashMap, HashSet};
use std::os::raw::{c_char, c_void};
use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};
use pyo3::Bound;

use ::xlog_gpu::logic as gpu_logic;
use xlog_core::{symbol, MemoryBudget, ScalarType, Schema};
#[cfg(feature = "arrow-device-import")]
use xlog_cuda::{ArrowDeviceArray, ArrowDeviceArrayOwned};
use xlog_cuda::{CudaDevice, CudaKernelProvider, DlpackManagedTensor, GpuMemoryManager};
#[cfg(feature = "host-io")]
use xlog_logic::ast::ArithExpr;
use xlog_logic::ast::{Atom, BodyLiteral, ProbEngine, Rule, Term};
use xlog_logic::parse_program;
use xlog_neural::{NetworkConfig, NetworkRegistry, TensorMetadata, TensorSourceRegistry};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
#[cfg(feature = "host-io")]
use xlog_prob::exact::{ExactResultWithGrads, QueryProbability};
use xlog_prob::mc::{McEvalConfig, McProgram};
use xlog_prob::neural_fast_path::{GpuWeightSlots, NeuralFastPathConfig};

use std::collections::HashMap as StdHashMap;

mod neural_registry;
use neural_registry::{NeuralPredicateInfo, NeuralPredicateRegistry};

const DLPACK_CAPSULE_NAME: &[u8] = b"dltensor\0";
const USED_DLPACK_CAPSULE_NAME: &[u8] = b"used_dltensor\0";

#[cfg(feature = "arrow-device-import")]
const ARROW_DEVICE_ARRAY_CAPSULE_NAME: &[u8] = b"arrow_device_array\0";
#[cfg(feature = "arrow-device-import")]
const USED_ARROW_DEVICE_ARRAY_CAPSULE_NAME: &[u8] = b"used_arrow_device_array\0";

/// Epsilon value for numerical stability in log computations
const NLL_EPSILON: f64 = 1e-38;

#[cfg(not(feature = "host-io"))]
fn host_io_disabled_pyerr() -> PyErr {
    PyRuntimeError::new_err(
        "Host output is disabled (feature \"host-io\" is OFF). Use device-resident APIs (DLPack) or rebuild with --features host-io.",
    )
}

/// Compute negative log-likelihood loss from probability.
///
/// NLL loss = -log(max(p, epsilon))
///
/// The epsilon prevents -log(0) = infinity.
#[inline]
fn nll_loss_value(probability: f64) -> f64 {
    -(probability.max(NLL_EPSILON)).ln()
}

/// Create a PyTorch tensor from a scalar f64 value.
fn create_torch_tensor(py: Python<'_>, value: f64) -> PyResult<PyObject> {
    let torch = py.import_bound("torch")?;
    let tensor = torch.call_method1("tensor", (value,))?;
    Ok(tensor.into())
}

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

fn dlpack_capsule_from_tensor(py: Python<'_>, tensor: DlpackManagedTensor) -> PyResult<PyObject> {
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

fn provider_from_config(config: GpuConfig) -> xlog_core::Result<CudaKernelProvider> {
    let device = Arc::new(CudaDevice::new(config.device_ordinal)?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(config.memory_bytes),
    ));
    CudaKernelProvider::new(device, memory)
}

fn parse_prob_engine_override(s: &str) -> PyResult<ProbEngine> {
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

fn dlpack_from_py(obj: &Bound<'_, PyAny>) -> PyResult<DlpackManagedTensor> {
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
        provider_from_config(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let device_array = provider
        .to_arrow_device_record_batch(buffer)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
        provider_from_config(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let device_array = arrow_device_from_py(device_array)?;

    let buffer = provider
        .from_arrow_device_record_batch(device_array)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
        provider_from_config(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let managed = dlpack_from_py(tensor)?;
    let buffer = provider
        .from_dlpack_tensors(vec![managed])
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = provider
        .to_dlpack_table(buffer)
        .column(0)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    dlpack_capsule_from_tensor(py, out)
}

#[pyclass]
pub struct Program;

#[pymethods]
impl Program {
    #[staticmethod]
    #[pyo3(signature = (source, device=0, memory_mb=32768, prob_engine=None))]
    pub fn compile(
        source: &str,
        device: usize,
        memory_mb: u64,
        prob_engine: Option<String>,
    ) -> PyResult<CompiledProgram> {
        if memory_mb == 0 {
            return Err(PyValueError::new_err("memory_mb must be > 0"));
        }

        let config = GpuConfig {
            device_ordinal: device,
            memory_bytes: memory_mb * 1024 * 1024,
        };

        // Parse the AST to get prob_engine and neural predicates
        let ast = xlog_logic::parse_program(source)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Extract declared neural network names
        let declared_networks: HashSet<String> = ast
            .neural_predicates
            .iter()
            .map(|np| np.network.clone())
            .collect();
        let neural_registry =
            NeuralPredicateRegistry::from_ast(&ast).map_err(|e| PyValueError::new_err(e))?;

        let engine = match prob_engine {
            Some(s) => parse_prob_engine_override(&s)?,
            None => ast.prob_engine(),
        };

        let program = match engine {
            ProbEngine::ExactDdnnf => CompiledProbProgram::Exact(
                ExactDdnnfProgram::compile_source_with_gpu(source, config)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            ),
            ProbEngine::Mc => CompiledProbProgram::Mc(
                McProgram::compile_source_with_gpu(source, config)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            ),
        };
        let provider =
            provider_from_config(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(CompiledProgram {
            program,
            output_provider: Arc::new(provider),
            network_registry: NetworkRegistry::new(),
            neural_registry,
            declared_networks,
            tensor_sources: TensorSourceRegistry::new(),
            _source: source.to_string(),
            ast,
            _gpu_config: config,
            _prob_engine: engine,
            query_signature_cache: StdHashMap::new(),
            circuit_cache: StdHashMap::new(),
            template_compile_count: 0,
        })
    }
}

/// A cached circuit for a specific query template.
///
/// The circuit structure is immutable - only weights change between queries.
/// Weight slots map network outputs to circuit variables.
struct CachedCircuit {
    /// The compiled program containing the GPU circuit
    program: ExactDdnnfProgram,

    /// Device-resident mapping from neural output slots to CNF variable ids.
    slots: GpuWeightSlots,

    /// Ordered target domain for Targeted signatures. Empty for Boolean.
    target_domain: Vec<String>,
}

#[derive(Debug, Clone)]
enum InputSource {
    QueryArg(usize),
    ImplicitSlot(usize),
}

#[derive(Debug, Clone)]
struct NeuralGroup {
    info: NeuralPredicateInfo,
    input_source: InputSource,
    #[cfg(feature = "host-io")]
    output_var: Option<String>,
}

#[derive(Debug, Clone)]
enum QuerySignature {
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

enum CompiledProbProgram {
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

#[pyclass(unsendable)]
pub struct CompiledProgram {
    program: CompiledProbProgram,
    output_provider: Arc<CudaKernelProvider>,
    /// Registry for neural networks
    network_registry: NetworkRegistry,
    /// Registry for neural predicate metadata (predicate -> network/labels)
    neural_registry: NeuralPredicateRegistry,
    /// Names of neural networks declared in the program (from nn() declarations)
    declared_networks: HashSet<String>,
    /// Registry for tensor data sources (images, embeddings, etc.)
    tensor_sources: TensorSourceRegistry,
    /// Original program source (for dynamic query compilation)
    _source: String,
    /// Parsed program AST (for signature analysis)
    ast: xlog_logic::ast::Program,
    /// GPU configuration
    _gpu_config: GpuConfig,
    /// Probabilistic inference engine
    _prob_engine: ProbEngine,
    /// Cache of analyzed query signatures.
    query_signature_cache: StdHashMap<String, QuerySignature>,
    /// Cache of compiled circuits by template signature
    circuit_cache: StdHashMap<String, CachedCircuit>,
    /// Number of times the template compilation path executed.
    template_compile_count: usize,
}

#[pymethods]
impl CompiledProgram {
    #[pyo3(signature = (return_grads=false, samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024))]
    pub fn evaluate(
        &self,
        _py: Python<'_>,
        return_grads: bool,
        samples: Option<usize>,
        seed: Option<u64>,
        confidence: f64,
        max_nonmonotone_iterations: usize,
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
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        self.pack_result_with_grads(_py, result)
                    } else {
                        let result = _program
                            .evaluate()
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        self.pack_result_probs(_py, result.query_probs)
                    }
                }
                #[cfg(not(feature = "host-io"))]
                {
                    let _ = return_grads;
                    Err(host_io_disabled_pyerr())
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
                };
                #[cfg(feature = "host-io")]
                {
                    let result = _program
                        .evaluate(cfg)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    self.pack_result_mc(_py, result)
                }
                #[cfg(not(feature = "host-io"))]
                {
                    let _ = cfg;
                    Err(host_io_disabled_pyerr())
                }
            }
        }
    }

    /// Evaluate Monte Carlo programs and return device-only result counts via DLPack.
    ///
    /// This is the primary GPU-native API surface for MC inference. It never performs
    /// device->host reads for result data (only returns device buffers).
    #[pyo3(signature = (samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024))]
    pub fn evaluate_device(
        &self,
        py: Python<'_>,
        samples: Option<usize>,
        seed: Option<u64>,
        confidence: f64,
        max_nonmonotone_iterations: usize,
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
        ) = match &self.program {
            CompiledProbProgram::Mc(program) => {
                let cfg = McEvalConfig {
                    samples: samples.unwrap_or(10000),
                    seed: seed.unwrap_or(0),
                    confidence,
                    max_nonmonotone_iterations,
                };

                let result = program
                    .evaluate_gpu_device_with_provider(cfg, self.output_provider.clone())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                (
                    result.query_counts,
                    result.evidence_count,
                    result.total_samples,
                    result.seed,
                    result.confidence,
                    result.nonmonotone_sccs,
                    result.nonmonotone_cycles,
                    result.nonmonotone_iteration_limit_hits,
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
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                self.output_provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
        })
    }

    /// Register a PyTorch neural network with this program.
    ///
    /// The network name must match an `nn()` declaration in the program source.
    ///
    /// # Arguments
    /// * `name` - Network name (must match nn() declaration)
    /// * `module` - PyTorch nn.Module instance
    /// * `optimizer` - PyTorch optimizer (e.g., Adam, SGD)
    /// * `scheduler` - Optional learning rate scheduler
    /// * `batching` - Whether to batch inputs for GPU efficiency (default: true)
    /// * `k` - Top-k sampling: only consider top k outputs (default: None = all)
    /// * `det` - Deterministic mode: use argmax instead of sampling (default: false)
    /// * `cache` - Whether to cache network outputs (default: true)
    /// * `cache_size` - Maximum cache entries (default: 10000)
    #[pyo3(signature = (name, module, optimizer, scheduler=None, batching=true, k=None, det=false, cache=true, cache_size=10000))]
    fn register_network(
        &mut self,
        name: String,
        module: PyObject,
        optimizer: PyObject,
        scheduler: Option<PyObject>,
        batching: bool,
        k: Option<usize>,
        det: bool,
        cache: bool,
        cache_size: usize,
    ) -> PyResult<()> {
        // Validate network name exists in neural predicates
        if !self.declared_networks.contains(&name) {
            return Err(PyValueError::new_err(format!(
                "Network '{}' not declared in program. Declared networks: {:?}",
                name,
                self.declared_networks.iter().collect::<Vec<_>>()
            )));
        }

        let config = NetworkConfig {
            name: name.clone(),
            batching,
            k,
            det,
            cache_enabled: cache,
            cache_size,
        };

        self.network_registry.register(config);

        // Store PyTorch objects in the handle
        if let Some(handle) = self.network_registry.get_mut(&name) {
            handle.set_module(module);
            handle.set_optimizer(optimizer);
            if let Some(sched) = scheduler {
                handle.set_scheduler(sched);
            }
        }

        Ok(())
    }

    /// Get names of all registered neural networks.
    fn network_names(&self) -> Vec<String> {
        self.network_registry
            .names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Number of cached circuit templates currently stored for this program.
    fn template_cache_size(&self) -> usize {
        self.circuit_cache.len()
    }

    /// Number of times template compilation has been executed.
    fn template_compile_count(&self) -> usize {
        self.template_compile_count
    }

    /// Get names of all declared neural networks (from nn() declarations).
    fn declared_network_names(&self) -> Vec<String> {
        self.declared_networks.iter().cloned().collect()
    }

    /// Get neural predicate metadata (network name + labels).
    fn neural_predicate_info(&self, py: Python<'_>, predicate: &str) -> PyResult<PyObject> {
        let infos = self.neural_registry.get(predicate).ok_or_else(|| {
            PyValueError::new_err(format!("Unknown neural predicate '{}'", predicate))
        })?;
        let info = match infos.len() {
            0 => {
                return Err(PyValueError::new_err(format!(
                    "Unknown neural predicate '{}'",
                    predicate
                )))
            }
            1 => &infos[0],
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Predicate '{}' has multiple nn/4 declarations; provide a concrete query atom",
                    predicate
                )))
            }
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("network", info.network.clone())?;
        dict.set_item("labels", info.labels.clone())?;
        Ok(dict.into())
    }

    /// Resolve a label to its index using the declared nn/4 label list.
    fn label_to_index(&self, predicate: &str, label: &str) -> PyResult<usize> {
        let infos = self.neural_registry.get(predicate).ok_or_else(|| {
            PyValueError::new_err(format!("Unknown neural predicate '{}'", predicate))
        })?;
        let info = match infos.as_slice() {
            [] => {
                return Err(PyValueError::new_err(format!(
                    "Unknown neural predicate '{}'",
                    predicate
                )))
            }
            [single] => single,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Predicate '{}' has multiple nn/4 declarations; provide a concrete query atom",
                    predicate
                )))
            }
        };
        let labels = info.labels.as_ref().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Predicate '{}' does not declare a label list in nn/4",
                predicate
            ))
        })?;
        labels
            .iter()
            .position(|l| l == label)
            .ok_or_else(|| PyValueError::new_err(format!("Label '{}' not in declared list", label)))
    }

    /// Check if a network is declared in the program.
    fn has_neural_predicate(&self, name: &str) -> bool {
        self.declared_networks.contains(name)
    }

    /// Set training mode for all registered networks.
    fn set_train_mode(&mut self, train: bool) {
        self.network_registry.set_train_mode(train);
    }

    // =========================================================================
    // Tensor Source Management
    // =========================================================================

    /// Add a tensor source (e.g., training images, test images).
    ///
    /// The tensor should be a PyTorch tensor where the first dimension
    /// is the number of samples. Neural predicates will index into this
    /// tensor by sample index.
    ///
    /// The first source added automatically becomes the active source.
    ///
    /// # Arguments
    /// * `name` - Name for this source (e.g., "train", "test")
    /// * `tensor` - PyTorch tensor with shape [N, ...]
    fn add_tensor_source(
        &mut self,
        py: Python<'_>,
        name: String,
        tensor: PyObject,
    ) -> PyResult<()> {
        // Extract size from tensor.shape[0]
        // In PyO3 0.21, use .bind(py) to get Bound<PyAny> for method calls
        let shape_obj = tensor.getattr(py, "shape")?;
        let shape_bound = shape_obj.bind(py);
        let size: usize = shape_bound.get_item(0)?.extract()?;

        // Extract full shape for metadata
        let shape_tuple: Vec<usize> = shape_bound.extract()?;
        let sample_shape = if shape_tuple.len() > 1 {
            shape_tuple[1..].to_vec()
        } else {
            vec![]
        };

        // Extract dtype
        let dtype_obj = tensor.getattr(py, "dtype")?;
        let dtype_str = dtype_obj.bind(py).str()?.to_string();

        let metadata = TensorMetadata::with_dtype(size, sample_shape, &dtype_str);
        self.tensor_sources.add(&name, tensor, metadata);

        Ok(())
    }

    /// Set the active tensor source.
    ///
    /// The active source is used when neural predicates are evaluated.
    fn set_active_tensor_source(&mut self, name: String) -> PyResult<()> {
        self.tensor_sources
            .set_active(&name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get the name of the currently active tensor source.
    fn active_tensor_source(&self) -> Option<String> {
        self.tensor_sources.active_name().map(|s| s.to_string())
    }

    /// Get the size (number of samples) of the active tensor source.
    fn active_tensor_source_size(&self) -> PyResult<usize> {
        self.tensor_sources
            .active_size()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get names of all tensor sources.
    fn tensor_source_names(&self) -> Vec<String> {
        self.tensor_sources.source_names()
    }

    /// Check if a tensor source exists.
    fn has_tensor_source(&self, name: &str) -> bool {
        self.tensor_sources.contains(name)
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
        Ok(nll_loss_value(prob))
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
        Ok(probs.iter().map(|&p| nll_loss_value(p)).sum())
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
        let sum: f64 = probs.iter().map(|&p| nll_loss_value(p)).sum();
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
        create_torch_tensor(py, loss)
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
        create_torch_tensor(py, loss)
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

    /// Step the learning rate scheduler for all registered networks.
    ///
    /// This updates learning rates according to the scheduler policy.
    /// Should be called once per epoch or as specified by the scheduler.
    fn scheduler_step(&self, py: Python<'_>) -> PyResult<()> {
        for name in self.network_registry.names() {
            if let Some(handle) = self.network_registry.get(name) {
                if let Some(ref scheduler) = handle.scheduler() {
                    scheduler.call_method0(py, "step")?;
                }
            }
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
    #[pyo3(signature = (queries, batch_size=32))]
    fn train_epoch(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        batch_size: usize,
    ) -> PyResult<EpochStats> {
        let mut history = TrainingHistory::new();
        self.train_epoch_internal(py, &queries, batch_size, usize::MAX, &mut history)
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
        let total_loss: f64 = probs.iter().map(|&p| nll_loss_value(p)).sum();
        Ok(total_loss / queries.len() as f64)
    }

    /// Perform forward pass through neural network and backward pass for gradients.
    ///
    /// This method supports two types of queries:
    /// 1. Direct neural predicate queries: `digit(0, 5)` - runs network on input 0, computes loss for label 5
    /// 2. Complex queries with neural predicates: `addition(0, 1, 7)` - expands neural predicates,
    ///    runs probabilistic inference, and backpropagates through the circuit to networks
    ///
    /// # Arguments
    /// * `query` - Query atom, e.g., "digit(0, 5)" or "addition(0, 1, 7)"
    ///
    /// # Returns
    /// The NLL loss value
    ///
    /// # Note
    /// Call zero_grad() before this and optimizer_step() after.
    #[pyo3(signature = (query, expected=true))]
    fn forward_backward(&mut self, py: Python<'_>, query: &str, expected: bool) -> PyResult<f64> {
        let loss = self.forward_backward_tensor_internal(py, query, expected)?;
        let loss_bound = loss.bind(py);
        loss_bound.call_method0("item")?.extract()
    }

    /// GPU-native forward-backward that returns the scalar NLL loss as a CUDA tensor (no `.item()` / `.tolist()`).
    ///
    /// This is the preferred API for strict GPU-native training loops. If you need a host `f64`,
    /// call `forward_backward(...)` which will read back a single scalar.
    #[pyo3(signature = (query, expected=true))]
    fn forward_backward_tensor(
        &mut self,
        py: Python<'_>,
        query: &str,
        expected: bool,
    ) -> PyResult<PyObject> {
        self.forward_backward_tensor_internal(py, query, expected)
    }

    /// Train for one epoch with GPU-native loss accumulation (no per-query .item()).
    #[pyo3(signature = (queries, batch_size=32))]
    fn train_epoch_tensor(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        batch_size: usize,
    ) -> PyResult<EpochStats> {
        let mut history = TrainingHistory::new();
        self.train_epoch_tensor_internal(py, &queries, batch_size, usize::MAX, &mut history)
    }

    /// Clear the circuit template cache, forcing recompilation on next query.
    /// Used for cache ablation benchmarks.
    fn clear_circuit_cache(&mut self) {
        self.circuit_cache.clear();
    }
}

impl CompiledProgram {
    fn forward_backward_tensor_internal(
        &mut self,
        py: Python<'_>,
        query: &str,
        expected: bool,
    ) -> PyResult<PyObject> {
        // Try to parse as a direct neural predicate query first.
        match self.try_parse_direct_neural_query(query) {
            Ok((predicate, network_name, input_idx, target_label)) => self
                .forward_backward_direct_tensor(
                    py,
                    &predicate,
                    &network_name,
                    input_idx,
                    &target_label,
                    expected,
                ),
            Err(_) => {
                let atom = self.parse_query_atom(query)?;
                self.forward_backward_complex_tensor(py, &atom, expected)
            }
        }
    }

    /// Internal implementation of train_epoch that tracks batch losses.
    pub(crate) fn train_epoch_internal(
        &mut self,
        py: Python<'_>,
        queries: &[String],
        batch_size: usize,
        log_iter: usize,
        history: &mut TrainingHistory,
    ) -> PyResult<EpochStats> {
        if queries.is_empty() {
            return Ok(EpochStats {
                avg_loss: 0.0,
                num_batches: 0,
                total_queries: 0,
            });
        }

        let num_batches = (queries.len() + batch_size - 1) / batch_size;
        let mut total_loss = 0.0;

        for (batch_idx, batch) in queries.chunks(batch_size).enumerate() {
            // Zero gradients at start of batch
            self.zero_grad(py)?;

            // Forward-backward for each query in batch
            let mut batch_loss = 0.0;
            for query in batch {
                batch_loss += self.forward_backward(py, query, true)?;
            }
            batch_loss /= batch.len() as f64;

            // Update parameters
            self.optimizer_step(py)?;

            total_loss += batch_loss;
            history.add_batch(batch_loss);

            // Log progress periodically
            if log_iter < usize::MAX && (batch_idx + 1) % log_iter == 0 {
                println!(
                    "  Batch {}/{}: loss={:.6}",
                    batch_idx + 1,
                    num_batches,
                    batch_loss
                );
            }
        }

        Ok(EpochStats {
            avg_loss: total_loss / num_batches as f64,
            num_batches,
            total_queries: queries.len(),
        })
    }

    /// GPU-native training epoch — no per-query .item() host sync.
    ///
    /// Accumulates batch loss as a CUDA tensor via torch.Tensor.add().
    /// Single .item() call per batch for logging/history only.
    pub(crate) fn train_epoch_tensor_internal(
        &mut self,
        py: Python<'_>,
        queries: &[String],
        batch_size: usize,
        log_iter: usize,
        history: &mut TrainingHistory,
    ) -> PyResult<EpochStats> {
        if queries.is_empty() {
            return Ok(EpochStats {
                avg_loss: 0.0,
                num_batches: 0,
                total_queries: 0,
            });
        }

        let num_batches = (queries.len() + batch_size - 1) / batch_size;
        let mut total_loss = 0.0;

        for (batch_idx, batch) in queries.chunks(batch_size).enumerate() {
            self.zero_grad(py)?;

            // Accumulate loss on device — no .item() per query
            let mut batch_loss_tensor: Option<PyObject> = None;
            for query in batch {
                let loss_t = self.forward_backward_tensor_internal(py, query, true)?;
                // Detach from computation graph (backward already called inside)
                let loss_val = loss_t.bind(py).call_method0("detach")?.unbind();
                batch_loss_tensor = Some(match batch_loss_tensor {
                    None => loss_val,
                    Some(acc) => {
                        acc.bind(py).call_method1("add_", (loss_val.bind(py),))?;
                        acc
                    }
                });
            }

            self.optimizer_step(py)?;

            // Single host sync per batch for logging
            let batch_loss_scalar: f64 = match batch_loss_tensor {
                Some(t) => {
                    let raw: f64 = t.bind(py).call_method0("item")?.extract()?;
                    raw / batch.len() as f64
                }
                None => 0.0,
            };

            total_loss += batch_loss_scalar;
            history.add_batch(batch_loss_scalar);

            if log_iter < usize::MAX && (batch_idx + 1) % log_iter == 0 {
                println!(
                    "  Batch {}/{}: loss={:.6}",
                    batch_idx + 1,
                    num_batches,
                    batch_loss_scalar
                );
            }
        }

        Ok(EpochStats {
            avg_loss: total_loss / num_batches as f64,
            num_batches,
            total_queries: queries.len(),
        })
    }

    /// Try to parse a query as a direct neural predicate query.
    ///
    /// Returns Ok((predicate_name, network_name, input_idx, target_label)) if the query is a direct neural predicate.
    /// Returns Err if the query is not a direct neural predicate (e.g., it's a complex query).
    fn try_parse_direct_neural_query(
        &self,
        query: &str,
    ) -> PyResult<(String, String, usize, String)> {
        let atom = self.parse_query_atom(query)?;
        let info = self
            .neural_registry
            .resolve_atom(&atom)
            .map_err(PyValueError::new_err)?;

        if info.input_arity != 1 {
            return Err(PyValueError::new_err(format!(
                "Predicate '{}' is associated with a {}-argument network, not a direct-call query",
                info.predicate, info.input_arity
            )));
        }

        let input_idx = self
            .term_to_input_idx(&atom.terms[info.input_positions[0]])
            .map_err(|_| PyValueError::new_err("Not a direct neural predicate query"))?;

        let target_label = self
            .term_to_label(&atom.terms[info.output_position])
            .map_err(|_| PyValueError::new_err("Not a direct neural predicate query"))?;

        Ok((
            atom.predicate,
            info.network.clone(),
            input_idx,
            target_label,
        ))
    }

    fn parse_query_atom(&self, query: &str) -> PyResult<Atom> {
        let source = format!("?- {}.", query.trim());
        let program = parse_program(&source).map_err(|e| {
            PyValueError::new_err(format!("Invalid query format '{}': {}", query, e))
        })?;
        if program.queries.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "Expected exactly one query, got {}",
                program.queries.len()
            )));
        }

        let atom = program.queries.into_iter().next().unwrap().atom;
        Ok(atom)
    }

    fn term_to_input_idx(&self, term: &Term) -> PyResult<usize> {
        match term {
            Term::Integer(idx) => usize::try_from(*idx)
                .map_err(|_| PyValueError::new_err(format!("Invalid input index: {}", idx))),
            _ => Err(PyValueError::new_err(format!(
                "Expected integer input index, got term {:?}",
                term
            ))),
        }
    }

    fn term_to_label(&self, term: &Term) -> PyResult<String> {
        match term {
            Term::Integer(v) => Ok(v.to_string()),
            Term::String(v) => Ok(v.clone()),
            Term::Symbol(id) => symbol::resolve_checked(*id)
                .ok_or_else(|| PyValueError::new_err("Invalid symbol id in query")),
            _ => Err(PyValueError::new_err("Expected constant label term")),
        }
    }

    /// Forward-backward for a direct neural predicate query.
    ///
    /// E.g., `digit(0, 5)` - runs network on input 0, computes NLL loss for label 5.
    fn forward_backward_direct_tensor(
        &self,
        py: Python<'_>,
        predicate: &str,
        network_name: &str,
        input_idx: usize,
        target_label: &str,
        expected: bool,
    ) -> PyResult<PyObject> {
        // Get the network handle
        let handle = self.network_registry.get(network_name).ok_or_else(|| {
            PyValueError::new_err(format!("Network '{}' not registered", network_name))
        })?;

        let module = handle.module().ok_or_else(|| {
            PyValueError::new_err(format!("Network '{}' has no module", network_name))
        })?;

        // Get input from tensor source
        let input_tensor = self.get_input_tensor(py, input_idx)?;

        // Forward pass through network (with gradient tracking).
        // Network expects batched input, so add batch dimension if needed.
        let input_bound = input_tensor.bind(py);
        let input_unsqueezed = input_bound.call_method1("unsqueeze", (0i32,))?;

        let output = module.call_method1(py, "__call__", (input_unsqueezed,))?;
        let output_bound = output.bind(py);

        // Output shape is [batch=1, num_classes], squeeze to [num_classes]
        let output_squeezed = output_bound.call_method1("squeeze", (0i32,))?;

        // Select the target probability tensor and compute loss on GPU:
        // loss = -log(clamp(prob, min=epsilon)).
        let label_idx = self.get_label_index(predicate, network_name, target_label)?;
        let prob_tensor = output_squeezed.get_item(label_idx)?;

        let torch = py.import_bound("torch")?;
        let clamp_kwargs = PyDict::new_bound(py);
        clamp_kwargs.set_item("min", NLL_EPSILON)?;
        let loss = if expected {
            let prob_clamped = prob_tensor.call_method("clamp", (), Some(&clamp_kwargs))?;
            let log_p = prob_clamped.call_method0("log")?;
            log_p.call_method0("__neg__")?
        } else {
            let device = prob_tensor.call_method0("device")?;
            let dtype = prob_tensor.call_method0("dtype")?;
            let one = torch.call_method1("tensor", (1.0f64,))?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("device", device)?;
            kwargs.set_item("dtype", dtype)?;
            let one_on_device = one.call_method("to", (), Some(&kwargs))?;
            let complement = one_on_device.call_method1("__sub__", (&prob_tensor,))?;
            let complement_clamped = complement.call_method("clamp", (), Some(&clamp_kwargs))?;
            let log_p = complement_clamped.call_method0("log")?;
            log_p.call_method0("__neg__")?
        };

        // Backprop through the network.
        loss.call_method0("backward")?;

        Ok(loss.into_py(py))
    }

    /// Forward-backward for a complex query involving neural predicates through rules.
    ///
    /// E.g., `addition(0, 1, 7)` where:
    /// - `addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.`
    /// - `nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).`
    ///
    /// This method uses circuit caching to avoid D4 recompilation:
    /// 1. Extracts input indices and runs neural networks
    /// 2. Generates template cache key from query structure
    /// 3. If cached: update weights and evaluate
    /// 4. If not cached: compile circuit, cache it, evaluate
    /// 5. Backpropagate gradients through networks
    fn forward_backward_complex_tensor(
        &mut self,
        py: Python<'_>,
        atom: &Atom,
        expected: bool,
    ) -> PyResult<PyObject> {
        let signature = self
            .get_or_build_query_signature(&atom.predicate, atom.terms.len())?
            .clone();
        let pred_name = atom.predicate.clone();

        let input_indices: Vec<usize> = signature
            .groups()
            .iter()
            .map(|group| match &group.input_source {
                InputSource::QueryArg(pos) => self.term_to_input_idx(&atom.terms[*pos]),
                InputSource::ImplicitSlot(slot) => Ok(*slot),
            })
            .collect::<PyResult<Vec<_>>>()?;
        if input_indices.is_empty() {
            return Err(PyValueError::new_err(format!(
                "No input indices found in query: {}. Make sure the query references neural predicate inputs.",
                atom.predicate
            )));
        }

        let target_label = match &signature {
            QuerySignature::Boolean { .. } => None,
            QuerySignature::Targeted {
                target_position, ..
            } => Some(self.term_to_label(&atom.terms[*target_position])?),
        };

        // Run neural networks and import the outputs as device-resident buffers via DLPack (no .tolist()).
        let torch = py.import_bound("torch")?;
        let schema_f32 = Schema::new(vec![("col0".to_string(), ScalarType::F32)]);
        let schema_f64 = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);

        #[derive(Clone)]
        struct NeuralCall {
            input_idx: usize,
            order_idx: usize,
        }

        let mut calls_by_network: HashMap<String, Vec<NeuralCall>> = HashMap::new();
        for (order_idx, &input_idx) in input_indices.iter().enumerate() {
            let network_name = signature
                .groups()
                .get(order_idx)
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "Invalid group index while building complex query call batches",
                    )
                })?
                .info
                .network
                .clone();
            calls_by_network
                .entry(network_name)
                .or_default()
                .push(NeuralCall {
                    input_idx,
                    order_idx,
                });
        }

        let mut out_tensors: Vec<Option<PyObject>> = std::iter::repeat_with(|| None)
            .take(input_indices.len())
            .collect();
        let mut grad_tensors: Vec<Option<PyObject>> = std::iter::repeat_with(|| None)
            .take(input_indices.len())
            .collect();
        let mut prob_bufs: Vec<Option<xlog_cuda::CudaBuffer>> = std::iter::repeat_with(|| None)
            .take(input_indices.len())
            .collect();
        let mut grad_bufs: Vec<Option<xlog_cuda::CudaBuffer>> = std::iter::repeat_with(|| None)
            .take(input_indices.len())
            .collect();

        for (network_name, calls) in calls_by_network {
            let handle = self.network_registry.get(&network_name).ok_or_else(|| {
                PyValueError::new_err(format!("Network '{}' not registered", network_name))
            })?;

            let module = handle.module().ok_or_else(|| {
                PyValueError::new_err(format!("Network '{}' has no module", network_name))
            })?;

            let mut inputs: Vec<PyObject> = Vec::with_capacity(calls.len());
            for call in &calls {
                inputs.push(self.get_input_tensor(py, call.input_idx)?);
            }

            let input_list = pyo3::types::PyList::new_bound(py, &inputs);
            let batch = torch.call_method1("stack", (input_list, 0i32))?;

            // Forward pass with gradient tracking (single batched forward).
            let output = module.call_method1(py, "__call__", (batch,))?;
            let output_bound = output.bind(py);

            for (batch_idx, call) in calls.iter().enumerate() {
                let output_row = output_bound.get_item(batch_idx)?;
                let output_row = output_row.call_method0("contiguous")?;

                let output_detached = output_row.call_method0("detach")?;
                let managed = dlpack_from_py(&output_detached)?;
                let prob_buf = self
                    .output_provider
                    .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![managed])
                    .map_err(|e| PyRuntimeError::new_err(format!("DLPack import failed: {}", e)))?;

                let grad_tensor = torch.call_method1("zeros_like", (&output_row,))?;
                let grad_tensor = grad_tensor.call_method0("contiguous")?;
                let grad_managed = dlpack_from_py(&grad_tensor)?;
                let grad_buf = self
                    .output_provider
                    .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![grad_managed])
                    .map_err(|e| PyRuntimeError::new_err(format!("DLPack import failed: {}", e)))?;

                out_tensors[call.order_idx] = Some(output_row.into());
                grad_tensors[call.order_idx] = Some(grad_tensor.into());
                prob_bufs[call.order_idx] = Some(prob_buf);
                grad_bufs[call.order_idx] = Some(grad_buf);
            }
        }

        let out_tensors: Vec<PyObject> = out_tensors
            .into_iter()
            .map(|v| v.ok_or_else(|| PyRuntimeError::new_err("Missing output tensor")))
            .collect::<PyResult<_>>()?;
        let grad_tensors: Vec<PyObject> = grad_tensors
            .into_iter()
            .map(|v| v.ok_or_else(|| PyRuntimeError::new_err("Missing grad tensor")))
            .collect::<PyResult<_>>()?;
        let prob_bufs: Vec<xlog_cuda::CudaBuffer> = prob_bufs
            .into_iter()
            .map(|v| v.ok_or_else(|| PyRuntimeError::new_err("Missing prob buffer")))
            .collect::<PyResult<_>>()?;
        let mut grad_bufs: Vec<xlog_cuda::CudaBuffer> = grad_bufs
            .into_iter()
            .map(|v| v.ok_or_else(|| PyRuntimeError::new_err("Missing grad buffer")))
            .collect::<PyResult<_>>()?;

        // Ensure template circuit is available (compile-once per shape).
        let cache_key =
            self.generate_cache_key_for_signature(&signature, &pred_name, atom.terms.len());
        if !self.circuit_cache.contains_key(&cache_key) {
            let cached =
                self.compile_circuit_for_template(&signature, &pred_name, atom.terms.len())?;
            self.circuit_cache.insert(cache_key.clone(), cached);
            self.template_compile_count = self.template_compile_count.saturating_add(1);
        }

        let cached = self
            .circuit_cache
            .get(&cache_key)
            .expect("cache populated above");

        let query_idx: usize = match target_label {
            Some(label) => cached
                .target_domain
                .iter()
                .position(|entry| entry == &label)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Target '{}' not in domain {:?} for predicate '{}'",
                        label, cached.target_domain, pred_name
                    ))
                })?,
            None => 0,
        };

        // Robustness: order Torch stream work vs XLOG's CUDA work-stream (legacy default stream).
        // Avoid global `cuda.synchronize()`; use stream-to-stream dependencies instead.
        if torch
            .getattr("cuda")
            .and_then(|c| c.call_method0("is_available"))
            .and_then(|b| b.extract::<bool>())
            .unwrap_or(false)
        {
            let cuda = torch.getattr("cuda")?;
            let current = cuda.call_method0("current_stream")?;
            let default_stream = cuda.call_method0("default_stream")?;
            default_stream.call_method1("wait_stream", (current,))?;
        }

        let cfg = NeuralFastPathConfig::default();
        let loss_dev = cached
            .program
            .neural_backward_nll_buffers_with_device_loss(
                &cached.slots,
                query_idx,
                &prob_bufs,
                &mut grad_bufs,
                cfg,
                expected,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Neural fast-path error: {}", e)))?;

        // `CudaBuffer` carries the row count on-device. For a single scalar loss, upload 1.
        let mut d_num_rows = self
            .output_provider
            .memory()
            .alloc::<u32>(1)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU allocation failed: {}", e)))?;
        self.output_provider
            .device()
            .inner()
            .htod_sync_copy_into(&[1u32], &mut d_num_rows)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set row count: {}", e)))?;

        let loss_buf = xlog_cuda::CudaBuffer::from_columns(
            vec![loss_dev.into_bytes().into()],
            1,
            d_num_rows,
            schema_f64,
        );
        let loss_dl = self
            .output_provider
            .to_dlpack_table(loss_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(format!("DLPack export failed: {}", e)))?;
        let loss_capsule = dlpack_capsule_from_tensor(py, loss_dl)?;
        let loss_tensor = torch.getattr("from_dlpack")?.call1((loss_capsule,))?;

        if torch
            .getattr("cuda")
            .and_then(|c| c.call_method0("is_available"))
            .and_then(|b| b.extract::<bool>())
            .unwrap_or(false)
        {
            let cuda = torch.getattr("cuda")?;
            let current = cuda.call_method0("current_stream")?;
            let default_stream = cuda.call_method0("default_stream")?;
            current.call_method1("wait_stream", (default_stream,))?;
        }

        // Backward through the networks using the device-resident gradients we filled on GPU.
        let last_idx = out_tensors.len().saturating_sub(1);
        for (idx, (out, grad)) in out_tensors.iter().zip(grad_tensors.iter()).enumerate() {
            let out_bound = out.bind(py);
            if idx == last_idx {
                out_bound.call_method1("backward", (grad.bind(py),))?;
            } else {
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("retain_graph", true)?;
                out_bound.call_method("backward", (grad.bind(py),), Some(&kwargs))?;
            }
        }

        Ok(loss_tensor.into_py(py))
    }

    fn get_or_build_query_signature(
        &mut self,
        pred_name: &str,
        arity: usize,
    ) -> PyResult<&QuerySignature> {
        let key = format!("{}:{}", pred_name, arity);
        if !self.query_signature_cache.contains_key(&key) {
            let signature = self.build_query_signature(pred_name, arity)?;
            self.query_signature_cache.insert(key.clone(), signature);
        }
        self.query_signature_cache
            .get(&key)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Failed to build signature for {key}")))
    }

    fn build_query_signature(&self, pred_name: &str, arity: usize) -> PyResult<QuerySignature> {
        let rule = self.find_query_rule(pred_name, arity)?;

        let mut groups: Vec<NeuralGroup> = Vec::with_capacity(rule.body.len());
        for literal in &rule.body {
            let body_atom = match literal {
                BodyLiteral::Positive(atom) => atom,
                _ => continue,
            };

            if self.neural_registry.get(&body_atom.predicate).is_none() {
                continue;
            }

            let info = self.match_neural_decl_for_atom(body_atom)?;

            if info.input_positions.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "Query signature currently supports exactly one input position per neural declaration; '{}' has {}",
                    info.predicate,
                    info.input_positions.len()
                )));
            }

            let input_position = info.input_positions[0];
            let input_term = body_atom.terms.get(input_position).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Malformed declaration for '{}': missing input variable at {}",
                    info.predicate, input_position
                ))
            })?;
            let input_var = match input_term {
                Term::Variable(name) => name.as_str(),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Expected variable at input position {} of neural call '{}'",
                        input_position, info.predicate
                    )))
                }
            };

            let input_source =
                if let Some(head_position) = Self::find_head_position(&rule.head, input_var) {
                    InputSource::QueryArg(head_position)
                } else {
                    let next_slot = groups
                        .iter()
                        .filter(|group| matches!(group.input_source, InputSource::ImplicitSlot(_)))
                        .count();
                    InputSource::ImplicitSlot(next_slot)
                };

            let _output_term = body_atom.terms.get(info.output_position).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Malformed declaration for '{}': missing output variable at {}",
                    info.predicate, info.output_position
                ))
            })?;
            #[cfg(feature = "host-io")]
            let output_var = match _output_term {
                Term::Variable(name) => Some(name.clone()),
                _ => None,
            };

            groups.push(NeuralGroup {
                info: info.clone(),
                input_source,
                #[cfg(feature = "host-io")]
                output_var,
            });
        }

        if groups.is_empty() {
            return Err(PyValueError::new_err(format!(
                "No neural groups found for query predicate '{}'",
                pred_name
            )));
        }

        if arity == 0 {
            return Ok(QuerySignature::Boolean { groups });
        }

        let mut used_head_positions: Vec<usize> = groups
            .iter()
            .filter_map(|group| match group.input_source {
                InputSource::QueryArg(pos) => Some(pos),
                InputSource::ImplicitSlot(_) => None,
            })
            .collect();
        used_head_positions.sort_unstable();
        used_head_positions.dedup();

        let target_positions: Vec<usize> = (0..arity)
            .filter(|pos| !used_head_positions.contains(pos))
            .collect();
        if target_positions.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "Could not determine unique target position for '{}': target positions {:?}",
                pred_name, target_positions
            )));
        }

        Ok(QuerySignature::Targeted {
            target_position: target_positions[0],
            groups,
        })
    }

    fn find_query_rule(&self, pred_name: &str, arity: usize) -> PyResult<&Rule> {
        let mut matches = 0usize;
        let mut found: Option<&Rule> = None;
        for rule in &self.ast.rules {
            if rule.head.predicate == pred_name && rule.head.arity() == arity {
                matches += 1;
                found = Some(rule);
            }
        }

        if matches == 0 {
            return Err(PyValueError::new_err(format!(
                "No rule defines predicate '{}' with arity {}",
                pred_name, arity
            )));
        }

        if matches > 1 {
            return Err(PyValueError::new_err(format!(
                "Query predicate '{}' has {} defining rules; expected exactly 1 for v0.4.0-alpha",
                pred_name, matches
            )));
        }

        found.ok_or_else(|| {
            PyValueError::new_err(format!(
                "Failed to locate defining rule for '{}' / arity {}",
                pred_name, arity
            ))
        })
    }

    fn match_neural_decl_for_atom(&self, atom: &Atom) -> PyResult<&NeuralPredicateInfo> {
        self.neural_registry
            .resolve_atom(atom)
            .map_err(PyValueError::new_err)
    }

    fn find_head_position(rule_head: &Atom, var_name: &str) -> Option<usize> {
        rule_head
            .terms
            .iter()
            .position(|term| matches!(term, Term::Variable(name) if name == var_name))
    }

    #[cfg(feature = "host-io")]
    fn generate_template_ast(
        &self,
        signature: &QuerySignature,
        query_pred: &str,
        query_arity: usize,
    ) -> PyResult<(xlog_logic::ast::Program, Vec<String>)> {
        let template_rule = self.find_query_rule(query_pred, query_arity)?;
        let mut template_program = xlog_logic::ast::Program::default();
        let mut target_domain = self.generate_target_domain(query_pred, query_arity, signature)?;

        for (group_idx, group) in signature.groups().iter().enumerate() {
            let labels = group.info.labels.as_ref().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Neural predicate '{}' does not declare labels",
                    group.info.predicate
                ))
            })?;
            if labels.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Neural predicate '{}' must declare at least one label",
                    group.info.predicate
                )));
            }

            let p = 0.9999999 / (labels.len() as f64);
            let mut choices = Vec::with_capacity(labels.len());
            for label in labels {
                let mut terms = Vec::with_capacity(group.info.predicate_arity);
                for pos in 0..group.info.predicate_arity {
                    if group.info.input_positions.contains(&pos) {
                        terms.push(Term::Integer(group_idx as i64));
                    } else if pos == group.info.output_position {
                        terms.push(self.label_string_to_term(label));
                    } else {
                        terms.push(group.info.predicate_terms[pos].clone());
                    }
                }

                choices.push(xlog_logic::ast::ProbFact {
                    prob: p,
                    atom: Atom {
                        predicate: group.info.predicate.clone(),
                        terms,
                    },
                });
            }

            template_program
                .annotated_disjunctions
                .push(xlog_logic::ast::AnnotatedDisjunction { choices });
        }

        template_program.rules.push(template_rule.clone());

        match signature {
            QuerySignature::Boolean { .. } => {
                template_program
                    .prob_queries
                    .push(xlog_logic::ast::ProbQuery {
                        atom: Atom {
                            predicate: query_pred.to_string(),
                            terms: Vec::new(),
                        },
                    });
            }
            QuerySignature::Targeted {
                target_position, ..
            } => {
                if target_domain.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "Targeted signature '{}' has empty target domain",
                        query_pred
                    )));
                }

                let mut head_to_group: StdHashMap<usize, usize> = StdHashMap::new();
                for (idx, group) in signature.groups().iter().enumerate() {
                    if let InputSource::QueryArg(pos) = group.input_source {
                        head_to_group.entry(pos).or_insert(idx);
                    }
                }

                for target in target_domain.iter() {
                    let mut terms = Vec::with_capacity(query_arity);
                    for pos in 0..query_arity {
                        if pos == *target_position {
                            terms.push(self.label_string_to_term(target));
                        } else if let Some(group_idx) = head_to_group.get(&pos) {
                            terms.push(Term::Integer(*group_idx as i64));
                        } else {
                            let head_term = template_rule
                                .head
                                .terms
                                .get(pos)
                                .ok_or_else(|| {
                                    PyValueError::new_err(format!(
                                        "Rule head position {} out of range in '{}' / arity {}",
                                        pos, query_pred, query_arity
                                    ))
                                })?
                                .clone();
                            terms.push(head_term);
                        }
                    }

                    template_program
                        .prob_queries
                        .push(xlog_logic::ast::ProbQuery {
                            atom: Atom {
                                predicate: query_pred.to_string(),
                                terms,
                            },
                        });
                }
            }
        }

        target_domain.sort_unstable();
        Ok((template_program, target_domain))
    }

    #[cfg(feature = "host-io")]
    fn generate_target_domain(
        &self,
        query_pred: &str,
        query_arity: usize,
        signature: &QuerySignature,
    ) -> PyResult<Vec<String>> {
        let rule = self.find_query_rule(query_pred, query_arity)?;

        match signature {
            QuerySignature::Boolean { .. } => Ok(Vec::new()),
            QuerySignature::Targeted {
                target_position,
                groups,
                ..
            } => {
                let target_term = rule
                    .head
                    .terms
                    .get(*target_position)
                    .ok_or_else(|| PyValueError::new_err("Target position is out of range"))?;
                let target_var = target_term.variable_name().ok_or_else(|| {
                    PyValueError::new_err("Target in query rule must be a variable")
                })?;

                let output_map: StdHashMap<&str, Vec<&NeuralGroup>> = groups
                    .iter()
                    .filter_map(|g| g.output_var.as_deref().map(|name| (name, g)))
                    .fold(StdHashMap::new(), |mut acc, (name, group)| {
                        acc.entry(name).or_default().push(group);
                        acc
                    });

                if let Some(domain) =
                    self.extract_addition_domain_from_rule(rule, target_var, &output_map)?
                {
                    return Ok(domain);
                }

                if let Some(groups_for_target) = output_map.get(target_var) {
                    let mut domain = groups_for_target
                        .iter()
                        .flat_map(|group| {
                            group
                                .info
                                .labels
                                .as_ref()
                                .into_iter()
                                .flat_map(|labels| labels.iter())
                        })
                        .map(ToString::to_string)
                        .collect::<Vec<_>>();
                    domain.sort_unstable();
                    domain.dedup();
                    if domain.is_empty() {
                        return Err(PyValueError::new_err(format!(
                            "No labels found for target variable '{}'",
                            target_var
                        )));
                    }
                    return Ok(domain);
                }

                Err(PyValueError::new_err(format!(
                    "Could not determine target domain for '{}'",
                    target_var
                )))
            }
        }
    }

    #[cfg(feature = "host-io")]
    fn extract_addition_domain_from_rule(
        &self,
        rule: &Rule,
        target_var: &str,
        output_map: &StdHashMap<&str, Vec<&NeuralGroup>>,
    ) -> PyResult<Option<Vec<String>>> {
        use std::collections::BTreeSet;
        for literal in &rule.body {
            let is_expr = match literal {
                BodyLiteral::IsExpr(is_expr) if is_expr.target == target_var => is_expr,
                _ => continue,
            };

            let (left_var, right_var) = match &is_expr.expr {
                ArithExpr::Add(lhs, rhs) => {
                    (self.extract_arith_var(lhs)?, self.extract_arith_var(rhs)?)
                }
                _ => {
                    return Ok(None);
                }
            };

            let left_groups = output_map.get(left_var.as_str()).ok_or_else(|| {
                PyValueError::new_err(format!("No neural group produces variable '{}'", left_var))
            })?;
            let right_groups = output_map.get(right_var.as_str()).ok_or_else(|| {
                PyValueError::new_err(format!("No neural group produces variable '{}'", right_var))
            })?;
            if left_groups.is_empty() || right_groups.is_empty() {
                return Err(PyValueError::new_err(
                    "Missing neural groups for target expression",
                ));
            }

            let left_labels = left_groups[0]
                .info
                .labels
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("Missing labels for source variable"))?;
            let right_labels = right_groups[0]
                .info
                .labels
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("Missing labels for source variable"))?;

            let mut values = BTreeSet::new();
            for left_label in left_labels {
                let left_value = left_label.parse::<i64>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Non-integer label '{}' in addition target domain",
                        left_label
                    ))
                })?;
                for right_label in right_labels {
                    let right_value = right_label.parse::<i64>().map_err(|_| {
                        PyValueError::new_err(format!(
                            "Non-integer label '{}' in addition target domain",
                            right_label
                        ))
                    })?;
                    values.insert((left_value + right_value).to_string());
                }
            }

            return Ok(Some(values.into_iter().collect()));
        }

        Ok(None)
    }

    #[cfg(feature = "host-io")]
    fn extract_arith_var(&self, expr: &ArithExpr) -> PyResult<String> {
        match expr {
            ArithExpr::Variable(name) => Ok(name.clone()),
            _ => Err(PyValueError::new_err(
                "Only variable terms are supported for target-domain arithmetic pattern",
            )),
        }
    }

    fn generate_cache_key_for_signature(
        &self,
        signature: &QuerySignature,
        query_pred: &str,
        query_arity: usize,
    ) -> String {
        let mut key = format!("{}:{}", query_pred, query_arity);
        for group in signature.groups() {
            let labels = group.info.labels.as_ref().map_or(0, |labels| labels.len());
            key.push(':');
            key.push_str(&group.info.network);
            key.push(':');
            key.push_str(&labels.to_string());
        }
        key
    }

    #[cfg(feature = "host-io")]
    fn label_string_to_term(&self, label: &str) -> Term {
        if let Ok(v) = label.parse::<i64>() {
            Term::Integer(v)
        } else {
            Term::Symbol(xlog_core::symbol::intern(label))
        }
    }

    fn compile_circuit_for_template(
        &self,
        signature: &QuerySignature,
        query_pred: &str,
        query_arity: usize,
    ) -> PyResult<CachedCircuit> {
        #[cfg(not(feature = "host-io"))]
        {
            let _ = (signature, query_pred, query_arity);
            return Err(host_io_disabled_pyerr());
        }

        #[cfg(feature = "host-io")]
        {
            let (expanded_ast, target_domain) =
                self.generate_template_ast(signature, query_pred, query_arity)?;
            let program = ExactDdnnfProgram::compile_from_program(&expanded_ast, self._gpu_config)
                .map_err(|e| PyRuntimeError::new_err(format!("Query compilation error: {}", e)))?;

            let random_vars = program.random_var_indices();
            let group_label_counts: Vec<usize> = signature
                .groups()
                .iter()
                .map(|g| g.info.labels.as_ref().map(|l| l.len()).unwrap_or(0))
                .collect();

            let expected_total: usize = group_label_counts.iter().sum();
            if random_vars.len() != expected_total {
                return Err(PyRuntimeError::new_err(format!(
                    "Template compilation produced {} random vars, expected {} (groups: {:?})",
                    random_vars.len(),
                    expected_total,
                    group_label_counts
                )));
            }

            let mut slot_groups = Vec::with_capacity(group_label_counts.len());
            let mut offset = 0usize;
            for count in group_label_counts.iter() {
                let end = offset + *count;
                if end > random_vars.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "Template compilation random vars exhausted at offset {} count {}",
                        offset, count
                    )));
                }
                slot_groups.push(random_vars[offset..end].to_vec());
                offset = end;
            }

            let slots = GpuWeightSlots::upload(self.output_provider.as_ref(), &slot_groups)
                .map_err(|e| PyRuntimeError::new_err(format!("Slot map upload error: {}", e)))?;

            Ok(CachedCircuit {
                program,
                slots,
                target_domain,
            })
        }
    }
    /// Get the label index for a given label string from declared labels.
    fn get_label_index(&self, predicate: &str, network: &str, label: &str) -> PyResult<usize> {
        let infos = self.neural_registry.get(predicate).ok_or_else(|| {
            PyValueError::new_err(format!("Unknown neural predicate '{}'", predicate))
        })?;
        let info = infos.iter().find(|i| i.network == network).ok_or_else(|| {
            PyValueError::new_err(format!(
                "No nn/4 declaration for predicate '{}' with network '{}'",
                predicate, network
            ))
        })?;
        let labels = info.labels.as_ref().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Predicate '{}' does not declare a label list in nn/4",
                predicate
            ))
        })?;
        labels
            .iter()
            .position(|l| l == label)
            .ok_or_else(|| PyValueError::new_err(format!("Label '{}' not in declared list", label)))
    }

    /// Get input tensor for a given index from the active tensor source.
    fn get_input_tensor(&self, py: Python<'_>, index: usize) -> PyResult<PyObject> {
        let tensor = self
            .tensor_sources
            .get_active()
            .map_err(|e| PyValueError::new_err(format!("No active tensor source: {}", e)))?;

        // Index into the tensor: tensor[index]
        let tensor_bound = tensor.bind(py);
        let indexed = tensor_bound.get_item(index)?;
        Ok(indexed.into())
    }

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
            return Err(host_io_disabled_pyerr());
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
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Query compilation error: {}", e))
                    })?;

                    program
                        .evaluate()
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("Query evaluation error: {}", e))
                        })?
                        .query_probs
                }
                ProbEngine::Mc => {
                    let program =
                        McProgram::compile_source_with_gpu(&source_with_queries, self._gpu_config)
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("Query compilation error: {}", e))
                            })?;

                    let cfg = McEvalConfig::default();
                    program
                        .evaluate(cfg)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("Query evaluation error: {}", e))
                        })?
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
            .create_buffer_from_f64_slice(&probs, schema.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_f64_slice(&log_probs, schema)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
                .create_buffer_from_f64_slice(&q.grad_true, schema.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let grad_false_buf = self
                .output_provider
                .create_buffer_from_f64_slice(&q.grad_false, schema.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let grad_true_tensor = self
                .output_provider
                .to_dlpack_table(grad_true_buf)
                .column(0)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let grad_false_tensor = self
                .output_provider
                .to_dlpack_table(grad_false_buf)
                .column(0)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            grad_true_caps.push(dlpack_capsule_from_tensor(py, grad_true_tensor)?);
            grad_false_caps.push(dlpack_capsule_from_tensor(py, grad_false_tensor)?);
        }

        let prob_buf = self
            .output_provider
            .create_buffer_from_f64_slice(&probs, schema.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_f64_slice(&log_probs, schema)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
            .create_buffer_from_f64_slice(&probs, schema.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let log_prob_buf = self
            .output_provider
            .create_buffer_from_f64_slice(&log_probs, schema.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let stderr_buf = self
            .output_provider
            .create_buffer_from_f64_slice(&stderrs, schema.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let ci_low_buf = self
            .output_provider
            .create_buffer_from_f64_slice(&ci_lows, schema.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let ci_high_buf = self
            .output_provider
            .create_buffer_from_f64_slice(&ci_highs, schema)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let prob_tensor = self
            .output_provider
            .to_dlpack_table(prob_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let log_prob_tensor = self
            .output_provider
            .to_dlpack_table(log_prob_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let stderr_tensor = self
            .output_provider
            .to_dlpack_table(stderr_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let ci_low_tensor = self
            .output_provider
            .to_dlpack_table(ci_low_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let ci_high_tensor = self
            .output_provider
            .to_dlpack_table(ci_high_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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
        })
    }
}

#[pyclass]
pub struct LogicProgram;

#[pymethods]
impl LogicProgram {
    #[staticmethod]
    #[pyo3(signature = (source, device=0, memory_mb=32768))]
    pub fn compile(source: &str, device: usize, memory_mb: u64) -> PyResult<CompiledLogicProgram> {
        if memory_mb == 0 {
            return Err(PyValueError::new_err("memory_mb must be > 0"));
        }

        let config = GpuConfig {
            device_ordinal: device,
            memory_bytes: memory_mb * 1024 * 1024,
        };

        let program = gpu_logic::LogicProgram::compile(source)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let provider =
            provider_from_config(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(CompiledLogicProgram {
            program,
            provider: Arc::new(provider),
        })
    }
}

#[pyclass(unsendable)]
pub struct CompiledLogicProgram {
    program: gpu_logic::LogicProgram,
    provider: Arc<CudaKernelProvider>,
}

#[pymethods]
impl CompiledLogicProgram {
    #[pyo3(signature = (dlpack_inputs=None))]
    pub fn evaluate(
        &self,
        py: Python<'_>,
        dlpack_inputs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<LogicEvalResult> {
        let mut inputs: HashMap<String, xlog_cuda::CudaBuffer> = HashMap::new();

        if let Some(dict) = dlpack_inputs {
            for (k, v) in dict.iter() {
                let name: String = k.extract()?;
                let schema = self.program.schema(&name).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Unknown input relation {} (not present in compiled schemas)",
                        name
                    ))
                })?;

                let seq = v.downcast::<PySequence>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Input relation {} must be a sequence of DLPack columns",
                        name
                    ))
                })?;

                let mut tensors: Vec<DlpackManagedTensor> = Vec::with_capacity(seq.len()? as usize);
                for item in seq.iter()? {
                    let item = item?;
                    tensors.push(dlpack_from_py(&item)?);
                }

                let buffer = self
                    .provider
                    .from_dlpack_tensors_with_schema(schema.clone(), tensors)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                inputs.insert(name, buffer);
            }
        }

        let result = self
            .program
            .evaluate(self.provider.clone(), inputs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.pack_logic_result(py, result)
    }
}

impl CompiledLogicProgram {
    fn pack_logic_result(
        &self,
        py: Python<'_>,
        result: gpu_logic::LogicEvalResult,
    ) -> PyResult<LogicEvalResult> {
        let mut queries: Vec<Py<LogicQueryResult>> = Vec::with_capacity(result.queries.len());

        for q in result.queries {
            let num_rows = q.buffer.num_rows() as usize;
            let is_true = q.columns.is_empty() && num_rows > 0;
            let arity = q.buffer.arity();
            let mut tensors: Vec<PyObject> = Vec::new();
            if !q.columns.is_empty() {
                let table = self.provider.to_dlpack_table(q.buffer);
                tensors.reserve(arity);
                for col_idx in 0..arity {
                    let tensor = table
                        .column(col_idx)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    tensors.push(dlpack_capsule_from_tensor(py, tensor)?);
                }
            }

            queries.push(Py::new(
                py,
                LogicQueryResult {
                    relation_name: q.relation_name,
                    columns: q.columns,
                    tensors,
                    num_rows,
                    is_true,
                },
            )?);
        }

        Ok(LogicEvalResult { queries })
    }
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
    /// Loss for each batch across all epochs
    #[pyo3(get)]
    pub batch_losses: Vec<f64>,
}

impl TrainingHistory {
    fn new() -> Self {
        Self {
            epoch_losses: Vec::new(),
            batch_losses: Vec::new(),
        }
    }

    fn add_epoch(&mut self, loss: f64) {
        self.epoch_losses.push(loss);
    }

    fn add_batch(&mut self, loss: f64) {
        self.batch_losses.push(loss);
    }
}

/// Train a program for multiple epochs.
///
/// This is the main training entry point that runs the full training loop:
/// - For each epoch: shuffle queries (optional), process batches, record stats
/// - Supports learning rate scheduling via scheduler_step() after each epoch
///
/// # Arguments
/// * `program` - Compiled program with registered networks
/// * `queries` - Training queries (e.g., ["addition(0, 1, 5)", "addition(2, 3, 7)")
/// * `epochs` - Number of training epochs (default: 10)
/// * `batch_size` - Number of queries per batch (default: 32)
/// * `log_iter` - Log progress every N batches (default: 100)
/// * `shuffle` - Whether to shuffle queries each epoch (default: true)
///
/// # Returns
/// TrainingHistory with epoch and batch losses
#[pyfunction]
#[pyo3(signature = (program, queries, epochs=10, batch_size=32, log_iter=100, shuffle=true))]
pub fn train_model(
    py: Python<'_>,
    program: &mut CompiledProgram,
    queries: Vec<String>,
    epochs: usize,
    batch_size: usize,
    log_iter: usize,
    shuffle: bool,
) -> PyResult<TrainingHistory> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut history = TrainingHistory::new();

    for epoch in 0..epochs {
        let mut epoch_queries = queries.clone();

        if shuffle {
            let mut rng = thread_rng();
            epoch_queries.shuffle(&mut rng);
        }

        let stats =
            program.train_epoch_internal(py, &epoch_queries, batch_size, log_iter, &mut history)?;
        history.add_epoch(stats.avg_loss);

        // Print epoch progress (visible in Python output)
        println!(
            "Epoch {}/{}: avg_loss={:.6}",
            epoch + 1,
            epochs,
            stats.avg_loss
        );
        // Flush stdout so epoch progress is visible when output is redirected to a file
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    Ok(history)
}

/// GPU-native training loop — no per-query host synchronization.
///
/// Identical to train_model but uses forward_backward_tensor internally.
/// Loss stays on CUDA device; .item() called once per batch only.
#[pyfunction]
#[pyo3(signature = (program, queries, epochs=10, batch_size=32, log_iter=100, shuffle=true))]
pub fn train_model_tensor(
    py: Python<'_>,
    program: &mut CompiledProgram,
    queries: Vec<String>,
    epochs: usize,
    batch_size: usize,
    log_iter: usize,
    shuffle: bool,
) -> PyResult<TrainingHistory> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut history = TrainingHistory::new();

    for epoch in 0..epochs {
        let mut epoch_queries = queries.clone();

        if shuffle {
            let mut rng = thread_rng();
            epoch_queries.shuffle(&mut rng);
        }

        let stats =
            program.train_epoch_tensor_internal(py, &epoch_queries, batch_size, log_iter, &mut history)?;
        history.add_epoch(stats.avg_loss);

        println!(
            "Epoch {}/{}: avg_loss={:.6}",
            epoch + 1,
            epochs,
            stats.avg_loss
        );
        // Flush stdout so epoch progress is visible when output is redirected to a file
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    Ok(history)
}

#[pymodule]
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
    m.add_function(wrap_pyfunction!(train_model, m)?)?;
    m.add_function(wrap_pyfunction!(train_model_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(dlpack_roundtrip, m)?)?;
    #[cfg(feature = "arrow-device-import")]
    m.add_function(wrap_pyfunction!(export_arrow_device, m)?)?;
    #[cfg(feature = "arrow-device-import")]
    m.add_function(wrap_pyfunction!(import_arrow_device, m)?)?;
    Ok(())
}
