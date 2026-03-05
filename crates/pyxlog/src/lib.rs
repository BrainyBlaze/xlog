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

use xlog_core::RelId;
use xlog_logic::ast::Program as AstProgram;
use xlog_runtime::{Executor, read_device_row_count};
use xlog_ir::{ExecutionPlan, RirNode};
use xlog_cuda::JoinType;

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

fn scalar_type_name(typ: &ScalarType) -> String {
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
            batch_queries: true,
            last_compile_profile: None,
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

#[pyclass]
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
    /// When true, batch queries sharing the same circuit template in training.
    batch_queries: bool,
    /// Latest circuit compilation profile (populated on cache miss when profiling).
    last_compile_profile: Option<xlog_prob::compilation::CircuitCompileProfile>,
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

    /// Enable or disable multi-query batching for training.
    ///
    /// When enabled (default), queries sharing the same circuit template are
    /// grouped and processed with batched network forward/backward passes,
    /// reducing kernel launch, DLPack, and GIL transition overhead.
    #[pyo3(signature = (enabled=true))]
    fn set_batch_queries(&mut self, enabled: bool) {
        self.batch_queries = enabled;
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

            if self.batch_queries {
                // ── Batched path: group queries by template ─────────────
                // Use insertion-ordered grouping so gradient accumulation
                // ordering is deterministic across runs.
                let mut complex_group_order: Vec<String> = Vec::new();
                let mut complex_groups: StdHashMap<String, Vec<Atom>> = StdHashMap::new();

                for query in batch {
                    match self.try_parse_direct_neural_query(query) {
                        Ok((predicate, network_name, input_idx, target_label)) => {
                            // Direct queries: process individually (pure PyTorch, no circuit)
                            let loss = self.forward_backward_direct_tensor(
                                py,
                                &predicate,
                                &network_name,
                                input_idx,
                                &target_label,
                                true,
                            )?;
                            let loss_val = loss.bind(py).call_method0("detach")?.unbind();
                            batch_loss_tensor = Some(match batch_loss_tensor {
                                None => loss_val,
                                Some(acc) => {
                                    acc.bind(py).call_method1("add_", (loss_val.bind(py),))?;
                                    acc
                                }
                            });
                        }
                        Err(_) => {
                            // Complex query: group by template for batching.
                            let atom = self.parse_query_atom(query)?;
                            let sig = self
                                .get_or_build_query_signature(&atom.predicate, atom.terms.len())?
                                .clone();
                            let key = self.generate_cache_key_for_signature(
                                &sig,
                                &atom.predicate,
                                atom.terms.len(),
                            );
                            if !complex_groups.contains_key(&key) {
                                complex_group_order.push(key.clone());
                            }
                            complex_groups.entry(key).or_default().push(atom);
                        }
                    }
                }

                // Batch-process each complex group in insertion order.
                for key in &complex_group_order {
                    let group = complex_groups.remove(key).unwrap();
                    let loss = self.forward_backward_batch_complex_tensor(py, &group, true)?;
                    let loss_val = loss.bind(py).call_method0("detach")?.unbind();
                    batch_loss_tensor = Some(match batch_loss_tensor {
                        None => loss_val,
                        Some(acc) => {
                            acc.bind(py).call_method1("add_", (loss_val.bind(py),))?;
                            acc
                        }
                    });
                }
            } else {
                // ── Sequential path (for regression testing / fallback) ─
                for query in batch {
                    let loss_t = self.forward_backward_tensor_internal(py, query, true)?;
                    let loss_val = loss_t.bind(py).call_method0("detach")?.unbind();
                    batch_loss_tensor = Some(match batch_loss_tensor {
                        None => loss_val,
                        Some(acc) => {
                            acc.bind(py).call_method1("add_", (loss_val.bind(py),))?;
                            acc
                        }
                    });
                }
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
            let (cached, profile) =
                self.compile_circuit_for_template(&signature, &pred_name, atom.terms.len())?;
            self.circuit_cache.insert(cache_key.clone(), cached);
            self.template_compile_count = self.template_compile_count.saturating_add(1);
            if profile.is_some() {
                self.last_compile_profile = profile;
            }
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
        // Release the GIL during GPU circuit evaluation + backward pass.
        // This lets Python threads (e.g. data loaders) run while CUDA kernels execute.
        let loss_dev = py
            .allow_threads(|| {
                cached.program.neural_backward_nll_buffers_with_device_loss(
                    &cached.slots,
                    query_idx,
                    &prob_bufs,
                    &mut grad_bufs,
                    cfg,
                    expected,
                )
            })
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

    /// Batch-process multiple queries that share the same circuit template.
    ///
    /// Instead of N separate forward passes, DLPack cycles, stream syncs, and
    /// backward passes, this method:
    /// 1. Stacks inputs per-network → single batched forward pass
    /// 2. Performs per-query circuit evaluation (same API, just looped)
    /// 3. Single stream sync before/after the circuit eval batch
    /// 4. Uses `torch.autograd.backward(outputs, grads)` for one backward pass
    fn forward_backward_batch_complex_tensor(
        &mut self,
        py: Python<'_>,
        atoms: &[Atom],
        expected: bool,
    ) -> PyResult<PyObject> {
        let n_queries = atoms.len();
        if n_queries == 0 {
            return Err(PyRuntimeError::new_err(
                "forward_backward_batch_complex_tensor called with empty atoms slice",
            ));
        }
        let n_queries_u32 = u32::try_from(n_queries)
            .map_err(|_| PyRuntimeError::new_err("Query batch size exceeds u32"))?;

        // ── 1. Shared setup from first atom ─────────────────────────────
        let signature = self
            .get_or_build_query_signature(&atoms[0].predicate, atoms[0].terms.len())?
            .clone();
        let n_groups = signature.groups().len();
        let pred_name = atoms[0].predicate.clone();

        // Ensure circuit is compiled/cached.
        let cache_key =
            self.generate_cache_key_for_signature(&signature, &pred_name, atoms[0].terms.len());
        if !self.circuit_cache.contains_key(&cache_key) {
            let (cached, profile) =
                self.compile_circuit_for_template(&signature, &pred_name, atoms[0].terms.len())?;
            self.circuit_cache.insert(cache_key.clone(), cached);
            self.template_compile_count = self.template_compile_count.saturating_add(1);
            if profile.is_some() {
                self.last_compile_profile = profile;
            }
        }

        // ── 2. Per-query data: input indices + query_idx ────────────────
        let mut per_query_inputs: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
        let mut per_query_idx: Vec<usize> = Vec::with_capacity(n_queries);

        for atom in atoms {
            let input_indices: Vec<usize> = signature
                .groups()
                .iter()
                .map(|group| match &group.input_source {
                    InputSource::QueryArg(pos) => self.term_to_input_idx(&atom.terms[*pos]),
                    InputSource::ImplicitSlot(slot) => Ok(*slot),
                })
                .collect::<PyResult<Vec<_>>>()?;

            let target_label = match &signature {
                QuerySignature::Boolean { .. } => None,
                QuerySignature::Targeted {
                    target_position, ..
                } => Some(self.term_to_label(&atom.terms[*target_position])?),
            };

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

            per_query_inputs.push(input_indices);
            per_query_idx.push(query_idx);
        }

        // ── 3. Group network calls: network → [(query, group, input_idx)] ─
        // Insertion-ordered Vec for deterministic gradient accumulation.
        struct NetCall {
            query: usize,
            group: usize,
            input_idx: usize,
        }
        let mut calls_by_network: Vec<(String, Vec<NetCall>)> = Vec::new();
        let mut net_index: StdHashMap<String, usize> = StdHashMap::new();

        for (q, inputs) in per_query_inputs.iter().enumerate() {
            for (g, group) in signature.groups().iter().enumerate() {
                let name = &group.info.network;
                let idx = match net_index.get(name) {
                    Some(&i) => i,
                    None => {
                        let i = calls_by_network.len();
                        net_index.insert(name.clone(), i);
                        calls_by_network.push((name.clone(), Vec::new()));
                        i
                    }
                };
                calls_by_network[idx].1.push(NetCall {
                    query: q,
                    group: g,
                    input_idx: inputs[g],
                });
            }
        }

        // ── 4. Batched forward per network ──────────────────────────────
        let torch = py.import_bound("torch")?;
        let schema_f32 = Schema::new(vec![("col0".to_string(), ScalarType::F32)]);
        let schema_f64 = Schema::new(vec![("col0".to_string(), ScalarType::F64)]);

        // Storage indexed by (query, group).
        let mut prob_map: StdHashMap<(usize, usize), xlog_cuda::CudaBuffer> = StdHashMap::new();
        let mut grad_map: StdHashMap<(usize, usize), xlog_cuda::CudaBuffer> = StdHashMap::new();
        // All output rows + grad tensors for the batched backward.
        let mut all_out_tensors: Vec<PyObject> = Vec::new();
        let mut all_grad_tensors: Vec<PyObject> = Vec::new();

        for (net_name, calls) in &calls_by_network {
            let handle = self.network_registry.get(net_name).ok_or_else(|| {
                PyValueError::new_err(format!("Network '{}' not registered", net_name))
            })?;
            let module = handle.module().ok_or_else(|| {
                PyValueError::new_err(format!("Network '{}' has no module", net_name))
            })?;

            // Stack all inputs for this network into a single batch.
            let mut inputs: Vec<PyObject> = Vec::with_capacity(calls.len());
            for c in calls {
                inputs.push(self.get_input_tensor(py, c.input_idx)?);
            }
            let input_list = pyo3::types::PyList::new_bound(py, &inputs);
            let stacked = torch.call_method1("stack", (input_list, 0i32))?;

            // Single batched forward: [N_total, num_classes]
            let batch_output = module.call_method1(py, "__call__", (stacked,))?;
            let batch_output_bound = batch_output.bind(py);

            // Slice each row, validate label count, create DLPack buffers.
            for (i, call) in calls.iter().enumerate() {
                let row = batch_output_bound.get_item(i)?;
                let row = row.call_method0("contiguous")?;

                // Validate network output width matches declared labels.
                let n: usize = row
                    .call_method0("numel")?
                    .extract::<i64>()?
                    .try_into()
                    .map_err(|_| PyValueError::new_err("Invalid output numel"))?;
                let expected_labels = signature
                    .groups()
                    .get(call.group)
                    .and_then(|g| g.info.labels.as_ref())
                    .map(|l| l.len())
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "No declared labels for group {}",
                            call.group
                        ))
                    })?;
                if n != expected_labels {
                    return Err(PyValueError::new_err(format!(
                        "Network output size {} does not match declared label count {} for group {}",
                        n, expected_labels, call.group
                    )));
                }

                // DLPack import: detach for prob values, keep original for autograd.
                let row_detached = row.call_method0("detach")?;
                let managed = dlpack_from_py(&row_detached)?;
                let prob_buf = self
                    .output_provider
                    .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![managed])
                    .map_err(|e| PyRuntimeError::new_err(format!("DLPack import failed: {}", e)))?;

                let grad_tensor = torch.call_method1("zeros_like", (&row,))?;
                let grad_tensor = grad_tensor.call_method0("contiguous")?;
                let grad_managed = dlpack_from_py(&grad_tensor)?;
                let grad_buf = self
                    .output_provider
                    .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![grad_managed])
                    .map_err(|e| PyRuntimeError::new_err(format!("DLPack import failed: {}", e)))?;

                prob_map.insert((call.query, call.group), prob_buf);
                grad_map.insert((call.query, call.group), grad_buf);
                all_out_tensors.push(row.into());
                all_grad_tensors.push(grad_tensor.into());
            }
        }

        // Arrange into per-query Vec<CudaBuffer> in group order.
        let mut per_query_probs: Vec<Vec<xlog_cuda::CudaBuffer>> = Vec::with_capacity(n_queries);
        let mut per_query_grads: Vec<Vec<xlog_cuda::CudaBuffer>> = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let mut probs = Vec::with_capacity(n_groups);
            let mut grads = Vec::with_capacity(n_groups);
            for g in 0..n_groups {
                probs.push(prob_map.remove(&(q, g)).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "Missing prob buffer for query {} group {}",
                        q, g
                    ))
                })?);
                grads.push(grad_map.remove(&(q, g)).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "Missing grad buffer for query {} group {}",
                        q, g
                    ))
                })?);
            }
            per_query_probs.push(probs);
            per_query_grads.push(grads);
        }

        // ── 5. Stream sync: torch current → default (once) ─────────────
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

        // ── 6. Per-query circuit evaluation ─────────────────────────────
        // Collect device-resident loss scalars first; DLPack export happens
        // AFTER the stream sync to avoid a race between XLOG's default
        // stream writes and PyTorch's current stream reads.
        let cfg = NeuralFastPathConfig::default();
        let cached = self
            .circuit_cache
            .get(&cache_key)
            .expect("cache populated above");

        let batched_loss_dev = py.allow_threads(|| {
            cached
                .program
                .neural_backward_nll_buffers_batch_with_device_loss(
                    &cached.slots,
                    &per_query_idx,
                    &per_query_probs,
                    &mut per_query_grads,
                    cfg,
                    expected,
                )
        });

        // ── 7. Stream sync: default → torch current (once) ─────────────
        // All circuit work (loss writes + grad fills) is on XLOG's default
        // stream. Sync before handing memory to PyTorch or calling backward.
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

        // ── 7b. Export losses via DLPack and accumulate on device ────────
        let batch_loss_tensor: PyObject = match batched_loss_dev {
            Ok(loss_dev) => {
                let mut d_num_rows =
                    self.output_provider.memory().alloc::<u32>(1).map_err(|e| {
                        PyRuntimeError::new_err(format!("GPU allocation failed: {}", e))
                    })?;
                self.output_provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&[n_queries_u32], &mut d_num_rows)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to set row count: {}", e))
                    })?;

                let loss_buf = xlog_cuda::CudaBuffer::from_columns(
                    vec![loss_dev.into_bytes().into()],
                    n_queries_u32 as u64,
                    d_num_rows,
                    schema_f64.clone(),
                );
                let loss_dl = self
                    .output_provider
                    .to_dlpack_table(loss_buf)
                    .column(0)
                    .map_err(|e| PyRuntimeError::new_err(format!("DLPack export failed: {}", e)))?;
                let loss_capsule = dlpack_capsule_from_tensor(py, loss_dl)?;
                let loss_tensor = torch.getattr("from_dlpack")?.call1((loss_capsule,))?;
                loss_tensor.call_method0("sum")?.into_py(py)
            }
            Err(_batch_err) => {
                // Fallback path: preserve prior semantics if batched circuit path
                // is unavailable for this circuit.
                let mut accum: Option<PyObject> = None;
                for q in 0..n_queries {
                    let loss_dev = py
                        .allow_threads(|| {
                            cached.program.neural_backward_nll_buffers_with_device_loss(
                                &cached.slots,
                                per_query_idx[q],
                                &per_query_probs[q],
                                &mut per_query_grads[q],
                                cfg,
                                expected,
                            )
                        })
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("Neural fast-path error: {}", e))
                        })?;

                    let mut d_num_rows =
                        self.output_provider.memory().alloc::<u32>(1).map_err(|e| {
                            PyRuntimeError::new_err(format!("GPU allocation failed: {}", e))
                        })?;
                    self.output_provider
                        .device()
                        .inner()
                        .htod_sync_copy_into(&[1u32], &mut d_num_rows)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("Failed to set row count: {}", e))
                        })?;

                    let loss_buf = xlog_cuda::CudaBuffer::from_columns(
                        vec![loss_dev.into_bytes().into()],
                        1,
                        d_num_rows,
                        schema_f64.clone(),
                    );
                    let loss_dl = self
                        .output_provider
                        .to_dlpack_table(loss_buf)
                        .column(0)
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("DLPack export failed: {}", e))
                        })?;
                    let loss_capsule = dlpack_capsule_from_tensor(py, loss_dl)?;
                    let loss_tensor = torch.getattr("from_dlpack")?.call1((loss_capsule,))?;

                    accum = Some(match accum {
                        None => loss_tensor.into_py(py),
                        Some(acc) => {
                            acc.bind(py).call_method1("add_", (loss_tensor,))?;
                            acc
                        }
                    });
                }
                accum.ok_or_else(|| PyRuntimeError::new_err("No loss computed in batch"))?
            }
        };

        // ── 8. Batched backward through all networks ────────────────────
        // torch.autograd.backward(tensors, grad_tensors) — single backward pass
        // through the shared batched computation graph.
        let autograd = torch.getattr("autograd")?;
        let out_list = pyo3::types::PyList::new_bound(py, all_out_tensors.iter());
        let grad_list = pyo3::types::PyList::new_bound(py, all_grad_tensors.iter());
        autograd.call_method1("backward", (out_list, grad_list))?;

        // ── 9. Return accumulated loss ──────────────────────────────────
        Ok(batch_loss_tensor)
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
    ) -> PyResult<(CachedCircuit, Option<xlog_prob::compilation::CircuitCompileProfile>)> {
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

            let compile_profile = program.last_compile_profile().cloned();

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

            Ok((CachedCircuit {
                program,
                slots,
                target_domain,
            }, compile_profile))
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

#[pyclass]
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
    /// Wall-clock time (seconds) for each epoch
    #[pyo3(get)]
    pub epoch_times: Vec<f64>,
    /// Loss for each batch across all epochs
    #[pyo3(get)]
    pub batch_losses: Vec<f64>,
}

impl TrainingHistory {
    fn new() -> Self {
        Self {
            epoch_losses: Vec::new(),
            epoch_times: Vec::new(),
            batch_losses: Vec::new(),
        }
    }

    fn add_epoch(&mut self, loss: f64, epoch_time_sec: f64) {
        self.epoch_losses.push(loss);
        self.epoch_times.push(epoch_time_sec);
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
    use std::time::Instant;

    let mut history = TrainingHistory::new();

    for epoch in 0..epochs {
        let mut epoch_queries = queries.clone();

        if shuffle {
            let mut rng = thread_rng();
            epoch_queries.shuffle(&mut rng);
        }

        let epoch_start = Instant::now();
        let stats =
            program.train_epoch_internal(py, &epoch_queries, batch_size, log_iter, &mut history)?;
        history.add_epoch(stats.avg_loss, epoch_start.elapsed().as_secs_f64());

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
    use std::time::Instant;

    let mut history = TrainingHistory::new();

    for epoch in 0..epochs {
        let mut epoch_queries = queries.clone();

        if shuffle {
            let mut rng = thread_rng();
            epoch_queries.shuffle(&mut rng);
        }

        let epoch_start = Instant::now();
        let stats = program.train_epoch_tensor_internal(
            py,
            &epoch_queries,
            batch_size,
            log_iter,
            &mut history,
        )?;
        history.add_epoch(stats.avg_loss, epoch_start.elapsed().as_secs_f64());

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

// ---------------------------------------------------------------------------
// ILP (Inductive Logic Programming) Python bindings
// ---------------------------------------------------------------------------

fn push_term_bytes(out: &mut Vec<u8>, term: &Term, typ: ScalarType) -> xlog_core::Result<()> {
    use xlog_core::XlogError;
    match (typ, term) {
        (ScalarType::U32, Term::Integer(v)) => {
            let v = u32::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("u32 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::U64, Term::Integer(v)) => {
            let v = u64::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("u64 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::I32, Term::Integer(v)) => {
            let v = i32::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("i32 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::I64, Term::Integer(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::F32, Term::Float(v)) => {
            out.extend_from_slice(&(*v as f32).to_le_bytes());
        }
        (ScalarType::F64, Term::Float(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::F32, Term::Integer(v)) => {
            out.extend_from_slice(&(*v as f32).to_le_bytes());
        }
        (ScalarType::F64, Term::Integer(v)) => {
            out.extend_from_slice(&(*v as f64).to_le_bytes());
        }
        (ScalarType::Bool, Term::Integer(v)) => {
            let b = match *v { 0 => 0u8, 1 => 1u8, other => {
                return Err(XlogError::Execution(format!("bool expects 0/1, got {}", other)));
            }};
            out.push(b);
        }
        (ScalarType::Bool, Term::Symbol(id)) => {
            let s = symbol::resolve(*id);
            if s == "true" || s == "false" {
                out.push(if s == "true" { 1u8 } else { 0u8 });
            } else {
                return Err(XlogError::Execution(format!("Expected boolean symbol, got '{}'", s)));
            }
        }
        (ScalarType::Symbol, Term::String(s)) => {
            out.extend_from_slice(&symbol::intern(s).to_le_bytes());
        }
        (ScalarType::Symbol, Term::Symbol(id)) => {
            out.extend_from_slice(&id.to_le_bytes());
        }
        (_, Term::Variable(v)) => {
            return Err(XlogError::Execution(format!("Fact cannot contain variable {}", v)));
        }
        (_, Term::Anonymous) => {
            return Err(XlogError::Execution("Fact cannot contain anonymous wildcard '_'".into()));
        }
        (_, Term::Aggregate(_)) => {
            return Err(XlogError::Execution("Fact cannot contain aggregate".into()));
        }
        (expected, got) => {
            return Err(XlogError::Execution(format!("Type mismatch: expected {:?}, got {:?}", expected, got)));
        }
    }
    Ok(())
}

/// Pack `i64` fact values into typed byte columns according to schema.
/// Returns one `Vec<u8>` per column with correctly-encoded LE bytes.
/// Rejects F32/F64 columns (not supported in batch APIs).
fn pack_i64_columns_typed(
    relation: &str,
    facts: &[Vec<i64>],
    schema: &Schema,
) -> PyResult<Vec<Vec<u8>>> {
    let arity = schema.arity();
    for (idx, fact) in facts.iter().enumerate() {
        if fact.len() != arity {
            return Err(PyValueError::new_err(format!(
                "Relation '{}': fact {} has {} values, expected {}",
                relation, idx, fact.len(), arity,
            )));
        }
    }

    let mut columns: Vec<Vec<u8>> = (0..arity)
        .map(|col_idx| {
            let elem_size = schema.column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            Vec::with_capacity(facts.len() * elem_size)
        })
        .collect();

    for fact in facts {
        for (col_idx, &val) in fact.iter().enumerate() {
            let col_type = schema.column_type(col_idx);
            let col = &mut columns[col_idx];
            match col_type {
                Some(ScalarType::U32) => {
                    let v = u32::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (U32): value {} out of range [0, {}]",
                        relation, col_idx, val, u32::MAX,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                Some(ScalarType::I32) => {
                    let v = i32::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (I32): value {} out of range [{}, {}]",
                        relation, col_idx, val, i32::MIN, i32::MAX,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                Some(ScalarType::U64) => {
                    let v = u64::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (U64): value {} is negative; U64 requires non-negative values",
                        relation, col_idx, val,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                Some(ScalarType::I64) => {
                    col.extend_from_slice(&val.to_le_bytes());
                }
                Some(ScalarType::Bool) => {
                    match val {
                        0 => col.push(0u8),
                        1 => col.push(1u8),
                        _ => return Err(PyValueError::new_err(format!(
                            "Relation '{}' column {} (Bool): value {} not in {{0, 1}}",
                            relation, col_idx, val,
                        ))),
                    }
                }
                Some(ScalarType::Symbol) => {
                    let v = u32::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (Symbol): value {} out of range [0, {}]",
                        relation, col_idx, val, u32::MAX,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                Some(ScalarType::F32) => {
                    return Err(PyValueError::new_err(format!(
                        "Relation '{}' column {} (F32): float columns not supported in batch APIs",
                        relation, col_idx,
                    )));
                }
                Some(ScalarType::F64) => {
                    return Err(PyValueError::new_err(format!(
                        "Relation '{}' column {} (F64): float columns not supported in batch APIs",
                        relation, col_idx,
                    )));
                }
                None => {
                    return Err(PyValueError::new_err(format!(
                        "Relation '{}' column {}: no type in schema",
                        relation, col_idx,
                    )));
                }
            }
        }
    }

    Ok(columns)
}

fn load_facts_into_store(
    ast: &AstProgram,
    provider: &CudaKernelProvider,
    executor: &mut Executor,
    schemas: &HashMap<String, Schema>,
) -> xlog_core::Result<()> {
    use xlog_core::XlogError;
    let mut rows_by_pred: HashMap<&str, Vec<&[Term]>> = HashMap::new();
    for fact in ast.facts() {
        rows_by_pred
            .entry(fact.head.predicate.as_str())
            .or_default()
            .push(&fact.head.terms);
    }

    for (pred, rows) in rows_by_pred {
        let schema = schemas.get(pred).ok_or_else(|| {
            XlogError::Execution(format!("Missing schema for fact predicate {}", pred))
        })?;

        if rows.iter().any(|r| r.len() != schema.arity()) {
            return Err(XlogError::Execution(format!(
                "Fact arity mismatch for {} (expected {})", pred, schema.arity()
            )));
        }

        let mut columns: Vec<Vec<u8>> = vec![Vec::new(); schema.arity()];
        for row in &rows {
            for (col_idx, term) in row.iter().enumerate() {
                let typ = schema.column_type(col_idx).ok_or_else(|| {
                    XlogError::Execution(format!("Missing type for col {}", col_idx))
                })?;
                push_term_bytes(&mut columns[col_idx], term, typ)?;
            }
        }

        let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
        let fact_buf = provider.create_buffer_from_slices(&slices, schema.clone())?;

        let existing = executor.store().get(pred).ok_or_else(|| {
            XlogError::Execution(format!("Missing base relation {} while loading facts", pred))
        })?;
        let merged = provider.union(existing, &fact_buf)?;
        executor.store_mut().put(pred, merged);
    }
    Ok(())
}

/// Extracted TensorMaskedJoin metadata from the execution plan.
struct TmjMeta {
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
    head_projection: Vec<usize>,
    schema_size: usize,
    head_rel_name: String,
}

fn walk_tmj(node: &RirNode, target_mask: Option<&str>) -> Option<TmjMeta> {
    match node {
        RirNode::TensorMaskedJoin {
            mask_name, left_keys, right_keys, head_projection, schema_size, head_rel_name, ..
        } => {
            if target_mask.is_none() || target_mask == Some(mask_name.as_str()) {
                Some(TmjMeta {
                    left_keys: left_keys.clone(),
                    right_keys: right_keys.clone(),
                    head_projection: head_projection.clone(),
                    schema_size: *schema_size,
                    head_rel_name: head_rel_name.clone(),
                })
            } else {
                None
            }
        }
        RirNode::Fixpoint { base, recursive, .. } => {
            walk_tmj(base, target_mask).or_else(|| walk_tmj(recursive, target_mask))
        }
        RirNode::Union { inputs } => {
            inputs.iter().find_map(|n| walk_tmj(n, target_mask))
        }
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::Distinct { input, .. }
        | RirNode::GroupBy { input, .. } => walk_tmj(input, target_mask),
        RirNode::Join { left, right, .. }
        | RirNode::Diff { left, right } => {
            walk_tmj(left, target_mask).or_else(|| walk_tmj(right, target_mask))
        }
        _ => None,
    }
}

fn extract_tmj_meta(plan: &ExecutionPlan) -> TmjMeta {
    extract_tmj_meta_for_mask(plan, None)
}

fn extract_tmj_meta_for_mask(plan: &ExecutionPlan, mask_name: Option<&str>) -> TmjMeta {
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            if let Some(meta) = walk_tmj(&rule.body, mask_name) {
                return meta;
            }
        }
    }
    TmjMeta { left_keys: vec![], right_keys: vec![], head_projection: vec![], schema_size: 0, head_rel_name: String::new() }
}

fn strip_learnable_declarations(source: &str) -> String {
    source.lines()
        .filter(|line| !line.trim_start().starts_with("learnable("))
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_learnable_declarations(source: &str) -> String {
    source.lines()
        .filter(|line| line.trim_start().starts_with("learnable("))
        .collect::<Vec<_>>()
        .join("\n")
}

#[pyclass]
pub struct IlpProgramFactory;

#[pymethods]
impl IlpProgramFactory {
    #[staticmethod]
    #[pyo3(signature = (source, device=0, memory_mb=512, max_active_rules=None))]
    pub fn compile(
        source: &str,
        device: usize,
        memory_mb: u64,
        max_active_rules: Option<usize>,
    ) -> PyResult<CompiledIlpProgram> {
        // Validate max_active_rules range
        if let Some(max) = max_active_rules {
            if !(16..=128).contains(&max) {
                return Err(PyValueError::new_err(format!(
                    "max_active_rules must be between 16 and 128, got {}", max
                )));
            }
        }

        let ast = xlog_logic::parse_program(source)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let base_source = strip_learnable_declarations(source);
        let learnable_source = extract_learnable_declarations(source);

        let mut compiler = xlog_logic::Compiler::new();
        if let Some(max) = max_active_rules {
            compiler.set_max_active_rules(max);
        }
        let plan = compiler.compile_program(&ast)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut rel_index: Vec<(RelId, String)> = compiler.rel_ids().iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect();
        rel_index.sort_by_key(|(id, _)| id.0);
        let schemas = compiler.schemas().clone();

        let config = GpuConfig {
            device_ordinal: device,
            memory_bytes: memory_mb * 1024 * 1024,
        };
        let provider = Arc::new(
            provider_from_config(config)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        );

        let mut executor = Executor::new(provider.clone());

        for (name, rel_id) in compiler.rel_ids() {
            executor.register_relation(*rel_id, name);
        }

        for (name, schema) in &schemas {
            let empty = provider.create_empty_buffer(schema.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            executor.store_mut().put(name, empty);
        }

        load_facts_into_store(&ast, &provider, &mut executor, &schemas)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        executor.execute_plan(&plan)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let tmj = extract_tmj_meta(&plan);

        let active_rules = max_active_rules.unwrap_or(32);

        Ok(CompiledIlpProgram {
            base_source, _learnable_source: learnable_source, ast, executor, provider,
            plan, rel_index, schemas,
            left_keys: tmj.left_keys, right_keys: tmj.right_keys,
            head_projection: tmj.head_projection,
            compiled_schema_size: tmj.schema_size,
            head_rel_name: tmj.head_rel_name,
            max_active_rules: active_rules,
            candidate_map: None,
            coo_chunk_budget: 16 * 1024 * 1024,
            strict_zero_dtoh: false,
        })
    }
}

#[pyclass]
pub struct CompiledIlpProgram {
    base_source: String,
    _learnable_source: String,
    ast: AstProgram,
    executor: Executor,
    provider: Arc<CudaKernelProvider>,
    plan: ExecutionPlan,
    rel_index: Vec<(RelId, String)>,
    schemas: HashMap<String, Schema>,
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
    head_projection: Vec<usize>,
    compiled_schema_size: usize,
    head_rel_name: String,
    max_active_rules: usize,
    candidate_map: Option<HashMap<(u32, u32, u32), u32>>,
    /// Maximum bytes for per-chunk temp allocations (masks, prefix sums,
    /// chunk-local COO scratch). The final merged COO buffer is exact-NNZ
    /// sized and may exceed this budget. Default: 16 MB.
    coo_chunk_budget: u64,
    /// When true, raise instead of falling back to chunked COO path.
    /// Use in zero-D2H benchmarks and CI gates. Default: false.
    strict_zero_dtoh: bool,
}

#[pymethods]
impl CompiledIlpProgram {
    /// Upload candidate (i,j,k) -> index mapping. Called once per attempt.
    pub fn set_candidate_map(&mut self, candidates: Vec<(u32, u32, u32)>) -> PyResult<()> {
        let mut map = HashMap::with_capacity(candidates.len());
        for (cidx, &(i, j, k)) in candidates.iter().enumerate() {
            map.insert((i, j, k), cidx as u32);
        }
        self.candidate_map = Some(map);
        Ok(())
    }

    /// Length of current candidate map (0 if not set).
    pub fn candidate_map_len(&self) -> usize {
        self.candidate_map.as_ref().map_or(0, |m| m.len())
    }

    /// Set the per-chunk temp allocation budget in bytes. The final merged
    /// COO buffer is exact-NNZ sized and may exceed this budget. Default: 16 MB.
    pub fn set_coo_chunk_budget(&mut self, bytes: u64) {
        self.coo_chunk_budget = bytes;
    }

    /// Deprecated: use `set_coo_chunk_budget`. Kept for one release cycle.
    #[allow(deprecated)]
    pub fn set_coo_memory_cap(&mut self, bytes: u64) {
        self.coo_chunk_budget = bytes;
    }

    /// Enable strict zero-D2H mode. When true, raises RuntimeError instead
    /// of falling back to the chunked COO path (which uses D2H transfers).
    /// Use for zero-D2H benchmarks and CI gates.
    pub fn set_strict_zero_dtoh(&mut self, strict: bool) {
        self.strict_zero_dtoh = strict;
    }

    /// GPU-resident ILP loss + gradient computation.
    ///
    /// Builds a sparse CSR structure from retained per-entry membership masks
    /// and launches forward (credit gather + NLL loss) and backward (gradient
    /// scatter) CUDA kernels.  Returns `(loss_dlpack, grad_dlpack)` where both
    /// are GPU-resident tensors exported via DLPack.
    ///
    /// # Arguments
    /// * `positives` — list of `(relation, [col_values])` positive examples
    /// * `negatives` — list of `(relation, [col_values])` negative examples
    /// * `cand_probs_obj` — DLPack/PyTorch tensor of candidate probabilities on GPU
    pub fn compute_ilp_loss_grad_gpu<'py>(
        &self,
        py: Python<'py>,
        positives: Vec<(String, Vec<i64>)>,
        negatives: Vec<(String, Vec<i64>)>,
        cand_probs_obj: &Bound<'py, PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        // ── Phase A: Validation + DLPack import ──
        let candidate_map = self.candidate_map.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "candidate_map not set — call set_candidate_map() before compute_ilp_loss_grad_gpu()",
            )
        })?;
        let num_cands = candidate_map.len() as u32;

        let managed = dlpack_from_py(cand_probs_obj)?;
        let cand_buf = self
            .provider
            .from_dlpack_tensors(vec![managed])
            .map_err(|e| PyRuntimeError::new_err(format!("DLPack import: {}", e)))?;

        // Determine dtype from imported buffer
        let cand_schema = cand_buf.schema().clone();
        let cand_dtype = cand_schema
            .column_type(0)
            .ok_or_else(|| PyRuntimeError::new_err("cand_probs has no column type"))?;
        let is_f64 = match cand_dtype {
            ScalarType::F32 => false,
            ScalarType::F64 => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "cand_probs must be F32 or F64, got {:?}",
                    other
                )));
            }
        };

        let cand_rows = read_device_row_count(&self.provider, &cand_buf)
            .map_err(|e| PyRuntimeError::new_err(format!("row count: {}", e)))?;
        if cand_rows != num_cands as usize {
            return Err(PyValueError::new_err(format!(
                "cand_probs length ({}) != candidate_map length ({})",
                cand_rows, num_cands
            )));
        }

        // ── Phase B: Build fact list, group by relation ──
        let num_pos = positives.len();
        let num_neg = negatives.len();
        let num_facts = (num_pos + num_neg) as u32;

        // Build all_facts: (relation, values, is_positive, global_fact_idx)
        struct FactInfo {
            relation: String,
            values: Vec<i64>,
            is_positive: bool,
        }
        let mut all_facts: Vec<FactInfo> = Vec::with_capacity(num_pos + num_neg);
        for (rel, vals) in positives {
            all_facts.push(FactInfo {
                relation: rel,
                values: vals,
                is_positive: true,
            });
        }
        for (rel, vals) in negatives {
            all_facts.push(FactInfo {
                relation: rel,
                values: vals,
                is_positive: false,
            });
        }

        // Handle empty facts edge case: return zero loss + zero grad
        if num_facts == 0 {
            return self.build_zero_loss_grad(py, num_cands, is_f64);
        }

        // Build is_positive array for upload
        let is_positive_host: Vec<u8> = all_facts.iter().map(|f| if f.is_positive { 1u8 } else { 0u8 }).collect();

        // Group facts by relation, preserving global fact_idx
        let mut groups: HashMap<String, Vec<(u32, Vec<i64>)>> = HashMap::new();
        for (global_idx, fact) in all_facts.iter().enumerate() {
            groups
                .entry(fact.relation.clone())
                .or_default()
                .push((global_idx as u32, fact.values.clone()));
        }

        // Get ILP tagged result
        let tagged = self
            .executor
            .ilp_last_result()
            .ok_or_else(|| PyRuntimeError::new_err("No ILP result — call evaluate() first"))?;

        // ── Phase C: Device-side COO build (zero D2H) ──
        //
        // Two-pass approach over (relation, candidate) tasks:
        //   Pass 1: compute GPU membership masks
        //   Pass 2: scatter COO entries at host-computed offsets via device kernel
        //
        // The key insight is that each task's num_query is known on the host
        // (it equals the number of facts in that relation group), so we can
        // compute COO write offsets entirely on the host without any D2H reads.
        // We over-allocate COO arrays at upper_bound (sum of all num_query)
        // and fill sentinel values for unused slots.

        struct CooTask {
            cidx: u32,
            num_query: u32,
            d_mask: xlog_cuda::memory::TrackedCudaSlice<u8>,
            fact_indices_idx: usize, // index into fact_indices_buffers
        }
        let mut tasks: Vec<CooTask> = Vec::new();
        let mut fact_indices_buffers: Vec<xlog_cuda::memory::TrackedCudaSlice<u32>> = Vec::new();

        for (relation, facts_with_idx) in &groups {
            let k_idx = self
                .rel_index
                .iter()
                .position(|(_, name)| name == relation)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("Relation '{}' not in ILP schema", relation))
                })? as u32;

            let relevant_entries: Vec<&xlog_runtime::ilp_registry::IlpTagEntry> = tagged
                .entries
                .iter()
                .filter(|e| e.k == k_idx && e.num_rows > 0 && e.buffer.is_some())
                .collect();

            if relevant_entries.is_empty() {
                continue;
            }

            let first_buf = relevant_entries[0].buffer.as_ref().unwrap();
            let arity = first_buf.arity();
            if arity == 0 {
                continue;
            }
            let schema = first_buf.schema().clone();

            let fact_values: Vec<Vec<i64>> =
                facts_with_idx.iter().map(|(_, v)| v.clone()).collect();
            let col_bytes = pack_i64_columns_typed(relation, &fact_values, &schema)?;
            let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
            let query_buf = self
                .provider
                .create_buffer_from_slices(&col_slices, schema)
                .map_err(|e| PyRuntimeError::new_err(format!("create_buffer: {}", e)))?;

            let keys: Vec<usize> = (0..arity).collect();
            let num_query = fact_values.len() as u32;

            // Upload global fact indices for this relation group (H2D, allowed).
            // Shared across all entries in this relation group via index.
            let global_indices: Vec<u32> =
                facts_with_idx.iter().map(|(idx, _)| *idx).collect();
            let mut d_fact_indices = self
                .provider
                .memory()
                .alloc::<u32>(num_query as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc fact_indices: {}", e)))?;
            self.provider
                .device()
                .inner()
                .htod_sync_copy_into(&global_indices, &mut d_fact_indices)
                .map_err(|e| PyRuntimeError::new_err(format!("htod fact_indices: {}", e)))?;
            let fi_idx = fact_indices_buffers.len();
            fact_indices_buffers.push(d_fact_indices);

            for entry in &relevant_entries {
                let cidx = match candidate_map.get(&(entry.i, entry.j, entry.k)) {
                    Some(c) => *c,
                    None => continue,
                };

                let entry_buf = entry.buffer.as_ref().unwrap();
                let d_mask = self
                    .provider
                    .membership_mask_device(&query_buf, entry_buf, &keys, &keys)
                    .map_err(|e| PyRuntimeError::new_err(format!("membership_mask: {}", e)))?;

                tasks.push(CooTask {
                    cidx,
                    num_query,
                    d_mask,
                    fact_indices_idx: fi_idx,
                });
            }
        }

        let num_tasks = tasks.len();
        let upper_bound: u32 = tasks.iter().map(|t| t.num_query).sum();

        if upper_bound == 0 || num_tasks == 0 {
            return self.build_loss_grad_empty_coo(
                py,
                &is_positive_host,
                num_facts,
                num_cands,
                is_f64,
            );
        }

        // Check if COO allocation would exceed memory cap.
        // Each COO entry uses 8 bytes (4 for fact_idx + 4 for cand_idx).
        let coo_bytes = (upper_bound as u64) * 8;
        let needs_chunking = coo_bytes > self.coo_chunk_budget;

        // Upload is_positive once (shared across all paths, H2D allowed)
        let mut d_is_positive = self
            .provider
            .memory()
            .alloc::<u8>(num_facts as usize)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc is_positive: {}", e)))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&is_positive_host, &mut d_is_positive)
            .map_err(|e| PyRuntimeError::new_err(format!("htod is_positive: {}", e)))?;

        let cand_col = cand_buf
            .column(0)
            .ok_or_else(|| PyRuntimeError::new_err("cand_probs has no column"))?;

        let eps_f32 = 1e-8f32;
        let eps_f64 = 1e-8f64;

        // Build COO arrays. If chunking is needed, we fill in chunks on device,
        // D2H each chunk, merge on host, then H2D the merged result.
        // This ensures a single CSR + forward/backward pass over the complete
        // COO, which is mathematically correct (NLL loss is nonlinear, so we
        // cannot sum per-chunk losses independently).
        let (mut d_coo_facts, mut d_coo_cands, actual_nnz) = if !needs_chunking {
            // ── Phase C (non-chunked): single-pass COO fill on device ──
            let mut offsets_host = vec![0u32; num_tasks];
            {
                let mut running = 0u32;
                for i in 0..num_tasks {
                    offsets_host[i] = running;
                    running += tasks[i].num_query;
                }
                debug_assert_eq!(running, upper_bound);
            }

            let mut d_offsets = self
                .provider
                .memory()
                .alloc::<u32>(num_tasks)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc d_offsets: {}", e)))?;
            self.provider
                .device()
                .inner()
                .htod_sync_copy_into(&offsets_host, &mut d_offsets)
                .map_err(|e| PyRuntimeError::new_err(format!("htod d_offsets: {}", e)))?;

            let sentinel_fact = num_facts;
            let sentinel_cand = num_cands;
            let mut d_coo_facts = self
                .provider
                .memory()
                .alloc::<u32>(upper_bound as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc coo_facts: {}", e)))?;
            let mut d_coo_cands = self
                .provider
                .memory()
                .alloc::<u32>(upper_bound as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc coo_cands: {}", e)))?;
            {
                let sentinel_facts_vec = vec![sentinel_fact; upper_bound as usize];
                let sentinel_cands_vec = vec![sentinel_cand; upper_bound as usize];
                self.provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&sentinel_facts_vec, &mut d_coo_facts)
                    .map_err(|e| PyRuntimeError::new_err(format!("sentinel facts: {}", e)))?;
                self.provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&sentinel_cands_vec, &mut d_coo_cands)
                    .map_err(|e| PyRuntimeError::new_err(format!("sentinel cands: {}", e)))?;
            }

            for (task_idx, task) in tasks.iter().enumerate() {
                let d_prefix = self
                    .provider
                    .scan_u8_mask_device(&task.d_mask, task.num_query)
                    .map_err(|e| PyRuntimeError::new_err(format!("scan mask: {}", e)))?;

                self.provider
                    .ilp_coo_fill_from_mask_launch(
                        &task.d_mask,
                        &d_prefix,
                        &fact_indices_buffers[task.fact_indices_idx],
                        task_idx as u32,
                        task.cidx,
                        task.num_query,
                        &d_offsets,
                        &mut d_coo_facts,
                        &mut d_coo_cands,
                    )
                    .map_err(|e| PyRuntimeError::new_err(format!("coo fill: {}", e)))?;
            }

            (d_coo_facts, d_coo_cands, upper_bound)
        } else {
            // ── Phase C (chunked): Two-pass GPU-only bounded-memory merge ──
            //
            // Pass 1: Count NNZ per task on device (bounded temp per chunk).
            // Pass 2: Fill COO at pre-computed offsets (bounded temp per chunk).
            // The final COO buffer is exact-NNZ sized and may exceed
            // coo_chunk_budget — this is safe because actual NNZ << upper_bound.

            let max_queries_per_chunk = (self.coo_chunk_budget / 8).max(1) as u32;

            // Allocate task_counts array (num_tasks + 1 for exclusive scan).
            let tc_len = num_tasks + 1;
            let mut d_task_counts = self
                .provider
                .memory()
                .alloc::<u32>(tc_len)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc task_counts: {}", e)))?;
            // Zero-init the entire array (counts default to 0, last slot = 0 for scan).
            {
                let zeros = vec![0u32; tc_len];
                self.provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&zeros, &mut d_task_counts)
                    .map_err(|e| PyRuntimeError::new_err(format!("zero task_counts: {}", e)))?;
            }

            // ── Pass 1: Count NNZ per task ──
            {
                let mut chunk_start = 0usize;
                while chunk_start < tasks.len() {
                    let mut chunk_end = chunk_start;
                    let mut chunk_sum = 0u32;
                    while chunk_end < tasks.len() {
                        let nq = tasks[chunk_end].num_query;
                        if chunk_sum + nq > max_queries_per_chunk && chunk_sum > 0 {
                            break;
                        }
                        chunk_sum += nq;
                        chunk_end += 1;
                    }
                    if chunk_end == chunk_start {
                        chunk_end = chunk_start + 1;
                    }

                    for task_idx in chunk_start..chunk_end {
                        let task = &tasks[task_idx];
                        self.provider
                            .count_mask_into_slot(
                                &task.d_mask,
                                task.num_query,
                                &mut d_task_counts,
                                task_idx,
                            )
                            .map_err(|e| PyRuntimeError::new_err(format!(
                                "count_mask_into_slot: {}", e
                            )))?;
                    }

                    chunk_start = chunk_end;
                }
            }

            // Synchronize to ensure all count kernels have completed.
            self.provider.device().synchronize()
                .map_err(|e| PyRuntimeError::new_err(format!("sync pass1: {}", e)))?;

            // Compute per-task write offsets via exclusive scan.
            // d_task_counts[0..num_tasks] has counts; after scan,
            // d_task_counts[i] = sum of counts[0..i] = write offset for task i.
            // d_task_counts[num_tasks] = total_nnz.
            self.provider
                .exclusive_scan_u32_inplace(&mut d_task_counts, tc_len as u32)
                .map_err(|e| PyRuntimeError::new_err(format!("scan task_counts: {}", e)))?;

            // Read total_nnz from the last element (metadata-only, untracked).
            let total_nnz: u32 = self.provider
                .dtoh_scalar_untracked(&d_task_counts, num_tasks)
                .map_err(|e| PyRuntimeError::new_err(format!("read total_nnz: {}", e)))?;

            if total_nnz == 0 {
                return self.build_loss_grad_empty_coo(
                    py,
                    &is_positive_host,
                    num_facts,
                    num_cands,
                    is_f64,
                );
            }

            // Allocate exact-NNZ global COO buffers (may exceed coo_chunk_budget).
            let mut d_coo_facts = self
                .provider
                .memory()
                .alloc::<u32>(total_nnz as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc global coo_facts: {}", e)))?;
            let mut d_coo_cands = self
                .provider
                .memory()
                .alloc::<u32>(total_nnz as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc global coo_cands: {}", e)))?;

            // ── Pass 2: Fill COO at pre-computed offsets ──
            // d_task_counts now contains offsets (after exclusive scan).
            // ilp_coo_fill_from_mask_launch reads d_offsets[offset_idx] for the
            // write base position. We pass d_task_counts as d_offsets with
            // offset_idx = task_idx.
            {
                let mut chunk_start = 0usize;
                while chunk_start < tasks.len() {
                    let mut chunk_end = chunk_start;
                    let mut chunk_sum = 0u32;
                    while chunk_end < tasks.len() {
                        let nq = tasks[chunk_end].num_query;
                        if chunk_sum + nq > max_queries_per_chunk && chunk_sum > 0 {
                            break;
                        }
                        chunk_sum += nq;
                        chunk_end += 1;
                    }
                    if chunk_end == chunk_start {
                        chunk_end = chunk_start + 1;
                    }

                    for task_idx in chunk_start..chunk_end {
                        let task = &tasks[task_idx];

                        let d_prefix = self
                            .provider
                            .scan_u8_mask_device(&task.d_mask, task.num_query)
                            .map_err(|e| PyRuntimeError::new_err(format!("scan mask: {}", e)))?;

                        self.provider
                            .ilp_coo_fill_from_mask_launch(
                                &task.d_mask,
                                &d_prefix,
                                &fact_indices_buffers[task.fact_indices_idx],
                                task_idx as u32,
                                task.cidx,
                                task.num_query,
                                &d_task_counts,  // offsets after scan
                                &mut d_coo_facts,
                                &mut d_coo_cands,
                            )
                            .map_err(|e| PyRuntimeError::new_err(format!("coo fill: {}", e)))?;
                        // d_prefix dropped here (bounded temp).
                    }

                    chunk_start = chunk_end;
                }
            }

            (d_coo_facts, d_coo_cands, total_nnz)
        };

        // ── Phase D: Sort COO + device-side CSR build ──
        //
        // Sort all entries by fact index. In the non-chunked path, sentinels
        // (fact = num_facts) sort to the end and are ignored by the histogram
        // kernel's f < num_facts guard. In the chunked path, the COO is
        // exact-NNZ (no sentinels) from the two-pass GPU-only merge.
        let mut scratch =
            xlog_cuda::provider::RadixSortScratch::new(&self.provider, actual_nnz)
                .map_err(|e| PyRuntimeError::new_err(format!("sort scratch: {}", e)))?;
        self.provider
            .radix_sort_u32_pairs(
                &mut d_coo_facts,
                &mut d_coo_cands,
                actual_nnz,
                &mut scratch,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("radix sort: {}", e)))?;

        let d_hist = self
            .provider
            .ilp_csr_histogram_launch(&d_coo_facts, actual_nnz, num_facts)
            .map_err(|e| PyRuntimeError::new_err(format!("csr histogram: {}", e)))?;

        let mut d_row_offsets = self
            .provider
            .memory()
            .alloc::<u32>((num_facts + 1) as usize)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc row_offsets: {}", e)))?;
        {
            let zeros = vec![0u32; (num_facts + 1) as usize];
            self.provider
                .device()
                .inner()
                .htod_sync_copy_into(&zeros, &mut d_row_offsets)
                .map_err(|e| PyRuntimeError::new_err(format!("zero row_offsets: {}", e)))?;
        }
        {
            let mut dst_view = d_row_offsets
                .try_slice_mut(0..num_facts as usize)
                .ok_or_else(|| {
                    PyRuntimeError::new_err("row_offsets slice failed")
                })?;
            self.provider
                .device()
                .inner()
                .dtod_copy(&d_hist, &mut dst_view)
                .map_err(|e| PyRuntimeError::new_err(format!("dtod hist->row_offsets: {}", e)))?;
        }
        self.provider
            .exclusive_scan_u32_inplace(&mut d_row_offsets, num_facts + 1)
            .map_err(|e| PyRuntimeError::new_err(format!("scan row_offsets: {}", e)))?;

        // ── Phase E: Forward + backward + device-side reduction ──
        if is_f64 {
            let (credit_out, loss_contrib) = self
                .provider
                .ilp_credit_forward_f64_launch(
                    &d_row_offsets,
                    &d_coo_cands,
                    cand_col,
                    &d_is_positive,
                    num_facts,
                    eps_f64,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("forward f64: {}", e)))?;

            let d_total_loss = self
                .provider
                .ilp_reduce_sum_f64_launch(&loss_contrib, num_facts)
                .map_err(|e| PyRuntimeError::new_err(format!("reduce f64: {}", e)))?;

            let d_grad = self
                .provider
                .ilp_credit_backward_f64_launch(
                    &d_row_offsets,
                    &d_coo_cands,
                    &credit_out,
                    &d_is_positive,
                    num_facts,
                    num_cands,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("backward f64: {}", e)))?;

            self.export_loss_grad_device_f64(py, d_total_loss, d_grad, num_cands)
        } else {
            let (credit_out, loss_contrib) = self
                .provider
                .ilp_credit_forward_f32_launch(
                    &d_row_offsets,
                    &d_coo_cands,
                    cand_col,
                    &d_is_positive,
                    num_facts,
                    eps_f32,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("forward f32: {}", e)))?;

            let d_total_loss = self
                .provider
                .ilp_reduce_sum_f32_launch(&loss_contrib, num_facts)
                .map_err(|e| PyRuntimeError::new_err(format!("reduce f32: {}", e)))?;

            let d_grad = self
                .provider
                .ilp_credit_backward_f32_launch(
                    &d_row_offsets,
                    &d_coo_cands,
                    &credit_out,
                    &d_is_positive,
                    num_facts,
                    num_cands,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("backward f32: {}", e)))?;

            self.export_loss_grad_device_f32(py, d_total_loss, d_grad, num_cands)
        }
    }

    pub fn set_rule_mask(
        &mut self,
        name: String,
        mask_hard_flat: &Bound<'_, PyAny>,
        mask_soft_flat: &Bound<'_, PyAny>,
        schema_size: usize,
    ) -> PyResult<()> {
        if self.compiled_schema_size > 0 && schema_size != self.compiled_schema_size {
            return Err(PyValueError::new_err(format!(
                "schema_size mismatch: mask has N={} but compiled program expects N={}",
                schema_size, self.compiled_schema_size,
            )));
        }

        let hard_dmt = dlpack_from_py(mask_hard_flat)?;
        let soft_dmt = dlpack_from_py(mask_soft_flat)?;

        let hard_buf = self.provider.from_dlpack_tensors(vec![hard_dmt])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let soft_buf = self.provider.from_dlpack_tensors(vec![soft_dmt])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.executor.ilp_registry_mut().insert_mask(
            name, hard_buf, soft_buf, schema_size,
        );
        Ok(())
    }

    /// Sparse mask API: candidate IDs + DLPack soft probabilities (GPU tensor).
    ///
    /// `candidate_ids` must be exactly `[0..C)` where C is the candidate count
    /// for this mask under the provided recursion policy.
    ///
    /// `soft_probs_dlpack` is a DLPack capsule (CUDA f64 tensor) passed from PyTorch.
    /// Rust imports it zero-copy, downloads values (not counted by the D2H counter),
    /// performs deterministic top-k (desc soft value, then lower id), and
    /// stores a sparse IlpMask (no dense N^3 materialization).
    #[pyo3(signature = (name, candidate_ids, soft_probs_dlpack, budget, allow_recursive=false))]
    pub fn set_rule_mask_sparse(
        &mut self,
        name: String,
        candidate_ids: Vec<u32>,
        soft_probs_dlpack: &Bound<'_, PyAny>,
        budget: usize,
        allow_recursive: bool,
    ) -> PyResult<()> {
        let tmj = extract_tmj_meta_for_mask(&self.plan, Some(&name));
        let n = tmj.schema_size;
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "no learnable mask '{}' found", name
            )));
        }
        if self.compiled_schema_size > 0 && n != self.compiled_schema_size {
            return Err(PyValueError::new_err(format!(
                "schema_size mismatch for '{}': plan N={} compiled N={}",
                name, n, self.compiled_schema_size
            )));
        }

        let expected_c = self.expected_candidate_count(&name, allow_recursive)?;
        if candidate_ids.len() != expected_c {
            return Err(PyValueError::new_err(format!(
                "candidate_ids length {} != expected candidate count {}",
                candidate_ids.len(), expected_c
            )));
        }
        for (idx, &cid) in candidate_ids.iter().enumerate() {
            if cid != idx as u32 {
                return Err(PyValueError::new_err(format!(
                    "candidate_ids must be [0..{}), got id {} at position {}",
                    expected_c, cid, idx
                )));
            }
        }

        // Import DLPack tensor (zero-copy, stays on GPU)
        let soft_dmt = dlpack_from_py(soft_probs_dlpack)?;
        let soft_buf = self.provider.from_dlpack_tensors(vec![soft_dmt])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Download f64 values (control-plane, NOT tracked by D2H counter)
        let soft_probs = self.provider.download_f64_untracked(&soft_buf, 0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if soft_probs.len() != candidate_ids.len() {
            return Err(PyValueError::new_err(format!(
                "soft_probs tensor length {} != candidate_ids length {}",
                soft_probs.len(), candidate_ids.len()
            )));
        }

        let candidate_triples = self.candidate_triples_for_mask(&name, allow_recursive)?;
        if candidate_triples.len() != expected_c {
            return Err(PyRuntimeError::new_err(format!(
                "internal candidate count mismatch: triples={} expected={}",
                candidate_triples.len(), expected_c
            )));
        }

        // Convert to f32 for top-k ranking in insert_mask_from_sparse
        let active_soft: Vec<f32> = soft_probs.iter().map(|&v| v as f32).collect();

        self.executor
            .ilp_registry_mut()
            .insert_mask_from_sparse(name, n, &candidate_triples, &active_soft, budget)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn evaluate(&mut self, py: Python<'_>) -> PyResult<()> {
        let _result: xlog_core::Result<xlog_cuda::CudaBuffer> = py.allow_threads(|| {
            self.executor.reset_for_mc();
            for (name, schema) in &self.schemas {
                let empty = self.provider.create_empty_buffer(schema.clone())?;
                self.executor.store_mut().put(name, empty);
            }
            load_facts_into_store(
                &self.ast, &self.provider, &mut self.executor, &self.schemas,
            )?;
            self.executor.execute_plan(&self.plan)
        });
        _result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Reset mutable runtime state for ILP attempt reuse.
    ///
    /// Clears ILP registry (masks/tagged results), executor store,
    /// join index cache, stats, and profiler. Then re-registers schemas
    /// with empty buffers, reloads base facts from AST, and re-executes
    /// the plan. Preserves all immutable compile artifacts (AST, plan,
    /// schemas, rel_index, provider, TMJ metadata, max_active_rules).
    ///
    /// After reset, the program is in the same state as a fresh compile()
    /// with the same source — ready for set_rule_mask / evaluate cycles.
    pub fn reset_runtime(&mut self, py: Python<'_>) -> PyResult<()> {
        let _result: xlog_core::Result<()> = py.allow_threads(|| {
            // 1. Clear all mutable state (ILP registry, store, caches, stats)
            self.executor.reset_for_ilp();

            // 2. Re-register schemas with empty buffers
            for (name, schema) in &self.schemas {
                let empty = self.provider.create_empty_buffer(schema.clone())?;
                self.executor.store_mut().put(name, empty);
            }

            // 3. Reload base facts from preserved AST
            load_facts_into_store(
                &self.ast, &self.provider, &mut self.executor, &self.schemas,
            )?;

            // 4. Re-execute plan (populates derived relations)
            self.executor.execute_plan(&self.plan)?;

            Ok(())
        });
        _result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // 5. Reset D2H transfer counter
        self.provider.reset_d2h_transfer_count();

        Ok(())
    }

    pub fn get_tagged_results(&self) -> PyResult<Vec<(u32, u32, u32, u32)>> {
        match self.executor.ilp_last_result() {
            Some(result) => Ok(result.entries.iter()
                .map(|e| (e.i, e.j, e.k, e.num_rows))
                .collect()),
            None => Ok(Vec::new()),
        }
    }

    pub fn fact_exists(
        &self,
        relation: &str,
        values: Vec<i64>,
    ) -> PyResult<bool> {
        let buf = self.executor.store().get(relation)
            .ok_or_else(|| PyValueError::new_err(
                format!("Relation '{}' not found", relation)
            ))?;

        Self::fact_exists_in_buffer(&self.provider, buf, &values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return all facts in the named relation as a list of int lists.
    #[pyo3(signature = (rel_name))]
    pub fn relation_facts(&self, rel_name: String) -> PyResult<Vec<Vec<i64>>> {
        let buf = self.executor.store().get(&rel_name)
            .ok_or_else(|| PyValueError::new_err(
                format!("Relation '{}' not found", rel_name)
            ))?;

        let num_rows = read_device_row_count(&self.provider, buf)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))? as usize;
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Download all columns (reuse fact_exists_in_buffer pattern)
        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for col_idx in 0..buf.arity() {
            let col_type = schema.column_type(col_idx)
                .ok_or_else(|| PyRuntimeError::new_err(
                    format!("Column {} type not found in schema", col_idx)
                ))?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => self.provider.download_column_i64(buf, col_idx)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                ScalarType::I32 => self.provider.download_column_i32(buf, col_idx)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into_iter().map(|v| v as i64).collect(),
                ScalarType::U32 | ScalarType::Symbol => {
                    self.provider.download_column_u32(buf, col_idx)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .into_iter().map(|v| v as i64).collect()
                }
                ScalarType::U64 => {
                    self.provider.download_column_u64(buf, col_idx)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .into_iter().map(|v| v as i64).collect()
                }
                ScalarType::Bool => {
                    self.provider.download_column_bool(buf, col_idx)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .into_iter().map(|v| if v { 1i64 } else { 0i64 }).collect()
                }
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(PyRuntimeError::new_err(
                        format!("relation_facts does not support float column type {:?}", col_type)
                    ));
                }
            };
            columns.push(col_i64);
        }

        let mut result = Vec::with_capacity(num_rows);
        for r in 0..num_rows {
            let mut row = Vec::with_capacity(buf.arity());
            for c in 0..buf.arity() {
                row.push(columns[c][r]);
            }
            result.push(row);
        }
        Ok(result)
    }

    /// Sample up to `max_n` derived facts for `head_rel` that are NOT in `exclude`.
    ///
    /// Returns `list[list[int]]` — each inner list is a tuple of column values.
    /// Uses the same column-download pattern as `relation_facts`.
    #[pyo3(signature = (head_rel, exclude, max_n))]
    pub fn sample_false_positives(
        &self,
        head_rel: String,
        exclude: Vec<(String, Vec<i64>)>,
        max_n: usize,
    ) -> PyResult<Vec<Vec<i64>>> {
        // Build exclude set: only consider tuples for the requested relation
        let exclude_set: HashSet<Vec<i64>> = exclude
            .into_iter()
            .filter(|(rel, _)| rel == &head_rel)
            .map(|(_, vals)| vals)
            .collect();

        // Download all facts using the same pattern as relation_facts
        let buf = self.executor.store().get(&head_rel)
            .ok_or_else(|| PyValueError::new_err(
                format!("Relation '{}' not found", head_rel)
            ))?;

        let num_rows = read_device_row_count(&self.provider, buf)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))? as usize;
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for col_idx in 0..buf.arity() {
            let col_type = schema.column_type(col_idx)
                .ok_or_else(|| PyRuntimeError::new_err(
                    format!("Column {} type not found in schema", col_idx)
                ))?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => self.provider.download_column_i64(buf, col_idx)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                ScalarType::I32 => self.provider.download_column_i32(buf, col_idx)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .into_iter().map(|v| v as i64).collect(),
                ScalarType::U32 | ScalarType::Symbol => {
                    self.provider.download_column_u32(buf, col_idx)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .into_iter().map(|v| v as i64).collect()
                }
                ScalarType::U64 => {
                    self.provider.download_column_u64(buf, col_idx)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .into_iter().map(|v| v as i64).collect()
                }
                ScalarType::Bool => {
                    self.provider.download_column_bool(buf, col_idx)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .into_iter().map(|v| if v { 1i64 } else { 0i64 }).collect()
                }
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(PyRuntimeError::new_err(
                        format!("sample_false_positives does not support float column type {:?}", col_type)
                    ));
                }
            };
            columns.push(col_i64);
        }

        // Filter out excluded tuples and cap at max_n
        let mut result = Vec::with_capacity(max_n.min(num_rows));
        for r in 0..num_rows {
            if result.len() >= max_n {
                break;
            }
            let mut row = Vec::with_capacity(buf.arity());
            for c in 0..buf.arity() {
                row.push(columns[c][r]);
            }
            if !exclude_set.contains(&row) {
                result.push(row);
            }
        }
        Ok(result)
    }

    pub fn tagged_entries_containing_fact(
        &self,
        relation: &str,
        values: Vec<i64>,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        let k_idx = self.rel_index.iter()
            .position(|(_, name)| name == relation)
            .ok_or_else(|| PyValueError::new_err(
                format!("Relation '{}' not in ILP schema", relation)
            ))? as u32;

        let tagged = match self.executor.ilp_last_result() {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        let mut result = Vec::new();
        for entry in &tagged.entries {
            if entry.k != k_idx || entry.num_rows == 0 {
                continue;
            }

            let (_, left_name) = &self.rel_index[entry.i as usize];
            let (_, right_name) = &self.rel_index[entry.j as usize];

            let left_buf = match self.executor.store().get(left_name) {
                Some(buf) if buf.arity() > 0 => buf,
                _ => continue,
            };
            let right_buf = match self.executor.store().get(right_name) {
                Some(buf) if buf.arity() > 0 => buf,
                _ => continue,
            };

            // Arity guard: same as executor (skip if join keys exceed columns)
            let left_max = self.left_keys.iter().copied().max().unwrap_or(0);
            let right_max = self.right_keys.iter().copied().max().unwrap_or(0);
            if left_buf.arity() <= left_max || right_buf.arity() <= right_max {
                continue;
            }

            let joined = self.provider.hash_join_v2(
                left_buf, right_buf,
                &self.left_keys, &self.right_keys,
                JoinType::Inner,
            ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Apply head_projection: check projected columns against values,
            // not the raw join output (which has more columns than the head).
            let found = if !self.head_projection.is_empty()
                && self.head_projection.len() == values.len()
            {
                Self::fact_exists_projected(
                    &self.provider, &joined, &values, &self.head_projection,
                ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            } else {
                Self::fact_exists_in_buffer(
                    &self.provider, &joined, &values,
                ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            };

            if found {
                result.push((entry.i, entry.j, entry.k));
            }
        }
        Ok(result)
    }

    pub fn ilp_schema_size(&self) -> usize { self.rel_index.len() }

    pub fn ilp_relation_names(&self) -> Vec<String> {
        self.rel_index.iter().map(|(_, name)| name.clone()).collect()
    }

    /// Return declared predicate types from source `pred` declarations.
    ///
    /// Output is a list of `(name, types)` tuples so callers can
    /// deterministically inspect whether metadata is available for
    /// relations used during promotion.
    pub fn relation_type_annotations(&self) -> Vec<(String, Vec<String>)> {
        self.ast
            .predicates
            .iter()
            .map(|pred| {
                let types = pred.types.iter().map(scalar_type_name).collect();
                (pred.name.clone(), types)
            })
            .collect()
    }

    /// Return the set of valid (i,j,k) candidates for the given learnable mask.
    ///
    /// Pruning rules:
    /// - k must be the head relation for this mask
    /// - At least one of (i,j) must have nonzero tuples in the store
    /// - Template+template body pairs (both have zero tuples) are pruned
    /// - If allow_recursive is false: i==k_head or j==k_head are pruned
    ///   (unless head already has base facts)
    ///
    /// Returns list of dicts: [{id, i, j, k, left_name, right_name, head_name}]
    /// IDs assigned 0..C-1 after sorting by (k, i, j) ascending.
    #[pyo3(signature = (mask_name, allow_recursive=false))]
    fn valid_candidates(
        &self,
        py: Python<'_>,
        mask_name: String,
        allow_recursive: bool,
    ) -> PyResult<Vec<StdHashMap<String, PyObject>>> {
        let candidates = self.candidate_triples_for_mask(&mask_name, allow_recursive)?;

        let result: Vec<StdHashMap<String, PyObject>> = candidates.iter()
            .enumerate()
            .map(|(id, &(i, j, k))| {
                let mut d = StdHashMap::new();
                d.insert("id".into(), id.into_py(py));
                d.insert("i".into(), i.into_py(py));
                d.insert("j".into(), j.into_py(py));
                d.insert("k".into(), k.into_py(py));
                d.insert("left_name".into(),
                         self.rel_index[i as usize].1.clone().into_py(py));
                d.insert("right_name".into(),
                         self.rel_index[j as usize].1.clone().into_py(py));
                d.insert("head_name".into(),
                         self.rel_index[k as usize].1.clone().into_py(py));
                d
            })
            .collect();

        Ok(result)
    }

    pub fn commit_induced_rule(&mut self, rule_source: &str) -> PyResult<()> {
        let new_base = format!("{}\n{}", self.base_source, rule_source);

        let ast = xlog_logic::parse_program(&new_base)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut compiler = xlog_logic::Compiler::new();
        compiler.set_max_active_rules(self.max_active_rules);
        let plan = compiler.compile_program(&ast)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let schemas = compiler.schemas().clone();

        self.executor.reset_for_mc();
        for (name, rel_id) in compiler.rel_ids() {
            self.executor.register_relation(*rel_id, name);
        }
        for (name, schema) in &schemas {
            let empty = self.provider.create_empty_buffer(schema.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.executor.store_mut().put(name, empty);
        }
        load_facts_into_store(&ast, &self.provider, &mut self.executor, &schemas)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.executor.execute_plan(&plan)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.base_source = new_base;
        self.ast = ast;
        let tmj = extract_tmj_meta(&plan);
        self.left_keys = tmj.left_keys;
        self.right_keys = tmj.right_keys;
        self.head_projection = tmj.head_projection;
        self.compiled_schema_size = tmj.schema_size;
        self.head_rel_name = tmj.head_rel_name;
        self.plan = plan;
        self.schemas = schemas;
        Ok(())
    }

    /// GPU-side batch fact membership check.
    /// Uploads `facts` (list of value-lists) to a temporary CudaBuffer,
    /// semi-joins against the named relation, returns per-fact boolean mask.
    /// Zero download_column_* calls — only downloads the u8 mask.
    pub fn batch_fact_membership(
        &self,
        relation: &str,
        facts: Vec<Vec<i64>>,
    ) -> PyResult<Vec<bool>> {
        if facts.is_empty() {
            return Ok(Vec::new());
        }

        let buf = self.executor.store().get(relation)
            .ok_or_else(|| PyValueError::new_err(
                format!("Relation '{}' not found", relation)
            ))?;

        let arity = buf.arity();
        if arity == 0 {
            return Ok(vec![false; facts.len()]);
        }

        // Schema-aware typed upload
        let col_bytes = pack_i64_columns_typed(relation, &facts, buf.schema())?;
        let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
        let query_buf = self.provider
            .create_buffer_from_slices(
                &col_slices,
                buf.schema().clone(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // All columns are keys (full-tuple match)
        let keys: Vec<usize> = (0..arity).collect();

        self.provider
            .membership_mask(&query_buf, buf, &keys, &keys)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// GPU-side batch credit assignment.
    ///
    /// For each fact in `facts`, returns the list of (i,j,k) entries whose
    /// join result contains that fact. Uses membership_mask against retained
    /// per-entry buffers — zero download_column_* calls.
    pub fn batch_tagged_credit(
        &self,
        relation: &str,
        facts: Vec<Vec<i64>>,
    ) -> PyResult<Vec<Vec<(u32, u32, u32)>>> {
        if facts.is_empty() {
            return Ok(Vec::new());
        }

        // Find k index for this relation
        let k_idx = self.rel_index.iter()
            .position(|(_, name)| name == relation)
            .ok_or_else(|| PyValueError::new_err(
                format!("Relation '{}' not in ILP schema", relation)
            ))? as u32;

        let tagged = match self.executor.ilp_last_result() {
            Some(t) => t,
            None => return Ok(vec![Vec::new(); facts.len()]),
        };

        // Filter entries to those matching target relation k, with retained buffers
        let relevant_entries: Vec<&xlog_runtime::ilp_registry::IlpTagEntry> = tagged.entries.iter()
            .filter(|e| e.k == k_idx && e.num_rows > 0 && e.buffer.is_some())
            .collect();

        if relevant_entries.is_empty() {
            return Ok(vec![Vec::new(); facts.len()]);
        }

        // Determine arity from the first entry's buffer
        let first_buf = relevant_entries[0].buffer.as_ref().unwrap();
        let arity = first_buf.arity();
        if arity == 0 {
            return Ok(vec![Vec::new(); facts.len()]);
        }

        // Schema-aware typed upload
        let schema = first_buf.schema().clone();
        let col_bytes = pack_i64_columns_typed(relation, &facts, &schema)?;
        let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
        let query_buf = self.provider
            .create_buffer_from_slices(&col_slices, schema)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let keys: Vec<usize> = (0..arity).collect();

        // For each relevant entry, compute membership mask against query facts
        let mut per_fact_credits: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); facts.len()];

        for entry in &relevant_entries {
            let entry_buf = entry.buffer.as_ref().unwrap();
            let mask = self.provider
                .membership_mask(&query_buf, entry_buf, &keys, &keys)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            for (fact_idx, &found) in mask.iter().enumerate() {
                if found {
                    per_fact_credits[fact_idx].push((entry.i, entry.j, entry.k));
                }
            }
        }

        Ok(per_fact_credits)
    }

    pub fn d2h_transfer_count(&self) -> u64 {
        self.provider.d2h_transfer_count()
    }

    pub fn reset_d2h_transfer_count(&self) {
        self.provider.reset_d2h_transfer_count()
    }

    pub fn host_transfer_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.provider.host_transfer_stats();
        let dict = PyDict::new_bound(py);
        dict.set_item("dtoh_bytes", stats.dtoh_bytes)?;
        dict.set_item("htod_bytes", stats.htod_bytes)?;
        dict.set_item("dtoh_calls", stats.dtoh_calls)?;
        dict.set_item("htod_calls", stats.htod_calls)?;
        Ok(dict.into())
    }

    pub fn reset_host_transfer_stats(&self) {
        self.provider.reset_host_transfer_stats()
    }
}

impl CompiledIlpProgram {
    // ─── GPU loss/grad export helpers ──────────────────────────────────

    /// Build zero loss (scalar) + zero grad (num_cands) on GPU, export via DLPack.
    fn build_zero_loss_grad(
        &self,
        py: Python<'_>,
        num_cands: u32,
        is_f64: bool,
    ) -> PyResult<(PyObject, PyObject)> {
        if is_f64 {
            let mut d_grad = self
                .provider
                .memory()
                .alloc::<f64>(num_cands as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc grad: {}", e)))?;
            if num_cands > 0 {
                self.provider
                    .device()
                    .inner()
                    .memset_zeros(&mut d_grad)
                    .map_err(|e| PyRuntimeError::new_err(format!("zero grad: {}", e)))?;
            }
            let mut d_loss = self
                .provider
                .memory()
                .alloc::<f64>(1)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc loss: {}", e)))?;
            self.provider
                .device()
                .inner()
                .memset_zeros(&mut d_loss)
                .map_err(|e| PyRuntimeError::new_err(format!("zero loss: {}", e)))?;
            self.export_loss_grad_device_f64(py, d_loss, d_grad, num_cands)
        } else {
            let mut d_grad = self
                .provider
                .memory()
                .alloc::<f32>(num_cands as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc grad: {}", e)))?;
            if num_cands > 0 {
                self.provider
                    .device()
                    .inner()
                    .memset_zeros(&mut d_grad)
                    .map_err(|e| PyRuntimeError::new_err(format!("zero grad: {}", e)))?;
            }
            let mut d_loss = self
                .provider
                .memory()
                .alloc::<f32>(1)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc loss: {}", e)))?;
            self.provider
                .device()
                .inner()
                .memset_zeros(&mut d_loss)
                .map_err(|e| PyRuntimeError::new_err(format!("zero loss: {}", e)))?;
            self.export_loss_grad_device_f32(py, d_loss, d_grad, num_cands)
        }
    }

    /// Handle the case where COO is empty but we have facts.
    /// Facts with no covering entries get: positive → -log(eps), negative → -log(1.0) = 0.
    /// We still run the kernels with an empty CSR for correctness.
    fn build_loss_grad_empty_coo(
        &self,
        py: Python<'_>,
        is_positive_host: &[u8],
        num_facts: u32,
        num_cands: u32,
        is_f64: bool,
    ) -> PyResult<(PyObject, PyObject)> {
        // Build CSR with all-zero row_offsets (every row has 0 non-zeros)
        let row_offsets = vec![0u32; (num_facts + 1) as usize];
        let mut d_row_offsets = self
            .provider
            .memory()
            .alloc::<u32>((num_facts + 1) as usize)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc: {}", e)))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&row_offsets, &mut d_row_offsets)
            .map_err(|e| PyRuntimeError::new_err(format!("htod: {}", e)))?;

        // Empty col_indices
        let d_col_indices = self
            .provider
            .memory()
            .alloc::<u32>(0)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc: {}", e)))?;

        let mut d_is_positive = self
            .provider
            .memory()
            .alloc::<u8>(num_facts as usize)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc: {}", e)))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(is_positive_host, &mut d_is_positive)
            .map_err(|e| PyRuntimeError::new_err(format!("htod: {}", e)))?;

        // Build a single-column CudaBuffer with 0 elements to represent an empty cand_probs
        // for the kernel launch (won't be read since row ranges are all empty).
        // We need a dummy CudaColumn. Use the actual cand count = num_cands.
        // Since COO is empty the kernel won't access cand_probs, but we need to pass something.
        let dummy_col: xlog_cuda::CudaColumn = if is_f64 {
            self.provider
                .memory()
                .alloc::<f64>(num_cands.max(1) as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc dummy: {}", e)))?
                .into_bytes()
                .into()
        } else {
            self.provider
                .memory()
                .alloc::<f32>(num_cands.max(1) as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("alloc dummy: {}", e)))?
                .into_bytes()
                .into()
        };

        if is_f64 {
            let (credit_out, loss_contrib) = self
                .provider
                .ilp_credit_forward_f64_launch(
                    &d_row_offsets,
                    &d_col_indices,
                    &dummy_col,
                    &d_is_positive,
                    num_facts,
                    1e-8f64,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("forward: {}", e)))?;

            let d_total_loss = self
                .provider
                .ilp_reduce_sum_f64_launch(&loss_contrib, num_facts)
                .map_err(|e| PyRuntimeError::new_err(format!("reduce: {}", e)))?;

            let d_grad = self
                .provider
                .ilp_credit_backward_f64_launch(
                    &d_row_offsets,
                    &d_col_indices,
                    &credit_out,
                    &d_is_positive,
                    num_facts,
                    num_cands,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("backward: {}", e)))?;

            self.export_loss_grad_device_f64(py, d_total_loss, d_grad, num_cands)
        } else {
            let (credit_out, loss_contrib) = self
                .provider
                .ilp_credit_forward_f32_launch(
                    &d_row_offsets,
                    &d_col_indices,
                    &dummy_col,
                    &d_is_positive,
                    num_facts,
                    1e-8f32,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("forward: {}", e)))?;

            let d_total_loss = self
                .provider
                .ilp_reduce_sum_f32_launch(&loss_contrib, num_facts)
                .map_err(|e| PyRuntimeError::new_err(format!("reduce: {}", e)))?;

            let d_grad = self
                .provider
                .ilp_credit_backward_f32_launch(
                    &d_row_offsets,
                    &d_col_indices,
                    &credit_out,
                    &d_is_positive,
                    num_facts,
                    num_cands,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("backward: {}", e)))?;

            self.export_loss_grad_device_f32(py, d_total_loss, d_grad, num_cands)
        }
    }

    /// Export a scalar f32 loss and f32 grad vector as DLPack capsules.
    /// Export a device-resident f32 loss scalar and f32 grad vector as DLPack capsules.
    fn export_loss_grad_device_f32(
        &self,
        py: Python<'_>,
        d_loss_val: xlog_cuda::memory::TrackedCudaSlice<f32>,
        d_grad: xlog_cuda::memory::TrackedCudaSlice<f32>,
        num_cands: u32,
    ) -> PyResult<(PyObject, PyObject)> {
        let schema_f32 = Schema::new(vec![("col_0".to_string(), ScalarType::F32)]);

        let mut d_loss_nrows = self
            .provider
            .memory()
            .alloc::<u32>(1)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc: {}", e)))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&[1u32], &mut d_loss_nrows)
            .map_err(|e| PyRuntimeError::new_err(format!("htod: {}", e)))?;

        let loss_buf = xlog_cuda::CudaBuffer::from_columns(
            vec![d_loss_val.into_bytes().into()],
            1,
            d_loss_nrows,
            schema_f32.clone(),
        );
        let loss_dl = self
            .provider
            .to_dlpack_table(loss_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(format!("DLPack loss: {}", e)))?;
        let loss_capsule = dlpack_capsule_from_tensor(py, loss_dl)?;

        let mut d_grad_nrows = self
            .provider
            .memory()
            .alloc::<u32>(1)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc: {}", e)))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&[num_cands], &mut d_grad_nrows)
            .map_err(|e| PyRuntimeError::new_err(format!("htod: {}", e)))?;

        let grad_buf = xlog_cuda::CudaBuffer::from_columns(
            vec![d_grad.into_bytes().into()],
            num_cands as u64,
            d_grad_nrows,
            schema_f32,
        );
        let grad_dl = self
            .provider
            .to_dlpack_table(grad_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(format!("DLPack grad: {}", e)))?;
        let grad_capsule = dlpack_capsule_from_tensor(py, grad_dl)?;

        Ok((loss_capsule, grad_capsule))
    }

    /// Export a device-resident f64 loss scalar and f64 grad vector as DLPack capsules.
    fn export_loss_grad_device_f64(
        &self,
        py: Python<'_>,
        d_loss_val: xlog_cuda::memory::TrackedCudaSlice<f64>,
        d_grad: xlog_cuda::memory::TrackedCudaSlice<f64>,
        num_cands: u32,
    ) -> PyResult<(PyObject, PyObject)> {
        let schema_f64 = Schema::new(vec![("col_0".to_string(), ScalarType::F64)]);

        let mut d_loss_nrows = self
            .provider
            .memory()
            .alloc::<u32>(1)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc: {}", e)))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&[1u32], &mut d_loss_nrows)
            .map_err(|e| PyRuntimeError::new_err(format!("htod: {}", e)))?;

        let loss_buf = xlog_cuda::CudaBuffer::from_columns(
            vec![d_loss_val.into_bytes().into()],
            1,
            d_loss_nrows,
            schema_f64.clone(),
        );
        let loss_dl = self
            .provider
            .to_dlpack_table(loss_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(format!("DLPack loss: {}", e)))?;
        let loss_capsule = dlpack_capsule_from_tensor(py, loss_dl)?;

        let mut d_grad_nrows = self
            .provider
            .memory()
            .alloc::<u32>(1)
            .map_err(|e| PyRuntimeError::new_err(format!("alloc: {}", e)))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&[num_cands], &mut d_grad_nrows)
            .map_err(|e| PyRuntimeError::new_err(format!("htod: {}", e)))?;

        let grad_buf = xlog_cuda::CudaBuffer::from_columns(
            vec![d_grad.into_bytes().into()],
            num_cands as u64,
            d_grad_nrows,
            schema_f64,
        );
        let grad_dl = self
            .provider
            .to_dlpack_table(grad_buf)
            .column(0)
            .map_err(|e| PyRuntimeError::new_err(format!("DLPack grad: {}", e)))?;
        let grad_capsule = dlpack_capsule_from_tensor(py, grad_dl)?;

        Ok((loss_capsule, grad_capsule))
    }

    /// Returns sorted (i,j,k) candidate triples for the given learnable mask.
    /// Pruning logic must stay aligned with `valid_candidates`.
    fn candidate_triples_for_mask(
        &self,
        mask_name: &str,
        allow_recursive: bool,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        let tmj = extract_tmj_meta_for_mask(&self.plan, Some(mask_name));
        let n = tmj.schema_size;
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "no learnable mask '{}' found in compiled program",
                mask_name
            )));
        }
        let head_name = &tmj.head_rel_name;
        let k_head = self
            .rel_index
            .iter()
            .position(|(_, name)| name == head_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "head relation '{}' not in rel_index for mask '{}'",
                    head_name, mask_name
                ))
            })? as u32;

        // Identify which relations currently have nonzero tuples in store.
        let has_tuples: Vec<bool> = self
            .rel_index
            .iter()
            .map(|(_, name)| {
                self.executor
                    .store()
                    .get(name)
                    .map(|buf| buf.num_rows() > 0)
                    .unwrap_or(false)
            })
            .collect();

        let mut triples: Vec<(u32, u32, u32)> = Vec::new();
        for i in 0..n as u32 {
            for j in 0..n as u32 {
                let k = k_head;

                // Prune template+template (both no tuples).
                if !has_tuples[i as usize] && !has_tuples[j as usize] {
                    continue;
                }

                // Keep behavior aligned with existing alpha candidate pruning:
                // recursive body refs are allowed only if head already has tuples.
                if !allow_recursive && (i == k || j == k) && !has_tuples[k as usize] {
                    continue;
                }

                triples.push((i, j, k));
            }
        }
        triples.sort_by_key(|&(i, j, k)| (k, i, j));
        Ok(triples)
    }

    fn expected_candidate_count(
        &self,
        mask_name: &str,
        allow_recursive: bool,
    ) -> PyResult<usize> {
        Ok(self
            .candidate_triples_for_mask(mask_name, allow_recursive)?
            .len())
    }

    fn fact_exists_in_buffer(
        provider: &CudaKernelProvider,
        buf: &xlog_cuda::CudaBuffer,
        values: &[i64],
    ) -> xlog_core::Result<bool> {
        use xlog_core::XlogError;
        let num_rows = read_device_row_count(provider, buf)? as usize;
        if num_rows == 0 { return Ok(false); }
        if values.len() != buf.arity() { return Ok(false); }

        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for col_idx in 0..buf.arity() {
            let col_type = schema.column_type(col_idx)
                .ok_or_else(|| XlogError::Kernel(
                    format!("Column {} type not found in schema", col_idx)
                ))?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => provider.download_column_i64(buf, col_idx)?,
                ScalarType::I32 => provider.download_column_i32(buf, col_idx)?
                    .into_iter().map(|v| v as i64).collect(),
                ScalarType::U32 | ScalarType::Symbol => {
                    provider.download_column_u32(buf, col_idx)?
                        .into_iter().map(|v| v as i64).collect()
                }
                ScalarType::U64 => {
                    let col_u64 = provider.download_column_u64(buf, col_idx)?;
                    col_u64.into_iter().map(|v| v as i64).collect()
                }
                ScalarType::Bool => {
                    provider.download_column_bool(buf, col_idx)?
                        .into_iter().map(|v| if v { 1i64 } else { 0i64 }).collect()
                }
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(XlogError::Kernel(
                        format!("fact_exists does not support float column type {:?}", col_type)
                    ));
                }
            };
            columns.push(col_i64);
        }

        for row in 0..num_rows {
            let mut matches = true;
            for (col_idx, val) in values.iter().enumerate() {
                if columns[col_idx][row] != *val {
                    matches = false;
                    break;
                }
            }
            if matches { return Ok(true); }
        }
        Ok(false)
    }

    /// Like fact_exists_in_buffer but checks only the projected columns.
    /// `projection[i]` is the column index in `buf` that corresponds to
    /// `values[i]` in the head relation.
    fn fact_exists_projected(
        provider: &CudaKernelProvider,
        buf: &xlog_cuda::CudaBuffer,
        values: &[i64],
        projection: &[usize],
    ) -> xlog_core::Result<bool> {
        use xlog_core::XlogError;
        let num_rows = read_device_row_count(provider, buf)? as usize;
        if num_rows == 0 { return Ok(false); }

        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for &col_idx in projection {
            if col_idx >= buf.arity() { return Ok(false); }
            let col_type = schema.column_type(col_idx)
                .ok_or_else(|| XlogError::Kernel(
                    format!("Column {} type not found in schema", col_idx)
                ))?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => provider.download_column_i64(buf, col_idx)?,
                ScalarType::I32 => provider.download_column_i32(buf, col_idx)?
                    .into_iter().map(|v| v as i64).collect(),
                ScalarType::U32 | ScalarType::Symbol => {
                    provider.download_column_u32(buf, col_idx)?
                        .into_iter().map(|v| v as i64).collect()
                }
                ScalarType::U64 => {
                    provider.download_column_u64(buf, col_idx)?
                        .into_iter().map(|v| v as i64).collect()
                }
                ScalarType::Bool => {
                    provider.download_column_bool(buf, col_idx)?
                        .into_iter().map(|v| if v { 1i64 } else { 0i64 }).collect()
                }
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(XlogError::Kernel(
                        format!("fact_exists does not support float column type {:?}", col_type)
                    ));
                }
            };
            columns.push(col_i64);
        }

        for row in 0..num_rows {
            let mut matches = true;
            for (i, val) in values.iter().enumerate() {
                if columns[i][row] != *val {
                    matches = false;
                    break;
                }
            }
            if matches { return Ok(true); }
        }
        Ok(false)
    }
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
    m.add_function(wrap_pyfunction!(train_model, m)?)?;
    m.add_function(wrap_pyfunction!(train_model_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(dlpack_roundtrip, m)?)?;
    #[cfg(feature = "arrow-device-import")]
    m.add_function(wrap_pyfunction!(export_arrow_device, m)?)?;
    #[cfg(feature = "arrow-device-import")]
    m.add_function(wrap_pyfunction!(import_arrow_device, m)?)?;
    Ok(())
}
