use std::collections::{HashMap, HashSet};
use std::os::raw::{c_char, c_void};
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use xlog_core::{MemoryBudget, Schema};
#[cfg(feature = "arrow-device-import")]
use xlog_cuda::{ArrowDeviceArray, ArrowDeviceArrayOwned};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, DlpackManagedTensor, GpuMemoryManager};
use xlog_gpu::logic as gpu_logic;
use xlog_logic::ast::ProbEngine;
use xlog_neural::{NetworkRegistry, TensorSourceRegistry};
use xlog_prob::exact::GpuConfig;

use xlog_core::RelId;
use xlog_ir::ExecutionPlan;
use xlog_logic::ast::Program as AstProgram;
use xlog_runtime::{Executor, RelationStore};

mod neural_registry;
use neural_registry::NeuralPredicateRegistry;
mod dlpack;
mod ilp;
mod ilp_exact;
mod ilp_gpu;
mod logic;
mod neural;
mod program;
mod training;
mod types;
pub(crate) use program::{
    CachedCircuit, CompiledProbProgram, InputSource, NeuralGroup, QuerySignature,
};

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

pub(crate) fn dlpack_capsule_from_tensor(
    py: Python<'_>,
    tensor: DlpackManagedTensor,
) -> PyResult<PyObject> {
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
pub(crate) fn arrow_device_capsule_from_device_array(
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
pub(crate) fn arrow_device_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ArrowDeviceArrayOwned> {
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

#[pyclass]
pub struct Program;

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
    pub(crate) query_signature_cache: HashMap<String, QuerySignature>,
    /// Cache of compiled circuits by template signature
    pub(crate) circuit_cache: HashMap<String, CachedCircuit>,
    /// Number of times the template compilation path executed.
    pub(crate) template_compile_count: usize,
    /// When true, batch queries sharing the same circuit template in training.
    pub(crate) batch_queries: bool,
    /// Latest circuit compilation profile (populated on cache miss when profiling).
    pub(crate) last_compile_profile: Option<xlog_prob::compilation::CircuitCompileProfile>,
}

#[pyclass]
pub struct LogicProgram;

#[pyclass]
pub struct CompiledLogicProgram {
    pub(crate) program: gpu_logic::LogicProgram,
    pub(crate) provider: Arc<CudaKernelProvider>,
}

#[pyclass]
pub struct LogicRelationSession {
    pub(crate) program: gpu_logic::LogicProgram,
    pub(crate) provider: Arc<CudaKernelProvider>,
    pub(crate) relation_store: RelationStore,
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
pub struct IlpTaggedCreditDeviceResult {
    #[pyo3(get)]
    pub fact_row_offsets: PyObject,
    #[pyo3(get)]
    pub entry_indices: PyObject,
    #[pyo3(get)]
    pub entry_i: PyObject,
    #[pyo3(get)]
    pub entry_j: PyObject,
    #[pyo3(get)]
    pub entry_k: PyObject,
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
    pub(crate) candidate_order: Option<Vec<(u32, u32, u32)>>,
    pub(crate) relation_overrides: HashMap<String, CudaBuffer>,
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
    m.add_class::<LogicRelationSession>()?;
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
    m.add_class::<IlpTaggedCreditDeviceResult>()?;
    m.add_function(wrap_pyfunction!(training::train_model, m)?)?;
    m.add_function(wrap_pyfunction!(training::train_model_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(dlpack::dlpack_roundtrip, m)?)?;
    #[cfg(feature = "arrow-device-import")]
    m.add_function(wrap_pyfunction!(dlpack::export_arrow_device, m)?)?;
    #[cfg(feature = "arrow-device-import")]
    m.add_function(wrap_pyfunction!(dlpack::import_arrow_device, m)?)?;
    Ok(())
}
