use std::collections::{HashMap, HashSet};
use std::os::raw::{c_char, c_void};
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::{PyDict, PySequence};

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, DlpackManagedTensor, GpuMemoryManager};
use ::xlog_gpu::logic as gpu_logic;
use xlog_logic::ast::ProbEngine;
use xlog_neural::{NetworkConfig, NetworkRegistry, TensorMetadata, TensorSourceRegistry};
use xlog_prob::exact::{ExactDdnnfProgram, ExactResultWithGrads, GpuConfig, QueryProbability};
use xlog_prob::mc::{McEvalConfig, McProgram};

const DLPACK_CAPSULE_NAME: &[u8] = b"dltensor\0";
const USED_DLPACK_CAPSULE_NAME: &[u8] = b"used_dltensor\0";

/// Epsilon value for numerical stability in log computations
const NLL_EPSILON: f64 = 1e-38;

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

    let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, DLPACK_CAPSULE_NAME.as_ptr() as *const c_char);
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
    #[pyo3(signature = (source, device=0, memory_mb=1024, prob_engine=None))]
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
            declared_networks,
            tensor_sources: TensorSourceRegistry::new(),
            source: source.to_string(),
            gpu_config: config,
            prob_engine: engine,
        })
    }
}

enum CompiledProbProgram {
    Exact(ExactDdnnfProgram),
    Mc(McProgram),
}

impl CompiledProbProgram {
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
    /// Names of neural networks declared in the program (from nn() declarations)
    declared_networks: HashSet<String>,
    /// Registry for tensor data sources (images, embeddings, etc.)
    tensor_sources: TensorSourceRegistry,
    /// Original program source (for dynamic query compilation)
    source: String,
    /// GPU configuration
    gpu_config: GpuConfig,
    /// Probabilistic inference engine
    prob_engine: ProbEngine,
}

#[pymethods]
impl CompiledProgram {
    #[pyo3(signature = (return_grads=false, samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024))]
    pub fn evaluate(
        &self,
        py: Python<'_>,
        return_grads: bool,
        samples: Option<usize>,
        seed: Option<u64>,
        confidence: f64,
        max_nonmonotone_iterations: usize,
    ) -> PyResult<EvalResult> {
        match &self.program {
            CompiledProbProgram::Exact(program) => {
                if samples.is_some() || seed.is_some() {
                    return Err(PyValueError::new_err(
                        "samples/seed are only supported for prob_engine='mc'",
                    ));
                }
                if return_grads {
                    let result = program
                        .evaluate_gpu_with_grads()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    self.pack_result_with_grads(py, result)
                } else {
                    let result = program
                        .evaluate()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    self.pack_result_probs(py, result.query_probs)
                }
            }
            CompiledProbProgram::Mc(program) => {
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
                let result = program
                    .evaluate(cfg)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                self.pack_result_mc(py, result)
            }
        }
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

    /// Get names of all declared neural networks (from nn() declarations).
    fn declared_network_names(&self) -> Vec<String> {
        self.declared_networks.iter().cloned().collect()
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
    fn add_tensor_source(&mut self, py: Python<'_>, name: String, tensor: PyObject) -> PyResult<()> {
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
    fn train_epoch(&mut self, py: Python<'_>, queries: Vec<String>, batch_size: usize) -> PyResult<EpochStats> {
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
    /// This is the core training method that:
    /// 1. Extracts the neural network call from the query
    /// 2. Runs the network forward pass with gradient tracking
    /// 3. Computes NLL loss from the output probability
    /// 4. Backpropagates gradients to network parameters
    ///
    /// # Arguments
    /// * `query` - Query atom, e.g., "pred(0, a)" where pred is a neural predicate
    ///
    /// # Returns
    /// The NLL loss value
    ///
    /// # Note
    /// Call zero_grad() before this and optimizer_step() after.
    fn forward_backward(&mut self, py: Python<'_>, query: &str) -> PyResult<f64> {
        // Parse the query to extract network name, input index, and target label
        let (network_name, input_idx, target_label) = self.parse_neural_query(query)?;

        // Get the network handle
        let handle = self.network_registry.get(&network_name).ok_or_else(|| {
            PyValueError::new_err(format!("Network '{}' not registered", network_name))
        })?;

        let module = handle.module().ok_or_else(|| {
            PyValueError::new_err(format!("Network '{}' has no module", network_name))
        })?;

        // Get input from tensor source
        let input_tensor = self.get_input_tensor(py, input_idx)?;

        // Forward pass through network (with gradient tracking)
        // Network expects batched input, so add batch dimension if needed
        let input_bound = input_tensor.bind(py);
        let input_unsqueezed = input_bound.call_method1("unsqueeze", (0i32,))?;

        let output = module.call_method1(py, "__call__", (input_unsqueezed,))?;
        let output_bound = output.bind(py);

        // Output shape is [batch=1, num_classes], squeeze to [num_classes]
        let output_squeezed = output_bound.call_method1("squeeze", (0i32,))?;

        // Get probability of target label
        let label_idx = self.get_label_index(&network_name, &target_label)?;
        let prob_tensor = output_squeezed.get_item(label_idx)?;
        let prob: f64 = prob_tensor.call_method0("item")?.extract()?;

        // Compute NLL loss: -log(prob)
        let loss = nll_loss_value(prob);

        // Get gradient of loss w.r.t. prob: d(-log(p))/dp = -1/p
        let grad_prob = -1.0 / prob;

        // Create gradient tensor for backward through network
        // The gradient is with respect to the squeezed output (shape [num_classes])
        let grad_tensor = self.create_output_gradient(py, &output_squeezed, label_idx, grad_prob)?;

        // Add batch dimension back for backward pass through network output
        let grad_unsqueezed = grad_tensor.bind(py).call_method1("unsqueeze", (0i32,))?;

        // Backward through network
        output_bound.call_method1("backward", (grad_unsqueezed,))?;

        Ok(loss)
    }
}

impl CompiledProgram {
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
                batch_loss += self.forward_backward(py, query)?;
            }
            batch_loss /= batch.len() as f64;

            // Update parameters
            self.optimizer_step(py)?;

            total_loss += batch_loss;
            history.add_batch(batch_loss);

            // Log progress periodically
            if log_iter < usize::MAX && (batch_idx + 1) % log_iter == 0 {
                println!("  Batch {}/{}: loss={:.6}", batch_idx + 1, num_batches, batch_loss);
            }
        }

        Ok(EpochStats {
            avg_loss: total_loss / num_batches as f64,
            num_batches,
            total_queries: queries.len(),
        })
    }

    /// Parse a neural predicate query to extract network name, input index, and label.
    ///
    /// E.g., "pred(0, a)" -> ("net_name", 0, "a") where pred is defined by nn(net_name, ...) :: pred(...).
    fn parse_neural_query(&self, query: &str) -> PyResult<(String, usize, String)> {
        // Simple parser for queries like "pred(idx, label)" or "pred(idx,label)"
        let query = query.trim();

        // Find predicate name and arguments
        let paren_start = query.find('(').ok_or_else(|| {
            PyValueError::new_err(format!("Invalid query format: {}", query))
        })?;
        let paren_end = query.rfind(')').ok_or_else(|| {
            PyValueError::new_err(format!("Invalid query format: {}", query))
        })?;

        let pred_name = &query[..paren_start];
        let args_str = &query[paren_start + 1..paren_end];

        // Split arguments
        let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();
        if args.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Neural predicate query must have 2 arguments (index, label), got: {}",
                query
            )));
        }

        let input_idx: usize = args[0].parse().map_err(|_| {
            PyValueError::new_err(format!("Invalid input index: {}", args[0]))
        })?;
        let target_label = args[1].to_string();

        // Find network name for this predicate
        // For now, we assume predicate name matches a declared network or there's a mapping
        // In a full implementation, this would look up the nn() declaration
        let network_name = self.find_network_for_predicate(pred_name)?;

        Ok((network_name, input_idx, target_label))
    }

    /// Find the network name associated with a predicate.
    fn find_network_for_predicate(&self, pred_name: &str) -> PyResult<String> {
        // For v0.4.0-alpha, use simple heuristic: if only one network, use it
        // Otherwise, look for matching predicate pattern
        let network_names: Vec<_> = self.network_registry.names().iter().cloned().collect();

        if network_names.len() == 1 {
            return Ok(network_names[0].to_string());
        }

        // Try to find a network with matching name pattern
        for name in &network_names {
            if pred_name.contains(&**name) || name.contains(pred_name) {
                return Ok(name.to_string());
            }
        }

        // If no match, try the first declared network
        if let Some(name) = self.declared_networks.iter().next() {
            if self.network_registry.contains(name) {
                return Ok(name.clone());
            }
        }

        Err(PyValueError::new_err(format!(
            "Could not find network for predicate '{}'. Registered networks: {:?}",
            pred_name, network_names
        )))
    }

    /// Get the label index for a given label string.
    fn get_label_index(&self, _network_name: &str, label: &str) -> PyResult<usize> {
        // Try to parse as integer first
        if let Ok(idx) = label.parse::<usize>() {
            return Ok(idx);
        }

        // For symbolic labels, we need to look up in the label list
        // For v0.4.0-alpha, use simple mapping: a=0, b=1, c=2, etc.
        if label.len() == 1 {
            let c = label.chars().next().unwrap();
            if c.is_ascii_lowercase() {
                return Ok((c as usize) - ('a' as usize));
            }
        }

        Err(PyValueError::new_err(format!(
            "Could not resolve label '{}' to index",
            label
        )))
    }

    /// Get input tensor for a given index from the active tensor source.
    fn get_input_tensor(&self, py: Python<'_>, index: usize) -> PyResult<PyObject> {
        let tensor = self.tensor_sources.get_active().map_err(|e| {
            PyValueError::new_err(format!("No active tensor source: {}", e))
        })?;

        // Index into the tensor: tensor[index]
        let tensor_bound = tensor.bind(py);
        let indexed = tensor_bound.get_item(index)?;
        Ok(indexed.into())
    }

    /// Create gradient tensor for backward pass through network output.
    fn create_output_gradient(
        &self,
        py: Python<'_>,
        output: &Bound<'_, PyAny>,
        target_idx: usize,
        grad_value: f64,
    ) -> PyResult<PyObject> {
        let torch = py.import_bound("torch")?;

        // Get output shape
        let shape = output.getattr("shape")?;
        let size: usize = shape.get_item(0)?.extract().unwrap_or(1);

        // Create zero gradient tensor
        let zeros = torch.call_method1("zeros", (size,))?;

        // Set gradient at target index
        zeros.set_item(target_idx, grad_value)?;

        Ok(zeros.into())
    }

    /// Evaluate probability of a single query by compiling a temporary program.
    fn evaluate_query_probability(&self, query: &str) -> PyResult<f64> {
        let probs = self.evaluate_query_probabilities(&[query.to_string()])?;
        probs.into_iter().next().ok_or_else(|| {
            PyRuntimeError::new_err("Query evaluation returned no results")
        })
    }

    /// Evaluate probabilities for multiple queries by compiling a temporary program.
    fn evaluate_query_probabilities(&self, queries: &[String]) -> PyResult<Vec<f64>> {
        // Build source with queries appended
        let mut source_with_queries = self.source.clone();
        for query in queries {
            source_with_queries.push_str(&format!("\nquery({}).", query));
        }

        // Compile and evaluate the temporary program
        let result = match self.prob_engine {
            ProbEngine::ExactDdnnf => {
                let program = ExactDdnnfProgram::compile_source_with_gpu(
                    &source_with_queries,
                    self.gpu_config,
                ).map_err(|e| PyRuntimeError::new_err(format!("Query compilation error: {}", e)))?;

                program
                    .evaluate()
                    .map_err(|e| PyRuntimeError::new_err(format!("Query evaluation error: {}", e)))?
                    .query_probs
            }
            ProbEngine::Mc => {
                let program = McProgram::compile_source_with_gpu(
                    &source_with_queries,
                    self.gpu_config,
                ).map_err(|e| PyRuntimeError::new_err(format!("Query compilation error: {}", e)))?;

                let cfg = McEvalConfig::default();
                program
                    .evaluate(cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("Query evaluation error: {}", e)))?
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

    fn pack_result_probs(&self, py: Python<'_>, query_probs: Vec<QueryProbability>) -> PyResult<EvalResult> {
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

    fn pack_result_with_grads(&self, py: Python<'_>, result: ExactResultWithGrads) -> PyResult<EvalResult> {
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
    #[pyo3(signature = (source, device=0, memory_mb=1024))]
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
/// * `queries` - Training queries (e.g., ["addition(0, 1, 5)", "addition(2, 3, 7)"])
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

        let stats = program.train_epoch_internal(py, &epoch_queries, batch_size, log_iter, &mut history)?;
        history.add_epoch(stats.avg_loss);

        // Print epoch progress (visible in Python output)
        println!("Epoch {}/{}: avg_loss={:.6}", epoch + 1, epochs, stats.avg_loss);
    }

    Ok(history)
}

#[pymodule]
fn xlog_gpu(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Program>()?;
    m.add_class::<CompiledProgram>()?;
    m.add_class::<LogicProgram>()?;
    m.add_class::<CompiledLogicProgram>()?;
    m.add_class::<LogicQueryResult>()?;
    m.add_class::<LogicEvalResult>()?;
    m.add_class::<EvalResult>()?;
    // Training infrastructure
    m.add_class::<EpochStats>()?;
    m.add_class::<TrainingHistory>()?;
    m.add_function(wrap_pyfunction!(train_model, m)?)?;
    m.add_function(wrap_pyfunction!(dlpack_roundtrip, m)?)?;
    Ok(())
}
