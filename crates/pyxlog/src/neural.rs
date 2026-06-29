use std::collections::HashMap;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use xlog_core::{symbol, ScalarType, Schema};
#[cfg(feature = "host-io")]
use xlog_logic::ast::ArithExpr;
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::parse_program;
use xlog_neural::{EmbeddingHandle, NetworkConfig, NetworkHandle, TensorMetadata};
use xlog_prob::exact::ExactDdnnfProgram;
use xlog_prob::neural_fast_path::{GpuWeightSlots, NeuralFastPathConfig};

use std::collections::HashMap as StdHashMap;

use super::neural_registry::NeuralPredicateInfo;
use super::{
    dlpack_capsule_from_tensor, dlpack_from_py, types, CachedCircuit, CompiledProgram, EpochStats,
    HardFilter, InputSource, JoinPlan, NeuralGroup, QuerySignature, TrainingHistory,
};

/// Build the standard 1-column schema for probability values.
fn prob_schema(scalar_type: ScalarType) -> Schema {
    Schema::new(vec![("col0".to_string(), scalar_type)])
}

/// How a neural group's forward input row is fetched. The placeholder/slot path
/// reads the active per-query examples source; Stage-B real-domain groups read a
/// fixed row of the join-domain source (per-event features) or a dummy row (an
/// input-independent leaf such as the rule-weight guard).
#[derive(Clone)]
enum Fetch {
    Active(usize),
    Domain(usize),
    Dummy,
}

/// Comparable key for a constant term, used to test hard join conditions over
/// the program's ground facts.
#[derive(Debug, Clone, PartialEq)]
enum ConstKey {
    Int(i64),
    Sym(u32),
    Str(String),
    Float(u64),
}

fn const_key(term: &Term) -> Option<ConstKey> {
    match term {
        Term::Integer(v) => Some(ConstKey::Int(*v)),
        Term::Symbol(s) => Some(ConstKey::Sym(*s)),
        Term::String(s) => Some(ConstKey::Str(s.clone())),
        Term::Float(f) => Some(ConstKey::Float(f.to_bits())),
        _ => None,
    }
}

/// Accumulate one producing call's loss into a running device-resident total
/// via in-place add. The backward pass already ran inside the producing call,
/// so the loss is detached before accumulation; it is cast to f64 so f32
/// (direct) and f64 (batched) losses accumulate in a single dtype. No host
/// sync happens here — the total is read back once by the caller.
fn accumulate_device_loss(
    py: Python<'_>,
    total: Option<PyObject>,
    loss: PyObject,
) -> PyResult<PyObject> {
    let detached = loss
        .bind(py)
        .call_method0("detach")?
        .call_method0("double")?
        .unbind();
    Ok(match total {
        None => detached,
        Some(acc) => {
            acc.bind(py).call_method1("add_", (detached.bind(py),))?;
            acc
        }
    })
}

fn apply_network_output_mode(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    handle: &NetworkHandle,
) -> PyResult<PyObject> {
    let take = if handle.det { Some(1) } else { handle.k };
    let Some(k) = take else {
        return Ok(values.clone().unbind());
    };
    if k == 0 {
        return Err(PyValueError::new_err("k must be > 0"));
    }

    let ndim: usize = values.getattr("ndim")?.extract()?;
    if ndim != 1 {
        return Err(PyValueError::new_err(format!(
            "registered network output mode expects a 1-D tensor, got {}D",
            ndim
        )));
    }
    let numel: usize = values
        .call_method0("numel")?
        .extract::<i64>()?
        .try_into()
        .map_err(|_| PyValueError::new_err("tensor numel is negative"))?;
    let take = k.min(numel);
    let take_i64 = i64::try_from(take).map_err(|_| PyValueError::new_err("k exceeds i64::MAX"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("descending", true)?;
    kwargs.set_item("stable", true)?;
    let order = values.call_method("argsort", (), Some(&kwargs))?;
    let indices = order.call_method1("narrow", (0i32, 0i64, take_i64))?;

    let torch = py.import("torch")?;
    let mask = torch.call_method1("zeros_like", (values,))?;
    let ones = torch.call_method1("ones_like", (values,))?;
    let selected_ones = ones.call_method1("index_select", (0i32, &indices))?;
    mask.call_method1("scatter_", (0i32, &indices, &selected_ones))?;

    let selected = values.call_method1("__mul__", (&mask,))?;
    let clamp_kwargs = PyDict::new(py);
    clamp_kwargs.set_item("min", types::NLL_EPSILON)?;
    let denom = selected
        .call_method0("sum")?
        .call_method("clamp", (), Some(&clamp_kwargs))?;
    let soft = selected.call_method1("__truediv__", (&denom,))?;
    if handle.det {
        let detached_soft = soft.call_method0("detach")?;
        let straight_through = mask.call_method1(
            "__add__",
            (&soft.call_method1("__sub__", (&detached_soft,))?,),
        )?;
        Ok(straight_through.unbind())
    } else {
        Ok(soft.unbind())
    }
}

// =============================================================================
// Neural-specific #[pymethods]
// =============================================================================

#[pymethods]
impl CompiledProgram {
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
        if k == Some(0) {
            return Err(PyValueError::new_err("k must be > 0"));
        }

        // Validate network name exists in neural predicates
        if !self.declared_networks.contains(&name) {
            return Err(PyValueError::new_err(format!(
                "Network '{}' not declared in program. Declared networks: {:?}",
                name,
                self.declared_networks.iter().collect::<Vec<_>>()
            )));
        }

        // Cross-registration: reject if this network is declared as an embedding
        if let Some(&true) = self.declared_network_forms.get(&name) {
            return Err(PyValueError::new_err(format!(
                "declaration '{}' is an embedding; use register_embedding()",
                name
            )));
        }

        let mut config = NetworkConfig::default(&name);
        config.batching = batching;
        config.k = k;
        config.det = det;
        config.cache_enabled = cache;
        config.cache_size = cache_size;

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

    /// Register an embedding module for a declared embedding predicate.
    ///
    /// # Arguments
    ///
    /// * `name` - Must match an nn() declaration with no label list
    /// * `module_or_tensor` - PyTorch nn.Embedding or 2D torch.Tensor
    /// * `trainable` - Whether gradients flow; must be false for raw tensors
    #[pyo3(signature = (name, module_or_tensor, trainable=true))]
    fn register_embedding(
        &mut self,
        py: Python<'_>,
        name: String,
        module_or_tensor: PyObject,
        trainable: bool,
    ) -> PyResult<()> {
        // Validate network name exists
        if !self.declared_networks.contains(&name) {
            return Err(PyValueError::new_err(format!(
                "Network '{}' not declared in program. Declared networks: {:?}",
                name,
                self.declared_networks.iter().collect::<Vec<_>>()
            )));
        }

        // Cross-registration: reject if declared as classification
        match self.declared_network_forms.get(&name) {
            Some(&false) => {
                return Err(PyValueError::new_err(format!(
                    "declaration '{}' is a classification network; use register_network()",
                    name
                )));
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "Network '{}' not found in form index",
                    name
                )));
            }
            _ => {} // is_embedding = true, correct
        }

        let torch = py.import("torch")?;
        let nn = py.import("torch.nn")?;
        let obj = module_or_tensor.bind(py);

        // Detect payload type and extract shape
        let embedding_cls = nn.getattr("Embedding")?;
        let is_nn_embedding = obj.is_instance(&embedding_cls)?;

        let (vocab_size, dim, stored_obj) = if is_nn_embedding {
            // nn.Embedding: read .weight shape
            let weight = obj.getattr("weight")?;
            let shape = weight.getattr("shape")?;
            let vs: usize = shape.get_item(0)?.extract()?;
            let d: usize = shape.get_item(1)?.extract()?;

            // Validate dtype is float
            let dtype = weight.getattr("dtype")?;
            let float32 = torch.getattr("float32")?;
            let float64 = torch.getattr("float64")?;
            let float16 = torch.getattr("float16")?;
            let bfloat16 = torch.getattr("bfloat16")?;
            if !dtype.eq(&float32)?
                && !dtype.eq(&float64)?
                && !dtype.eq(&float16)?
                && !dtype.eq(&bfloat16)?
            {
                return Err(PyValueError::new_err(
                    "nn.Embedding weight must have float dtype",
                ));
            }

            (vs, d, module_or_tensor.clone_ref(py))
        } else {
            // Raw torch.Tensor: must be frozen
            let tensor_cls = torch.getattr("Tensor")?;
            if !obj.is_instance(&tensor_cls)? {
                return Err(PyValueError::new_err(
                    "module_or_tensor must be nn.Embedding or torch.Tensor",
                ));
            }

            if trainable {
                return Err(PyValueError::new_err(
                    "trainable=True requires nn.Embedding; raw torch.Tensor is always frozen",
                ));
            }

            // Validate rank == 2
            let ndim: usize = obj.getattr("ndim")?.extract()?;
            if ndim != 2 {
                return Err(PyValueError::new_err(format!(
                    "embedding tensor must be 2D [vocab_size, dim], got {}D",
                    ndim
                )));
            }

            let shape = obj.getattr("shape")?;
            let vs: usize = shape.get_item(0)?.extract()?;
            let d: usize = shape.get_item(1)?.extract()?;

            // Validate dtype is float
            let dtype = obj.getattr("dtype")?;
            let float32 = torch.getattr("float32")?;
            let float64 = torch.getattr("float64")?;
            let float16 = torch.getattr("float16")?;
            let bfloat16 = torch.getattr("bfloat16")?;
            if !dtype.eq(&float32)?
                && !dtype.eq(&float64)?
                && !dtype.eq(&float16)?
                && !dtype.eq(&bfloat16)?
            {
                return Err(PyValueError::new_err(
                    "embedding tensor must have float dtype",
                ));
            }

            // Detach raw tensor to enforce frozen contract
            let detached = obj.call_method0("detach")?;
            (vs, d, detached.unbind())
        };

        let mut handle = EmbeddingHandle::new(name.clone(), trainable, dim, vocab_size);
        handle.set_module(stored_obj);
        self.network_registry.register_embedding(handle);

        Ok(())
    }

    /// Look up embedding vectors by integer IDs.
    ///
    /// Returns a batched torch.Tensor with shape [len(ids), dim].
    /// For nn.Embedding: tensor has autograd graph (grad-enabled).
    /// For frozen torch.Tensor: tensor has requires_grad=False.
    ///
    /// This is the only gradient-carrying embedding API in v0.5.
    fn forward_embedding(&self, py: Python<'_>, name: String, ids: Vec<i64>) -> PyResult<PyObject> {
        let handle = self.network_registry.get_embedding(&name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Embedding '{}' not registered. Did you call register_embedding()?",
                name
            ))
        })?;

        let module = handle
            .module()
            .ok_or_else(|| PyValueError::new_err(format!("Embedding '{}' has no module", name)))?;

        let torch = py.import("torch")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", torch.getattr("long")?)?;

        // Determine device from the embedding weight/tensor so ids_tensor
        // is created on the same device (prevents CPU/CUDA mismatch).
        let device = if handle.trainable {
            module.getattr(py, "weight")?.getattr(py, "device")?
        } else {
            module.getattr(py, "device")?
        };
        kwargs.set_item("device", device)?;

        if handle.trainable {
            // nn.Embedding: call module(ids_tensor)
            let ids_tensor = torch.call_method("tensor", (ids,), Some(&kwargs))?;
            let result = module.call_method1(py, "__call__", (ids_tensor,))?;
            Ok(result)
        } else {
            // Frozen tensor: index directly
            let ids_tensor = torch.call_method("tensor", (ids,), Some(&kwargs))?;
            let result = module.call_method1(py, "__getitem__", (ids_tensor,))?;
            Ok(result)
        }
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

    /// Return neural bridge cache and deterministic-mode telemetry.
    fn neural_cache_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = PyDict::new(py);
        stats.set_item("circuit_cache_size", self.circuit_cache.len())?;
        stats.set_item("circuit_cache_hits", self.circuit_cache_hits)?;
        stats.set_item("circuit_cache_misses", self.circuit_cache_misses)?;
        stats.set_item("template_compile_count", self.template_compile_count)?;
        stats.set_item(
            "query_signature_cache_size",
            self.query_signature_cache.len(),
        )?;

        let networks = PyDict::new(py);
        for (name, handle) in self.network_registry.iter() {
            let entry = PyDict::new(py);
            entry.set_item("cache_enabled", handle.cache_enabled)?;
            entry.set_item("cache_size", handle.cache_size)?;
            entry.set_item("top_k", handle.k)?;
            entry.set_item("deterministic", handle.det)?;
            networks.set_item(name, entry)?;
        }
        stats.set_item("networks", networks)?;
        Ok(stats.into())
    }

    /// Deterministic stable top-k over a 1-D tensor.
    ///
    /// Ties are resolved by lower input index via stable descending argsort.
    fn deterministic_topk(
        &self,
        py: Python<'_>,
        values: &Bound<'_, PyAny>,
        k: usize,
    ) -> PyResult<PyObject> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be > 0"));
        }
        let ndim: usize = values.getattr("ndim")?.extract()?;
        if ndim != 1 {
            return Err(PyValueError::new_err(format!(
                "deterministic_topk expects a 1-D tensor, got {}D",
                ndim
            )));
        }
        let numel: usize = values
            .call_method0("numel")?
            .extract::<i64>()?
            .try_into()
            .map_err(|_| PyValueError::new_err("tensor numel is negative"))?;
        let take = k.min(numel);
        let take_i64 =
            i64::try_from(take).map_err(|_| PyValueError::new_err("k exceeds i64::MAX"))?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("descending", true)?;
        kwargs.set_item("stable", true)?;
        let order = values.call_method("argsort", (), Some(&kwargs))?;
        let indices = order.call_method1("narrow", (0i32, 0i64, take_i64))?;
        let top_values = values.call_method1("index_select", (0i32, &indices))?;

        let dict = PyDict::new(py);
        dict.set_item("indices", indices)?;
        dict.set_item("values", top_values)?;
        Ok(dict.into())
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
        let dict = PyDict::new(py);
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

    /// Register the Stage-B existential-join domain tensor source: stores the
    /// per-event feature batch under `name` AND records `name` as the join-domain
    /// source the forward reads (for `DomainRow`/`ConstDummy` groups). The name is
    /// owned by the Python driver and flows in as data — the engine holds no
    /// hardcoded source name.
    fn register_domain_tensor_source(
        &mut self,
        py: Python<'_>,
        name: String,
        tensor: PyObject,
    ) -> PyResult<()> {
        self.add_tensor_source(py, name.clone(), tensor)?;
        self.domain_source = Some(name);
        Ok(())
    }

    /// Set the active tensor source.
    ///
    /// The active source is used when neural predicates are evaluated.
    fn set_active_tensor_source(&mut self, name: String) -> PyResult<()> {
        self.tensor_sources
            .set_active(&name)
            .map_err(types::val_err)
    }

    /// Get the name of the currently active tensor source.
    fn active_tensor_source(&self) -> Option<String> {
        self.tensor_sources.active_name().map(|s| s.to_string())
    }

    /// Get the size (number of samples) of the active tensor source.
    fn active_tensor_source_size(&self) -> PyResult<usize> {
        self.tensor_sources.active_size().map_err(types::val_err)
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
    // Forward-Backward (Python-facing)
    // =========================================================================

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

    /// GPU-resident batched forward-backward over many queries with PER-QUERY
    /// `expected` labels, accumulating loss on device with a SINGLE host sync.
    ///
    /// The scalar [`forward_backward`] reads the loss back with `.item()` on
    /// every call, so a training step that loops it over N queries pays N host
    /// syncs and goes CPU-bound — the GPU sits idle between syncs. This method
    /// keeps the whole step device-resident: queries are partitioned by
    /// hard-filter eligibility and grouped by `(expected, circuit template)` so
    /// each group runs as one batched circuit pass, the losses accumulate on
    /// device, and exactly one `.item()` reads the summed loss back — one host
    /// sync per call regardless of N.
    ///
    /// Mixed positive/negative supervision is supported via the per-query
    /// `expected` vector; the existing batched epoch path fixes `expected=true`
    /// for every query and so cannot express it. Hard-filter gating is preserved
    /// exactly as in the scalar path: an ineligible query contributes a constant
    /// loss with no neural forward and no gradient (gradient isolation).
    ///
    /// Call `zero_grad()` before and `optimizer_step()` after, like
    /// `forward_backward`. Returns the summed NLL loss over all queries.
    #[pyo3(signature = (queries, expected))]
    fn forward_backward_grouped(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        expected: Vec<bool>,
    ) -> PyResult<f64> {
        if queries.len() != expected.len() {
            return Err(PyValueError::new_err(format!(
                "forward_backward_grouped: {} queries but {} expected labels",
                queries.len(),
                expected.len()
            )));
        }
        if queries.is_empty() {
            return Ok(0.0);
        }

        // Eligible complex queries grouped by (expected, circuit template) so
        // each group runs as one batched circuit pass; insertion-ordered for
        // deterministic gradient accumulation.
        let mut group_order: Vec<(bool, String)> = Vec::new();
        let mut groups: StdHashMap<(bool, String), Vec<Atom>> = StdHashMap::new();

        // Eligible (batched + direct) losses accumulate on device; ineligible
        // queries contribute a host-side constant that carries no gradient, so
        // it never enters the device path (and never costs a sync).
        let mut device_loss: Option<PyObject> = None;
        let mut host_const_loss: f64 = 0.0;

        for (query, &exp) in queries.iter().zip(expected.iter()) {
            match self.try_parse_direct_neural_query(query) {
                Ok((predicate, network_name, input_idx, target_label)) => {
                    let loss = self.forward_backward_direct_tensor(
                        py,
                        &predicate,
                        &network_name,
                        input_idx,
                        &target_label,
                        exp,
                    )?;
                    device_loss = Some(accumulate_device_loss(py, device_loss, loss)?);
                }
                Err(_) => {
                    let atom = self.parse_query_atom(query)?;
                    let signature = self
                        .get_or_build_query_signature(&atom.predicate, atom.terms.len())?
                        .clone();
                    // Same short-circuit as forward_backward_complex_tensor: an
                    // ineligible query is probability 0 with no neural forward,
                    // so no gradient flows through it. The constant matches
                    // zero_probability_loss(py, exp) exactly.
                    if !signature.hard_filters().is_empty()
                        && !self.hard_filters_satisfied(&atom, signature.hard_filters())?
                    {
                        host_const_loss += if exp { -(types::NLL_EPSILON.ln()) } else { 0.0 };
                        continue;
                    }
                    let key = self.generate_cache_key_for_signature(
                        &signature,
                        &atom.predicate,
                        atom.terms.len(),
                    );
                    let group_key = (exp, key);
                    if !groups.contains_key(&group_key) {
                        group_order.push(group_key.clone());
                    }
                    groups.entry(group_key).or_default().push(atom);
                }
            }
        }

        // One batched circuit pass per (expected, template) group.
        for group_key in &group_order {
            let atoms = groups.remove(group_key).expect("group populated above");
            // The batched pass returns per-query losses; sum them for this group.
            let loss = self.forward_backward_batch_complex_tensor(py, &atoms, group_key.0)?;
            let loss_sum = loss.bind(py).call_method0("sum")?.unbind();
            device_loss = Some(accumulate_device_loss(py, device_loss, loss_sum)?);
        }

        // Single host sync: read the device-accumulated loss back once.
        let device_total: f64 = match device_loss {
            Some(t) => t.bind(py).call_method0("item")?.extract()?,
            None => 0.0,
        };
        Ok(device_total + host_const_loss)
    }

    /// GPU-resident batched query probabilities (one circuit pass per template).
    ///
    /// Mirrors `exp(-forward_backward(query, true))` for every query but without
    /// the per-query host sync: eligible queries are grouped by circuit template
    /// and evaluated in one batched pass each (host syncs are O(templates), not
    /// O(N)), and an ineligible (hard-filtered) query takes the same near-zero
    /// probability as the scalar path. Used for the post-training probability
    /// readout so the whole training surface — not just the step loop — avoids
    /// per-query host syncs at corpus scale.
    #[pyo3(signature = (queries))]
    fn query_probabilities_grouped(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
    ) -> PyResult<Vec<f64>> {
        let n = queries.len();
        // Ineligible (hard-filtered) queries take exp(-zero_probability_loss(true))
        // = exp(-(-ln eps)) = eps, the same near-zero probability as the scalar path.
        let mut probs = vec![types::NLL_EPSILON; n];
        if n == 0 {
            return Ok(probs);
        }

        // Eligible complex queries grouped by template, tracking original indices.
        let mut group_order: Vec<String> = Vec::new();
        let mut groups: StdHashMap<String, (Vec<Atom>, Vec<usize>)> = StdHashMap::new();

        for (i, query) in queries.iter().enumerate() {
            match self.try_parse_direct_neural_query(query) {
                Ok(_) => {
                    // Direct queries are not template-batched here; evaluate one.
                    let loss = self.forward_backward(py, query, true)?;
                    probs[i] = (-loss).exp();
                }
                Err(_) => {
                    let atom = self.parse_query_atom(query)?;
                    let signature = self
                        .get_or_build_query_signature(&atom.predicate, atom.terms.len())?
                        .clone();
                    if !signature.hard_filters().is_empty()
                        && !self.hard_filters_satisfied(&atom, signature.hard_filters())?
                    {
                        continue; // ineligible: keep the eps default
                    }
                    let key = self.generate_cache_key_for_signature(
                        &signature,
                        &atom.predicate,
                        atom.terms.len(),
                    );
                    if !groups.contains_key(&key) {
                        group_order.push(key.clone());
                    }
                    let entry = groups
                        .entry(key)
                        .or_insert_with(|| (Vec::new(), Vec::new()));
                    entry.0.push(atom);
                    entry.1.push(i);
                }
            }
        }

        for key in &group_order {
            let (atoms, indices) = groups.remove(key).expect("group populated above");
            // Per-query NLL losses for the group (expected=true); prob = exp(-loss).
            let losses = self.forward_backward_batch_complex_tensor(py, &atoms, true)?;
            let prob_tensor = losses.bind(py).call_method0("neg")?.call_method0("exp")?;
            let prob_list: Vec<f64> = prob_tensor.call_method0("tolist")?.extract()?;
            for (j, &orig_i) in indices.iter().enumerate() {
                probs[orig_i] = prob_list[j];
            }
        }
        Ok(probs)
    }

    /// Per-candidate hard-filter eligibility for the joint multi-rule same-head
    /// mixture (ST-TRC Phase-1b, guard-only candidates). For each rule defining
    /// `(head_pred, arity)`, returns `(guard_predicate_name, mask)` where
    /// `mask[i]` is whether the ground head binding `head_pred(i)` satisfies that
    /// rule's hard join conditions (its ordinary-relation body atoms). The
    /// differentiable noisy-OR over `(mask × guard sigmoid)` is assembled
    /// torch-side by the caller; this exposes only the engine's relational
    /// eligibility, so a query contributes through candidate k exactly where
    /// candidate k's join holds — the OR-amalgamation gate.
    ///
    /// Guard-only candidates only: each defining rule must carry exactly the
    /// trainable guard as its neural group (the relational joins are the hard
    /// conditions). General multi-rule OR over candidates with neural predicates
    /// beyond the guard requires the circuit backward and is a documented
    /// follow-up.
    #[pyo3(signature = (head_pred, arity, num_queries))]
    fn joint_candidate_eligibility(
        &self,
        head_pred: &str,
        arity: usize,
        num_queries: usize,
    ) -> PyResult<Vec<(String, Vec<bool>)>> {
        let rules = self.find_query_rules(head_pred, arity);
        if rules.is_empty() {
            return Err(PyValueError::new_err(format!(
                "No rule defines query predicate '{}' with arity {}",
                head_pred, arity
            )));
        }
        let mut out: Vec<(String, Vec<bool>)> = Vec::with_capacity(rules.len());
        for rule in rules {
            let signature = self.build_query_signature_for_rule(rule, head_pred, arity)?;
            // Guard-only candidate: exactly one neural group (the guard).
            let guard_pred = signature
                .groups()
                .first()
                .map(|g| g.info.predicate.clone())
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Candidate rule for '{}' has no neural guard group; the joint \
                         multi-rule mixture requires guard-only candidates (relational joins \
                         plus a single trainable guard)",
                        head_pred
                    ))
                })?;
            let filters = signature.hard_filters();
            let mut mask = Vec::with_capacity(num_queries);
            for i in 0..num_queries {
                let atom = Atom {
                    predicate: head_pred.to_string(),
                    terms: (0..arity.max(1))
                        .map(|p| {
                            if p == 0 {
                                Term::Integer(i as i64)
                            } else {
                                Term::Integer(0)
                            }
                        })
                        .collect(),
                };
                let eligible = if filters.is_empty() {
                    true
                } else {
                    self.hard_filters_satisfied(&atom, filters)?
                };
                mask.push(eligible);
            }
            out.push((guard_pred, mask));
        }
        Ok(out)
    }

    /// Belnap-aware dual-channel loss terms for bridge training.
    #[pyo3(signature = (pro, contra, quarantine, pro_reward=1.0, contra_penalty=1.0, quarantine_penalty=1.0, reduction="mean"))]
    fn belnap_loss(
        &self,
        py: Python<'_>,
        pro: &Bound<'_, PyAny>,
        contra: &Bound<'_, PyAny>,
        quarantine: &Bound<'_, PyAny>,
        pro_reward: f64,
        contra_penalty: f64,
        quarantine_penalty: f64,
        reduction: &str,
    ) -> PyResult<PyObject> {
        let pro_term = pro.call_method1("__mul__", (pro_reward,))?.unbind();
        let contra_term = contra.call_method1("__mul__", (contra_penalty,))?.unbind();
        let quarantine_term = quarantine
            .call_method1("__mul__", (quarantine_penalty,))?
            .unbind();
        let penalty = contra_term
            .bind(py)
            .call_method1("__add__", (quarantine_term.bind(py),))?
            .unbind();
        let unreduced = penalty
            .bind(py)
            .call_method1("__sub__", (pro_term.bind(py),))?
            .unbind();

        let dict = PyDict::new(py);
        dict.set_item(
            "loss",
            reduce_tensor_obj(py, unreduced.clone_ref(py), reduction)?,
        )?;
        dict.set_item(
            "pro_reward",
            reduce_tensor_obj(py, pro_term.clone_ref(py), reduction)?,
        )?;
        dict.set_item(
            "contra_penalty",
            reduce_tensor_obj(py, contra_term.clone_ref(py), reduction)?,
        )?;
        dict.set_item(
            "quarantine_penalty",
            reduce_tensor_obj(py, quarantine_term.clone_ref(py), reduction)?,
        )?;
        dict.set_item(
            "cfr_regret_proxy",
            reduce_tensor_obj(py, penalty.clone_ref(py), reduction)?,
        )?;
        dict.set_item(
            "formula",
            "contra_penalty + quarantine_penalty - pro_reward",
        )?;
        Ok(dict.into())
    }

    /// Non-negative semantic violation loss.
    #[pyo3(signature = (violations, weight=1.0, reduction="mean"))]
    fn semantic_loss_tensor(
        &self,
        py: Python<'_>,
        violations: &Bound<'_, PyAny>,
        weight: f64,
        reduction: &str,
    ) -> PyResult<PyObject> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("min", 0.0)?;
        let clipped = violations.call_method("clamp", (), Some(&kwargs))?;
        let weighted = clipped.call_method1("__mul__", (weight,))?.unbind();
        reduce_tensor_obj(py, weighted, reduction)
    }

    /// Mean-squared-error tensor helper.
    #[pyo3(signature = (pred, target, weight=1.0, reduction="mean"))]
    fn mse_loss_tensor(
        &self,
        py: Python<'_>,
        pred: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
        weight: f64,
        reduction: &str,
    ) -> PyResult<PyObject> {
        let diff = pred.call_method1("__sub__", (target,))?;
        let sq = diff.call_method1("pow", (2.0f64,))?;
        let weighted = sq.call_method1("__mul__", (weight,))?.unbind();
        reduce_tensor_obj(py, weighted, reduction)
    }

    /// Information loss `-log(prob)` with clamping.
    #[pyo3(signature = (prob, weight=1.0, eps=types::NLL_EPSILON, reduction="mean"))]
    fn infoloss_tensor(
        &self,
        py: Python<'_>,
        prob: &Bound<'_, PyAny>,
        weight: f64,
        eps: f64,
        reduction: &str,
    ) -> PyResult<PyObject> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("min", eps)?;
        let clamped = prob.call_method("clamp", (), Some(&kwargs))?;
        let log_p = clamped.call_method0("log")?;
        let neg = log_p.call_method0("__neg__")?;
        let weighted = neg.call_method1("__mul__", (weight,))?.unbind();
        reduce_tensor_obj(py, weighted, reduction)
    }
}

// =============================================================================
// Private helpers (plain impl)
// =============================================================================

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
        max_grad_norm: Option<f64>,
        history: &mut TrainingHistory,
    ) -> PyResult<EpochStats> {
        if queries.is_empty() {
            return Ok(EpochStats {
                avg_loss: 0.0,
                num_batches: 0,
                total_queries: 0,
            });
        }

        let num_batches = queries.len().div_ceil(batch_size);
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

            // Clip gradients if requested
            if let Some(max_norm) = max_grad_norm {
                self.clip_grad_norms(py, max_norm)?;
            }

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
        max_grad_norm: Option<f64>,
        history: &mut TrainingHistory,
    ) -> PyResult<EpochStats> {
        if queries.is_empty() {
            return Ok(EpochStats {
                avg_loss: 0.0,
                num_batches: 0,
                total_queries: 0,
            });
        }

        let num_batches = queries.len().div_ceil(batch_size);
        let mut total_loss = 0.0;

        for (batch_idx, batch) in queries.chunks(batch_size).enumerate() {
            self.zero_grad(py)?;

            // Accumulate loss on device — no .item() per query
            let mut batch_loss_tensor: Option<PyObject> = None;

            if self.batch_queries {
                // -- Batched path: group queries by template -----
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
                    // The batched pass returns per-query losses; sum for the batch.
                    let loss = self.forward_backward_batch_complex_tensor(py, &group, true)?;
                    let loss_val = loss
                        .bind(py)
                        .call_method0("sum")?
                        .call_method0("detach")?
                        .unbind();
                    batch_loss_tensor = Some(match batch_loss_tensor {
                        None => loss_val,
                        Some(acc) => {
                            acc.bind(py).call_method1("add_", (loss_val.bind(py),))?;
                            acc
                        }
                    });
                }
            } else {
                // -- Sequential path (for regression testing / fallback) -
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

            // Clip gradients if requested
            if let Some(max_norm) = max_grad_norm {
                self.clip_grad_norms(py, max_norm)?;
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
            .map_err(types::val_err)?;

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

    /// Whether every hard join condition holds for this query's head bindings.
    /// Each condition is an ordinary relation that must contain the tuple formed
    /// by the query's (ground) head arguments at the recorded positions.
    fn hard_filters_satisfied(&self, atom: &Atom, filters: &[HardFilter]) -> PyResult<bool> {
        for filter in filters {
            let mut key: Vec<ConstKey> = Vec::with_capacity(filter.arg_head_positions.len());
            for &head_pos in &filter.arg_head_positions {
                let term = atom.terms.get(head_pos).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "hard condition references head position {} out of range for query '{}'",
                        head_pos, atom.predicate
                    ))
                })?;
                let k = const_key(term).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "query '{}' head argument at position {} is not a constant; \
                         trainable-rule queries must be ground",
                        atom.predicate, head_pos
                    ))
                })?;
                key.push(k);
            }
            if !self.relation_has_tuple(&filter.relation, &key) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether the program's facts contain `relation(key...)`. Scans ground
    /// facts (rules with an empty body); the relation's facts are compiled into
    /// the program.
    fn relation_has_tuple(&self, relation: &str, key: &[ConstKey]) -> bool {
        for fact in self.ast.facts() {
            if fact.head.predicate != relation || fact.head.terms.len() != key.len() {
                continue;
            }
            let matches = fact
                .head
                .terms
                .iter()
                .zip(key.iter())
                .all(|(ft, k)| const_key(ft).as_ref() == Some(k));
            if matches {
                return true;
            }
        }
        false
    }

    /// Loss for a derived atom that is hard-false (probability 0): NLL is
    /// `-log(eps)` when the example expects it true, else 0. Returned as a plain
    /// constant tensor — detached from every network graph, so the backward pass
    /// flows no gradient through a filtered-out query (gradient isolation).
    fn zero_probability_loss(&self, py: Python<'_>, expected: bool) -> PyResult<PyObject> {
        let loss_value = if expected {
            -(types::NLL_EPSILON.ln())
        } else {
            0.0
        };
        let torch = py.import("torch")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", torch.getattr("float64")?)?;
        let tensor = torch.call_method("tensor", (loss_value,), Some(&kwargs))?;
        Ok(tensor.into())
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
        let output_mode = apply_network_output_mode(py, &output_squeezed, handle)?;
        let output_squeezed = output_mode.bind(py);

        // Select the target probability tensor and compute loss on GPU:
        // loss = -log(clamp(prob, min=epsilon)).
        let label_idx = self.get_label_index(predicate, network_name, target_label)?;
        let prob_tensor = output_squeezed.get_item(label_idx)?;

        let torch = py.import("torch")?;
        let clamp_kwargs = PyDict::new(py);
        clamp_kwargs.set_item("min", types::NLL_EPSILON)?;
        let loss = if expected {
            let prob_clamped = prob_tensor.call_method("clamp", (), Some(&clamp_kwargs))?;
            let log_p = prob_clamped.call_method0("log")?;
            log_p.call_method0("__neg__")?
        } else {
            let device = prob_tensor.call_method0("device")?;
            let dtype = prob_tensor.call_method0("dtype")?;
            let one = torch.call_method1("tensor", (1.0f64,))?;
            let kwargs = PyDict::new(py);
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

        Ok(loss.unbind())
    }

    /// Forward-backward for a complex query involving neural predicates through rules.
    ///
    /// E.g., `addition(0, 1, 7)` where:
    /// - `addition(X, Y, Z) :- digit(X, LeftDigit), digit(Y, RightDigit),
    ///   Z is LeftDigit + RightDigit.`
    /// - `nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).`
    ///
    /// This method uses circuit caching to avoid Decision-DNNF circuit recompilation:
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

        // Hard join conditions (ordinary relations) gate which groundings can
        // fire. If any fails for this query's head bindings, the derived atom
        // is false: short-circuit to probability 0 with NO neural forward, so
        // no gradient flows through the fact atoms. Evaluating before the
        // network forward is what enforces the gradient isolation.
        if !signature.hard_filters().is_empty()
            && !self.hard_filters_satisfied(atom, signature.hard_filters())?
        {
            return self.zero_probability_loss(py, expected);
        }

        let input_indices: Vec<Fetch> = signature
            .groups()
            .iter()
            .map(|group| match &group.input_source {
                InputSource::QueryArg(pos) => {
                    self.term_to_input_idx(&atom.terms[*pos]).map(Fetch::Active)
                }
                InputSource::ImplicitSlot(slot) => Ok(Fetch::Active(*slot)),
                InputSource::DomainRow(row) => Ok(Fetch::Domain(*row)),
                InputSource::ConstDummy => Ok(Fetch::Dummy),
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
        let torch = py.import("torch")?;
        let schema_f32 = prob_schema(ScalarType::F32);

        #[derive(Clone)]
        struct NeuralCall {
            fetch: Fetch,
            order_idx: usize,
        }

        let mut calls_by_network: HashMap<String, Vec<NeuralCall>> = HashMap::new();
        for (order_idx, fetch) in input_indices.iter().enumerate() {
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
                    fetch: fetch.clone(),
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
                inputs.push(self.resolve_input_tensor(py, &call.fetch)?);
            }

            let input_list = pyo3::types::PyList::new(py, &inputs)?;
            let batch = torch.call_method1("stack", (input_list, 0i32))?;

            // Forward pass with gradient tracking (single batched forward).
            let output = module.call_method1(py, "__call__", (batch,))?;
            let output_bound = output.bind(py);

            for (batch_idx, call) in calls.iter().enumerate() {
                let output_row = output_bound.get_item(batch_idx)?;
                let output_row_mode = apply_network_output_mode(py, &output_row, handle)?;
                let output_row = output_row_mode.bind(py);
                let output_row = output_row.call_method0("contiguous")?;

                let output_detached = output_row.call_method0("detach")?;
                let managed = dlpack_from_py(&output_detached)?;
                let prob_buf = self
                    .output_provider
                    .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![managed])
                    .map_err(|e| types::gpu_err("DLPack import failed", e))?;

                let grad_tensor = torch.call_method1("zeros_like", (&output_row,))?;
                let grad_tensor = grad_tensor.call_method0("contiguous")?;
                let grad_managed = dlpack_from_py(&grad_tensor)?;
                let grad_buf = self
                    .output_provider
                    .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![grad_managed])
                    .map_err(|e| types::gpu_err("DLPack import failed", e))?;

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
        if self.circuit_cache.contains_key(&cache_key) {
            self.circuit_cache_hits = self.circuit_cache_hits.saturating_add(1);
        } else {
            self.circuit_cache_misses = self.circuit_cache_misses.saturating_add(1);
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
            .map_err(|e| types::gpu_err("Neural fast-path error", e))?;

        // `CudaBuffer` carries the row count on-device. For a single scalar loss, upload 1.
        let schema_f64 = prob_schema(ScalarType::F64);
        let mut d_num_rows = self
            .output_provider
            .memory()
            .alloc::<u32>(1)
            .map_err(|e| types::gpu_err("GPU allocation failed", e))?;
        self.output_provider
            .device()
            .inner()
            .htod_sync_copy_into(&[1u32], &mut d_num_rows)
            .map_err(|e| types::gpu_err("Failed to set row count", e))?;

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
            .map_err(|e| types::gpu_err("DLPack export failed", e))?;
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
                let kwargs = PyDict::new(py);
                kwargs.set_item("retain_graph", true)?;
                out_bound.call_method("backward", (grad.bind(py),), Some(&kwargs))?;
            }
        }

        Ok(loss_tensor.into())
    }

    /// Batch-process multiple queries that share the same circuit template.
    ///
    /// Instead of N separate forward passes, DLPack cycles, stream syncs, and
    /// backward passes, this method:
    /// 1. Stacks inputs per-network -> single batched forward pass
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

        // -- 1. Shared setup from first atom -----
        let signature = self
            .get_or_build_query_signature(&atoms[0].predicate, atoms[0].terms.len())?
            .clone();
        let n_groups = signature.groups().len();
        let pred_name = atoms[0].predicate.clone();

        // Ensure circuit is compiled/cached.
        let cache_key =
            self.generate_cache_key_for_signature(&signature, &pred_name, atoms[0].terms.len());
        if self.circuit_cache.contains_key(&cache_key) {
            self.circuit_cache_hits = self.circuit_cache_hits.saturating_add(1);
        } else {
            self.circuit_cache_misses = self.circuit_cache_misses.saturating_add(1);
            let (cached, profile) =
                self.compile_circuit_for_template(&signature, &pred_name, atoms[0].terms.len())?;
            self.circuit_cache.insert(cache_key.clone(), cached);
            self.template_compile_count = self.template_compile_count.saturating_add(1);
            if profile.is_some() {
                self.last_compile_profile = profile;
            }
        }

        // -- 2. Per-query data: input fetches + query_idx ----
        let mut per_query_inputs: Vec<Vec<Fetch>> = Vec::with_capacity(n_queries);
        let mut per_query_idx: Vec<usize> = Vec::with_capacity(n_queries);

        for atom in atoms {
            let input_indices: Vec<Fetch> = signature
                .groups()
                .iter()
                .map(|group| match &group.input_source {
                    InputSource::QueryArg(pos) => {
                        self.term_to_input_idx(&atom.terms[*pos]).map(Fetch::Active)
                    }
                    InputSource::ImplicitSlot(slot) => Ok(Fetch::Active(*slot)),
                    InputSource::DomainRow(row) => Ok(Fetch::Domain(*row)),
                    InputSource::ConstDummy => Ok(Fetch::Dummy),
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

        // -- 3. Group network calls: network -> [(query, group, input_idx)] -
        // Insertion-ordered Vec for deterministic gradient accumulation.
        struct NetCall {
            query: usize,
            group: usize,
            fetch: Fetch,
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
                    fetch: inputs[g].clone(),
                });
            }
        }

        // -- 4. Batched forward per network ------
        let torch = py.import("torch")?;
        let schema_f32 = prob_schema(ScalarType::F32);

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
                inputs.push(self.resolve_input_tensor(py, &c.fetch)?);
            }
            let input_list = pyo3::types::PyList::new(py, &inputs)?;
            let stacked = torch.call_method1("stack", (input_list, 0i32))?;

            // Single batched forward: [N_total, num_classes]
            let batch_output = module.call_method1(py, "__call__", (stacked,))?;
            let batch_output_bound = batch_output.bind(py);

            // Slice each row, validate label count, create DLPack buffers.
            for (i, call) in calls.iter().enumerate() {
                let row = batch_output_bound.get_item(i)?;
                let row_mode = apply_network_output_mode(py, &row, handle)?;
                let row = row_mode.bind(py);
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
                    .map_err(|e| types::gpu_err("DLPack import failed", e))?;

                let grad_tensor = torch.call_method1("zeros_like", (&row,))?;
                let grad_tensor = grad_tensor.call_method0("contiguous")?;
                let grad_managed = dlpack_from_py(&grad_tensor)?;
                let grad_buf = self
                    .output_provider
                    .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![grad_managed])
                    .map_err(|e| types::gpu_err("DLPack import failed", e))?;

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

        // -- 5. Stream sync: torch current -> default (once) -----
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

        // -- 6. Per-query circuit evaluation -----
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

        // -- 7. Stream sync: default -> torch current (once) -----
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

        // -- 7b. Export losses via DLPack and accumulate on device ----
        let schema_f64 = prob_schema(ScalarType::F64);
        let batch_loss_tensor: PyObject = match batched_loss_dev {
            Ok(loss_dev) => {
                let mut d_num_rows = self
                    .output_provider
                    .memory()
                    .alloc::<u32>(1)
                    .map_err(|e| types::gpu_err("GPU allocation failed", e))?;
                self.output_provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&[n_queries_u32], &mut d_num_rows)
                    .map_err(|e| types::gpu_err("Failed to set row count", e))?;

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
                    .map_err(|e| types::gpu_err("DLPack export failed", e))?;
                let loss_capsule = dlpack_capsule_from_tensor(py, loss_dl)?;
                // Per-query 1D loss tensor (length n_queries), in input order.
                torch
                    .getattr("from_dlpack")?
                    .call1((loss_capsule,))?
                    .unbind()
            }
            Err(_batch_err) => {
                // Fallback path: preserve prior semantics if batched circuit path
                // is unavailable for this circuit. Collect per-query 1D losses and
                // concatenate so the return shape matches the batched path.
                let mut per_query_losses: Vec<PyObject> = Vec::with_capacity(n_queries);
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
                        .map_err(|e| types::gpu_err("Neural fast-path error", e))?;

                    let mut d_num_rows = self
                        .output_provider
                        .memory()
                        .alloc::<u32>(1)
                        .map_err(|e| types::gpu_err("GPU allocation failed", e))?;
                    self.output_provider
                        .device()
                        .inner()
                        .htod_sync_copy_into(&[1u32], &mut d_num_rows)
                        .map_err(|e| types::gpu_err("Failed to set row count", e))?;

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
                        .map_err(|e| types::gpu_err("DLPack export failed", e))?;
                    let loss_capsule = dlpack_capsule_from_tensor(py, loss_dl)?;
                    let loss_tensor = torch.getattr("from_dlpack")?.call1((loss_capsule,))?;
                    per_query_losses.push(loss_tensor.unbind());
                }
                if per_query_losses.is_empty() {
                    return Err(PyRuntimeError::new_err("No loss computed in batch"));
                }
                let loss_list = pyo3::types::PyList::new(py, per_query_losses.iter())?;
                torch.call_method1("cat", (loss_list,))?.unbind()
            }
        };

        // -- 8. Batched backward through all networks ----
        // torch.autograd.backward(tensors, grad_tensors) — single backward pass
        // through the shared batched computation graph.
        let autograd = torch.getattr("autograd")?;
        let out_list = pyo3::types::PyList::new(py, all_out_tensors.iter())?;
        let grad_list = pyo3::types::PyList::new(py, all_grad_tensors.iter())?;
        autograd.call_method1("backward", (out_list, grad_list))?;

        // -- 9. Return per-query losses (1D, input order) ------
        // Callers that want the scalar batch loss call `.sum()`; callers that
        // want per-query probabilities use the per-query losses directly.
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
        self.build_query_signature_for_rule(rule, pred_name, arity)
    }

    /// Build the query signature from ONE specific defining rule. The single-rule
    /// path reaches this via `build_query_signature` (after `find_query_rule`);
    /// the multi-rule same-head path (ST-TRC Phase-1b) builds one signature per
    /// candidate rule and OR-amalgamates them in the forward.
    fn build_query_signature_for_rule(
        &self,
        rule: &Rule,
        pred_name: &str,
        arity: usize,
    ) -> PyResult<QuerySignature> {
        // ---- Pass 1: collect neural occurrences and ordinary-relation atoms. ----
        // The query template grounds ONLY nn/4 predicates; an ordinary relation
        // is either a HARD pre-filter (head-bound args) or a Stage-B existential
        // join (an existential arg that is a neural predicate's input variable).
        struct NeuralOccurrence {
            info: NeuralPredicateInfo,
            input_var: String,
            input_head_pos: Option<usize>,
            #[cfg(feature = "host-io")]
            output_var: Option<String>,
        }
        let mut neural_occ: Vec<NeuralOccurrence> = Vec::new();
        let mut ordinary_atoms: Vec<&Atom> = Vec::new();

        for literal in &rule.body {
            let body_atom = match literal {
                BodyLiteral::Positive(atom) => atom,
                BodyLiteral::Negated(atom) => {
                    return Err(PyValueError::new_err(format!(
                        "Query rule for '{}' has negated literal 'not {}(..)'; \
                         negation is not supported by the neural query template path",
                        pred_name, atom.predicate
                    )));
                }
                BodyLiteral::Epistemic(_) => {
                    return Err(PyValueError::new_err(format!(
                        "Query rule for '{}' has an epistemic literal; \
                         epistemic operators are not supported by the neural query template path",
                        pred_name
                    )));
                }
                // Comparisons / is-expressions / univ are grounded by the
                // circuit compiler itself and remain supported.
                _ => continue,
            };

            if self.neural_registry.get(&body_atom.predicate).is_none() {
                ordinary_atoms.push(body_atom);
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
                Term::Variable(name) => name.clone(),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Expected variable at input position {} of neural call '{}'",
                        input_position, info.predicate
                    )))
                }
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
            let input_head_pos = Self::find_head_position(&rule.head, &input_var);
            neural_occ.push(NeuralOccurrence {
                info: info.clone(),
                input_var,
                input_head_pos,
                #[cfg(feature = "host-io")]
                output_var,
            });
        }

        // ---- Pass 2: classify ordinary atoms as hard filters or Stage-B joins. --
        // A join is an ordinary relation with EXACTLY one existential (non-head)
        // argument variable that IS the input variable of a neural occurrence; its
        // remaining argument variables bind head positions. Such a relation is
        // grounded INSIDE the circuit (real-domain), not stripped as a pre-filter.
        struct JoinRecord {
            relation: String,
            neural_idx: usize,
            join_var_arg_pos: usize,
            head_key: Vec<(usize, usize)>, // (relation arg position, head position)
        }
        let mut joins: Vec<JoinRecord> = Vec::new();
        let mut hard_filters: Vec<HardFilter> = Vec::new();

        for atom in &ordinary_atoms {
            let mut existential_join: Option<(usize, usize)> = None;
            let mut existential_nonjoin: Option<String> = None;
            let mut head_key: Vec<(usize, usize)> = Vec::new();
            let mut arg_head_positions: Vec<usize> = Vec::with_capacity(atom.terms.len());
            let mut non_variable: Option<usize> = None;
            for (i, term) in atom.terms.iter().enumerate() {
                match term {
                    Term::Variable(name) => match Self::find_head_position(&rule.head, name) {
                        Some(pos) => {
                            head_key.push((i, pos));
                            arg_head_positions.push(pos);
                        }
                        None => match neural_occ.iter().position(|o| &o.input_var == name) {
                            Some(nidx) => {
                                if existential_join.is_some() {
                                    return Err(PyValueError::new_err(format!(
                                        "Query rule for '{}' relation '{}' joins on more than one \
                                         existential neural-input variable; only a single \
                                         existential join variable is supported",
                                        pred_name, atom.predicate
                                    )));
                                }
                                existential_join = Some((i, nidx));
                            }
                            None => existential_nonjoin = Some(name.clone()),
                        },
                    },
                    _ => non_variable = Some(i),
                }
            }

            if let Some(name) = existential_nonjoin {
                return Err(PyValueError::new_err(format!(
                    "Query rule for '{}' joins ordinary relation '{}' on existential variable '{}'; \
                     existential-join conditions are supported only when the variable is the input \
                     of a neural predicate (Stage-B). Only head-variable hard filters and \
                     neural-input existential joins are supported.",
                    pred_name, atom.predicate, name
                )));
            }

            // Ground-facts-only requirement shared by joins and hard filters: a
            // derived relation has no materialized extension and would not ground.
            let is_derived = self
                .ast
                .rules
                .iter()
                .any(|r| r.head.predicate == atom.predicate && !r.body.is_empty());

            if let Some((join_var_arg_pos, neural_idx)) = existential_join {
                if is_derived {
                    return Err(PyValueError::new_err(format!(
                        "Query rule for '{}' uses derived relation '{}' as an existential join; \
                         the join domain is materialized from ground facts only. Materialize it \
                         upstream and supply it as ground facts.",
                        pred_name, atom.predicate
                    )));
                }
                joins.push(JoinRecord {
                    relation: atom.predicate.clone(),
                    neural_idx,
                    join_var_arg_pos,
                    head_key,
                });
                continue;
            }

            if let Some(i) = non_variable {
                return Err(PyValueError::new_err(format!(
                    "Query rule for '{}' hard-condition relation '{}' has a non-variable argument \
                     at position {}; only variables bound to head positions are supported",
                    pred_name, atom.predicate, i
                )));
            }

            if is_derived {
                return Err(PyValueError::new_err(format!(
                    "Query rule for '{}' uses derived relation '{}' as a hard condition; \
                     hard conditions are checked against ground facts only, so a derived \
                     relation would silently exclude every grounding. Materialize it \
                     upstream and supply it as ground facts.",
                    pred_name, atom.predicate
                )));
            }
            hard_filters.push(HardFilter {
                relation: atom.predicate.clone(),
                arg_head_positions,
            });
        }

        let joins_present = !joins.is_empty();

        // ---- Pass 3: build neural groups (expanded over the real join domain). --
        let mut groups: Vec<NeuralGroup> = Vec::with_capacity(neural_occ.len());
        for (occ_idx, occ) in neural_occ.iter().enumerate() {
            if let Some(join) = joins.iter().find(|j| j.neural_idx == occ_idx) {
                // Join-target group (e.g. saliency): one leaf per real event id.
                let event_domain =
                    self.materialize_int_domain(&join.relation, join.join_var_arg_pos)?;
                if event_domain.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "Query rule for '{}': existential-join relation '{}' has no ground facts \
                         to materialize the join domain for variable '{}'",
                        pred_name, join.relation, occ.input_var
                    )));
                }
                for (row, ev) in event_domain.iter().enumerate() {
                    groups.push(NeuralGroup {
                        info: occ.info.clone(),
                        input_source: InputSource::DomainRow(row),
                        ground_const: Some(Term::Integer(*ev)),
                        #[cfg(feature = "host-io")]
                        output_var: occ.output_var.clone(),
                    });
                }
            } else if joins_present {
                // In a join rule, the remaining neural predicates must be
                // head-keyed (e.g. the rule-weight guard): one leaf per real head
                // constant, fed a dummy input (the leaf weight is input-independent).
                let head_pos = occ.input_head_pos.ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Query rule for '{}': neural predicate '{}' over existential variable '{}' \
                         is neither the join target nor head-bound; unsupported in a Stage-B join rule",
                        pred_name, occ.info.predicate, occ.input_var
                    ))
                })?;
                let (rel, arg_pos) = joins
                    .iter()
                    .find_map(|j| {
                        j.head_key
                            .iter()
                            .find(|(_, hp)| *hp == head_pos)
                            .map(|(ap, _)| (j.relation.clone(), *ap))
                    })
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Query rule for '{}': head-keyed neural predicate '{}' is not bound by \
                             any existential-join relation at head position {}",
                            pred_name, occ.info.predicate, head_pos
                        ))
                    })?;
                let edge_domain = self.materialize_int_domain(&rel, arg_pos)?;
                if edge_domain.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "Query rule for '{}': join relation '{}' has no ground facts to materialize \
                         the head-key domain",
                        pred_name, rel
                    )));
                }
                for edge in edge_domain.iter() {
                    groups.push(NeuralGroup {
                        info: occ.info.clone(),
                        input_source: InputSource::ConstDummy,
                        ground_const: Some(Term::Integer(*edge)),
                        #[cfg(feature = "host-io")]
                        output_var: occ.output_var.clone(),
                    });
                }
            } else {
                // No join present: the original placeholder/slot behavior.
                let input_source = match occ.input_head_pos {
                    Some(head_position) => InputSource::QueryArg(head_position),
                    None => {
                        let next_slot = groups
                            .iter()
                            .filter(|g| matches!(g.input_source, InputSource::ImplicitSlot(_)))
                            .count();
                        InputSource::ImplicitSlot(next_slot)
                    }
                };
                groups.push(NeuralGroup {
                    info: occ.info.clone(),
                    input_source,
                    ground_const: None,
                    #[cfg(feature = "host-io")]
                    output_var: occ.output_var.clone(),
                });
            }
        }

        if groups.is_empty() {
            return Err(PyValueError::new_err(format!(
                "No neural groups found for query predicate '{}'",
                pred_name
            )));
        }

        if joins_present {
            // Stage-B existential-join signature: the head ranges over the real
            // head-key domain (e.g. edges) and the groups are real-domain-grounded.
            let mut head_positions: Vec<usize> = joins
                .iter()
                .flat_map(|j| j.head_key.iter().map(|(_, hp)| *hp))
                .collect();
            head_positions.sort_unstable();
            head_positions.dedup();
            if head_positions.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "Query rule for '{}': Stage-B existential join currently supports exactly one \
                     head-key position; found {:?}",
                    pred_name, head_positions
                )));
            }
            let target_position = head_positions[0];
            let (rel, arg_pos) = joins
                .iter()
                .find_map(|j| {
                    j.head_key
                        .iter()
                        .find(|(_, hp)| *hp == target_position)
                        .map(|(ap, _)| (j.relation.clone(), *ap))
                })
                .expect("target position came from a join head_key");
            let head_domain: Vec<String> = self
                .materialize_int_domain(&rel, arg_pos)?
                .into_iter()
                .map(|v| v.to_string())
                .collect();
            let mut relations: Vec<String> = joins.iter().map(|j| j.relation.clone()).collect();
            relations.sort();
            relations.dedup();
            return Ok(QuerySignature::Targeted {
                target_position,
                groups,
                hard_filters,
                join: Some(JoinPlan {
                    relations,
                    head_domain,
                }),
            });
        }

        if arity == 0 {
            return Ok(QuerySignature::Boolean {
                groups,
                hard_filters,
            });
        }

        let mut used_head_positions: Vec<usize> = groups
            .iter()
            .filter_map(|group| match group.input_source {
                InputSource::QueryArg(pos) => Some(pos),
                _ => None,
            })
            .collect();
        used_head_positions.sort_unstable();
        used_head_positions.dedup();

        let target_positions: Vec<usize> = (0..arity)
            .filter(|pos| !used_head_positions.contains(pos))
            .collect();
        if target_positions.is_empty() {
            // Every head position is consumed by a neural input, so the query
            // atom is fully ground: supervise the truth of the derived atom
            // itself (boolean NLL with the caller's `expected` flag).
            return Ok(QuerySignature::Boolean {
                groups,
                hard_filters,
            });
        }
        if target_positions.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "Could not determine unique target position for '{}': target positions {:?}",
                pred_name, target_positions
            )));
        }

        Ok(QuerySignature::Targeted {
            target_position: target_positions[0],
            groups,
            hard_filters,
            join: None,
        })
    }

    /// Materialize the sorted, distinct integer constants appearing at argument
    /// position `arg_pos` of a relation's GROUND facts — the real domain a
    /// Stage-B existential join grounds the neural predicate over. Row order in
    /// this vector is the canonical domain order: it indexes `DomainRow(_)` for
    /// the per-event leaves and orders the per-edge `prob_queries`.
    fn materialize_int_domain(&self, relation: &str, arg_pos: usize) -> PyResult<Vec<i64>> {
        use std::collections::BTreeSet;
        let mut set: BTreeSet<i64> = BTreeSet::new();
        for fact in self.ast.facts() {
            if fact.head.predicate != relation {
                continue;
            }
            let term = fact.head.terms.get(arg_pos).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "relation '{}' ground fact has no argument at position {}",
                    relation, arg_pos
                ))
            })?;
            match term {
                Term::Integer(v) => {
                    set.insert(*v);
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Stage-B join domain for relation '{}' arg {} must be integer constants",
                        relation, arg_pos
                    )))
                }
            }
        }
        Ok(set.into_iter().collect())
    }

    /// All rules defining `(pred_name, arity)` — the multi-rule same-head
    /// candidate set for the ST-TRC joint soft-mixture. The single-rule helper
    /// `find_query_rule` enforces exactly one for the legacy path; the multi-rule
    /// forward consumes the full set and OR-amalgamates them.
    fn find_query_rules(&self, pred_name: &str, arity: usize) -> Vec<&Rule> {
        self.ast
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == pred_name && rule.head.arity() == arity)
            .collect()
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
                "Query predicate '{}' has {} defining rules; expected exactly 1 matching rule",
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
            .map_err(types::val_err)
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

        // Canonical placeholder per head position: groups that read the same
        // head variable must share one placeholder constant so their template
        // facts unify with the (single) binding of that variable.
        let mut head_to_group: StdHashMap<usize, usize> = StdHashMap::new();
        for (idx, group) in signature.groups().iter().enumerate() {
            if let InputSource::QueryArg(pos) = group.input_source {
                head_to_group.entry(pos).or_insert(idx);
            }
        }

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
                        // Stage-B real-domain grounding pins the input position at
                        // a real constant (event/edge id); otherwise use the
                        // placeholder shared per head variable.
                        let term = match &group.ground_const {
                            Some(c) => c.clone(),
                            None => {
                                let placeholder = match group.input_source {
                                    InputSource::QueryArg(head_pos) => {
                                        *head_to_group.get(&head_pos).unwrap_or(&group_idx)
                                    }
                                    _ => group_idx,
                                };
                                Term::Integer(placeholder as i64)
                            }
                        };
                        terms.push(term);
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

        // The probabilistic circuit covers only the neural part of the body.
        // Hard join conditions (ordinary relations) are evaluated separately as
        // a pre-filter, so strip them from the template rule — otherwise the
        // circuit would try to satisfy a relation that has no facts in the
        // synthetic template program and collapse the query to empty.
        let hard_relations: std::collections::HashSet<&str> = signature
            .hard_filters()
            .iter()
            .map(|f| f.relation.as_str())
            .collect();
        let mut circuit_rule = template_rule.clone();
        if !hard_relations.is_empty() {
            circuit_rule.body.retain(|lit| match lit {
                BodyLiteral::Positive(atom) => !hard_relations.contains(atom.predicate.as_str()),
                _ => true,
            });
        }
        template_program.rules.push(circuit_rule);

        // Stage-B existential join: the join relations stay INSIDE the circuit
        // rule (not stripped) so provenance OR-aggregates
        // `OR_event(neural(event) ∧ join(event, head))` at each head binding. Add
        // their ground facts to the template program so the join grounds against
        // the real domain.
        if let Some(join) = signature.join() {
            for fact in self.ast.facts() {
                if join.relations.iter().any(|r| r == &fact.head.predicate) {
                    template_program.rules.push((*fact).clone());
                }
            }
        }

        match signature {
            QuerySignature::Boolean { .. } => {
                // Fully-ground boolean query: each head position carries the
                // canonical placeholder of the neural group that consumes it.
                let mut terms = Vec::with_capacity(query_arity);
                for pos in 0..query_arity {
                    let group_idx = head_to_group.get(&pos).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Boolean query '{}' head position {} is not bound to any neural input",
                            query_pred, pos
                        ))
                    })?;
                    terms.push(Term::Integer(*group_idx as i64));
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
            QuerySignature::Targeted {
                target_position, ..
            } => {
                if target_domain.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "Targeted signature '{}' has empty target domain",
                        query_pred
                    )));
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
        // Stage-B existential join: the target domain IS the real head-key domain
        // (e.g. edge ids), materialized in the signature builder.
        if let Some(join) = signature.join() {
            return Ok(join.head_domain.clone());
        }

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
    ) -> PyResult<(
        CachedCircuit,
        Option<xlog_prob::compilation::CircuitCompileProfile>,
    )> {
        #[cfg(not(feature = "host-io"))]
        {
            let _ = (signature, query_pred, query_arity);
            return Err(types::host_io_disabled_pyerr());
        }

        #[cfg(feature = "host-io")]
        {
            let (expanded_ast, target_domain) =
                self.generate_template_ast(signature, query_pred, query_arity)?;
            let program = ExactDdnnfProgram::compile_from_program(&expanded_ast, self._gpu_config)
                .map_err(|e| types::gpu_err("Query compilation error", e))?;

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
                .map_err(|e| types::gpu_err("Slot map upload error", e))?;

            Ok((
                CachedCircuit {
                    program,
                    slots,
                    target_domain,
                },
                compile_profile,
            ))
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

    /// Get input tensor for a given index from a specific NAMED tensor source
    /// (regardless of which source is active). Used by Stage-B join groups to read
    /// the per-event feature batch from the registered domain source.
    fn get_input_tensor_named(
        &self,
        py: Python<'_>,
        source: &str,
        index: usize,
    ) -> PyResult<PyObject> {
        let tensor = self
            .tensor_sources
            .get_named(source)
            .map_err(|e| PyValueError::new_err(format!("No tensor source '{}': {}", source, e)))?;
        let indexed = tensor.bind(py).get_item(index)?;
        Ok(indexed.into())
    }

    /// Resolve a forward input row for one neural group per its fetch kind. The
    /// domain/dummy fetches read the Stage-B join-domain source whose NAME was
    /// supplied by the Python driver (`register_domain_tensor_source`); the engine
    /// holds no hardcoded source name.
    fn resolve_input_tensor(&self, py: Python<'_>, fetch: &Fetch) -> PyResult<PyObject> {
        match fetch {
            Fetch::Active(idx) => self.get_input_tensor(py, *idx),
            Fetch::Domain(idx) => self.get_input_tensor_named(py, self.domain_source_name()?, *idx),
            // Input-independent leaf (rule-weight guard): reuse domain row 0 — it
            // is on the correct device/dtype, and the module ignores input values.
            Fetch::Dummy => self.get_input_tensor_named(py, self.domain_source_name()?, 0),
        }
    }

    /// The Stage-B join-domain source name supplied by the Python driver. A
    /// `Domain`/`Dummy` fetch is only produced for a join signature, which is built
    /// only after the domain source is registered — so a missing name here is an
    /// internal invariant violation, surfaced as a clear error rather than papered
    /// over with a hardcoded fallback name.
    fn domain_source_name(&self) -> PyResult<&str> {
        self.domain_source.as_deref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "Stage-B join forward requested the domain tensor source, but none was \
                 registered; call register_domain_tensor_source before training a join rule",
            )
        })
    }
}

fn reduce_tensor_obj(py: Python<'_>, tensor: PyObject, reduction: &str) -> PyResult<PyObject> {
    match reduction {
        "none" => Ok(tensor),
        "sum" => Ok(tensor.bind(py).call_method0("sum")?.unbind()),
        "mean" => Ok(tensor.bind(py).call_method0("mean")?.unbind()),
        other => Err(PyValueError::new_err(format!(
            "Unsupported reduction '{}'; expected 'none', 'sum', or 'mean'",
            other
        ))),
    }
}
