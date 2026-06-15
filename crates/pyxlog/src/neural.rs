use std::collections::BTreeSet;
use std::collections::HashMap;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use xlog_core::{symbol, ScalarType, Schema};
#[cfg(feature = "host-io")]
use xlog_logic::ast::ArithExpr;
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::parse_program;
use xlog_logic::{classify_trainable_body_literal, TrainableBodyClass};
use xlog_neural::{EmbeddingHandle, NetworkConfig, NetworkHandle, TensorMetadata};
use xlog_prob::exact::ExactDdnnfProgram;
use xlog_prob::neural_fast_path::{GpuWeightSlots, NeuralFastPathConfig};

use std::collections::HashMap as StdHashMap;

use super::neural_registry::NeuralPredicateInfo;
use super::{
    dlpack_capsule_from_tensor, dlpack_from_py, types, CachedCircuit, CompiledProgram, EpochStats,
    GateLiteral, InputSource, NeuralGroup, QuerySignature, TrainingHistory,
};

/// Build the standard 1-column schema for probability values.
fn prob_schema(scalar_type: ScalarType) -> Schema {
    Schema::new(vec![("col0".to_string(), scalar_type)])
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

        // Per-example boolean truth of each Stage-A gate at this query's key.
        let gate_truths = self.evaluate_gate_truths(&signature, atom)?;

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
        let torch = py.import("torch")?;
        let schema_f32 = prob_schema(ScalarType::F32);

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
        let mut prob_bufs: Vec<xlog_cuda::CudaBuffer> = prob_bufs
            .into_iter()
            .map(|v| v.ok_or_else(|| PyRuntimeError::new_err("Missing prob buffer")))
            .collect::<PyResult<_>>()?;
        let mut grad_bufs: Vec<xlog_cuda::CudaBuffer> = grad_bufs
            .into_iter()
            .map(|v| v.ok_or_else(|| PyRuntimeError::new_err("Missing grad buffer")))
            .collect::<PyResult<_>>()?;

        // Stage-A gate leaves: append one fixed (no-grad) buffer per gate,
        // carrying the per-example boolean truth, matching the gate slots in
        // compile_circuit_for_template (keeps probs.len() == num_groups). They
        // are filled into the circuit weights but never backpropagated (no
        // network owns them), so the gate stays constant and gets no gradient.
        // The source tensors are kept alive until after the GPU kernel runs.
        let mut _gate_keepalive: Vec<PyObject> = Vec::with_capacity(gate_truths.len() * 2);
        for truth in &gate_truths {
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", torch.getattr("float32")?)?;
            kwargs.set_item("device", "cuda")?;
            let prob_t = torch
                .call_method("tensor", (vec![*truth],), Some(&kwargs))?
                .call_method0("contiguous")?;
            let prob_managed = dlpack_from_py(&prob_t)?;
            let prob_buf = self
                .output_provider
                .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![prob_managed])
                .map_err(|e| types::gpu_err("Gate DLPack import failed", e))?;
            let grad_t = torch
                .call_method("zeros", (vec![1i64],), Some(&kwargs))?
                .call_method0("contiguous")?;
            let grad_managed = dlpack_from_py(&grad_t)?;
            let grad_buf = self
                .output_provider
                .from_dlpack_tensors_with_schema(schema_f32.clone(), vec![grad_managed])
                .map_err(|e| types::gpu_err("Gate DLPack import failed", e))?;
            prob_bufs.push(prob_buf);
            grad_bufs.push(grad_buf);
            _gate_keepalive.push(prob_t.unbind());
            _gate_keepalive.push(grad_t.unbind());
        }

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

        // -- 2. Per-query data: input indices + query_idx ----
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

        // -- 3. Group network calls: network -> [(query, group, input_idx)] -
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
                inputs.push(self.get_input_tensor(py, c.input_idx)?);
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
                let loss_tensor = torch.getattr("from_dlpack")?.call1((loss_capsule,))?;
                loss_tensor.call_method0("sum")?.unbind()
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

                    accum = Some(match accum {
                        None => loss_tensor.unbind(),
                        Some(acc) => {
                            acc.bind(py).call_method1("add_", (loss_tensor,))?;
                            acc
                        }
                    });
                }
                accum.ok_or_else(|| PyRuntimeError::new_err("No loss computed in batch"))?
            }
        };

        // -- 8. Batched backward through all networks ----
        // torch.autograd.backward(tensors, grad_tensors) — single backward pass
        // through the shared batched computation graph.
        let autograd = torch.getattr("autograd")?;
        let out_list = pyo3::types::PyList::new(py, all_out_tensors.iter())?;
        let grad_list = pyo3::types::PyList::new(py, all_grad_tensors.iter())?;
        autograd.call_method1("backward", (out_list, grad_list))?;

        // -- 9. Return accumulated loss ------
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

        // Bound variables for Stage-A gate classification: the rule head
        // variables plus every variable appearing in a neural-predicate input
        // position. Computed up front (order-independent) so a gate may share a
        // neural input variable that a later body literal binds.
        let mut bound_vars: BTreeSet<String> = BTreeSet::new();
        for term in &rule.head.terms {
            if let Term::Variable(name) = term {
                bound_vars.insert(name.clone());
            }
        }
        for literal in &rule.body {
            if let BodyLiteral::Positive(atom) = literal {
                if self.neural_registry.get(&atom.predicate).is_some() {
                    let info = self.match_neural_decl_for_atom(atom)?;
                    for &pos in &info.input_positions {
                        if let Some(Term::Variable(name)) = atom.terms.get(pos) {
                            bound_vars.insert(name.clone());
                        }
                    }
                }
            }
        }

        let is_neural = |predicate: &str| self.neural_registry.get(predicate).is_some();

        let mut groups: Vec<NeuralGroup> = Vec::with_capacity(rule.body.len());
        let mut gates: Vec<GateLiteral> = Vec::new();
        for literal in &rule.body {
            // Stage-A relaxation: a non-neural relation that introduces no new
            // variable is a per-example gate (fixed circuit leaf), not a
            // fail-closed rejection. Unbound joins (Stage B), negation, and
            // epistemics remain typed errors.
            match classify_trainable_body_literal(literal, &bound_vars, &is_neural) {
                // Comparisons / is-expressions / univ are grounded by the
                // circuit compiler itself and remain supported.
                TrainableBodyClass::Builtin => continue,
                TrainableBodyClass::Negated => {
                    let predicate = match literal {
                        BodyLiteral::Negated(atom) => atom.predicate.as_str(),
                        _ => unreachable!("classifier returns Negated only for negated literals"),
                    };
                    return Err(PyValueError::new_err(format!(
                        "Query rule for '{}' has negated literal 'not {}(..)'; \
                         negation is not supported by the neural query template path",
                        pred_name, predicate
                    )));
                }
                TrainableBodyClass::Epistemic => {
                    return Err(PyValueError::new_err(format!(
                        "Query rule for '{}' has an epistemic literal; \
                         epistemic operators are not supported by the neural query template path",
                        pred_name
                    )));
                }
                TrainableBodyClass::UnboundJoin { var } => {
                    let predicate = match literal {
                        BodyLiteral::Positive(atom) => atom.predicate.as_str(),
                        _ => unreachable!("classifier returns UnboundJoin only for positive literals"),
                    };
                    return Err(PyValueError::new_err(format!(
                        "Query rule for '{}' relation '{}' introduces unbound join variable '{}'; \
                         mixed-body joins (Stage B) are not yet supported (bind it via the head or \
                         a neural input, or rename it to '_' for an existence check)",
                        pred_name, predicate, var
                    )));
                }
                TrainableBodyClass::Gate => {
                    let body_atom = match literal {
                        BodyLiteral::Positive(atom) => atom,
                        _ => unreachable!("classifier returns Gate only for positive literals"),
                    };
                    // Fail closed on genuine typos: a Stage-A gate must name a
                    // relation actually defined in the program. Distinct from
                    // the Stage-B "unbound join" message above.
                    if !self.relation_is_defined(&body_atom.predicate) {
                        return Err(PyValueError::new_err(format!(
                            "Query rule for '{}' references undefined relation '{}/{}'",
                            pred_name,
                            body_atom.predicate,
                            body_atom.terms.len()
                        )));
                    }
                    gates.push(GateLiteral {
                        atom: body_atom.clone(),
                    });
                    continue;
                }
                TrainableBodyClass::Neural => {}
            }

            let body_atom = match literal {
                BodyLiteral::Positive(atom) => atom,
                _ => unreachable!("classifier returns Neural only for positive literals"),
            };

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
            return Ok(QuerySignature::Boolean { groups, gates });
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
        if target_positions.is_empty() {
            // Every head position is consumed by a neural input, so the query
            // atom is fully ground: supervise the truth of the derived atom
            // itself (boolean NLL with the caller's `expected` flag).
            return Ok(QuerySignature::Boolean { groups, gates });
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
            gates,
        })
    }

    /// Whether a relation is defined in the program — as a fact or rule head, a
    /// probabilistic fact, or an annotated-disjunction choice. Used to fail
    /// closed on a misspelled Stage-A gate relation.
    fn relation_is_defined(&self, predicate: &str) -> bool {
        self.ast.rules.iter().any(|r| r.head.predicate == predicate)
            || self
                .ast
                .prob_facts
                .iter()
                .any(|f| f.atom.predicate == predicate)
            || self
                .ast
                .annotated_disjunctions
                .iter()
                .any(|ad| ad.choices.iter().any(|c| c.atom.predicate == predicate))
    }

    /// Per-example boolean truth (1.0/0.0) of each Stage-A gate for a ground
    /// query atom. The query's ground terms are substituted into the rule head
    /// to bind the gate variables, then membership is checked against the
    /// program's facts. Order matches `signature.gates()` (and the gate slots).
    fn evaluate_gate_truths(
        &self,
        signature: &QuerySignature,
        query_atom: &Atom,
    ) -> PyResult<Vec<f32>> {
        let gates = signature.gates();
        if gates.is_empty() {
            return Ok(Vec::new());
        }
        let rule = self.find_query_rule(&query_atom.predicate, query_atom.terms.len())?;
        let mut subst: StdHashMap<&str, &Term> = StdHashMap::new();
        for (pos, head_term) in rule.head.terms.iter().enumerate() {
            if let Term::Variable(name) = head_term {
                if let Some(query_term) = query_atom.terms.get(pos) {
                    subst.insert(name.as_str(), query_term);
                }
            }
        }
        let mut truths = Vec::with_capacity(gates.len());
        for gate in gates {
            let ground_terms: Vec<Term> = gate
                .atom
                .terms
                .iter()
                .map(|term| match term {
                    Term::Variable(name) => subst
                        .get(name.as_str())
                        .map(|t| (*t).clone())
                        .unwrap_or_else(|| term.clone()),
                    other => other.clone(),
                })
                .collect();
            let holds = self.ground_atom_holds(&gate.atom.predicate, &ground_terms);
            truths.push(if holds { 1.0 } else { 0.0 });
        }
        Ok(truths)
    }

    /// Whether a ground atom holds as an EDB fact (empty-body rule) in the
    /// program. Stage-A gates are deterministic relations; derived (rule-headed)
    /// gates are out of scope here.
    fn ground_atom_holds(&self, predicate: &str, terms: &[Term]) -> bool {
        self.ast.rules.iter().any(|rule| {
            rule.body.is_empty()
                && rule.head.predicate == predicate
                && rule.head.terms.as_slice() == terms
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
                        let placeholder = match group.input_source {
                            InputSource::QueryArg(head_pos) => {
                                *head_to_group.get(&head_pos).unwrap_or(&group_idx)
                            }
                            InputSource::ImplicitSlot(_) => group_idx,
                        };
                        terms.push(Term::Integer(placeholder as i64));
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

        // Stage-A gates: emit one fixed (single-choice) leaf per gate, grounded
        // at the same placeholder constants as the query so the compiled circuit
        // conjoins `neural ∧ guard ∧ gate`. The leaf's per-example truth weight
        // (1.0/0.0) is injected in the forward pass (Task 5). Gate leaves are
        // emitted AFTER the neural disjunctions so their circuit random vars sort
        // after the neural slots (see compile_circuit_for_template).
        let head_var_pos: StdHashMap<&str, usize> = template_rule
            .head
            .terms
            .iter()
            .enumerate()
            .filter_map(|(pos, term)| match term {
                Term::Variable(name) => Some((name.as_str(), pos)),
                _ => None,
            })
            .collect();
        for gate in signature.gates() {
            let mut terms = Vec::with_capacity(gate.atom.terms.len());
            for term in &gate.atom.terms {
                match term {
                    Term::Variable(name) => {
                        let head_pos = head_var_pos.get(name.as_str()).ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "Gate relation '{}' variable '{}' is not a head variable; \
                                 only head-bound gates are supported (Stage A)",
                                gate.atom.predicate, name
                            ))
                        })?;
                        let placeholder = head_to_group.get(head_pos).ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "Gate relation '{}' variable '{}' maps to head position {} which \
                                 is not consumed by a neural input (Stage A limitation)",
                                gate.atom.predicate, name, head_pos
                            ))
                        })?;
                        terms.push(Term::Integer(*placeholder as i64));
                    }
                    other => terms.push(other.clone()),
                }
            }
            template_program
                .annotated_disjunctions
                .push(xlog_logic::ast::AnnotatedDisjunction {
                    choices: vec![xlog_logic::ast::ProbFact {
                        prob: 0.5,
                        atom: Atom {
                            predicate: gate.atom.predicate.clone(),
                            terms,
                        },
                    }],
                });
        }

        template_program.rules.push(template_rule.clone());

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

            // Each Stage-A gate adds exactly one fixed circuit random var,
            // emitted after the neural disjunctions (see generate_template_ast).
            let num_gates = signature.gates().len();
            let expected_total: usize = group_label_counts.iter().sum::<usize>() + num_gates;
            if random_vars.len() != expected_total {
                return Err(PyRuntimeError::new_err(format!(
                    "Template compilation produced {} random vars, expected {} (groups: {:?}, gates: {})",
                    random_vars.len(),
                    expected_total,
                    group_label_counts,
                    num_gates
                )));
            }

            let mut slot_groups = Vec::with_capacity(group_label_counts.len() + num_gates);
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
            // Gate slots: one random var each, in signature.gates() order.
            for _ in 0..num_gates {
                let end = offset + 1;
                if end > random_vars.len() {
                    return Err(PyRuntimeError::new_err(format!(
                        "Template compilation random vars exhausted for gate slot at offset {}",
                        offset
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
