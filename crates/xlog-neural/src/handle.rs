//! Network handle for managing PyTorch modules.
//!
//! Each registered neural network is represented by a `NetworkHandle` which holds:
//! - The PyTorch module (nn.Module) - when `python` feature is enabled
//! - Optional optimizer for training
//! - Optional learning rate scheduler
//! - Configuration flags for batching, caching, etc.

#[cfg(feature = "python")]
use pyo3::PyObject;

/// Handle to a registered neural network.
///
/// This struct holds the PyTorch module and associated training state.
/// When the `python` feature is enabled, it can hold PyO3 PyObject references.
#[derive(Debug)]
pub struct NetworkHandle {
    /// Unique name identifying this network
    pub name: String,

    /// The PyTorch nn.Module (set via Python API)
    /// Only available with the `python` feature
    #[cfg(feature = "python")]
    pub module: Option<PyObject>,

    /// The optimizer for training (e.g., Adam, SGD)
    /// Only available with the `python` feature
    #[cfg(feature = "python")]
    pub optimizer: Option<PyObject>,

    /// Learning rate scheduler
    /// Only available with the `python` feature
    #[cfg(feature = "python")]
    pub scheduler: Option<PyObject>,

    /// Whether to batch inputs for efficient GPU processing
    pub batching: bool,

    /// Top-k sampling: if Some(k), only consider top k outputs
    pub k: Option<usize>,

    /// Deterministic mode: use argmax instead of sampling
    pub det: bool,

    /// Whether the network is in training mode
    pub train_mode: bool,

    /// Whether output caching is enabled
    pub cache_enabled: bool,

    /// Maximum number of cached outputs
    pub cache_size: usize,
}

impl NetworkHandle {
    /// Create a new network handle with the given name and default settings.
    pub fn new(name: String) -> Self {
        Self {
            name,
            #[cfg(feature = "python")]
            module: None,
            #[cfg(feature = "python")]
            optimizer: None,
            #[cfg(feature = "python")]
            scheduler: None,
            batching: true,
            k: None,
            det: false,
            train_mode: false,
            cache_enabled: true,
            cache_size: 10000,
        }
    }

    /// Create a handle from a configuration.
    pub fn from_config(config: &crate::NetworkConfig) -> Self {
        Self {
            name: config.name.clone(),
            #[cfg(feature = "python")]
            module: None,
            #[cfg(feature = "python")]
            optimizer: None,
            #[cfg(feature = "python")]
            scheduler: None,
            batching: config.batching,
            k: config.k,
            det: config.det,
            train_mode: false,
            cache_enabled: config.cache_enabled,
            cache_size: config.cache_size,
        }
    }

    /// Check if the PyTorch module has been set.
    #[cfg(feature = "python")]
    pub fn has_module(&self) -> bool {
        self.module.is_some()
    }

    /// Check if the PyTorch module has been set.
    /// Without Python feature, always returns false.
    #[cfg(not(feature = "python"))]
    /// Report whether a Python module/tensor handle is attached.
    pub fn has_module(&self) -> bool {
        false
    }

    /// Check if an optimizer has been configured.
    #[cfg(feature = "python")]
    pub fn has_optimizer(&self) -> bool {
        self.optimizer.is_some()
    }

    /// Check if an optimizer has been configured.
    /// Without Python feature, always returns false.
    #[cfg(not(feature = "python"))]
    pub fn has_optimizer(&self) -> bool {
        false
    }

    /// Check if a scheduler has been configured.
    #[cfg(feature = "python")]
    pub fn has_scheduler(&self) -> bool {
        self.scheduler.is_some()
    }

    /// Check if a scheduler has been configured.
    /// Without Python feature, always returns false.
    #[cfg(not(feature = "python"))]
    pub fn has_scheduler(&self) -> bool {
        false
    }

    /// Set the PyTorch module.
    #[cfg(feature = "python")]
    pub fn set_module(&mut self, module: PyObject) {
        self.module = Some(module);
    }

    /// Set the optimizer.
    #[cfg(feature = "python")]
    pub fn set_optimizer(&mut self, optimizer: PyObject) {
        self.optimizer = Some(optimizer);
    }

    /// Set the learning rate scheduler.
    #[cfg(feature = "python")]
    pub fn set_scheduler(&mut self, scheduler: PyObject) {
        self.scheduler = Some(scheduler);
    }

    /// Get a reference to the PyTorch module.
    #[cfg(feature = "python")]
    pub fn module(&self) -> Option<&PyObject> {
        self.module.as_ref()
    }

    /// Get a reference to the optimizer.
    #[cfg(feature = "python")]
    pub fn optimizer(&self) -> Option<&PyObject> {
        self.optimizer.as_ref()
    }

    /// Get a reference to the scheduler.
    #[cfg(feature = "python")]
    pub fn scheduler(&self) -> Option<&PyObject> {
        self.scheduler.as_ref()
    }

    /// Clear the module and training state.
    #[cfg(feature = "python")]
    pub fn clear(&mut self) {
        self.module = None;
        self.optimizer = None;
        self.scheduler = None;
    }

    /// Clear the module and training state.
    /// Without Python feature, this is a no-op.
    #[cfg(not(feature = "python"))]
    pub fn clear(&mut self) {
        // No-op without Python feature
    }
}

/// Handle to a registered embedding module.
///
/// Wraps either a trainable `nn.Embedding` or a frozen `torch.Tensor`.
/// Created via `CompiledProgram.register_embedding()` in Python.
#[derive(Debug)]
pub struct EmbeddingHandle {
    /// Unique name matching the nn() declaration
    pub name: String,

    /// The PyTorch nn.Embedding or tensor
    #[cfg(feature = "python")]
    pub module: Option<PyObject>,

    /// Whether gradients flow through this embedding
    pub trainable: bool,

    /// Embedding vector dimension (second axis of weight matrix)
    pub dim: usize,

    /// Number of embedding entries (first axis of weight matrix)
    pub vocab_size: usize,
}

impl EmbeddingHandle {
    /// Create a new embedding handle.
    pub fn new(name: String, trainable: bool, dim: usize, vocab_size: usize) -> Self {
        Self {
            name,
            #[cfg(feature = "python")]
            module: None,
            trainable,
            dim,
            vocab_size,
        }
    }

    /// Check if the PyTorch module/tensor has been set.
    #[cfg(feature = "python")]
    pub fn has_module(&self) -> bool {
        self.module.is_some()
    }

    /// Check if the PyTorch module/tensor has been set.
    /// Without Python feature, always returns false.
    #[cfg(not(feature = "python"))]
    pub fn has_module(&self) -> bool {
        false
    }

    /// Set the PyTorch module/tensor.
    #[cfg(feature = "python")]
    pub fn set_module(&mut self, module: PyObject) {
        self.module = Some(module);
    }

    /// Get a reference to the PyTorch module/tensor.
    #[cfg(feature = "python")]
    pub fn module(&self) -> Option<&PyObject> {
        self.module.as_ref()
    }
}

impl Default for NetworkHandle {
    fn default() -> Self {
        Self::new(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_new() {
        let handle = NetworkHandle::new("test".to_string());
        assert_eq!(handle.name, "test");
        assert!(!handle.has_module());
        assert!(!handle.has_optimizer());
        assert!(handle.batching);
        assert!(!handle.train_mode);
    }

    #[test]
    fn test_handle_from_config() {
        let config = crate::NetworkConfig {
            name: "configured".to_string(),
            batching: false,
            k: Some(5),
            det: true,
            cache_enabled: false,
            cache_size: 500,
        };

        let handle = NetworkHandle::from_config(&config);
        assert_eq!(handle.name, "configured");
        assert!(!handle.batching);
        assert_eq!(handle.k, Some(5));
        assert!(handle.det);
        assert!(!handle.cache_enabled);
        assert_eq!(handle.cache_size, 500);
    }

    #[test]
    fn test_embedding_handle_new() {
        let handle = EmbeddingHandle::new("test_embed".to_string(), true, 64, 1000);
        assert_eq!(handle.name, "test_embed");
        assert!(handle.trainable);
        assert_eq!(handle.dim, 64);
        assert_eq!(handle.vocab_size, 1000);
        assert!(!handle.has_module());
    }

    #[test]
    fn test_embedding_handle_frozen() {
        let handle = EmbeddingHandle::new("frozen".to_string(), false, 128, 500);
        assert!(!handle.trainable);
        assert_eq!(handle.dim, 128);
        assert_eq!(handle.vocab_size, 500);
    }
}
