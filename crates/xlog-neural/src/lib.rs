//! Neural network integration for XLOG probabilistic logic programs.
#![warn(missing_docs)]
//!
//! This crate provides the infrastructure for integrating PyTorch neural networks
//! with XLOG's probabilistic inference engine, following the DeepProbLog paradigm.
//!
//! # Architecture
//!
//! The neural integration consists of:
//!
//! - **NetworkRegistry**: Central registry managing all neural networks
//! - **NetworkHandle**: Holds PyTorch module, optimizer, and configuration
//! - **NetworkConfig**: Configuration options for network behavior
//!
//! # Features
//!
//! - `python` - Enable Python interop via PyO3. Required for actual PyTorch integration.
//!
//! # Example
//!
//! ```
//! use xlog_neural::{NetworkRegistry, NetworkConfig};
//!
//! let mut registry = NetworkRegistry::new();
//! registry.register(NetworkConfig::default("mnist_net"));
//!
//! // Set all networks to training mode
//! registry.set_train_mode(true);
//! ```

pub mod batch;
pub mod bridge;
pub mod handle;
pub mod registry;
pub mod tensor_source;

pub use batch::{BatchCollector, BatchMapping, BatchResult, NeuralCall};
pub use bridge::{ADProbability, CircuitLeaf, NeuralBridge, NeuralOutput};
pub use handle::{EmbeddingHandle, NetworkHandle};
pub use registry::{NetworkConfig, NetworkRegistry};
pub use tensor_source::{TensorMetadata, TensorSourceError, TensorSourceRegistry};

/// Re-export pyo3 when the python feature is enabled
#[cfg(feature = "python")]
pub use pyo3;

/// Error types for neural network operations
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NeuralError {
    /// Network not found in registry
    #[error("Network not found: {0}")]
    NetworkNotFound(String),

    /// Network already registered
    #[error("Network already registered: {0}")]
    NetworkAlreadyExists(String),

    /// PyTorch module error
    #[error("PyTorch error: {0}")]
    PyTorchError(String),

    /// Invalid network configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Batch processing error
    #[error("Batch error: {0}")]
    BatchError(String),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),
}

/// Result type for neural operations
pub type NeuralResult<T> = std::result::Result<T, NeuralError>;

// Error conversion seams — orphan rule: NeuralError is defined here.
impl From<NeuralError> for xlog_core::XlogError {
    fn from(e: NeuralError) -> Self {
        xlog_core::XlogError::Execution(e.to_string())
    }
}

impl From<tensor_source::TensorSourceError> for xlog_core::XlogError {
    fn from(e: tensor_source::TensorSourceError) -> Self {
        xlog_core::XlogError::Execution(e.to_string())
    }
}
