//! Tensor Source Registry
//!
//! This module manages external tensor data (images, embeddings, etc.) that
//! can be indexed by neural predicates during proof search.
//!
//! # Architecture
//!
//! In DeepProbLog-style programs, neural predicates reference external data:
//!
//! ```text
//! nn(mnist_net, [X], Y, [0..9]) :: digit(X, Y).
//! ?- digit(42, Y).  // X=42 indexes into tensor source
//! ```
//!
//! The tensor source registry:
//! - Stores named tensor sources (train, test, etc.)
//! - Tracks the "active" source for current evaluation
//! - Validates indices are within bounds
//! - Holds PyTorch tensors via PyO3 (when `python` feature enabled)
//!
//! # Usage
//!
//! ```
//! use xlog_neural::tensor_source::{TensorSourceRegistry, TensorMetadata};
//!
//! let mut registry = TensorSourceRegistry::new();
//!
//! // Add sources with metadata
//! registry.add_with_metadata("train", TensorMetadata::new(60000, vec![1, 28, 28]));
//! registry.add_with_metadata("test", TensorMetadata::new(10000, vec![1, 28, 28]));
//!
//! // Set active source
//! registry.set_active("train").unwrap();
//!
//! // Validate index before neural call
//! registry.check_index(42).unwrap();
//! ```

use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "python")]
use pyo3::PyObject;

/// Errors from tensor source operations.
#[derive(Error, Debug)]
pub enum TensorSourceError {
    /// Tensor source not found in registry
    #[error("Tensor source '{0}' not found")]
    NotFound(String),

    /// No active tensor source set
    #[error("No active tensor source set")]
    NoActive,

    /// Index out of bounds for the active source
    #[error("Index {0} out of bounds for source with {1} entries")]
    IndexOutOfBounds(usize, usize),
}

/// Metadata about a tensor source (without the actual tensor data).
///
/// Used for validation and introspection without requiring Python GIL.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Number of samples in the tensor (first dimension)
    pub size: usize,
    /// Shape of each sample (excluding batch dimension)
    pub shape: Vec<usize>,
    /// Data type as string (e.g., "float32", "float64")
    pub dtype: String,
}

impl TensorMetadata {
    /// Create metadata with default dtype (float32).
    pub fn new(size: usize, shape: Vec<usize>) -> Self {
        Self {
            size,
            shape,
            dtype: "float32".to_string(),
        }
    }

    /// Create metadata with explicit dtype.
    pub fn with_dtype(size: usize, shape: Vec<usize>, dtype: &str) -> Self {
        Self {
            size,
            shape,
            dtype: dtype.to_string(),
        }
    }

    /// Total number of elements per sample.
    pub fn sample_numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Full shape including batch dimension.
    pub fn full_shape(&self) -> Vec<usize> {
        let mut shape = vec![self.size];
        shape.extend(&self.shape);
        shape
    }
}

/// Internal storage for a tensor source.
#[allow(dead_code)] // metadata field used only with python feature
struct TensorSource {
    /// Metadata about the tensor
    metadata: TensorMetadata,
    /// The actual PyTorch tensor (when python feature enabled)
    #[cfg(feature = "python")]
    tensor: PyObject,
}

/// Registry for managing tensor sources.
///
/// Tensor sources are named collections of data (e.g., "train", "test")
/// that neural predicates can index into.
pub struct TensorSourceRegistry {
    /// Map from source name to source data
    #[cfg(feature = "python")]
    sources: HashMap<String, TensorSource>,
    #[cfg(not(feature = "python"))]
    sources: HashMap<String, TensorMetadata>,

    /// Currently active source name
    active: Option<String>,

    /// Metadata stored separately for non-python access
    #[cfg(feature = "python")]
    metadata: HashMap<String, TensorMetadata>,
}

impl TensorSourceRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            active: None,
            #[cfg(feature = "python")]
            metadata: HashMap::new(),
        }
    }

    /// Add a tensor source with metadata only (for testing without Python).
    pub fn add_with_metadata(&mut self, name: &str, metadata: TensorMetadata) {
        #[cfg(feature = "python")]
        {
            self.metadata.insert(name.to_string(), metadata);
        }
        #[cfg(not(feature = "python"))]
        {
            self.sources.insert(name.to_string(), metadata);
        }

        // Auto-set first source as active
        if self.active.is_none() {
            self.active = Some(name.to_string());
        }
    }

    /// Add a tensor source with PyTorch tensor.
    #[cfg(feature = "python")]
    pub fn add(&mut self, name: &str, tensor: PyObject, metadata: TensorMetadata) {
        let source = TensorSource {
            metadata: metadata.clone(),
            tensor,
        };
        self.sources.insert(name.to_string(), source);
        self.metadata.insert(name.to_string(), metadata);

        // Auto-set first source as active
        if self.active.is_none() {
            self.active = Some(name.to_string());
        }
    }

    /// Set the active tensor source.
    pub fn set_active(&mut self, name: &str) -> Result<(), TensorSourceError> {
        #[cfg(feature = "python")]
        let exists = self.metadata.contains_key(name);
        #[cfg(not(feature = "python"))]
        let exists = self.sources.contains_key(name);

        if exists {
            self.active = Some(name.to_string());
            Ok(())
        } else {
            Err(TensorSourceError::NotFound(name.to_string()))
        }
    }

    /// Get the name of the active source.
    pub fn active_name(&self) -> Option<&str> {
        self.active.as_deref()
    }

    /// Get the size of the active source.
    pub fn active_size(&self) -> Result<usize, TensorSourceError> {
        match &self.active {
            Some(name) => {
                #[cfg(feature = "python")]
                let meta = self.metadata.get(name);
                #[cfg(not(feature = "python"))]
                let meta = self.sources.get(name);

                meta.map(|m| m.size)
                    .ok_or_else(|| TensorSourceError::NotFound(name.clone()))
            }
            None => Err(TensorSourceError::NoActive),
        }
    }

    /// Get the PyTorch tensor for the active source.
    #[cfg(feature = "python")]
    pub fn get_active(&self) -> Result<&PyObject, TensorSourceError> {
        match &self.active {
            Some(name) => self
                .sources
                .get(name)
                .map(|s| &s.tensor)
                .ok_or_else(|| TensorSourceError::NotFound(name.clone())),
            None => Err(TensorSourceError::NoActive),
        }
    }

    /// Get metadata for a specific source.
    pub fn get_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        #[cfg(feature = "python")]
        {
            self.metadata.get(name)
        }
        #[cfg(not(feature = "python"))]
        {
            self.sources.get(name)
        }
    }

    /// Check if a source exists.
    pub fn contains(&self, name: &str) -> bool {
        #[cfg(feature = "python")]
        {
            self.metadata.contains_key(name)
        }
        #[cfg(not(feature = "python"))]
        {
            self.sources.contains_key(name)
        }
    }

    /// Check if an index is valid for the active source.
    pub fn check_index(&self, index: usize) -> Result<(), TensorSourceError> {
        let size = self.active_size()?;
        if index < size {
            Ok(())
        } else {
            Err(TensorSourceError::IndexOutOfBounds(index, size))
        }
    }

    /// Validate multiple indices at once.
    pub fn validate_indices(&self, indices: &[usize]) -> Result<(), TensorSourceError> {
        let size = self.active_size()?;
        for &idx in indices {
            if idx >= size {
                return Err(TensorSourceError::IndexOutOfBounds(idx, size));
            }
        }
        Ok(())
    }

    /// Get names of all sources.
    pub fn source_names(&self) -> Vec<String> {
        #[cfg(feature = "python")]
        {
            self.metadata.keys().cloned().collect()
        }
        #[cfg(not(feature = "python"))]
        {
            self.sources.keys().cloned().collect()
        }
    }

    /// Number of sources.
    pub fn len(&self) -> usize {
        #[cfg(feature = "python")]
        {
            self.metadata.len()
        }
        #[cfg(not(feature = "python"))]
        {
            self.sources.len()
        }
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove a source.
    pub fn remove(&mut self, name: &str) {
        #[cfg(feature = "python")]
        {
            self.sources.remove(name);
            self.metadata.remove(name);
        }
        #[cfg(not(feature = "python"))]
        {
            self.sources.remove(name);
        }

        // Clear active if removed
        if self.active.as_deref() == Some(name) {
            self.active = None;
        }
    }

    /// Clear all sources.
    pub fn clear(&mut self) {
        self.sources.clear();
        #[cfg(feature = "python")]
        self.metadata.clear();
        self.active = None;
    }

    /// Iterate over source names and metadata.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorMetadata)> {
        #[cfg(feature = "python")]
        {
            self.metadata.iter().map(|(k, v)| (k.as_str(), v))
        }
        #[cfg(not(feature = "python"))]
        {
            self.sources.iter().map(|(k, v)| (k.as_str(), v))
        }
    }
}

impl Default for TensorSourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_sample_numel() {
        let meta = TensorMetadata::new(100, vec![3, 224, 224]);
        assert_eq!(meta.sample_numel(), 3 * 224 * 224);
    }

    #[test]
    fn test_metadata_full_shape() {
        let meta = TensorMetadata::new(1000, vec![1, 28, 28]);
        assert_eq!(meta.full_shape(), vec![1000, 1, 28, 28]);
    }

    #[test]
    fn test_registry_default() {
        let registry = TensorSourceRegistry::default();
        assert!(registry.is_empty());
    }
}
