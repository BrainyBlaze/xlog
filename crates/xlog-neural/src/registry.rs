//! Network registry for managing registered neural networks.
//!
//! The registry is the central point for managing all neural networks used
//! in a probabilistic logic program. It handles:
//!
//! - Registration of networks with their configurations
//! - Train/eval mode switching for all networks
//! - Network lookup by name

use crate::handle::NetworkHandle;
use std::collections::HashMap;

/// Configuration for registering a neural network.
///
/// This mirrors the DeepProbLog `register_network` options.
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Unique name identifying this network (must match nn() declarations)
    pub name: String,

    /// Whether to batch inputs for efficient GPU processing.
    /// When true, multiple queries are grouped into a single forward pass.
    pub batching: bool,

    /// Top-k sampling: if Some(k), only consider the top k outputs.
    /// Useful for large output spaces where most classes have near-zero probability.
    pub k: Option<usize>,

    /// Deterministic mode: use argmax instead of probabilistic sampling.
    /// Useful for debugging and when you want reproducible results.
    pub det: bool,

    /// Whether to cache network outputs.
    /// Caching avoids redundant forward passes for repeated inputs.
    pub cache_enabled: bool,

    /// Maximum number of entries in the output cache.
    pub cache_size: usize,
}

impl NetworkConfig {
    /// Create a default configuration for a network with the given name.
    ///
    /// Default settings:
    /// - batching: true
    /// - k: None (consider all outputs)
    /// - det: false (probabilistic mode)
    /// - cache_enabled: true
    /// - cache_size: 10000
    pub fn default(name: &str) -> Self {
        Self {
            name: name.to_string(),
            batching: true,
            k: None,
            det: false,
            cache_enabled: true,
            cache_size: 10000,
        }
    }

    /// Create a configuration for a deterministic network.
    pub fn deterministic(name: &str) -> Self {
        Self {
            name: name.to_string(),
            batching: true,
            k: None,
            det: true,
            cache_enabled: true,
            cache_size: 10000,
        }
    }

    /// Create a configuration with top-k sampling.
    pub fn with_top_k(name: &str, k: usize) -> Self {
        Self {
            name: name.to_string(),
            batching: true,
            k: Some(k),
            det: false,
            cache_enabled: true,
            cache_size: 10000,
        }
    }

    /// Builder method to set batching.
    pub fn batching(mut self, enabled: bool) -> Self {
        self.batching = enabled;
        self
    }

    /// Builder method to set top-k.
    pub fn k(mut self, k: Option<usize>) -> Self {
        self.k = k;
        self
    }

    /// Builder method to set deterministic mode.
    pub fn det(mut self, det: bool) -> Self {
        self.det = det;
        self
    }

    /// Builder method to set cache.
    pub fn cache(mut self, enabled: bool, size: usize) -> Self {
        self.cache_enabled = enabled;
        self.cache_size = size;
        self
    }
}

/// Registry for managing neural networks.
///
/// The registry maintains a collection of `NetworkHandle` instances,
/// each identified by a unique name. Networks are registered with
/// configurations and then have their PyTorch modules attached via
/// the Python API.
pub struct NetworkRegistry {
    /// Map from network name to handle
    networks: HashMap<String, NetworkHandle>,
}

impl NetworkRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            networks: HashMap::new(),
        }
    }

    /// Register a network with the given configuration.
    ///
    /// If a network with the same name already exists, it will be replaced.
    pub fn register(&mut self, config: NetworkConfig) {
        let handle = NetworkHandle::from_config(&config);
        self.networks.insert(config.name, handle);
    }

    /// Get a reference to a network handle by name.
    pub fn get(&self, name: &str) -> Option<&NetworkHandle> {
        self.networks.get(name)
    }

    /// Get a mutable reference to a network handle by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut NetworkHandle> {
        self.networks.get_mut(name)
    }

    /// Check if a network is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.networks.contains_key(name)
    }

    /// Remove a network from the registry.
    pub fn unregister(&mut self, name: &str) -> Option<NetworkHandle> {
        self.networks.remove(name)
    }

    /// Set train mode for all registered networks.
    ///
    /// This affects both the `train_mode` flag on handles and should
    /// be used to call `.train()` or `.eval()` on PyTorch modules.
    pub fn set_train_mode(&mut self, train: bool) {
        for handle in self.networks.values_mut() {
            handle.train_mode = train;
        }
    }

    /// Get the names of all registered networks.
    pub fn names(&self) -> Vec<&str> {
        self.networks.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered networks.
    pub fn len(&self) -> usize {
        self.networks.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.networks.is_empty()
    }

    /// Remove all networks from the registry.
    pub fn clear(&mut self) {
        self.networks.clear();
    }

    /// Iterate over all network handles.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &NetworkHandle)> {
        self.networks.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Iterate mutably over all network handles.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut NetworkHandle)> {
        self.networks.iter_mut().map(|(k, v)| (k.as_str(), v))
    }
}

impl Default for NetworkRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = NetworkConfig::default("test");
        assert_eq!(config.name, "test");
        assert!(config.batching);
        assert!(config.k.is_none());
        assert!(!config.det);
        assert!(config.cache_enabled);
        assert_eq!(config.cache_size, 10000);
    }

    #[test]
    fn test_config_deterministic() {
        let config = NetworkConfig::deterministic("det_test");
        assert!(config.det);
    }

    #[test]
    fn test_config_with_top_k() {
        let config = NetworkConfig::with_top_k("top_k_test", 5);
        assert_eq!(config.k, Some(5));
    }

    #[test]
    fn test_config_builder() {
        let config = NetworkConfig::default("builder_test")
            .batching(false)
            .k(Some(3))
            .det(true)
            .cache(false, 0);

        assert!(!config.batching);
        assert_eq!(config.k, Some(3));
        assert!(config.det);
        assert!(!config.cache_enabled);
        assert_eq!(config.cache_size, 0);
    }

    #[test]
    fn test_registry_new() {
        let registry = NetworkRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register_get() {
        let mut registry = NetworkRegistry::new();
        registry.register(NetworkConfig::default("net1"));

        assert!(registry.contains("net1"));
        assert!(registry.get("net1").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_iter() {
        let mut registry = NetworkRegistry::new();
        registry.register(NetworkConfig::default("a"));
        registry.register(NetworkConfig::default("b"));

        let names: Vec<&str> = registry.iter().map(|(name, _)| name).collect();
        assert_eq!(names.len(), 2);
    }
}
