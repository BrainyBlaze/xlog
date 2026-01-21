//! Tests for the neural network registry
//!
//! The network registry manages PyTorch modules and their configurations
//! for integration with probabilistic logic programs.

use xlog_neural::{NetworkConfig, NetworkRegistry};

#[test]
fn test_registry_register_and_get() {
    let mut registry = NetworkRegistry::new();

    let config = NetworkConfig {
        name: "mnist_net".to_string(),
        batching: true,
        k: None,
        det: false,
        cache_enabled: true,
        cache_size: 10000,
    };

    registry.register(config);

    assert!(registry.get("mnist_net").is_some());
    assert!(registry.get("unknown").is_none());
}

#[test]
fn test_registry_set_train_mode() {
    let mut registry = NetworkRegistry::new();
    registry.register(NetworkConfig::default("test_net"));

    registry.set_train_mode(true);
    assert!(registry.get("test_net").unwrap().train_mode);

    registry.set_train_mode(false);
    assert!(!registry.get("test_net").unwrap().train_mode);
}

#[test]
fn test_registry_multiple_networks() {
    let mut registry = NetworkRegistry::new();

    registry.register(NetworkConfig::default("net1"));
    registry.register(NetworkConfig::default("net2"));
    registry.register(NetworkConfig::default("net3"));

    assert_eq!(registry.len(), 3);
    assert!(registry.get("net1").is_some());
    assert!(registry.get("net2").is_some());
    assert!(registry.get("net3").is_some());
}

#[test]
fn test_registry_names() {
    let mut registry = NetworkRegistry::new();

    registry.register(NetworkConfig::default("alpha"));
    registry.register(NetworkConfig::default("beta"));

    let names = registry.names();
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"beta"));
}

#[test]
fn test_network_config_with_k() {
    let mut registry = NetworkRegistry::new();

    let config = NetworkConfig {
        name: "top_k_net".to_string(),
        batching: true,
        k: Some(5),
        det: false,
        cache_enabled: true,
        cache_size: 1000,
    };

    registry.register(config);

    let handle = registry.get("top_k_net").unwrap();
    assert_eq!(handle.k, Some(5));
}

#[test]
fn test_network_config_deterministic() {
    let mut registry = NetworkRegistry::new();

    let config = NetworkConfig {
        name: "det_net".to_string(),
        batching: false,
        k: None,
        det: true,
        cache_enabled: false,
        cache_size: 0,
    };

    registry.register(config);

    let handle = registry.get("det_net").unwrap();
    assert!(handle.det);
    assert!(!handle.batching);
    assert!(!handle.cache_enabled);
}

#[test]
fn test_registry_get_mut() {
    let mut registry = NetworkRegistry::new();
    registry.register(NetworkConfig::default("mutable_net"));

    // Modify through get_mut
    if let Some(handle) = registry.get_mut("mutable_net") {
        handle.train_mode = true;
        handle.k = Some(10);
    }

    let handle = registry.get("mutable_net").unwrap();
    assert!(handle.train_mode);
    assert_eq!(handle.k, Some(10));
}

#[test]
fn test_registry_unregister() {
    let mut registry = NetworkRegistry::new();
    registry.register(NetworkConfig::default("temp_net"));

    assert!(registry.get("temp_net").is_some());

    registry.unregister("temp_net");

    assert!(registry.get("temp_net").is_none());
}

#[test]
fn test_registry_clear() {
    let mut registry = NetworkRegistry::new();
    registry.register(NetworkConfig::default("net1"));
    registry.register(NetworkConfig::default("net2"));

    assert_eq!(registry.len(), 2);

    registry.clear();

    assert_eq!(registry.len(), 0);
    assert!(registry.is_empty());
}

#[test]
fn test_registry_contains() {
    let mut registry = NetworkRegistry::new();
    registry.register(NetworkConfig::default("existing"));

    assert!(registry.contains("existing"));
    assert!(!registry.contains("nonexistent"));
}
