//! Tests for tensor source registry
//!
//! The tensor source registry manages external data (images, embeddings, etc.)
//! that can be indexed by neural predicates.

#![cfg(feature = "python")]

use xlog_neural::tensor_source::{TensorMetadata, TensorSourceError, TensorSourceRegistry};

#[test]
fn test_add_and_set_active_source() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("train", TensorMetadata::new(1000, vec![1, 28, 28]));
    registry.add_with_metadata("test", TensorMetadata::new(200, vec![1, 28, 28]));

    registry.set_active("train").unwrap();
    assert_eq!(registry.active_name(), Some("train"));

    registry.set_active("test").unwrap();
    assert_eq!(registry.active_name(), Some("test"));
}

#[test]
fn test_set_active_invalid_source() {
    let mut registry = TensorSourceRegistry::new();

    let result = registry.set_active("nonexistent");
    assert!(result.is_err());

    match result {
        Err(TensorSourceError::NotFound(name)) => {
            assert_eq!(name, "nonexistent");
        }
        _ => panic!("Expected NotFound error"),
    }
}

#[test]
fn test_first_source_is_auto_active() {
    let mut registry = TensorSourceRegistry::new();

    assert!(registry.active_name().is_none());

    registry.add_with_metadata("first", TensorMetadata::new(100, vec![10]));

    // First added source becomes active automatically
    assert_eq!(registry.active_name(), Some("first"));
}

#[test]
fn test_source_size() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("data", TensorMetadata::new(500, vec![1, 28, 28]));
    registry.set_active("data").unwrap();

    assert_eq!(registry.active_size().unwrap(), 500);
}

#[test]
fn test_source_shape() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("images", TensorMetadata::new(100, vec![3, 224, 224]));

    let meta = registry.get_metadata("images").unwrap();
    assert_eq!(meta.shape, vec![3, 224, 224]);
}

#[test]
fn test_multiple_sources() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("train", TensorMetadata::new(60000, vec![1, 28, 28]));
    registry.add_with_metadata("val", TensorMetadata::new(10000, vec![1, 28, 28]));
    registry.add_with_metadata("test", TensorMetadata::new(10000, vec![1, 28, 28]));

    let names = registry.source_names();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"train".to_string()));
    assert!(names.contains(&"val".to_string()));
    assert!(names.contains(&"test".to_string()));
}

#[test]
fn test_remove_source() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("temp", TensorMetadata::new(10, vec![5]));
    assert!(registry.contains("temp"));

    registry.remove("temp");
    assert!(!registry.contains("temp"));
}

#[test]
fn test_clear_sources() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("a", TensorMetadata::new(10, vec![1]));
    registry.add_with_metadata("b", TensorMetadata::new(20, vec![2]));

    assert_eq!(registry.len(), 2);

    registry.clear();

    assert_eq!(registry.len(), 0);
    assert!(registry.is_empty());
    assert!(registry.active_name().is_none());
}

#[test]
fn test_no_active_error() {
    let registry = TensorSourceRegistry::new();

    let result = registry.active_size();
    assert!(result.is_err());

    match result {
        Err(TensorSourceError::NoActive) => {}
        _ => panic!("Expected NoActive error"),
    }
}

#[test]
fn test_index_bounds_check() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("small", TensorMetadata::new(10, vec![5]));
    registry.set_active("small").unwrap();

    // Index 9 is valid (0-9 for size 10)
    assert!(registry.check_index(9).is_ok());

    // Index 10 is out of bounds
    let result = registry.check_index(10);
    assert!(result.is_err());

    match result {
        Err(TensorSourceError::IndexOutOfBounds(idx, size)) => {
            assert_eq!(idx, 10);
            assert_eq!(size, 10);
        }
        _ => panic!("Expected IndexOutOfBounds error"),
    }
}

#[test]
fn test_tensor_metadata_creation() {
    let meta = TensorMetadata::new(1000, vec![3, 224, 224]);

    assert_eq!(meta.size, 1000);
    assert_eq!(meta.shape, vec![3, 224, 224]);
    assert_eq!(meta.dtype, "float32"); // default
}

#[test]
fn test_tensor_metadata_with_dtype() {
    let meta = TensorMetadata::with_dtype(100, vec![10], "float64");

    assert_eq!(meta.dtype, "float64");
}

#[test]
fn test_validate_indices() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("data", TensorMetadata::new(100, vec![10]));
    registry.set_active("data").unwrap();

    // All valid
    assert!(registry.validate_indices(&[0, 50, 99]).is_ok());

    // One invalid
    let result = registry.validate_indices(&[0, 100, 50]);
    assert!(result.is_err());
}

#[test]
fn test_source_iteration() {
    let mut registry = TensorSourceRegistry::new();

    registry.add_with_metadata("x", TensorMetadata::new(10, vec![1]));
    registry.add_with_metadata("y", TensorMetadata::new(20, vec![2]));

    let entries: Vec<_> = registry.iter().collect();
    assert_eq!(entries.len(), 2);
}
