// crates/xlog-cuda/tests/sort_tests.rs
use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

#[test]
fn test_sort_single_column() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Unsorted: [5, 2, 8, 1, 9, 3]
    // Sorted:   [1, 2, 3, 5, 8, 9]
    let keys: Vec<u32> = vec![5, 2, 8, 1, 9, 3];
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&keys, schema.clone())
        .unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result = provider.download_column_u32(&sorted, 0).unwrap();
    assert_eq!(result, vec![1, 2, 3, 5, 8, 9]);
}

#[test]
fn test_sort_preserves_other_columns() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Sort by col0, col1 should follow
    // col0: [3, 1, 2] -> [1, 2, 3]
    // col1: [30, 10, 20] -> [10, 20, 30]
    let col0: Vec<u32> = vec![3, 1, 2];
    let col1: Vec<u32> = vec![30, 10, 20];
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&col0, &col1], schema)
        .unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result0 = provider.download_column_u32(&sorted, 0).unwrap();
    let result1 = provider.download_column_u32(&sorted, 1).unwrap();

    assert_eq!(result0, vec![1, 2, 3]);
    assert_eq!(result1, vec![10, 20, 30]);
}

#[test]
fn test_sort_empty() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
    let buffer = provider.create_empty_buffer(schema.clone()).unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    assert_eq!(sorted.num_rows(), 0);
}

#[test]
fn test_sort_already_sorted() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let keys: Vec<u32> = vec![1, 2, 3, 4, 5];
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&keys, schema)
        .unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result = provider.download_column_u32(&sorted, 0).unwrap();
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_sort_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Input with duplicates: [3, 1, 2, 1, 3, 2]
    // Sorted: [1, 1, 2, 2, 3, 3]
    let keys: Vec<u32> = vec![3, 1, 2, 1, 3, 2];
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&keys, schema)
        .unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result = provider.download_column_u32(&sorted, 0).unwrap();
    assert_eq!(result, vec![1, 1, 2, 2, 3, 3]);
}

#[test]
fn test_sort_respects_device_row_count_after_filter() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
    let keys: Vec<u32> = vec![3, 1, 2];
    let buffer = provider
        .create_buffer_from_u32_slice(&keys, schema)
        .unwrap();

    let mask: Vec<u8> = vec![0, 1, 1];
    let d_mask = provider
        .device()
        .inner()
        .htod_sync_copy(&mask)
        .unwrap();
    let filtered = provider.filter_by_device_mask(&buffer, &d_mask).unwrap();

    let filtered_vals = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(filtered_vals, vec![1, 2]);

    let sorted = provider.sort(&filtered, &[0]).unwrap();
    let sorted_vals = provider.download_column_u32(&sorted, 0).unwrap();
    assert_eq!(sorted_vals, vec![1, 2]);
}

#[test]
fn test_sort_reverse() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Reverse sorted: [5, 4, 3, 2, 1]
    // Sorted: [1, 2, 3, 4, 5]
    let keys: Vec<u32> = vec![5, 4, 3, 2, 1];
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&keys, schema)
        .unwrap();
    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result = provider.download_column_u32(&sorted, 0).unwrap();
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}
