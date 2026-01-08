// crates/xlog-cuda/tests/filter_tests.rs
use std::sync::Arc;
use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CompareOp, CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).unwrap());
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

#[test]
fn test_filter_u32_eq() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Filter col0 == 2
    // col0: [1, 2, 3, 2, 4] -> [2, 2]
    // col1: [10, 20, 30, 40, 50] -> [20, 40]
    let col0: Vec<u32> = vec![1, 2, 3, 2, 4];
    let col1: Vec<u32> = vec![10, 20, 30, 40, 50];
    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U32),
        ("col1".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&col0, &col1], schema)
        .unwrap();
    let filtered = provider.filter_u32_eq(&buffer, 0, 2).unwrap();

    let result0 = provider.download_column_u32(&filtered, 0).unwrap();
    let result1 = provider.download_column_u32(&filtered, 1).unwrap();

    assert_eq!(result0, vec![2, 2]);
    assert_eq!(result1, vec![20, 40]);
}

#[test]
fn test_filter_u32_gt() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Filter col0 > 2
    let col0: Vec<u32> = vec![1, 2, 3, 4, 5];
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&col0, schema)
        .unwrap();
    let filtered = provider.filter_u32_gt(&buffer, 0, 2).unwrap();

    let result = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(result, vec![3, 4, 5]);
}

#[test]
fn test_filter_no_matches() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let col0: Vec<u32> = vec![1, 2, 3];
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&col0, schema)
        .unwrap();
    let filtered = provider.filter_u32_eq(&buffer, 0, 99).unwrap();

    assert_eq!(filtered.num_rows, 0);
}

#[test]
fn test_filter_by_mask() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let col0: Vec<u32> = vec![10, 20, 30, 40, 50];
    let mask: Vec<u8> = vec![1, 0, 1, 0, 1];
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&col0, schema)
        .unwrap();
    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();

    let result = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(result, vec![10, 30, 50]);
}

#[test]
fn test_filter_u32_over_limit() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // More than 256 rows should error
    let col0: Vec<u32> = (0..300).collect();
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);

    let buffer = provider.create_buffer_from_u32_slice(&col0, schema).unwrap();
    let result = provider.filter_u32_eq(&buffer, 0, 100);

    assert!(result.is_err(), "Should error for > 256 rows");
}

#[test]
fn test_filter_u32_all_ops() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let col0: Vec<u32> = vec![1, 2, 3, 4, 5];
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);
    let buffer = provider.create_buffer_from_u32_slice(&col0, schema).unwrap();

    // Test Lt (less than 3): [1, 2]
    let filtered = provider.filter_u32(&buffer, 0, 3, CompareOp::Lt).unwrap();
    let result = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(result, vec![1, 2]);

    // Test Le (less than or equal 3): [1, 2, 3]
    let filtered = provider.filter_u32(&buffer, 0, 3, CompareOp::Le).unwrap();
    let result = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(result, vec![1, 2, 3]);

    // Test Ge (greater than or equal 3): [3, 4, 5]
    let filtered = provider.filter_u32(&buffer, 0, 3, CompareOp::Ge).unwrap();
    let result = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(result, vec![3, 4, 5]);

    // Test Ne (not equal 3): [1, 2, 4, 5]
    let filtered = provider.filter_u32(&buffer, 0, 3, CompareOp::Ne).unwrap();
    let result = provider.download_column_u32(&filtered, 0).unwrap();
    assert_eq!(result, vec![1, 2, 4, 5]);
}
