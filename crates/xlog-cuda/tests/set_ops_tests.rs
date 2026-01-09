// crates/xlog-cuda/tests/set_ops_tests.rs
//! Tests for GPU-native set operations (union, diff)

use std::sync::Arc;
use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).unwrap());
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

// ============== Union Tests ==============

#[test]
fn test_union_gpu_basic() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = [3, 4, 5]
    // Union should deduplicate: [1, 2, 3, 4, 5]
    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![3, 4, 5];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_union_gpu_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2], b = [3, 4]
    // Union: [1, 2, 3, 4]
    let a: Vec<u32> = vec![1, 2];
    let b: Vec<u32> = vec![3, 4];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3, 4]);
}

#[test]
fn test_union_gpu_complete_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = [1, 2, 3]
    // Union: [1, 2, 3]
    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![1, 2, 3];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_union_gpu_empty_a() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [], b = [1, 2]
    // Union: [1, 2]
    let b: Vec<u32> = vec![1, 2];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider.create_empty_buffer(schema.clone()).unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2]);
}

#[test]
fn test_union_gpu_empty_b() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2], b = []
    // Union: [1, 2]
    let a: Vec<u32> = vec![1, 2];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider.create_empty_buffer(schema.clone()).unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2]);
}

#[test]
fn test_union_gpu_both_empty() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [], b = []
    // Union: []
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider.create_empty_buffer(schema.clone()).unwrap();
    let buf_b = provider.create_empty_buffer(schema.clone()).unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();

    assert_eq!(result.num_rows, 0);
}

#[test]
fn test_union_gpu_with_duplicates_in_input() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 1, 2], b = [2, 3, 3]
    // Union (deduplicated): [1, 2, 3]
    let a: Vec<u32> = vec![1, 1, 2];
    let b: Vec<u32> = vec![2, 3, 3];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3]);
}

// ============== Diff Tests ==============

#[test]
fn test_diff_gpu_basic() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3, 4], b = [2, 4]
    // Diff: a - b = [1, 3]
    let a: Vec<u32> = vec![1, 2, 3, 4];
    let b: Vec<u32> = vec![2, 4];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 3]);
}

#[test]
fn test_diff_gpu_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = [4, 5, 6]
    // Diff: a - b = [1, 2, 3] (no overlap)
    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![4, 5, 6];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_diff_gpu_complete_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = [1, 2, 3]
    // Diff: a - b = [] (all removed)
    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![1, 2, 3];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();

    assert_eq!(result.num_rows, 0);
}

#[test]
fn test_diff_gpu_empty_a() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [], b = [1, 2]
    // Diff: [] - [1,2] = []
    let b: Vec<u32> = vec![1, 2];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider.create_empty_buffer(schema.clone()).unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();

    assert_eq!(result.num_rows, 0);
}

#[test]
fn test_diff_gpu_empty_b() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [1, 2, 3], b = []
    // Diff: [1,2,3] - [] = [1, 2, 3]
    let a: Vec<u32> = vec![1, 2, 3];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider.create_empty_buffer(schema.clone()).unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_diff_gpu_b_superset() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a = [2, 3], b = [1, 2, 3, 4, 5]
    // Diff: [2,3] - [1,2,3,4,5] = []
    let a: Vec<u32> = vec![2, 3];
    let b: Vec<u32> = vec![1, 2, 3, 4, 5];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();

    assert_eq!(result.num_rows, 0);
}

#[test]
fn test_diff_gpu_unsorted_inputs() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Unsorted inputs should still work (sorted internally)
    // a = [4, 1, 3, 2], b = [2, 4]
    // Diff: {1,2,3,4} - {2,4} = [1, 3]
    let a: Vec<u32> = vec![4, 1, 3, 2];
    let b: Vec<u32> = vec![2, 4];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    // Result should be sorted: [1, 3]
    assert_eq!(result_data, vec![1, 3]);
}

#[test]
fn test_diff_gpu_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // a has duplicates: [1, 1, 2, 2, 3], b = [2]
    // After dedup and diff: {1, 2, 3} - {2} = [1, 3]
    let a: Vec<u32> = vec![1, 1, 2, 2, 3];
    let b: Vec<u32> = vec![2];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    assert_eq!(result_data, vec![1, 3]);
}
