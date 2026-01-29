// crates/xlog-cuda/tests/set_ops_tests.rs
//! Tests for GPU-native set operations (union, diff)

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

    assert_eq!(result.num_rows(), 0);
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

    assert_eq!(result.num_rows(), 0);
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

    assert_eq!(result.num_rows(), 0);
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

    assert_eq!(result.num_rows(), 0);
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

// ============== U64 Union Tests ==============

#[test]
fn test_union_u64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3];
    let b_vals: Vec<u64> = vec![2, 3, 4];

    let a_buf = provider
        .create_buffer_from_u64_slice(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_u64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a_buf, &b_buf).unwrap();
    assert_eq!(result.num_rows(), 4); // 1, 2, 3, 4

    let result_data = provider.download_column_u64(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2, 3, 4]);
}

#[test]
fn test_union_u64_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    // Input with duplicates
    let a_vals: Vec<u64> = vec![1, 1, 2];
    let b_vals: Vec<u64> = vec![2, 3, 3];

    let a_buf = provider
        .create_buffer_from_u64_slice(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_u64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a_buf, &b_buf).unwrap();
    assert_eq!(result.num_rows(), 3); // 1, 2, 3

    let result_data = provider.download_column_u64(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_union_u64_empty_a() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let b_vals: Vec<u64> = vec![1, 2, 3];

    let a_buf = provider.create_empty_buffer(schema.clone()).unwrap();
    let b_buf = provider
        .create_buffer_from_u64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a_buf, &b_buf).unwrap();
    assert_eq!(result.num_rows(), 3);

    let result_data = provider.download_column_u64(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2, 3]);
}

// ============== U64 Diff Tests ==============

#[test]
fn test_diff_u64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3, 4];
    let b_vals: Vec<u64> = vec![2, 4];

    let a_buf = provider
        .create_buffer_from_u64_slice(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_u64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a_buf, &b_buf).unwrap();
    assert_eq!(result.num_rows(), 2); // 1, 3

    let result_data = provider.download_column_u64(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 3]);
}

#[test]
fn test_diff_u64_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3];
    let b_vals: Vec<u64> = vec![4, 5, 6];

    let a_buf = provider
        .create_buffer_from_u64_slice(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_u64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a_buf, &b_buf).unwrap();
    assert_eq!(result.num_rows(), 3); // All remain

    let result_data = provider.download_column_u64(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2, 3]);
}

#[test]
fn test_diff_u64_complete_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3];
    let b_vals: Vec<u64> = vec![1, 2, 3];

    let a_buf = provider
        .create_buffer_from_u64_slice(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider
        .create_buffer_from_u64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a_buf, &b_buf).unwrap();
    assert_eq!(result.num_rows(), 0); // All removed
}

#[test]
fn test_diff_u64_empty_b() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let a_vals: Vec<u64> = vec![1, 2, 3];

    let a_buf = provider
        .create_buffer_from_u64_slice(&a_vals, schema.clone())
        .unwrap();
    let b_buf = provider.create_empty_buffer(schema).unwrap();

    let result = provider.diff_gpu(&a_buf, &b_buf).unwrap();
    assert_eq!(result.num_rows(), 3); // All remain

    let result_data = provider.download_column_u64(&result, 0).unwrap();
    assert_eq!(result_data, vec![1, 2, 3]);
}

// ============== I64 Union Tests ==============

#[test]
fn test_union_i64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    // Include negative values to test proper ordering
    let a_vals: Vec<i64> = vec![-10, -5, 0, 5];
    let b_vals: Vec<i64> = vec![-5, 0, 10, 20];

    let a = provider
        .create_buffer_from_i64_slice(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_i64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 6); // -10, -5, 0, 5, 10, 20
}

#[test]
fn test_union_i64_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    let a_vals: Vec<i64> = vec![-5, -5, 0];
    let b_vals: Vec<i64> = vec![0, 5, 5];

    let a = provider
        .create_buffer_from_i64_slice(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_i64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 3); // -5, 0, 5
}

// ============== F64 Union Tests ==============

#[test]
fn test_union_f64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let a_vals: Vec<f64> = vec![1.5, 2.5, 3.5];
    let b_vals: Vec<f64> = vec![2.5, 3.5, 4.5];

    let a = provider
        .create_buffer_from_f64_slice(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_f64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 4); // 1.5, 2.5, 3.5, 4.5
}

#[test]
fn test_union_f64_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let a_vals: Vec<f64> = vec![1.5, 1.5, 2.5];
    let b_vals: Vec<f64> = vec![2.5, 3.5, 3.5];

    let a = provider
        .create_buffer_from_f64_slice(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_f64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.union_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 3); // 1.5, 2.5, 3.5
}

// ============== I64 Diff Tests ==============

#[test]
fn test_diff_i64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    let a_vals: Vec<i64> = vec![-10, -5, 0, 5, 10];
    let b_vals: Vec<i64> = vec![-5, 5];

    let a = provider
        .create_buffer_from_i64_slice(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_i64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 3); // -10, 0, 10
}

#[test]
fn test_diff_i64_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    let a_vals: Vec<i64> = vec![-10, -5, 0];
    let b_vals: Vec<i64> = vec![5, 10, 15];

    let a = provider
        .create_buffer_from_i64_slice(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_i64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 3); // All remain: -10, -5, 0
}

// ============== F64 Diff Tests ==============

#[test]
fn test_diff_f64() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let a_vals: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5];
    let b_vals: Vec<f64> = vec![2.5, 4.5];

    let a = provider
        .create_buffer_from_f64_slice(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_f64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 2); // 1.5, 3.5
}

#[test]
fn test_diff_f64_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let a_vals: Vec<f64> = vec![1.5, 2.5, 3.5];
    let b_vals: Vec<f64> = vec![4.5, 5.5, 6.5];

    let a = provider
        .create_buffer_from_f64_slice(&a_vals, schema.clone())
        .unwrap();
    let b = provider
        .create_buffer_from_f64_slice(&b_vals, schema)
        .unwrap();

    let result = provider.diff_gpu(&a, &b).unwrap();
    assert_eq!(result.num_rows(), 3); // All remain: 1.5, 2.5, 3.5
}
