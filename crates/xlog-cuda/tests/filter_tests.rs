// crates/xlog-cuda/tests/filter_tests.rs
mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::{CompareOp, CudaKernelProvider};

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
    let filtered = provider.filter::<u32>(&buffer, 0, 2, CompareOp::Eq).unwrap();

    let result0 = provider.download_column::<u32>(&filtered, 0).unwrap();
    let result1 = provider.download_column::<u32>(&filtered, 1).unwrap();

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
        .create_buffer_from_slice::<u32>(&col0, schema)
        .unwrap();
    let filtered = provider.filter::<u32>(&buffer, 0, 2, CompareOp::Gt).unwrap();

    let result = provider.download_column::<u32>(&filtered, 0).unwrap();
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
        .create_buffer_from_slice::<u32>(&col0, schema)
        .unwrap();
    let filtered = provider.filter::<u32>(&buffer, 0, 99, CompareOp::Eq).unwrap();

    let result = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert!(result.is_empty());
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
        .create_buffer_from_slice::<u32>(&col0, schema)
        .unwrap();
    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();

    let result = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(result, vec![10, 30, 50]);
}

#[test]
fn test_filter_u32_large() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // More than 256 rows now works with multi-block prefix sum
    let col0: Vec<u32> = (0..300).collect();
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_slice::<u32>(&col0, schema)
        .unwrap();
    let result = provider.filter::<u32>(&buffer, 0, 100, CompareOp::Eq);

    assert!(result.is_ok(), "Filter should work with > 256 rows");
    let filtered = result.unwrap();
    let values = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(values, vec![100u32]);
}

#[test]
fn test_filter_u32_all_ops() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let col0: Vec<u32> = vec![1, 2, 3, 4, 5];
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);
    let buffer = provider
        .create_buffer_from_slice::<u32>(&col0, schema)
        .unwrap();

    // Test Lt (less than 3): [1, 2]
    let filtered = provider.filter::<u32>(&buffer, 0, 3, CompareOp::Lt).unwrap();
    let result = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(result, vec![1, 2]);

    // Test Le (less than or equal 3): [1, 2, 3]
    let filtered = provider.filter::<u32>(&buffer, 0, 3, CompareOp::Le).unwrap();
    let result = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(result, vec![1, 2, 3]);

    // Test Ge (greater than or equal 3): [3, 4, 5]
    let filtered = provider.filter::<u32>(&buffer, 0, 3, CompareOp::Ge).unwrap();
    let result = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(result, vec![3, 4, 5]);

    // Test Ne (not equal 3): [1, 2, 4, 5]
    let filtered = provider.filter::<u32>(&buffer, 0, 3, CompareOp::Ne).unwrap();
    let result = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(result, vec![1, 2, 4, 5]);
}

// ============== F64 Filter Tests ==============

/// Helper to create an F64 test buffer from a slice of f64 values
fn create_f64_buffer(
    provider: &CudaKernelProvider,
    values: &[f64],
    name: &str,
) -> xlog_cuda::CudaBuffer {
    let schema = Schema::new(vec![(name.to_string(), ScalarType::F64)]);
    provider
        .create_buffer_from_slice::<f64>(values, schema)
        .unwrap()
}

/// Helper to download F64 column from GPU
fn download_f64_column(
    provider: &CudaKernelProvider,
    buffer: &xlog_cuda::CudaBuffer,
    col_idx: usize,
) -> Vec<f64> {
    provider.download_column::<f64>(buffer, col_idx).unwrap()
}

#[test]
fn test_filter_f64_eq() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Filter col0 == 2.5
    // col0: [1.0, 2.5, 3.0, 2.5, 4.0] -> [2.5, 2.5]
    let values: Vec<f64> = vec![1.0, 2.5, 3.0, 2.5, 4.0];
    let buffer = create_f64_buffer(&provider, &values, "col0");

    let filtered = provider.filter::<f64>(&buffer, 0, 2.5, CompareOp::Eq).unwrap();
    let result = download_f64_column(&provider, &filtered, 0);

    assert_eq!(result, vec![2.5, 2.5]);
}

#[test]
fn test_filter_f64_gt() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Filter col0 > 2.0
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let buffer = create_f64_buffer(&provider, &values, "col0");

    let filtered = provider.filter::<f64>(&buffer, 0, 2.0, CompareOp::Gt).unwrap();
    let result = download_f64_column(&provider, &filtered, 0);

    assert_eq!(result, vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_filter_i32_u64_f32_bool_and_column_column_compare() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema_i32 = Schema::new(vec![("v".to_string(), ScalarType::I32)]);
    let buf_i32 = provider
        .create_buffer_from_slice::<i32>(&[-2, 0, 3], schema_i32)
        .unwrap();
    let filtered_i32 = provider.filter::<i32>(&buf_i32, 0, -1, CompareOp::Gt).unwrap();
    let vals_i32 = provider.download_column::<i32>(&filtered_i32, 0).unwrap();
    assert_eq!(vals_i32, vec![0, 3]);

    let schema_u64 = Schema::new(vec![("v".to_string(), ScalarType::U64)]);
    let buf_u64 = provider
        .create_buffer_from_slice::<u64>(&[1, 5, 9], schema_u64)
        .unwrap();
    let filtered_u64 = provider.filter::<u64>(&buf_u64, 0, 5, CompareOp::Ge).unwrap();
    let vals_u64 = provider.download_column::<u64>(&filtered_u64, 0).unwrap();
    assert_eq!(vals_u64, vec![5, 9]);

    let schema_f32 = Schema::new(vec![("v".to_string(), ScalarType::F32)]);
    let buf_f32 = provider
        .create_buffer_from_slice::<f32>(&[1.0, -1.5, 2.5], schema_f32)
        .unwrap();
    let filtered_f32 = provider
        .filter::<f32>(&buf_f32, 0, 0.0, CompareOp::Gt)
        .unwrap();
    let vals_f32 = provider.download_column::<f32>(&filtered_f32, 0).unwrap();
    assert_eq!(vals_f32.len(), 2);

    let schema_bool = Schema::new(vec![("v".to_string(), ScalarType::Bool)]);
    let buf_bool = provider
        .create_buffer_from_slice::<u8>(&[0, 1, 1, 0], schema_bool)
        .unwrap();
    let filtered_bool = provider
        .filter::<bool>(&buf_bool, 0, true, CompareOp::Eq)
        .unwrap();
    let vals_bool = provider.download_column::<u8>(&filtered_bool, 0).unwrap();
    assert_eq!(vals_bool, vec![1, 1]);

    let schema_u32 = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
    ]);
    let buf_u32 = provider
        .create_buffer_from_u32_columns(&[&[1, 2, 3], &[1, 9, 3]], schema_u32)
        .unwrap();
    let mask = provider
        .compare_columns::<u32>(&buf_u32, 0, 1, CompareOp::Eq)
        .unwrap();
    let filtered = provider.filter_by_device_mask(&buf_u32, &mask).unwrap();
    let vals = provider.download_column::<u32>(&filtered, 0).unwrap();
    assert_eq!(vals, vec![1, 3]);
}

#[test]
fn test_filter_f64_lt() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Filter col0 < 3.0
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let buffer = create_f64_buffer(&provider, &values, "col0");

    let filtered = provider.filter::<f64>(&buffer, 0, 3.0, CompareOp::Lt).unwrap();
    let result = download_f64_column(&provider, &filtered, 0);

    assert_eq!(result, vec![1.0, 2.0]);
}

#[test]
fn test_filter_f64_all_ops() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let buffer = create_f64_buffer(&provider, &values, "col0");

    // Test Lt (less than 3.0): [1.0, 2.0]
    let filtered = provider.filter::<f64>(&buffer, 0, 3.0, CompareOp::Lt).unwrap();
    let result = download_f64_column(&provider, &filtered, 0);
    assert_eq!(result, vec![1.0, 2.0]);

    // Test Le (less than or equal 3.0): [1.0, 2.0, 3.0]
    let filtered = provider.filter::<f64>(&buffer, 0, 3.0, CompareOp::Le).unwrap();
    let result = download_f64_column(&provider, &filtered, 0);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);

    // Test Ge (greater than or equal 3.0): [3.0, 4.0, 5.0]
    let filtered = provider.filter::<f64>(&buffer, 0, 3.0, CompareOp::Ge).unwrap();
    let result = download_f64_column(&provider, &filtered, 0);
    assert_eq!(result, vec![3.0, 4.0, 5.0]);

    // Test Ne (not equal 3.0): [1.0, 2.0, 4.0, 5.0]
    let filtered = provider.filter::<f64>(&buffer, 0, 3.0, CompareOp::Ne).unwrap();
    let result = download_f64_column(&provider, &filtered, 0);
    assert_eq!(result, vec![1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn test_filter_f64_no_matches() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let values: Vec<f64> = vec![1.0, 2.0, 3.0];
    let buffer = create_f64_buffer(&provider, &values, "col0");

    let filtered = provider.filter::<f64>(&buffer, 0, 99.0, CompareOp::Eq).unwrap();
    let result = provider.download_column::<f64>(&filtered, 0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_filter_f64_type_validation() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create a U32 buffer and try to filter with F64 - should fail
    let col0: Vec<u32> = vec![1, 2, 3];
    let schema = Schema::new(vec![("col0".to_string(), ScalarType::U32)]);
    let buffer = provider
        .create_buffer_from_slice::<u32>(&col0, schema)
        .unwrap();

    let result = provider.filter::<f64>(&buffer, 0, 2.0, CompareOp::Eq);
    assert!(
        result.is_err(),
        "Should fail when filtering U32 column with F64 filter"
    );
}
