// crates/xlog-cuda/tests/groupby_tests.rs
//! Tests for multi-aggregation groupby operations
use std::sync::Arc;
use xlog_core::{AggOp, MemoryBudget, Schema, ScalarType};
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

#[test]
fn test_groupby_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // key: [1, 1, 2, 2, 2]
    // Expected: group 1: count=2, group 2: count=3
    let keys: Vec<u32> = vec![1, 1, 2, 2, 2];
    let values: Vec<u32> = vec![10, 20, 5, 15, 25]; // values don't matter for count
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    // Perform groupby with count aggregation
    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Count)])
        .unwrap();

    // Result should have 2 rows (2 groups)
    assert_eq!(result.num_rows(), 2);

    // Download and verify results
    let result_keys = provider.download_column_u32(&result, 0).unwrap();
    let result_counts = provider.download_column_u32(&result, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_counts, vec![2, 3]);
}

#[test]
fn test_groupby_sum() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // key: [1, 1, 2, 2, 2]
    // val: [10, 20, 5, 15, 25]
    // Expected: group 1: sum=30, group 2: sum=45
    let keys: Vec<u32> = vec![1, 1, 2, 2, 2];
    let values: Vec<u32> = vec![10, 20, 5, 15, 25];
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    // Perform groupby with sum aggregation
    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum)])
        .unwrap();

    assert_eq!(result.num_rows(), 2);

    let result_keys = provider.download_column_u32(&result, 0).unwrap();
    // Sum results are u64 internally but we truncate to u32 for download
    let result_sums = provider.download_column_u32(&result, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_sums, vec![30, 45]);
}

#[test]
fn test_groupby_min_max() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // key: [1, 1, 2, 2, 2]
    // val: [10, 20, 5, 15, 25]
    // Expected: group 1: min=10, max=20; group 2: min=5, max=25
    let keys: Vec<u32> = vec![1, 1, 2, 2, 2];
    let values: Vec<u32> = vec![10, 20, 5, 15, 25];
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    // Test Min
    let result_min = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Min)])
        .unwrap();

    assert_eq!(result_min.num_rows(), 2);
    let result_keys = provider.download_column_u32(&result_min, 0).unwrap();
    let result_mins = provider.download_column_u32(&result_min, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_mins, vec![10, 5]);

    // Test Max
    let result_max = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Max)])
        .unwrap();

    assert_eq!(result_max.num_rows(), 2);
    let result_keys = provider.download_column_u32(&result_max, 0).unwrap();
    let result_maxs = provider.download_column_u32(&result_max, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_maxs, vec![20, 25]);
}

#[test]
fn test_groupby_multi_agg() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Same data, multiple aggregations at once
    // key: [1, 1, 2, 2, 2]
    // val: [10, 20, 5, 15, 25]
    let keys: Vec<u32> = vec![1, 1, 2, 2, 2];
    let values: Vec<u32> = vec![10, 20, 5, 15, 25];
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    // Multiple aggregations: Sum, Count, Min, Max
    let result = provider
        .groupby_multi_agg(
            &buffer,
            &[0],
            &[
                (1, AggOp::Sum),
                (1, AggOp::Count),
                (1, AggOp::Min),
                (1, AggOp::Max),
            ],
        )
        .unwrap();

    // Result: key, sum, count, min, max = 5 columns
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.arity(), 5);

    let result_keys = provider.download_column_u32(&result, 0).unwrap();
    let result_sums = provider.download_column_u32(&result, 1).unwrap();
    let result_counts = provider.download_column_u32(&result, 2).unwrap();
    let result_mins = provider.download_column_u32(&result, 3).unwrap();
    let result_maxs = provider.download_column_u32(&result, 4).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_sums, vec![30, 45]);
    assert_eq!(result_counts, vec![2, 3]);
    assert_eq!(result_mins, vec![10, 5]);
    assert_eq!(result_maxs, vec![20, 25]);
}

#[test]
fn test_groupby_empty_input() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider.create_empty_buffer(schema).unwrap();

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum)])
        .unwrap();

    assert_eq!(result.num_rows(), 0);
}

#[test]
fn test_groupby_single_group() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // All same key: single group
    let keys: Vec<u32> = vec![5, 5, 5, 5];
    let values: Vec<u32> = vec![10, 20, 30, 40];
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum), (1, AggOp::Count)])
        .unwrap();

    assert_eq!(result.num_rows(), 1);

    let result_keys = provider.download_column_u32(&result, 0).unwrap();
    let result_sums = provider.download_column_u32(&result, 1).unwrap();
    let result_counts = provider.download_column_u32(&result, 2).unwrap();

    assert_eq!(result_keys, vec![5]);
    assert_eq!(result_sums, vec![100]); // 10+20+30+40
    assert_eq!(result_counts, vec![4]);
}

#[test]
fn test_groupby_unsorted_input() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Unsorted input - groupby_multi_agg should sort internally
    // key: [2, 1, 2, 1, 2]
    // val: [5, 10, 15, 20, 25]
    // After sort by key: key=[1,1,2,2,2], val=[10,20,5,15,25]
    // Expected: group 1: sum=30, group 2: sum=45
    let keys: Vec<u32> = vec![2, 1, 2, 1, 2];
    let values: Vec<u32> = vec![5, 10, 15, 20, 25];
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum)])
        .unwrap();

    assert_eq!(result.num_rows(), 2);

    let result_keys = provider.download_column_u32(&result, 0).unwrap();
    let result_sums = provider.download_column_u32(&result, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_sums, vec![30, 45]);
}

#[test]
fn test_groupby_many_groups() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Each element is its own group (all unique keys)
    let keys: Vec<u32> = vec![5, 3, 1, 4, 2];
    let values: Vec<u32> = vec![50, 30, 10, 40, 20];
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::U32),
    ]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum), (1, AggOp::Count)])
        .unwrap();

    assert_eq!(result.num_rows(), 5);

    let result_keys = provider.download_column_u32(&result, 0).unwrap();
    let result_sums = provider.download_column_u32(&result, 1).unwrap();
    let result_counts = provider.download_column_u32(&result, 2).unwrap();

    // Keys should be sorted: 1, 2, 3, 4, 5
    assert_eq!(result_keys, vec![1, 2, 3, 4, 5]);
    // Each group has sum = its own value
    assert_eq!(result_sums, vec![10, 20, 30, 40, 50]);
    // Each group has count = 1
    assert_eq!(result_counts, vec![1, 1, 1, 1, 1]);
}
