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
    // Sum results are u64 (to prevent overflow)
    let result_sums = provider.download_column_u64(&result, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_sums, vec![30u64, 45u64]);
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
    // Sum results are u64 (to prevent overflow)
    let result_sums = provider.download_column_u64(&result, 1).unwrap();
    let result_counts = provider.download_column_u32(&result, 2).unwrap();
    let result_mins = provider.download_column_u32(&result, 3).unwrap();
    let result_maxs = provider.download_column_u32(&result, 4).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_sums, vec![30u64, 45u64]);
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
    // Sum results are u64 (to prevent overflow)
    let result_sums = provider.download_column_u64(&result, 1).unwrap();
    let result_counts = provider.download_column_u32(&result, 2).unwrap();

    assert_eq!(result_keys, vec![5]);
    assert_eq!(result_sums, vec![100u64]); // 10+20+30+40
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
    // Sum results are u64 (to prevent overflow)
    let result_sums = provider.download_column_u64(&result, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_sums, vec![30u64, 45u64]);
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
    // Sum results are u64 (to prevent overflow)
    let result_sums = provider.download_column_u64(&result, 1).unwrap();
    let result_counts = provider.download_column_u32(&result, 2).unwrap();

    // Keys should be sorted: 1, 2, 3, 4, 5
    assert_eq!(result_keys, vec![1, 2, 3, 4, 5]);
    // Each group has sum = its own value
    assert_eq!(result_sums, vec![10u64, 20u64, 30u64, 40u64, 50u64]);
    // Each group has count = 1
    assert_eq!(result_counts, vec![1, 1, 1, 1, 1]);
}

// ============== LogSumExp Edge Case Tests ==============
//
// LogSumExp special value behavior:
// - NaN inputs: If any input is NaN, the result is NaN
// - +INFINITY: If max value is +INFINITY, result is +INFINITY
// - -INFINITY: Treated as a very small value, effectively ignored in the sum
// - Single element: logsumexp(x) = x (since log(exp(x)) = x)

/// Helper to create a buffer with u32 keys and f64 values for logsumexp tests
fn create_keyed_f64_buffer(
    provider: &CudaKernelProvider,
    keys: &[u32],
    values: &[f64],
) -> xlog_cuda::CudaBuffer {
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::F64),
    ]);

    // Create key column
    let key_bytes: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut key_col = provider
        .memory()
        .alloc::<u8>(key_bytes.len())
        .expect("alloc key");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&key_bytes, &mut key_col)
        .expect("upload key");

    // Create value column
    let val_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut val_col = provider
        .memory()
        .alloc::<u8>(val_bytes.len())
        .expect("alloc val");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&val_bytes, &mut val_col)
        .expect("upload val");

    xlog_cuda::CudaBuffer::from_columns(vec![key_col.into(), val_col.into()], keys.len() as u64, schema)
}

#[test]
fn test_groupby_logsumexp_single_element_groups() {
    // Test that logsumexp(x) = x for single-element groups
    // This verifies the identity property: log(exp(x)) = x
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Each key appears exactly once
    let keys: Vec<u32> = vec![1, 2, 3, 4, 5];
    let values: Vec<f64> = vec![0.0, 1.0, -1.0, 10.0, -10.0];

    let buffer = create_keyed_f64_buffer(&provider, &keys, &values);

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::LogSumExp)])
        .expect("groupby_multi_agg should succeed");

    assert_eq!(result.num_rows(), 5, "Should have 5 single-element groups");

    let result_values = provider.download_column_f64(&result, 1).expect("download result");

    // For single element groups, logsumexp(x) = x
    let tolerance = 1e-10;
    for (i, (&expected, &actual)) in values.iter().zip(result_values.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < tolerance,
            "Group {} (key={}): logsumexp({}) should equal {}, got {}",
            i,
            keys[i],
            expected,
            expected,
            actual
        );
    }
}

#[test]
fn test_groupby_logsumexp_mixed_positive_negative() {
    // Test logsumexp with mixed positive and negative values
    // Including large negative values to test for underflow handling
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Group 1: mixed values [5.0, -100.0]
    //   max = 5.0
    //   sumexp = exp(5-5) + exp(-100-5) = 1 + exp(-105) ≈ 1 (exp(-105) is negligible)
    //   logsumexp ≈ 5.0 + log(1) = 5.0
    //
    // Group 2: similar negative values [-1.0, -2.0, -3.0]
    //   max = -1.0
    //   sumexp = exp(-1-(-1)) + exp(-2-(-1)) + exp(-3-(-1))
    //          = exp(0) + exp(-1) + exp(-2)
    //          = 1 + 0.3679 + 0.1353 ≈ 1.5032
    //   logsumexp = -1.0 + log(1.5032) ≈ -0.5924
    //
    // Group 3: large negative values [-500.0, -501.0]
    //   max = -500.0
    //   sumexp = exp(0) + exp(-1) = 1.3679
    //   logsumexp = -500.0 + log(1.3679) ≈ -499.687
    let keys: Vec<u32> = vec![1, 1, 2, 2, 2, 3, 3];
    let values: Vec<f64> = vec![5.0, -100.0, -1.0, -2.0, -3.0, -500.0, -501.0];

    let buffer = create_keyed_f64_buffer(&provider, &keys, &values);

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::LogSumExp)])
        .expect("groupby_multi_agg should succeed");

    assert_eq!(result.num_rows(), 3);

    let result_values = provider.download_column_f64(&result, 1).expect("download result");

    let tolerance = 1e-5;

    // Group 1: logsumexp(5.0, -100.0) ≈ 5.0 (exp(-105) is negligible)
    let expected_0 = 5.0_f64 + (1.0_f64 + (-105.0_f64).exp()).ln();
    assert!(
        (result_values[0] - expected_0).abs() < tolerance,
        "Group 1: expected {}, got {}",
        expected_0,
        result_values[0]
    );

    // Group 2: logsumexp(-1.0, -2.0, -3.0)
    let expected_1 = -1.0_f64 + (1.0_f64 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln();
    assert!(
        (result_values[1] - expected_1).abs() < tolerance,
        "Group 2: expected {}, got {}",
        expected_1,
        result_values[1]
    );

    // Group 3: logsumexp(-500.0, -501.0)
    let expected_2 = -500.0_f64 + (1.0_f64 + (-1.0_f64).exp()).ln();
    assert!(
        (result_values[2] - expected_2).abs() < tolerance,
        "Group 3: expected {}, got {}",
        expected_2,
        result_values[2]
    );
}

#[test]
fn test_groupby_logsumexp_infinity() {
    // Test that +INFINITY is correctly handled in LogSumExp
    // When a group contains +INFINITY, the result should be +INFINITY
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Group 1: contains +INFINITY -> should return +INFINITY
    // Group 2: normal values -> should return normal result
    let keys: Vec<u32> = vec![1, 1, 2, 2];
    let values: Vec<f64> = vec![
        f64::INFINITY, 1.0,  // group 1: has infinity
        2.0, 3.0,            // group 2: normal values
    ];

    let buffer = create_keyed_f64_buffer(&provider, &keys, &values);

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::LogSumExp)])
        .expect("groupby_multi_agg should succeed");

    assert_eq!(result.num_rows(), 2);

    let result_values = provider.download_column_f64(&result, 1).expect("download result");

    // Group 1 should be +INFINITY
    assert!(result_values[0].is_infinite() && result_values[0] > 0.0,
        "Group with +INFINITY should return +INFINITY, got {}", result_values[0]);

    // Group 2 should be finite (logsumexp(2, 3) ≈ 3.3133)
    assert!(result_values[1].is_finite(),
        "Group with normal values should return finite, got {}", result_values[1]);
}
