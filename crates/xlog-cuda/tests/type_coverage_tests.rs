// crates/xlog-cuda/tests/type_coverage_tests.rs
//! Comprehensive type coverage tests for xlog-cuda operations.
//!
//! This module tests that GPU operations work correctly with different scalar types,
//! focusing on what's actually supported: filter, join, sort, groupby, union, and diff.

use std::sync::Arc;
use xlog_core::{AggOp, MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

// ============== Test Setup ==============

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

fn make_schema(cols: &[(&str, ScalarType)]) -> Schema {
    Schema::new(cols.iter().map(|(n, t)| (n.to_string(), *t)).collect())
}

// ============== Filter Tests with Different Types ==============

/// Test filter_by_mask with i64 values
/// The filter_by_mask operation is type-agnostic as it works on raw bytes
#[test]
fn test_filter_i64_by_mask() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create i64 data: [100, -200, 300, -400, 500]
    let data: Vec<i64> = vec![100, -200, 300, -400, 500];
    let mask: Vec<u8> = vec![1, 0, 1, 0, 1]; // Select indices 0, 2, 4

    // Convert i64 to bytes and create buffer
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let schema = make_schema(&[("val", ScalarType::I64)]);

    // Allocate and upload
    let mut col = provider
        .memory()
        .alloc::<u8>(bytes.len())
        .expect("alloc failed");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes, &mut col)
        .expect("upload failed");

    let buffer = xlog_cuda::CudaBuffer::from_columns(vec![col.into()], data.len() as u64, schema);

    // Apply filter by mask
    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();

    // Verify result
    assert_eq!(filtered.num_rows, 3);

    // Download and check values
    let result_col = filtered.column(0).unwrap();
    let mut result_bytes = vec![0u8; (filtered.num_rows as usize) * 8];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(result_col, &mut result_bytes)
        .expect("download failed");

    let result: Vec<i64> = result_bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(result, vec![100, 300, 500]);
}

/// Test filter_by_mask with f64 values
#[test]
fn test_filter_f64_by_mask() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create f64 data
    let data: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let mask: Vec<u8> = vec![0, 1, 1, 0, 1]; // Select indices 1, 2, 4

    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let schema = make_schema(&[("val", ScalarType::F64)]);

    let mut col = provider
        .memory()
        .alloc::<u8>(bytes.len())
        .expect("alloc failed");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes, &mut col)
        .expect("upload failed");

    let buffer = xlog_cuda::CudaBuffer::from_columns(vec![col.into()], data.len() as u64, schema);
    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();

    assert_eq!(filtered.num_rows, 3);

    let result_col = filtered.column(0).unwrap();
    let mut result_bytes = vec![0u8; (filtered.num_rows as usize) * 8];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(result_col, &mut result_bytes)
        .expect("download failed");

    let result: Vec<f64> = result_bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(result, vec![2.5, 3.5, 5.5]);
}

/// Test filter_by_mask with i32 values
#[test]
fn test_filter_i32_by_mask() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let data: Vec<i32> = vec![-10, 20, -30, 40, -50];
    let mask: Vec<u8> = vec![1, 1, 0, 0, 1];

    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let schema = make_schema(&[("val", ScalarType::I32)]);

    let mut col = provider
        .memory()
        .alloc::<u8>(bytes.len())
        .expect("alloc failed");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes, &mut col)
        .expect("upload failed");

    let buffer = xlog_cuda::CudaBuffer::from_columns(vec![col.into()], data.len() as u64, schema);
    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();

    assert_eq!(filtered.num_rows, 3);

    let result_col = filtered.column(0).unwrap();
    let mut result_bytes = vec![0u8; (filtered.num_rows as usize) * 4];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(result_col, &mut result_bytes)
        .expect("download failed");

    let result: Vec<i32> = result_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(result, vec![-10, 20, -50]);
}

/// Test filter_by_mask with f32 values
#[test]
fn test_filter_f32_by_mask() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let data: Vec<f32> = vec![1.1, 2.2, 3.3, 4.4];
    let mask: Vec<u8> = vec![1, 0, 1, 0];

    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let schema = make_schema(&[("val", ScalarType::F32)]);

    let mut col = provider
        .memory()
        .alloc::<u8>(bytes.len())
        .expect("alloc failed");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes, &mut col)
        .expect("upload failed");

    let buffer = xlog_cuda::CudaBuffer::from_columns(vec![col.into()], data.len() as u64, schema);
    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();

    assert_eq!(filtered.num_rows, 2);

    let result_col = filtered.column(0).unwrap();
    let mut result_bytes = vec![0u8; (filtered.num_rows as usize) * 4];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(result_col, &mut result_bytes)
        .expect("download failed");

    let result: Vec<f32> = result_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    assert!((result[0] - 1.1).abs() < 0.001);
    assert!((result[1] - 3.3).abs() < 0.001);
}

// ============== Sort Tests with U32 Keys ==============

/// Test sort with u32 keys (primary supported type)
#[test]
fn test_sort_u32_keys() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let keys: Vec<u32> = vec![5, 2, 8, 1, 9, 3];
    let values: Vec<u32> = vec![50, 20, 80, 10, 90, 30];
    let schema = make_schema(&[("key", ScalarType::U32), ("value", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    let sorted = provider.sort(&buffer, &[0]).unwrap();

    let result_keys = provider.download_column_u32(&sorted, 0).unwrap();
    let result_values = provider.download_column_u32(&sorted, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2, 3, 5, 8, 9]);
    assert_eq!(result_values, vec![10, 20, 30, 50, 80, 90]);
}

/// Test sort stability with duplicate keys
#[test]
fn test_sort_u32_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let keys: Vec<u32> = vec![3, 1, 2, 1, 3, 2];
    let schema = make_schema(&[("key", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&keys, schema)
        .unwrap();

    let sorted = provider.sort(&buffer, &[0]).unwrap();
    let result = provider.download_column_u32(&sorted, 0).unwrap();

    assert_eq!(result, vec![1, 1, 2, 2, 3, 3]);
}

// ============== Join Tests with U32 Keys ==============

/// Test inner join with u32 keys
#[test]
fn test_join_u32_keys_inner() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let left_key: Vec<u32> = vec![1, 2, 3];
    let left_val: Vec<u32> = vec![10, 20, 30];
    let right_key: Vec<u32> = vec![2, 3, 4];
    let right_val: Vec<u32> = vec![200, 300, 400];

    let left_schema = make_schema(&[("key", ScalarType::U32), ("lval", ScalarType::U32)]);
    let right_schema = make_schema(&[("key", ScalarType::U32), ("rval", ScalarType::U32)]);

    let left = provider
        .create_buffer_from_u32_columns(&[&left_key, &left_val], left_schema)
        .unwrap();
    let right = provider
        .create_buffer_from_u32_columns(&[&right_key, &right_val], right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::Inner)
        .unwrap();

    // Should have 2 matches: key 2 and key 3
    assert_eq!(result.num_rows, 2);
    assert_eq!(result.arity(), 4);
}

/// Test semi join with u32 keys
#[test]
fn test_join_u32_keys_semi() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let left: Vec<u32> = vec![1, 2, 3, 4];
    let right: Vec<u32> = vec![2, 4];

    let left_schema = make_schema(&[("val", ScalarType::U32)]);
    let right_schema = make_schema(&[("val", ScalarType::U32)]);

    let left_buf = provider
        .create_buffer_from_u32_slice(&left, left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_u32_slice(&right, right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Semi)
        .unwrap();

    assert_eq!(result.num_rows, 2);
    let vals = provider.download_column_u32(&result, 0).unwrap();
    assert!(vals.contains(&2));
    assert!(vals.contains(&4));
}

/// Test anti join with u32 keys
#[test]
fn test_join_u32_keys_anti() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let left: Vec<u32> = vec![1, 2, 3, 4];
    let right: Vec<u32> = vec![2, 4];

    let left_schema = make_schema(&[("val", ScalarType::U32)]);
    let right_schema = make_schema(&[("val", ScalarType::U32)]);

    let left_buf = provider
        .create_buffer_from_u32_slice(&left, left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_u32_slice(&right, right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Anti)
        .unwrap();

    assert_eq!(result.num_rows, 2);
    let vals = provider.download_column_u32(&result, 0).unwrap();
    assert!(vals.contains(&1));
    assert!(vals.contains(&3));
}

// ============== GroupBy Tests with U32 ==============

/// Test groupby sum with u32 values
#[test]
fn test_groupby_sum_u32() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let keys: Vec<u32> = vec![1, 1, 2, 2, 2];
    let values: Vec<u32> = vec![10, 20, 5, 15, 25];
    let schema = make_schema(&[("key", ScalarType::U32), ("value", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum)])
        .unwrap();

    assert_eq!(result.num_rows(), 2);

    let result_keys = provider.download_column_u32(&result, 0).unwrap();
    let result_sums = provider.download_column_u64(&result, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_sums, vec![30u64, 45u64]);
}

/// Test groupby count with u32 keys
#[test]
fn test_groupby_count_u32() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let keys: Vec<u32> = vec![1, 1, 2, 2, 2, 3];
    let values: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
    let schema = make_schema(&[("key", ScalarType::U32), ("value", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    let result = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Count)])
        .unwrap();

    assert_eq!(result.num_rows(), 3);

    let result_keys = provider.download_column_u32(&result, 0).unwrap();
    // Count results are u64 (to match predicate declaration types)
    let result_counts = provider.download_column_u64(&result, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2, 3]);
    assert_eq!(result_counts, vec![2u64, 3u64, 1u64]);
}

/// Test groupby min/max with u32 values
#[test]
fn test_groupby_min_max_u32() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let keys: Vec<u32> = vec![1, 1, 2, 2, 2];
    let values: Vec<u32> = vec![10, 20, 5, 15, 25];
    let schema = make_schema(&[("key", ScalarType::U32), ("value", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    // Test Min
    let result_min = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Min)])
        .unwrap();

    let result_keys = provider.download_column_u32(&result_min, 0).unwrap();
    let result_mins = provider.download_column_u32(&result_min, 1).unwrap();

    assert_eq!(result_keys, vec![1, 2]);
    assert_eq!(result_mins, vec![10, 5]);

    // Test Max
    let result_max = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Max)])
        .unwrap();

    let result_maxs = provider.download_column_u32(&result_max, 1).unwrap();
    assert_eq!(result_maxs, vec![20, 25]);
}

// ============== Union Tests with U32 ==============

/// Test union with u32 single column
#[test]
fn test_union_u32() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![3, 4, 5];
    let schema = make_schema(&[("val", ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    // Union should deduplicate: [1, 2, 3, 4, 5]
    assert_eq!(result_data, vec![1, 2, 3, 4, 5]);
}

/// Test union with no overlap
#[test]
fn test_union_u32_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let a: Vec<u32> = vec![1, 2];
    let b: Vec<u32> = vec![3, 4];
    let schema = make_schema(&[("val", ScalarType::U32)]);

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

// ============== Diff Tests with U32 ==============

/// Test diff with u32 single column
#[test]
fn test_diff_u32() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let a: Vec<u32> = vec![1, 2, 3, 4];
    let b: Vec<u32> = vec![2, 4];
    let schema = make_schema(&[("val", ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();
    let result_data = provider.download_column_u32(&result, 0).unwrap();

    // Diff: a - b = [1, 3]
    assert_eq!(result_data, vec![1, 3]);
}

/// Test diff with no overlap (all elements remain)
#[test]
fn test_diff_u32_no_overlap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let a: Vec<u32> = vec![1, 2, 3];
    let b: Vec<u32> = vec![4, 5, 6];
    let schema = make_schema(&[("val", ScalarType::U32)]);

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

// ============== Edge Case Tests ==============

/// Test operations with empty buffers
#[test]
fn test_operations_empty_buffer() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = make_schema(&[("val", ScalarType::U32)]);
    let empty = provider.create_empty_buffer(schema.clone()).unwrap();
    let non_empty: Vec<u32> = vec![1, 2, 3];
    let non_empty_buf = provider
        .create_buffer_from_u32_slice(&non_empty, schema.clone())
        .unwrap();

    // Filter empty buffer with mask
    let mask: Vec<u8> = vec![];
    let filtered = provider.filter_by_mask(&empty, &mask).unwrap();
    assert_eq!(filtered.num_rows, 0);

    // Sort empty buffer
    let sorted = provider.sort(&empty, &[0]).unwrap();
    assert_eq!(sorted.num_rows, 0);

    // Join with empty buffer
    let join_result = provider
        .hash_join_v2(&empty, &non_empty_buf, &[0], &[0], JoinType::Inner)
        .unwrap();
    assert_eq!(join_result.num_rows, 0);

    // Union with empty buffer
    let union_result = provider.union_gpu(&empty, &non_empty_buf).unwrap();
    let union_data = provider.download_column_u32(&union_result, 0).unwrap();
    assert_eq!(union_data, vec![1, 2, 3]);

    // Diff with empty buffer
    let diff_result = provider.diff_gpu(&non_empty_buf, &empty).unwrap();
    let diff_data = provider.download_column_u32(&diff_result, 0).unwrap();
    assert_eq!(diff_data, vec![1, 2, 3]);

    // Diff from empty buffer
    let diff_result2 = provider.diff_gpu(&empty, &non_empty_buf).unwrap();
    assert_eq!(diff_result2.num_rows, 0);
}

/// Test groupby with empty input
#[test]
fn test_groupby_empty_input() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = make_schema(&[("key", ScalarType::U32), ("value", ScalarType::U32)]);
    let empty = provider.create_empty_buffer(schema).unwrap();

    let result = provider
        .groupby_multi_agg(&empty, &[0], &[(1, AggOp::Sum)])
        .unwrap();

    assert_eq!(result.num_rows(), 0);
}

/// Test buffer operations with multiple aggregations
#[test]
fn test_buffer_operations_with_groupby() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create buffer with small dataset (same as existing groupby tests)
    let keys: Vec<u32> = vec![1, 1, 2, 2, 2, 3];
    let values: Vec<u32> = vec![10, 20, 5, 15, 25, 100];
    let schema = make_schema(&[("key", ScalarType::U32), ("value", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    // Test groupby with multiple aggregations
    let grouped = provider
        .groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Count), (1, AggOp::Sum), (1, AggOp::Min), (1, AggOp::Max)])
        .unwrap();
    assert_eq!(grouped.num_rows(), 3); // 3 unique keys

    // Verify results
    let result_keys = provider.download_column_u32(&grouped, 0).unwrap();
    // Count results are u64 (to match predicate declaration types)
    let result_counts = provider.download_column_u64(&grouped, 1).unwrap();
    let result_sums = provider.download_column_u64(&grouped, 2).unwrap();
    let result_mins = provider.download_column_u32(&grouped, 3).unwrap();
    let result_maxs = provider.download_column_u32(&grouped, 4).unwrap();

    assert_eq!(result_keys, vec![1, 2, 3]);
    assert_eq!(result_counts, vec![2u64, 3u64, 1u64]);
    assert_eq!(result_sums, vec![30u64, 45u64, 100u64]);
    assert_eq!(result_mins, vec![10, 5, 100]);
    assert_eq!(result_maxs, vec![20, 25, 100]);
}

/// Test sort operation with small data
#[test]
fn test_sort_operation() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create small buffer for sort testing (within current limits)
    let keys: Vec<u32> = vec![5, 2, 8, 1, 9, 3, 7, 4, 6, 0];
    let values: Vec<u32> = vec![50, 20, 80, 10, 90, 30, 70, 40, 60, 0];
    let schema = make_schema(&[("key", ScalarType::U32), ("value", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_columns(&[&keys, &values], schema)
        .unwrap();

    let sorted = provider.sort(&buffer, &[0]).unwrap();
    assert_eq!(sorted.num_rows, 10);

    // Verify sorted order
    let sorted_keys = provider.download_column_u32(&sorted, 0).unwrap();
    for i in 1..sorted_keys.len() {
        assert!(
            sorted_keys[i - 1] <= sorted_keys[i],
            "Sort order violated at index {}: {} > {}",
            i,
            sorted_keys[i - 1],
            sorted_keys[i]
        );
    }
}

/// Test large union operation
#[test]
fn test_large_union() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create two large buffers with some overlap
    let a: Vec<u32> = (0..5000).collect();
    let b: Vec<u32> = (2500..7500).collect();
    let schema = make_schema(&[("val", ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.union_gpu(&buf_a, &buf_b).unwrap();

    // Union should have 7500 unique values (0..7500)
    assert_eq!(result.num_rows, 7500);
}

/// Test diff operation with moderate size
/// Note: Current diff implementation has a 256 element limit for prefix_sum
#[test]
fn test_moderate_diff() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create two buffers within the 256 element limit
    let a: Vec<u32> = (0..100).collect();
    let b: Vec<u32> = (50..100).collect();
    let schema = make_schema(&[("val", ScalarType::U32)]);

    let buf_a = provider
        .create_buffer_from_u32_slice(&a, schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_u32_slice(&b, schema.clone())
        .unwrap();

    let result = provider.diff_gpu(&buf_a, &buf_b).unwrap();

    // Diff should have 50 values (0..50)
    assert_eq!(result.num_rows, 50);
}

/// Test multi-column filter by mask with mixed types
#[test]
fn test_filter_multi_column_mixed_types() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create a buffer with u32 and u64 columns
    let col0: Vec<u32> = vec![1, 2, 3, 4, 5];
    let col1: Vec<u64> = vec![100, 200, 300, 400, 500];
    let mask: Vec<u8> = vec![1, 0, 1, 0, 1];

    let schema = make_schema(&[("u32_col", ScalarType::U32), ("u64_col", ScalarType::U64)]);

    // Upload u32 column
    let col0_bytes: Vec<u8> = col0.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut cuda_col0 = provider
        .memory()
        .alloc::<u8>(col0_bytes.len())
        .expect("alloc failed");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&col0_bytes, &mut cuda_col0)
        .expect("upload failed");

    // Upload u64 column
    let col1_bytes: Vec<u8> = col1.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut cuda_col1 = provider
        .memory()
        .alloc::<u8>(col1_bytes.len())
        .expect("alloc failed");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&col1_bytes, &mut cuda_col1)
        .expect("upload failed");

    let buffer = xlog_cuda::CudaBuffer::from_columns(
        vec![cuda_col0.into(), cuda_col1.into()],
        col0.len() as u64,
        schema,
    );

    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();
    assert_eq!(filtered.num_rows, 3);

    // Download and verify u32 column
    let result_col0 = filtered.column(0).unwrap();
    let mut result0_bytes = vec![0u8; (filtered.num_rows as usize) * 4];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(result_col0, &mut result0_bytes)
        .expect("download failed");
    let result0: Vec<u32> = result0_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result0, vec![1, 3, 5]);

    // Download and verify u64 column
    let result_col1 = filtered.column(1).unwrap();
    let mut result1_bytes = vec![0u8; (filtered.num_rows as usize) * 8];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(result_col1, &mut result1_bytes)
        .expect("download failed");
    let result1: Vec<u64> = result1_bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result1, vec![100, 300, 500]);
}

/// Test single element buffers
#[test]
fn test_single_element_buffer() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let data: Vec<u32> = vec![42];
    let schema = make_schema(&[("val", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&data, schema.clone())
        .unwrap();

    // Sort single element
    let sorted = provider.sort(&buffer, &[0]).unwrap();
    let sorted_data = provider.download_column_u32(&sorted, 0).unwrap();
    assert_eq!(sorted_data, vec![42]);

    // Filter with mask=1
    let filtered = provider.filter_by_mask(&buffer, &[1]).unwrap();
    assert_eq!(filtered.num_rows, 1);

    // Filter with mask=0
    let filtered0 = provider.filter_by_mask(&buffer, &[0]).unwrap();
    assert_eq!(filtered0.num_rows, 0);
}

/// Test all mask values are 1 (select all)
#[test]
fn test_filter_select_all() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let mask: Vec<u8> = vec![1, 1, 1, 1, 1];
    let schema = make_schema(&[("val", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&data, schema)
        .unwrap();

    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();
    let result = provider.download_column_u32(&filtered, 0).unwrap();

    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

/// Test all mask values are 0 (select none)
#[test]
fn test_filter_select_none() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let mask: Vec<u8> = vec![0, 0, 0, 0, 0];
    let schema = make_schema(&[("val", ScalarType::U32)]);

    let buffer = provider
        .create_buffer_from_u32_slice(&data, schema)
        .unwrap();

    let filtered = provider.filter_by_mask(&buffer, &mask).unwrap();
    assert_eq!(filtered.num_rows, 0);
}

#[test]
fn test_arith_u32_i32_u64_f32() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema_u32 = Schema::new(vec![("v".to_string(), ScalarType::U32)]);
    let a_u32 = provider
        .create_buffer_from_u32_slice(&[1, 2, 3], schema_u32.clone())
        .unwrap();
    let b_u32 = provider
        .create_buffer_from_u32_slice(&[4, 5, 6], schema_u32.clone())
        .unwrap();
    let sum_u32 = provider.add_columns(&a_u32, &b_u32).unwrap();
    let vals_u32 = provider.download_column_u32(&sum_u32, 0).unwrap();
    assert_eq!(vals_u32, vec![5, 7, 9]);

    let b_u32_div = provider
        .create_buffer_from_u32_slice(&[1, 0, 2], schema_u32.clone())
        .unwrap();
    let div_u32 = provider.div_columns(&a_u32, &b_u32_div).unwrap();
    let vals_u32_div = provider.download_column_u32(&div_u32, 0).unwrap();
    assert_eq!(vals_u32_div, vec![1, u32::MAX, 1]);

    let schema_i32 = Schema::new(vec![("v".to_string(), ScalarType::I32)]);
    let a_i32 = provider
        .create_buffer_from_i32_slice(&[-3, 4, -5], schema_i32.clone())
        .unwrap();
    let abs_i32 = provider.abs_column(&a_i32).unwrap();
    let vals_i32 = provider.download_column_i32(&abs_i32, 0).unwrap();
    assert_eq!(vals_i32, vec![3, 4, 5]);

    let schema_u64 = Schema::new(vec![("v".to_string(), ScalarType::U64)]);
    let a_u64 = provider
        .create_buffer_from_u64_slice(&[10, 20, 30], schema_u64.clone())
        .unwrap();
    let b_u64 = provider
        .create_buffer_from_u64_slice(&[1, 2, 3], schema_u64.clone())
        .unwrap();
    let diff_u64 = provider.sub_columns(&a_u64, &b_u64).unwrap();
    let vals_u64 = provider.download_column_u64(&diff_u64, 0).unwrap();
    assert_eq!(vals_u64, vec![9, 18, 27]);

    let b_u64_div = provider
        .create_buffer_from_u64_slice(&[2, 0, 3], schema_u64.clone())
        .unwrap();
    let div_u64 = provider.div_columns(&a_u64, &b_u64_div).unwrap();
    let vals_u64_div = provider.download_column_u64(&div_u64, 0).unwrap();
    assert_eq!(vals_u64_div, vec![5, u64::MAX, 10]);

    let schema_f32 = Schema::new(vec![("v".to_string(), ScalarType::F32)]);
    let a_f32 = provider
        .create_buffer_from_f32_slice(&[1.5, -2.0, 3.0], schema_f32.clone())
        .unwrap();
    let b_f32 = provider
        .create_buffer_from_f32_slice(&[2.0, 2.0, 0.5], schema_f32.clone())
        .unwrap();
    let prod_f32 = provider.mul_columns(&a_f32, &b_f32).unwrap();
    let vals_f32 = provider.download_column_f32(&prod_f32, 0).unwrap();
    assert!((vals_f32[0] - 3.0).abs() < 1e-6);
    assert!((vals_f32[1] + 4.0).abs() < 1e-6);
    assert!((vals_f32[2] - 1.5).abs() < 1e-6);
}
