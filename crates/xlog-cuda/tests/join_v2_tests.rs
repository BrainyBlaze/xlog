// crates/xlog-cuda/tests/join_v2_tests.rs
//! Tests for hash_join_v2 with multi-column keys and typed join support

use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

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

#[test]
fn test_hash_join_v2_single_key() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: key, payload
    // Right: key, value
    // Join on key
    let left_key: Vec<u32> = vec![1, 2, 3];
    let left_payload: Vec<u32> = vec![10, 20, 30];
    let right_key: Vec<u32> = vec![2, 3, 4];
    let right_value: Vec<u32> = vec![200, 300, 400];

    let left_schema = make_schema(&[("key", ScalarType::U32), ("payload", ScalarType::U32)]);
    let right_schema = make_schema(&[("key", ScalarType::U32), ("value", ScalarType::U32)]);

    let left = provider
        .create_buffer_from_u32_columns(&[&left_key, &left_payload], left_schema)
        .unwrap();
    let right = provider
        .create_buffer_from_u32_columns(&[&right_key, &right_value], right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::Inner)
        .unwrap();

    // Expected: rows with matching keys 2 and 3
    // Result should have 4 columns: left.key, left.payload, right.key, right.value
    let rows = provider.download_column_u32(&result, 0).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(result.arity(), 4);
}

#[test]
fn test_semi_join() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: [1, 2, 3, 4]  Right: [2, 4]
    // Semi-join: left rows that exist in right -> [2, 4]
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

    let vals = provider.download_column_u32(&result, 0).unwrap();
    // Order may vary, just check count and values
    assert_eq!(vals.len(), 2);
    assert!(vals.contains(&2));
    assert!(vals.contains(&4));
}

#[test]
fn test_anti_join() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: [1, 2, 3, 4]  Right: [2, 4]
    // Anti-join: left rows NOT in right -> [1, 3]
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

    let vals = provider.download_column_u32(&result, 0).unwrap();
    assert_eq!(vals.len(), 2);
    assert!(vals.contains(&1));
    assert!(vals.contains(&3));
}

#[test]
fn test_join_no_matches() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let left: Vec<u32> = vec![1, 2, 3];
    let right: Vec<u32> = vec![4, 5, 6];

    let left_schema = make_schema(&[("val", ScalarType::U32)]);
    let right_schema = make_schema(&[("val", ScalarType::U32)]);

    let left_buf = provider
        .create_buffer_from_u32_slice(&left, left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_u32_slice(&right, right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
        .unwrap();

    let vals = provider.download_column_u32(&result, 0).unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_inner_join_with_duplicates() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: [1, 2, 2], Right: [2, 2]
    // Inner join on key should produce 4 results (2 from left x 2 from right)
    let left: Vec<u32> = vec![1, 2, 2];
    let right: Vec<u32> = vec![2, 2];

    let left_schema = make_schema(&[("key", ScalarType::U32)]);
    let right_schema = make_schema(&[("key", ScalarType::U32)]);

    let left_buf = provider
        .create_buffer_from_u32_slice(&left, left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_u32_slice(&right, right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::Inner)
        .unwrap();

    // 2 matching left rows x 2 matching right rows = 4
    let vals = provider.download_column_u32(&result, 0).unwrap();
    assert_eq!(vals.len(), 4);
}

#[test]
fn test_semi_join_preserves_left_columns() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left has two columns: key and payload
    // Semi-join should return left rows (with all columns) that have matches
    let left_key: Vec<u32> = vec![1, 2, 3];
    let left_payload: Vec<u32> = vec![10, 20, 30];
    let right_key: Vec<u32> = vec![2, 3, 4];

    let left_schema = make_schema(&[("key", ScalarType::U32), ("payload", ScalarType::U32)]);
    let right_schema = make_schema(&[("key", ScalarType::U32)]);

    let left = provider
        .create_buffer_from_u32_columns(&[&left_key, &left_payload], left_schema)
        .unwrap();
    let right = provider
        .create_buffer_from_u32_slice(&right_key, right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::Semi)
        .unwrap();

    // Semi-join keeps left schema
    assert_eq!(result.arity(), 2);
    let keys = provider.download_column_u32(&result, 0).unwrap();
    let payloads = provider.download_column_u32(&result, 1).unwrap();
    assert_eq!(keys.len(), 2);

    // Should have rows for keys 2 and 3
    assert!(keys.contains(&2));
    assert!(keys.contains(&3));
    // Payloads should correspond to keys
    for i in 0..keys.len() {
        if keys[i] == 2 {
            assert_eq!(payloads[i], 20);
        } else if keys[i] == 3 {
            assert_eq!(payloads[i], 30);
        }
    }
}

#[test]
fn test_left_outer_join() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: [1, 2, 3], Right: [2]
    // Left outer should return:
    // - (1, 0) - no match, right value is null/0
    // - (2, 2) - matched
    // - (3, 0) - no match, right value is null/0
    let left: Vec<u32> = vec![1, 2, 3];
    let right: Vec<u32> = vec![2];

    let left_schema = make_schema(&[("lval", ScalarType::U32)]);
    let right_schema = make_schema(&[("rval", ScalarType::U32)]);

    let left_buf = provider
        .create_buffer_from_u32_slice(&left, left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_u32_slice(&right, right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::LeftOuter)
        .unwrap();

    // Should have 3 rows (all left rows preserved)
    assert_eq!(result.arity(), 2);

    let left_vals = provider.download_column_u32(&result, 0).unwrap();
    let right_vals = provider.download_column_u32(&result, 1).unwrap();
    assert_eq!(left_vals.len(), 3);

    // All left values should be present
    assert!(left_vals.contains(&1));
    assert!(left_vals.contains(&2));
    assert!(left_vals.contains(&3));

    // The matched row (key=2) should have right value 2
    // Unmatched rows should have right value 0 (null represented as 0)
    for i in 0..left_vals.len() {
        if left_vals[i] == 2 {
            assert_eq!(right_vals[i], 2);
        } else {
            assert_eq!(right_vals[i], 0);
        }
    }
}

#[test]
fn test_left_outer_all_unmatched() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: [1, 2, 3], Right: [4, 5]
    // No matches - all left rows with null right columns
    let left: Vec<u32> = vec![1, 2, 3];
    let right: Vec<u32> = vec![4, 5];

    let left_schema = make_schema(&[("lval", ScalarType::U32)]);
    let right_schema = make_schema(&[("rval", ScalarType::U32)]);

    let left_buf = provider
        .create_buffer_from_u32_slice(&left, left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_u32_slice(&right, right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::LeftOuter)
        .unwrap();

    let right_vals = provider.download_column_u32(&result, 1).unwrap();
    assert_eq!(right_vals.len(), 3);
    // All right values should be 0 (null)
    assert!(right_vals.iter().all(|&v| v == 0));
}

#[test]
fn test_multi_column_join() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: (a, b, payload)
    // Right: (x, y, value)
    // Join on (a, b) = (x, y)

    // Left side
    let left_a: Vec<u32> = vec![1, 1, 2];
    let left_b: Vec<u32> = vec![10, 20, 10];
    let left_p: Vec<u32> = vec![100, 200, 300];

    // Right side
    let right_x: Vec<u32> = vec![1, 2, 1];
    let right_y: Vec<u32> = vec![10, 10, 20];
    let right_v: Vec<u32> = vec![1000, 2000, 3000];

    let left_schema = make_schema(&[
        ("a", ScalarType::U32),
        ("b", ScalarType::U32),
        ("payload", ScalarType::U32),
    ]);
    let right_schema = make_schema(&[
        ("x", ScalarType::U32),
        ("y", ScalarType::U32),
        ("value", ScalarType::U32),
    ]);

    let left = provider
        .create_buffer_from_u32_columns(&[&left_a, &left_b, &left_p], left_schema)
        .unwrap();
    let right = provider
        .create_buffer_from_u32_columns(&[&right_x, &right_y, &right_v], right_schema)
        .unwrap();

    // Join on columns 0,1 (a,b) = (x,y)
    let result = provider
        .hash_join_v2(&left, &right, &[0, 1], &[0, 1], JoinType::Inner)
        .unwrap();

    // Expected matches:
    // (1, 10, 100) joins (1, 10, 1000) -> result row
    // (1, 20, 200) joins (1, 20, 3000) -> result row
    // (2, 10, 300) joins (2, 10, 2000) -> result row
    let rows = provider.download_column_u32(&result, 0).unwrap();
    assert_eq!(rows.len(), 3, "Should have 3 matches for multi-column join");
}

#[test]
fn test_left_outer_empty_right() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Left: [1, 2, 3], Right: empty
    let left: Vec<u32> = vec![1, 2, 3];
    let right: Vec<u32> = vec![];

    let left_schema = make_schema(&[("lval", ScalarType::U32)]);
    let right_schema = make_schema(&[("rval", ScalarType::U32)]);

    let left_buf = provider
        .create_buffer_from_u32_slice(&left, left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_u32_slice(&right, right_schema)
        .unwrap();

    let result = provider
        .hash_join_v2(&left_buf, &right_buf, &[0], &[0], JoinType::LeftOuter)
        .unwrap();

    // All left rows should be returned with null right columns
    assert_eq!(result.arity(), 2);

    let right_vals = provider.download_column_u32(&result, 1).unwrap();
    assert_eq!(right_vals.len(), 3);
    assert!(right_vals.iter().all(|&v| v == 0));
}
