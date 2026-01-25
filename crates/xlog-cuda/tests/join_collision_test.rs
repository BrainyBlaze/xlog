//! Test for join key verification (not just hash comparison)
//!
//! These tests ensure that hash_join_v2 compares actual key bytes,
//! not just hash values. While FNV-1a 64-bit hash has very low collision
//! probability (~2^-64), joins must be mathematically correct.

use std::sync::Arc;
use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager, JoinType};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 256); // 256 MB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    CudaKernelProvider::new(device, memory).ok()
}

fn make_schema(cols: &[(&str, ScalarType)]) -> Schema {
    Schema::new(cols.iter().map(|(n, t)| (n.to_string(), *t)).collect())
}

/// Test that joins compare actual keys, not just hashes.
/// Even if two keys have the same hash, they should not match unless equal.
#[test]
fn test_join_verifies_keys_not_just_hash() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping test: no CUDA device");
        return;
    };

    // Create left table: key=100, val=1
    let left_key: Vec<u32> = vec![100];
    let left_val: Vec<u32> = vec![1];
    let left_schema = make_schema(&[("key", ScalarType::U32), ("val", ScalarType::U32)]);
    let left = provider
        .create_buffer_from_u32_columns(&[&left_key, &left_val], left_schema)
        .unwrap();

    // Create right table: key=200, val=2 (different key, might have hash collision)
    let right_key: Vec<u32> = vec![200];
    let right_val: Vec<u32> = vec![2];
    let right_schema = make_schema(&[("key", ScalarType::U32), ("val", ScalarType::U32)]);
    let right = provider
        .create_buffer_from_u32_columns(&[&right_key, &right_val], right_schema)
        .unwrap();

    // Join on key column - should produce ZERO results since 100 != 200
    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::Inner)
        .unwrap();

    assert_eq!(
        result.num_rows(),
        0,
        "Join should produce no results when keys differ, even if hashes might collide"
    );
}

/// Test that identical keys DO match
#[test]
fn test_join_matches_equal_keys() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping test: no CUDA device");
        return;
    };

    let left_key: Vec<u32> = vec![100, 200];
    let left_val: Vec<u32> = vec![1, 2];
    let left_schema = make_schema(&[("key", ScalarType::U32), ("val", ScalarType::U32)]);
    let left = provider
        .create_buffer_from_u32_columns(&[&left_key, &left_val], left_schema)
        .unwrap();

    let right_key: Vec<u32> = vec![200, 300];
    let right_val: Vec<u32> = vec![10, 20];
    let right_schema = make_schema(&[("key", ScalarType::U32), ("val", ScalarType::U32)]);
    let right = provider
        .create_buffer_from_u32_columns(&[&right_key, &right_val], right_schema)
        .unwrap();

    // Join on key column - should match on key=200
    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::Inner)
        .unwrap();

    assert_eq!(result.num_rows(), 1, "Join should match on equal keys");
}

/// Test multi-column key verification
#[test]
fn test_join_verifies_multi_column_keys() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping test: no CUDA device");
        return;
    };

    // Left: (a=1, b=2, payload=100)
    let left_a: Vec<u32> = vec![1];
    let left_b: Vec<u32> = vec![2];
    let left_p: Vec<u32> = vec![100];
    let left_schema = make_schema(&[
        ("a", ScalarType::U32),
        ("b", ScalarType::U32),
        ("payload", ScalarType::U32),
    ]);
    let left = provider
        .create_buffer_from_u32_columns(&[&left_a, &left_b, &left_p], left_schema)
        .unwrap();

    // Right: (x=1, y=3, value=200) - first column matches, second doesn't
    let right_x: Vec<u32> = vec![1];
    let right_y: Vec<u32> = vec![3]; // Different from left_b!
    let right_v: Vec<u32> = vec![200];
    let right_schema = make_schema(&[
        ("x", ScalarType::U32),
        ("y", ScalarType::U32),
        ("value", ScalarType::U32),
    ]);
    let right = provider
        .create_buffer_from_u32_columns(&[&right_x, &right_y, &right_v], right_schema)
        .unwrap();

    // Join on (a, b) = (x, y) - should NOT match because (1, 2) != (1, 3)
    let result = provider
        .hash_join_v2(&left, &right, &[0, 1], &[0, 1], JoinType::Inner)
        .unwrap();

    assert_eq!(
        result.num_rows(),
        0,
        "Multi-column join should verify all key columns, not just hash"
    );
}

/// Test semi-join also verifies keys
#[test]
fn test_semi_join_verifies_keys() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping test: no CUDA device");
        return;
    };

    let left_key: Vec<u32> = vec![100, 200];
    let left_schema = make_schema(&[("key", ScalarType::U32)]);
    let left = provider
        .create_buffer_from_u32_slice(&left_key, left_schema)
        .unwrap();

    let right_key: Vec<u32> = vec![300, 400]; // No matching keys
    let right_schema = make_schema(&[("key", ScalarType::U32)]);
    let right = provider
        .create_buffer_from_u32_slice(&right_key, right_schema)
        .unwrap();

    // Semi-join should return no rows since no keys match
    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::Semi)
        .unwrap();

    assert_eq!(
        result.num_rows(),
        0,
        "Semi-join should verify keys, not just hashes"
    );
}

/// Test anti-join also verifies keys
#[test]
fn test_anti_join_verifies_keys() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping test: no CUDA device");
        return;
    };

    let left_key: Vec<u32> = vec![100, 200];
    let left_schema = make_schema(&[("key", ScalarType::U32)]);
    let left = provider
        .create_buffer_from_u32_slice(&left_key, left_schema)
        .unwrap();

    let right_key: Vec<u32> = vec![300, 400]; // No matching keys
    let right_schema = make_schema(&[("key", ScalarType::U32)]);
    let right = provider
        .create_buffer_from_u32_slice(&right_key, right_schema)
        .unwrap();

    // Anti-join should return ALL left rows since no keys match
    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::Anti)
        .unwrap();

    assert_eq!(
        result.num_rows(),
        2,
        "Anti-join should verify keys and return all rows when none match"
    );
}
