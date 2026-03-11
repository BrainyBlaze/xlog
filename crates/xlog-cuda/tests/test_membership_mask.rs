// crates/xlog-cuda/tests/test_membership_mask.rs
//! Tests for the GPU-side membership_mask primitive on CudaKernelProvider.

mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema};

#[test]
fn membership_mask_basic() {
    let Some(provider) = setup_provider() else {
        return;
    };
    // Relation: edge(1,2), edge(2,3), edge(3,4)
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    let relation = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2, 3], &[2u32, 3, 4]], schema.clone())
        .unwrap();
    // Query: (1,2) and (5,6)
    let queries = provider
        .create_buffer_from_u32_columns(&[&[1u32, 5], &[2u32, 6]], schema.clone())
        .unwrap();
    let keys: Vec<usize> = vec![0, 1];
    let mask = provider
        .membership_mask(&queries, &relation, &keys, &keys)
        .unwrap();
    assert_eq!(mask, vec![true, false]);
}

#[test]
fn membership_mask_all_match() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    let relation = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2], &[2u32, 3]], schema.clone())
        .unwrap();
    let queries = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2], &[2u32, 3]], schema.clone())
        .unwrap();
    let keys: Vec<usize> = vec![0, 1];
    let mask = provider
        .membership_mask(&queries, &relation, &keys, &keys)
        .unwrap();
    assert_eq!(mask, vec![true, true]);
}

#[test]
fn membership_mask_none_match() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    let relation = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2], &[2u32, 3]], schema.clone())
        .unwrap();
    let queries = provider
        .create_buffer_from_u32_columns(&[&[10u32, 20], &[20u32, 30]], schema.clone())
        .unwrap();
    let keys: Vec<usize> = vec![0, 1];
    let mask = provider
        .membership_mask(&queries, &relation, &keys, &keys)
        .unwrap();
    assert_eq!(mask, vec![false, false]);
}

#[test]
fn membership_mask_empty_relation() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    let relation = provider.create_empty_buffer(schema.clone()).unwrap();
    let queries = provider
        .create_buffer_from_u32_columns(&[&[1u32], &[2u32]], schema.clone())
        .unwrap();
    let keys: Vec<usize> = vec![0, 1];
    let mask = provider
        .membership_mask(&queries, &relation, &keys, &keys)
        .unwrap();
    assert_eq!(mask, vec![false]);
}

#[test]
fn membership_mask_projected_keys() {
    let Some(provider) = setup_provider() else {
        return;
    };
    // Relation with 3 columns: (src, mid, dst)
    let rel_schema = Schema::new(vec![
        ("src".to_string(), ScalarType::U32),
        ("mid".to_string(), ScalarType::U32),
        ("dst".to_string(), ScalarType::U32),
    ]);
    let relation = provider
        .create_buffer_from_u32_columns(
            &[&[1u32, 2], &[10u32, 20], &[3u32, 4]],
            rel_schema,
        )
        .unwrap();

    // Query: check if (src=1, dst=3) exists (columns 0 and 2 only)
    let query_schema = Schema::new(vec![
        ("src".to_string(), ScalarType::U32),
        ("dst".to_string(), ScalarType::U32),
    ]);
    let queries = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2], &[3u32, 5]], query_schema)
        .unwrap();

    // Query keys [0,1] map to relation keys [0,2]
    let mask = provider
        .membership_mask(&queries, &relation, &[0, 1], &[0, 2])
        .unwrap();
    assert_eq!(mask, vec![true, false]);
}
