// crates/xlog-cuda/tests/test_membership_mask.rs
//! Tests for the GPU-side membership_mask primitive on CudaKernelProvider.

mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema, XlogError};
use xlog_cuda::CudaKernelProvider;

struct StrictDeterministicD2hGuard<'a> {
    provider: &'a CudaKernelProvider,
    was_enabled: bool,
}

impl<'a> StrictDeterministicD2hGuard<'a> {
    fn enable(provider: &'a CudaKernelProvider) -> Self {
        let was_enabled = provider.strict_deterministic_d2h_enabled();
        provider.enable_strict_deterministic_d2h();
        Self {
            provider,
            was_enabled,
        }
    }
}

impl Drop for StrictDeterministicD2hGuard<'_> {
    fn drop(&mut self) {
        if self.was_enabled {
            self.provider.enable_strict_deterministic_d2h();
        } else {
            self.provider.disable_strict_deterministic_d2h();
        }
    }
}

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
        .create_buffer_from_u32_columns(&[&[1u32, 2], &[10u32, 20], &[3u32, 4]], rel_schema)
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

#[test]
fn membership_mask_records_one_downloaded_byte_per_probe_row() {
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
        .create_buffer_from_u32_columns(&[&[1u32, 2, 9], &[2u32, 3, 10]], schema)
        .unwrap();

    provider.reset_host_transfer_stats();
    let mask = provider
        .membership_mask(&queries, &relation, &[0, 1], &[0, 1])
        .unwrap();
    let stats = provider.host_transfer_stats();

    assert_eq!(mask, vec![true, true, false]);
    assert_eq!(stats.dtoh_bytes, 3, "one mask byte per probe row");
    assert_eq!(stats.dtoh_calls, 1, "the mask is downloaded in one call");
}

#[test]
fn membership_mask_tracks_download_larger_than_metadata_cap() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
    let probe_row_count = CudaKernelProvider::DTOH_SMALL_METADATA_MAX_BYTES + 1;
    let query_values: Vec<u32> = (0..probe_row_count).map(|value| value as u32).collect();
    let relation_values = [0u32, (probe_row_count - 1) as u32];
    let relation = provider
        .create_buffer_from_u32_columns(&[&relation_values], schema.clone())
        .unwrap();
    let queries = provider
        .create_buffer_from_u32_columns(&[&query_values], schema)
        .unwrap();

    provider.reset_host_transfer_stats();
    let mask = provider
        .membership_mask(&queries, &relation, &[0], &[0])
        .unwrap();
    let stats = provider.host_transfer_stats();

    assert_eq!(mask.len(), probe_row_count);
    assert!(mask[0]);
    assert!(mask[probe_row_count - 1]);
    assert_eq!(mask.iter().filter(|&&matched| matched).count(), 2);
    assert_eq!(stats.dtoh_bytes, probe_row_count as u64);
    assert_eq!(stats.dtoh_calls, 1);
}

#[test]
fn membership_mask_fails_closed_under_strict_deterministic_d2h() {
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
        .create_buffer_from_u32_columns(&[&[1u32, 2, 9], &[2u32, 3, 10]], schema)
        .unwrap();

    provider.reset_deterministic_d2h_violations();
    provider.reset_host_transfer_stats();
    let _strict_gate = StrictDeterministicD2hGuard::enable(&provider);
    let error = provider
        .membership_mask(&queries, &relation, &[0, 1], &[0, 1])
        .expect_err("membership_mask must reject its three-byte mask download");
    let stats = provider.host_transfer_stats();

    match error {
        XlogError::Execution(message) => {
            assert!(
                message.contains("dtoh_sync_copy_into_tracked"),
                "error must identify the tracked D2H operation: {message}"
            );
            assert!(
                message.contains("3 bytes"),
                "error must report the attempted mask byte count: {message}"
            );
        }
        other => panic!("expected a strict-gate execution error, got {other}"),
    }
    assert_eq!(provider.deterministic_d2h_violation_count(), 1);
    assert_eq!(stats.dtoh_bytes, 0, "rejected copies must not record bytes");
    assert_eq!(
        stats.dtoh_calls, 0,
        "rejected copies must not record a call"
    );
}

#[test]
fn strict_d2h_guard_restores_prior_enabled_state() {
    let Some(provider) = setup_provider() else {
        return;
    };

    let _outer_guard = StrictDeterministicD2hGuard::enable(&provider);
    {
        let _nested_guard = StrictDeterministicD2hGuard::enable(&provider);
        assert!(provider.strict_deterministic_d2h_enabled());
    }

    assert!(
        provider.strict_deterministic_d2h_enabled(),
        "dropping a nested guard must restore the previously enabled state"
    );
}
