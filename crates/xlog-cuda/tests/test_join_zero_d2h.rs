// crates/xlog-cuda/tests/test_join_zero_d2h.rs
//! Acceptance tests for the v0.5.5 GPU-resident binary-join contract.
//!
//! Each test exercises one of the four binary-join entry points
//! (`hash_join_inner_v2`, `hash_join_inner_v2_indexed`,
//! `hash_join_left_outer_*` non-indexed and indexed) and asserts the
//! triple-zero contract on success:
//!
//!   * `deterministic_d2h_violation_count() == 0`
//!   * `control_plane_d2h_read_count() == 0`
//!   * `sanctioned_status_read_count() == 0`
//!
//! The `control_plane_d2h_read_count` counter is incremented by every
//! `device_row_count` cache-miss and every `dtoh_scalar_untracked`
//! call. On the pre-fix baseline these tests **fail** because the
//! join entry points call `device_row_count(left)` / `device_row_count(right)`
//! and then `read_join_output_count_metadata` (which goes through
//! `dtoh_scalar_untracked`).
//!
//! After the fix lands the kernels accept device row-count pointers
//! directly and use a count→prefix-scan→materialize flow with a
//! deferred GPU overflow status, so the join hot path issues zero
//! control-plane reads on success.

mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::{CudaKernelProvider, JoinType};

fn buf_u32x1(provider: &CudaKernelProvider, name: &str, data: &[u32]) -> xlog_cuda::CudaBuffer {
    let schema = Schema::new(vec![(name.to_string(), ScalarType::U32)]);
    provider
        .create_buffer_from_slice::<u32>(data, schema)
        .unwrap()
}

fn assert_join_path_clean(provider: &CudaKernelProvider, label: &str) {
    let v = provider.deterministic_d2h_violation_count();
    let c = provider.control_plane_d2h_read_count();
    let s = provider.sanctioned_status_read_count();
    assert_eq!(
        v, 0,
        "{}: deterministic_d2h_violation_count expected 0, got {}",
        label, v
    );
    assert_eq!(
        c, 0,
        "{}: control_plane_d2h_read_count expected 0, got {} \
         (device_row_count cache miss or dtoh_scalar_untracked in the join hot path)",
        label, c
    );
    assert_eq!(
        s, 0,
        "{}: sanctioned_status_read_count expected 0 on success path, got {}",
        label, s
    );
}

fn reset_counters(provider: &CudaKernelProvider) {
    provider.reset_deterministic_d2h_violations();
    provider.reset_control_plane_d2h_reads();
    provider.reset_sanctioned_status_reads();
    // Prime row-count caches on inputs so the join entry points don't
    // double-count the row-count read for the test's own inputs. The
    // counter then measures only what the join itself does.
}

fn prime_row_count_cache(provider: &CudaKernelProvider, buffer: &xlog_cuda::CudaBuffer) {
    // Force a row-count cache fill via the public API, then reset the
    // counter. Without this, the very first `device_row_count` inside
    // the join would inevitably increment the counter for legitimate
    // input-side metadata reads that aren't the contract under test.
    let _ = provider.device_row_count(buffer);
}

#[test]
fn inner_join_zero_d2h_on_success_path() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let left = buf_u32x1(&provider, "lval", &[1, 2, 3]);
    let right = buf_u32x1(&provider, "rval", &[2, 3, 4]);

    prime_row_count_cache(&provider, &left);
    prime_row_count_cache(&provider, &right);
    reset_counters(&provider);

    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::Inner)
        .expect("inner join must succeed");

    assert_join_path_clean(&provider, "inner_join_v2");
    let v = provider.download_column::<u32>(&result, 0).unwrap();
    let mut got = v;
    got.sort_unstable();
    assert_eq!(got, vec![2u32, 3]);
}

#[test]
fn indexed_inner_join_zero_d2h_on_success_path() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let left = buf_u32x1(&provider, "lval", &[1, 2, 3, 4]);
    let right = buf_u32x1(&provider, "rval", &[2, 4]);
    let index = provider.build_join_index_v2(&right, &[0]).unwrap();

    prime_row_count_cache(&provider, &left);
    prime_row_count_cache(&provider, &right);
    reset_counters(&provider);

    let result = provider
        .hash_join_v2_with_index(&left, &right, &[0], &[0], JoinType::Inner, &index, None)
        .expect("indexed inner join must succeed");

    assert_join_path_clean(&provider, "indexed_inner_join_v2");
    let v = provider.download_column::<u32>(&result, 0).unwrap();
    let mut got = v;
    got.sort_unstable();
    assert_eq!(got, vec![2u32, 4]);
}

#[test]
fn left_outer_join_zero_d2h_on_success_path() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let left = buf_u32x1(&provider, "lval", &[1, 2, 3]);
    let right = buf_u32x1(&provider, "rval", &[2]);

    prime_row_count_cache(&provider, &left);
    prime_row_count_cache(&provider, &right);
    reset_counters(&provider);

    let result = provider
        .hash_join_v2(&left, &right, &[0], &[0], JoinType::LeftOuter)
        .expect("left-outer join must succeed");

    assert_join_path_clean(&provider, "left_outer_join_v2");
    assert_eq!(result.arity(), 2);
    let l = provider.download_column::<u32>(&result, 0).unwrap();
    let r = provider.download_column::<u32>(&result, 1).unwrap();
    let mut pairs: Vec<(u32, u32)> = l.into_iter().zip(r).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1u32, 0), (2, 2), (3, 0)]);
}

/// Error-path test for the deferred GPU overflow status. Forces the
/// inner-join allocation cap below the actual match count via a tiny
/// `max_output`, then consumes the overflow flag explicitly.
///
/// Counter expectations on the error path:
///   * `deterministic_d2h_violation_count() == 0` (no gate violation —
///     overflow stays device-resident until consumed).
///   * `control_plane_d2h_read_count() == 0` (the materialize path
///     issues no host scalar reads even on overflow).
///   * `sanctioned_status_read_count() == 1` (one explicit consume of
///     the deferred flag, named and counted).
#[test]
fn inner_join_overflow_status_is_deferred_and_named() {
    let Some(provider) = setup_provider() else {
        return;
    };
    // Cartesian-heavy match space: every left row matches every right
    // row. Pre-cap the output to 1 row so the join definitely
    // overflows the allocated capacity.
    let left = buf_u32x1(&provider, "lval", &[1u32, 1, 1, 1]);
    let right = buf_u32x1(&provider, "rval", &[1u32, 1, 1, 1]);

    prime_row_count_cache(&provider, &left);
    prime_row_count_cache(&provider, &right);
    reset_counters(&provider);

    let result = provider
        .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Inner, Some(1))
        .expect("inner join with capped max_output must succeed");

    // Counters before consuming the deferred flag: zero on every axis.
    assert_eq!(provider.deterministic_d2h_violation_count(), 0);
    assert_eq!(provider.control_plane_d2h_read_count(), 0);
    assert_eq!(provider.sanctioned_status_read_count(), 0);

    // Consume the flag. This is the named/counted sanctioned status read.
    let overflowed = provider
        .try_consume_overflow_status(&result)
        .expect("consume overflow status");
    assert!(overflowed, "expected overflow flag set on capped join");
    assert_eq!(
        provider.sanctioned_status_read_count(),
        1,
        "consume must increment sanctioned_status_reads by exactly 1"
    );
    // Control-plane and gate counters unaffected by the sanctioned read.
    assert_eq!(provider.deterministic_d2h_violation_count(), 0);
    assert_eq!(provider.control_plane_d2h_read_count(), 0);
}

#[test]
fn indexed_left_outer_join_zero_d2h_on_success_path() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let left = buf_u32x1(&provider, "lval", &[1, 2, 3]);
    let right = buf_u32x1(&provider, "rval", &[2]);
    let index = provider.build_join_index_v2(&right, &[0]).unwrap();

    prime_row_count_cache(&provider, &left);
    prime_row_count_cache(&provider, &right);
    reset_counters(&provider);

    let result = provider
        .hash_join_v2_with_index(&left, &right, &[0], &[0], JoinType::LeftOuter, &index, None)
        .expect("indexed left-outer join must succeed");

    assert_join_path_clean(&provider, "indexed_left_outer_join_v2");
    assert_eq!(result.arity(), 2);
    let l = provider.download_column::<u32>(&result, 0).unwrap();
    let r = provider.download_column::<u32>(&result, 1).unwrap();
    let mut pairs: Vec<(u32, u32)> = l.into_iter().zip(r).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1u32, 0), (2, 2), (3, 0)]);
}
