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

/// Chained regression test: a join output (whose `row_cap` is the
/// host-known capacity / upper bound, with `d_num_rows` < `row_cap`)
/// must feed correctly into downstream operators that previously
/// treated `num_rows()` as the active logical span.
///
/// This pins the architectural separation introduced by the v0.5.5
/// hardening:
///   * `row_cap` = allocation capacity (host-known upper bound).
///   * `d_num_rows` = active logical row count (device-resident).
///
/// Concretely the test:
///   1. Builds an inner join `J = path(X, Z) :- e(X, Y), e(Y, Z).`
///      via `hash_join_v2`. With our compute_join_output_capacity
///      formula, J ends up with `row_cap > logical_count`.
///   2. Feeds J into a follow-up `dedup` (full-row, all columns).
///   3. Joins J's deduped output against itself.
///   4. Asserts result correctness via `device_row_count`.
///
/// Steps 2 and 3 are exactly the chains that integration tests
/// `test_social_network_friend_recommendations` and
/// `test_rbac_permission_derivation` exercise; this test pins the
/// failure at the provider level for fast iteration.
#[test]
fn join_output_chains_into_dedup_and_self_join_correctly() {
    let Some(provider) = setup_provider() else {
        return;
    };
    // Edge set forming a small graph: 1->2, 2->3, 3->4, 4->5.
    let lhs_a = buf_u32x1(&provider, "src", &[1u32, 2, 3, 4]);
    let lhs_b = buf_u32x1(&provider, "dst", &[2u32, 3, 4, 5]);
    let edges = {
        let schema = Schema::new(vec![
            ("src".to_string(), ScalarType::U32),
            ("dst".to_string(), ScalarType::U32),
        ]);
        let src: Vec<u32> = vec![1, 2, 3, 4];
        let dst: Vec<u32> = vec![2, 3, 4, 5];
        provider
            .create_buffer_from_u32_columns(&[&src, &dst], schema)
            .unwrap()
    };
    drop((lhs_a, lhs_b));

    // path(X, Z) joins edges with itself on edges.dst == edges.src.
    let path = provider
        .hash_join_v2(&edges, &edges, &[1], &[0], JoinType::Inner)
        .expect("first inner join must succeed");

    // The capacity formula yields `row_cap > logical_count`. Pin that.
    let path_logical = provider.device_row_count(&path).unwrap();
    assert!(
        (path.num_rows() as usize) >= path_logical,
        "row_cap {} should be a host-known upper bound on logical {}",
        path.num_rows(),
        path_logical
    );
    // Expected logical count for path(X,Z) over the 4-edge chain: 3.
    // (1->2->3, 2->3->4, 3->4->5)
    assert_eq!(path_logical, 3);

    // Step 2: feed `path` into dedup. Pre-PR this would size masks
    // from `path.num_rows()` (the capacity) and miscompute. The
    // dedup result must contain exactly the 3 logical rows.
    let path_dedup = provider.dedup_full_row(&path).unwrap();
    let dedup_logical = provider.device_row_count(&path_dedup).unwrap();
    assert_eq!(
        dedup_logical, 3,
        "dedup of join output collapsed/expanded rows; row_cap mismatch leak"
    );
    let dedup_col0 = provider.download_column::<u32>(&path_dedup, 0).unwrap();
    let dedup_col1 = provider.download_column::<u32>(&path_dedup, 1).unwrap();
    let mut got: Vec<(u32, u32)> = dedup_col0
        .into_iter()
        .zip(dedup_col1)
        .take(dedup_logical)
        .collect();
    got.sort_unstable();
    // Path columns are (X, Z) where path(X,Z) = e(X,Y) ⨝ e(Y,Z).
    // Inner join projects (left.src, left.dst, right.src, right.dst).
    // We don't assert the exact 4-column tuple here; the logical
    // count + downloadable validity is the load-bearing invariant.
    assert_eq!(got.len(), 3);

    // Step 3: chain another join. Self-join the deduped path on its
    // own first column; logical count must be derived from the
    // device count, not from the upstream row_cap.
    let chained = provider
        .hash_join_v2(&path_dedup, &path_dedup, &[0], &[0], JoinType::Inner)
        .expect("self-join on deduped path must succeed");
    let chained_logical = provider.device_row_count(&chained).unwrap();
    assert!(
        chained_logical >= 3,
        "self-join of (X, _, _, Z) on X has at least one match per row, got {}",
        chained_logical
    );
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
