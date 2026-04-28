// crates/xlog-cuda/tests/test_join_pipeline_invariants.rs
//! Small deterministic kernel-shape tests for the v0.5.5 prototype's
//! count → exclusive-scan → materialize inner-join pipeline.
//!
//! These reproduce the structural shape of the integration tests that
//! flake under parallel execution (social/RBAC: self-join, duplicate
//! keys, multi-match probe rows). They drive `provider.hash_join_v2`
//! at the public API and assert the result row count + content match
//! a host-computed reference, so any off-by-one in per-probe counts,
//! scan offsets, total-from-scan, or per-offset writes shows up as a
//! result-correctness failure rather than a downstream-operator
//! flake.
//!
//! Each test runs the join MULTIPLE times in a tight loop. Any
//! non-determinism in the pipeline (race in atomic count, missing
//! synchronize, scan-tail edge case) shows up as result variance
//! across iterations.

mod common;
use common::setup_provider;
use std::collections::BTreeSet;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaKernelProvider, JoinType};

fn buf_u32x1(provider: &CudaKernelProvider, name: &str, data: &[u32]) -> CudaBuffer {
    let schema = Schema::new(vec![(name.to_string(), ScalarType::U32)]);
    provider
        .create_buffer_from_slice::<u32>(data, schema)
        .unwrap()
}

fn buf_u32x2(provider: &CudaKernelProvider, rows: &[(u32, u32)]) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    let c0: Vec<u32> = rows.iter().map(|(a, _)| *a).collect();
    let c1: Vec<u32> = rows.iter().map(|(_, b)| *b).collect();
    provider
        .create_buffer_from_u32_columns(&[&c0, &c1], schema)
        .unwrap()
}

fn read_join_pairs(provider: &CudaKernelProvider, buf: &CudaBuffer) -> Vec<(u32, u32, u32, u32)> {
    let logical = provider.device_row_count(buf).unwrap();
    let c0 = provider.download_column::<u32>(buf, 0).unwrap();
    let c1 = provider.download_column::<u32>(buf, 1).unwrap();
    let c2 = provider.download_column::<u32>(buf, 2).unwrap();
    let c3 = provider.download_column::<u32>(buf, 3).unwrap();
    (0..logical).map(|i| (c0[i], c1[i], c2[i], c3[i])).collect()
}

/// Self-join `friend(X, Y) ⨝ friend(Y, Z)`: every probe row matches
/// some build rows; the friends graph is symmetric so multiple matches
/// per probe key are expected.
///
/// Reference graph (from `test_social_network_friend_recommendations`):
/// 1<->2, 2<->3, 3<->4, 1<->5, 2<->5, 5<->6.
///
/// Expected `path(X, Y, Y, Z)` (raw inner-join projection): the set of
/// (X, Y, Y', Z) tuples with `Y == Y'` formed by joining each
/// `friend(X, Y)` against `friend(Y, Z)`. This is a deterministic set
/// — neither the order of rows in the input nor the parallel scheduling
/// of GPU threads should change it.
#[test]
fn self_join_on_friend_graph_is_deterministic_across_repeats() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let edges = vec![
        (1u32, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (1, 5),
        (5, 1),
        (2, 5),
        (5, 2),
        (5, 6),
        (6, 5),
    ];
    let buf = buf_u32x2(&provider, &edges);

    // Host-compute the expected (X, Y_l, Y_r, Z) tuples via the same
    // semantic the join produces (inner join, no projection).
    let mut expected: BTreeSet<(u32, u32, u32, u32)> = BTreeSet::new();
    for (xa, ya) in &edges {
        for (yb, zb) in &edges {
            if ya == yb {
                expected.insert((*xa, *ya, *yb, *zb));
            }
        }
    }

    // Run the join repeatedly; if there is non-determinism in the
    // count/scan/materialize pipeline, different runs will produce
    // different answer sets.
    let mut last: Option<BTreeSet<(u32, u32, u32, u32)>> = None;
    for trial in 0..16 {
        let result = provider
            .hash_join_v2(&buf, &buf, &[1], &[0], JoinType::Inner)
            .unwrap_or_else(|e| panic!("self-join trial {} failed: {}", trial, e));
        let pairs = read_join_pairs(&provider, &result);
        let got: BTreeSet<(u32, u32, u32, u32)> = pairs.into_iter().collect();
        assert_eq!(
            got, expected,
            "self-join trial {} produced wrong tuples; got {:?}, expected {:?}",
            trial, got, expected
        );
        if let Some(prev) = &last {
            assert_eq!(
                &got, prev,
                "self-join trial {} differs from previous trial — non-deterministic pipeline",
                trial
            );
        }
        last = Some(got);
    }
}

/// Chained inner joins: A ⨝ B ⨝ C with multi-match probe rows at each
/// stage. Mimics the RBAC effective_role derivation
/// (`user_role(U, R) ⨝ inherits(R, R')` to expand transitive roles).
///
/// Inputs:
///   * A = `{(10, 1), (10, 2), (20, 2), (30, 3)}` (user → role)
///   * B = `{(1, 1), (1, 100), (2, 2), (2, 100), (3, 3)}` (role → super_role)
///
/// First join: A ⨝ B on A[1] == B[0]. Second join: result ⨝ A on
/// result[3] == A[0] (inverse lookup). Each stage produces multi-match
/// outputs whose row counts depend on duplicates.
#[test]
fn chained_inner_join_multi_match_is_deterministic() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let a = buf_u32x2(&provider, &[(10, 1), (10, 2), (20, 2), (30, 3)]);
    let b = buf_u32x2(&provider, &[(1, 1), (1, 100), (2, 2), (2, 100), (3, 3)]);

    // Host-compute first-join expected:
    //   for (u, r) in A, for (r2, sr) in B, if r == r2: emit (u, r, r2, sr).
    let a_pairs: Vec<(u32, u32)> = vec![(10, 1), (10, 2), (20, 2), (30, 3)];
    let b_pairs: Vec<(u32, u32)> = vec![(1, 1), (1, 100), (2, 2), (2, 100), (3, 3)];
    let mut first_expected: BTreeSet<(u32, u32, u32, u32)> = BTreeSet::new();
    for (u, r) in &a_pairs {
        for (r2, sr) in &b_pairs {
            if r == r2 {
                first_expected.insert((*u, *r, *r2, *sr));
            }
        }
    }

    // Repeat to expose non-determinism.
    for trial in 0..16 {
        let first = provider
            .hash_join_v2(&a, &b, &[1], &[0], JoinType::Inner)
            .unwrap();
        let first_got: BTreeSet<(u32, u32, u32, u32)> =
            read_join_pairs(&provider, &first).into_iter().collect();
        assert_eq!(
            first_got, first_expected,
            "first-join trial {} produced wrong tuples",
            trial
        );

        // Chain: take col 3 of `first` (the super_role) and join it
        // against A col 1 (role). The point is that `first` has
        // row_capacity > logical_count after my count-scan-materialize
        // pipeline, so feeding it into the second join exercises the
        // chain consumer.
        let first_logical = provider.device_row_count(&first).unwrap();
        assert!(first_logical > 0, "first-join produced empty");

        let second = provider.hash_join_v2(&first, &a, &[3], &[1], JoinType::Inner);
        // We don't compute the exact second-join answer set; we
        // assert that the call succeeds and produces a buffer with
        // logical row count <= the worst-case Cartesian.
        match second {
            Ok(buf) => {
                let logical = provider.device_row_count(&buf).unwrap();
                let cap = buf.num_rows();
                assert!(
                    (logical as u64) <= cap,
                    "trial {}: second-join logical {} > row_cap {}",
                    trial,
                    logical,
                    cap
                );
            }
            Err(e) => panic!("second-join trial {} failed: {}", trial, e),
        }
    }
}

/// Forward-computation shape: a join whose input has duplicate keys
/// and whose result feeds a downstream filter. Reproduces the
/// `test_forward_computation` scenario where the first-join output's
/// `row_cap > logical_count` flows into another operator.
#[test]
fn forward_computation_chain_is_deterministic() {
    let Some(provider) = setup_provider() else {
        return;
    };
    // Mimics step(N, N+1) with base val(0, 1), val(1, 2), val(2, 4), …
    let base = buf_u32x2(&provider, &[(0u32, 1)]);
    let step = buf_u32x2(&provider, &[(0u32, 1), (1, 2), (2, 4), (3, 8), (4, 16)]);

    // Reference: result of base ⨝ step on base[1] = step[0].
    let mut expected: BTreeSet<(u32, u32, u32, u32)> = BTreeSet::new();
    for (a, b) in [(0u32, 1)].iter() {
        for (c, d) in &[(0u32, 1), (1, 2), (2, 4), (3, 8), (4, 16)] {
            if b == c {
                expected.insert((*a, *b, *c, *d));
            }
        }
    }

    for trial in 0..16 {
        let result = provider
            .hash_join_v2(&base, &step, &[1], &[0], JoinType::Inner)
            .unwrap();
        let got: BTreeSet<(u32, u32, u32, u32)> =
            read_join_pairs(&provider, &result).into_iter().collect();
        assert_eq!(
            got, expected,
            "forward-computation trial {} produced wrong tuples; got {:?}, expected {:?}",
            trial, got, expected
        );
    }
}

/// Tiny smoke test: a single probe row matching a single build row.
/// If the count→scan→materialize pipeline has any boundary bug at
/// `num_probe == 1`, this test catches it.
#[test]
fn single_row_probe_single_match() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let probe = buf_u32x1(&provider, "k", &[7u32]);
    let build = buf_u32x1(&provider, "k", &[7u32]);
    for trial in 0..32 {
        let result = provider
            .hash_join_v2(&probe, &build, &[0], &[0], JoinType::Inner)
            .unwrap();
        let logical = provider.device_row_count(&result).unwrap();
        assert_eq!(
            logical, 1,
            "trial {}: single-row join expected 1 match",
            trial
        );
    }
}
