// crates/xlog-cuda/tests/test_deterministic_d2h_gate.rs
//! Tests for the strict deterministic-Datalog D2H gate.
//!
//! The gate is the v0.5.5 regression-detection scaffold for the larger
//! deterministic-hardening work (GPU-native multi-column dedup/diff,
//! deterministic count-prefix-materialize binary join). It is opt-in by
//! design and these tests verify two contracts:
//!
//! 1. Public column-download paths (`download_column`,
//!    `download_column_untracked`) trip the gate before any host buffer
//!    is allocated.
//! 2. The chokepoint inside `dtoh_sync_copy_into_tracked` trips the gate
//!    even when reached through an internal relational op (`diff`), so
//!    paths that bypass the column API still cannot regress silently.

mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema};

#[test]
fn gate_starts_disabled_with_zero_violations() {
    let Some(provider) = setup_provider() else {
        return;
    };
    assert!(!provider.strict_deterministic_d2h_enabled());
    assert_eq!(provider.deterministic_d2h_violation_count(), 0);
}

#[test]
fn enable_disable_round_trip() {
    let Some(provider) = setup_provider() else {
        return;
    };
    provider.enable_strict_deterministic_d2h();
    assert!(provider.strict_deterministic_d2h_enabled());
    provider.disable_strict_deterministic_d2h();
    assert!(!provider.strict_deterministic_d2h_enabled());
}

#[test]
fn download_column_trips_gate() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let buf = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2, 3, 4]], schema)
        .unwrap();

    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    let res = provider.download_column::<u32>(&buf, 0);
    provider.disable_strict_deterministic_d2h();

    assert!(res.is_err(), "download_column must error while gate is on");
    assert_eq!(
        provider.deterministic_d2h_violation_count(),
        1,
        "download_column must record exactly one violation"
    );
}

#[test]
fn download_column_untracked_trips_gate() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let buf = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2, 3]], schema)
        .unwrap();

    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    let res = provider.download_column_untracked::<u32>(&buf, 0);
    provider.disable_strict_deterministic_d2h();

    assert!(
        res.is_err(),
        "download_column_untracked must error while gate is on"
    );
    assert_eq!(provider.deterministic_d2h_violation_count(), 1);
}

#[test]
fn empty_download_does_not_trip_gate() {
    let Some(provider) = setup_provider() else {
        return;
    };
    // Empty buffer: download issues no D2H copy and must not violate.
    let schema = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let buf = provider
        .create_buffer_from_u32_columns(&[&[] as &[u32]], schema)
        .unwrap();

    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    let res = provider.download_column::<u32>(&buf, 0);
    provider.disable_strict_deterministic_d2h();

    assert!(res.is_ok(), "empty download must succeed");
    assert_eq!(res.unwrap().len(), 0);
    assert_eq!(provider.deterministic_d2h_violation_count(), 0);
}

#[test]
fn diff_trips_gate_via_chokepoint() {
    // `diff` is one of the relational ops that still falls back to host-side
    // set algebra (replacement is the next PR in the v0.5.5 chain). Today
    // it issues `dtoh_sync_copy_into_tracked` calls for both inputs, so it
    // is the canonical exercise for the chokepoint.
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema_a = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let schema_b = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let a = provider
        .create_buffer_from_u32_columns(&[&[1u32, 2, 3, 4]], schema_a)
        .unwrap();
    let b = provider
        .create_buffer_from_u32_columns(&[&[2u32, 4]], schema_b)
        .unwrap();

    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    let res = provider.diff(&a, &b);
    provider.disable_strict_deterministic_d2h();

    assert!(res.is_err(), "diff must error while gate is on");
    assert!(
        provider.deterministic_d2h_violation_count() >= 1,
        "diff must record at least one violation, got {}",
        provider.deterministic_d2h_violation_count()
    );
}

#[test]
fn metadata_reads_remain_allowed() {
    // `dtoh_scalar_untracked` is the explicit metadata escape hatch and
    // must continue to work with the gate engaged.
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let buf = provider
        .create_buffer_from_u32_columns(&[&[10u32, 11, 12]], schema)
        .unwrap();

    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    let n = provider.dtoh_scalar_untracked::<u32>(buf.num_rows_device(), 0);
    provider.disable_strict_deterministic_d2h();

    assert_eq!(n.unwrap(), 3);
    assert_eq!(provider.deterministic_d2h_violation_count(), 0);
}

#[test]
fn gate_recovers_after_violation() {
    // After the gate fires, disabling it must allow normal D2H to resume.
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let buf = provider
        .create_buffer_from_u32_columns(&[&[7u32, 8]], schema)
        .unwrap();

    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    assert!(provider.download_column::<u32>(&buf, 0).is_err());
    provider.disable_strict_deterministic_d2h();

    let v = provider.download_column::<u32>(&buf, 0).unwrap();
    assert_eq!(v, vec![7u32, 8]);
}

/// Targeted strict-gate coverage for `JoinType::LeftOuter` on the
/// non-indexed path (`hash_join_left_outer_impl`).
///
/// The runtime integration test (`*_inner_join_materialize_clean`)
/// exercises only the inner-join path through Datalog rule lowering;
/// `LeftOuter` is an internal IR-level join type not directly reachable
/// from a Datalog rule body. This kernel-level test calls
/// `hash_join_v2(.., JoinType::LeftOuter)` directly with the strict
/// gate enabled, asserting both result correctness and zero
/// `deterministic_d2h_violation_count()`. The indexed sibling path
/// is covered by the next test.
#[test]
fn left_outer_join_strict_gate_clean() {
    let Some(provider) = setup_provider() else {
        return;
    };

    let left_schema = Schema::new(vec![("lval".to_string(), ScalarType::U32)]);
    let right_schema = Schema::new(vec![("rval".to_string(), ScalarType::U32)]);
    let left_buf = provider
        .create_buffer_from_slice::<u32>(&[1u32, 2, 3], left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_slice::<u32>(&[2u32], right_schema)
        .unwrap();

    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    let result = provider.hash_join_v2(
        &left_buf,
        &right_buf,
        &[0],
        &[0],
        xlog_cuda::JoinType::LeftOuter,
    );
    let violations = provider.deterministic_d2h_violation_count();
    provider.disable_strict_deterministic_d2h();

    let result = result.expect("left-outer join must run clean under strict gate");
    assert_eq!(
        violations, 0,
        "left-outer join tripped the gate; output-count reads must be \
         routed through `read_join_output_count_metadata`"
    );

    // Verify shape with the gate disengaged. Left-outer preserves all
    // left rows; the matched row (key=2) carries the right value, the
    // others carry the null sentinel (0).
    assert_eq!(result.arity(), 2);
    let left_vals = provider.download_column::<u32>(&result, 0).unwrap();
    let right_vals = provider.download_column::<u32>(&result, 1).unwrap();
    assert_eq!(left_vals.len(), 3);
    let mut pairs: Vec<(u32, u32)> = left_vals.into_iter().zip(right_vals).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1, 0), (2, 2), (3, 0)]);
}

/// Targeted strict-gate coverage for `JoinType::LeftOuter` on the
/// indexed path (`hash_join_left_outer_indexed`).
///
/// The non-indexed path goes through `hash_join_left_outer_impl`; this
/// test exercises the sibling helper invoked when the caller has built
/// a `JoinIndexV2` on the right buffer and dispatches via
/// `hash_join_v2_with_index`. Without this, the indexed left-outer
/// count path was statically reviewed only.
#[test]
fn left_outer_join_indexed_strict_gate_clean() {
    let Some(provider) = setup_provider() else {
        return;
    };

    let left_schema = Schema::new(vec![("lval".to_string(), ScalarType::U32)]);
    let right_schema = Schema::new(vec![("rval".to_string(), ScalarType::U32)]);
    let left_buf = provider
        .create_buffer_from_slice::<u32>(&[1u32, 2, 3], left_schema)
        .unwrap();
    let right_buf = provider
        .create_buffer_from_slice::<u32>(&[2u32], right_schema)
        .unwrap();

    let index = provider
        .build_join_index_v2(&right_buf, &[0])
        .expect("build join index");

    provider.reset_deterministic_d2h_violations();
    provider.enable_strict_deterministic_d2h();
    let result = provider.hash_join_v2_with_index(
        &left_buf,
        &right_buf,
        &[0],
        &[0],
        xlog_cuda::JoinType::LeftOuter,
        &index,
        None,
    );
    let violations = provider.deterministic_d2h_violation_count();
    provider.disable_strict_deterministic_d2h();

    let result = result.expect("indexed left-outer join must run clean under strict gate");
    assert_eq!(
        violations, 0,
        "indexed left-outer join tripped the gate; the indexed count \
         path must also route output-count reads through metadata"
    );

    // Same expected shape as the non-indexed test: all left rows preserved.
    assert_eq!(result.arity(), 2);
    let left_vals = provider.download_column::<u32>(&result, 0).unwrap();
    let right_vals = provider.download_column::<u32>(&result, 1).unwrap();
    assert_eq!(left_vals.len(), 3);
    let mut pairs: Vec<(u32, u32)> = left_vals.into_iter().zip(right_vals).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1, 0), (2, 2), (3, 0)]);
}
