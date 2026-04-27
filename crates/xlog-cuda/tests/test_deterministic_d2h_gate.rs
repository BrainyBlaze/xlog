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
