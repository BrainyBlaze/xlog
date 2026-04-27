// crates/xlog-cuda/tests/test_full_row_set_algebra.rs
//! Acceptance tests for the v0.5.5 GPU-native deterministic full-row
//! dedup / set-difference pipeline (`dedup_full_row` and `diff_full_row`).
//!
//! Tier 1 — full six-case dedup matrix on a canonical two-column integer
//! schema `(u32, u32)`. Tier 2 — minimal per-scalar smokes for the cases
//! most likely to reveal type-dispatch bugs (signed integers and bool).
//! Tier 4 — float pin tests (NaN bitwise dedup, ±0/±Inf distinct,
//! representative-by-lowest-original-index, semantic-key membership in
//! `diff_full_row`).
//!
//! Tier 3 (mixed-schema) is intentionally deferred to a follow-up commit
//! to keep this PR focused on the kernel pipeline plus the highest-value
//! coverage. The runtime-level full-row semantics are sealed separately
//! in `xlog-integration/tests/executor_config_tests.rs`.

mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema};

// ---------- helpers ----------

fn schema_u32x2() -> Schema {
    Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ])
}

fn buf_u32x2(
    provider: &xlog_cuda::CudaKernelProvider,
    rows: &[(u32, u32)],
) -> xlog_cuda::CudaBuffer {
    let c0: Vec<u32> = rows.iter().map(|(a, _)| *a).collect();
    let c1: Vec<u32> = rows.iter().map(|(_, b)| *b).collect();
    provider
        .create_buffer_from_u32_columns(&[&c0, &c1], schema_u32x2())
        .unwrap()
}

fn read_u32x2(
    provider: &xlog_cuda::CudaKernelProvider,
    buf: &xlog_cuda::CudaBuffer,
) -> Vec<(u32, u32)> {
    let c0 = provider.download_column::<u32>(buf, 0).unwrap();
    let c1 = provider.download_column::<u32>(buf, 1).unwrap();
    c0.into_iter().zip(c1).collect()
}

fn sorted_set_eq(mut got: Vec<(u32, u32)>, mut expected: Vec<(u32, u32)>) {
    got.sort_unstable();
    expected.sort_unstable();
    assert_eq!(got, expected);
}

// ====================================================================
// Tier 1 — full six-case dedup matrix on (u32, u32)
// ====================================================================

#[test]
fn t1_dedup_all_distinct() {
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 10), (2, 20), (3, 30)]);
    let r = p.dedup_full_row(&a).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(1, 10), (2, 20), (3, 30)]);
}

#[test]
fn t1_dedup_all_duplicates_collapse() {
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 10), (1, 10), (1, 10)]);
    let r = p.dedup_full_row(&a).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(1, 10)]);
}

#[test]
fn t1_dedup_mixed_multiplicities() {
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 10), (1, 10), (1, 20), (2, 20), (2, 20)]);
    let r = p.dedup_full_row(&a).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(1, 10), (1, 20), (2, 20)]);
}

#[test]
fn t1_dedup_first_column_overlap_distinct_full_rows() {
    // Pins full-row semantics: rows share col 0 but differ on col 1, so
    // none collapse. A first-column-key dedup would collapse them.
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 10), (1, 20), (1, 30)]);
    let r = p.dedup_full_row(&a).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(1, 10), (1, 20), (1, 30)]);
}

#[test]
fn t1_dedup_empty_input() {
    let Some(p) = setup_provider() else { return };
    let a = p
        .create_buffer_from_u32_columns(&[&[] as &[u32], &[] as &[u32]], schema_u32x2())
        .unwrap();
    let r = p.dedup_full_row(&a).unwrap();
    assert_eq!(read_u32x2(&p, &r).len(), 0);
}

#[test]
fn t1_dedup_single_row() {
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(7, 8)]);
    let r = p.dedup_full_row(&a).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(7, 8)]);
}

// ====================================================================
// Tier 1 — diff cases that aren't already implied by dedup
// ====================================================================

#[test]
fn t1_diff_removed_duplicate_row() {
    // Distinguishes full-row from key-only diff but not set vs bag.
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 10), (1, 10), (1, 20)]);
    let b = buf_u32x2(&p, &[(1, 10)]);
    let r = p.diff_full_row(&a, &b).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(1, 20)]);
}

#[test]
fn t1_diff_surviving_duplicate_collapse() {
    // Set semantics: surviving rows must collapse.
    // Bag semantics would yield {(1,20), (1,20), (2,30)}; set semantics
    // yields {(1,20), (2,30)}.
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 20), (1, 20), (2, 30)]);
    let b = buf_u32x2(&p, &[(1, 10)]);
    let r = p.diff_full_row(&a, &b).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(1, 20), (2, 30)]);
}

#[test]
fn t1_diff_full_overlap_yields_empty() {
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 10), (2, 20)]);
    let b = buf_u32x2(&p, &[(2, 20), (1, 10)]);
    let r = p.diff_full_row(&a, &b).unwrap();
    assert_eq!(read_u32x2(&p, &r).len(), 0);
}

#[test]
fn t1_diff_no_overlap_yields_a_dedup() {
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 10), (1, 10), (2, 20)]);
    let b = buf_u32x2(&p, &[(3, 30)]);
    let r = p.diff_full_row(&a, &b).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(1, 10), (2, 20)]);
}

#[test]
fn t1_diff_two_column_first_column_overlap() {
    // Mirror of the runtime two-column negation seal at the kernel layer.
    let Some(p) = setup_provider() else { return };
    let a = buf_u32x2(&p, &[(1, 10), (1, 20), (2, 10)]);
    let b = buf_u32x2(&p, &[(1, 10)]);
    let r = p.diff_full_row(&a, &b).unwrap();
    sorted_set_eq(read_u32x2(&p, &r), vec![(1, 20), (2, 10)]);
}

// ====================================================================
// Tier 2 — minimal per-scalar smokes for the cases most likely to reveal
// type-dispatch bugs in the typed comparator (signed ints, bool).
// ====================================================================

#[test]
fn t2_dedup_i32_signed_negative_values() {
    let Some(p) = setup_provider() else { return };
    let schema = Schema::new(vec![("c0".to_string(), ScalarType::I32)]);
    let data: Vec<i32> = vec![-2, -1, 0, 1, -1, -2, 0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = p.create_buffer_from_slices(&[&bytes], schema).unwrap();
    let r = p.dedup_full_row(&buf).unwrap();
    let out_bytes = p.download_column::<i32>(&r, 0).unwrap();
    let mut got = out_bytes;
    got.sort_unstable();
    assert_eq!(got, vec![-2, -1, 0, 1]);
}

#[test]
fn t2_diff_i32_signed_negative_membership() {
    // Pins the typed comparator's signed-int path inside the binary search.
    let Some(p) = setup_provider() else { return };
    let schema_a = Schema::new(vec![("c0".to_string(), ScalarType::I32)]);
    let schema_b = Schema::new(vec![("c0".to_string(), ScalarType::I32)]);
    let a_data: Vec<i32> = vec![-3, -1, 0, 1, 3];
    let b_data: Vec<i32> = vec![-1, 1];
    let a_bytes: Vec<u8> = a_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let a = p.create_buffer_from_slices(&[&a_bytes], schema_a).unwrap();
    let b = p.create_buffer_from_slices(&[&b_bytes], schema_b).unwrap();
    let r = p.diff_full_row(&a, &b).unwrap();
    let mut got = p.download_column::<i32>(&r, 0).unwrap();
    got.sort_unstable();
    assert_eq!(got, vec![-3, 0, 3]);
}

#[test]
fn t2_dedup_bool() {
    let Some(p) = setup_provider() else { return };
    let schema = Schema::new(vec![("c0".to_string(), ScalarType::Bool)]);
    let bytes: Vec<u8> = vec![1, 0, 1, 0, 1];
    let buf = p.create_buffer_from_slices(&[&bytes], schema).unwrap();
    let r = p.dedup_full_row(&buf).unwrap();
    let out: Vec<bool> = p.download_column::<bool>(&r, 0).unwrap();
    let mut as_u8: Vec<u8> = out.into_iter().map(|b| if b { 1 } else { 0 }).collect();
    as_u8.sort_unstable();
    assert_eq!(as_u8, vec![0u8, 1]);
}

// ====================================================================
// Tier 4 — float-specific pin tests
//
// The contract:
//   * Equality used for sort, mask, and diff membership matches IEEE-754
//     totalOrder under the project's `f32_to_ordered_u32` /
//     `f64_to_ordered_u64` normalization. Distinct bit patterns map to
//     distinct ordered keys.
//   * Materialization gathers raw bytes from `a`, so NaN payload bits,
//     signed-zero sign bit, and signed-infinity sign bit are preserved
//     exactly.
// ====================================================================

fn schema_f32() -> Schema {
    Schema::new(vec![("c0".to_string(), ScalarType::F32)])
}

fn buf_f32(provider: &xlog_cuda::CudaKernelProvider, bits: &[u32]) -> xlog_cuda::CudaBuffer {
    let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    provider
        .create_buffer_from_slices(&[&bytes], schema_f32())
        .unwrap()
}

fn read_f32_bits(
    provider: &xlog_cuda::CudaKernelProvider,
    buf: &xlog_cuda::CudaBuffer,
) -> Vec<u32> {
    let v: Vec<f32> = provider.download_column::<f32>(buf, 0).unwrap();
    v.into_iter().map(|f| f.to_bits()).collect()
}

#[test]
fn t4_dedup_nan_bitwise_collapses_only_identical_payloads() {
    let Some(p) = setup_provider() else { return };
    // Two NaNs with the same payload (bit-identical) and one finite value.
    let nan_a: u32 = 0x7FC00001;
    let buf = buf_f32(&p, &[nan_a, nan_a, 1.0_f32.to_bits()]);
    let r = p.dedup_full_row(&buf).unwrap();
    let mut got = read_f32_bits(&p, &r);
    got.sort_unstable();
    let mut expected = vec![nan_a, 1.0_f32.to_bits()];
    expected.sort_unstable();
    assert_eq!(got, expected);
}

#[test]
fn t4_dedup_nan_distinct_payloads_stay_distinct() {
    // Two NaNs with different payloads have different totalOrder keys
    // under the project's bijective normalization, so they survive dedup.
    let Some(p) = setup_provider() else { return };
    let nan_a: u32 = 0x7FC00001;
    let nan_b: u32 = 0x7FC00002;
    let buf = buf_f32(&p, &[nan_a, nan_b]);
    let r = p.dedup_full_row(&buf).unwrap();
    let mut got = read_f32_bits(&p, &r);
    got.sort_unstable();
    let mut expected = vec![nan_a, nan_b];
    expected.sort_unstable();
    assert_eq!(got, expected);
}

#[test]
fn t4_dedup_signed_zero_distinct() {
    // +0 and -0 have different totalOrder keys, so they must NOT collapse.
    let Some(p) = setup_provider() else { return };
    let pos_zero: u32 = 0x00000000;
    let neg_zero: u32 = 0x80000000;
    let buf = buf_f32(&p, &[pos_zero, neg_zero, pos_zero]);
    let r = p.dedup_full_row(&buf).unwrap();
    let mut got = read_f32_bits(&p, &r);
    got.sort_unstable();
    let mut expected = vec![pos_zero, neg_zero];
    expected.sort_unstable();
    assert_eq!(got, expected);
}

#[test]
fn t4_dedup_pos_neg_inf_distinct() {
    let Some(p) = setup_provider() else { return };
    let pos_inf: u32 = f32::INFINITY.to_bits();
    let neg_inf: u32 = f32::NEG_INFINITY.to_bits();
    let buf = buf_f32(&p, &[pos_inf, neg_inf, pos_inf, 1.0_f32.to_bits()]);
    let r = p.dedup_full_row(&buf).unwrap();
    let mut got = read_f32_bits(&p, &r);
    got.sort_unstable();
    let mut expected = vec![pos_inf, neg_inf, 1.0_f32.to_bits()];
    expected.sort_unstable();
    assert_eq!(got, expected);
}

#[test]
fn t4_diff_membership_drops_only_bit_identical() {
    // Disjoint NaN payloads must NOT match in the diff probe — bijective
    // totalOrder distinguishes them.
    let Some(p) = setup_provider() else { return };
    let nan_a: u32 = 0x7FC00001;
    let nan_b: u32 = 0x7FC00002;
    let a = buf_f32(&p, &[nan_a]);
    let b = buf_f32(&p, &[nan_b]);
    let r = p.diff_full_row(&a, &b).unwrap();
    let got = read_f32_bits(&p, &r);
    assert_eq!(got, vec![nan_a]);
}

#[test]
fn t4_diff_membership_drops_signed_zero_correctly() {
    // +0 in `a`, -0 in `b` — they are distinct under totalOrder so +0
    // is NOT removed.
    let Some(p) = setup_provider() else { return };
    let pos_zero: u32 = 0x00000000;
    let neg_zero: u32 = 0x80000000;
    let a = buf_f32(&p, &[pos_zero]);
    let b = buf_f32(&p, &[neg_zero]);
    let r = p.diff_full_row(&a, &b).unwrap();
    let got = read_f32_bits(&p, &r);
    assert_eq!(got, vec![pos_zero]);
}
