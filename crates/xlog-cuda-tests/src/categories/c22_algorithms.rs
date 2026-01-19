//! Category 22: Algorithm-specific edge cases
//!
//! Tests specific algorithm behaviors with edge cases including sort variants,
//! join patterns, dedup edge cases, and groupby scenarios.

use crate::harness::{CategoryResult, TestResult, TestContext};
use crate::harness::xgcf;
use std::collections::HashSet;
use std::time::Instant;
use xlog_core::{AggOp, Schema, ScalarType};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c22_algorithms");
    let start = Instant::now();

    results.add_result(test_sort_all_equal(ctx));
    results.add_result(test_sort_already_sorted(ctx));
    results.add_result(test_sort_reverse_sorted(ctx));
    results.add_result(test_join_no_matches(ctx));
    results.add_result(test_join_all_matches(ctx));
    results.add_result(test_join_high_cardinality(ctx));
    results.add_result(test_dedup_all_unique(ctx));
    results.add_result(test_dedup_all_same(ctx));
    results.add_result(test_groupby_single_group(ctx));
    results.add_result(test_groupby_all_unique_keys(ctx));
    results.add_result(test_mc_sample_matches_cpu_reference(ctx));
    results.add_result(test_xgcf_forward_tiny_matches_expected(ctx));
    results.add_result(test_xgcf_backward_tiny_matches_expected(ctx));

    results.set_duration(start.elapsed());
    results
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn mc_sample_cpu_reference(probs: &[f32], num_samples: usize, seed: u64) -> Vec<u8> {
    let num_vars = probs.len();
    let total = num_vars * num_samples;
    let mut out = vec![0u8; total];
    if total == 0 {
        return out;
    }

    let stride = 0x9e3779b97f4a7c15u64;
    let inv_u32: f32 = 1.0f32 / 4294967296.0f32;

    for tid in 0..(total as u64) {
        let var_idx = (tid % (num_vars as u64)) as usize;
        let p = probs[var_idx];
        let p_clamped = if p <= 0.0 {
            0.0
        } else if p >= 1.0 {
            1.0
        } else {
            p
        };

        let x = splitmix64(seed ^ tid.wrapping_mul(stride));
        let r = (x >> 32) as u32;
        let u = (r as f32 + 0.5f32) * inv_u32;
        out[tid as usize] = if u < p_clamped { 1 } else { 0 };
    }
    out
}

/// Test 11: MC sampling kernel produces expected bit patterns for a fixed seed.
fn test_mc_sample_matches_cpu_reference(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let probs: Vec<f32> = vec![0.0, 1.0, 0.25, 0.75];
    let num_samples = 256usize;
    let seed = 123456789u64;

    let expected = mc_sample_cpu_reference(&probs, num_samples, seed);
    let got = match ctx.provider.sample_bernoulli_matrix(&probs, num_samples, seed) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_mc_sample_matches_cpu_reference",
                start.elapsed(),
                format!("sample_bernoulli_matrix failed: {}", e),
            )
        }
    };

    if got != expected {
        // Provide a small diagnostic sample.
        let mut first_mismatch = None;
        for (i, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
            if a != b {
                first_mismatch = Some((i, a, b));
                break;
            }
        }
        let diag = if let Some((i, a, b)) = first_mismatch {
            format!("first mismatch at idx {}: got {}, expected {}", i, a, b)
        } else {
            "unknown mismatch".to_string()
        };

        return TestResult::error(
            "test_mc_sample_matches_cpu_reference",
            start.elapsed(),
            format!(
                "MC sampling output differed from CPU reference (len={}): {}",
                got.len(),
                diag
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_mc_sample_matches_cpu_reference",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_mc_sample_matches_cpu_reference", start.elapsed())
}

/// Test 12: XGCF forward kernel matches CPU expectations on a tiny circuit.
fn test_xgcf_forward_tiny_matches_expected(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = xgcf::tiny_xgcf_spec();
    let values = match xgcf::run_tiny_xgcf_forward(ctx, &spec) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_xgcf_forward_tiny_matches_expected",
                start.elapsed(),
                format!("xgcf forward failed: {}", e),
            )
        }
    };

    if values.len() != spec.expected_values.len() {
        return TestResult::error(
            "test_xgcf_forward_tiny_matches_expected",
            start.elapsed(),
            format!(
                "Unexpected values length: got {}, expected {}",
                values.len(),
                spec.expected_values.len()
            ),
        );
    }

    const EPS: f64 = 1e-9;
    for (i, (&got, &expected)) in values.iter().zip(spec.expected_values.iter()).enumerate() {
        if expected.is_infinite() && expected.is_sign_negative() {
            if !(got.is_infinite() && got.is_sign_negative()) {
                return TestResult::error(
                    "test_xgcf_forward_tiny_matches_expected",
                    start.elapsed(),
                    format!("node {}: expected -inf, got {}", i, got),
                );
            }
            continue;
        }
        let diff = (got - expected).abs();
        if diff > EPS || got.is_nan() || expected.is_nan() {
            return TestResult::error(
                "test_xgcf_forward_tiny_matches_expected",
                start.elapsed(),
                format!(
                    "node {}: got {}, expected {} (|diff|={})",
                    i, got, expected, diff
                ),
            );
        }
    }

    TestResult::passed("test_xgcf_forward_tiny_matches_expected", start.elapsed())
}

/// Test 13: XGCF backward kernels produce expected gradients on a tiny circuit.
fn test_xgcf_backward_tiny_matches_expected(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let spec = xgcf::tiny_xgcf_spec();
    let run = match xgcf::run_tiny_xgcf_backward(ctx, &spec) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_xgcf_backward_tiny_matches_expected",
                start.elapsed(),
                format!("xgcf backward failed: {}", e),
            )
        }
    };

    const EPS: f64 = 1e-9;

    if run.values.len() != spec.expected_values.len() {
        return TestResult::error(
            "test_xgcf_backward_tiny_matches_expected",
            start.elapsed(),
            format!(
                "Unexpected values length: got {}, expected {}",
                run.values.len(),
                spec.expected_values.len()
            ),
        );
    }
    for (i, (&got, &expected)) in run.values.iter().zip(spec.expected_values.iter()).enumerate() {
        if expected.is_infinite() && expected.is_sign_negative() {
            if !(got.is_infinite() && got.is_sign_negative()) {
                return TestResult::error(
                    "test_xgcf_backward_tiny_matches_expected",
                    start.elapsed(),
                    format!("node {}: expected -inf, got {}", i, got),
                );
            }
            continue;
        }
        let diff = (got - expected).abs();
        if diff > EPS || got.is_nan() || expected.is_nan() {
            return TestResult::error(
                "test_xgcf_backward_tiny_matches_expected",
                start.elapsed(),
                format!(
                    "node {}: got {}, expected {} (|diff|={})",
                    i, got, expected, diff
                ),
            );
        }
    }

    if run.grad_true.len() != spec.expected_grad_true.len()
        || run.grad_false.len() != spec.expected_grad_false.len()
    {
        return TestResult::error(
            "test_xgcf_backward_tiny_matches_expected",
            start.elapsed(),
            format!(
                "Unexpected gradient lengths: got true/false {}/{} expected {}/{}",
                run.grad_true.len(),
                run.grad_false.len(),
                spec.expected_grad_true.len(),
                spec.expected_grad_false.len()
            ),
        );
    }

    for (i, (&got, &expected)) in run
        .grad_true
        .iter()
        .zip(spec.expected_grad_true.iter())
        .enumerate()
    {
        let diff = (got - expected).abs();
        if diff > EPS || got.is_nan() || expected.is_nan() {
            return TestResult::error(
                "test_xgcf_backward_tiny_matches_expected",
                start.elapsed(),
                format!(
                    "grad_true[{}]: got {}, expected {} (|diff|={})",
                    i, got, expected, diff
                ),
            );
        }
    }
    for (i, (&got, &expected)) in run
        .grad_false
        .iter()
        .zip(spec.expected_grad_false.iter())
        .enumerate()
    {
        let diff = (got - expected).abs();
        if diff > EPS || got.is_nan() || expected.is_nan() {
            return TestResult::error(
                "test_xgcf_backward_tiny_matches_expected",
                start.elapsed(),
                format!(
                    "grad_false[{}]: got {}, expected {} (|diff|={})",
                    i, got, expected, diff
                ),
            );
        }
    }

    TestResult::passed("test_xgcf_backward_tiny_matches_expected", start.elapsed())
}

/// Test 1: Sort when all values are equal.
///
/// Verifies that sorting a buffer where all values are identical produces
/// correct results with all values preserved.
fn test_sort_all_equal(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;
    const VALUE: u32 = 42;

    // Create data where all values are the same
    let data: Vec<u32> = vec![VALUE; SIZE];

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_all_equal",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sort_all_equal",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Verify row count
    if sorted.num_rows != SIZE as u64 {
        return TestResult::error(
            "test_sort_all_equal",
            start.elapsed(),
            format!("Sort returned {} rows, expected {}", sorted.num_rows, SIZE),
        );
    }

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_all_equal",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify all values are preserved and equal
    for (i, &val) in sorted_data.iter().enumerate() {
        if val != VALUE {
            return TestResult::error(
                "test_sort_all_equal",
                start.elapsed(),
                format!("Value at index {} is {}, expected {}", i, val, VALUE),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sort_all_equal",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sort_all_equal", start.elapsed())
}

/// Test 2: Sort when input is already sorted.
///
/// Verifies that sorting already-sorted data maintains the order and doesn't
/// corrupt the data.
fn test_sort_already_sorted(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;

    // Create already-sorted data
    let data: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_already_sorted",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sort_already_sorted",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_already_sorted",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify data is still sorted and unchanged
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_sort_already_sorted",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    for (i, (&expected, &actual)) in data.iter().zip(sorted_data.iter()).enumerate() {
        if expected != actual {
            return TestResult::error(
                "test_sort_already_sorted",
                start.elapsed(),
                format!(
                    "Value at index {} is {}, expected {}",
                    i, actual, expected
                ),
            );
        }
    }

    // Verify sort order is maintained
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_sort_already_sorted",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} < {}",
                    i, sorted_data[i], sorted_data[i - 1]
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sort_already_sorted",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sort_already_sorted", start.elapsed())
}

/// Test 3: Sort when input is reverse sorted.
///
/// Verifies that sorting reverse-sorted data (worst case for some algorithms)
/// produces correctly sorted output.
fn test_sort_reverse_sorted(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;

    // Create reverse-sorted data
    let data: Vec<u32> = (0..SIZE as u32).rev().collect();

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_reverse_sorted",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sort_reverse_sorted",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_reverse_sorted",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify correct length
    if sorted_data.len() != SIZE {
        return TestResult::error(
            "test_sort_reverse_sorted",
            start.elapsed(),
            format!("Sort returned {} rows, expected {}", sorted_data.len(), SIZE),
        );
    }

    // Verify data is now sorted in ascending order
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_sort_reverse_sorted",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} < {}",
                    i, sorted_data[i], sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify first element is 0 and last is SIZE-1
    if sorted_data[0] != 0 {
        return TestResult::error(
            "test_sort_reverse_sorted",
            start.elapsed(),
            format!("First element should be 0, got {}", sorted_data[0]),
        );
    }

    if sorted_data[SIZE - 1] != (SIZE - 1) as u32 {
        return TestResult::error(
            "test_sort_reverse_sorted",
            start.elapsed(),
            format!(
                "Last element should be {}, got {}",
                SIZE - 1,
                sorted_data[SIZE - 1]
            ),
        );
    }

    // Verify all values are present
    let sorted_set: HashSet<u32> = sorted_data.iter().copied().collect();
    let expected_set: HashSet<u32> = (0..SIZE as u32).collect();
    if sorted_set != expected_set {
        return TestResult::error(
            "test_sort_reverse_sorted",
            start.elapsed(),
            "Some values missing or duplicated after sort".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sort_reverse_sorted",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sort_reverse_sorted", start.elapsed())
}

/// Test 4: Hash join with no matching keys.
///
/// Verifies that a hash join with completely disjoint key sets produces
/// an empty result.
fn test_join_no_matches(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    const SIZE: usize = 1000;

    // Left keys: 0, 1, 2, ..., SIZE-1
    let left_keys: Vec<u32> = (0..SIZE as u32).collect();
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k * 10).collect();

    // Right keys: SIZE, SIZE+1, ..., 2*SIZE-1 (no overlap with left)
    let right_keys: Vec<u32> = (SIZE as u32..(2 * SIZE) as u32).collect();
    let right_vals: Vec<u32> = right_keys.iter().map(|&k| k * 100).collect();

    let left_buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&left_keys, &left_vals],
        left_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_join_no_matches",
                start.elapsed(),
                format!("Failed to create left buffer: {}", e),
            )
        }
    };

    let right_buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&right_keys, &right_vals],
        right_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_join_no_matches",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
            )
        }
    };

    // Perform hash join
    let joined = match ctx.provider.hash_join(&left_buffer, &right_buffer, &[0], &[0]) {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_join_no_matches",
                start.elapsed(),
                format!("Hash join failed: {}", e),
            )
        }
    };

    // Verify result is empty
    if joined.num_rows != 0 {
        return TestResult::error(
            "test_join_no_matches",
            start.elapsed(),
            format!(
                "Join with no matching keys should return 0 rows, got {}",
                joined.num_rows
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_join_no_matches",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_join_no_matches", start.elapsed())
}

/// Test 5: Hash join where every left matches every right (Cartesian-like).
///
/// Verifies that a hash join where all rows have the same key produces
/// a Cartesian product (N*M rows).
fn test_join_all_matches(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    const LEFT_SIZE: usize = 100;
    const RIGHT_SIZE: usize = 50;
    const COMMON_KEY: u32 = 42;

    // All left keys are the same
    let left_keys: Vec<u32> = vec![COMMON_KEY; LEFT_SIZE];
    let left_vals: Vec<u32> = (0..LEFT_SIZE as u32).collect();

    // All right keys are the same (matching left)
    let right_keys: Vec<u32> = vec![COMMON_KEY; RIGHT_SIZE];
    let right_vals: Vec<u32> = (0..RIGHT_SIZE as u32).map(|v| v * 100).collect();

    let left_buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&left_keys, &left_vals],
        left_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_join_all_matches",
                start.elapsed(),
                format!("Failed to create left buffer: {}", e),
            )
        }
    };

    let right_buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&right_keys, &right_vals],
        right_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_join_all_matches",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
            )
        }
    };

    // Perform hash join
    let joined = match ctx.provider.hash_join(&left_buffer, &right_buffer, &[0], &[0]) {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_join_all_matches",
                start.elapsed(),
                format!("Hash join failed: {}", e),
            )
        }
    };

    // Verify Cartesian product size
    let expected_rows = LEFT_SIZE * RIGHT_SIZE;
    if joined.num_rows != expected_rows as u64 {
        return TestResult::error(
            "test_join_all_matches",
            start.elapsed(),
            format!(
                "Join should produce {} rows ({}x{}), got {}",
                expected_rows, LEFT_SIZE, RIGHT_SIZE, joined.num_rows
            ),
        );
    }

    // Download and verify all keys are COMMON_KEY
    let joined_keys = match ctx.provider.download_column_u32(&joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_all_matches",
                start.elapsed(),
                format!("Failed to download joined keys: {}", e),
            )
        }
    };

    for (i, &key) in joined_keys.iter().enumerate() {
        if key != COMMON_KEY {
            return TestResult::error(
                "test_join_all_matches",
                start.elapsed(),
                format!("Key at row {} is {}, expected {}", i, key, COMMON_KEY),
            );
        }
    }

    // Verify we have all combinations of lval and rval
    let joined_lvals = match ctx.provider.download_column_u32(&joined, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_all_matches",
                start.elapsed(),
                format!("Failed to download joined lvals: {}", e),
            )
        }
    };

    let joined_rvals = match ctx.provider.download_column_u32(&joined, 2) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_all_matches",
                start.elapsed(),
                format!("Failed to download joined rvals: {}", e),
            )
        }
    };

    // Collect all (lval, rval) pairs
    let pairs: HashSet<(u32, u32)> = joined_lvals
        .iter()
        .zip(joined_rvals.iter())
        .map(|(&l, &r)| (l, r))
        .collect();

    // Verify all expected combinations exist
    for lval in 0..LEFT_SIZE as u32 {
        for rval_idx in 0..RIGHT_SIZE as u32 {
            let rval = rval_idx * 100;
            if !pairs.contains(&(lval, rval)) {
                return TestResult::error(
                    "test_join_all_matches",
                    start.elapsed(),
                    format!("Missing combination: lval={}, rval={}", lval, rval),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_join_all_matches",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_join_all_matches", start.elapsed())
}

/// Test 6: Join with many distinct keys (high cardinality).
///
/// Verifies that hash join works correctly with high cardinality (many
/// distinct keys) that may stress hash table sizing.
fn test_join_high_cardinality(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    const SIZE: usize = 10000;
    const OVERLAP_START: u32 = 5000;

    // Left keys: 0, 1, 2, ..., SIZE-1
    let left_keys: Vec<u32> = (0..SIZE as u32).collect();
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k * 10).collect();

    // Right keys: OVERLAP_START, OVERLAP_START+1, ..., OVERLAP_START+SIZE-1
    // This creates an overlap of SIZE - OVERLAP_START keys
    let right_keys: Vec<u32> = (OVERLAP_START..(OVERLAP_START + SIZE as u32)).collect();
    let right_vals: Vec<u32> = right_keys.iter().map(|&k| k * 100).collect();

    let left_buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&left_keys, &left_vals],
        left_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!("Failed to create left buffer: {}", e),
            )
        }
    };

    let right_buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&right_keys, &right_vals],
        right_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
            )
        }
    };

    // Perform hash join
    let joined = match ctx.provider.hash_join(&left_buffer, &right_buffer, &[0], &[0]) {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!("Hash join failed: {}", e),
            )
        }
    };

    // Calculate expected overlap: keys from OVERLAP_START to SIZE-1
    let expected_matches = SIZE - OVERLAP_START as usize;
    if joined.num_rows != expected_matches as u64 {
        return TestResult::error(
            "test_join_high_cardinality",
            start.elapsed(),
            format!(
                "Join should produce {} rows, got {}",
                expected_matches, joined.num_rows
            ),
        );
    }

    // Download and verify results
    let joined_keys = match ctx.provider.download_column_u32(&joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!("Failed to download joined keys: {}", e),
            )
        }
    };

    let joined_lvals = match ctx.provider.download_column_u32(&joined, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!("Failed to download joined lvals: {}", e),
            )
        }
    };

    let joined_rvals = match ctx.provider.download_column_u32(&joined, 2) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!("Failed to download joined rvals: {}", e),
            )
        }
    };

    // Verify each joined row is correct
    for i in 0..joined.num_rows as usize {
        let key = joined_keys[i];
        let lval = joined_lvals[i];
        let rval = joined_rvals[i];

        // Key should be in overlap range
        if key < OVERLAP_START || key >= SIZE as u32 {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!("Key {} at row {} is out of overlap range", key, i),
            );
        }

        // Verify lval and rval match key
        let expected_lval = key * 10;
        let expected_rval = key * 100;

        if lval != expected_lval {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!(
                    "Row {}: lval {} doesn't match expected {} for key {}",
                    i, lval, expected_lval, key
                ),
            );
        }

        if rval != expected_rval {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!(
                    "Row {}: rval {} doesn't match expected {} for key {}",
                    i, rval, expected_rval, key
                ),
            );
        }
    }

    // Verify all expected keys are present
    let joined_key_set: HashSet<u32> = joined_keys.iter().copied().collect();
    for key in OVERLAP_START..SIZE as u32 {
        if !joined_key_set.contains(&key) {
            return TestResult::error(
                "test_join_high_cardinality",
                start.elapsed(),
                format!("Missing expected key {} in join result", key),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_join_high_cardinality",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_join_high_cardinality", start.elapsed())
}

/// Test 7: Dedup when all values are unique.
///
/// Verifies that dedup on a buffer where all keys are unique returns all rows.
fn test_dedup_all_unique(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    const SIZE: usize = 5000;

    // All keys are unique
    let keys: Vec<u32> = (0..SIZE as u32).collect();
    let vals: Vec<u32> = keys.iter().map(|&k| k * 10).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_columns(&[&keys, &vals], schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_dedup_all_unique",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Dedup on key column
    let deduped = match ctx.provider.dedup(&buffer, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dedup_all_unique",
                start.elapsed(),
                format!("Dedup failed: {}", e),
            )
        }
    };

    // Should return all rows since all keys are unique
    if deduped.num_rows != SIZE as u64 {
        return TestResult::error(
            "test_dedup_all_unique",
            start.elapsed(),
            format!(
                "Dedup should return all {} rows when all unique, got {}",
                SIZE, deduped.num_rows
            ),
        );
    }

    // Download and verify keys
    let deduped_keys = match ctx.provider.download_column_u32(&deduped, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dedup_all_unique",
                start.elapsed(),
                format!("Failed to download deduped keys: {}", e),
            )
        }
    };

    // Verify all original keys are present
    let deduped_set: HashSet<u32> = deduped_keys.iter().copied().collect();
    let original_set: HashSet<u32> = keys.iter().copied().collect();
    if deduped_set != original_set {
        return TestResult::error(
            "test_dedup_all_unique",
            start.elapsed(),
            "Dedup lost or changed some unique keys".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_dedup_all_unique",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_dedup_all_unique", start.elapsed())
}

/// Test 8: Dedup when all values are the same.
///
/// Verifies that dedup on a buffer where all keys are the same returns
/// exactly one row.
fn test_dedup_all_same(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    const SIZE: usize = 5000;
    const COMMON_KEY: u32 = 42;

    // All keys are the same
    let keys: Vec<u32> = vec![COMMON_KEY; SIZE];
    let vals: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_columns(&[&keys, &vals], schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_dedup_all_same",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Dedup on key column
    let deduped = match ctx.provider.dedup(&buffer, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dedup_all_same",
                start.elapsed(),
                format!("Dedup failed: {}", e),
            )
        }
    };

    // Should return exactly 1 row since all keys are the same
    if deduped.num_rows != 1 {
        return TestResult::error(
            "test_dedup_all_same",
            start.elapsed(),
            format!(
                "Dedup should return 1 row when all keys are same, got {}",
                deduped.num_rows
            ),
        );
    }

    // Download and verify the single key
    let deduped_keys = match ctx.provider.download_column_u32(&deduped, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dedup_all_same",
                start.elapsed(),
                format!("Failed to download deduped keys: {}", e),
            )
        }
    };

    if deduped_keys.len() != 1 || deduped_keys[0] != COMMON_KEY {
        return TestResult::error(
            "test_dedup_all_same",
            start.elapsed(),
            format!(
                "Expected single key {}, got {:?}",
                COMMON_KEY, deduped_keys
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_dedup_all_same",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_dedup_all_same", start.elapsed())
}

/// Test 9: Groupby when all keys are the same (single group).
///
/// Verifies that groupby with a single group aggregates all values correctly.
fn test_groupby_single_group(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    const SIZE: usize = 1000;
    const COMMON_KEY: u32 = 42;

    // All keys are the same
    let keys: Vec<u32> = vec![COMMON_KEY; SIZE];
    // Values: 1, 2, 3, ..., SIZE
    let vals: Vec<u32> = (1..=SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_columns(&[&keys, &vals], schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test Count aggregation
    let count_result = match ctx.provider.groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Count)]) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Groupby count failed: {}", e),
            )
        }
    };

    if count_result.num_rows() != 1 {
        return TestResult::error(
            "test_groupby_single_group",
            start.elapsed(),
            format!(
                "Groupby count should return 1 group, got {}",
                count_result.num_rows()
            ),
        );
    }

    let count_keys = match ctx.provider.download_column_u32(&count_result, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Failed to download count keys: {}", e),
            )
        }
    };

    // Count results are u64 (to match predicate declaration types)
    let counts = match ctx.provider.download_column_u64(&count_result, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Failed to download counts: {}", e),
            )
        }
    };

    if count_keys[0] != COMMON_KEY {
        return TestResult::error(
            "test_groupby_single_group",
            start.elapsed(),
            format!("Count key should be {}, got {}", COMMON_KEY, count_keys[0]),
        );
    }

    if counts[0] != SIZE as u64 {
        return TestResult::error(
            "test_groupby_single_group",
            start.elapsed(),
            format!("Count should be {}, got {}", SIZE, counts[0]),
        );
    }

    // Test Sum aggregation
    let sum_result = match ctx.provider.groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum)]) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Groupby sum failed: {}", e),
            )
        }
    };

    let sums = match ctx.provider.download_column_u64(&sum_result, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Failed to download sums: {}", e),
            )
        }
    };

    // Sum of 1..SIZE = SIZE*(SIZE+1)/2
    let expected_sum = (SIZE as u64) * ((SIZE as u64) + 1) / 2;
    if sums[0] != expected_sum {
        return TestResult::error(
            "test_groupby_single_group",
            start.elapsed(),
            format!("Sum should be {}, got {}", expected_sum, sums[0]),
        );
    }

    // Test Min aggregation
    let min_result = match ctx.provider.groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Min)]) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Groupby min failed: {}", e),
            )
        }
    };

    let mins = match ctx.provider.download_column_u32(&min_result, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Failed to download mins: {}", e),
            )
        }
    };

    if mins[0] != 1 {
        return TestResult::error(
            "test_groupby_single_group",
            start.elapsed(),
            format!("Min should be 1, got {}", mins[0]),
        );
    }

    // Test Max aggregation
    let max_result = match ctx.provider.groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Max)]) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Groupby max failed: {}", e),
            )
        }
    };

    let maxs = match ctx.provider.download_column_u32(&max_result, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_single_group",
                start.elapsed(),
                format!("Failed to download maxs: {}", e),
            )
        }
    };

    if maxs[0] != SIZE as u32 {
        return TestResult::error(
            "test_groupby_single_group",
            start.elapsed(),
            format!("Max should be {}, got {}", SIZE, maxs[0]),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_groupby_single_group",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_groupby_single_group", start.elapsed())
}

/// Test 10: Groupby when every key is unique.
///
/// Verifies that groupby with all unique keys creates one group per row.
fn test_groupby_all_unique_keys(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    const SIZE: usize = 1000;

    // All keys are unique
    let keys: Vec<u32> = (0..SIZE as u32).collect();
    let vals: Vec<u32> = keys.iter().map(|&k| k * 10).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_columns(&[&keys, &vals], schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test Count aggregation - each group should have count=1
    let count_result = match ctx.provider.groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Count)]) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!("Groupby count failed: {}", e),
            )
        }
    };

    if count_result.num_rows() != SIZE as u64 {
        return TestResult::error(
            "test_groupby_all_unique_keys",
            start.elapsed(),
            format!(
                "Groupby count should return {} groups, got {}",
                SIZE,
                count_result.num_rows()
            ),
        );
    }

    let count_keys = match ctx.provider.download_column_u32(&count_result, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!("Failed to download count keys: {}", e),
            )
        }
    };

    // Count results are u64 (to match predicate declaration types)
    let counts = match ctx.provider.download_column_u64(&count_result, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!("Failed to download counts: {}", e),
            )
        }
    };

    // Verify all counts are 1
    for (i, &count) in counts.iter().enumerate() {
        if count != 1u64 {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!("Count for group {} should be 1, got {}", i, count),
            );
        }
    }

    // Verify all original keys are present
    let result_key_set: HashSet<u32> = count_keys.iter().copied().collect();
    let expected_key_set: HashSet<u32> = keys.iter().copied().collect();
    if result_key_set != expected_key_set {
        return TestResult::error(
            "test_groupby_all_unique_keys",
            start.elapsed(),
            "Groupby result missing some keys".to_string(),
        );
    }

    // Test Sum aggregation - each sum should equal the corresponding val
    let sum_result = match ctx.provider.groupby_multi_agg(&buffer, &[0], &[(1, AggOp::Sum)]) {
        Ok(r) => r,
        Err(e) => {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!("Groupby sum failed: {}", e),
            )
        }
    };

    let sum_keys = match ctx.provider.download_column_u32(&sum_result, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!("Failed to download sum keys: {}", e),
            )
        }
    };

    let sums = match ctx.provider.download_column_u64(&sum_result, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!("Failed to download sums: {}", e),
            )
        }
    };

    // Verify each sum matches the value (which is key * 10)
    for i in 0..sum_keys.len() {
        let key = sum_keys[i];
        let sum = sums[i];
        let expected = (key * 10) as u64;
        if sum != expected {
            return TestResult::error(
                "test_groupby_all_unique_keys",
                start.elapsed(),
                format!(
                    "Sum for key {} should be {}, got {}",
                    key, expected, sum
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_groupby_all_unique_keys",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_groupby_all_unique_keys", start.elapsed())
}
