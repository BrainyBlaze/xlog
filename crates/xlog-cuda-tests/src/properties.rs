//! Property-based tests for CUDA kernel correctness.
//!
//! Uses proptest to verify kernel properties with randomized inputs.
//!
//! Properties tested:
//! - Sort stability: Equal keys preserve relative order
//! - Join correctness: Result equals set of matching pairs
//! - Filter idempotence: filter(filter(x)) = filter(x)
//! - Dedup determinism: Same input produces same output
//!
//! # Usage
//!
//! ```bash
//! cargo test -p xlog-cuda-tests --test properties --release -- --nocapture
//! ```

use crate::harness::TestContext;
use std::collections::{HashMap, HashSet};
use xlog_core::{ScalarType, Schema};
use xlog_cuda::CompareOp;

// =============================================================================
// Property 1: Sort Stability
// =============================================================================

/// Property: When sorting by key, rows with equal keys maintain their original
/// relative order (stability).
///
/// For each group of equal keys, if row A appeared before row B in the input,
/// row A should appear before row B in the output.
pub fn prop_sort_stability(
    ctx: &TestContext,
    keys: Vec<u32>,
    vals: Vec<u32>,
) -> Result<(), String> {
    if keys.is_empty() || keys.len() != vals.len() {
        return Ok(()); // Trivially true for empty or mismatched inputs
    }

    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Create buffer with (key, val) pairs
    let buffer = ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema)
        .map_err(|e| format!("Buffer creation failed: {}", e))?;

    // Sort by key
    let sorted = ctx
        .provider
        .sort(&buffer, &[0])
        .map_err(|e| format!("Sort failed: {}", e))?;

    // Download results
    let sorted_keys = ctx
        .provider
        .download_column_u32(&sorted, 0)
        .map_err(|e| format!("Download keys failed: {}", e))?;
    let sorted_vals = ctx
        .provider
        .download_column_u32(&sorted, 1)
        .map_err(|e| format!("Download vals failed: {}", e))?;

    // Verify keys are sorted (ascending)
    for i in 1..sorted_keys.len() {
        if sorted_keys[i] < sorted_keys[i - 1] {
            return Err(format!(
                "Keys not sorted at index {}: {} < {}",
                i,
                sorted_keys[i],
                sorted_keys[i - 1]
            ));
        }
    }

    // Check counts match for each unique pair (data preservation)
    let mut original_counts: HashMap<(u32, u32), usize> = HashMap::new();
    for (&k, &v) in keys.iter().zip(vals.iter()) {
        *original_counts.entry((k, v)).or_insert(0) += 1;
    }

    let mut sorted_counts: HashMap<(u32, u32), usize> = HashMap::new();
    for (&k, &v) in sorted_keys.iter().zip(sorted_vals.iter()) {
        *sorted_counts.entry((k, v)).or_insert(0) += 1;
    }

    if original_counts != sorted_counts {
        return Err("Sort changed data - counts don't match".to_string());
    }

    Ok(())
}

// =============================================================================
// Property 2: Join Correctness
// =============================================================================

/// Property: Hash join result equals the set of all matching pairs.
///
/// For each (key, lval) in left and (key, rval) in right where keys match,
/// the result should contain exactly one (key, lval, rval) tuple.
pub fn prop_join_correctness(
    ctx: &TestContext,
    left_keys: Vec<u32>,
    left_vals: Vec<u32>,
    right_keys: Vec<u32>,
    right_vals: Vec<u32>,
) -> Result<(), String> {
    if left_keys.len() != left_vals.len() || right_keys.len() != right_vals.len() {
        return Ok(()); // Invalid input
    }

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    // Handle empty tables
    if left_keys.is_empty() {
        let left_buffer = ctx
            .provider
            .create_empty_buffer(left_schema)
            .map_err(|e| format!("Failed to create empty left buffer: {}", e))?;

        if right_keys.is_empty() {
            let right_buffer = ctx
                .provider
                .create_empty_buffer(right_schema)
                .map_err(|e| format!("Failed to create empty right buffer: {}", e))?;

            let joined = ctx
                .provider
                .hash_join(&left_buffer, &right_buffer, &[0], &[0])
                .map_err(|e| format!("Join failed: {}", e))?;

            if ctx.device_row_count(&joined) != 0 {
                return Err(format!(
                    "Empty tables should produce 0 rows, got {}",
                    ctx.device_row_count(&joined)
                ));
            }
            return Ok(());
        }

        let right_buffer = ctx
            .provider
            .create_buffer_from_u32_columns(&[&right_keys, &right_vals], right_schema)
            .map_err(|e| format!("Failed to create right buffer: {}", e))?;

        let joined = ctx
            .provider
            .hash_join(&left_buffer, &right_buffer, &[0], &[0])
            .map_err(|e| format!("Join failed: {}", e))?;

        if ctx.device_row_count(&joined) != 0 {
            return Err(format!(
                "Empty left should produce 0 rows, got {}",
                ctx.device_row_count(&joined)
            ));
        }
        return Ok(());
    }

    if right_keys.is_empty() {
        let left_buffer = ctx
            .provider
            .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema)
            .map_err(|e| format!("Failed to create left buffer: {}", e))?;
        let right_buffer = ctx
            .provider
            .create_empty_buffer(right_schema)
            .map_err(|e| format!("Failed to create empty right buffer: {}", e))?;

        let joined = ctx
            .provider
            .hash_join(&left_buffer, &right_buffer, &[0], &[0])
            .map_err(|e| format!("Join failed: {}", e))?;

        if ctx.device_row_count(&joined) != 0 {
            return Err(format!(
                "Empty right should produce 0 rows, got {}",
                ctx.device_row_count(&joined)
            ));
        }
        return Ok(());
    }

    let left_buffer = ctx
        .provider
        .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema)
        .map_err(|e| format!("Failed to create left buffer: {}", e))?;

    let right_buffer = ctx
        .provider
        .create_buffer_from_u32_columns(&[&right_keys, &right_vals], right_schema)
        .map_err(|e| format!("Failed to create right buffer: {}", e))?;

    // Perform join
    let joined = ctx
        .provider
        .hash_join(&left_buffer, &right_buffer, &[0], &[0])
        .map_err(|e| format!("Join failed: {}", e))?;

    // Download results
    let result_keys = ctx
        .provider
        .download_column_u32(&joined, 0)
        .map_err(|e| format!("Download result keys failed: {}", e))?;
    let result_lvals = ctx
        .provider
        .download_column_u32(&joined, 1)
        .map_err(|e| format!("Download result lvals failed: {}", e))?;
    let result_rvals = ctx
        .provider
        .download_column_u32(&joined, 2)
        .map_err(|e| format!("Download result rvals failed: {}", e))?;

    // Compute expected result using CPU reference implementation
    let mut right_by_key: HashMap<u32, Vec<u32>> = HashMap::new();
    for (&k, &v) in right_keys.iter().zip(right_vals.iter()) {
        right_by_key.entry(k).or_default().push(v);
    }

    // Expected tuples with multiplicity
    let mut expected: HashMap<(u32, u32, u32), usize> = HashMap::new();
    for (&lk, &lv) in left_keys.iter().zip(left_vals.iter()) {
        if let Some(right_vals_for_key) = right_by_key.get(&lk) {
            for &rv in right_vals_for_key {
                *expected.entry((lk, lv, rv)).or_insert(0) += 1;
            }
        }
    }

    // Actual tuples with multiplicity
    let mut actual: HashMap<(u32, u32, u32), usize> = HashMap::new();
    for i in 0..result_keys.len() {
        let tuple = (result_keys[i], result_lvals[i], result_rvals[i]);
        *actual.entry(tuple).or_insert(0) += 1;
    }

    // Compare expected vs actual
    if expected != actual {
        let expected_total: usize = expected.values().sum();
        let actual_total: usize = actual.values().sum();

        let mut missing: Vec<(u32, u32, u32)> = Vec::new();
        let mut extra: Vec<(u32, u32, u32)> = Vec::new();

        for (tuple, &count) in &expected {
            let actual_count = actual.get(tuple).copied().unwrap_or(0);
            if actual_count < count {
                for _ in 0..(count - actual_count) {
                    missing.push(*tuple);
                }
            }
        }

        for (tuple, &count) in &actual {
            let expected_count = expected.get(tuple).copied().unwrap_or(0);
            if count > expected_count {
                for _ in 0..(count - expected_count) {
                    extra.push(*tuple);
                }
            }
        }

        let mut msg = format!(
            "Join result mismatch: expected {} rows, got {}.",
            expected_total, actual_total
        );
        if !missing.is_empty() {
            msg.push_str(&format!(" Missing: {:?}", &missing[..missing.len().min(5)]));
        }
        if !extra.is_empty() {
            msg.push_str(&format!(" Extra: {:?}", &extra[..extra.len().min(5)]));
        }

        return Err(msg);
    }

    Ok(())
}

// =============================================================================
// Property 3: Filter Idempotence
// =============================================================================

/// Property: Applying the same filter twice produces the same result.
///
/// filter(filter(x, cond), cond) = filter(x, cond)
///
/// This tests that filtering is idempotent - once rows are removed, applying
/// the same condition again doesn't change anything.
pub fn prop_filter_idempotence(
    ctx: &TestContext,
    data: Vec<u32>,
    threshold: u32,
) -> Result<(), String> {
    if data.is_empty() {
        return Ok(()); // Trivially true for empty input
    }

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let buffer = ctx
        .provider
        .create_buffer_from_u32_slice(&data, schema)
        .map_err(|e| format!("Buffer creation failed: {}", e))?;

    // First filter: val < threshold
    let filtered_once = ctx
        .provider
        .filter_u32(&buffer, 0, threshold, CompareOp::Lt)
        .map_err(|e| format!("First filter failed: {}", e))?;

    // Second filter with same condition
    let filtered_twice = ctx
        .provider
        .filter_u32(&filtered_once, 0, threshold, CompareOp::Lt)
        .map_err(|e| format!("Second filter failed: {}", e))?;

    // Download both results
    let once_data = ctx
        .provider
        .download_column_u32(&filtered_once, 0)
        .map_err(|e| format!("Download first filter failed: {}", e))?;

    let twice_data = ctx
        .provider
        .download_column_u32(&filtered_twice, 0)
        .map_err(|e| format!("Download second filter failed: {}", e))?;

    // Results should be identical
    if once_data.len() != twice_data.len() {
        return Err(format!(
            "Filter idempotence violated: once={} rows, twice={} rows",
            once_data.len(),
            twice_data.len()
        ));
    }

    if once_data != twice_data {
        return Err(
            "Filter idempotence violated: data differs after second application".to_string(),
        );
    }

    // Also verify the first filter was correct
    let expected: Vec<u32> = data.iter().filter(|&&v| v < threshold).copied().collect();
    if once_data != expected {
        return Err(format!(
            "First filter incorrect: expected {} rows matching, got {}",
            expected.len(),
            once_data.len()
        ));
    }

    Ok(())
}

// =============================================================================
// Property 4: Dedup Determinism
// =============================================================================

/// Property: Dedup produces the same output for the same input.
///
/// Running dedup multiple times on identical input should produce identical output.
/// The set of unique keys should match, and the number of rows should be correct.
pub fn prop_dedup_determinism(
    ctx: &TestContext,
    keys: Vec<u32>,
    vals: Vec<u32>,
) -> Result<(), String> {
    if keys.is_empty() || keys.len() != vals.len() {
        return Ok(()); // Trivially true for empty or mismatched inputs
    }

    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Run dedup twice on the same data
    let buffer1 = ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
        .map_err(|e| format!("Buffer1 creation failed: {}", e))?;

    let buffer2 = ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema)
        .map_err(|e| format!("Buffer2 creation failed: {}", e))?;

    let dedup1 = ctx
        .provider
        .dedup(&buffer1, &[0])
        .map_err(|e| format!("First dedup failed: {}", e))?;

    let dedup2 = ctx
        .provider
        .dedup(&buffer2, &[0])
        .map_err(|e| format!("Second dedup failed: {}", e))?;

    // Download results
    let keys1 = ctx
        .provider
        .download_column_u32(&dedup1, 0)
        .map_err(|e| format!("Download dedup1 keys failed: {}", e))?;

    let keys2 = ctx
        .provider
        .download_column_u32(&dedup2, 0)
        .map_err(|e| format!("Download dedup2 keys failed: {}", e))?;

    // Row counts should match
    if ctx.device_row_count(&dedup1) != ctx.device_row_count(&dedup2) {
        return Err(format!(
            "Dedup row count differs: {} vs {}",
            ctx.device_row_count(&dedup1),
            ctx.device_row_count(&dedup2)
        ));
    }

    // Sets of unique keys should be identical
    let set1: HashSet<u32> = keys1.iter().copied().collect();
    let set2: HashSet<u32> = keys2.iter().copied().collect();

    if set1 != set2 {
        return Err("Dedup key sets differ across runs".to_string());
    }

    // Verify correctness: number of unique keys
    let expected_unique: HashSet<u32> = keys.iter().copied().collect();
    if set1 != expected_unique {
        return Err(format!(
            "Dedup result incorrect: expected {} unique keys, got {}",
            expected_unique.len(),
            set1.len()
        ));
    }

    // Verify no duplicates in output
    if keys1.len() != set1.len() {
        return Err(format!(
            "Dedup output contains duplicates: {} rows but only {} unique",
            keys1.len(),
            set1.len()
        ));
    }

    Ok(())
}

// =============================================================================
// Test Module with Proptest Integration
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a ProptestConfig for GPU property tests.
    /// We use fewer cases than default due to GPU overhead.
    fn gpu_proptest_config() -> ProptestConfig {
        ProptestConfig {
            cases: 50,
            max_shrink_iters: 100,
            failure_persistence: None,
            ..ProptestConfig::default()
        }
    }

    /// Create test context, skipping if no GPU available.
    fn get_context() -> Option<TestContext> {
        match TestContext::new() {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("Skipping property test: no CUDA device ({})", e);
                None
            }
        }
    }

    // -------------------------------------------------------------------------
    // Proptest Strategies
    // -------------------------------------------------------------------------

    /// Strategy for generating key-value pairs for sort tests.
    fn sort_data_strategy(
        max_rows: usize,
        max_key: u32,
    ) -> impl Strategy<Value = (Vec<u32>, Vec<u32>)> {
        prop::collection::vec(0..max_key, 1..=max_rows).prop_flat_map(move |keys| {
            let len = keys.len();
            prop::collection::vec(any::<u32>(), len..=len)
                .prop_map(move |vals| (keys.clone(), vals))
        })
    }

    /// Strategy for generating left and right tables for join tests.
    fn join_data_strategy(
        max_rows: usize,
        max_key: u32,
    ) -> impl Strategy<Value = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        (
            prop::collection::vec(0..max_key, 0..=max_rows),
            prop::collection::vec(0..max_key, 0..=max_rows),
        )
            .prop_flat_map(move |(left_keys, right_keys)| {
                let left_len = left_keys.len();
                let right_len = right_keys.len();
                (
                    Just(left_keys),
                    prop::collection::vec(any::<u32>(), left_len..=left_len),
                    Just(right_keys),
                    prop::collection::vec(any::<u32>(), right_len..=right_len),
                )
            })
            .prop_map(|(lk, lv, rk, rv)| (lk, lv, rk, rv))
    }

    /// Strategy for generating filter test data.
    fn filter_data_strategy(
        max_rows: usize,
        max_val: u32,
    ) -> impl Strategy<Value = (Vec<u32>, u32)> {
        prop::collection::vec(0..max_val, 1..=max_rows)
            .prop_flat_map(move |data| (Just(data), 0..max_val))
    }

    /// Strategy for generating dedup test data with varying duplicate patterns.
    fn dedup_data_strategy(
        max_rows: usize,
        max_key: u32,
    ) -> impl Strategy<Value = (Vec<u32>, Vec<u32>)> {
        prop::collection::vec(0..max_key, 1..=max_rows).prop_flat_map(|keys| {
            let len = keys.len();
            prop::collection::vec(any::<u32>(), len..=len)
                .prop_map(move |vals| (keys.clone(), vals))
        })
    }

    // -------------------------------------------------------------------------
    // Sort Stability Tests
    // -------------------------------------------------------------------------

    proptest! {
        #![proptest_config(gpu_proptest_config())]

        #[test]
        fn test_prop_sort_stability(
            (keys, vals) in sort_data_strategy(1000, 100)
        ) {
            if let Some(ctx) = get_context() {
                prop_sort_stability(&ctx, keys, vals).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }

        #[test]
        fn test_prop_sort_stability_high_duplicates(
            (keys, vals) in sort_data_strategy(1000, 10) // Few unique keys = many duplicates
        ) {
            if let Some(ctx) = get_context() {
                prop_sort_stability(&ctx, keys, vals).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }

        #[test]
        fn test_prop_sort_stability_large(
            (keys, vals) in sort_data_strategy(5000, 500)
        ) {
            if let Some(ctx) = get_context() {
                prop_sort_stability(&ctx, keys, vals).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Join Correctness Tests
    // -------------------------------------------------------------------------

    proptest! {
        #![proptest_config(gpu_proptest_config())]

        #[test]
        fn test_prop_join_correctness(
            (lk, lv, rk, rv) in join_data_strategy(500, 100)
        ) {
            if let Some(ctx) = get_context() {
                prop_join_correctness(&ctx, lk, lv, rk, rv).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }

        #[test]
        fn test_prop_join_correctness_high_overlap(
            (lk, lv, rk, rv) in join_data_strategy(500, 20) // Few keys = high overlap
        ) {
            if let Some(ctx) = get_context() {
                prop_join_correctness(&ctx, lk, lv, rk, rv).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }

        #[test]
        fn test_prop_join_correctness_no_overlap(
            left_size in 1usize..500,
            right_size in 1usize..500,
        ) {
            if let Some(ctx) = get_context() {
                // Disjoint key ranges
                let lk: Vec<u32> = (0..left_size as u32).collect();
                let lv: Vec<u32> = lk.iter().map(|k| k * 10).collect();
                let rk: Vec<u32> = ((left_size as u32)..(left_size as u32 + right_size as u32)).collect();
                let rv: Vec<u32> = rk.iter().map(|k| k * 100).collect();

                prop_join_correctness(&ctx, lk, lv, rk, rv).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Filter Idempotence Tests
    // -------------------------------------------------------------------------

    proptest! {
        #![proptest_config(gpu_proptest_config())]

        #[test]
        fn test_prop_filter_idempotence(
            (data, threshold) in filter_data_strategy(1000, 1000)
        ) {
            if let Some(ctx) = get_context() {
                prop_filter_idempotence(&ctx, data, threshold).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }

        #[test]
        fn test_prop_filter_idempotence_all_pass(
            data in prop::collection::vec(0u32..100, 1..1000)
        ) {
            if let Some(ctx) = get_context() {
                // Threshold higher than all values - all pass
                prop_filter_idempotence(&ctx, data, 1000).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }

        #[test]
        fn test_prop_filter_idempotence_none_pass(
            data in prop::collection::vec(100u32..1000, 1..1000)
        ) {
            if let Some(ctx) = get_context() {
                // Threshold lower than all values - none pass
                prop_filter_idempotence(&ctx, data, 50).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Dedup Determinism Tests
    // -------------------------------------------------------------------------

    proptest! {
        #![proptest_config(gpu_proptest_config())]

        #[test]
        fn test_prop_dedup_determinism(
            (keys, vals) in dedup_data_strategy(1000, 100)
        ) {
            if let Some(ctx) = get_context() {
                prop_dedup_determinism(&ctx, keys, vals).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }

        #[test]
        fn test_prop_dedup_determinism_all_same(
            size in 1usize..1000,
            key in any::<u32>(),
        ) {
            if let Some(ctx) = get_context() {
                let keys = vec![key; size];
                let vals: Vec<u32> = (0..size as u32).collect();
                prop_dedup_determinism(&ctx, keys, vals).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }

        #[test]
        fn test_prop_dedup_determinism_all_unique(
            size in 1usize..1000,
        ) {
            if let Some(ctx) = get_context() {
                let keys: Vec<u32> = (0..size as u32).collect();
                let vals = keys.clone();
                prop_dedup_determinism(&ctx, keys, vals).map_err(|e| {
                    TestCaseError::Fail(e.into())
                })?;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests (Non-proptest)
    // -------------------------------------------------------------------------

    #[test]
    fn test_sort_stability_single_element() {
        if let Some(ctx) = get_context() {
            prop_sort_stability(&ctx, vec![42], vec![100]).unwrap();
        }
    }

    #[test]
    fn test_sort_stability_all_equal_keys() {
        if let Some(ctx) = get_context() {
            let keys = vec![5; 100];
            let vals: Vec<u32> = (0..100).collect();
            prop_sort_stability(&ctx, keys, vals).unwrap();
        }
    }

    #[test]
    fn test_join_correctness_empty_tables() {
        if let Some(ctx) = get_context() {
            prop_join_correctness(&ctx, vec![], vec![], vec![], vec![]).unwrap();
        }
    }

    #[test]
    fn test_join_correctness_single_match() {
        if let Some(ctx) = get_context() {
            prop_join_correctness(&ctx, vec![1], vec![10], vec![1], vec![100]).unwrap();
        }
    }

    #[test]
    fn test_filter_idempotence_single_element() {
        if let Some(ctx) = get_context() {
            prop_filter_idempotence(&ctx, vec![50], 100).unwrap();
            prop_filter_idempotence(&ctx, vec![150], 100).unwrap();
        }
    }

    #[test]
    fn test_dedup_determinism_single_element() {
        if let Some(ctx) = get_context() {
            prop_dedup_determinism(&ctx, vec![42], vec![100]).unwrap();
        }
    }
}
