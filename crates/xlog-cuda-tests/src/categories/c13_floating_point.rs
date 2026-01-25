//! Category 13: Floating-point edge cases
//!
//! Tests floating-point special values and precision including infinity,
//! NaN handling, zero signs, subnormal values, and precision extremes.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c13_floating_point");
    let start = Instant::now();

    results.add_result(test_f64_infinity(ctx));
    results.add_result(test_f64_nan_handling(ctx));
    results.add_result(test_f64_zero_signs(ctx));
    results.add_result(test_f64_subnormal(ctx));
    results.add_result(test_f64_precision_extremes(ctx));
    results.add_result(test_f64_sort_ordering(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Test f64::INFINITY and f64::NEG_INFINITY in sort/filter.
///
/// Verifies that positive and negative infinity values are correctly handled
/// in sorting operations, with proper ordering relative to finite values.
fn test_f64_infinity(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Create data with infinities mixed with finite values
    let data: Vec<f64> = vec![
        1.0,
        f64::INFINITY,
        -1.0,
        f64::NEG_INFINITY,
        0.0,
        f64::INFINITY,
        100.0,
        f64::NEG_INFINITY,
        -100.0,
        f64::MAX,
        f64::MIN,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_f64_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_infinity",
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
                "test_f64_infinity",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_f64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_f64_infinity",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_f64_infinity",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Verify sorted order using total_cmp for consistent infinity ordering
    for i in 1..sorted_data.len() {
        if sorted_data[i].total_cmp(&sorted_data[i - 1]) == std::cmp::Ordering::Less {
            return TestResult::error(
                "test_f64_infinity",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} should be >= {}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify NEG_INFINITY comes first (there are 2 of them)
    if !sorted_data[0].is_infinite() || sorted_data[0] > 0.0 {
        return TestResult::error(
            "test_f64_infinity",
            start.elapsed(),
            format!(
                "First element should be NEG_INFINITY, got {}",
                sorted_data[0]
            ),
        );
    }

    // Verify INFINITY comes last (there are 2 of them)
    let last = sorted_data[sorted_data.len() - 1];
    if !last.is_infinite() || last < 0.0 {
        return TestResult::error(
            "test_f64_infinity",
            start.elapsed(),
            format!("Last element should be INFINITY, got {}", last),
        );
    }

    // Test filter: keep only positive values (should include +INFINITY)
    let mask: Vec<u8> = data.iter().map(|&v| if v > 0.0 { 1 } else { 0 }).collect();
    let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_f64_infinity",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    let filtered_data = match ctx.provider.download_column_f64(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_f64_infinity",
                start.elapsed(),
                format!("Failed to download filtered column: {}", e),
            )
        }
    };

    // Count expected positive values (including +INFINITY)
    let expected_positive: Vec<f64> = data.iter().copied().filter(|&v| v > 0.0).collect();
    if filtered_data.len() != expected_positive.len() {
        return TestResult::error(
            "test_f64_infinity",
            start.elapsed(),
            format!(
                "Filtered {} rows, expected {}",
                filtered_data.len(),
                expected_positive.len()
            ),
        );
    }

    // Verify +INFINITY is in the filtered results
    let has_pos_inf = filtered_data.iter().any(|&v| v == f64::INFINITY);
    if !has_pos_inf {
        return TestResult::error(
            "test_f64_infinity",
            start.elapsed(),
            "Filtered positive values should include INFINITY".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_infinity",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_infinity", start.elapsed())
}

/// Test 2: Test f64::NAN handling (NaN propagation in operations).
///
/// Verifies that NaN values are handled correctly in sorting and filtering.
/// NaN should sort to a consistent position (typically at the end).
fn test_f64_nan_handling(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Create data with NaN values mixed with regular values
    let data: Vec<f64> = vec![
        1.0,
        f64::NAN,
        -1.0,
        f64::NAN,
        0.0,
        2.0,
        f64::NAN,
        -2.0,
        3.0,
        f64::INFINITY,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_f64_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_handling",
                start.elapsed(),
                format!("Failed to create buffer with NaN values: {}", e),
            )
        }
    };

    // Sort the buffer - NaN handling is implementation-defined but should be consistent
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_handling",
                start.elapsed(),
                format!("Sort with NaN values failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_f64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_f64_nan_handling",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_f64_nan_handling",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Count NaN values - should be preserved
    let nan_count = sorted_data.iter().filter(|v| v.is_nan()).count();
    let expected_nan_count = data.iter().filter(|v| v.is_nan()).count();
    if nan_count != expected_nan_count {
        return TestResult::error(
            "test_f64_nan_handling",
            start.elapsed(),
            format!("NaN count changed: {} -> {}", expected_nan_count, nan_count),
        );
    }

    // Verify non-NaN values are sorted correctly
    // Using total_cmp which treats NaN consistently
    let non_nan: Vec<f64> = sorted_data
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .collect();
    for i in 1..non_nan.len() {
        if non_nan[i].total_cmp(&non_nan[i - 1]) == std::cmp::Ordering::Less {
            return TestResult::error(
                "test_f64_nan_handling",
                start.elapsed(),
                format!(
                    "Non-NaN sort order incorrect at index {}: {} should be >= {}",
                    i,
                    non_nan[i],
                    non_nan[i - 1]
                ),
            );
        }
    }

    // Verify all NaN values are grouped together (either at start or end)
    // Based on IEEE total ordering, NaN should sort to end
    let first_nan_idx = sorted_data.iter().position(|v| v.is_nan());
    if let Some(idx) = first_nan_idx {
        // All remaining values after first NaN should also be NaN
        for (i, &val) in sorted_data[idx..].iter().enumerate() {
            if !val.is_nan() {
                return TestResult::error(
                    "test_f64_nan_handling",
                    start.elapsed(),
                    format!(
                        "NaN values should be grouped together, found {} at position {}",
                        val,
                        idx + i
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_nan_handling",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_nan_handling", start.elapsed())
}

/// Test 3: Test +0.0 and -0.0 distinction.
///
/// Verifies that positive and negative zero are handled correctly.
/// While +0.0 == -0.0 mathematically, they have different bit representations
/// and should maintain their distinction through operations.
fn test_f64_zero_signs(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Create data with positive and negative zeros
    let data: Vec<f64> = vec![0.0, -0.0, 1.0, -0.0, 0.0, -1.0, 0.0, -0.0, 0.5, -0.5];

    let buffer = match ctx
        .provider
        .create_buffer_from_f64_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_zero_signs",
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
                "test_f64_zero_signs",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_f64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_f64_zero_signs",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_f64_zero_signs",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Verify sort order using total_cmp which distinguishes -0.0 < +0.0
    for i in 1..sorted_data.len() {
        if sorted_data[i].total_cmp(&sorted_data[i - 1]) == std::cmp::Ordering::Less {
            return TestResult::error(
                "test_f64_zero_signs",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} should be >= {}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Count zeros (both +0.0 and -0.0)
    let total_zeros = sorted_data.iter().filter(|&&v| v == 0.0).count();
    let expected_zeros = data.iter().filter(|&&v| v == 0.0).count();
    if total_zeros != expected_zeros {
        return TestResult::error(
            "test_f64_zero_signs",
            start.elapsed(),
            format!("Zero count changed: {} -> {}", expected_zeros, total_zeros),
        );
    }

    // Verify zeros are grouped together in sorted output
    let first_zero_idx = sorted_data.iter().position(|&v| v == 0.0);
    if let Some(start_idx) = first_zero_idx {
        for i in start_idx..(start_idx + total_zeros) {
            if sorted_data[i] != 0.0 {
                return TestResult::error(
                    "test_f64_zero_signs",
                    start.elapsed(),
                    format!(
                        "Zeros should be grouped together, found {} at index {}",
                        sorted_data[i], i
                    ),
                );
            }
        }
    }

    // Count negative zeros using bit representation
    let neg_zero_count_original = data
        .iter()
        .filter(|&&v| v.to_bits() == (-0.0f64).to_bits())
        .count();
    let neg_zero_count_sorted = sorted_data
        .iter()
        .filter(|&&v| v.to_bits() == (-0.0f64).to_bits())
        .count();

    // The count of negative zeros should be preserved
    if neg_zero_count_sorted != neg_zero_count_original {
        return TestResult::error(
            "test_f64_zero_signs",
            start.elapsed(),
            format!(
                "Negative zero count changed: {} -> {}",
                neg_zero_count_original, neg_zero_count_sorted
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_zero_signs",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_zero_signs", start.elapsed())
}

/// Test 4: Test subnormal/denormalized values (very small numbers).
///
/// Verifies that subnormal (denormalized) floating-point numbers are handled
/// correctly. These are numbers smaller than the smallest normal f64.
fn test_f64_subnormal(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // MIN_POSITIVE is the smallest positive normal number
    // Subnormals are smaller than MIN_POSITIVE but still representable
    let smallest_subnormal = f64::from_bits(1); // Smallest positive subnormal
    let mid_subnormal = f64::from_bits(1u64 << 50); // A mid-range subnormal

    // Create data with subnormal values
    let data: Vec<f64> = vec![
        0.0,
        smallest_subnormal,
        mid_subnormal,
        f64::MIN_POSITIVE,         // Smallest normal
        f64::MIN_POSITIVE * 0.5,   // Subnormal (half of min normal)
        f64::MIN_POSITIVE * 0.25,  // Smaller subnormal
        f64::MIN_POSITIVE * 0.125, // Even smaller subnormal
        -smallest_subnormal,
        -mid_subnormal,
        -f64::MIN_POSITIVE,
        1.0,
        -1.0,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_f64_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_subnormal",
                start.elapsed(),
                format!("Failed to create buffer with subnormal values: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_f64_subnormal",
                start.elapsed(),
                format!("Sort with subnormal values failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_f64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_f64_subnormal",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_f64_subnormal",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Verify sort order
    for i in 1..sorted_data.len() {
        if sorted_data[i].total_cmp(&sorted_data[i - 1]) == std::cmp::Ordering::Less {
            return TestResult::error(
                "test_f64_subnormal",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {:e} should be >= {:e}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify subnormal values are preserved (not flushed to zero)
    let subnormal_count_original = data
        .iter()
        .filter(|&&v| v != 0.0 && v.abs() < f64::MIN_POSITIVE)
        .count();
    let subnormal_count_sorted = sorted_data
        .iter()
        .filter(|&&v| v != 0.0 && v.abs() < f64::MIN_POSITIVE)
        .count();

    if subnormal_count_sorted != subnormal_count_original {
        return TestResult::error(
            "test_f64_subnormal",
            start.elapsed(),
            format!(
                "Subnormal values may have been flushed to zero: {} -> {}",
                subnormal_count_original, subnormal_count_sorted
            ),
        );
    }

    // Verify specific subnormal values are present
    let has_smallest = sorted_data.iter().any(|&v| v == smallest_subnormal);
    if !has_smallest {
        return TestResult::error(
            "test_f64_subnormal",
            start.elapsed(),
            format!(
                "Smallest subnormal {:e} not found in output",
                smallest_subnormal
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_subnormal",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_subnormal", start.elapsed())
}

/// Test 5: Test very large and very small values together.
///
/// Verifies that sorting works correctly when combining values across
/// the full range of f64 representation.
fn test_f64_precision_extremes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Create data spanning the full f64 range
    let data: Vec<f64> = vec![
        f64::MAX,           // Largest positive
        f64::MIN,           // Most negative (largest magnitude negative)
        f64::MIN_POSITIVE,  // Smallest positive normal
        -f64::MIN_POSITIVE, // Smallest magnitude negative normal
        1e308,              // Very large
        -1e308,             // Very large negative
        1e-308,             // Very small positive
        -1e-308,            // Very small negative
        1.0,                // Unity
        -1.0,
        0.0,
        1e100,
        -1e100,
        1e-100,
        -1e-100,
        // Values near overflow boundary
        f64::MAX * 0.99,
        f64::MIN * 0.99,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_f64_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_precision_extremes",
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
                "test_f64_precision_extremes",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_f64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_f64_precision_extremes",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_f64_precision_extremes",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Verify sort order
    for i in 1..sorted_data.len() {
        if sorted_data[i].total_cmp(&sorted_data[i - 1]) == std::cmp::Ordering::Less {
            return TestResult::error(
                "test_f64_precision_extremes",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {:e} should be >= {:e}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify MIN is first (most negative)
    if sorted_data[0] != f64::MIN {
        return TestResult::error(
            "test_f64_precision_extremes",
            start.elapsed(),
            format!("First element should be f64::MIN, got {:e}", sorted_data[0]),
        );
    }

    // Verify MAX is last (largest positive)
    if sorted_data[sorted_data.len() - 1] != f64::MAX {
        return TestResult::error(
            "test_f64_precision_extremes",
            start.elapsed(),
            format!(
                "Last element should be f64::MAX, got {:e}",
                sorted_data[sorted_data.len() - 1]
            ),
        );
    }

    // Verify all original values are present (no precision loss)
    for &original in &data {
        let found = sorted_data.iter().any(|&v| v == original);
        if !found {
            return TestResult::error(
                "test_f64_precision_extremes",
                start.elapsed(),
                format!("Original value {:e} not found in sorted output", original),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_precision_extremes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_precision_extremes", start.elapsed())
}

/// Test 6: Verify f64 sort produces correct total ordering.
///
/// Tests that sorting produces a proper total ordering that is consistent
/// with IEEE 754 totalOrder predicate, handling all special cases correctly.
fn test_f64_sort_ordering(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Create comprehensive test data with all IEEE 754 categories
    let data: Vec<f64> = vec![
        // Negative special values
        f64::NEG_INFINITY,
        f64::MIN, // Most negative finite
        -f64::MAX * 0.5,
        -1e200,
        -1e100,
        -1e10,
        -1000.0,
        -100.0,
        -10.0,
        -1.0,
        -0.5,
        -0.1,
        -1e-10,
        -1e-100,
        -1e-200,
        -f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE * 0.5, // Negative subnormal
        -0.0,
        // Positive values
        0.0,
        f64::MIN_POSITIVE * 0.5, // Positive subnormal
        f64::MIN_POSITIVE,
        1e-200,
        1e-100,
        1e-10,
        0.1,
        0.5,
        1.0,
        10.0,
        100.0,
        1000.0,
        1e10,
        1e100,
        1e200,
        f64::MAX * 0.5,
        f64::MAX,
        f64::INFINITY,
    ];

    // Shuffle the data to test sorting
    let mut shuffled = data.clone();
    // Deterministic shuffle using indices
    for i in 0..shuffled.len() {
        let j = (i * 17 + 7) % shuffled.len();
        shuffled.swap(i, j);
    }

    let buffer = match ctx
        .provider
        .create_buffer_from_f64_slice(&shuffled, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_f64_sort_ordering",
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
                "test_f64_sort_ordering",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download sorted data
    let sorted_data = match ctx.provider.download_column_f64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_f64_sort_ordering",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify row count preserved
    if sorted_data.len() != data.len() {
        return TestResult::error(
            "test_f64_sort_ordering",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted_data.len(),
                data.len()
            ),
        );
    }

    // Verify strict total ordering using total_cmp
    for i in 1..sorted_data.len() {
        let cmp = sorted_data[i].total_cmp(&sorted_data[i - 1]);
        if cmp == std::cmp::Ordering::Less {
            return TestResult::error(
                "test_f64_sort_ordering",
                start.elapsed(),
                format!(
                    "Total ordering violated at index {}: {:e} < {:e}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify expected sort produces same ordering as Rust's sort with total_cmp
    let mut expected = data.clone();
    expected.sort_by(|a, b| a.total_cmp(b));

    for (i, (&actual, &expect)) in sorted_data.iter().zip(expected.iter()).enumerate() {
        // Use total_cmp to check equality (handles -0.0 vs 0.0)
        if actual.total_cmp(&expect) != std::cmp::Ordering::Equal {
            return TestResult::error(
                "test_f64_sort_ordering",
                start.elapsed(),
                format!(
                    "Mismatch at index {}: got {:e}, expected {:e}",
                    i, actual, expect
                ),
            );
        }
    }

    // Verify transitivity: if a <= b and b <= c then a <= c
    // Sample some triplets
    for i in 0..sorted_data.len().saturating_sub(2) {
        let a = sorted_data[i];
        let b = sorted_data[i + 1];
        let c = sorted_data[i + 2];

        // a <= b (verified above)
        // b <= c (verified above)
        // Check a <= c
        if a.total_cmp(&c) == std::cmp::Ordering::Greater {
            return TestResult::error(
                "test_f64_sort_ordering",
                start.elapsed(),
                format!(
                    "Transitivity violated: {:e} > {:e} but {:e} <= {:e} <= {:e}",
                    a, c, a, b, c
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_f64_sort_ordering",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_f64_sort_ordering", start.elapsed())
}
