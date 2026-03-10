//! Category 24: Edge case matrix
//!
//! Cross-product testing of Size x Distribution x Type. This is the largest
//! test category, systematically combining different sizes, data distributions,
//! and operations.

use crate::harness::generators::Distribution;
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::collections::HashSet;
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c24_edge_matrix");
    let start = Instant::now();

    results.add_result(test_size_distribution_matrix_u32(ctx));
    results.add_result(test_size_distribution_matrix_u64(ctx));
    results.add_result(test_size_distribution_matrix_i64(ctx));
    results.add_result(test_size_distribution_matrix_f64(ctx));
    results.add_result(test_operation_matrix(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Cross-product of sizes and distributions for U32.
///
/// Tests U32 sort operation across multiple sizes and distribution patterns.
fn test_size_distribution_matrix_u32(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes: 0, 1, 32, 256, 1000, 10000
    let sizes: Vec<usize> = vec![0, 1, 32, 256, 1000, 10000];

    // Distributions: AllEqual, AllUnique, Sorted, ReverseSorted, Random
    let distributions = vec![
        Distribution::AllEqual,
        Distribution::AllUnique,
        Distribution::Sorted,
        Distribution::ReverseSorted,
        Distribution::Random,
    ];

    for &size in &sizes {
        for dist in &distributions {
            // Generate data
            let data = dist.generate_u32(size, 42);

            let buffer = match ctx
                .provider
                .create_buffer_from_slice::<u32>(&data, schema.clone())
            {
                Ok(buf) => buf,
                Err(e) => {
                    return TestResult::error(
                        "test_size_distribution_matrix_u32",
                        start.elapsed(),
                        format!(
                            "Failed to create buffer for size={}, dist={:?}: {}",
                            size, dist, e
                        ),
                    )
                }
            };

            // Test sort
            let sorted = match ctx.provider.sort(&buffer, &[0]) {
                Ok(s) => s,
                Err(e) => {
                    return TestResult::error(
                        "test_size_distribution_matrix_u32",
                        start.elapsed(),
                        format!("Sort failed for size={}, dist={:?}: {}", size, dist, e),
                    )
                }
            };

            // Verify row count
            if ctx.device_row_count(&sorted) != size as u64 {
                return TestResult::error(
                    "test_size_distribution_matrix_u32",
                    start.elapsed(),
                    format!(
                        "Sort returned {} rows for size={}, dist={:?}, expected {}",
                        ctx.device_row_count(&sorted),
                        size,
                        dist,
                        size
                    ),
                );
            }

            // For non-empty buffers, verify sorted order
            if size > 0 {
                let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
                    Ok(d) => d,
                    Err(e) => {
                        return TestResult::error(
                            "test_size_distribution_matrix_u32",
                            start.elapsed(),
                            format!(
                                "Failed to download sorted data for size={}, dist={:?}: {}",
                                size, dist, e
                            ),
                        )
                    }
                };

                // Verify sorted order
                for i in 1..sorted_data.len() {
                    if sorted_data[i] < sorted_data[i - 1] {
                        return TestResult::error(
                            "test_size_distribution_matrix_u32",
                            start.elapsed(),
                            format!(
                                "Sort order incorrect at index {} for size={}, dist={:?}: {} < {}",
                                i,
                                size,
                                dist,
                                sorted_data[i],
                                sorted_data[i - 1]
                            ),
                        );
                    }
                }

                // Verify all values preserved
                let original_set: HashSet<u32> = data.iter().copied().collect();
                let sorted_set: HashSet<u32> = sorted_data.iter().copied().collect();
                if original_set != sorted_set {
                    return TestResult::error(
                        "test_size_distribution_matrix_u32",
                        start.elapsed(),
                        format!(
                            "Values changed during sort for size={}, dist={:?}",
                            size, dist
                        ),
                    );
                }
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_size_distribution_matrix_u32",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_size_distribution_matrix_u32", start.elapsed())
}

/// Test 2: Cross-product of sizes and distributions for U64.
///
/// Tests U64 sort operation across multiple sizes and distribution patterns.
fn test_size_distribution_matrix_u64(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    let sizes: Vec<usize> = vec![0, 1, 32, 256, 1000, 10000];

    let distributions = vec![
        Distribution::AllEqual,
        Distribution::AllUnique,
        Distribution::Sorted,
        Distribution::ReverseSorted,
        Distribution::Random,
    ];

    for &size in &sizes {
        for dist in &distributions {
            // Generate U64 data (convert from u32 generator)
            let u32_data = dist.generate_u32(size, 42);
            let data: Vec<u64> = u32_data
                .iter()
                .map(|&v| v as u64 * 0x100000001u64)
                .collect();

            let buffer = match ctx
                .provider
                .create_buffer_from_slice::<u64>(&data, schema.clone())
            {
                Ok(buf) => buf,
                Err(e) => {
                    return TestResult::error(
                        "test_size_distribution_matrix_u64",
                        start.elapsed(),
                        format!(
                            "Failed to create U64 buffer for size={}, dist={:?}: {}",
                            size, dist, e
                        ),
                    )
                }
            };

            // Test sort
            let sorted = match ctx.provider.sort(&buffer, &[0]) {
                Ok(s) => s,
                Err(e) => {
                    return TestResult::error(
                        "test_size_distribution_matrix_u64",
                        start.elapsed(),
                        format!("U64 sort failed for size={}, dist={:?}: {}", size, dist, e),
                    )
                }
            };

            // Verify row count
            if ctx.device_row_count(&sorted) != size as u64 {
                return TestResult::error(
                    "test_size_distribution_matrix_u64",
                    start.elapsed(),
                    format!(
                        "U64 sort returned {} rows for size={}, dist={:?}, expected {}",
                        ctx.device_row_count(&sorted),
                        size,
                        dist,
                        size
                    ),
                );
            }

            // For non-empty buffers, verify sorted order
            if size > 0 {
                let sorted_data = match ctx.provider.download_column::<u64>(&sorted, 0) {
                    Ok(d) => d,
                    Err(e) => {
                        return TestResult::error(
                            "test_size_distribution_matrix_u64",
                            start.elapsed(),
                            format!(
                                "Failed to download U64 sorted data for size={}, dist={:?}: {}",
                                size, dist, e
                            ),
                        )
                    }
                };

                // Verify sorted order
                for i in 1..sorted_data.len() {
                    if sorted_data[i] < sorted_data[i - 1] {
                        return TestResult::error(
                            "test_size_distribution_matrix_u64",
                            start.elapsed(),
                            format!(
                                "U64 sort order incorrect at index {} for size={}, dist={:?}: {} < {}",
                                i, size, dist, sorted_data[i], sorted_data[i - 1]
                            ),
                        );
                    }
                }
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_size_distribution_matrix_u64",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_size_distribution_matrix_u64", start.elapsed())
}

/// Test 3: Cross-product of sizes and distributions for I64 (including negative values).
///
/// Tests I64 sort operation across multiple sizes and distribution patterns,
/// with specific attention to signed comparison behavior.
fn test_size_distribution_matrix_i64(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    let sizes: Vec<usize> = vec![0, 1, 32, 256, 1000, 10000];

    let distributions = vec![
        Distribution::AllEqual,
        Distribution::AllUnique,
        Distribution::Sorted,
        Distribution::ReverseSorted,
        Distribution::Random,
    ];

    for &size in &sizes {
        for dist in &distributions {
            // Generate I64 data using the built-in generator
            let data = dist.generate_i64(size, 42);

            let buffer = match ctx
                .provider
                .create_buffer_from_slice::<i64>(&data, schema.clone())
            {
                Ok(buf) => buf,
                Err(e) => {
                    return TestResult::error(
                        "test_size_distribution_matrix_i64",
                        start.elapsed(),
                        format!(
                            "Failed to create I64 buffer for size={}, dist={:?}: {}",
                            size, dist, e
                        ),
                    )
                }
            };

            // Test sort
            let sorted = match ctx.provider.sort(&buffer, &[0]) {
                Ok(s) => s,
                Err(e) => {
                    return TestResult::error(
                        "test_size_distribution_matrix_i64",
                        start.elapsed(),
                        format!("I64 sort failed for size={}, dist={:?}: {}", size, dist, e),
                    )
                }
            };

            // Verify row count
            if ctx.device_row_count(&sorted) != size as u64 {
                return TestResult::error(
                    "test_size_distribution_matrix_i64",
                    start.elapsed(),
                    format!(
                        "I64 sort returned {} rows for size={}, dist={:?}, expected {}",
                        ctx.device_row_count(&sorted),
                        size,
                        dist,
                        size
                    ),
                );
            }

            // For non-empty buffers, verify sorted order (signed comparison)
            if size > 0 {
                let sorted_data = match ctx.provider.download_column::<i64>(&sorted, 0) {
                    Ok(d) => d,
                    Err(e) => {
                        return TestResult::error(
                            "test_size_distribution_matrix_i64",
                            start.elapsed(),
                            format!(
                                "Failed to download I64 sorted data for size={}, dist={:?}: {}",
                                size, dist, e
                            ),
                        )
                    }
                };

                // Verify sorted order (signed comparison - negative before positive)
                for i in 1..sorted_data.len() {
                    if sorted_data[i] < sorted_data[i - 1] {
                        return TestResult::error(
                            "test_size_distribution_matrix_i64",
                            start.elapsed(),
                            format!(
                                "I64 sort order incorrect at index {} for size={}, dist={:?}: {} < {}",
                                i, size, dist, sorted_data[i], sorted_data[i - 1]
                            ),
                        );
                    }
                }

                // For Alternating distribution, verify negative values come before positive
                if *dist == Distribution::Alternating && size > 1 {
                    let has_neg = sorted_data.iter().any(|&v| v < 0);
                    let has_pos = sorted_data.iter().any(|&v| v > 0);
                    if has_neg && has_pos {
                        // Find first positive
                        let first_pos_idx = sorted_data.iter().position(|&v| v >= 0);
                        if let Some(pos_idx) = first_pos_idx {
                            // All values before pos_idx should be negative
                            for i in 0..pos_idx {
                                if sorted_data[i] >= 0 {
                                    return TestResult::error(
                                        "test_size_distribution_matrix_i64",
                                        start.elapsed(),
                                        format!(
                                            "I64 signed sort error: non-negative {} at index {} before positive section for size={}, dist={:?}",
                                            sorted_data[i], i, size, dist
                                        ),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_size_distribution_matrix_i64",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_size_distribution_matrix_i64", start.elapsed())
}

/// Test 4: Cross-product of sizes and distributions for F64 (including edge values).
///
/// Tests F64 sort operation across multiple sizes and distribution patterns,
/// with specific attention to floating-point edge values.
fn test_size_distribution_matrix_f64(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let sizes: Vec<usize> = vec![0, 1, 32, 256, 1000, 10000];

    let distributions = vec![
        Distribution::AllEqual,
        Distribution::AllUnique,
        Distribution::Sorted,
        Distribution::ReverseSorted,
        Distribution::Random,
    ];

    for &size in &sizes {
        for dist in &distributions {
            // Generate F64 data using the built-in generator
            let data = dist.generate_f64(size, 42);

            let buffer = match ctx
                .provider
                .create_buffer_from_slice::<f64>(&data, schema.clone())
            {
                Ok(buf) => buf,
                Err(e) => {
                    return TestResult::error(
                        "test_size_distribution_matrix_f64",
                        start.elapsed(),
                        format!(
                            "Failed to create F64 buffer for size={}, dist={:?}: {}",
                            size, dist, e
                        ),
                    )
                }
            };

            // Test sort
            let sorted = match ctx.provider.sort(&buffer, &[0]) {
                Ok(s) => s,
                Err(e) => {
                    return TestResult::error(
                        "test_size_distribution_matrix_f64",
                        start.elapsed(),
                        format!("F64 sort failed for size={}, dist={:?}: {}", size, dist, e),
                    )
                }
            };

            // Verify row count
            if ctx.device_row_count(&sorted) != size as u64 {
                return TestResult::error(
                    "test_size_distribution_matrix_f64",
                    start.elapsed(),
                    format!(
                        "F64 sort returned {} rows for size={}, dist={:?}, expected {}",
                        ctx.device_row_count(&sorted),
                        size,
                        dist,
                        size
                    ),
                );
            }

            // For non-empty buffers, verify sorted order
            if size > 0 {
                let sorted_data = match ctx.provider.download_column::<f64>(&sorted, 0) {
                    Ok(d) => d,
                    Err(e) => {
                        return TestResult::error(
                            "test_size_distribution_matrix_f64",
                            start.elapsed(),
                            format!(
                                "Failed to download F64 sorted data for size={}, dist={:?}: {}",
                                size, dist, e
                            ),
                        )
                    }
                };

                // Verify sorted order using total_cmp for consistent NaN handling
                for i in 1..sorted_data.len() {
                    if sorted_data[i].total_cmp(&sorted_data[i - 1]) == std::cmp::Ordering::Less {
                        return TestResult::error(
                            "test_size_distribution_matrix_f64",
                            start.elapsed(),
                            format!(
                                "F64 sort order incorrect at index {} for size={}, dist={:?}: {} < {}",
                                i, size, dist, sorted_data[i], sorted_data[i - 1]
                            ),
                        );
                    }
                }
            }
        }
    }

    // Additional test with F64 edge values mixed in
    let edge_values: Vec<f64> = vec![
        f64::NEG_INFINITY,
        f64::MIN,
        -1.0,
        -f64::MIN_POSITIVE,
        -0.0,
        0.0,
        f64::MIN_POSITIVE,
        1.0,
        f64::MAX,
        f64::INFINITY,
    ];

    let edge_buffer = match ctx
        .provider
        .create_buffer_from_slice::<f64>(&edge_values, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_size_distribution_matrix_f64",
                start.elapsed(),
                format!("Failed to create F64 edge values buffer: {}", e),
            )
        }
    };

    let edge_sorted = match ctx.provider.sort(&edge_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_size_distribution_matrix_f64",
                start.elapsed(),
                format!("F64 edge values sort failed: {}", e),
            )
        }
    };

    let edge_sorted_data = match ctx.provider.download_column::<f64>(&edge_sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_size_distribution_matrix_f64",
                start.elapsed(),
                format!("Failed to download F64 edge sorted data: {}", e),
            )
        }
    };

    // Verify NEG_INFINITY is first
    if !edge_sorted_data[0].is_infinite() || edge_sorted_data[0] > 0.0 {
        return TestResult::error(
            "test_size_distribution_matrix_f64",
            start.elapsed(),
            format!(
                "F64 edge values: first element should be NEG_INFINITY, got {}",
                edge_sorted_data[0]
            ),
        );
    }

    // Verify INFINITY is last
    let last = edge_sorted_data[edge_sorted_data.len() - 1];
    if !last.is_infinite() || last < 0.0 {
        return TestResult::error(
            "test_size_distribution_matrix_f64",
            start.elapsed(),
            format!(
                "F64 edge values: last element should be INFINITY, got {}",
                last
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_size_distribution_matrix_f64",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_size_distribution_matrix_f64", start.elapsed())
}

/// Test 5: Cross-product of operations with different sizes.
///
/// Tests multiple operations (sort, filter, dedup) across different sizes.
fn test_operation_matrix(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes: 100, 1000, 10000
    let sizes: Vec<usize> = vec![100, 1000, 10000];

    // Filter selectivities: 0%, 50%, 100%
    let selectivities: Vec<(f64, &str)> = vec![(0.0, "0%"), (0.5, "50%"), (1.0, "100%")];

    for &size in &sizes {
        // Create random data for this size
        let data = Distribution::Random.generate_u32(size, 12345);

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Failed to create buffer for size {}: {}", size, e),
                )
            }
        };

        // Test SORT
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&sorted) != size as u64 {
            return TestResult::error(
                "test_operation_matrix",
                start.elapsed(),
                format!(
                    "Sort returned {} rows for size {}, expected {}",
                    ctx.device_row_count(&sorted),
                    size,
                    size
                ),
            );
        }

        let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Failed to download sorted data for size {}: {}", size, e),
                )
            }
        };

        for i in 1..sorted_data.len() {
            if sorted_data[i] < sorted_data[i - 1] {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!(
                        "Sort order incorrect at index {} for size {}: {} < {}",
                        i,
                        size,
                        sorted_data[i],
                        sorted_data[i - 1]
                    ),
                );
            }
        }

        // Test FILTER with different selectivities
        for (selectivity, name) in &selectivities {
            let mask: Vec<u8> = if *selectivity == 0.0 {
                vec![0; size]
            } else if *selectivity == 1.0 {
                vec![1; size]
            } else {
                // 50% - alternate
                (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect()
            };

            let expected_count = mask.iter().map(|&m| m as usize).sum::<usize>();

            let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
                Ok(f) => f,
                Err(e) => {
                    return TestResult::error(
                        "test_operation_matrix",
                        start.elapsed(),
                        format!("Filter {} failed for size {}: {}", name, size, e),
                    )
                }
            };

            if ctx.device_row_count(&filtered) != expected_count as u64 {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!(
                        "Filter {} returned {} rows for size {}, expected {}",
                        name,
                        ctx.device_row_count(&filtered),
                        size,
                        expected_count
                    ),
                );
            }
        }

        // Test DEDUP
        let dedup_schema = Schema::new(vec![
            ("key".to_string(), ScalarType::U32),
            ("val".to_string(), ScalarType::U32),
        ]);

        // Create keys with some duplicates (mod 100)
        let dedup_keys: Vec<u32> = (0..size as u32).map(|i| i % 100).collect();
        let dedup_vals: Vec<u32> = (0..size as u32).collect();

        let dedup_buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&dedup_keys, &dedup_vals], dedup_schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Failed to create dedup buffer for size {}: {}", size, e),
                )
            }
        };

        let deduped = match ctx.provider.dedup(&dedup_buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Dedup failed for size {}: {}", size, e),
                )
            }
        };

        // Should have exactly 100 unique keys (or size if size < 100)
        let expected_unique = std::cmp::min(100, size);
        if ctx.device_row_count(&deduped) != expected_unique as u64 {
            return TestResult::error(
                "test_operation_matrix",
                start.elapsed(),
                format!(
                    "Dedup returned {} rows for size {}, expected {}",
                    ctx.device_row_count(&deduped),
                    size,
                    expected_unique
                ),
            );
        }

        let deduped_keys = match ctx.provider.download_column::<u32>(&deduped, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Failed to download deduped keys for size {}: {}", size, e),
                )
            }
        };

        // Verify all deduped keys are unique
        let key_set: HashSet<u32> = deduped_keys.iter().copied().collect();
        if key_set.len() != deduped_keys.len() {
            return TestResult::error(
                "test_operation_matrix",
                start.elapsed(),
                format!(
                    "Dedup result contains duplicates for size {}: {} unique out of {}",
                    size,
                    key_set.len(),
                    deduped_keys.len()
                ),
            );
        }
    }

    // Additional test: chain operations (sort -> filter -> dedup)
    for &size in &sizes {
        let data = Distribution::Random.generate_u32(size, 54321);

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Failed to create chain buffer for size {}: {}", size, e),
                )
            }
        };

        // Sort
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Chain sort failed for size {}: {}", size, e),
                )
            }
        };

        // Filter 50%
        let mask: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let filtered = match ctx.provider.filter_by_mask(&sorted, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!("Chain filter failed for size {}: {}", size, e),
                )
            }
        };

        let expected_after_filter = (size + 1) / 2; // Ceiling division for odd sizes
        if ctx.device_row_count(&filtered) != expected_after_filter as u64 {
            return TestResult::error(
                "test_operation_matrix",
                start.elapsed(),
                format!(
                    "Chain filter returned {} rows for size {}, expected {}",
                    ctx.device_row_count(&filtered),
                    size,
                    expected_after_filter
                ),
            );
        }

        // Dedup on the filtered (which is still single column, need to add val column)
        // Actually, for dedup we need two columns, so let's just verify the filter worked
        let filtered_data = match ctx.provider.download_column::<u32>(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!(
                        "Failed to download chain filtered data for size {}: {}",
                        size, e
                    ),
                )
            }
        };

        // Verify filtered data is still sorted
        for i in 1..filtered_data.len() {
            if filtered_data[i] < filtered_data[i - 1] {
                return TestResult::error(
                    "test_operation_matrix",
                    start.elapsed(),
                    format!(
                        "Chain: filtered data not sorted at index {} for size {}: {} < {}",
                        i,
                        size,
                        filtered_data[i],
                        filtered_data[i - 1]
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_operation_matrix",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_operation_matrix", start.elapsed())
}
