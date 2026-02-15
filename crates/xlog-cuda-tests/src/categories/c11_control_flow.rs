//! Category 11: Control Flow and Predication
//!
//! Tests conditional execution patterns including various filter selectivity
//! levels from 0% to 100%, sparse and dense predicates, alternating patterns,
//! and random predicate distributions.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c11_control_flow");
    let start = Instant::now();

    results.add_result(test_filter_all_pass(ctx));
    results.add_result(test_filter_none_pass(ctx));
    results.add_result(test_filter_half_pass(ctx));
    results.add_result(test_sparse_predicate(ctx));
    results.add_result(test_dense_predicate(ctx));
    results.add_result(test_alternating_predicate(ctx));
    results.add_result(test_random_predicate_distribution(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Filter where all elements pass (100% selectivity).
///
/// When all elements pass the filter, the output should equal the input.
/// This tests the edge case where the predicate is always true.
fn test_filter_all_pass(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various sizes with 100% selectivity
    let sizes: Vec<usize> = vec![100, 1000, 10000, 100000];

    for size in sizes {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_filter_all_pass",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // All ones mask - 100% pass
        let mask: Vec<u8> = vec![1; size];

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_filter_all_pass",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        // All elements should pass
        if ctx.device_row_count(&filtered) != size as u64 {
            return TestResult::error(
                "test_filter_all_pass",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {} (100% selectivity)",
                    size,
                    ctx.device_row_count(&filtered),
                    size
                ),
            );
        }

        // Download and verify data is identical
        let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_filter_all_pass",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        if filtered_data != data {
            return TestResult::error(
                "test_filter_all_pass",
                start.elapsed(),
                format!("Size {}: filtered data doesn't match original", size),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_filter_all_pass",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_filter_all_pass", start.elapsed())
}

/// Test 2: Filter where no elements pass (0% selectivity).
///
/// When no elements pass the filter, the output should be empty.
/// This tests the edge case where the predicate is always false.
fn test_filter_none_pass(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various sizes with 0% selectivity
    let sizes: Vec<usize> = vec![100, 1000, 10000, 100000];

    for size in sizes {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_filter_none_pass",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // All zeros mask - 0% pass
        let mask: Vec<u8> = vec![0; size];

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_filter_none_pass",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        // No elements should pass
        if ctx.device_row_count(&filtered) != 0 {
            return TestResult::error(
                "test_filter_none_pass",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected 0 (0% selectivity)",
                    size,
                    ctx.device_row_count(&filtered)
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_filter_none_pass",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_filter_none_pass", start.elapsed())
}

/// Test 3: Filter where half pass (50% selectivity).
///
/// Tests the common case where approximately half of elements pass,
/// which represents balanced predicate evaluation.
fn test_filter_half_pass(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various sizes with 50% selectivity
    let sizes: Vec<usize> = vec![100, 1000, 10000, 100000];

    for size in sizes {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_filter_half_pass",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Keep first half
        let mask: Vec<u8> = (0..size)
            .map(|i| if i < size / 2 { 1 } else { 0 })
            .collect();
        let expected_count = size / 2;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_filter_half_pass",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_filter_half_pass",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {} (50% selectivity)",
                    size,
                    ctx.device_row_count(&filtered),
                    expected_count
                ),
            );
        }

        // Download and verify
        let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_filter_half_pass",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify values are 0, 1, 2, ..., (size/2 - 1)
        for (i, &val) in filtered_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "test_filter_half_pass",
                    start.elapsed(),
                    format!("Size {}: filtered[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_filter_half_pass",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_filter_half_pass", start.elapsed())
}

/// Test 4: Filter with very few passing (1% selectivity).
///
/// Sparse predicates test the efficiency of compaction when most elements
/// are filtered out. This is common in highly selective queries.
fn test_sparse_predicate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various sizes with ~1% selectivity
    let sizes: Vec<usize> = vec![1000, 10000, 100000];

    for size in sizes {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_sparse_predicate",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Keep every 100th element (~1% selectivity)
        let mask: Vec<u8> = (0..size)
            .map(|i| if i % 100 == 0 { 1 } else { 0 })
            .collect();
        let expected_count: usize = mask.iter().map(|&m| m as usize).sum();

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_sparse_predicate",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_sparse_predicate",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {} (~1% selectivity)",
                    size,
                    ctx.device_row_count(&filtered),
                    expected_count
                ),
            );
        }

        // Download and verify
        let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_sparse_predicate",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify values are 0, 100, 200, ...
        for (i, &val) in filtered_data.iter().enumerate() {
            let expected = (i * 100) as u32;
            if val != expected {
                return TestResult::error(
                    "test_sparse_predicate",
                    start.elapsed(),
                    format!(
                        "Size {}: filtered[{}] = {}, expected {}",
                        size, i, val, expected
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sparse_predicate",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sparse_predicate", start.elapsed())
}

/// Test 5: Filter with most passing (99% selectivity).
///
/// Dense predicates test the efficiency when almost all elements pass.
/// This is common in queries with loose filters.
fn test_dense_predicate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various sizes with ~99% selectivity
    let sizes: Vec<usize> = vec![1000, 10000, 100000];

    for size in sizes {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_dense_predicate",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Keep all except every 100th element (~99% selectivity)
        let mask: Vec<u8> = (0..size)
            .map(|i| if i % 100 != 0 { 1 } else { 0 })
            .collect();
        let expected_count: usize = mask.iter().map(|&m| m as usize).sum();

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_dense_predicate",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_dense_predicate",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {} (~99% selectivity)",
                    size,
                    ctx.device_row_count(&filtered),
                    expected_count
                ),
            );
        }

        // Download and verify
        let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dense_predicate",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify no value is a multiple of 100
        for (i, &val) in filtered_data.iter().enumerate() {
            if val % 100 == 0 {
                return TestResult::error(
                    "test_dense_predicate",
                    start.elapsed(),
                    format!(
                        "Size {}: filtered[{}] = {} is a multiple of 100 (should be filtered)",
                        size, i, val
                    ),
                );
            }
        }

        // Verify the expected values are present
        let expected_values: Vec<u32> = (0..size as u32).filter(|&v| v % 100 != 0).collect();
        if filtered_data != expected_values {
            return TestResult::error(
                "test_dense_predicate",
                start.elapsed(),
                format!("Size {}: filtered data doesn't match expected values", size),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_dense_predicate",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_dense_predicate", start.elapsed())
}

/// Test 6: Alternating pass/fail pattern.
///
/// Tests the case where the predicate alternates between true and false,
/// which creates maximum divergence within warps and blocks.
fn test_alternating_predicate(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various sizes with alternating pattern
    let sizes: Vec<usize> = vec![100, 1000, 10000, 100000];

    for size in sizes {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_alternating_predicate",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Alternating mask: 1, 0, 1, 0, ...
        let mask: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let expected_count = (size + 1) / 2;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_alternating_predicate",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_alternating_predicate",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {} (alternating)",
                    size,
                    ctx.device_row_count(&filtered),
                    expected_count
                ),
            );
        }

        // Download and verify
        let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_alternating_predicate",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify values are 0, 2, 4, 6, ... (even indices)
        for (i, &val) in filtered_data.iter().enumerate() {
            let expected = (i * 2) as u32;
            if val != expected {
                return TestResult::error(
                    "test_alternating_predicate",
                    start.elapsed(),
                    format!(
                        "Size {}: filtered[{}] = {}, expected {}",
                        size, i, val, expected
                    ),
                );
            }
        }

        // Also test the opposite alternating pattern: 0, 1, 0, 1, ...
        let mask2: Vec<u8> = (0..size).map(|i| if i % 2 != 0 { 1 } else { 0 }).collect();
        let expected_count2 = size / 2;

        let filtered2 = match ctx.provider.filter_by_mask(&buffer, &mask2) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_alternating_predicate",
                    start.elapsed(),
                    format!("Size {}: second filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered2) != expected_count2 as u64 {
            return TestResult::error(
                "test_alternating_predicate",
                start.elapsed(),
                format!(
                    "Size {}: second filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered2),
                    expected_count2
                ),
            );
        }

        let filtered_data2 = match ctx.provider.download_column_u32(&filtered2, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_alternating_predicate",
                    start.elapsed(),
                    format!("Size {}: failed to download second filter: {}", size, e),
                )
            }
        };

        // Verify values are 1, 3, 5, 7, ... (odd indices)
        for (i, &val) in filtered_data2.iter().enumerate() {
            let expected = (i * 2 + 1) as u32;
            if val != expected {
                return TestResult::error(
                    "test_alternating_predicate",
                    start.elapsed(),
                    format!(
                        "Size {}: second filtered[{}] = {}, expected {}",
                        size, i, val, expected
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_alternating_predicate",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_alternating_predicate", start.elapsed())
}

/// Test 7: Random predicate distribution.
///
/// Tests the filter with pseudo-random predicate patterns that vary
/// selectivity and distribution across the data.
fn test_random_predicate_distribution(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various sizes with pseudo-random patterns
    let sizes: Vec<usize> = vec![1000, 10000, 100000];

    for size in sizes {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_random_predicate_distribution",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Pattern 1: LCG-based pseudo-random (~50% selectivity)
        // Uses linear congruential generator for deterministic randomness
        let mask1: Vec<u8> = (0..size)
            .map(|i| {
                let hash = ((i as u64 * 1103515245 + 12345) >> 16) & 0x7FFF;
                if hash % 2 == 0 {
                    1
                } else {
                    0
                }
            })
            .collect();
        let expected_count1: usize = mask1.iter().map(|&m| m as usize).sum();

        let filtered1 = match ctx.provider.filter_by_mask(&buffer, &mask1) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_random_predicate_distribution",
                    start.elapsed(),
                    format!("Size {}: first filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered1) != expected_count1 as u64 {
            return TestResult::error(
                "test_random_predicate_distribution",
                start.elapsed(),
                format!(
                    "Size {}: first filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered1),
                    expected_count1
                ),
            );
        }

        // Verify filtered values match expected
        let filtered_data1 = match ctx.provider.download_column_u32(&filtered1, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_random_predicate_distribution",
                    start.elapsed(),
                    format!("Size {}: failed to download first filter: {}", size, e),
                )
            }
        };

        let expected_values1: Vec<u32> = (0..size)
            .filter(|&i| mask1[i] == 1)
            .map(|i| i as u32)
            .collect();

        if filtered_data1 != expected_values1 {
            return TestResult::error(
                "test_random_predicate_distribution",
                start.elapsed(),
                format!("Size {}: first filter values mismatch", size),
            );
        }

        // Pattern 2: Different LCG parameters (~30% selectivity)
        let mask2: Vec<u8> = (0..size)
            .map(|i| {
                // Use wrapping arithmetic so debug builds don't panic on overflow.
                let hash = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let digit = (hash >> 40) % 10;
                if digit < 3 {
                    1
                } else {
                    0
                }
            })
            .collect();
        let expected_count2: usize = mask2.iter().map(|&m| m as usize).sum();

        let filtered2 = match ctx.provider.filter_by_mask(&buffer, &mask2) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_random_predicate_distribution",
                    start.elapsed(),
                    format!("Size {}: second filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered2) != expected_count2 as u64 {
            return TestResult::error(
                "test_random_predicate_distribution",
                start.elapsed(),
                format!(
                    "Size {}: second filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered2),
                    expected_count2
                ),
            );
        }

        // Pattern 3: Burst pattern - groups of passing followed by groups of failing
        let mask3: Vec<u8> = (0..size)
            .map(|i| {
                let group = (i / 17) % 5; // Groups of 17, 5 phases
                if group < 2 {
                    1
                } else {
                    0
                } // 2/5 = 40% selectivity
            })
            .collect();
        let expected_count3: usize = mask3.iter().map(|&m| m as usize).sum();

        let filtered3 = match ctx.provider.filter_by_mask(&buffer, &mask3) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_random_predicate_distribution",
                    start.elapsed(),
                    format!("Size {}: third filter failed: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered3) != expected_count3 as u64 {
            return TestResult::error(
                "test_random_predicate_distribution",
                start.elapsed(),
                format!(
                    "Size {}: third filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered3),
                    expected_count3
                ),
            );
        }

        let filtered_data3 = match ctx.provider.download_column_u32(&filtered3, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_random_predicate_distribution",
                    start.elapsed(),
                    format!("Size {}: failed to download third filter: {}", size, e),
                )
            }
        };

        let expected_values3: Vec<u32> = (0..size)
            .filter(|&i| mask3[i] == 1)
            .map(|i| i as u32)
            .collect();

        if filtered_data3 != expected_values3 {
            return TestResult::error(
                "test_random_predicate_distribution",
                start.elapsed(),
                format!("Size {}: third filter values mismatch", size),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_random_predicate_distribution",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_random_predicate_distribution", start.elapsed())
}
