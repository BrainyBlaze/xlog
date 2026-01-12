//! Category 23: Testing blind spots
//!
//! Tests commonly overlooked edge cases including non-power-of-two sizes,
//! misaligned boundaries, near-overflow indices, alternating patterns,
//! and empty/single element cases.

use crate::harness::{CategoryResult, TestResult, TestContext};
use crate::harness::generators::SizeGen;
use std::collections::HashSet;
use std::time::Instant;
use xlog_core::{Schema, ScalarType};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c23_blind_spots");
    let start = Instant::now();

    results.add_result(test_non_power_of_two_sizes(ctx));
    results.add_result(test_misaligned_boundaries(ctx));
    results.add_result(test_near_overflow_indices(ctx));
    results.add_result(test_alternating_patterns(ctx));
    results.add_result(test_empty_and_single(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Prime and odd sizes across all operations.
///
/// Verifies that operations work correctly with sizes that are not powers of two,
/// including prime numbers which are worst-case for many algorithms.
fn test_non_power_of_two_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test a selection of prime sizes from SizeGen
    let prime_sizes: Vec<usize> = vec![7, 13, 31, 67, 127, 251, 509, 1021, 2039, 4093];

    for &size in &prime_sizes {
        // Create data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Failed to create buffer for size {}: {}", size, e),
                )
            }
        };

        // Test sort
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Sort failed for prime size {}: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_non_power_of_two_sizes",
                start.elapsed(),
                format!(
                    "Sort for size {} returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }

        // Test filter with ~50% selectivity
        let mask: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let expected_filtered = mask.iter().map(|&m| m as usize).sum::<usize>();

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Filter failed for prime size {}: {}", size, e),
                )
            }
        };

        if filtered.num_rows != expected_filtered as u64 {
            return TestResult::error(
                "test_non_power_of_two_sizes",
                start.elapsed(),
                format!(
                    "Filter for size {} returned {} rows, expected {}",
                    size, filtered.num_rows, expected_filtered
                ),
            );
        }

        // Test dedup with some duplicates
        let dedup_schema = Schema::new(vec![
            ("key".to_string(), ScalarType::U32),
            ("val".to_string(), ScalarType::U32),
        ]);
        let dedup_keys: Vec<u32> = (0..size as u32).map(|i| i % ((size / 2).max(1) as u32)).collect();
        let dedup_vals: Vec<u32> = (0..size as u32).collect();

        let dedup_buffer = match ctx.provider.create_buffer_from_u32_columns(
            &[&dedup_keys, &dedup_vals],
            dedup_schema.clone(),
        ) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Failed to create dedup buffer for size {}: {}", size, e),
                )
            }
        };

        let deduped = match ctx.provider.dedup(&dedup_buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Dedup failed for prime size {}: {}", size, e),
                )
            }
        };

        // Should have at most (size/2) unique keys, but at least 1
        let unique_keys: HashSet<u32> = dedup_keys.iter().copied().collect();
        if deduped.num_rows != unique_keys.len() as u64 {
            return TestResult::error(
                "test_non_power_of_two_sizes",
                start.elapsed(),
                format!(
                    "Dedup for size {} returned {} rows, expected {}",
                    size, deduped.num_rows, unique_keys.len()
                ),
            );
        }
    }

    // Also test odd sizes that are not prime
    let odd_sizes: Vec<usize> = vec![15, 21, 33, 45, 63, 99, 255, 511, 1023];

    for &size in &odd_sizes {
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Failed to create buffer for odd size {}: {}", size, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Sort failed for odd size {}: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_non_power_of_two_sizes",
                start.elapsed(),
                format!(
                    "Sort for odd size {} returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_non_power_of_two_sizes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_non_power_of_two_sizes", start.elapsed())
}

/// Test 2: Sizes that don't align to warp/block boundaries.
///
/// Verifies that operations work correctly with sizes that cross warp (32)
/// and block (256) boundaries at various offsets.
fn test_misaligned_boundaries(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test sizes around warp boundary (32)
    let warp_related: Vec<usize> = SizeGen::warp_related();

    for &size in &warp_related {
        let data: Vec<u32> = (0..size as u32).rev().collect(); // Reverse sorted for sorting test

        let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_misaligned_boundaries",
                    start.elapsed(),
                    format!("Failed to create buffer for warp size {}: {}", size, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_misaligned_boundaries",
                    start.elapsed(),
                    format!("Sort failed for warp-related size {}: {}", size, e),
                )
            }
        };

        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_misaligned_boundaries",
                    start.elapsed(),
                    format!("Failed to download sorted data for size {}: {}", size, e),
                )
            }
        };

        // Verify sorted
        for i in 1..sorted_data.len() {
            if sorted_data[i] < sorted_data[i - 1] {
                return TestResult::error(
                    "test_misaligned_boundaries",
                    start.elapsed(),
                    format!(
                        "Sort order incorrect at index {} for warp size {}: {} < {}",
                        i, size, sorted_data[i], sorted_data[i - 1]
                    ),
                );
            }
        }
    }

    // Test sizes around block boundary (256)
    let block_related: Vec<usize> = SizeGen::block_related();

    for &size in &block_related {
        let data: Vec<u32> = (0..size as u32).collect();

        // Create filter mask with alternating pattern
        let mask: Vec<u8> = (0..size).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let expected_count: usize = mask.iter().map(|&m| m as usize).sum();

        let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_misaligned_boundaries",
                    start.elapsed(),
                    format!("Failed to create buffer for block size {}: {}", size, e),
                )
            }
        };

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_misaligned_boundaries",
                    start.elapsed(),
                    format!("Filter failed for block-related size {}: {}", size, e),
                )
            }
        };

        if filtered.num_rows != expected_count as u64 {
            return TestResult::error(
                "test_misaligned_boundaries",
                start.elapsed(),
                format!(
                    "Filter for block size {} returned {} rows, expected {}",
                    size, filtered.num_rows, expected_count
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_misaligned_boundaries",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_misaligned_boundaries", start.elapsed())
}

/// Test 3: Sizes near u32/i32 limits (practical sizes).
///
/// Verifies operations with large indices that approach practical limits
/// without exceeding available memory.
fn test_near_overflow_indices(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Test with sizes that have values near overflow boundaries
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test u32 max value handling in sort
    let edge_values: Vec<u32> = vec![
        0,
        1,
        u32::MAX - 2,
        u32::MAX - 1,
        u32::MAX,
        u32::MAX / 2,
        u32::MAX / 2 + 1,
    ];

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&edge_values, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Failed to create buffer with edge values: {}", e),
            )
        }
    };

    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Sort with edge u32 values failed: {}", e),
            )
        }
    };

    let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Failed to download sorted edge values: {}", e),
            )
        }
    };

    // Verify sort order
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} < {}",
                    i, sorted_data[i], sorted_data[i - 1]
                ),
            );
        }
    }

    // Verify u32::MAX is last
    if sorted_data[sorted_data.len() - 1] != u32::MAX {
        return TestResult::error(
            "test_near_overflow_indices",
            start.elapsed(),
            format!(
                "Last element should be u32::MAX ({}), got {}",
                u32::MAX,
                sorted_data[sorted_data.len() - 1]
            ),
        );
    }

    // Test i64 values near overflow boundaries
    let i64_schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);
    let i64_edge_values: Vec<i64> = vec![
        i64::MIN,
        i64::MIN + 1,
        -1,
        0,
        1,
        i64::MAX - 1,
        i64::MAX,
    ];

    let i64_buffer = match ctx.provider.create_buffer_from_i64_slice(&i64_edge_values, i64_schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Failed to create i64 buffer with edge values: {}", e),
            )
        }
    };

    let i64_sorted = match ctx.provider.sort(&i64_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Sort with edge i64 values failed: {}", e),
            )
        }
    };

    let i64_sorted_data = match ctx.provider.download_column_i64(&i64_sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Failed to download sorted i64 edge values: {}", e),
            )
        }
    };

    // Verify i64::MIN is first
    if i64_sorted_data[0] != i64::MIN {
        return TestResult::error(
            "test_near_overflow_indices",
            start.elapsed(),
            format!(
                "First i64 element should be i64::MIN ({}), got {}",
                i64::MIN, i64_sorted_data[0]
            ),
        );
    }

    // Verify i64::MAX is last
    if i64_sorted_data[i64_sorted_data.len() - 1] != i64::MAX {
        return TestResult::error(
            "test_near_overflow_indices",
            start.elapsed(),
            format!(
                "Last i64 element should be i64::MAX ({}), got {}",
                i64::MAX,
                i64_sorted_data[i64_sorted_data.len() - 1]
            ),
        );
    }

    // Test with moderately large sizes that have specific index patterns
    let large_size: usize = 100000;
    let large_data: Vec<u32> = (0..large_size as u32).collect();

    let large_buffer = match ctx.provider.create_buffer_from_u32_slice(&large_data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Failed to create large buffer: {}", e),
            )
        }
    };

    // Filter to keep last few elements (high indices)
    let mask: Vec<u8> = (0..large_size).map(|i| {
        if i >= large_size - 100 { 1 } else { 0 }
    }).collect();

    let filtered = match ctx.provider.filter_by_mask(&large_buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Filter with high indices failed: {}", e),
            )
        }
    };

    if filtered.num_rows != 100 {
        return TestResult::error(
            "test_near_overflow_indices",
            start.elapsed(),
            format!("Filter should return 100 rows, got {}", filtered.num_rows),
        );
    }

    let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!("Failed to download filtered data: {}", e),
            )
        }
    };

    // Verify filtered values are from the end
    for (i, &val) in filtered_data.iter().enumerate() {
        let expected = (large_size - 100 + i) as u32;
        if val != expected {
            return TestResult::error(
                "test_near_overflow_indices",
                start.elapsed(),
                format!(
                    "Filtered value {} at index {} should be {}",
                    val, i, expected
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_near_overflow_indices",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_near_overflow_indices", start.elapsed())
}

/// Test 4: Data with maximum divergence patterns.
///
/// Verifies operations handle data patterns that cause maximum thread
/// divergence (alternating, strided, random-looking deterministic).
fn test_alternating_patterns(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;

    // Pattern 1: Alternating 0 and MAX
    let alternating_extreme: Vec<u32> = (0..SIZE).map(|i| {
        if i % 2 == 0 { 0 } else { u32::MAX }
    }).collect();

    let buffer1 = match ctx.provider.create_buffer_from_u32_slice(&alternating_extreme, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Failed to create alternating extreme buffer: {}", e),
            )
        }
    };

    let sorted1 = match ctx.provider.sort(&buffer1, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Sort failed for alternating extreme pattern: {}", e),
            )
        }
    };

    let sorted_data1 = match ctx.provider.download_column_u32(&sorted1, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Failed to download alternating extreme sorted: {}", e),
            )
        }
    };

    // First half should be 0s, second half should be MAX
    let zeros_count = sorted_data1.iter().filter(|&&v| v == 0).count();
    let max_count = sorted_data1.iter().filter(|&&v| v == u32::MAX).count();

    if zeros_count != SIZE / 2 || max_count != SIZE / 2 {
        return TestResult::error(
            "test_alternating_patterns",
            start.elapsed(),
            format!(
                "Alternating extreme sort: expected {}/{} zeros/MAX, got {}/{}",
                SIZE / 2,
                SIZE / 2,
                zeros_count,
                max_count
            ),
        );
    }

    // Pattern 2: Alternating bits (checkerboard)
    let checkerboard: Vec<u32> = (0..SIZE).map(|i| {
        if i % 2 == 0 { 0xAAAAAAAA } else { 0x55555555 }
    }).collect();

    let buffer2 = match ctx.provider.create_buffer_from_u32_slice(&checkerboard, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Failed to create checkerboard buffer: {}", e),
            )
        }
    };

    // Test filter with alternating mask
    let alternating_mask: Vec<u8> = (0..SIZE).map(|i| (i % 2) as u8).collect();
    let filtered2 = match ctx.provider.filter_by_mask(&buffer2, &alternating_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Filter failed for checkerboard pattern: {}", e),
            )
        }
    };

    // Should keep odd indices (0x55555555 values)
    if filtered2.num_rows != (SIZE / 2) as u64 {
        return TestResult::error(
            "test_alternating_patterns",
            start.elapsed(),
            format!(
                "Alternating filter should keep {} rows, got {}",
                SIZE / 2,
                filtered2.num_rows
            ),
        );
    }

    // Pattern 3: Stride pattern (every 3rd element differs)
    let stride3: Vec<u32> = (0..SIZE).map(|i| {
        match i % 3 {
            0 => 0,
            1 => 100,
            _ => 200,
        }
    }).collect();

    let _buffer3 = match ctx.provider.create_buffer_from_u32_slice(&stride3, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Failed to create stride3 buffer: {}", e),
            )
        }
    };

    // Test dedup - should result in exactly 3 unique values
    let dedup_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);
    let stride3_vals: Vec<u32> = (0..SIZE as u32).collect();

    let dedup_buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&stride3, &stride3_vals],
        dedup_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Failed to create stride3 dedup buffer: {}", e),
            )
        }
    };

    let deduped3 = match ctx.provider.dedup(&dedup_buffer, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Dedup failed for stride3 pattern: {}", e),
            )
        }
    };

    if deduped3.num_rows != 3 {
        return TestResult::error(
            "test_alternating_patterns",
            start.elapsed(),
            format!("Stride3 dedup should return 3 unique rows, got {}", deduped3.num_rows),
        );
    }

    // Pattern 4: Pseudo-random looking but deterministic (LCG pattern)
    let lcg_pattern: Vec<u32> = {
        let mut vals = Vec::with_capacity(SIZE);
        let mut x: u32 = 1;
        for _ in 0..SIZE {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            vals.push(x);
        }
        vals
    };

    let buffer4 = match ctx.provider.create_buffer_from_u32_slice(&lcg_pattern, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Failed to create LCG pattern buffer: {}", e),
            )
        }
    };

    let sorted4 = match ctx.provider.sort(&buffer4, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Sort failed for LCG pattern: {}", e),
            )
        }
    };

    let sorted_data4 = match ctx.provider.download_column_u32(&sorted4, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!("Failed to download LCG sorted: {}", e),
            )
        }
    };

    // Verify sorted
    for i in 1..sorted_data4.len() {
        if sorted_data4[i] < sorted_data4[i - 1] {
            return TestResult::error(
                "test_alternating_patterns",
                start.elapsed(),
                format!(
                    "LCG sort order incorrect at index {}: {} < {}",
                    i, sorted_data4[i], sorted_data4[i - 1]
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_alternating_patterns",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_alternating_patterns", start.elapsed())
}

/// Test 5: Empty and single-element cases for all operations.
///
/// Verifies that all operations handle the edge cases of empty buffers
/// and single-element buffers correctly.
fn test_empty_and_single(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test empty buffer
    let empty_data: Vec<u32> = vec![];

    let empty_buffer = match ctx.provider.create_buffer_from_u32_slice(&empty_data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Failed to create empty buffer: {}", e),
            )
        }
    };

    // Sort empty
    let sorted_empty = match ctx.provider.sort(&empty_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Sort on empty buffer failed: {}", e),
            )
        }
    };

    if sorted_empty.num_rows != 0 {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Sort on empty buffer should return 0 rows, got {}", sorted_empty.num_rows),
        );
    }

    // Filter empty
    let empty_mask: Vec<u8> = vec![];
    let filtered_empty = match ctx.provider.filter_by_mask(&empty_buffer, &empty_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Filter on empty buffer failed: {}", e),
            )
        }
    };

    if filtered_empty.num_rows != 0 {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Filter on empty buffer should return 0 rows, got {}", filtered_empty.num_rows),
        );
    }

    // Test single element
    let single_data: Vec<u32> = vec![42];

    let single_buffer = match ctx.provider.create_buffer_from_u32_slice(&single_data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Failed to create single-element buffer: {}", e),
            )
        }
    };

    // Sort single
    let sorted_single = match ctx.provider.sort(&single_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Sort on single-element buffer failed: {}", e),
            )
        }
    };

    if sorted_single.num_rows != 1 {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Sort on single-element buffer should return 1 row, got {}", sorted_single.num_rows),
        );
    }

    let sorted_single_data = match ctx.provider.download_column_u32(&sorted_single, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Failed to download single sorted: {}", e),
            )
        }
    };

    if sorted_single_data != vec![42] {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Single element should remain 42 after sort, got {:?}", sorted_single_data),
        );
    }

    // Filter single - keep
    let keep_mask: Vec<u8> = vec![1];
    let filtered_keep = match ctx.provider.filter_by_mask(&single_buffer, &keep_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Filter (keep) on single-element buffer failed: {}", e),
            )
        }
    };

    if filtered_keep.num_rows != 1 {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Filter (keep) on single-element should return 1 row, got {}", filtered_keep.num_rows),
        );
    }

    // Filter single - discard
    let discard_mask: Vec<u8> = vec![0];
    let filtered_discard = match ctx.provider.filter_by_mask(&single_buffer, &discard_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Filter (discard) on single-element buffer failed: {}", e),
            )
        }
    };

    if filtered_discard.num_rows != 0 {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Filter (discard) on single-element should return 0 rows, got {}", filtered_discard.num_rows),
        );
    }

    // Dedup single element
    let dedup_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    let single_dedup_buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&single_data, &single_data],
        dedup_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Failed to create single dedup buffer: {}", e),
            )
        }
    };

    let deduped_single = match ctx.provider.dedup(&single_dedup_buffer, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Dedup on single-element buffer failed: {}", e),
            )
        }
    };

    if deduped_single.num_rows != 1 {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Dedup on single-element should return 1 row, got {}", deduped_single.num_rows),
        );
    }

    // Join with empty/single
    let join_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    let left_single = match ctx.provider.create_buffer_from_u32_columns(
        &[&vec![42u32], &vec![100u32]],
        join_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Failed to create left single buffer: {}", e),
            )
        }
    };

    let right_matching = match ctx.provider.create_buffer_from_u32_columns(
        &[&vec![42u32], &vec![200u32]],
        join_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Failed to create right matching buffer: {}", e),
            )
        }
    };

    // Join two single-element tables with matching key
    let joined_match = match ctx.provider.hash_join(&left_single, &right_matching, &[0], &[0]) {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Join with single matching elements failed: {}", e),
            )
        }
    };

    if joined_match.num_rows != 1 {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Join of single matching elements should return 1 row, got {}", joined_match.num_rows),
        );
    }

    // Join two single-element tables with non-matching key
    let right_nonmatch = match ctx.provider.create_buffer_from_u32_columns(
        &[&vec![99u32], &vec![200u32]],
        join_schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Failed to create right non-matching buffer: {}", e),
            )
        }
    };

    let joined_nomatch = match ctx.provider.hash_join(&left_single, &right_nonmatch, &[0], &[0]) {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_empty_and_single",
                start.elapsed(),
                format!("Join with single non-matching elements failed: {}", e),
            )
        }
    };

    if joined_nomatch.num_rows != 0 {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Join of single non-matching elements should return 0 rows, got {}", joined_nomatch.num_rows),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_empty_and_single",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_empty_and_single", start.elapsed())
}
