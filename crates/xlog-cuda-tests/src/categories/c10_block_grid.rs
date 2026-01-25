//! Category 10: Block and Grid Coordination
//!
//! Tests cross-block and grid-level behavior including single-block operations,
//! multi-block operations, block boundary correctness, grid-stride loops, and
//! cross-block data patterns.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c10_block_grid");
    let start = Instant::now();

    results.add_result(test_single_block_operations(ctx));
    results.add_result(test_multi_block_operations(ctx));
    results.add_result(test_block_boundary_correctness(ctx));
    results.add_result(test_grid_stride_correctness(ctx));
    results.add_result(test_cross_block_data_patterns(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Sizes that fit in one block (1-256).
///
/// Tests operations that fit within a single thread block, verifying that
/// intra-block operations work correctly without inter-block communication.
fn test_single_block_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Sizes that fit in a single block (assuming 256 threads/block)
    let sizes: Vec<usize> = vec![1, 16, 32, 64, 128, 192, 256];

    for size in sizes {
        // Create reverse-sorted keys
        let keys: Vec<u32> = (0..size as u32).rev().collect();
        let vals: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_single_block_operations",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Sort
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_single_block_operations",
                    start.elapsed(),
                    format!("Size {}: sort failed: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_single_block_operations",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }

        // Download and verify
        let sorted_keys = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_single_block_operations",
                    start.elapsed(),
                    format!("Size {}: failed to download keys: {}", size, e),
                )
            }
        };

        for (i, &key) in sorted_keys.iter().enumerate() {
            if key != i as u32 {
                return TestResult::error(
                    "test_single_block_operations",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted_keys[{}] = {}, expected {}",
                        size, i, key, i
                    ),
                );
            }
        }

        // Test filter in single block
        let mask: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let expected_count = (size + 1) / 2;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_single_block_operations",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if filtered.num_rows != expected_count as u64 {
            return TestResult::error(
                "test_single_block_operations",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size, filtered.num_rows, expected_count
                ),
            );
        }

        // Test dedup in single block
        let deduped = match ctx.provider.dedup(&buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_single_block_operations",
                    start.elapsed(),
                    format!("Size {}: dedup failed: {}", size, e),
                )
            }
        };

        // All values unique
        if deduped.num_rows != size as u64 {
            return TestResult::error(
                "test_single_block_operations",
                start.elapsed(),
                format!(
                    "Size {}: dedup returned {} rows, expected {}",
                    size, deduped.num_rows, size
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_single_block_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_single_block_operations", start.elapsed())
}

/// Test 2: Sizes requiring multiple blocks (1K, 10K, 100K, 1M).
///
/// Tests operations that require multiple thread blocks, verifying that
/// inter-block communication and coordination work correctly.
fn test_multi_block_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes requiring multiple blocks
    let sizes: Vec<usize> = vec![1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        // Create reverse-sorted data
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_multi_block_operations",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Sort requires cross-block coordination
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_multi_block_operations",
                    start.elapsed(),
                    format!("Size {}: sort failed: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_multi_block_operations",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }

        // Download and verify key positions
        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_multi_block_operations",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify first, middle, and last
        if sorted_data[0] != 0 {
            return TestResult::error(
                "test_multi_block_operations",
                start.elapsed(),
                format!(
                    "Size {}: first element is {}, expected 0",
                    size, sorted_data[0]
                ),
            );
        }

        let mid = size / 2;
        if sorted_data[mid] != mid as u32 {
            return TestResult::error(
                "test_multi_block_operations",
                start.elapsed(),
                format!(
                    "Size {}: middle element [{}] is {}, expected {}",
                    size, mid, sorted_data[mid], mid
                ),
            );
        }

        let last = size - 1;
        if sorted_data[last] != last as u32 {
            return TestResult::error(
                "test_multi_block_operations",
                start.elapsed(),
                format!(
                    "Size {}: last element is {}, expected {}",
                    size, sorted_data[last], last
                ),
            );
        }

        // Verify sorted order by sampling
        let mut prev = sorted_data[0];
        for &idx in &[
            100,
            1000,
            size / 4,
            size / 2,
            size * 3 / 4,
            size - 100,
            size - 1,
        ] {
            if idx < size {
                if sorted_data[idx] < prev {
                    return TestResult::error(
                        "test_multi_block_operations",
                        start.elapsed(),
                        format!(
                            "Size {}: not sorted at {}: {} > {}",
                            size, idx, prev, sorted_data[idx]
                        ),
                    );
                }
                prev = sorted_data[idx];
            }
        }

        // Test filter across multiple blocks
        let mask: Vec<u8> = (0..size).map(|i| if i % 10 == 0 { 1 } else { 0 }).collect();
        let expected_count = (size + 9) / 10;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_multi_block_operations",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if filtered.num_rows != expected_count as u64 {
            return TestResult::error(
                "test_multi_block_operations",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size, filtered.num_rows, expected_count
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_multi_block_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_multi_block_operations", start.elapsed())
}

/// Test 3: Test at block boundaries (255, 256, 257, 511, 512, 513).
///
/// Block boundaries are common sources of off-by-one errors. This test
/// verifies correct behavior at exact block boundaries and +-1.
fn test_block_boundary_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes at block boundaries (assuming 256 threads/block)
    let sizes: Vec<usize> = vec![255, 256, 257, 511, 512, 513];

    for size in sizes {
        // Create reverse-sorted data to maximize data movement
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_block_boundary_correctness",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Sort
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_block_boundary_correctness",
                    start.elapsed(),
                    format!("Size {}: sort failed: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_block_boundary_correctness",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }

        // Download and verify every element
        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_block_boundary_correctness",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "test_block_boundary_correctness",
                    start.elapsed(),
                    format!("Size {}: sorted[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }

        // Test filter at block boundary
        // Create a mask that has boundary transition
        let mask: Vec<u8> = (0..size)
            .map(|i| {
                // Keep first element of each potential block
                if i % 256 == 0 || i % 256 == 255 {
                    1
                } else {
                    0
                }
            })
            .collect();
        let expected_count: usize = mask.iter().map(|&m| m as usize).sum();

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_block_boundary_correctness",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if filtered.num_rows != expected_count as u64 {
            return TestResult::error(
                "test_block_boundary_correctness",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size, filtered.num_rows, expected_count
                ),
            );
        }

        // Verify filtered values
        let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_block_boundary_correctness",
                    start.elapsed(),
                    format!("Size {}: failed to download filtered: {}", size, e),
                )
            }
        };

        let expected_values: Vec<u32> = (0..size)
            .filter(|&i| mask[i] == 1)
            .map(|i| data[i])
            .collect();

        if filtered_data != expected_values {
            return TestResult::error(
                "test_block_boundary_correctness",
                start.elapsed(),
                format!("Size {}: filtered data mismatch", size),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_block_boundary_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_block_boundary_correctness", start.elapsed())
}

/// Test 4: Very large sizes (5M, 10M) requiring grid-stride loops.
///
/// When the number of elements exceeds the maximum grid size, kernels must
/// use grid-stride loops. This test verifies that grid-stride behavior is correct.
fn test_grid_stride_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Large sizes requiring grid-stride loops
    let sizes: Vec<usize> = vec![5_000_000, 10_000_000];

    for size in sizes {
        // Create data with repeating pattern (modulo) to limit value range
        let data: Vec<u32> = (0..size).map(|i| (i % 10000) as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_grid_stride_correctness",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Sort exercises grid-stride loops
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_grid_stride_correctness",
                    start.elapsed(),
                    format!("Size {}: sort failed: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_grid_stride_correctness",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }

        // Download and verify
        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_grid_stride_correctness",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify sorted order
        if sorted_data[0] != 0 {
            return TestResult::error(
                "test_grid_stride_correctness",
                start.elapsed(),
                format!(
                    "Size {}: first element is {}, expected 0",
                    size, sorted_data[0]
                ),
            );
        }

        // Last element should be 9999 (max of modulo 10000)
        let last = size - 1;
        if sorted_data[last] != 9999 {
            return TestResult::error(
                "test_grid_stride_correctness",
                start.elapsed(),
                format!(
                    "Size {}: last element is {}, expected 9999",
                    size, sorted_data[last]
                ),
            );
        }

        // Verify sorted order by sampling
        let mut prev = sorted_data[0];
        for idx in [
            1000,
            100000,
            1000000,
            2500000,
            4000000,
            size - 1000,
            size - 1,
        ] {
            if idx < size {
                if sorted_data[idx] < prev {
                    return TestResult::error(
                        "test_grid_stride_correctness",
                        start.elapsed(),
                        format!(
                            "Size {}: not sorted at {}: {} > {}",
                            size, idx, prev, sorted_data[idx]
                        ),
                    );
                }
                prev = sorted_data[idx];
            }
        }

        // Verify value counts - each value 0-9999 should appear size/10000 times
        let count_per_value = size / 10000;
        // Check boundaries - position of first occurrence of value 1
        let first_one_pos = count_per_value;
        if sorted_data[first_one_pos] > 1 {
            return TestResult::error(
                "test_grid_stride_correctness",
                start.elapsed(),
                format!(
                    "Size {}: expected value <= 1 at position {}, got {}",
                    size, first_one_pos, sorted_data[first_one_pos]
                ),
            );
        }

        // Test filter across grid-stride iterations
        let mask: Vec<u8> = (0..size)
            .map(|i| if i % 100 == 0 { 1 } else { 0 })
            .collect();
        let expected_count = (size + 99) / 100;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_grid_stride_correctness",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if filtered.num_rows != expected_count as u64 {
            return TestResult::error(
                "test_grid_stride_correctness",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size, filtered.num_rows, expected_count
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_grid_stride_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_grid_stride_correctness", start.elapsed())
}

/// Test 5: Data patterns that span block boundaries.
///
/// Tests data patterns that specifically cross block boundaries to ensure
/// that cross-block data dependencies are handled correctly.
fn test_cross_block_data_patterns(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    // Pattern 1: Keys that span multiple blocks with matches across boundaries
    const SIZE: usize = 2048; // 8 blocks of 256

    // Left table: sequential keys
    let left_keys: Vec<u32> = (0..SIZE as u32).collect();
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k * 10).collect();

    // Right table: keys at block boundaries (every 256th key, plus neighbors)
    let mut right_keys: Vec<u32> = Vec::new();
    let mut right_vals: Vec<u32> = Vec::new();
    for block in 0..8 {
        let boundary = (block * 256) as u32;
        if boundary > 0 {
            right_keys.push(boundary - 1);
            right_vals.push((boundary - 1) * 100);
        }
        right_keys.push(boundary);
        right_vals.push(boundary * 100);
        if boundary + 1 < SIZE as u32 {
            right_keys.push(boundary + 1);
            right_vals.push((boundary + 1) * 100);
        }
    }

    let left_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Failed to create left buffer: {}", e),
            )
        }
    };

    let right_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&right_keys, &right_vals], right_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
            )
        }
    };

    // Hash join with keys spanning block boundaries
    let joined = match ctx
        .provider
        .hash_join(&left_buffer, &right_buffer, &[0], &[0])
    {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Hash join failed: {}", e),
            )
        }
    };

    // Verify join count equals right table size (all right keys have matches)
    if joined.num_rows != right_keys.len() as u64 {
        return TestResult::error(
            "test_cross_block_data_patterns",
            start.elapsed(),
            format!(
                "Join returned {} rows, expected {}",
                joined.num_rows,
                right_keys.len()
            ),
        );
    }

    // Download and verify join results
    let joined_keys = match ctx.provider.download_column_u32(&joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Failed to download joined keys: {}", e),
            )
        }
    };

    let joined_lvals = match ctx.provider.download_column_u32(&joined, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Failed to download joined lvals: {}", e),
            )
        }
    };

    let joined_rvals = match ctx.provider.download_column_u32(&joined, 2) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Failed to download joined rvals: {}", e),
            )
        }
    };

    // Verify each join result
    for i in 0..joined.num_rows as usize {
        let key = joined_keys[i];
        let lval = joined_lvals[i];
        let rval = joined_rvals[i];

        // lval should be key * 10
        let expected_lval = key * 10;
        if lval != expected_lval {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!(
                    "Row {}: lval {} doesn't match expected {} for key {}",
                    i, lval, expected_lval, key
                ),
            );
        }

        // rval should be key * 100
        let expected_rval = key * 100;
        if rval != expected_rval {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!(
                    "Row {}: rval {} doesn't match expected {} for key {}",
                    i, rval, expected_rval, key
                ),
            );
        }
    }

    // Pattern 2: Sort with data that creates cross-block dependencies
    let sort_schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Create data where each block has values that belong in other blocks
    let cross_block_data: Vec<u32> = (0..SIZE)
        .map(|i| {
            let block = i / 256;
            let lane = i % 256;
            // Interleave: even lanes get high values, odd lanes get low values
            if lane % 2 == 0 {
                ((7 - block) * 256 + lane) as u32
            } else {
                (block * 256 + lane) as u32
            }
        })
        .collect();

    let sort_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&cross_block_data, sort_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Failed to create cross-block sort buffer: {}", e),
            )
        }
    };

    let sorted = match ctx.provider.sort(&sort_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Sort of cross-block data failed: {}", e),
            )
        }
    };

    let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!("Failed to download sorted cross-block data: {}", e),
            )
        }
    };

    // Verify sorted order
    for i in 1..SIZE {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_cross_block_data_patterns",
                start.elapsed(),
                format!(
                    "Cross-block sort: not sorted at {}: {} > {}",
                    i,
                    sorted_data[i - 1],
                    sorted_data[i]
                ),
            );
        }
    }

    // Verify same values (sorted)
    let mut expected = cross_block_data.clone();
    expected.sort();
    if sorted_data != expected {
        return TestResult::error(
            "test_cross_block_data_patterns",
            start.elapsed(),
            "Cross-block sort: values not preserved".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cross_block_data_patterns",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_cross_block_data_patterns", start.elapsed())
}
