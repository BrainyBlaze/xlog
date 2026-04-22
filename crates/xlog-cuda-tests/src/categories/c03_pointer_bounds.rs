//! Category 3: Pointer arithmetic, indexing, bounds edge cases
//!
//! This category tests:
//! - Edge case sizes from SizeGen (including 0, 1, powers of 2, and off-by-one)
//! - Off-by-one errors in filter (last element handling)
//! - Off-by-one errors in sort (element preservation)
//! - Grid-stride loops for large data sizes
//! - Tail handling for partial warps/blocks
//! - Boundary index correctness (first/last element selection)
//! - Multi-column stride handling

use crate::harness::generators::SizeGen;
use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub(crate) fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c03_pointer_bounds");
    let start = Instant::now();

    results.add_result(test_edge_case_sizes(ctx));
    results.add_result(test_off_by_one_filter(ctx));
    results.add_result(test_off_by_one_sort(ctx));
    results.add_result(test_grid_stride_loop(ctx));
    results.add_result(test_tail_handling_filter(ctx));
    results.add_result(test_tail_handling_sort(ctx));
    results.add_result(test_boundary_indices(ctx));
    results.add_result(test_multi_column_strides(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Test all edge case sizes from SizeGen.
///
/// Uses SizeGen::edge_cases() which includes 0, 1, 2, 3, 7, 15, 16, 17, etc.
/// up to 65537. For each non-zero size, creates a buffer and applies an
/// alternating mask filter, verifying the count matches expected.
fn test_edge_case_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let sizes = SizeGen::edge_cases();

    for size in sizes {
        // Skip size 0 as it's tested elsewhere and filter needs non-empty mask
        if size == 0 {
            continue;
        }

        // Create sequential data: 0, 1, 2, ..., size-1
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_edge_case_sizes",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Create alternating mask: 1, 0, 1, 0, ... (keeps even indices)
        let mask: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let expected_count = (size + 1) / 2; // Ceiling division for odd sizes

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_edge_case_sizes",
                    start.elapsed(),
                    format!("Filter failed for size {}: {}", size, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_edge_case_sizes",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered),
                    expected_count
                ),
            );
        }

        // Verify filtered values are the even indices (0, 2, 4, ...)
        let filtered_data = match ctx.provider.download_column::<u32>(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_edge_case_sizes",
                    start.elapsed(),
                    format!("Size {}: failed to download filtered column: {}", size, e),
                )
            }
        };

        for (idx, &val) in filtered_data.iter().enumerate() {
            let expected_val = (idx * 2) as u32;
            if val != expected_val {
                return TestResult::error(
                    "test_edge_case_sizes",
                    start.elapsed(),
                    format!(
                        "Size {}: filtered[{}] = {}, expected {}",
                        size, idx, val, expected_val
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_edge_case_sizes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_edge_case_sizes", start.elapsed())
}

/// Test 2: Filter with all 1s except last element.
///
/// Tests off-by-one handling in filter operations by creating masks where
/// all elements are selected except the last one. This catches bugs in
/// boundary handling.
fn test_off_by_one_filter(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes near warp/block boundaries
    let sizes: Vec<usize> = vec![31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257];

    for size in sizes {
        // Create sequential data: 0, 1, 2, ..., size-1
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_off_by_one_filter",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Create mask: all 1s except last element
        let mut mask: Vec<u8> = vec![1; size];
        mask[size - 1] = 0;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_off_by_one_filter",
                    start.elapsed(),
                    format!("Filter failed for size {}: {}", size, e),
                )
            }
        };

        // Should have size-1 rows
        if ctx.device_row_count(&filtered) != (size - 1) as u64 {
            return TestResult::error(
                "test_off_by_one_filter",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered),
                    size - 1
                ),
            );
        }

        // Download and verify the excluded element is NOT in the result
        let filtered_data = match ctx.provider.download_column::<u32>(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_off_by_one_filter",
                    start.elapsed(),
                    format!("Size {}: failed to download filtered column: {}", size, e),
                )
            }
        };

        // The excluded element value is size-1
        let excluded_val = (size - 1) as u32;
        if filtered_data.contains(&excluded_val) {
            return TestResult::error(
                "test_off_by_one_filter",
                start.elapsed(),
                format!(
                    "Size {}: excluded value {} found in filtered result",
                    size, excluded_val
                ),
            );
        }

        // Verify all included elements are present (0 to size-2)
        for i in 0..(size - 1) {
            if filtered_data[i] != i as u32 {
                return TestResult::error(
                    "test_off_by_one_filter",
                    start.elapsed(),
                    format!(
                        "Size {}: filtered[{}] = {}, expected {}",
                        size, i, filtered_data[i], i
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_off_by_one_filter",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_off_by_one_filter", start.elapsed())
}

/// Test 3: Sort preserves all elements including the last one.
///
/// Tests that sort operations don't lose elements due to off-by-one errors.
/// Creates a 2-column buffer with keys in reverse order and values that
/// allow verification of key-value pairing after sort.
fn test_off_by_one_sort(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Sizes near warp/block boundaries
    let sizes: Vec<usize> = vec![31, 32, 33, 63, 64, 65, 127, 128, 129];

    for size in sizes {
        // Create keys in reverse order: size-1, size-2, ..., 1, 0
        let keys: Vec<u32> = (0..size as u32).rev().collect();
        // Create values: i * 10 (so val[i] = i * 10 for original index i)
        let vals: Vec<u32> = (0..size as u32).map(|i| i * 10).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_off_by_one_sort",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Sort by key column (index 0)
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_off_by_one_sort",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify all elements present
        if ctx.device_row_count(&sorted) != size as u64 {
            return TestResult::error(
                "test_off_by_one_sort",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&sorted),
                    size
                ),
            );
        }

        // Download sorted columns
        let sorted_keys = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_off_by_one_sort",
                    start.elapsed(),
                    format!("Size {}: failed to download sorted keys: {}", size, e),
                )
            }
        };

        let sorted_vals = match ctx.provider.download_column::<u32>(&sorted, 1) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_off_by_one_sort",
                    start.elapsed(),
                    format!("Size {}: failed to download sorted vals: {}", size, e),
                )
            }
        };

        // Verify keys are sorted: 0, 1, 2, ..., size-1
        for (i, &key) in sorted_keys.iter().enumerate() {
            if key != i as u32 {
                return TestResult::error(
                    "test_off_by_one_sort",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted_keys[{}] = {}, expected {}",
                        size, i, key, i
                    ),
                );
            }
        }

        // Verify key-value pairing is preserved
        // Original: key[i] = size-1-i, val[i] = i*10
        // After sort by key: sorted_keys[j] = j, so original index was size-1-j
        // Therefore sorted_vals[j] = (size-1-j)*10
        for i in 0..size {
            let expected_val = ((size - 1 - i) * 10) as u32;
            if sorted_vals[i] != expected_val {
                return TestResult::error(
                    "test_off_by_one_sort",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted_vals[{}] = {}, expected {}",
                        size, i, sorted_vals[i], expected_val
                    ),
                );
            }
        }

        // Specifically verify: after sorting, key 0 should have val = (size-1)*10
        if sorted_vals[0] != ((size - 1) * 10) as u32 {
            return TestResult::error(
                "test_off_by_one_sort",
                start.elapsed(),
                format!(
                    "Size {}: key 0 has val {}, expected {}",
                    size,
                    sorted_vals[0],
                    (size - 1) * 10
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_off_by_one_sort",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_off_by_one_sort", start.elapsed())
}

/// Test 4: Large sizes requiring grid-stride loops.
///
/// Tests that kernels correctly handle sizes larger than the grid can
/// process in a single iteration, requiring grid-stride loops.
fn test_grid_stride_loop(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Large sizes that require grid-stride loops
    let sizes: Vec<usize> = vec![100_000, 500_000, 1_000_000];

    for size in sizes {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_grid_stride_loop",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Create alternating mask: 0, 1, 0, 1, ... (keeps odd indices)
        let mask: Vec<u8> = (0..size).map(|i| (i % 2) as u8).collect();
        let expected_count = size / 2;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_grid_stride_loop",
                    start.elapsed(),
                    format!("Filter failed for size {}: {}", size, e),
                )
            }
        };

        // Verify count = size / 2
        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_grid_stride_loop",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered),
                    expected_count
                ),
            );
        }

        // Sample check: verify some values
        let filtered_data = match ctx.provider.download_column::<u32>(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_grid_stride_loop",
                    start.elapsed(),
                    format!("Size {}: failed to download filtered column: {}", size, e),
                )
            }
        };

        // First filtered element should be 1 (first odd index)
        if filtered_data[0] != 1 {
            return TestResult::error(
                "test_grid_stride_loop",
                start.elapsed(),
                format!(
                    "Size {}: first filtered element is {}, expected 1",
                    size, filtered_data[0]
                ),
            );
        }

        // Last filtered element should be size-1 (if size is even) or size-2 (if odd)
        let last_expected = if size % 2 == 0 { size - 1 } else { size - 2 };
        let last_idx = filtered_data.len() - 1;
        if filtered_data[last_idx] != last_expected as u32 {
            return TestResult::error(
                "test_grid_stride_loop",
                start.elapsed(),
                format!(
                    "Size {}: last filtered element is {}, expected {}",
                    size, filtered_data[last_idx], last_expected
                ),
            );
        }

        // Sample check at various positions
        let sample_indices = [0, 100, 1000, expected_count / 2, expected_count - 1];
        for &idx in &sample_indices {
            if idx < filtered_data.len() {
                let expected_val = (idx * 2 + 1) as u32; // Odd values: 1, 3, 5, ...
                if filtered_data[idx] != expected_val {
                    return TestResult::error(
                        "test_grid_stride_loop",
                        start.elapsed(),
                        format!(
                            "Size {}: filtered[{}] = {}, expected {}",
                            size, idx, filtered_data[idx], expected_val
                        ),
                    );
                }
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_grid_stride_loop",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_grid_stride_loop", start.elapsed())
}

/// Test 5: Sizes with partial warps/blocks (filter).
///
/// Tests filter operations on sizes that are one past a power of 2,
/// which require handling partial warps/blocks correctly.
fn test_tail_handling_filter(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes one past power of 2
    let sizes: Vec<usize> = vec![33, 65, 129, 257, 513, 1025];

    for size in sizes {
        // Create sequential data: 0, 1, 2, ..., size-1
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_tail_handling_filter",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Apply all-1s mask (keep all elements)
        let mask: Vec<u8> = vec![1; size];

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_tail_handling_filter",
                    start.elapsed(),
                    format!("Filter failed for size {}: {}", size, e),
                )
            }
        };

        // All elements should be present
        if ctx.device_row_count(&filtered) != size as u64 {
            return TestResult::error(
                "test_tail_handling_filter",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered),
                    size
                ),
            );
        }

        // Download and verify all elements
        let filtered_data = match ctx.provider.download_column::<u32>(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_tail_handling_filter",
                    start.elapsed(),
                    format!("Size {}: failed to download filtered column: {}", size, e),
                )
            }
        };

        for (i, &val) in filtered_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "test_tail_handling_filter",
                    start.elapsed(),
                    format!("Size {}: filtered[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_tail_handling_filter",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_tail_handling_filter", start.elapsed())
}

/// Test 6: Sizes with partial warps/blocks (sort).
///
/// Tests sort operations on sizes that are one past a power of 2,
/// which require handling partial warps/blocks correctly.
fn test_tail_handling_sort(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes one past power of 2
    let sizes: Vec<usize> = vec![33, 65, 129, 257, 513, 1025];

    for size in sizes {
        // Create reverse-sorted data: size-1, size-2, ..., 1, 0
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_tail_handling_sort",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Sort the buffer
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_tail_handling_sort",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify row count
        if ctx.device_row_count(&sorted) != size as u64 {
            return TestResult::error(
                "test_tail_handling_sort",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&sorted),
                    size
                ),
            );
        }

        // Download and verify sorted order: 0, 1, 2, ..., size-1
        let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_tail_handling_sort",
                    start.elapsed(),
                    format!("Size {}: failed to download sorted column: {}", size, e),
                )
            }
        };

        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "test_tail_handling_sort",
                    start.elapsed(),
                    format!("Size {}: sorted[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_tail_handling_sort",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_tail_handling_sort", start.elapsed())
}

/// Test 7: First and last index correctness.
///
/// Tests that filter operations correctly handle selection of only the
/// first element or only the last element in a buffer.
fn test_boundary_indices(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let size: usize = 1000;

    // Create sequential data: 0, 1, 2, ..., 999
    let data: Vec<u32> = (0..size as u32).collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_boundary_indices",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Test 1: Select only first element
    let mut mask_first: Vec<u8> = vec![0; size];
    mask_first[0] = 1;

    let filtered_first = match ctx.provider.filter_by_mask(&buffer, &mask_first) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_boundary_indices",
                start.elapsed(),
                format!("Filter for first element failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&filtered_first) != 1 {
        return TestResult::error(
            "test_boundary_indices",
            start.elapsed(),
            format!(
                "Filter first: returned {} rows, expected 1",
                ctx.device_row_count(&filtered_first)
            ),
        );
    }

    let first_data = match ctx.provider.download_column::<u32>(&filtered_first, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_boundary_indices",
                start.elapsed(),
                format!("Failed to download first filtered column: {}", e),
            )
        }
    };

    if first_data.len() != 1 || first_data[0] != 0 {
        return TestResult::error(
            "test_boundary_indices",
            start.elapsed(),
            format!("Filter first: result is {:?}, expected [0]", first_data),
        );
    }

    // Test 2: Select only last element
    let mut mask_last: Vec<u8> = vec![0; size];
    mask_last[size - 1] = 1;

    let filtered_last = match ctx.provider.filter_by_mask(&buffer, &mask_last) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_boundary_indices",
                start.elapsed(),
                format!("Filter for last element failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&filtered_last) != 1 {
        return TestResult::error(
            "test_boundary_indices",
            start.elapsed(),
            format!(
                "Filter last: returned {} rows, expected 1",
                ctx.device_row_count(&filtered_last)
            ),
        );
    }

    let last_data = match ctx.provider.download_column::<u32>(&filtered_last, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_boundary_indices",
                start.elapsed(),
                format!("Failed to download last filtered column: {}", e),
            )
        }
    };

    if last_data.len() != 1 || last_data[0] != 999 {
        return TestResult::error(
            "test_boundary_indices",
            start.elapsed(),
            format!("Filter last: result is {:?}, expected [999]", last_data),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_boundary_indices",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_boundary_indices", start.elapsed())
}

/// Test 8: Multiple columns with different strides.
///
/// Tests that filter operations correctly handle multi-column buffers
/// where each column may have different memory layouts.
fn test_multi_column_strides(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Test with 2, 3, and 4 columns
    for num_cols in [2, 3, 4] {
        let schema = Schema::new(
            (0..num_cols)
                .map(|i| (format!("col{}", i), ScalarType::U32))
                .collect(),
        );

        // Test with different sizes
        for size in [100, 256, 1000] {
            // Create columns: col[c][i] = c * 1000 + i
            let columns: Vec<Vec<u32>> = (0..num_cols)
                .map(|c| (0..size).map(|i| (c * 1000 + i) as u32).collect())
                .collect();

            let column_refs: Vec<&[u32]> = columns.iter().map(|c| c.as_slice()).collect();

            let buffer = match ctx
                .provider
                .create_buffer_from_u32_columns(&column_refs, schema.clone())
            {
                Ok(buf) => buf,
                Err(e) => {
                    return TestResult::error(
                        "test_multi_column_strides",
                        start.elapsed(),
                        format!(
                            "Failed to create buffer with {} cols, size {}: {}",
                            num_cols, size, e
                        ),
                    )
                }
            };

            // Apply alternating mask (filter every other row)
            let mask: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
            let expected_count = (size + 1) / 2;

            let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
                Ok(f) => f,
                Err(e) => {
                    return TestResult::error(
                        "test_multi_column_strides",
                        start.elapsed(),
                        format!("Filter failed for {} cols, size {}: {}", num_cols, size, e),
                    )
                }
            };

            if ctx.device_row_count(&filtered) != expected_count as u64 {
                return TestResult::error(
                    "test_multi_column_strides",
                    start.elapsed(),
                    format!(
                        "{} cols, size {}: filter returned {} rows, expected {}",
                        num_cols,
                        size,
                        ctx.device_row_count(&filtered),
                        expected_count
                    ),
                );
            }

            // Verify each column in output has correct values
            for col_idx in 0..num_cols {
                let col_data = match ctx.provider.download_column::<u32>(&filtered, col_idx) {
                    Ok(d) => d,
                    Err(e) => {
                        return TestResult::error(
                            "test_multi_column_strides",
                            start.elapsed(),
                            format!(
                                "{} cols, size {}: failed to download column {}: {}",
                                num_cols, size, col_idx, e
                            ),
                        )
                    }
                };

                // Verify values: col[c][filtered_idx] = c * 1000 + original_idx
                // where original_idx is the even indices: 0, 2, 4, ...
                for (filtered_idx, &val) in col_data.iter().enumerate() {
                    let original_idx = filtered_idx * 2;
                    let expected = (col_idx * 1000 + original_idx) as u32;
                    if val != expected {
                        return TestResult::error(
                            "test_multi_column_strides",
                            start.elapsed(),
                            format!(
                                "{} cols, size {}: col{}[{}] = {}, expected {}",
                                num_cols, size, col_idx, filtered_idx, val, expected
                            ),
                        );
                    }
                }
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_multi_column_strides",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_multi_column_strides", start.elapsed())
}
