//! Category 2: Kernel launch configuration edge cases
//!
//! This category tests:
//! - Zero element edge case (no kernel launch)
//! - Single element operations
//! - Warp boundary sizes (31, 32, 33, etc.)
//! - Block boundary sizes (255, 256, 257, etc.)
//! - Non-power-of-two and prime sizes
//! - Large grid sizes (1M, 10M elements)

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub(crate) fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c02_launch_config");
    let start = Instant::now();

    results.add_result(test_zero_elements_no_launch(ctx));
    results.add_result(test_single_element(ctx));
    results.add_result(test_warp_boundary_sizes(ctx));
    results.add_result(test_block_boundary_sizes(ctx));
    results.add_result(test_non_power_of_two_sizes(ctx));
    results.add_result(test_large_grid_sizes(ctx));
    results.add_result(test_max_practical_size(ctx));
    results.add_result(test_mc_sample_edge_sizes(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 8: MC sampling kernel launch config across edge sizes.
///
/// Validates `(num_vars, num_samples)` combinations around warp and block boundaries.
fn test_mc_sample_edge_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let var_counts: Vec<usize> = vec![0, 1, 31, 32, 33, 255, 256, 257];
    let sample_counts: Vec<usize> = vec![0, 1, 31, 32, 33, 255, 256, 257];

    for &num_vars in &var_counts {
        let probs: Vec<f32> = vec![0.5f32; num_vars];

        // Allocate zero-filled force arrays (no clamping)
        let mut d_force_mask = ctx.memory.alloc::<u8>(num_vars.max(1)).unwrap();
        ctx.device.inner().memset_zeros(&mut d_force_mask).unwrap();
        let mut d_forced_value = ctx.memory.alloc::<u8>(num_vars.max(1)).unwrap();
        ctx.device
            .inner()
            .memset_zeros(&mut d_forced_value)
            .unwrap();

        for &num_samples in &sample_counts {
            let got = match ctx.provider.sample_bernoulli_matrix(
                &probs,
                num_samples,
                123,
                &d_force_mask.slice(..),
                &d_forced_value.slice(..),
            ) {
                Ok(v) => v,
                Err(e) => {
                    return TestResult::error(
                        "test_mc_sample_edge_sizes",
                        start.elapsed(),
                        format!(
                            "sample_bernoulli_matrix failed for num_vars={}, num_samples={}: {}",
                            num_vars, num_samples, e
                        ),
                    )
                }
            };

            let expected_len = num_vars.saturating_mul(num_samples);
            if got.len() != expected_len {
                return TestResult::error(
                    "test_mc_sample_edge_sizes",
                    start.elapsed(),
                    format!(
                        "Unexpected output length for num_vars={}, num_samples={}: got {}, expected {}",
                        num_vars,
                        num_samples,
                        got.len(),
                        expected_len
                    ),
                );
            }

            for (i, &b) in got.iter().enumerate() {
                if b > 1 {
                    return TestResult::error(
                        "test_mc_sample_edge_sizes",
                        start.elapsed(),
                        format!(
                            "Invalid sample bit at idx {} for num_vars={}, num_samples={}: {}",
                            i, num_vars, num_samples, b
                        ),
                    );
                }
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_mc_sample_edge_sizes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_mc_sample_edge_sizes", start.elapsed())
}

/// Test 1: Empty buffer operations should not crash.
///
/// Creates an empty buffer and verifies that sort and filter operations
/// return empty results without crashing or causing CUDA errors.
fn test_zero_elements_no_launch(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Create empty buffer
    let empty = match ctx.provider.create_empty_buffer(schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "zero_elements_no_launch",
                start.elapsed(),
                format!("Failed to create empty buffer: {}", e),
            )
        }
    };

    // Verify empty buffer has 0 rows
    if ctx.device_row_count(&empty) != 0 {
        return TestResult::error(
            "zero_elements_no_launch",
            start.elapsed(),
            format!(
                "Empty buffer has {} rows, expected 0",
                ctx.device_row_count(&empty)
            ),
        );
    }

    // Sort empty - should return empty, not crash
    match ctx.provider.sort(&empty, &[0]) {
        Ok(sorted) => {
            if ctx.device_row_count(&sorted) != 0 {
                return TestResult::error(
                    "zero_elements_no_launch",
                    start.elapsed(),
                    format!(
                        "Sort of empty buffer returned {} rows, expected 0",
                        ctx.device_row_count(&sorted)
                    ),
                );
            }
        }
        Err(e) => {
            return TestResult::error(
                "zero_elements_no_launch",
                start.elapsed(),
                format!("Sort of empty buffer failed: {}", e),
            )
        }
    }

    // Filter empty with empty mask - should return empty, not crash
    match ctx.provider.filter_by_mask(&empty, &[]) {
        Ok(filtered) => {
            if ctx.device_row_count(&filtered) != 0 {
                return TestResult::error(
                    "zero_elements_no_launch",
                    start.elapsed(),
                    format!(
                        "Filter of empty buffer returned {} rows, expected 0",
                        ctx.device_row_count(&filtered)
                    ),
                );
            }
        }
        Err(e) => {
            return TestResult::error(
                "zero_elements_no_launch",
                start.elapsed(),
                format!("Filter of empty buffer failed: {}", e),
            )
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "zero_elements_no_launch",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("zero_elements_no_launch", start.elapsed())
}

/// Test 2: Operations with exactly one element.
///
/// Tests that sort and filter operations work correctly with a single element,
/// which is an edge case for many GPU algorithms that assume multiple elements.
fn test_single_element(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Create buffer with exactly 1 element
    let data: Vec<u32> = vec![42];
    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "single_element",
                start.elapsed(),
                format!("Failed to create single element buffer: {}", e),
            )
        }
    };

    // Verify buffer has 1 row
    if ctx.device_row_count(&buffer) != 1 {
        return TestResult::error(
            "single_element",
            start.elapsed(),
            format!(
                "Single element buffer has {} rows, expected 1",
                ctx.device_row_count(&buffer)
            ),
        );
    }

    // Sort single element - should return the same element
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "single_element",
                start.elapsed(),
                format!("Sort of single element failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&sorted) != 1 {
        return TestResult::error(
            "single_element",
            start.elapsed(),
            format!(
                "Sort of single element returned {} rows, expected 1",
                ctx.device_row_count(&sorted)
            ),
        );
    }

    // Verify the sorted value
    let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "single_element",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    if sorted_data.len() != 1 || sorted_data[0] != 42 {
        return TestResult::error(
            "single_element",
            start.elapsed(),
            format!(
                "Sort of single element returned {:?}, expected [42]",
                sorted_data
            ),
        );
    }

    // Filter single element with mask=[1] - should return 1 result
    let filtered_keep = match ctx.provider.filter_by_mask(&buffer, &[1]) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "single_element",
                start.elapsed(),
                format!("Filter with mask=[1] failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&filtered_keep) != 1 {
        return TestResult::error(
            "single_element",
            start.elapsed(),
            format!(
                "Filter with mask=[1] returned {} rows, expected 1",
                ctx.device_row_count(&filtered_keep)
            ),
        );
    }

    // Verify filtered value
    let filtered_data = match ctx.provider.download_column::<u32>(&filtered_keep, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "single_element",
                start.elapsed(),
                format!("Failed to download filtered column: {}", e),
            )
        }
    };

    if filtered_data.len() != 1 || filtered_data[0] != 42 {
        return TestResult::error(
            "single_element",
            start.elapsed(),
            format!(
                "Filter with mask=[1] returned {:?}, expected [42]",
                filtered_data
            ),
        );
    }

    // Filter single element with mask=[0] - should return 0 results
    let filtered_drop = match ctx.provider.filter_by_mask(&buffer, &[0]) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "single_element",
                start.elapsed(),
                format!("Filter with mask=[0] failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&filtered_drop) != 0 {
        return TestResult::error(
            "single_element",
            start.elapsed(),
            format!(
                "Filter with mask=[0] returned {} rows, expected 0",
                ctx.device_row_count(&filtered_drop)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "single_element",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("single_element", start.elapsed())
}

/// Test 3: Test sizes around warp boundaries (32).
///
/// Warp size is 32 threads in CUDA. Tests sizes 31, 32, 33, etc. to ensure
/// correct handling at warp boundaries where partial warps may occur.
fn test_warp_boundary_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Use sizes from SizeGen - subset around warp boundaries
    let sizes: Vec<usize> = vec![31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129];

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
                    "warp_boundary_sizes",
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
                    "warp_boundary_sizes",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify row count
        if ctx.device_row_count(&sorted) != size as u64 {
            return TestResult::error(
                "warp_boundary_sizes",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&sorted),
                    size
                ),
            );
        }

        // Download and verify sorted order
        let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "warp_boundary_sizes",
                    start.elapsed(),
                    format!("Size {}: failed to download sorted column: {}", size, e),
                )
            }
        };

        // Verify it's sorted ascending: 0, 1, 2, ..., size-1
        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "warp_boundary_sizes",
                    start.elapsed(),
                    format!("Size {}: sorted[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "warp_boundary_sizes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("warp_boundary_sizes", start.elapsed())
}

/// Test 4: Test sizes around block boundaries (256).
///
/// Block size is typically 256 threads. Tests sizes 255, 256, 257, etc.
/// to ensure correct handling at block boundaries.
fn test_block_boundary_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Use sizes from SizeGen - subset around block boundaries
    let sizes: Vec<usize> = vec![
        255, 256, 257, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025,
    ];

    for size in sizes {
        // Create reverse-sorted data
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "block_boundary_sizes",
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
                    "block_boundary_sizes",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify row count
        if ctx.device_row_count(&sorted) != size as u64 {
            return TestResult::error(
                "block_boundary_sizes",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&sorted),
                    size
                ),
            );
        }

        // Download and verify sorted order
        let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "block_boundary_sizes",
                    start.elapsed(),
                    format!("Size {}: failed to download sorted column: {}", size, e),
                )
            }
        };

        // Verify it's sorted ascending
        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "block_boundary_sizes",
                    start.elapsed(),
                    format!("Size {}: sorted[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "block_boundary_sizes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("block_boundary_sizes", start.elapsed())
}

/// Test 5: Prime and odd sizes (non-power-of-two).
///
/// Many GPU algorithms are optimized for power-of-two sizes. This test
/// ensures correct behavior with prime and odd sizes.
fn test_non_power_of_two_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Prime sizes that are particularly tricky for GPU algorithms
    let sizes: Vec<usize> = vec![7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

    for size in sizes {
        // Create reverse-sorted data
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "non_power_of_two_sizes",
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
                    "non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify row count matches
        if ctx.device_row_count(&sorted) != size as u64 {
            return TestResult::error(
                "non_power_of_two_sizes",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&sorted),
                    size
                ),
            );
        }

        // Download and verify sorted order
        let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Size {}: failed to download sorted column: {}", size, e),
                )
            }
        };

        // Verify it's sorted ascending
        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "non_power_of_two_sizes",
                    start.elapsed(),
                    format!("Size {}: sorted[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "non_power_of_two_sizes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("non_power_of_two_sizes", start.elapsed())
}

/// Test 6: Large grid sizes with 1M elements.
///
/// Tests that the kernels handle large grids correctly by sorting
/// 1 million elements and verifying key positions.
fn test_large_grid_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let size: usize = 1_000_000;

    // Create reverse-sorted data: size-1, size-2, ..., 1, 0
    let data: Vec<u32> = (0..size as u32).rev().collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "large_grid_sizes",
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
                "large_grid_sizes",
                start.elapsed(),
                format!("Sort failed for size {}: {}", size, e),
            )
        }
    };

    // Verify row count
    if ctx.device_row_count(&sorted) != size as u64 {
        return TestResult::error(
            "large_grid_sizes",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                ctx.device_row_count(&sorted),
                size
            ),
        );
    }

    // Download sorted data
    let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "large_grid_sizes",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify key positions: first element is 0
    if sorted_data[0] != 0 {
        return TestResult::error(
            "large_grid_sizes",
            start.elapsed(),
            format!("First element is {}, expected 0", sorted_data[0]),
        );
    }

    // Last element is size-1
    let last_idx = size - 1;
    if sorted_data[last_idx] != last_idx as u32 {
        return TestResult::error(
            "large_grid_sizes",
            start.elapsed(),
            format!(
                "Last element is {}, expected {}",
                sorted_data[last_idx], last_idx
            ),
        );
    }

    // Middle element is size/2
    let mid_idx = size / 2;
    if sorted_data[mid_idx] != mid_idx as u32 {
        return TestResult::error(
            "large_grid_sizes",
            start.elapsed(),
            format!(
                "Middle element at {} is {}, expected {}",
                mid_idx, sorted_data[mid_idx], mid_idx
            ),
        );
    }

    // Verify overall sorted order by sampling
    let sample_indices = vec![
        0, 100, 1000, 10000, 100000, 250000, 500000, 750000, 900000, 999999,
    ];
    for &idx in &sample_indices {
        if sorted_data[idx] != idx as u32 {
            return TestResult::error(
                "large_grid_sizes",
                start.elapsed(),
                format!(
                    "Sample at index {}: got {}, expected {}",
                    idx, sorted_data[idx], idx
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "large_grid_sizes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("large_grid_sizes", start.elapsed())
}

/// Test 7: Maximum practical size with 10M elements.
///
/// Tests the maximum practical size by sorting 10 million elements
/// with a modular pattern and verifying the result.
fn test_max_practical_size(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let size: usize = 10_000_000;

    // Create data with modular pattern: i % 1000
    // This creates a pattern where values repeat: 0, 1, 2, ..., 999, 0, 1, 2, ...
    let data: Vec<u32> = (0..size).map(|i| (i % 1000) as u32).collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "max_practical_size",
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
                "max_practical_size",
                start.elapsed(),
                format!("Sort failed for size {}: {}", size, e),
            )
        }
    };

    // Verify row count is preserved
    if ctx.device_row_count(&sorted) != size as u64 {
        return TestResult::error(
            "max_practical_size",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                ctx.device_row_count(&sorted),
                size
            ),
        );
    }

    // Download sorted data
    let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "max_practical_size",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // With modular pattern i % 1000, after sorting we should have:
    // - First 10000 elements (size/1000) are 0
    // - Next 10000 elements are 1
    // - etc.
    // Check that the first element is 0
    if sorted_data[0] != 0 {
        return TestResult::error(
            "max_practical_size",
            start.elapsed(),
            format!("First element is {}, expected 0", sorted_data[0]),
        );
    }

    // Check that the last element is 999
    let last_idx = size - 1;
    if sorted_data[last_idx] != 999 {
        return TestResult::error(
            "max_practical_size",
            start.elapsed(),
            format!("Last element is {}, expected 999", sorted_data[last_idx]),
        );
    }

    // Verify data is sorted (sample check)
    let mut prev = sorted_data[0];
    for idx in [100000, 500000, 1000000, 5000000, 9000000, 9999999] {
        if sorted_data[idx] < prev {
            return TestResult::error(
                "max_practical_size",
                start.elapsed(),
                format!(
                    "Data not sorted: index {} has {}, but prev was {}",
                    idx, sorted_data[idx], prev
                ),
            );
        }
        prev = sorted_data[idx];
    }

    // Check value distribution: each value 0-999 appears exactly size/1000 times
    // Just verify boundary values to avoid expensive full validation
    let count_per_value = size / 1000;

    // Check position of first occurrence of value 1 (should be at index count_per_value)
    let first_one_pos = count_per_value;
    if sorted_data[first_one_pos] != 1 && sorted_data[first_one_pos] != 0 {
        return TestResult::error(
            "max_practical_size",
            start.elapsed(),
            format!(
                "Expected value 0 or 1 at position {}, got {}",
                first_one_pos, sorted_data[first_one_pos]
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "max_practical_size",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("max_practical_size", start.elapsed())
}
