//! Category 5: Global Memory Hazards
//!
//! Tests global memory access patterns and potential hazards including
//! large allocations, aligned access patterns, coalesced access, and
//! repeated buffer access.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{Schema, ScalarType};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c05_global_memory");
    let start = Instant::now();

    results.add_result(test_large_allocation(ctx));
    results.add_result(test_aligned_access_patterns(ctx));
    results.add_result(test_coalesced_access(ctx));
    results.add_result(test_repeated_access(ctx));
    results.add_result(test_buffer_reuse(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Allocate and verify large buffer (10M elements).
///
/// Tests that large memory allocations work correctly and data integrity
/// is maintained across the entire buffer.
fn test_large_allocation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10_000_000; // 10M elements = 40MB

    // Create large sequential data
    let data: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_large_allocation",
                start.elapsed(),
                format!("Failed to create 10M element buffer: {}", e),
            )
        }
    };

    // Verify row count
    if buffer.num_rows != SIZE as u64 {
        return TestResult::error(
            "test_large_allocation",
            start.elapsed(),
            format!(
                "Buffer has {} rows, expected {}",
                buffer.num_rows, SIZE
            ),
        );
    }

    // Apply a filter to exercise the buffer (keep elements divisible by 1000)
    let mask: Vec<u8> = (0..SIZE).map(|i| if i % 1000 == 0 { 1 } else { 0 }).collect();
    let expected_count = SIZE / 1000;

    let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_large_allocation",
                start.elapsed(),
                format!("Filter of large buffer failed: {}", e),
            )
        }
    };

    if filtered.num_rows != expected_count as u64 {
        return TestResult::error(
            "test_large_allocation",
            start.elapsed(),
            format!(
                "Filter returned {} rows, expected {}",
                filtered.num_rows, expected_count
            ),
        );
    }

    // Verify some filtered values
    let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_large_allocation",
                start.elapsed(),
                format!("Failed to download filtered column: {}", e),
            )
        }
    };

    // Check first, middle, and last filtered values
    let test_indices = [0, expected_count / 2, expected_count - 1];
    for &idx in &test_indices {
        let expected = (idx * 1000) as u32;
        if filtered_data[idx] != expected {
            return TestResult::error(
                "test_large_allocation",
                start.elapsed(),
                format!(
                    "filtered[{}] = {}, expected {}",
                    idx, filtered_data[idx], expected
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_large_allocation",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_large_allocation", start.elapsed())
}

/// Test 2: Test aligned memory access with different sizes.
///
/// Tests that memory access works correctly with various buffer sizes
/// that exercise different alignment scenarios (aligned to 4, 8, 16,
/// 32 bytes, etc.).
fn test_aligned_access_patterns(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes that test various alignment scenarios
    // 32 elements = 128 bytes (aligned to 128)
    // 64 elements = 256 bytes (aligned to 256)
    // 128 elements = 512 bytes (aligned to 512)
    // 256 elements = 1024 bytes (aligned to 1KB)
    let aligned_sizes: Vec<usize> = vec![32, 64, 128, 256, 512, 1024, 2048, 4096];

    for size in aligned_sizes {
        // Create data where value = index
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_aligned_access_patterns",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Sort the buffer (this reads and writes all memory)
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_aligned_access_patterns",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Download and verify
        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_aligned_access_patterns",
                    start.elapsed(),
                    format!("Failed to download for size {}: {}", size, e),
                )
            }
        };

        // Data was already sorted, so it should be unchanged
        if sorted_data != data {
            return TestResult::error(
                "test_aligned_access_patterns",
                start.elapsed(),
                format!(
                    "Size {}: data corrupted after sort",
                    size
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_aligned_access_patterns",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_aligned_access_patterns", start.elapsed())
}

/// Test 3: Test that coalesced access patterns work correctly.
///
/// Tests memory access patterns that should result in coalesced reads
/// and writes on the GPU. Uses contiguous, sequential data patterns.
fn test_coalesced_access(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Create data that will exercise coalesced access during sort
    // Size is multiple of warp size (32)
    const SIZE: usize = 32768; // 32 warps * 32 threads * 32

    // Keys are shuffled (reverse order) to force data movement
    let keys: Vec<u32> = (0..SIZE as u32).rev().collect();
    // Values match original key positions
    let vals: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_columns(&[&keys, &vals], schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_coalesced_access",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort by key
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_coalesced_access",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download and verify
    let sorted_keys = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_coalesced_access",
                start.elapsed(),
                format!("Failed to download sorted keys: {}", e),
            )
        }
    };

    let sorted_vals = match ctx.provider.download_column_u32(&sorted, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_coalesced_access",
                start.elapsed(),
                format!("Failed to download sorted vals: {}", e),
            )
        }
    };

    // Verify keys are sorted 0, 1, 2, ..., SIZE-1
    for (i, &key) in sorted_keys.iter().enumerate() {
        if key != i as u32 {
            return TestResult::error(
                "test_coalesced_access",
                start.elapsed(),
                format!("sorted_keys[{}] = {}, expected {}", i, key, i),
            );
        }
    }

    // Verify key-value pairing is preserved
    // Original: key[i] = SIZE-1-i, val[i] = i
    // After sort: sorted_keys[j] = j, so original index was SIZE-1-j
    // Therefore sorted_vals[j] = SIZE-1-j
    for i in 0..SIZE {
        let expected_val = (SIZE - 1 - i) as u32;
        if sorted_vals[i] != expected_val {
            return TestResult::error(
                "test_coalesced_access",
                start.elapsed(),
                format!(
                    "sorted_vals[{}] = {}, expected {}",
                    i, sorted_vals[i], expected_val
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_coalesced_access",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_coalesced_access", start.elapsed())
}

/// Test 4: Access same buffer multiple times with different operations.
///
/// Tests that a buffer can be read multiple times and that different
/// operations on the same buffer produce correct results.
fn test_repeated_access(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;

    // Create buffer with random-ish but deterministic data
    let data: Vec<u32> = (0..SIZE).map(|i| ((i * 7 + 13) % 1000) as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Operation 1: Sort the buffer
    let sorted1 = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("First sort failed: {}", e),
            )
        }
    };

    // Operation 2: Sort again (should give same result)
    let sorted2 = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("Second sort failed: {}", e),
            )
        }
    };

    // Operation 3: Filter (keep values < 500)
    let mask: Vec<u8> = data.iter().map(|&v| if v < 500 { 1 } else { 0 }).collect();
    let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    // Verify sorted1 and sorted2 are identical
    let sorted_data1 = match ctx.provider.download_column_u32(&sorted1, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("Failed to download sorted1: {}", e),
            )
        }
    };

    let sorted_data2 = match ctx.provider.download_column_u32(&sorted2, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("Failed to download sorted2: {}", e),
            )
        }
    };

    if sorted_data1 != sorted_data2 {
        return TestResult::error(
            "test_repeated_access",
            start.elapsed(),
            "Two sorts of same buffer produced different results".to_string(),
        );
    }

    // Verify sorted data is correct
    let mut expected_sorted = data.clone();
    expected_sorted.sort();
    if sorted_data1 != expected_sorted {
        return TestResult::error(
            "test_repeated_access",
            start.elapsed(),
            "Sorted data is incorrect".to_string(),
        );
    }

    // Verify filtered data
    let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("Failed to download filtered: {}", e),
            )
        }
    };

    let expected_filtered: Vec<u32> = data.iter().copied().filter(|&v| v < 500).collect();
    if filtered_data != expected_filtered {
        return TestResult::error(
            "test_repeated_access",
            start.elapsed(),
            format!(
                "Filtered data incorrect: got {} elements, expected {}",
                filtered_data.len(),
                expected_filtered.len()
            ),
        );
    }

    // Operation 4: One more sort after filter to ensure buffer is still valid
    let sorted3 = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("Third sort failed: {}", e),
            )
        }
    };

    let sorted_data3 = match ctx.provider.download_column_u32(&sorted3, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_repeated_access",
                start.elapsed(),
                format!("Failed to download sorted3: {}", e),
            )
        }
    };

    if sorted_data3 != expected_sorted {
        return TestResult::error(
            "test_repeated_access",
            start.elapsed(),
            "Buffer corrupted after multiple operations".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_repeated_access",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_repeated_access", start.elapsed())
}

/// Test 5: Create buffer, use it, verify data integrity on reuse.
///
/// Tests that buffer contents remain valid across multiple read operations
/// and that the GPU memory manager correctly handles buffer lifecycle.
fn test_buffer_reuse(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 5000;

    // Create initial buffer
    let data: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_buffer_reuse",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Read 1: Download and verify
    let read1 = match ctx.provider.download_column_u32(&buffer, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_buffer_reuse",
                start.elapsed(),
                format!("First read failed: {}", e),
            )
        }
    };

    if read1 != data {
        return TestResult::error(
            "test_buffer_reuse",
            start.elapsed(),
            "First read returned incorrect data".to_string(),
        );
    }

    // Use buffer in an operation
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_buffer_reuse",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Read 2: Verify original buffer unchanged after sort
    let read2 = match ctx.provider.download_column_u32(&buffer, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_buffer_reuse",
                start.elapsed(),
                format!("Second read failed: {}", e),
            )
        }
    };

    if read2 != data {
        return TestResult::error(
            "test_buffer_reuse",
            start.elapsed(),
            "Original buffer modified by sort operation".to_string(),
        );
    }

    // Use in filter operation
    let mask: Vec<u8> = (0..SIZE).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_buffer_reuse",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    // Read 3: Verify original buffer still unchanged
    let read3 = match ctx.provider.download_column_u32(&buffer, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_buffer_reuse",
                start.elapsed(),
                format!("Third read failed: {}", e),
            )
        }
    };

    if read3 != data {
        return TestResult::error(
            "test_buffer_reuse",
            start.elapsed(),
            "Original buffer modified by filter operation".to_string(),
        );
    }

    // Verify sorted result is correct
    let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_buffer_reuse",
                start.elapsed(),
                format!("Failed to download sorted: {}", e),
            )
        }
    };

    // Data was already sorted, so should be identical
    if sorted_data != data {
        return TestResult::error(
            "test_buffer_reuse",
            start.elapsed(),
            "Sorted result incorrect".to_string(),
        );
    }

    // Verify filtered result is correct
    let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_buffer_reuse",
                start.elapsed(),
                format!("Failed to download filtered: {}", e),
            )
        }
    };

    let expected_filtered: Vec<u32> = data.iter().enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, &v)| v)
        .collect();

    if filtered_data != expected_filtered {
        return TestResult::error(
            "test_buffer_reuse",
            start.elapsed(),
            "Filtered result incorrect".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_buffer_reuse",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_buffer_reuse", start.elapsed())
}
