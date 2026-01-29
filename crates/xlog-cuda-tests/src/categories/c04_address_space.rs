//! Category 4: Address Space Correctness
//!
//! Tests that different address spaces (global memory) work correctly with
//! various data types. Verifies that values are preserved through GPU operations.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c04_address_space");
    let start = Instant::now();

    results.add_result(test_global_u32_correctness(ctx));
    results.add_result(test_global_u64_correctness(ctx));
    results.add_result(test_global_i64_correctness(ctx));
    results.add_result(test_global_f64_correctness(ctx));
    results.add_result(test_multi_buffer_isolation(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Create U32 buffer, sort, verify values preserved.
///
/// Tests that U32 values are correctly stored in global memory and preserved
/// through sort operations. Uses a variety of U32 values including edge cases.
fn test_global_u32_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Create test data with various U32 values including edge cases
    let data: Vec<u32> = vec![
        0,
        1,
        u32::MAX,
        u32::MAX - 1,
        0x8000_0000, // Sign bit position
        0x7FFF_FFFF,
        0xFFFF_0000,
        0x0000_FFFF,
        0xDEAD_BEEF,
        0xCAFE_BABE,
        42,
        100,
        1000,
        10000,
        100000,
        1000000,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_global_u32_correctness",
                start.elapsed(),
                format!("Failed to create U32 buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_global_u32_correctness",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Verify row count preserved
    if ctx.device_row_count(&sorted) != data.len() as u64 {
        return TestResult::error(
            "test_global_u32_correctness",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                ctx.device_row_count(&sorted),
                data.len()
            ),
        );
    }

    // Download and verify all original values are present (sorted)
    let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_global_u32_correctness",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify sorted order
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_global_u32_correctness",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} > {}",
                    i,
                    sorted_data[i - 1],
                    sorted_data[i]
                ),
            );
        }
    }

    // Verify all original values are present
    let mut original_sorted = data.clone();
    original_sorted.sort();
    if sorted_data != original_sorted {
        return TestResult::error(
            "test_global_u32_correctness",
            start.elapsed(),
            format!(
                "Values not preserved: expected {:?}, got {:?}",
                original_sorted, sorted_data
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_global_u32_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_global_u32_correctness", start.elapsed())
}

/// Test 2: Create U64 buffer, sort, verify U64::MAX values work.
///
/// Tests that U64 values including U64::MAX are correctly stored and processed
/// in global memory. This is critical for 64-bit addressing correctness.
fn test_global_u64_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U64)]);

    // Create test data with various U64 values including edge cases
    let data: Vec<u64> = vec![
        0,
        1,
        u64::MAX,
        u64::MAX - 1,
        0x8000_0000_0000_0000, // Sign bit position
        0x7FFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
        0x0000_0000_FFFF_FFFF,
        0xDEAD_BEEF_CAFE_BABE,
        0x0123_4567_89AB_CDEF,
        42,
        u32::MAX as u64,
        (u32::MAX as u64) + 1,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_u64_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_global_u64_correctness",
                start.elapsed(),
                format!("Failed to create U64 buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_global_u64_correctness",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Verify row count preserved
    if ctx.device_row_count(&sorted) != data.len() as u64 {
        return TestResult::error(
            "test_global_u64_correctness",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                ctx.device_row_count(&sorted),
                data.len()
            ),
        );
    }

    // Download and verify
    let sorted_data = match ctx.provider.download_column_u64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_global_u64_correctness",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify sorted order
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_global_u64_correctness",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} > {}",
                    i,
                    sorted_data[i - 1],
                    sorted_data[i]
                ),
            );
        }
    }

    // Verify all original values are present
    let mut original_sorted = data.clone();
    original_sorted.sort();
    if sorted_data != original_sorted {
        return TestResult::error(
            "test_global_u64_correctness",
            start.elapsed(),
            format!(
                "Values not preserved: expected {:?}, got {:?}",
                original_sorted, sorted_data
            ),
        );
    }

    // Specifically verify U64::MAX is present
    if !sorted_data.contains(&u64::MAX) {
        return TestResult::error(
            "test_global_u64_correctness",
            start.elapsed(),
            "U64::MAX not found in sorted output".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_global_u64_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_global_u64_correctness", start.elapsed())
}

/// Test 3: Create I64 buffer with negative values, sort, verify.
///
/// Tests that signed I64 values including negative numbers are correctly
/// stored and sorted in global memory. Verifies correct signed comparison.
fn test_global_i64_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::I64)]);

    // Create test data with various I64 values including negatives
    let data: Vec<i64> = vec![
        0,
        1,
        -1,
        i64::MAX,
        i64::MIN,
        i64::MIN + 1,
        i64::MAX - 1,
        -100,
        100,
        -1000000,
        1000000,
        -9223372036854775807, // i64::MIN + 1
        9223372036854775806,  // i64::MAX - 1
        42,
        -42,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_i64_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_global_i64_correctness",
                start.elapsed(),
                format!("Failed to create I64 buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_global_i64_correctness",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Verify row count preserved
    if ctx.device_row_count(&sorted) != data.len() as u64 {
        return TestResult::error(
            "test_global_i64_correctness",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                ctx.device_row_count(&sorted),
                data.len()
            ),
        );
    }

    // Download and verify
    let sorted_data = match ctx.provider.download_column_i64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_global_i64_correctness",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify sorted order (signed comparison)
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_global_i64_correctness",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} > {}",
                    i,
                    sorted_data[i - 1],
                    sorted_data[i]
                ),
            );
        }
    }

    // Verify all original values are present
    let mut original_sorted = data.clone();
    original_sorted.sort();
    if sorted_data != original_sorted {
        return TestResult::error(
            "test_global_i64_correctness",
            start.elapsed(),
            format!(
                "Values not preserved: expected {:?}, got {:?}",
                original_sorted, sorted_data
            ),
        );
    }

    // Specifically verify negative values are sorted correctly
    // i64::MIN should be first
    if sorted_data[0] != i64::MIN {
        return TestResult::error(
            "test_global_i64_correctness",
            start.elapsed(),
            format!(
                "First element should be i64::MIN ({}), got {}",
                i64::MIN,
                sorted_data[0]
            ),
        );
    }

    // i64::MAX should be last
    if sorted_data[sorted_data.len() - 1] != i64::MAX {
        return TestResult::error(
            "test_global_i64_correctness",
            start.elapsed(),
            format!(
                "Last element should be i64::MAX ({}), got {}",
                i64::MAX,
                sorted_data[sorted_data.len() - 1]
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_global_i64_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_global_i64_correctness", start.elapsed())
}

/// Test 4: Create F64 buffer with edge values, sort, verify.
///
/// Tests that F64 values including special floating-point values are
/// correctly stored and sorted. Tests infinity, negative values, and
/// values near zero.
fn test_global_f64_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    // Create test data with various F64 values including edge cases
    // Note: Avoiding NaN as it has special comparison behavior
    let data: Vec<f64> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        std::f64::consts::PI,
        std::f64::consts::E,
        1e-300,
        -1e-300,
        1e300,
        -1e300,
        0.1,
        0.2,
        0.3,
        42.5,
        -42.5,
    ];

    let buffer = match ctx
        .provider
        .create_buffer_from_f64_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_global_f64_correctness",
                start.elapsed(),
                format!("Failed to create F64 buffer: {}", e),
            )
        }
    };

    // Sort the buffer
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_global_f64_correctness",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Verify row count preserved
    if ctx.device_row_count(&sorted) != data.len() as u64 {
        return TestResult::error(
            "test_global_f64_correctness",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                ctx.device_row_count(&sorted),
                data.len()
            ),
        );
    }

    // Download and verify
    let sorted_data = match ctx.provider.download_column_f64(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_global_f64_correctness",
                start.elapsed(),
                format!("Failed to download sorted column: {}", e),
            )
        }
    };

    // Verify sorted order (handle -0.0 == 0.0 specially)
    for i in 1..sorted_data.len() {
        // Use total_cmp for consistent ordering including -0.0 vs 0.0
        if sorted_data[i].total_cmp(&sorted_data[i - 1]) == std::cmp::Ordering::Less {
            return TestResult::error(
                "test_global_f64_correctness",
                start.elapsed(),
                format!(
                    "Sort order incorrect at index {}: {} > {}",
                    i,
                    sorted_data[i - 1],
                    sorted_data[i]
                ),
            );
        }
    }

    // Verify NEG_INFINITY is first (or among first if -0.0 handling varies)
    if sorted_data[0] != f64::NEG_INFINITY {
        return TestResult::error(
            "test_global_f64_correctness",
            start.elapsed(),
            format!(
                "First element should be NEG_INFINITY, got {}",
                sorted_data[0]
            ),
        );
    }

    // Verify INFINITY is last
    if sorted_data[sorted_data.len() - 1] != f64::INFINITY {
        return TestResult::error(
            "test_global_f64_correctness",
            start.elapsed(),
            format!(
                "Last element should be INFINITY, got {}",
                sorted_data[sorted_data.len() - 1]
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_global_f64_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_global_f64_correctness", start.elapsed())
}

/// Test 5: Create multiple buffers, operate on each, verify no cross-contamination.
///
/// Tests that multiple independent buffers in global memory don't interfere
/// with each other. Operations on one buffer should not affect others.
fn test_multi_buffer_isolation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Create three distinct buffers with different data
    let data_a: Vec<u32> = (0..1000).map(|i| i * 3).collect();
    let data_b: Vec<u32> = (0..1000).map(|i| i * 5 + 1).collect();
    let data_c: Vec<u32> = (0..1000).map(|i| i * 7 + 2).collect();

    let buffer_a = match ctx
        .provider
        .create_buffer_from_u32_slice(&data_a, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Failed to create buffer A: {}", e),
            )
        }
    };

    let buffer_b = match ctx
        .provider
        .create_buffer_from_u32_slice(&data_b, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Failed to create buffer B: {}", e),
            )
        }
    };

    let buffer_c = match ctx
        .provider
        .create_buffer_from_u32_slice(&data_c, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Failed to create buffer C: {}", e),
            )
        }
    };

    // Operate on buffer B (sort it)
    let sorted_b = match ctx.provider.sort(&buffer_b, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Sort of buffer B failed: {}", e),
            )
        }
    };

    // Verify buffer A is unchanged by downloading and checking
    let downloaded_a = match ctx.provider.download_column_u32(&buffer_a, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Failed to download buffer A: {}", e),
            )
        }
    };

    if downloaded_a != data_a {
        return TestResult::error(
            "test_multi_buffer_isolation",
            start.elapsed(),
            "Buffer A was corrupted after operating on buffer B".to_string(),
        );
    }

    // Verify buffer C is unchanged
    let downloaded_c = match ctx.provider.download_column_u32(&buffer_c, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Failed to download buffer C: {}", e),
            )
        }
    };

    if downloaded_c != data_c {
        return TestResult::error(
            "test_multi_buffer_isolation",
            start.elapsed(),
            "Buffer C was corrupted after operating on buffer B".to_string(),
        );
    }

    // Verify sorted_b has correct sorted values
    let downloaded_sorted_b = match ctx.provider.download_column_u32(&sorted_b, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Failed to download sorted buffer B: {}", e),
            )
        }
    };

    let mut expected_sorted_b = data_b.clone();
    expected_sorted_b.sort();
    if downloaded_sorted_b != expected_sorted_b {
        return TestResult::error(
            "test_multi_buffer_isolation",
            start.elapsed(),
            "Sorted buffer B has incorrect values".to_string(),
        );
    }

    // Now filter buffer A and verify B and C are still fine
    let mask_a: Vec<u8> = (0..data_a.len())
        .map(|i| if i % 2 == 0 { 1 } else { 0 })
        .collect();
    let filtered_a = match ctx.provider.filter_by_mask(&buffer_a, &mask_a) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Filter of buffer A failed: {}", e),
            )
        }
    };

    // Re-verify buffer C is still unchanged
    let downloaded_c_again = match ctx.provider.download_column_u32(&buffer_c, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Failed to download buffer C again: {}", e),
            )
        }
    };

    if downloaded_c_again != data_c {
        return TestResult::error(
            "test_multi_buffer_isolation",
            start.elapsed(),
            "Buffer C was corrupted after filtering buffer A".to_string(),
        );
    }

    // Verify filtered_a has correct values (every other element from original)
    let downloaded_filtered_a = match ctx.provider.download_column_u32(&filtered_a, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_multi_buffer_isolation",
                start.elapsed(),
                format!("Failed to download filtered buffer A: {}", e),
            )
        }
    };

    let expected_filtered_a: Vec<u32> = data_a
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, &v)| v)
        .collect();

    if downloaded_filtered_a != expected_filtered_a {
        return TestResult::error(
            "test_multi_buffer_isolation",
            start.elapsed(),
            "Filtered buffer A has incorrect values".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_multi_buffer_isolation",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_multi_buffer_isolation", start.elapsed())
}
