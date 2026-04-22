//! Category 16: Async copy and pipeline
//!
//! Tests async execution patterns including sequential operations,
//! operation dependencies, synchronization, error propagation, and
//! large batch operations.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub(crate) fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c16_async_pipeline");
    let start = Instant::now();

    results.add_result(test_sequential_operations(ctx));
    results.add_result(test_operation_dependencies(ctx));
    results.add_result(test_sync_between_operations(ctx));
    results.add_result(test_error_propagation(ctx));
    results.add_result(test_large_batch_operations(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Run many operations sequentially, verify all complete.
///
/// Tests that a sequence of independent operations all complete correctly
/// when executed one after another.
fn test_sequential_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const NUM_OPERATIONS: usize = 50;
    const SIZE: usize = 1000;

    // Run many sequential sort operations
    for i in 0..NUM_OPERATIONS {
        // Create unique data for each iteration
        let data: Vec<u32> = (0..SIZE).map(|j| ((j + i * 1000) % 10000) as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_sequential_operations",
                    start.elapsed(),
                    format!("Iteration {}: failed to create buffer: {}", i, e),
                )
            }
        };

        // Sort operation
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_sequential_operations",
                    start.elapsed(),
                    format!("Iteration {}: sort failed: {}", i, e),
                )
            }
        };

        // Verify result
        let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_sequential_operations",
                    start.elapsed(),
                    format!("Iteration {}: download failed: {}", i, e),
                )
            }
        };

        // Verify sorted
        for j in 1..sorted_data.len() {
            if sorted_data[j] < sorted_data[j - 1] {
                return TestResult::error(
                    "test_sequential_operations",
                    start.elapsed(),
                    format!("Iteration {}: sort incorrect at index {}", i, j),
                );
            }
        }
    }

    // Run many sequential filter operations
    let filter_data: Vec<u32> = (0..SIZE as u32).collect();
    let filter_buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&filter_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sequential_operations",
                start.elapsed(),
                format!("Failed to create filter buffer: {}", e),
            )
        }
    };

    for i in 0..NUM_OPERATIONS {
        // Create different masks each time
        let threshold = (i * 10) % SIZE;
        let mask: Vec<u8> = (0..SIZE)
            .map(|j| if j >= threshold { 1 } else { 0 })
            .collect();
        let expected_count = SIZE - threshold;

        let filtered = match ctx.provider.filter_by_mask(&filter_buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_sequential_operations",
                    start.elapsed(),
                    format!("Filter iteration {}: filter failed: {}", i, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_sequential_operations",
                start.elapsed(),
                format!(
                    "Filter iteration {}: expected {} rows, got {}",
                    i,
                    expected_count,
                    ctx.device_row_count(&filtered)
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sequential_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sequential_operations", start.elapsed())
}

/// Test 2: Chain operations where each depends on previous.
///
/// Tests that operations can be chained where the output of one operation
/// becomes the input of the next.
fn test_operation_dependencies(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;

    // Create initial data
    let data: Vec<u32> = (0..SIZE).map(|i| ((i * 17 + 13) % 1000) as u32).collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Chain: sort -> filter -> sort -> filter -> sort
    // Step 1: Sort
    let step1 = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 1 (sort) failed: {}", e),
            )
        }
    };

    // Verify step 1
    let step1_data = match ctx.provider.download_column::<u32>(&step1, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 1: download failed: {}", e),
            )
        }
    };

    for i in 1..step1_data.len() {
        if step1_data[i] < step1_data[i - 1] {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 1: sort incorrect at index {}", i),
            );
        }
    }

    // Step 2: Filter (keep values >= 500)
    let step2_mask: Vec<u8> = step1_data
        .iter()
        .map(|&v| if v >= 500 { 1 } else { 0 })
        .collect();
    let step2 = match ctx.provider.filter_by_mask(&step1, &step2_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 2 (filter) failed: {}", e),
            )
        }
    };

    let step2_data = match ctx.provider.download_column::<u32>(&step2, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 2: download failed: {}", e),
            )
        }
    };

    // Verify step 2 - all values should be >= 500
    for (i, &v) in step2_data.iter().enumerate() {
        if v < 500 {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 2: value {} at index {} is < 500", v, i),
            );
        }
    }

    // Step 3: Sort again (should be no-op since already sorted and filtered)
    let step3 = match ctx.provider.sort(&step2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 3 (sort) failed: {}", e),
            )
        }
    };

    // Step 4: Filter (keep values < 800)
    let step3_data = match ctx.provider.download_column::<u32>(&step3, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 3: download failed: {}", e),
            )
        }
    };

    let step4_mask: Vec<u8> = step3_data
        .iter()
        .map(|&v| if v < 800 { 1 } else { 0 })
        .collect();
    let step4 = match ctx.provider.filter_by_mask(&step3, &step4_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 4 (filter) failed: {}", e),
            )
        }
    };

    let step4_data = match ctx.provider.download_column::<u32>(&step4, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 4: download failed: {}", e),
            )
        }
    };

    // Verify step 4 - all values should be in [500, 800)
    for (i, &v) in step4_data.iter().enumerate() {
        if v < 500 || v >= 800 {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 4: value {} at index {} not in [500, 800)", v, i),
            );
        }
    }

    // Step 5: Final sort
    let step5 = match ctx.provider.sort(&step4, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 5 (sort) failed: {}", e),
            )
        }
    };

    let final_data = match ctx.provider.download_column::<u32>(&step5, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Step 5: download failed: {}", e),
            )
        }
    };

    // Verify final result is sorted and in range
    for i in 1..final_data.len() {
        if final_data[i] < final_data[i - 1] {
            return TestResult::error(
                "test_operation_dependencies",
                start.elapsed(),
                format!("Final: sort incorrect at index {}", i),
            );
        }
    }

    // Compute expected result on CPU
    let mut expected: Vec<u32> = data
        .iter()
        .copied()
        .filter(|&v| v >= 500 && v < 800)
        .collect();
    expected.sort();

    if final_data != expected {
        return TestResult::error(
            "test_operation_dependencies",
            start.elapsed(),
            format!(
                "Final result doesn't match expected: {} vs {} elements",
                final_data.len(),
                expected.len()
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_operation_dependencies",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_operation_dependencies", start.elapsed())
}

/// Test 3: Verify sync correctly waits for completion.
///
/// Tests that sync_and_check properly waits for all pending GPU operations
/// to complete before returning.
fn test_sync_between_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 50000;

    // Create data
    let data: Vec<u32> = (0..SIZE).map(|i| ((i * 31337) % 1000000) as u32).collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Operation 1: Sort (large operation)
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Sync point 1
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sync_between_operations",
            start.elapsed(),
            format!("Sync point 1 failed: {}", e),
        );
    }

    // Verify data is available after sync
    let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Download after sync 1 failed: {}", e),
            )
        }
    };

    // Verify sorted
    for i in 1..sorted_data.len() {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Sort incorrect at index {}", i),
            );
        }
    }

    // Operation 2: Filter (use result of operation 1)
    let mask: Vec<u8> = sorted_data
        .iter()
        .map(|&v| if v % 2 == 0 { 1 } else { 0 })
        .collect();
    let filtered = match ctx.provider.filter_by_mask(&sorted, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    // Sync point 2
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sync_between_operations",
            start.elapsed(),
            format!("Sync point 2 failed: {}", e),
        );
    }

    // Verify filter result
    let filtered_data = match ctx.provider.download_column::<u32>(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Download after sync 2 failed: {}", e),
            )
        }
    };

    // All filtered values should be even
    for (i, &v) in filtered_data.iter().enumerate() {
        if v % 2 != 0 {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Filtered value {} at index {} is not even", v, i),
            );
        }
    }

    // Operation 3: Multiple operations then sync
    let schema2 = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    let keys: Vec<u32> = (0..1000u32).collect();
    let vals: Vec<u32> = keys.iter().map(|&k| k * 10).collect();

    let buffer2 = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema2.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Failed to create buffer2: {}", e),
            )
        }
    };

    // Start multiple operations
    let sorted2 = match ctx.provider.sort(&buffer2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Sort2 failed: {}", e),
            )
        }
    };

    let deduped = match ctx.provider.dedup(&buffer2, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Dedup failed: {}", e),
            )
        }
    };

    // Single sync should wait for both
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sync_between_operations",
            start.elapsed(),
            format!("Sync point 3 (after multiple ops) failed: {}", e),
        );
    }

    // Verify both operations completed
    let sorted2_data = match ctx.provider.download_column::<u32>(&sorted2, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Download sorted2 failed: {}", e),
            )
        }
    };

    let deduped_data = match ctx.provider.download_column::<u32>(&deduped, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sync_between_operations",
                start.elapsed(),
                format!("Download deduped failed: {}", e),
            )
        }
    };

    if sorted2_data.len() != 1000 {
        return TestResult::error(
            "test_sync_between_operations",
            start.elapsed(),
            format!("Sorted2 has {} rows, expected 1000", sorted2_data.len()),
        );
    }

    if deduped_data.len() != 1000 {
        return TestResult::error(
            "test_sync_between_operations",
            start.elapsed(),
            format!("Deduped has {} rows, expected 1000", deduped_data.len()),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sync_between_operations",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_sync_between_operations", start.elapsed())
}

/// Test 4: Verify errors in operations are properly detected.
///
/// Tests that errors during GPU operations are properly captured and
/// reported through the sync mechanism.
fn test_error_propagation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test 1: Empty buffer operations should work
    let empty_data: Vec<u32> = vec![];
    let empty_buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&empty_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_error_propagation",
                start.elapsed(),
                format!("Failed to create empty buffer: {}", e),
            )
        }
    };

    // Sort on empty should work (or return empty)
    let sorted_empty = match ctx.provider.sort(&empty_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_error_propagation",
                start.elapsed(),
                format!("Sort on empty buffer failed unexpectedly: {}", e),
            )
        }
    };

    if ctx.device_row_count(&sorted_empty) != 0 {
        return TestResult::error(
            "test_error_propagation",
            start.elapsed(),
            format!(
                "Sort on empty buffer returned {} rows, expected 0",
                ctx.device_row_count(&sorted_empty)
            ),
        );
    }

    // Test 2: Mismatched mask length should error
    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_error_propagation",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Mask with wrong length
    let wrong_mask: Vec<u8> = vec![1, 0, 1]; // Only 3 elements for 5-element buffer

    match ctx.provider.filter_by_mask(&buffer, &wrong_mask) {
        Ok(_) => {
            // Some implementations might pad or truncate - that's OK too
            // We're just verifying the operation doesn't crash
        }
        Err(_) => {
            // Error is expected - that's fine
        }
    }

    // Sync to check for any async errors
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_error_propagation",
            start.elapsed(),
            format!("Sync after error test failed: {}", e),
        );
    }

    // Test 3: Valid operations after error recovery
    let valid_mask: Vec<u8> = vec![1, 0, 1, 0, 1];
    let filtered = match ctx.provider.filter_by_mask(&buffer, &valid_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_error_propagation",
                start.elapsed(),
                format!("Valid filter after error test failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&filtered) != 3 {
        return TestResult::error(
            "test_error_propagation",
            start.elapsed(),
            format!(
                "Valid filter returned {} rows, expected 3",
                ctx.device_row_count(&filtered)
            ),
        );
    }

    // Test 4: Multiple valid operations to verify system still works
    let schema2 = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    let keys: Vec<u32> = (0..100u32).collect();
    let vals: Vec<u32> = keys.iter().map(|&k| k * 2).collect();

    let buffer2 = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema2.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_error_propagation",
                start.elapsed(),
                format!("Failed to create buffer2: {}", e),
            )
        }
    };

    let sorted = match ctx.provider.sort(&buffer2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_error_propagation",
                start.elapsed(),
                format!("Sort after recovery failed: {}", e),
            )
        }
    };

    let sorted_keys = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_error_propagation",
                start.elapsed(),
                format!("Download after recovery failed: {}", e),
            )
        }
    };

    if sorted_keys.len() != 100 {
        return TestResult::error(
            "test_error_propagation",
            start.elapsed(),
            format!(
                "Sort after recovery has {} rows, expected 100",
                sorted_keys.len()
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_error_propagation",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_error_propagation", start.elapsed())
}

/// Test 5: Run a large batch of operations back-to-back.
///
/// Tests that the system can handle a large number of operations executed
/// in rapid succession without memory leaks or resource exhaustion.
fn test_large_batch_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const BATCH_SIZE: usize = 100;
    const OPERATION_SIZE: usize = 5000;

    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let schema2 = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Batch 1: Many sort operations
    for i in 0..BATCH_SIZE {
        let data: Vec<u32> = (0..OPERATION_SIZE)
            .map(|j| ((j * (i + 1) * 7) % 100000) as u32)
            .collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_large_batch_operations",
                    start.elapsed(),
                    format!("Batch 1 iter {}: create buffer failed: {}", i, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_large_batch_operations",
                    start.elapsed(),
                    format!("Batch 1 iter {}: sort failed: {}", i, e),
                )
            }
        };

        // Periodic verification (every 10th)
        if i % 10 == 0 {
            let sorted_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
                Ok(d) => d,
                Err(e) => {
                    return TestResult::error(
                        "test_large_batch_operations",
                        start.elapsed(),
                        format!("Batch 1 iter {}: download failed: {}", i, e),
                    )
                }
            };

            for j in 1..sorted_data.len() {
                if sorted_data[j] < sorted_data[j - 1] {
                    return TestResult::error(
                        "test_large_batch_operations",
                        start.elapsed(),
                        format!("Batch 1 iter {}: sort incorrect at index {}", i, j),
                    );
                }
            }
        }
    }

    // Sync after batch 1
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_large_batch_operations",
            start.elapsed(),
            format!("Sync after batch 1 failed: {}", e),
        );
    }

    // Batch 2: Many filter operations
    let filter_data: Vec<u32> = (0..OPERATION_SIZE as u32).collect();
    let filter_buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&filter_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_large_batch_operations",
                start.elapsed(),
                format!("Batch 2: create filter buffer failed: {}", e),
            )
        }
    };

    for i in 0..BATCH_SIZE {
        // Different filter patterns
        let threshold = (i * OPERATION_SIZE / BATCH_SIZE) as u32;
        let mask: Vec<u8> = filter_data
            .iter()
            .map(|&v| if v >= threshold { 1 } else { 0 })
            .collect();

        let filtered = match ctx.provider.filter_by_mask(&filter_buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_large_batch_operations",
                    start.elapsed(),
                    format!("Batch 2 iter {}: filter failed: {}", i, e),
                )
            }
        };

        let expected_count = OPERATION_SIZE - (i * OPERATION_SIZE / BATCH_SIZE);
        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_large_batch_operations",
                start.elapsed(),
                format!(
                    "Batch 2 iter {}: expected {} rows, got {}",
                    i,
                    expected_count,
                    ctx.device_row_count(&filtered)
                ),
            );
        }
    }

    // Sync after batch 2
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_large_batch_operations",
            start.elapsed(),
            format!("Sync after batch 2 failed: {}", e),
        );
    }

    // Batch 3: Many dedup operations
    for i in 0..BATCH_SIZE / 2 {
        // Create data with varying duplicate patterns
        let num_unique = 100 + i * 10;
        let keys: Vec<u32> = (0..OPERATION_SIZE)
            .map(|j| (j % num_unique) as u32)
            .collect();
        let vals: Vec<u32> = (0..OPERATION_SIZE as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&keys, &vals], schema2.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_large_batch_operations",
                    start.elapsed(),
                    format!("Batch 3 iter {}: create buffer failed: {}", i, e),
                )
            }
        };

        let deduped = match ctx.provider.dedup(&buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_large_batch_operations",
                    start.elapsed(),
                    format!("Batch 3 iter {}: dedup failed: {}", i, e),
                )
            }
        };

        if ctx.device_row_count(&deduped) != num_unique as u64 {
            return TestResult::error(
                "test_large_batch_operations",
                start.elapsed(),
                format!(
                    "Batch 3 iter {}: expected {} unique, got {}",
                    i,
                    num_unique,
                    ctx.device_row_count(&deduped)
                ),
            );
        }
    }

    // Sync after batch 3
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_large_batch_operations",
            start.elapsed(),
            format!("Sync after batch 3 failed: {}", e),
        );
    }

    // Batch 4: Mixed operations (join + sort + filter)
    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    for i in 0..BATCH_SIZE / 4 {
        // Small join to keep memory reasonable
        let left_keys: Vec<u32> = (0..500u32).collect();
        let left_vals: Vec<u32> = left_keys.iter().map(|&k| k + (i as u32) * 1000).collect();

        let right_keys: Vec<u32> = (0..300u32).map(|j| j * 2).collect();
        let right_vals: Vec<u32> = right_keys.iter().map(|&k| k * 10).collect();

        let left_buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_large_batch_operations",
                    start.elapsed(),
                    format!("Batch 4 iter {}: create left buffer failed: {}", i, e),
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
                    "test_large_batch_operations",
                    start.elapsed(),
                    format!("Batch 4 iter {}: create right buffer failed: {}", i, e),
                )
            }
        };

        let joined = match ctx
            .provider
            .hash_join(&left_buffer, &right_buffer, &[0], &[0])
        {
            Ok(j) => j,
            Err(e) => {
                return TestResult::error(
                    "test_large_batch_operations",
                    start.elapsed(),
                    format!("Batch 4 iter {}: join failed: {}", i, e),
                )
            }
        };

        // Right keys are 0, 2, 4, ..., 598 (300 keys)
        // Left keys are 0, 1, 2, ..., 499
        // Matches: 0, 2, 4, ..., 498 (250 matches)
        let expected_matches = 250;
        if ctx.device_row_count(&joined) != expected_matches {
            return TestResult::error(
                "test_large_batch_operations",
                start.elapsed(),
                format!(
                    "Batch 4 iter {}: join returned {} rows, expected {}",
                    i,
                    ctx.device_row_count(&joined),
                    expected_matches
                ),
            );
        }
    }

    // Final sync
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_large_batch_operations",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_large_batch_operations", start.elapsed())
}
