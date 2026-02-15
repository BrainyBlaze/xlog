//! Category 21: Hardware reliability
//!
//! Tests hardware error handling and reliability, including error detection,
//! recovery after errors, stress operations, memory pressure, and sustained
//! operation stability.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c21_hardware");
    let start = Instant::now();

    results.add_result(test_error_detection(ctx));
    results.add_result(test_recovery_after_error(ctx));
    results.add_result(test_stress_operations(ctx));
    results.add_result(test_memory_pressure(ctx));
    results.add_result(test_sustained_operation(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Verify errors are properly detected via sync_and_check.
///
/// Tests that the sync_and_check mechanism properly detects and reports
/// GPU errors, and that successful operations don't generate false errors.
fn test_error_detection(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test 1: Successful operation should not trigger error
    let data: Vec<u32> = (0..10000u32).collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Buffer creation failed: {}", e),
            )
        }
    };

    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // sync_and_check should succeed for valid operations
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_error_detection",
            start.elapsed(),
            format!("sync_and_check failed for valid operation: {}", e),
        );
    }

    // Verify result is correct
    let result = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Download failed: {}", e),
            )
        }
    };

    for (i, &val) in result.iter().enumerate() {
        if val != i as u32 {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!(
                    "Result incorrect at index {}: expected {}, got {}",
                    i, i, val
                ),
            );
        }
    }

    // Test 2: Multiple sync_and_check calls should all succeed
    for i in 0..5 {
        let check_data: Vec<u32> = (0..1000u32).map(|j| j + i * 1000).collect();

        let check_buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&check_data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_error_detection",
                    start.elapsed(),
                    format!("Check {} buffer creation failed: {}", i, e),
                )
            }
        };

        let _check_sorted = match ctx.provider.sort(&check_buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_error_detection",
                    start.elapsed(),
                    format!("Check {} sort failed: {}", i, e),
                )
            }
        };

        if let Err(e) = ctx.sync_and_check() {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Check {} sync_and_check failed: {}", i, e),
            );
        }
    }

    // Test 3: Empty buffer operations should not cause errors
    let empty_data: Vec<u32> = vec![];
    let empty_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&empty_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Empty buffer creation failed: {}", e),
            )
        }
    };

    let _empty_sorted = match ctx.provider.sort(&empty_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Empty sort failed: {}", e),
            )
        }
    };

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_error_detection",
            start.elapsed(),
            format!("sync_and_check failed for empty operation: {}", e),
        );
    }

    // Test 4: Single element operations
    let single_data: Vec<u32> = vec![42];
    let single_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&single_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Single buffer creation failed: {}", e),
            )
        }
    };

    let single_sorted = match ctx.provider.sort(&single_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Single sort failed: {}", e),
            )
        }
    };

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_error_detection",
            start.elapsed(),
            format!("sync_and_check failed for single element: {}", e),
        );
    }

    let single_result = match ctx.provider.download_column_u32(&single_sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_error_detection",
                start.elapsed(),
                format!("Single download failed: {}", e),
            )
        }
    };

    if single_result != vec![42] {
        return TestResult::error(
            "test_error_detection",
            start.elapsed(),
            format!("Single result incorrect: {:?}", single_result),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_error_detection",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_error_detection", start.elapsed())
}

/// Test 2: Test recovery after sync_and_check detects issues.
///
/// Tests that the system can recover and continue operating correctly
/// after an error or edge case is handled.
fn test_recovery_after_error(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test 1: Recover after handling edge cases
    // First, do a valid operation
    let data1: Vec<u32> = (0..5000u32).collect();
    let buffer1 = match ctx
        .provider
        .create_buffer_from_u32_slice(&data1, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!("Buffer1 creation failed: {}", e),
            )
        }
    };

    let sorted1 = match ctx.provider.sort(&buffer1, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!("Sort1 failed: {}", e),
            )
        }
    };

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_recovery_after_error",
            start.elapsed(),
            format!("Sync1 failed: {}", e),
        );
    }

    // Try an edge case operation (mismatched mask size - may error or be handled)
    let edge_data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let edge_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&edge_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!("Edge buffer creation failed: {}", e),
            )
        }
    };

    // This may succeed with truncation or fail - both are acceptable
    let wrong_mask: Vec<u8> = vec![1, 0, 1]; // Wrong size
    let _edge_result = ctx.provider.filter_by_mask(&edge_buffer, &wrong_mask);

    // Sync to clear any pending errors
    let _ = ctx.sync_and_check();

    // Test 2: Verify system recovered - operations should work again
    let data2: Vec<u32> = (0..10000u32).collect();
    let buffer2 = match ctx
        .provider
        .create_buffer_from_u32_slice(&data2, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!("Recovery buffer creation failed: {}", e),
            )
        }
    };

    let sorted2 = match ctx.provider.sort(&buffer2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!("Recovery sort failed: {}", e),
            )
        }
    };

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_recovery_after_error",
            start.elapsed(),
            format!("Recovery sync failed: {}", e),
        );
    }

    // Verify recovery was complete
    let result2 = match ctx.provider.download_column_u32(&sorted2, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!("Recovery download failed: {}", e),
            )
        }
    };

    for (i, &val) in result2.iter().enumerate() {
        if val != i as u32 {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!(
                    "Recovery result incorrect at {}: expected {}, got {}",
                    i, i, val
                ),
            );
        }
    }

    // Test 3: Multiple recovery cycles
    for cycle in 0..3 {
        // Valid operation
        let valid_data: Vec<u32> = (0..1000u32).map(|j| j + cycle * 1000).collect();
        let valid_buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&valid_data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_recovery_after_error",
                    start.elapsed(),
                    format!("Cycle {} valid buffer failed: {}", cycle, e),
                )
            }
        };

        let valid_sorted = match ctx.provider.sort(&valid_buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_recovery_after_error",
                    start.elapsed(),
                    format!("Cycle {} valid sort failed: {}", cycle, e),
                )
            }
        };

        if let Err(e) = ctx.sync_and_check() {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!("Cycle {} valid sync failed: {}", cycle, e),
            );
        }

        // Verify
        let valid_result = match ctx.provider.download_column_u32(&valid_sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_recovery_after_error",
                    start.elapsed(),
                    format!("Cycle {} valid download failed: {}", cycle, e),
                )
            }
        };

        if valid_result.len() != 1000 {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!(
                    "Cycle {}: expected 1000 rows, got {}",
                    cycle,
                    valid_result.len()
                ),
            );
        }
    }

    // Test 4: Verify previous results still accessible
    let result1 = match ctx.provider.download_column_u32(&sorted1, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!("Previous result download failed: {}", e),
            )
        }
    };

    for (i, &val) in result1.iter().enumerate() {
        if val != i as u32 {
            return TestResult::error(
                "test_recovery_after_error",
                start.elapsed(),
                format!(
                    "Previous result corrupted at {}: expected {}, got {}",
                    i, i, val
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_recovery_after_error",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_recovery_after_error", start.elapsed())
}

/// Test 3: Run many operations to stress hardware.
///
/// Tests hardware reliability under heavy load by running a large number
/// of operations in rapid succession.
fn test_stress_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const NUM_OPERATIONS: usize = 200;
    const DATA_SIZE: usize = 10000;

    // Stress test: many sort operations
    for i in 0..NUM_OPERATIONS {
        let data: Vec<u32> = (0..DATA_SIZE)
            .map(|j| ((j * (i + 1) * 1103515245 + 12345) % DATA_SIZE) as u32)
            .collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!("Stress sort {}: buffer creation failed: {}", i, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!("Stress sort {}: sort failed: {}", i, e),
                )
            }
        };

        // Periodic verification (every 20 operations)
        if i % 20 == 0 {
            let result = match ctx.provider.download_column_u32(&sorted, 0) {
                Ok(d) => d,
                Err(e) => {
                    return TestResult::error(
                        "test_stress_operations",
                        start.elapsed(),
                        format!("Stress sort {}: download failed: {}", i, e),
                    )
                }
            };

            for j in 1..result.len() {
                if result[j] < result[j - 1] {
                    return TestResult::error(
                        "test_stress_operations",
                        start.elapsed(),
                        format!("Stress sort {}: incorrect at index {}", i, j),
                    );
                }
            }

            if let Err(e) = ctx.sync_and_check() {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!("Stress sort {}: sync failed: {}", i, e),
                );
            }
        }
    }

    // Stress test: many filter operations
    let filter_data: Vec<u32> = (0..DATA_SIZE as u32).collect();
    let filter_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&filter_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_stress_operations",
                start.elapsed(),
                format!("Filter buffer creation failed: {}", e),
            )
        }
    };

    for i in 0..NUM_OPERATIONS {
        let selectivity = (i % 10 + 1) * 10; // 10%, 20%, ..., 100%
        let mask: Vec<u8> = (0..DATA_SIZE)
            .map(|j| {
                if (j * 100 / DATA_SIZE) < selectivity {
                    1
                } else {
                    0
                }
            })
            .collect();

        let filtered = match ctx.provider.filter_by_mask(&filter_buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!("Stress filter {}: failed: {}", i, e),
                )
            }
        };

        // Periodic verification
        if i % 20 == 0 {
            let expected_min = (DATA_SIZE * selectivity / 100).saturating_sub(DATA_SIZE / 20);
            let expected_max = (DATA_SIZE * selectivity / 100) + DATA_SIZE / 20 + 1;

            let count = ctx.device_row_count(&filtered) as usize;
            if count < expected_min || count > expected_max {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!(
                        "Stress filter {}: got {} rows, expected ~{}",
                        i,
                        count,
                        DATA_SIZE * selectivity / 100
                    ),
                );
            }

            if let Err(e) = ctx.sync_and_check() {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!("Stress filter {}: sync failed: {}", i, e),
                );
            }
        }
    }

    // Stress test: mixed operations
    let schema2 = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    for i in 0..NUM_OPERATIONS / 2 {
        // Dedup
        let dedup_keys: Vec<u32> = (0..DATA_SIZE).map(|j| (j % 100) as u32).collect();
        let dedup_vals: Vec<u32> = (0..DATA_SIZE as u32).collect();

        let dedup_buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&dedup_keys, &dedup_vals], schema2.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!("Stress dedup {}: buffer creation failed: {}", i, e),
                )
            }
        };

        let deduped = match ctx.provider.dedup(&dedup_buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!("Stress dedup {}: failed: {}", i, e),
                )
            }
        };

        if ctx.device_row_count(&deduped) != 100 {
            return TestResult::error(
                "test_stress_operations",
                start.elapsed(),
                format!(
                    "Stress dedup {}: expected 100, got {}",
                    i,
                    ctx.device_row_count(&deduped)
                ),
            );
        }

        if i % 20 == 0 {
            if let Err(e) = ctx.sync_and_check() {
                return TestResult::error(
                    "test_stress_operations",
                    start.elapsed(),
                    format!("Stress dedup {}: sync failed: {}", i, e),
                );
            }
        }
    }

    // Final comprehensive sync
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_stress_operations",
            start.elapsed(),
            format!("Final stress sync failed: {}", e),
        );
    }

    TestResult::passed("test_stress_operations", start.elapsed())
}

/// Test 4: Test operations at high memory usage.
///
/// Tests system stability when operating near memory limits.
fn test_memory_pressure(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let budget = ctx.memory_budget();

    // Calculate size that uses significant portion of budget
    // Each u32 is 4 bytes, aim for ~20% of budget per buffer
    let buffer_size = (budget / 20 / 4).min(5_000_000);

    if buffer_size < 1000 {
        return TestResult::skipped(
            "test_memory_pressure",
            "Memory budget too small for pressure test",
        );
    }

    // Create multiple buffers to approach memory limit
    let mut buffers = Vec::new();
    let mut successful_allocations = 0;

    for i in 0..5 {
        let data: Vec<u32> = (0..buffer_size)
            .map(|j| ((j + i * buffer_size) % buffer_size) as u32)
            .collect();

        match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => {
                buffers.push(buf);
                successful_allocations += 1;
            }
            Err(_) => {
                // Memory limit reached - this is acceptable
                break;
            }
        }
    }

    // We should have been able to allocate at least one buffer
    if successful_allocations == 0 {
        return TestResult::error(
            "test_memory_pressure",
            start.elapsed(),
            "Could not allocate any buffers".to_string(),
        );
    }

    // Perform operations on allocated buffers
    for (i, buffer) in buffers.iter().enumerate() {
        let sorted = match ctx.provider.sort(buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_memory_pressure",
                    start.elapsed(),
                    format!("Buffer {}: sort failed under pressure: {}", i, e),
                )
            }
        };

        // Verify correctness
        let result = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_memory_pressure",
                    start.elapsed(),
                    format!("Buffer {}: download failed under pressure: {}", i, e),
                )
            }
        };

        // Spot check sorted
        for j in (1..result.len()).step_by(10000) {
            if result[j] < result[j - 1] {
                return TestResult::error(
                    "test_memory_pressure",
                    start.elapsed(),
                    format!("Buffer {}: sort incorrect at index {}", i, j),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_memory_pressure",
            start.elapsed(),
            format!("Sync under pressure failed: {}", e),
        );
    }

    // Test filter under pressure
    for (i, buffer) in buffers.iter().enumerate() {
        let mask: Vec<u8> = (0..buffer_size)
            .map(|j| if j % 2 == 0 { 1 } else { 0 })
            .collect();

        let filtered = match ctx.provider.filter_by_mask(buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_memory_pressure",
                    start.elapsed(),
                    format!("Buffer {}: filter failed under pressure: {}", i, e),
                )
            }
        };

        let expected = (buffer_size + 1) / 2;
        if ctx.device_row_count(&filtered) != expected as u64 {
            return TestResult::error(
                "test_memory_pressure",
                start.elapsed(),
                format!(
                    "Buffer {}: filter expected {} rows, got {}",
                    i,
                    expected,
                    ctx.device_row_count(&filtered)
                ),
            );
        }
    }

    // Release buffers and verify operations still work
    drop(buffers);

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_memory_pressure",
            start.elapsed(),
            format!("Sync after release failed: {}", e),
        );
    }

    // New operations should work after release
    let fresh_data: Vec<u32> = (0..10000u32).collect();
    let fresh_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&fresh_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_memory_pressure",
                start.elapsed(),
                format!("Fresh buffer after pressure failed: {}", e),
            )
        }
    };

    let fresh_sorted = match ctx.provider.sort(&fresh_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_memory_pressure",
                start.elapsed(),
                format!("Fresh sort after pressure failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&fresh_sorted) != 10000 {
        return TestResult::error(
            "test_memory_pressure",
            start.elapsed(),
            format!(
                "Fresh result after pressure: expected 10000, got {}",
                ctx.device_row_count(&fresh_sorted)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_memory_pressure",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_memory_pressure", start.elapsed())
}

/// Test 5: Long-running operations verify stability.
///
/// Tests hardware stability by running operations over an extended period
/// to detect any degradation or intermittent failures.
fn test_sustained_operation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 50000;
    const DURATION_SECONDS: u64 = 5; // Run for at least 5 seconds

    let deadline = start + std::time::Duration::from_secs(DURATION_SECONDS);
    let mut operation_count = 0;
    let mut error_count = 0;

    // Create reference data
    let reference_data: Vec<u32> = (0..SIZE)
        .map(|i| ((i * 1103515245 + 12345) % SIZE) as u32)
        .collect();

    let mut expected_sorted = reference_data.clone();
    expected_sorted.sort();

    // Run operations until deadline
    while Instant::now() < deadline {
        let data: Vec<u32> = reference_data.clone();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(_) => {
                error_count += 1;
                continue;
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(_) => {
                error_count += 1;
                continue;
            }
        };

        // Periodic full verification
        if operation_count % 10 == 0 {
            let result = match ctx.provider.download_column_u32(&sorted, 0) {
                Ok(d) => d,
                Err(_) => {
                    error_count += 1;
                    continue;
                }
            };

            if result != expected_sorted {
                return TestResult::error(
                    "test_sustained_operation",
                    start.elapsed(),
                    format!(
                        "Operation {} produced incorrect result after {:?}",
                        operation_count,
                        start.elapsed()
                    ),
                );
            }

            if let Err(e) = ctx.sync_and_check() {
                return TestResult::error(
                    "test_sustained_operation",
                    start.elapsed(),
                    format!("Sync failed at operation {}: {}", operation_count, e),
                );
            }
        }

        operation_count += 1;
    }

    // Should have completed many operations
    if operation_count < 10 {
        return TestResult::error(
            "test_sustained_operation",
            start.elapsed(),
            format!(
                "Too few operations completed: {} (expected >= 10)",
                operation_count
            ),
        );
    }

    // Error rate should be very low
    if error_count > operation_count / 100 {
        return TestResult::error(
            "test_sustained_operation",
            start.elapsed(),
            format!(
                "High error rate: {}/{} operations failed",
                error_count, operation_count
            ),
        );
    }

    // Run sustained filter operations
    let filter_data: Vec<u32> = (0..SIZE as u32).collect();
    let filter_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&filter_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sustained_operation",
                start.elapsed(),
                format!("Sustained filter buffer failed: {}", e),
            )
        }
    };

    let deadline2 = Instant::now() + std::time::Duration::from_secs(DURATION_SECONDS / 2);
    let mut filter_count = 0;

    while Instant::now() < deadline2 {
        let selectivity = (filter_count % 10 + 1) * 10;
        let mask: Vec<u8> = (0..SIZE)
            .map(|j| if (j * 100 / SIZE) < selectivity { 1 } else { 0 })
            .collect();

        let filtered = match ctx.provider.filter_by_mask(&filter_buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_sustained_operation",
                    start.elapsed(),
                    format!("Sustained filter {} failed: {}", filter_count, e),
                )
            }
        };

        // Verify row count is reasonable
        let expected_min = (SIZE * selectivity / 100).saturating_sub(SIZE / 10);
        let expected_max = (SIZE * selectivity / 100) + SIZE / 10 + 1;

        let count = ctx.device_row_count(&filtered) as usize;
        if count < expected_min || count > expected_max {
            return TestResult::error(
                "test_sustained_operation",
                start.elapsed(),
                format!(
                    "Sustained filter {}: got {} rows, expected ~{}",
                    filter_count,
                    count,
                    SIZE * selectivity / 100
                ),
            );
        }

        filter_count += 1;
    }

    // Run sustained join operations
    let schema2 = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    let left_keys: Vec<u32> = (0..1000u32).collect();
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k * 2).collect();

    let right_keys: Vec<u32> = (0..500u32).map(|i| i * 2).collect();
    let right_vals: Vec<u32> = right_keys.iter().map(|&k| k * 3).collect();

    let left_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&left_keys, &left_vals], schema2.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sustained_operation",
                start.elapsed(),
                format!("Sustained join left buffer failed: {}", e),
            )
        }
    };

    let right_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&right_keys, &right_vals], schema2)
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sustained_operation",
                start.elapsed(),
                format!("Sustained join right buffer failed: {}", e),
            )
        }
    };

    let deadline3 = Instant::now() + std::time::Duration::from_secs(DURATION_SECONDS / 2);
    let mut join_count = 0;

    while Instant::now() < deadline3 {
        let joined = match ctx
            .provider
            .hash_join(&left_buffer, &right_buffer, &[0], &[0])
        {
            Ok(j) => j,
            Err(e) => {
                return TestResult::error(
                    "test_sustained_operation",
                    start.elapsed(),
                    format!("Sustained join {} failed: {}", join_count, e),
                )
            }
        };

        if ctx.device_row_count(&joined) != 500 {
            return TestResult::error(
                "test_sustained_operation",
                start.elapsed(),
                format!(
                    "Sustained join {}: expected 500 rows, got {}",
                    join_count,
                    ctx.device_row_count(&joined)
                ),
            );
        }

        join_count += 1;
    }

    // Final verification
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sustained_operation",
            start.elapsed(),
            format!("Final sustained sync failed: {}", e),
        );
    }

    TestResult::passed("test_sustained_operation", start.elapsed())
}
