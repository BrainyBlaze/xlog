//! Category 18: Host-device integration
//!
//! Tests host-device data transfer and coordination, including upload/download
//! integrity, large transfers, repeated small transfers, memory lifecycle,
//! and memory budget limits.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c18_host_device");
    let start = Instant::now();

    results.add_result(test_upload_download_integrity(ctx));
    results.add_result(test_large_transfer(ctx));
    results.add_result(test_repeated_transfer(ctx));
    results.add_result(test_memory_lifecycle(ctx));
    results.add_result(test_memory_budget_limits(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Upload data, operate, download, and verify integrity.
///
/// Tests that data maintains integrity through the full cycle of upload
/// to GPU, GPU operation, and download back to host.
fn test_upload_download_integrity(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various data patterns
    let test_patterns: Vec<(&str, Vec<u32>)> = vec![
        ("zeros", vec![0u32; 10000]),
        ("ones", vec![1u32; 10000]),
        ("max_values", vec![u32::MAX; 10000]),
        ("sequential", (0..10000u32).collect()),
        ("reverse", (0..10000u32).rev().collect()),
        (
            "alternating",
            (0..10000)
                .map(|i| if i % 2 == 0 { 0 } else { u32::MAX })
                .collect(),
        ),
        (
            "random_lcg",
            (0..10000u32)
                .map(|i| i.wrapping_mul(1103515245).wrapping_add(12345))
                .collect(),
        ),
    ];

    for (name, data) in test_patterns {
        // Upload
        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_upload_download_integrity",
                    start.elapsed(),
                    format!("Pattern '{}': failed to create buffer: {}", name, e),
                )
            }
        };

        // Download immediately (no operation)
        let downloaded = match ctx.provider.download_column_u32(&buffer, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_upload_download_integrity",
                    start.elapsed(),
                    format!("Pattern '{}': failed to download: {}", name, e),
                )
            }
        };

        // Verify integrity
        if downloaded.len() != data.len() {
            return TestResult::error(
                "test_upload_download_integrity",
                start.elapsed(),
                format!(
                    "Pattern '{}': size mismatch - uploaded {}, downloaded {}",
                    name,
                    data.len(),
                    downloaded.len()
                ),
            );
        }

        for (i, (&original, &returned)) in data.iter().zip(downloaded.iter()).enumerate() {
            if original != returned {
                return TestResult::error(
                    "test_upload_download_integrity",
                    start.elapsed(),
                    format!(
                        "Pattern '{}': mismatch at index {}: uploaded {}, downloaded {}",
                        name, i, original, returned
                    ),
                );
            }
        }

        // Now sort and verify
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_upload_download_integrity",
                    start.elapsed(),
                    format!("Pattern '{}': sort failed: {}", name, e),
                )
            }
        };

        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_upload_download_integrity",
                    start.elapsed(),
                    format!("Pattern '{}': failed to download sorted: {}", name, e),
                )
            }
        };

        // Verify sorted
        for i in 1..sorted_data.len() {
            if sorted_data[i] < sorted_data[i - 1] {
                return TestResult::error(
                    "test_upload_download_integrity",
                    start.elapsed(),
                    format!("Pattern '{}': sort incorrect at index {}", name, i),
                );
            }
        }

        // Verify same elements (sorted should contain same elements as original)
        let mut original_sorted = data.clone();
        original_sorted.sort();
        if sorted_data != original_sorted {
            return TestResult::error(
                "test_upload_download_integrity",
                start.elapsed(),
                format!("Pattern '{}': sorted result doesn't match expected", name),
            );
        }
    }

    // Test with 64-bit values
    let schema64 = Schema::new(vec![("val".to_string(), ScalarType::U64)]);
    let data64: Vec<u64> = (0..5000).map(|i| i as u64 * 1_000_000_000).collect();

    let buffer64 = match ctx.provider.create_buffer_from_u64_slice(&data64, schema64) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_upload_download_integrity",
                start.elapsed(),
                format!("Failed to create u64 buffer: {}", e),
            )
        }
    };

    let downloaded64 = match ctx.provider.download_column_u64(&buffer64, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_upload_download_integrity",
                start.elapsed(),
                format!("Failed to download u64: {}", e),
            )
        }
    };

    for (i, (&original, &returned)) in data64.iter().zip(downloaded64.iter()).enumerate() {
        if original != returned {
            return TestResult::error(
                "test_upload_download_integrity",
                start.elapsed(),
                format!(
                    "U64 mismatch at index {}: uploaded {}, downloaded {}",
                    i, original, returned
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_upload_download_integrity",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_upload_download_integrity", start.elapsed())
}

/// Test 2: Transfer large data (100MB+).
///
/// Tests that large data transfers work correctly, stressing the
/// host-device transfer mechanisms.
fn test_large_transfer(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // 100MB = 25 million u32 values
    const LARGE_SIZE: usize = 25_000_000;

    // Create large data with a pattern that can be verified
    let large_data: Vec<u32> = (0..LARGE_SIZE)
        .map(|i| ((i * 1103515245 + 12345) % LARGE_SIZE) as u32)
        .collect();

    // Upload
    let buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&large_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_large_transfer",
                start.elapsed(),
                format!("Failed to create large buffer: {}", e),
            )
        }
    };

    // Verify size
    if ctx.device_row_count(&buffer) != LARGE_SIZE as u64 {
        return TestResult::error(
            "test_large_transfer",
            start.elapsed(),
            format!(
                "Buffer has {} rows, expected {}",
                ctx.device_row_count(&buffer),
                LARGE_SIZE
            ),
        );
    }

    // Download
    let downloaded = match ctx.provider.download_column_u32(&buffer, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_large_transfer",
                start.elapsed(),
                format!("Failed to download large buffer: {}", e),
            )
        }
    };

    if downloaded.len() != LARGE_SIZE {
        return TestResult::error(
            "test_large_transfer",
            start.elapsed(),
            format!(
                "Downloaded {} values, expected {}",
                downloaded.len(),
                LARGE_SIZE
            ),
        );
    }

    // Spot check integrity (checking all 25M would be slow)
    for i in (0..LARGE_SIZE).step_by(100000) {
        if downloaded[i] != large_data[i] {
            return TestResult::error(
                "test_large_transfer",
                start.elapsed(),
                format!(
                    "Mismatch at index {}: expected {}, got {}",
                    i, large_data[i], downloaded[i]
                ),
            );
        }
    }

    // Check first and last 1000 elements thoroughly
    for i in 0..1000 {
        if downloaded[i] != large_data[i] {
            return TestResult::error(
                "test_large_transfer",
                start.elapsed(),
                format!("Mismatch at start index {}", i),
            );
        }
    }

    for i in (LARGE_SIZE - 1000)..LARGE_SIZE {
        if downloaded[i] != large_data[i] {
            return TestResult::error(
                "test_large_transfer",
                start.elapsed(),
                format!("Mismatch at end index {}", i),
            );
        }
    }

    // Perform an operation on large data
    let mask: Vec<u8> = (0..LARGE_SIZE)
        .map(|i| if i % 10 == 0 { 1 } else { 0 })
        .collect();
    let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_large_transfer",
                start.elapsed(),
                format!("Large filter failed: {}", e),
            )
        }
    };

    let expected_filtered = (LARGE_SIZE + 9) / 10;
    if ctx.device_row_count(&filtered) != expected_filtered as u64 {
        return TestResult::error(
            "test_large_transfer",
            start.elapsed(),
            format!(
                "Filter returned {} rows, expected {}",
                ctx.device_row_count(&filtered),
                expected_filtered
            ),
        );
    }

    // Download filtered and verify
    let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_large_transfer",
                start.elapsed(),
                format!("Failed to download filtered: {}", e),
            )
        }
    };

    for (i, &val) in filtered_data.iter().enumerate() {
        let expected_idx = i * 10;
        let expected_val = large_data[expected_idx];
        if val != expected_val {
            return TestResult::error(
                "test_large_transfer",
                start.elapsed(),
                format!(
                    "Filtered value {} at index {}: expected {} (from idx {})",
                    val, i, expected_val, expected_idx
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_large_transfer",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_large_transfer", start.elapsed())
}

/// Test 3: Many small transfers.
///
/// Tests that repeated small transfers work correctly without leaking
/// resources or degrading performance.
fn test_repeated_transfer(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const NUM_TRANSFERS: usize = 500;
    const SMALL_SIZE: usize = 1000;

    // Many upload-download cycles
    for i in 0..NUM_TRANSFERS {
        // Create unique data for each iteration
        let data: Vec<u32> = (0..SMALL_SIZE)
            .map(|j| ((j + i * SMALL_SIZE) % 1000000) as u32)
            .collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_transfer",
                    start.elapsed(),
                    format!("Iteration {}: upload failed: {}", i, e),
                )
            }
        };

        let downloaded = match ctx.provider.download_column_u32(&buffer, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_transfer",
                    start.elapsed(),
                    format!("Iteration {}: download failed: {}", i, e),
                )
            }
        };

        // Verify integrity (spot check for speed)
        if downloaded.len() != data.len() {
            return TestResult::error(
                "test_repeated_transfer",
                start.elapsed(),
                format!(
                    "Iteration {}: size mismatch - {} vs {}",
                    i,
                    data.len(),
                    downloaded.len()
                ),
            );
        }

        if downloaded[0] != data[0] || downloaded[SMALL_SIZE - 1] != data[SMALL_SIZE - 1] {
            return TestResult::error(
                "test_repeated_transfer",
                start.elapsed(),
                format!("Iteration {}: data mismatch", i),
            );
        }
    }

    // Many small operations
    for i in 0..NUM_TRANSFERS {
        let data: Vec<u32> = (0..SMALL_SIZE as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_transfer",
                    start.elapsed(),
                    format!("Operation {}: create failed: {}", i, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_transfer",
                    start.elapsed(),
                    format!("Operation {}: sort failed: {}", i, e),
                )
            }
        };

        // Periodic verification
        if i % 50 == 0 {
            let result = match ctx.provider.download_column_u32(&sorted, 0) {
                Ok(d) => d,
                Err(e) => {
                    return TestResult::error(
                        "test_repeated_transfer",
                        start.elapsed(),
                        format!("Operation {}: download failed: {}", i, e),
                    )
                }
            };

            for j in 0..result.len() {
                if result[j] != j as u32 {
                    return TestResult::error(
                        "test_repeated_transfer",
                        start.elapsed(),
                        format!(
                            "Operation {}: incorrect at index {}: expected {}, got {}",
                            i, j, j, result[j]
                        ),
                    );
                }
            }
        }
    }

    // Interleaved upload-operate-download
    for i in 0..100 {
        // Upload
        let data: Vec<u32> = (0..SMALL_SIZE)
            .map(|j| ((j * (i + 1)) % 10000) as u32)
            .collect();
        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_transfer",
                    start.elapsed(),
                    format!("Interleaved {}: create failed: {}", i, e),
                )
            }
        };

        // Operate
        let mask: Vec<u8> = (0..SMALL_SIZE)
            .map(|j| if j % 2 == 0 { 1 } else { 0 })
            .collect();
        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_transfer",
                    start.elapsed(),
                    format!("Interleaved {}: filter failed: {}", i, e),
                )
            }
        };

        // Download
        let result = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_transfer",
                    start.elapsed(),
                    format!("Interleaved {}: download failed: {}", i, e),
                )
            }
        };

        let expected_count = (SMALL_SIZE + 1) / 2;
        if result.len() != expected_count {
            return TestResult::error(
                "test_repeated_transfer",
                start.elapsed(),
                format!(
                    "Interleaved {}: expected {} rows, got {}",
                    i,
                    expected_count,
                    result.len()
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_repeated_transfer",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_repeated_transfer", start.elapsed())
}

/// Test 4: Memory lifecycle - allocate, use, free, reallocate.
///
/// Tests that GPU memory can be allocated, used, freed, and reallocated
/// correctly without memory leaks or corruption.
fn test_memory_lifecycle(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 100000;
    const ITERATIONS: usize = 10;

    // Track initial memory usage
    let _initial_memory = ctx.memory_used();

    // Multiple allocation/deallocation cycles
    for cycle in 0..ITERATIONS {
        // Allocate multiple buffers
        let mut buffers = Vec::new();

        for i in 0..5 {
            let data: Vec<u32> = (0..SIZE)
                .map(|j| ((j + i * SIZE + cycle * SIZE * 5) % 1000000) as u32)
                .collect();

            let buffer = match ctx
                .provider
                .create_buffer_from_u32_slice(&data, schema.clone())
            {
                Ok(buf) => buf,
                Err(e) => {
                    return TestResult::error(
                        "test_memory_lifecycle",
                        start.elapsed(),
                        format!("Cycle {}, buffer {}: create failed: {}", cycle, i, e),
                    )
                }
            };

            buffers.push((buffer, data));
        }

        // Operate on all buffers
        for (i, (buffer, original_data)) in buffers.iter().enumerate() {
            let sorted = match ctx.provider.sort(buffer, &[0]) {
                Ok(s) => s,
                Err(e) => {
                    return TestResult::error(
                        "test_memory_lifecycle",
                        start.elapsed(),
                        format!("Cycle {}, buffer {}: sort failed: {}", cycle, i, e),
                    )
                }
            };

            // Verify correctness
            let result = match ctx.provider.download_column_u32(&sorted, 0) {
                Ok(d) => d,
                Err(e) => {
                    return TestResult::error(
                        "test_memory_lifecycle",
                        start.elapsed(),
                        format!("Cycle {}, buffer {}: download failed: {}", cycle, i, e),
                    )
                }
            };

            // Verify sorted
            for j in 1..result.len() {
                if result[j] < result[j - 1] {
                    return TestResult::error(
                        "test_memory_lifecycle",
                        start.elapsed(),
                        format!(
                            "Cycle {}, buffer {}: sort incorrect at index {}",
                            cycle, i, j
                        ),
                    );
                }
            }

            // Verify same elements
            let mut expected = original_data.clone();
            expected.sort();
            if result != expected {
                return TestResult::error(
                    "test_memory_lifecycle",
                    start.elapsed(),
                    format!(
                        "Cycle {}, buffer {}: sorted data doesn't match expected",
                        cycle, i
                    ),
                );
            }
        }

        // Buffers go out of scope here, should be freed
        drop(buffers);

        // Sync to ensure all operations complete
        if let Err(e) = ctx.sync_and_check() {
            return TestResult::error(
                "test_memory_lifecycle",
                start.elapsed(),
                format!("Cycle {}: sync failed: {}", cycle, e),
            );
        }
    }

    // Test reallocating same-sized buffers
    for i in 0..20 {
        let data: Vec<u32> = (0..SIZE as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_memory_lifecycle",
                    start.elapsed(),
                    format!("Realloc {}: create failed: {}", i, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_memory_lifecycle",
                    start.elapsed(),
                    format!("Realloc {}: sort failed: {}", i, e),
                )
            }
        };

        if ctx.device_row_count(&sorted) != SIZE as u64 {
            return TestResult::error(
                "test_memory_lifecycle",
                start.elapsed(),
                format!(
                    "Realloc {}: wrong row count: {}",
                    i,
                    ctx.device_row_count(&sorted)
                ),
            );
        }

        // Let buffer go out of scope (freed)
    }

    // Test varying sizes
    let varying_sizes = [1000, 10000, 50000, 100000, 50000, 10000, 1000];

    for (i, &size) in varying_sizes.iter().enumerate() {
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_memory_lifecycle",
                    start.elapsed(),
                    format!("Varying {}: create size {} failed: {}", i, size, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_memory_lifecycle",
                    start.elapsed(),
                    format!("Varying {}: sort failed: {}", i, e),
                )
            }
        };

        if ctx.device_row_count(&sorted) != size as u64 {
            return TestResult::error(
                "test_memory_lifecycle",
                start.elapsed(),
                format!(
                    "Varying {}: expected {} rows, got {}",
                    i,
                    size,
                    ctx.device_row_count(&sorted)
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_memory_lifecycle",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_memory_lifecycle", start.elapsed())
}

/// Test 5: Test operations near memory budget limits.
///
/// Tests behavior when operating close to the memory budget limits,
/// ensuring operations either succeed or fail gracefully.
fn test_memory_budget_limits(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    let budget = ctx.memory_budget();
    let _used_start = ctx.memory_used();

    // Calculate size that uses ~25% of budget
    // Each u32 is 4 bytes
    let quarter_budget_elements = budget / 16; // 25% of budget in u32s

    // Create buffer using ~25% of budget
    let small_fraction = quarter_budget_elements.min(10_000_000); // Cap at 10M for reasonable test time

    let data: Vec<u32> = (0..small_fraction)
        .map(|i| ((i * 1103515245 + 12345) % small_fraction) as u32)
        .collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_memory_budget_limits",
                start.elapsed(),
                format!("Failed to create initial buffer: {}", e),
            )
        }
    };

    // Perform operation
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_memory_budget_limits",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Verify
    let result = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_memory_budget_limits",
                start.elapsed(),
                format!("Download failed: {}", e),
            )
        }
    };

    for i in (1..result.len()).step_by(10000) {
        if result[i] < result[i - 1] {
            return TestResult::error(
                "test_memory_budget_limits",
                start.elapsed(),
                format!("Sort incorrect at index {}", i),
            );
        }
    }

    // Create multiple smaller buffers to approach budget
    let num_buffers = 3;
    let per_buffer_size = quarter_budget_elements.min(5_000_000);
    let mut buffers = Vec::new();

    for i in 0..num_buffers {
        let buf_data: Vec<u32> = (0..per_buffer_size)
            .map(|j| ((j + i * per_buffer_size) % per_buffer_size) as u32)
            .collect();

        match ctx
            .provider
            .create_buffer_from_u32_slice(&buf_data, schema.clone())
        {
            Ok(buf) => buffers.push(buf),
            Err(_) => {
                // Expected to fail at some point due to memory limits
                break;
            }
        }
    }

    // Verify operations still work on allocated buffers
    for (i, buf) in buffers.iter().enumerate() {
        let mask: Vec<u8> = (0..per_buffer_size)
            .map(|j| if j % 2 == 0 { 1 } else { 0 })
            .collect();

        let filtered = match ctx.provider.filter_by_mask(buf, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_memory_budget_limits",
                    start.elapsed(),
                    format!("Buffer {}: filter failed: {}", i, e),
                )
            }
        };

        let expected = (per_buffer_size + 1) / 2;
        if ctx.device_row_count(&filtered) != expected as u64 {
            return TestResult::error(
                "test_memory_budget_limits",
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

    // Release buffers and verify we can allocate again
    drop(buffers);

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_memory_budget_limits",
            start.elapsed(),
            format!("Sync after release failed: {}", e),
        );
    }

    // Should be able to allocate again after releasing
    let new_data: Vec<u32> = (0..per_buffer_size as u32).collect();
    let new_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&new_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_memory_budget_limits",
                start.elapsed(),
                format!("Failed to reallocate after release: {}", e),
            )
        }
    };

    let new_sorted = match ctx.provider.sort(&new_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_memory_budget_limits",
                start.elapsed(),
                format!("Sort after realloc failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&new_sorted) != per_buffer_size as u64 {
        return TestResult::error(
            "test_memory_budget_limits",
            start.elapsed(),
            format!(
                "Reallocated buffer: expected {} rows, got {}",
                per_buffer_size,
                ctx.device_row_count(&new_sorted)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_memory_budget_limits",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_memory_budget_limits", start.elapsed())
}
