//! Category 9: Warp-Level Programming
//!
//! Tests warp-level behavior (32 thread groups) including full and partial warps,
//! warp divergence patterns, uniform patterns, and multi-warp coordination.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c09_warp_level");
    let start = Instant::now();

    results.add_result(test_warp_size_operations(ctx));
    results.add_result(test_partial_warp_correctness(ctx));
    results.add_result(test_warp_divergence_patterns(ctx));
    results.add_result(test_warp_uniform_patterns(ctx));
    results.add_result(test_multi_warp_coordination(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Test sizes that exercise full and partial warps.
///
/// CUDA warps are 32 threads. This test verifies correct behavior with sizes
/// that result in exactly N warps (32, 64), N-1 threads (31, 63), and N+1 threads (33, 65).
fn test_warp_size_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes that exercise warp boundaries: partial warps, full warps, just over
    let sizes: Vec<usize> = vec![31, 32, 33, 63, 64, 65];

    for size in sizes {
        // Create reverse-sorted data to force data movement
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_warp_size_operations",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Sort exercises warp-level operations
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_warp_size_operations",
                    start.elapsed(),
                    format!("Size {}: sort failed: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_warp_size_operations",
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
                    "test_warp_size_operations",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify ascending order 0, 1, 2, ..., size-1
        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "test_warp_size_operations",
                    start.elapsed(),
                    format!("Size {}: sorted[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }

        // Also test filter at warp boundaries
        let mask: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let expected_count = (size + 1) / 2;

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_warp_size_operations",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if filtered.num_rows != expected_count as u64 {
            return TestResult::error(
                "test_warp_size_operations",
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
            "test_warp_size_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_warp_size_operations", start.elapsed())
}

/// Test 2: Test sizes 1-31 that result in partial warps.
///
/// Partial warps are common edge cases that can cause bugs if not handled correctly.
/// Tests all sizes from 1 to 31 to ensure partial warp handling is correct.
fn test_partial_warp_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // All partial warp sizes (1 to 31)
    for size in 1..32 {
        // Create reverse-sorted data
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_partial_warp_correctness",
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
                    "test_partial_warp_correctness",
                    start.elapsed(),
                    format!("Size {}: sort failed: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_partial_warp_correctness",
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
                    "test_partial_warp_correctness",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify sorted order
        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "test_partial_warp_correctness",
                    start.elapsed(),
                    format!("Size {}: sorted[{}] = {}, expected {}", size, i, val, i),
                );
            }
        }

        // Test dedup with partial warps - all unique
        let deduped = match ctx.provider.dedup(&buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_partial_warp_correctness",
                    start.elapsed(),
                    format!("Size {}: dedup failed: {}", size, e),
                )
            }
        };

        // All elements should be unique
        if deduped.num_rows != size as u64 {
            return TestResult::error(
                "test_partial_warp_correctness",
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
            "test_partial_warp_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_partial_warp_correctness", start.elapsed())
}

/// Test 3: Test adversarial data causing warp divergence in comparisons.
///
/// Warp divergence occurs when threads in a warp take different branches.
/// This test uses data patterns that maximize divergence during comparisons.
fn test_warp_divergence_patterns(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Pattern 1: Alternating high/low values - maximizes divergence in comparisons
    const SIZE: usize = 1024;
    let alternating: Vec<u32> = (0..SIZE)
        .map(|i| {
            if i % 2 == 0 {
                i as u32
            } else {
                (SIZE - i) as u32
            }
        })
        .collect();

    let buffer1 = match ctx
        .provider
        .create_buffer_from_u32_slice(&alternating, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Failed to create alternating buffer: {}", e),
            )
        }
    };

    let sorted1 = match ctx.provider.sort(&buffer1, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Sort of alternating pattern failed: {}", e),
            )
        }
    };

    // Verify sorted
    let sorted_data1 = match ctx.provider.download_column_u32(&sorted1, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Failed to download alternating sorted: {}", e),
            )
        }
    };

    for i in 1..SIZE {
        if sorted_data1[i] < sorted_data1[i - 1] {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!(
                    "Alternating: not sorted at {}: {} > {}",
                    i,
                    sorted_data1[i - 1],
                    sorted_data1[i]
                ),
            );
        }
    }

    // Pattern 2: Random-ish pattern within each warp - causes warp divergence
    let warp_chaos: Vec<u32> = (0..SIZE)
        .map(|i| {
            let warp_id = i / 32;
            let lane = i % 32;
            // Each thread in warp gets a different pseudo-random value
            ((lane * 7 + warp_id * 13 + i * 17) % 1000) as u32
        })
        .collect();

    let buffer2 = match ctx
        .provider
        .create_buffer_from_u32_slice(&warp_chaos, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Failed to create warp chaos buffer: {}", e),
            )
        }
    };

    let sorted2 = match ctx.provider.sort(&buffer2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Sort of warp chaos pattern failed: {}", e),
            )
        }
    };

    let sorted_data2 = match ctx.provider.download_column_u32(&sorted2, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Failed to download warp chaos sorted: {}", e),
            )
        }
    };

    // Verify sorted
    for i in 1..SIZE {
        if sorted_data2[i] < sorted_data2[i - 1] {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!(
                    "Warp chaos: not sorted at {}: {} > {}",
                    i,
                    sorted_data2[i - 1],
                    sorted_data2[i]
                ),
            );
        }
    }

    // Pattern 3: Saw-tooth pattern - rises and falls causing comparison divergence
    let sawtooth: Vec<u32> = (0..SIZE)
        .map(|i| {
            let cycle = i % 64;
            if cycle < 32 {
                cycle as u32
            } else {
                (63 - cycle) as u32
            }
        })
        .collect();

    let buffer3 = match ctx
        .provider
        .create_buffer_from_u32_slice(&sawtooth, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Failed to create sawtooth buffer: {}", e),
            )
        }
    };

    let sorted3 = match ctx.provider.sort(&buffer3, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Sort of sawtooth pattern failed: {}", e),
            )
        }
    };

    let sorted_data3 = match ctx.provider.download_column_u32(&sorted3, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!("Failed to download sawtooth sorted: {}", e),
            )
        }
    };

    // Verify sorted
    for i in 1..SIZE {
        if sorted_data3[i] < sorted_data3[i - 1] {
            return TestResult::error(
                "test_warp_divergence_patterns",
                start.elapsed(),
                format!(
                    "Sawtooth: not sorted at {}: {} > {}",
                    i,
                    sorted_data3[i - 1],
                    sorted_data3[i]
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_warp_divergence_patterns",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_warp_divergence_patterns", start.elapsed())
}

/// Test 4: Test uniform data within warps (32 same values repeating).
///
/// When all threads in a warp have the same value, warp-level optimizations
/// may apply. This test verifies correct handling of uniform warp data.
fn test_warp_uniform_patterns(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Pattern 1: Each warp has the same value - all 32 threads uniform
    const SIZE: usize = 1024; // 32 warps
    let warp_uniform: Vec<u32> = (0..SIZE).map(|i| (i / 32) as u32).collect();

    let buffer1 = match ctx
        .provider
        .create_buffer_from_u32_slice(&warp_uniform, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_warp_uniform_patterns",
                start.elapsed(),
                format!("Failed to create warp uniform buffer: {}", e),
            )
        }
    };

    // Sort - should produce grouped output
    let sorted1 = match ctx.provider.sort(&buffer1, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_warp_uniform_patterns",
                start.elapsed(),
                format!("Sort of warp uniform failed: {}", e),
            )
        }
    };

    let sorted_data1 = match ctx.provider.download_column_u32(&sorted1, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_warp_uniform_patterns",
                start.elapsed(),
                format!("Failed to download warp uniform sorted: {}", e),
            )
        }
    };

    // Verify sorted and count values
    let mut value_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &val in &sorted_data1 {
        *value_counts.entry(val).or_insert(0) += 1;
    }

    // Each warp value (0-31) should appear exactly 32 times
    for warp_id in 0..(SIZE / 32) {
        let count = *value_counts.get(&(warp_id as u32)).unwrap_or(&0);
        if count != 32 {
            return TestResult::error(
                "test_warp_uniform_patterns",
                start.elapsed(),
                format!("Warp {}: count = {}, expected 32", warp_id, count),
            );
        }
    }

    // Verify sorted order
    for i in 1..SIZE {
        if sorted_data1[i] < sorted_data1[i - 1] {
            return TestResult::error(
                "test_warp_uniform_patterns",
                start.elapsed(),
                format!(
                    "Warp uniform: not sorted at {}: {} > {}",
                    i,
                    sorted_data1[i - 1],
                    sorted_data1[i]
                ),
            );
        }
    }

    // Pattern 2: Dedup on uniform warps - should keep one per warp
    let deduped = match ctx.provider.dedup(&buffer1, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_warp_uniform_patterns",
                start.elapsed(),
                format!("Dedup of warp uniform failed: {}", e),
            )
        }
    };

    // Should have exactly 32 unique values (one per warp)
    let expected_unique = SIZE / 32;
    if deduped.num_rows != expected_unique as u64 {
        return TestResult::error(
            "test_warp_uniform_patterns",
            start.elapsed(),
            format!(
                "Dedup returned {} unique, expected {}",
                deduped.num_rows, expected_unique
            ),
        );
    }

    // Pattern 3: All same value - extreme uniform case
    let all_same: Vec<u32> = vec![42; SIZE];

    let buffer2 = match ctx
        .provider
        .create_buffer_from_u32_slice(&all_same, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_warp_uniform_patterns",
                start.elapsed(),
                format!("Failed to create all-same buffer: {}", e),
            )
        }
    };

    let deduped2 = match ctx.provider.dedup(&buffer2, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_warp_uniform_patterns",
                start.elapsed(),
                format!("Dedup of all-same failed: {}", e),
            )
        }
    };

    if deduped2.num_rows != 1 {
        return TestResult::error(
            "test_warp_uniform_patterns",
            start.elapsed(),
            format!(
                "Dedup of all-same returned {} rows, expected 1",
                deduped2.num_rows
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_warp_uniform_patterns",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_warp_uniform_patterns", start.elapsed())
}

/// Test 5: Test sizes requiring multiple warps to coordinate (1024, 2048, 4096).
///
/// Large operations require coordination across many warps. This test verifies
/// that multi-warp coordination (through shared memory or global sync) works correctly.
fn test_multi_warp_coordination(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Large sizes that require many warps: 32, 64, 128 warps
    let sizes: Vec<usize> = vec![1024, 2048, 4096];

    for size in sizes {
        // Create reverse-sorted keys with associated values
        let keys: Vec<u32> = (0..size as u32).rev().collect();
        let vals: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_multi_warp_coordination",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Sort by key - requires coordination across all warps
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_multi_warp_coordination",
                    start.elapsed(),
                    format!("Size {}: sort failed: {}", size, e),
                )
            }
        };

        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_multi_warp_coordination",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }

        let sorted_keys = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_multi_warp_coordination",
                    start.elapsed(),
                    format!("Size {}: failed to download keys: {}", size, e),
                )
            }
        };

        let sorted_vals = match ctx.provider.download_column_u32(&sorted, 1) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_multi_warp_coordination",
                    start.elapsed(),
                    format!("Size {}: failed to download vals: {}", size, e),
                )
            }
        };

        // Verify keys are sorted ascending 0, 1, 2, ..., size-1
        for (i, &key) in sorted_keys.iter().enumerate() {
            if key != i as u32 {
                return TestResult::error(
                    "test_multi_warp_coordination",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted_keys[{}] = {}, expected {}",
                        size, i, key, i
                    ),
                );
            }
        }

        // Verify key-value pairs are maintained
        // Original: keys[i] = size-1-i, vals[i] = i
        // After sort by key: sorted_keys[j] = j, so original row was (size-1-j, size-1-j)
        // Therefore sorted_vals[j] = size-1-j
        for i in 0..size {
            let expected_val = (size - 1 - i) as u32;
            if sorted_vals[i] != expected_val {
                return TestResult::error(
                    "test_multi_warp_coordination",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted_vals[{}] = {}, expected {}",
                        size, i, sorted_vals[i], expected_val
                    ),
                );
            }
        }

        // Test filter with a complex mask pattern across warps
        let mask: Vec<u8> = (0..size)
            .map(|i| {
                // Create a pattern that varies across warp boundaries
                let warp_id = i / 32;
                if warp_id % 2 == 0 {
                    // Even warps: keep every 3rd element
                    if i % 3 == 0 {
                        1
                    } else {
                        0
                    }
                } else {
                    // Odd warps: keep first half of warp
                    if (i % 32) < 16 {
                        1
                    } else {
                        0
                    }
                }
            })
            .collect();

        let expected_count: usize = mask.iter().map(|&m| m as usize).sum();

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_multi_warp_coordination",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        if filtered.num_rows != expected_count as u64 {
            return TestResult::error(
                "test_multi_warp_coordination",
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
            "test_multi_warp_coordination",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_multi_warp_coordination", start.elapsed())
}
