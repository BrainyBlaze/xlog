//! Category 6: Shared Memory Edge Cases
//!
//! Tests operations that use shared memory internally, such as sort and groupby.
//! Exercises shared memory bank conflicts, block boundaries, and size limits.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{Schema, ScalarType};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c06_shared_memory");
    let start = Instant::now();

    results.add_result(test_sort_uses_shared_memory(ctx));
    results.add_result(test_sort_bank_conflicts(ctx));
    results.add_result(test_sort_multiple_passes(ctx));
    results.add_result(test_block_boundary_shared_mem(ctx));
    results.add_result(test_shared_mem_size_limits(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Sort various sizes, verify correct (exercises shared memory).
///
/// Sort operations typically use shared memory for local sorting within blocks.
/// This test verifies sort correctness across various sizes that exercise
/// different shared memory usage patterns.
fn test_sort_uses_shared_memory(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Various sizes that exercise different shared memory tile sizes
    let sizes: Vec<usize> = vec![
        256,   // Single block, fits in shared memory
        512,   // Single block, larger shared memory usage
        1024,  // Typical shared memory tile size
        2048,  // Multiple tiles in shared memory
        4096,  // Larger working set
        8192,  // Multiple blocks
        16384, // Many blocks
    ];

    for size in sizes {
        // Create reverse-sorted data (worst case for many sort algorithms)
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_sort_uses_shared_memory",
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
                    "test_sort_uses_shared_memory",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify row count
        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_sort_uses_shared_memory",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }

        // Download and verify sorted order
        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_sort_uses_shared_memory",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify sorted: 0, 1, 2, ..., size-1
        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "test_sort_uses_shared_memory",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted[{}] = {}, expected {}",
                        size, i, val, i
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sort_uses_shared_memory",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sort_uses_shared_memory", start.elapsed())
}

/// Test 2: Sort adversarial data patterns that might cause bank conflicts.
///
/// Shared memory bank conflicts occur when multiple threads access the same
/// memory bank. This test uses data patterns that could trigger such conflicts.
fn test_sort_bank_conflicts(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Size that's a multiple of bank count (32 banks in most GPUs)
    const SIZE: usize = 4096;

    // Pattern 1: All same value (could cause serialization)
    let data_same: Vec<u32> = vec![42; SIZE];

    let buffer_same = match ctx.provider.create_buffer_from_u32_slice(&data_same, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Failed to create same-value buffer: {}", e),
            )
        }
    };

    let sorted_same = match ctx.provider.sort(&buffer_same, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Sort of same-value buffer failed: {}", e),
            )
        }
    };

    let downloaded_same = match ctx.provider.download_column_u32(&sorted_same, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Failed to download same-value result: {}", e),
            )
        }
    };

    if downloaded_same != data_same {
        return TestResult::error(
            "test_sort_bank_conflicts",
            start.elapsed(),
            "Same-value sort produced incorrect result".to_string(),
        );
    }

    // Pattern 2: Values that map to same bank (stride of 32)
    // This pattern could cause bank conflicts during comparison/swap
    let data_stride: Vec<u32> = (0..SIZE).map(|i| ((i % 32) * 1000 + i / 32) as u32).collect();

    let buffer_stride = match ctx.provider.create_buffer_from_u32_slice(&data_stride, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Failed to create stride buffer: {}", e),
            )
        }
    };

    let sorted_stride = match ctx.provider.sort(&buffer_stride, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Sort of stride buffer failed: {}", e),
            )
        }
    };

    let downloaded_stride = match ctx.provider.download_column_u32(&sorted_stride, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Failed to download stride result: {}", e),
            )
        }
    };

    // Verify sorted order
    for i in 1..downloaded_stride.len() {
        if downloaded_stride[i] < downloaded_stride[i - 1] {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!(
                    "Stride sort order incorrect at {}: {} > {}",
                    i, downloaded_stride[i - 1], downloaded_stride[i]
                ),
            );
        }
    }

    // Pattern 3: Power-of-2 values (many duplicate keys with specific bit patterns)
    let data_pow2: Vec<u32> = (0..SIZE).map(|i| 1u32 << (i % 16)).collect();

    let buffer_pow2 = match ctx.provider.create_buffer_from_u32_slice(&data_pow2, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Failed to create pow2 buffer: {}", e),
            )
        }
    };

    let sorted_pow2 = match ctx.provider.sort(&buffer_pow2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Sort of pow2 buffer failed: {}", e),
            )
        }
    };

    let downloaded_pow2 = match ctx.provider.download_column_u32(&sorted_pow2, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!("Failed to download pow2 result: {}", e),
            )
        }
    };

    // Verify sorted order
    for i in 1..downloaded_pow2.len() {
        if downloaded_pow2[i] < downloaded_pow2[i - 1] {
            return TestResult::error(
                "test_sort_bank_conflicts",
                start.elapsed(),
                format!(
                    "Pow2 sort order incorrect at {}: {} > {}",
                    i, downloaded_pow2[i - 1], downloaded_pow2[i]
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sort_bank_conflicts",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sort_bank_conflicts", start.elapsed())
}

/// Test 3: Sort large data requiring multiple passes.
///
/// When data exceeds what can be sorted in a single pass, the algorithm
/// must merge sorted chunks. This tests that multi-pass sorting works.
fn test_sort_multiple_passes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Large enough to require multiple passes
    const SIZE: usize = 1_000_000;

    // Create data with keys in random-ish order.
    //
    // Use parameters where (a * i + c) mod SIZE is a permutation of 0..SIZE-1
    // (i.e., gcd(a, SIZE) == 1).
    let a = 1_103_515_247u64;
    let c = 12_345u64;
    let m = SIZE as u64;
    let keys: Vec<u32> = (0..SIZE)
        .map(|i| (((i as u64) * a + c) % m) as u32)
        .collect();
    let vals: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_columns(&[&keys, &vals], schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_sort_multiple_passes",
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
                "test_sort_multiple_passes",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Verify row count
    if sorted.num_rows != SIZE as u64 {
        return TestResult::error(
            "test_sort_multiple_passes",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted.num_rows, SIZE
            ),
        );
    }

    // Download and verify
    let sorted_keys = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_sort_multiple_passes",
                start.elapsed(),
                format!("Failed to download sorted keys: {}", e),
            )
        }
    };

    // Verify keys are sorted
    for i in 1..sorted_keys.len() {
        if sorted_keys[i] < sorted_keys[i - 1] {
            return TestResult::error(
                "test_sort_multiple_passes",
                start.elapsed(),
                format!(
                    "Sort order incorrect at {}: {} > {}",
                    i, sorted_keys[i - 1], sorted_keys[i]
                ),
            );
        }
    }

    // Since the input keys are a permutation of 0..SIZE-1, a correct sort must produce
    // exactly 0, 1, 2, ..., SIZE-1.
    for (i, &k) in sorted_keys.iter().enumerate() {
        if k != i as u32 {
            return TestResult::error(
                "test_sort_multiple_passes",
                start.elapsed(),
                format!("Key at {} is {}, expected {}", i, k, i),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sort_multiple_passes",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sort_multiple_passes", start.elapsed())
}

/// Test 4: Test sizes at block boundaries for shared memory operations.
///
/// Block boundaries (256, 512, 1024 threads) often have edge cases in
/// shared memory allocation and synchronization.
fn test_block_boundary_shared_mem(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes at and around typical block sizes
    let boundary_sizes: Vec<usize> = vec![
        255, 256, 257,     // Around 256-thread block
        511, 512, 513,     // Around 512-thread block
        1023, 1024, 1025,  // Around 1024-thread block
        2047, 2048, 2049,  // Around 2048-element tile
        4095, 4096, 4097,  // Around 4096-element tile
    ];

    for size in boundary_sizes {
        // Create reverse-sorted data
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_block_boundary_shared_mem",
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
                    "test_block_boundary_shared_mem",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify row count
        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_block_boundary_shared_mem",
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
                    "test_block_boundary_shared_mem",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify sorted: 0, 1, 2, ..., size-1
        for (i, &val) in sorted_data.iter().enumerate() {
            if val != i as u32 {
                return TestResult::error(
                    "test_block_boundary_shared_mem",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted[{}] = {}, expected {}",
                        size, i, val, i
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_block_boundary_shared_mem",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_block_boundary_shared_mem", start.elapsed())
}

/// Test 5: Test sizes that stress shared memory capacity.
///
/// GPUs have limited shared memory per block (48KB-164KB typically).
/// This test uses sizes that approach or exceed typical shared memory limits.
fn test_shared_mem_size_limits(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Sizes that stress shared memory
    // 8192 u32s = 32KB (fits in most shared memory)
    // 12288 u32s = 48KB (at limit of standard shared memory)
    // 16384 u32s = 64KB (exceeds standard, requires extended shared memory)
    let stress_sizes: Vec<usize> = vec![
        8192,   // 32KB with 2 columns = 64KB total row data
        12288,  // 48KB with 2 columns = 96KB
        16384,  // 64KB with 2 columns = 128KB
        32768,  // 128KB with 2 columns = 256KB
        65536,  // Forces multi-block coordination
    ];

    for size in stress_sizes {
        // Create data with keys in reverse order
        let keys: Vec<u32> = (0..size as u32).rev().collect();
        let vals: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx.provider.create_buffer_from_u32_columns(&[&keys, &vals], schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_shared_mem_size_limits",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Sort by key
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_shared_mem_size_limits",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify row count
        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_shared_mem_size_limits",
                start.elapsed(),
                format!(
                    "Size {}: sort returned {} rows, expected {}",
                    size, sorted.num_rows, size
                ),
            );
        }

        // Download and verify keys
        let sorted_keys = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_shared_mem_size_limits",
                    start.elapsed(),
                    format!("Size {}: failed to download keys: {}", size, e),
                )
            }
        };

        let sorted_vals = match ctx.provider.download_column_u32(&sorted, 1) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_shared_mem_size_limits",
                    start.elapsed(),
                    format!("Size {}: failed to download vals: {}", size, e),
                )
            }
        };

        // Verify keys are sorted: 0, 1, 2, ..., size-1
        for (i, &key) in sorted_keys.iter().enumerate() {
            if key != i as u32 {
                return TestResult::error(
                    "test_shared_mem_size_limits",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted_keys[{}] = {}, expected {}",
                        size, i, key, i
                    ),
                );
            }
        }

        // Verify key-value pairing preserved
        // Original: key[i] = size-1-i, val[i] = i
        // Sorted: sorted_keys[j] = j, so original index was size-1-j
        // Therefore sorted_vals[j] = size-1-j
        for i in 0..size {
            let expected_val = (size - 1 - i) as u32;
            if sorted_vals[i] != expected_val {
                return TestResult::error(
                    "test_shared_mem_size_limits",
                    start.elapsed(),
                    format!(
                        "Size {}: sorted_vals[{}] = {}, expected {}",
                        size, i, sorted_vals[i], expected_val
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_shared_mem_size_limits",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_shared_mem_size_limits", start.elapsed())
}
