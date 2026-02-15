//! Category 8: Synchronization and Memory Ordering
//!
//! Tests synchronization primitives and memory ordering including atomics,
//! block synchronization barriers, and concurrent operations.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c08_synchronization");
    let start = Instant::now();

    results.add_result(test_hash_join_atomics(ctx));
    results.add_result(test_filter_scan_sync(ctx));
    results.add_result(test_sort_barrier_correctness(ctx));
    results.add_result(test_dedup_atomic_marking(ctx));
    results.add_result(test_concurrent_operations(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Hash join uses atomics for hash table - verify correctness.
///
/// Hash join operations use atomic operations for building the hash table.
/// This test verifies that concurrent atomic operations produce correct results.
fn test_hash_join_atomics(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    // Create left table with sequential keys
    const LEFT_SIZE: usize = 10000;
    let left_keys: Vec<u32> = (0..LEFT_SIZE as u32).collect();
    let left_vals: Vec<u32> = (0..LEFT_SIZE as u32).map(|i| i * 10).collect();

    // Create right table with overlapping keys (every 2nd key)
    const RIGHT_SIZE: usize = 5000;
    let right_keys: Vec<u32> = (0..RIGHT_SIZE as u32).map(|i| i * 2).collect();
    let right_vals: Vec<u32> = (0..RIGHT_SIZE as u32).map(|i| i * 100).collect();

    let left_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_hash_join_atomics",
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
                "test_hash_join_atomics",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
            )
        }
    };

    // Perform hash join
    let joined = match ctx
        .provider
        .hash_join(&left_buffer, &right_buffer, &[0], &[0])
    {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_hash_join_atomics",
                start.elapsed(),
                format!("Hash join failed: {}", e),
            )
        }
    };

    // Expected: matches for keys 0, 2, 4, ..., 9998 = 5000 matches
    let expected_matches = RIGHT_SIZE;
    if ctx.device_row_count(&joined) != expected_matches as u64 {
        return TestResult::error(
            "test_hash_join_atomics",
            start.elapsed(),
            format!(
                "Join returned {} rows, expected {}",
                ctx.device_row_count(&joined),
                expected_matches
            ),
        );
    }

    // Download and verify join results
    let joined_keys = match ctx.provider.download_column_u32(&joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_hash_join_atomics",
                start.elapsed(),
                format!("Failed to download joined keys: {}", e),
            )
        }
    };

    let joined_lvals = match ctx.provider.download_column_u32(&joined, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_hash_join_atomics",
                start.elapsed(),
                format!("Failed to download joined lvals: {}", e),
            )
        }
    };

    let joined_rvals = match ctx.provider.download_column_u32(&joined, 2) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_hash_join_atomics",
                start.elapsed(),
                format!("Failed to download joined rvals: {}", e),
            )
        }
    };

    // Verify that all joined rows have matching keys and correct values
    for i in 0..ctx.device_row_count(&joined) as usize {
        let key = joined_keys[i];
        let lval = joined_lvals[i];
        let rval = joined_rvals[i];

        // Key should be even (matching pattern)
        if key % 2 != 0 {
            return TestResult::error(
                "test_hash_join_atomics",
                start.elapsed(),
                format!("Row {}: key {} should be even", i, key),
            );
        }

        // lval should be key * 10
        let expected_lval = key * 10;
        if lval != expected_lval {
            return TestResult::error(
                "test_hash_join_atomics",
                start.elapsed(),
                format!(
                    "Row {}: lval {} doesn't match expected {} for key {}",
                    i, lval, expected_lval, key
                ),
            );
        }

        // rval should be (key/2) * 100
        let expected_rval = (key / 2) * 100;
        if rval != expected_rval {
            return TestResult::error(
                "test_hash_join_atomics",
                start.elapsed(),
                format!(
                    "Row {}: rval {} doesn't match expected {} for key {}",
                    i, rval, expected_rval, key
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_hash_join_atomics",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_hash_join_atomics", start.elapsed())
}

/// Test 2: Filter uses scan which requires block synchronization.
///
/// Filter operations use prefix scan to compute output positions. Prefix scan
/// requires __syncthreads() for correctness within blocks.
fn test_filter_scan_sync(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test various sizes that exercise different scan patterns
    let test_cases: Vec<(usize, Box<dyn Fn(usize) -> bool>)> = vec![
        // (size, predicate)
        (1000, Box::new(|i| i % 2 == 0)),          // Keep even
        (10000, Box::new(|i| i % 3 == 0)),         // Keep multiples of 3
        (50000, Box::new(|i| i < 25000)),          // Keep first half
        (100000, Box::new(|i| i % 7 == 0)),        // Keep multiples of 7
        (65536, Box::new(|i| (i / 256) % 2 == 0)), // Keep alternating chunks
    ];

    for (size, predicate) in test_cases {
        // Create sequential data
        let data: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_filter_scan_sync",
                    start.elapsed(),
                    format!("Size {}: failed to create buffer: {}", size, e),
                )
            }
        };

        // Create mask based on predicate
        let mask: Vec<u8> = (0..size)
            .map(|i| if predicate(i) { 1 } else { 0 })
            .collect();
        let expected_count: usize = mask.iter().map(|&m| m as usize).sum();

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_filter_scan_sync",
                    start.elapsed(),
                    format!("Size {}: filter failed: {}", size, e),
                )
            }
        };

        // Verify count
        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_filter_scan_sync",
                start.elapsed(),
                format!(
                    "Size {}: filter returned {} rows, expected {}",
                    size,
                    ctx.device_row_count(&filtered),
                    expected_count
                ),
            );
        }

        // Download and verify values
        let filtered_data = match ctx.provider.download_column_u32(&filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_filter_scan_sync",
                    start.elapsed(),
                    format!("Size {}: failed to download: {}", size, e),
                )
            }
        };

        // Verify each filtered value matches predicate
        let mut expected_idx = 0;
        for i in 0..size {
            if predicate(i) {
                if expected_idx >= filtered_data.len() {
                    return TestResult::error(
                        "test_filter_scan_sync",
                        start.elapsed(),
                        format!(
                            "Size {}: filtered data too short at expected index {}",
                            size, expected_idx
                        ),
                    );
                }
                if filtered_data[expected_idx] != i as u32 {
                    return TestResult::error(
                        "test_filter_scan_sync",
                        start.elapsed(),
                        format!(
                            "Size {}: filtered[{}] = {}, expected {}",
                            size, expected_idx, filtered_data[expected_idx], i
                        ),
                    );
                }
                expected_idx += 1;
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_filter_scan_sync",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_filter_scan_sync", start.elapsed())
}

/// Test 3: Sort uses __syncthreads() - verify correct ordering.
///
/// Sort algorithms like radix sort use block synchronization barriers
/// (__syncthreads) for correctness. This test verifies that synchronization
/// works correctly across various data patterns.
fn test_sort_barrier_correctness(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Test patterns that stress synchronization
    let test_patterns: Vec<(&str, Vec<u32>)> = vec![
        ("reverse", (0..10000u32).rev().collect()),
        (
            "alternating",
            (0..10000u32)
                .map(|i| if i % 2 == 0 { i } else { 10000 - i })
                .collect(),
        ),
        ("sawtooth", (0..10000u32).map(|i| i % 100).collect()),
        (
            "random_lcg",
            (0..10000usize)
                .map(|i| ((i * 1103515245 + 12345) % 10000) as u32)
                .collect(),
        ),
        (
            "blocks_reversed",
            (0..10000u32)
                .map(|i| {
                    let block = i / 256;
                    let offset = i % 256;
                    block * 256 + (255 - offset)
                })
                .collect(),
        ),
    ];

    for (name, keys) in test_patterns {
        let size = keys.len();
        let vals: Vec<u32> = (0..size as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_sort_barrier_correctness",
                    start.elapsed(),
                    format!("Pattern {}: failed to create buffer: {}", name, e),
                )
            }
        };

        // Sort by key
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_sort_barrier_correctness",
                    start.elapsed(),
                    format!("Pattern {}: sort failed: {}", name, e),
                )
            }
        };

        // Verify row count
        if ctx.device_row_count(&sorted) != size as u64 {
            return TestResult::error(
                "test_sort_barrier_correctness",
                start.elapsed(),
                format!(
                    "Pattern {}: sort returned {} rows, expected {}",
                    name,
                    ctx.device_row_count(&sorted),
                    size
                ),
            );
        }

        // Download sorted keys
        let sorted_keys = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_sort_barrier_correctness",
                    start.elapsed(),
                    format!("Pattern {}: failed to download sorted keys: {}", name, e),
                )
            }
        };

        // Verify sorted order
        for i in 1..size {
            if sorted_keys[i] < sorted_keys[i - 1] {
                return TestResult::error(
                    "test_sort_barrier_correctness",
                    start.elapsed(),
                    format!(
                        "Pattern {}: sort order incorrect at {}: {} > {}",
                        name,
                        i,
                        sorted_keys[i - 1],
                        sorted_keys[i]
                    ),
                );
            }
        }

        // Verify all original keys are present (same multiset)
        let mut original_sorted = keys.clone();
        original_sorted.sort();
        if sorted_keys != original_sorted {
            return TestResult::error(
                "test_sort_barrier_correctness",
                start.elapsed(),
                format!("Pattern {}: keys not preserved through sort", name),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sort_barrier_correctness",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sort_barrier_correctness", start.elapsed())
}

/// Test 4: Dedup uses atomic operations - verify correctness.
///
/// Dedup operations use atomic operations to mark duplicates. This test
/// verifies that concurrent atomic marking produces correct deduplication.
fn test_dedup_atomic_marking(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Test various duplicate patterns
    let test_cases: Vec<(&str, Vec<u32>, Vec<u32>)> = vec![
        // (name, keys, vals)
        ("all_same", vec![42; 10000], (0..10000u32).collect()),
        (
            "pairs",
            (0..5000u32).flat_map(|i| vec![i, i]).collect(),
            (0..10000u32).collect(),
        ),
        (
            "triples",
            (0..3333u32)
                .flat_map(|i| vec![i, i, i])
                .take(9999)
                .collect(),
            (0..9999u32).collect(),
        ),
        ("no_dups", (0..10000u32).collect(), (0..10000u32).collect()),
        (
            "many_dups",
            (0..10000u32).map(|i| i % 100).collect(),
            (0..10000u32).collect(),
        ),
    ];

    for (name, keys, vals) in test_cases {
        let buffer = match ctx
            .provider
            .create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_atomic_marking",
                    start.elapsed(),
                    format!("Pattern {}: failed to create buffer: {}", name, e),
                )
            }
        };

        // Dedup by key column
        let deduped = match ctx.provider.dedup(&buffer, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_atomic_marking",
                    start.elapsed(),
                    format!("Pattern {}: dedup failed: {}", name, e),
                )
            }
        };

        // Calculate expected unique count
        let mut unique_keys: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &k in &keys {
            unique_keys.insert(k);
        }
        let expected_unique = unique_keys.len();

        if ctx.device_row_count(&deduped) != expected_unique as u64 {
            return TestResult::error(
                "test_dedup_atomic_marking",
                start.elapsed(),
                format!(
                    "Pattern {}: dedup returned {} rows, expected {}",
                    name,
                    ctx.device_row_count(&deduped),
                    expected_unique
                ),
            );
        }

        // Download and verify uniqueness
        let deduped_keys = match ctx.provider.download_column_u32(&deduped, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dedup_atomic_marking",
                    start.elapsed(),
                    format!("Pattern {}: failed to download deduped keys: {}", name, e),
                )
            }
        };

        // Verify all keys are unique
        let mut seen_keys: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &k in &deduped_keys {
            if !seen_keys.insert(k) {
                return TestResult::error(
                    "test_dedup_atomic_marking",
                    start.elapsed(),
                    format!("Pattern {}: duplicate key {} in dedup result", name, k),
                );
            }
        }

        // Verify all original unique keys are present
        if seen_keys != unique_keys {
            return TestResult::error(
                "test_dedup_atomic_marking",
                start.elapsed(),
                format!("Pattern {}: dedup result missing some unique keys", name),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_dedup_atomic_marking",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_dedup_atomic_marking", start.elapsed())
}

/// Test 5: Run multiple independent operations, verify all correct.
///
/// Tests that multiple independent operations can run and complete correctly,
/// verifying that GPU resource management and synchronization work properly.
fn test_concurrent_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Create multiple buffers
    const NUM_BUFFERS: usize = 5;
    const BUFFER_SIZE: usize = 10000;

    let mut buffers = Vec::with_capacity(NUM_BUFFERS);
    let mut original_data = Vec::with_capacity(NUM_BUFFERS);

    for i in 0..NUM_BUFFERS {
        // Each buffer has different data pattern
        let data: Vec<u32> = (0..BUFFER_SIZE)
            .map(|j| ((j * (i + 1) * 7 + i * 13) % BUFFER_SIZE) as u32)
            .collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_operations",
                    start.elapsed(),
                    format!("Failed to create buffer {}: {}", i, e),
                )
            }
        };

        original_data.push(data);
        buffers.push(buffer);
    }

    // Perform different operations on each buffer
    let mut results = Vec::with_capacity(NUM_BUFFERS);

    // Buffer 0: Sort
    let sorted0 = match ctx.provider.sort(&buffers[0], &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_concurrent_operations",
                start.elapsed(),
                format!("Sort of buffer 0 failed: {}", e),
            )
        }
    };
    results.push(("sort", sorted0));

    // Buffer 1: Filter (keep first half)
    let mask1: Vec<u8> = (0..BUFFER_SIZE)
        .map(|i| if i < BUFFER_SIZE / 2 { 1 } else { 0 })
        .collect();
    let filtered1 = match ctx.provider.filter_by_mask(&buffers[1], &mask1) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_concurrent_operations",
                start.elapsed(),
                format!("Filter of buffer 1 failed: {}", e),
            )
        }
    };
    results.push(("filter", filtered1));

    // Buffer 2: Sort
    let sorted2 = match ctx.provider.sort(&buffers[2], &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_concurrent_operations",
                start.elapsed(),
                format!("Sort of buffer 2 failed: {}", e),
            )
        }
    };
    results.push(("sort", sorted2));

    // Buffer 3: Filter (keep even indices)
    let mask3: Vec<u8> = (0..BUFFER_SIZE)
        .map(|i| if i % 2 == 0 { 1 } else { 0 })
        .collect();
    let filtered3 = match ctx.provider.filter_by_mask(&buffers[3], &mask3) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_concurrent_operations",
                start.elapsed(),
                format!("Filter of buffer 3 failed: {}", e),
            )
        }
    };
    results.push(("filter", filtered3));

    // Buffer 4: Sort
    let sorted4 = match ctx.provider.sort(&buffers[4], &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_concurrent_operations",
                start.elapsed(),
                format!("Sort of buffer 4 failed: {}", e),
            )
        }
    };
    results.push(("sort", sorted4));

    // Verify all results
    // Sort results (0, 2, 4)
    for &idx in &[0usize, 2, 4] {
        let (op, ref result) = results[idx];
        assert!(op == "sort");

        if ctx.device_row_count(&result) != BUFFER_SIZE as u64 {
            return TestResult::error(
                "test_concurrent_operations",
                start.elapsed(),
                format!(
                    "Buffer {}: {} returned {} rows, expected {}",
                    idx,
                    op,
                    ctx.device_row_count(&result),
                    BUFFER_SIZE
                ),
            );
        }

        let sorted_data = match ctx.provider.download_column_u32(result, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_operations",
                    start.elapsed(),
                    format!("Buffer {}: failed to download: {}", idx, e),
                )
            }
        };

        // Verify sorted order
        for i in 1..BUFFER_SIZE {
            if sorted_data[i] < sorted_data[i - 1] {
                return TestResult::error(
                    "test_concurrent_operations",
                    start.elapsed(),
                    format!(
                        "Buffer {}: sort order incorrect at {}: {} > {}",
                        idx,
                        i,
                        sorted_data[i - 1],
                        sorted_data[i]
                    ),
                );
            }
        }

        // Verify same values (sorted)
        let mut expected = original_data[idx].clone();
        expected.sort();
        if sorted_data != expected {
            return TestResult::error(
                "test_concurrent_operations",
                start.elapsed(),
                format!("Buffer {}: sorted data doesn't match expected", idx),
            );
        }
    }

    // Filter results (1, 3)
    // Buffer 1: first half
    let (op1, ref result1) = results[1];
    assert!(op1 == "filter");
    if ctx.device_row_count(&result1) != (BUFFER_SIZE / 2) as u64 {
        return TestResult::error(
            "test_concurrent_operations",
            start.elapsed(),
            format!(
                "Buffer 1: filter returned {} rows, expected {}",
                ctx.device_row_count(&result1),
                BUFFER_SIZE / 2
            ),
        );
    }

    // Buffer 3: even indices
    let (op3, ref result3) = results[3];
    assert!(op3 == "filter");
    if ctx.device_row_count(&result3) != ((BUFFER_SIZE + 1) / 2) as u64 {
        return TestResult::error(
            "test_concurrent_operations",
            start.elapsed(),
            format!(
                "Buffer 3: filter returned {} rows, expected {}",
                ctx.device_row_count(&result3),
                (BUFFER_SIZE + 1) / 2
            ),
        );
    }

    // Verify original buffers are unchanged
    for i in 0..NUM_BUFFERS {
        let current_data = match ctx.provider.download_column_u32(&buffers[i], 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_concurrent_operations",
                    start.elapsed(),
                    format!("Failed to verify buffer {}: {}", i, e),
                )
            }
        };

        if current_data != original_data[i] {
            return TestResult::error(
                "test_concurrent_operations",
                start.elapsed(),
                format!("Buffer {} was modified by operations", i),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_concurrent_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_concurrent_operations", start.elapsed())
}
