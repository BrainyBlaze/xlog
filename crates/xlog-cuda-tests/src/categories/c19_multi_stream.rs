//! Category 19: Multi-stream concurrency
//!
//! Tests concurrent stream operations. Note that xlog uses a single stream,
//! so we test sequential patterns that would be parallel in a multi-stream
//! environment. This verifies operation isolation and batching behavior.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub(crate) fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c19_multi_stream");
    let start = Instant::now();

    results.add_result(test_sequential_batch_operations(ctx));
    results.add_result(test_interleaved_operations(ctx));
    results.add_result(test_operation_isolation(ctx));
    results.add_result(test_batch_completion(ctx));
    results.add_result(test_dependency_chain(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Run batches of independent operations sequentially.
///
/// Tests that batches of independent operations can be queued and executed
/// correctly, simulating what would be parallel streams.
fn test_sequential_batch_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const BATCH_SIZE: usize = 10;
    const DATA_SIZE: usize = 10000;

    // Create batch of independent buffers
    let mut buffers = Vec::new();
    let mut original_data = Vec::new();

    for i in 0..BATCH_SIZE {
        let data: Vec<u32> = (0..DATA_SIZE)
            .map(|j| ((j * (i + 1) * 17 + 13) % 100000) as u32)
            .collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_sequential_batch_operations",
                    start.elapsed(),
                    format!("Buffer {}: create failed: {}", i, e),
                )
            }
        };

        buffers.push(buffer);
        original_data.push(data);
    }

    // Execute all sort operations in batch (sequential but independent)
    let mut sorted_buffers = Vec::new();

    for (i, buffer) in buffers.iter().enumerate() {
        let sorted = match ctx.provider.sort(buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_sequential_batch_operations",
                    start.elapsed(),
                    format!("Buffer {}: sort failed: {}", i, e),
                )
            }
        };
        sorted_buffers.push(sorted);
    }

    // Verify all results after batch completes
    for (i, (sorted, original)) in sorted_buffers.iter().zip(original_data.iter()).enumerate() {
        let result = match ctx.provider.download_column::<u32>(sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_sequential_batch_operations",
                    start.elapsed(),
                    format!("Buffer {}: download failed: {}", i, e),
                )
            }
        };

        // Verify sorted order
        for j in 1..result.len() {
            if result[j] < result[j - 1] {
                return TestResult::error(
                    "test_sequential_batch_operations",
                    start.elapsed(),
                    format!("Buffer {}: sort incorrect at index {}", i, j),
                );
            }
        }

        // Verify same elements
        let mut expected = original.clone();
        expected.sort();
        if result != expected {
            return TestResult::error(
                "test_sequential_batch_operations",
                start.elapsed(),
                format!("Buffer {}: result doesn't match expected", i),
            );
        }
    }

    // Test batch of filter operations
    let mut filtered_buffers = Vec::new();
    let masks: Vec<Vec<u8>> = (0..BATCH_SIZE)
        .map(|i| {
            (0..DATA_SIZE)
                .map(|j| if (j + i) % 3 == 0 { 1 } else { 0 })
                .collect()
        })
        .collect();

    for (i, (buffer, mask)) in buffers.iter().zip(masks.iter()).enumerate() {
        let filtered = match ctx.provider.filter_by_mask(buffer, mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_sequential_batch_operations",
                    start.elapsed(),
                    format!("Filter {}: failed: {}", i, e),
                )
            }
        };
        filtered_buffers.push(filtered);
    }

    // Verify filter results
    for (i, (filtered, mask)) in filtered_buffers.iter().zip(masks.iter()).enumerate() {
        let expected_count: usize = mask.iter().map(|&m| m as usize).sum();
        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_sequential_batch_operations",
                start.elapsed(),
                format!(
                    "Filter {}: expected {} rows, got {}",
                    i,
                    expected_count,
                    ctx.device_row_count(&filtered)
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_sequential_batch_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_sequential_batch_operations", start.elapsed())
}

/// Test 2: Interleave sort and filter operations.
///
/// Tests that interleaving different operation types works correctly,
/// simulating concurrent different operations on different streams.
fn test_interleaved_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;
    const ITERATIONS: usize = 20;

    // Create buffers for interleaved operations
    let sort_data: Vec<u32> = (0..SIZE)
        .map(|i| ((i * 1103515245 + 12345) % SIZE) as u32)
        .collect();
    let filter_data: Vec<u32> = (0..SIZE as u32).collect();

    let sort_buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&sort_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_interleaved_operations",
                start.elapsed(),
                format!("Failed to create sort buffer: {}", e),
            )
        }
    };

    let filter_buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&filter_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_interleaved_operations",
                start.elapsed(),
                format!("Failed to create filter buffer: {}", e),
            )
        }
    };

    // Interleave sort and filter operations
    let mut sort_results = Vec::new();
    let mut filter_results = Vec::new();

    for i in 0..ITERATIONS {
        // Sort operation
        let sorted = match ctx.provider.sort(&sort_buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_interleaved_operations",
                    start.elapsed(),
                    format!("Iteration {}: sort failed: {}", i, e),
                )
            }
        };
        sort_results.push(sorted);

        // Filter operation with varying selectivity
        let selectivity = (i + 1) * 5; // 5%, 10%, 15%, ..., 100%
        let mask: Vec<u8> = (0..SIZE)
            .map(|j| if (j * 100 / SIZE) < selectivity { 1 } else { 0 })
            .collect();

        let filtered = match ctx.provider.filter_by_mask(&filter_buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_interleaved_operations",
                    start.elapsed(),
                    format!("Iteration {}: filter failed: {}", i, e),
                )
            }
        };
        filter_results.push((filtered, selectivity));
    }

    // Verify all sort results are correct and identical
    let mut first_sort_result: Option<Vec<u32>> = None;

    for (i, sorted) in sort_results.iter().enumerate() {
        let result = match ctx.provider.download_column::<u32>(sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_interleaved_operations",
                    start.elapsed(),
                    format!("Sort {}: download failed: {}", i, e),
                )
            }
        };

        for j in 1..result.len() {
            if result[j] < result[j - 1] {
                return TestResult::error(
                    "test_interleaved_operations",
                    start.elapsed(),
                    format!("Sort {}: incorrect at index {}", i, j),
                );
            }
        }

        match &first_sort_result {
            Some(first) => {
                if result != *first {
                    return TestResult::error(
                        "test_interleaved_operations",
                        start.elapsed(),
                        format!("Sort {}: differs from first sort", i),
                    );
                }
            }
            None => {
                first_sort_result = Some(result);
            }
        }
    }

    // Verify filter results
    for (i, (filtered, selectivity)) in filter_results.iter().enumerate() {
        // Expected count is approximately selectivity% of SIZE
        let expected_min = (SIZE * selectivity / 100).saturating_sub(SIZE / 20);
        let expected_max = (SIZE * selectivity / 100) + SIZE / 20 + 1;

        let count = ctx.device_row_count(&filtered) as usize;
        if count < expected_min || count > expected_max {
            return TestResult::error(
                "test_interleaved_operations",
                start.elapsed(),
                format!(
                    "Filter {} ({}%): got {} rows, expected ~{}",
                    i,
                    selectivity,
                    count,
                    SIZE * selectivity / 100
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_interleaved_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_interleaved_operations", start.elapsed())
}

/// Test 3: Verify operations don't interfere with each other.
///
/// Tests that concurrent-style operations maintain data isolation and
/// don't corrupt each other's results.
fn test_operation_isolation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;
    const NUM_BUFFERS: usize = 5;

    // Create distinct buffers with easily identifiable patterns
    let mut buffers = Vec::new();
    let mut expected_sums = Vec::new();

    for i in 0..NUM_BUFFERS {
        // Each buffer has values in range [i*1000, (i+1)*1000)
        let data: Vec<u32> = (0..SIZE)
            .map(|j| i as u32 * 10000 + (j as u32 % 10000))
            .collect();

        let sum: u64 = data.iter().map(|&x| x as u64).sum();
        expected_sums.push(sum);

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_operation_isolation",
                    start.elapsed(),
                    format!("Buffer {}: create failed: {}", i, e),
                )
            }
        };

        buffers.push(buffer);
    }

    // Perform operations on all buffers
    let mut sorted_buffers = Vec::new();

    for (i, buffer) in buffers.iter().enumerate() {
        let sorted = match ctx.provider.sort(buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_operation_isolation",
                    start.elapsed(),
                    format!("Buffer {}: sort failed: {}", i, e),
                )
            }
        };
        sorted_buffers.push(sorted);
    }

    // Verify each buffer maintains its unique identity (no cross-contamination)
    for (i, sorted) in sorted_buffers.iter().enumerate() {
        let result = match ctx.provider.download_column::<u32>(sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_operation_isolation",
                    start.elapsed(),
                    format!("Buffer {}: download failed: {}", i, e),
                )
            }
        };

        // Verify size unchanged
        if result.len() != SIZE {
            return TestResult::error(
                "test_operation_isolation",
                start.elapsed(),
                format!(
                    "Buffer {}: size changed from {} to {}",
                    i,
                    SIZE,
                    result.len()
                ),
            );
        }

        // Verify sum unchanged (proves same elements)
        let actual_sum: u64 = result.iter().map(|&x| x as u64).sum();
        if actual_sum != expected_sums[i] {
            return TestResult::error(
                "test_operation_isolation",
                start.elapsed(),
                format!(
                    "Buffer {}: sum changed from {} to {} (data contamination?)",
                    i, expected_sums[i], actual_sum
                ),
            );
        }

        // Verify range (all values should be in expected range)
        let min_expected = i as u32 * 10000;
        let max_expected = min_expected + 10000;

        for (j, &val) in result.iter().enumerate() {
            if val < min_expected || val >= max_expected {
                return TestResult::error(
                    "test_operation_isolation",
                    start.elapsed(),
                    format!(
                        "Buffer {}: value {} at index {} outside range [{}, {})",
                        i, val, j, min_expected, max_expected
                    ),
                );
            }
        }

        // Verify sorted
        for j in 1..result.len() {
            if result[j] < result[j - 1] {
                return TestResult::error(
                    "test_operation_isolation",
                    start.elapsed(),
                    format!("Buffer {}: not sorted at index {}", i, j),
                );
            }
        }
    }

    // Test isolation with filter operations
    let mut filtered_buffers = Vec::new();

    for (i, buffer) in buffers.iter().enumerate() {
        // Filter to keep first half
        let mask: Vec<u8> = (0..SIZE)
            .map(|j| if j < SIZE / 2 { 1 } else { 0 })
            .collect();

        let filtered = match ctx.provider.filter_by_mask(buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_operation_isolation",
                    start.elapsed(),
                    format!("Filter {}: failed: {}", i, e),
                )
            }
        };
        filtered_buffers.push(filtered);
    }

    // Verify filtered results maintain isolation
    for (i, filtered) in filtered_buffers.iter().enumerate() {
        let result = match ctx.provider.download_column::<u32>(filtered, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_operation_isolation",
                    start.elapsed(),
                    format!("Filtered {}: download failed: {}", i, e),
                )
            }
        };

        if result.len() != SIZE / 2 {
            return TestResult::error(
                "test_operation_isolation",
                start.elapsed(),
                format!(
                    "Filtered {}: expected {} rows, got {}",
                    i,
                    SIZE / 2,
                    result.len()
                ),
            );
        }

        // Verify all values in expected range
        let min_expected = i as u32 * 10000;
        let max_expected = min_expected + 10000;

        for &val in &result {
            if val < min_expected || val >= max_expected {
                return TestResult::error(
                    "test_operation_isolation",
                    start.elapsed(),
                    format!(
                        "Filtered {}: value {} outside range [{}, {})",
                        i, val, min_expected, max_expected
                    ),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_operation_isolation",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_operation_isolation", start.elapsed())
}

/// Test 4: Run batch of operations, sync, verify all completed.
///
/// Tests that sync properly waits for all queued operations to complete
/// before returning.
fn test_batch_completion(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 50000;
    const BATCH_SIZE: usize = 20;

    // Queue many operations
    let mut all_operations = Vec::new();

    for i in 0..BATCH_SIZE {
        let data: Vec<u32> = (0..SIZE)
            .map(|j| ((j * (i + 1) * 31337) % 1000000) as u32)
            .collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_batch_completion",
                    start.elapsed(),
                    format!("Op {}: create failed: {}", i, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_batch_completion",
                    start.elapsed(),
                    format!("Op {}: sort failed: {}", i, e),
                )
            }
        };

        all_operations.push((sorted, data));
    }

    // Sync - should wait for all operations
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_batch_completion",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    // Verify all operations completed correctly
    for (i, (sorted, original)) in all_operations.iter().enumerate() {
        let result = match ctx.provider.download_column::<u32>(sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_batch_completion",
                    start.elapsed(),
                    format!("Op {}: download failed after sync: {}", i, e),
                )
            }
        };

        // Verify sorted
        for j in 1..result.len() {
            if result[j] < result[j - 1] {
                return TestResult::error(
                    "test_batch_completion",
                    start.elapsed(),
                    format!("Op {}: sort incomplete at index {}", i, j),
                );
            }
        }

        // Verify same elements
        let mut expected = original.clone();
        expected.sort();
        if result != expected {
            return TestResult::error(
                "test_batch_completion",
                start.elapsed(),
                format!("Op {}: sorted result doesn't match expected", i),
            );
        }
    }

    // Test batch with mixed operation types
    let mut mixed_ops = Vec::new();

    let base_data: Vec<u32> = (0..SIZE as u32).collect();
    let base_buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&base_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_batch_completion",
                start.elapsed(),
                format!("Base buffer creation failed: {}", e),
            )
        }
    };

    // Queue sorts, filters, dedups
    for i in 0..5 {
        let sort_data: Vec<u32> = (0..SIZE).map(|j| ((j * i) % SIZE) as u32).collect();
        let sort_buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&sort_data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_batch_completion",
                    start.elapsed(),
                    format!("Mixed sort buffer {} failed: {}", i, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&sort_buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_batch_completion",
                    start.elapsed(),
                    format!("Mixed sort {} failed: {}", i, e),
                )
            }
        };
        mixed_ops.push(("sort", ctx.device_row_count(&sorted), SIZE as u64));

        let mask: Vec<u8> = (0..SIZE)
            .map(|j| if j % (i + 2) == 0 { 1 } else { 0 })
            .collect();
        let expected_filtered: usize = mask.iter().map(|&m| m as usize).sum();

        let filtered = match ctx.provider.filter_by_mask(&base_buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_batch_completion",
                    start.elapsed(),
                    format!("Mixed filter {} failed: {}", i, e),
                )
            }
        };
        mixed_ops.push((
            "filter",
            ctx.device_row_count(&filtered),
            expected_filtered as u64,
        ));
    }

    // Sync
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_batch_completion",
            start.elapsed(),
            format!("Mixed batch sync failed: {}", e),
        );
    }

    // Verify all mixed operations completed
    for (i, (op_type, actual, expected)) in mixed_ops.iter().enumerate() {
        if actual != expected {
            return TestResult::error(
                "test_batch_completion",
                start.elapsed(),
                format!(
                    "Mixed op {} ({}): expected {} rows, got {}",
                    i, op_type, expected, actual
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_batch_completion",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_batch_completion", start.elapsed())
}

/// Test 5: Chain operations with dependencies.
///
/// Tests that operations with data dependencies are handled correctly
/// when the output of one becomes the input of the next.
fn test_dependency_chain(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 50000;

    // Create initial data
    let data: Vec<u32> = (0..SIZE)
        .map(|i| ((i * 1103515245 + 12345) % 1000000) as u32)
        .collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Initial buffer creation failed: {}", e),
            )
        }
    };

    // Chain 1: sort -> filter -> sort
    // Step 1: Sort
    let step1 = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 1, step 1 (sort) failed: {}", e),
            )
        }
    };

    // Get intermediate result for filter condition
    let step1_data = match ctx.provider.download_column::<u32>(&step1, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 1, step 1 download failed: {}", e),
            )
        }
    };

    // Step 2: Filter (keep values in middle 50%)
    let min_val = step1_data[SIZE / 4];
    let max_val = step1_data[3 * SIZE / 4];
    let step2_mask: Vec<u8> = step1_data
        .iter()
        .map(|&v| if v >= min_val && v < max_val { 1 } else { 0 })
        .collect();

    let step2 = match ctx.provider.filter_by_mask(&step1, &step2_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 1, step 2 (filter) failed: {}", e),
            )
        }
    };

    // Step 3: Sort again (no-op, should already be sorted)
    let step3 = match ctx.provider.sort(&step2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 1, step 3 (sort) failed: {}", e),
            )
        }
    };

    // Verify chain 1 result
    let chain1_result = match ctx.provider.download_column::<u32>(&step3, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 1 result download failed: {}", e),
            )
        }
    };

    // Verify sorted
    for i in 1..chain1_result.len() {
        if chain1_result[i] < chain1_result[i - 1] {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 1: not sorted at index {}", i),
            );
        }
    }

    // Verify range
    for &val in &chain1_result {
        if val < min_val || val >= max_val {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!(
                    "Chain 1: value {} outside range [{}, {})",
                    val, min_val, max_val
                ),
            );
        }
    }

    // Chain 2: Multiple parallel chains that converge (via join)
    let schema2 = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    // Left chain: create -> sort
    let left_keys: Vec<u32> = (0..10000u32).collect();
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k * 2).collect();

    let left_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&left_keys, &left_vals], schema2.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 2 left buffer failed: {}", e),
            )
        }
    };

    let left_sorted = match ctx.provider.sort(&left_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 2 left sort failed: {}", e),
            )
        }
    };

    // Right chain: create -> filter -> sort
    let right_keys: Vec<u32> = (0..15000u32).map(|i| i * 2 / 3).collect();
    let right_vals: Vec<u32> = right_keys.iter().map(|&k| k * 3).collect();

    let right_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&right_keys, &right_vals], schema2.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 2 right buffer failed: {}", e),
            )
        }
    };

    // Filter right to keep ~half
    let right_mask: Vec<u8> = (0..15000).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let right_filtered = match ctx.provider.filter_by_mask(&right_buffer, &right_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 2 right filter failed: {}", e),
            )
        }
    };

    let right_sorted = match ctx.provider.sort(&right_filtered, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 2 right sort failed: {}", e),
            )
        }
    };

    // Converge: join left and right
    let joined = match ctx
        .provider
        .hash_join(&left_sorted, &right_sorted, &[0], &[0])
    {
        Ok(j) => j,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 2 join failed: {}", e),
            )
        }
    };

    // Verify join produced results
    if ctx.device_row_count(&joined) == 0 {
        return TestResult::error(
            "test_dependency_chain",
            start.elapsed(),
            "Chain 2: join produced no results".to_string(),
        );
    }

    // Download and verify join result
    let join_keys = match ctx.provider.download_column::<u32>(&joined, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 2 join keys download failed: {}", e),
            )
        }
    };

    let join_lvals = match ctx.provider.download_column::<u32>(&joined, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 2 join lvals download failed: {}", e),
            )
        }
    };

    // Verify join consistency: lval should equal key * 2
    for (i, (&key, &lval)) in join_keys.iter().zip(join_lvals.iter()).enumerate() {
        if lval != key * 2 {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!(
                    "Chain 2: join row {} inconsistent: key={}, lval={} (expected {})",
                    i,
                    key,
                    lval,
                    key * 2
                ),
            );
        }
    }

    // Chain 3: Deep dependency chain
    let deep_data: Vec<u32> = (0..10000).map(|i| (i % 1000) as u32).collect();
    let mut current = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&deep_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 3 initial buffer failed: {}", e),
            )
        }
    };

    // Chain of 5 operations
    for i in 0..5 {
        let sorted = match ctx.provider.sort(&current, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_dependency_chain",
                    start.elapsed(),
                    format!("Chain 3 step {} sort failed: {}", i, e),
                )
            }
        };

        // Filter to keep ~80%
        let current_data = match ctx.provider.download_column::<u32>(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_dependency_chain",
                    start.elapsed(),
                    format!("Chain 3 step {} download failed: {}", i, e),
                )
            }
        };

        let mask: Vec<u8> = (0..current_data.len())
            .map(|j| if j % 5 < 4 { 1 } else { 0 })
            .collect();

        current = match ctx.provider.filter_by_mask(&sorted, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_dependency_chain",
                    start.elapsed(),
                    format!("Chain 3 step {} filter failed: {}", i, e),
                )
            }
        };
    }

    // Verify deep chain result
    let deep_result = match ctx.provider.download_column::<u32>(&current, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 3 result download failed: {}", e),
            )
        }
    };

    // Should have about 10000 * 0.8^5 = ~3277 rows
    if deep_result.is_empty() {
        return TestResult::error(
            "test_dependency_chain",
            start.elapsed(),
            "Chain 3: deep chain produced no results".to_string(),
        );
    }

    // Verify still sorted
    for i in 1..deep_result.len() {
        if deep_result[i] < deep_result[i - 1] {
            return TestResult::error(
                "test_dependency_chain",
                start.elapsed(),
                format!("Chain 3: result not sorted at index {}", i),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_dependency_chain",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_dependency_chain", start.elapsed())
}
