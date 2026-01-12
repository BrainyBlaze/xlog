//! Category 7: Local Memory and Stack
//!
//! Tests operations that may use local memory (register spilling) and stack.
//! These tests stress the GPU's register file and local memory allocation.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{Schema, ScalarType};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c07_local_memory");
    let start = Instant::now();

    results.add_result(test_deep_sort_keys(ctx));
    results.add_result(test_repeated_operations(ctx));
    results.add_result(test_variable_workload(ctx));
    results.add_result(test_complex_filter_chains(ctx));
    results.add_result(test_local_memory_stress(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Sort with complex data (multi-column) that may spill registers.
///
/// When sorting with multiple key columns, the sort kernel needs to compare
/// multiple values per element, which may require register spilling to local memory.
fn test_deep_sort_keys(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Create schema with multiple columns (deep key comparison)
    let schema = Schema::new(vec![
        ("k1".to_string(), ScalarType::U32),
        ("k2".to_string(), ScalarType::U32),
        ("k3".to_string(), ScalarType::U32),
        ("k4".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    const SIZE: usize = 10000;

    // Create data where sorting requires comparing multiple columns
    // k1: 0,0,0,...,1,1,1,...,2,2,2,... (groups of 100)
    // k2: 0,0,...,1,1,...,2,2,... (groups of 10 within k1)
    // k3: random-ish within k1,k2
    // k4: 0,1,2,3,... within k1,k2,k3
    let mut k1: Vec<u32> = Vec::with_capacity(SIZE);
    let mut k2: Vec<u32> = Vec::with_capacity(SIZE);
    let mut k3: Vec<u32> = Vec::with_capacity(SIZE);
    let mut k4: Vec<u32> = Vec::with_capacity(SIZE);
    let mut val: Vec<u32> = Vec::with_capacity(SIZE);

    for i in 0..SIZE {
        k1.push((i / 1000) as u32);
        k2.push(((i % 1000) / 100) as u32);
        k3.push(((i % 100) / 10) as u32);
        k4.push((i % 10) as u32);
        val.push(i as u32);
    }

    // Shuffle the data (reverse order)
    k1.reverse();
    k2.reverse();
    k3.reverse();
    k4.reverse();
    val.reverse();

    let buffer = match ctx.provider.create_buffer_from_u32_columns(
        &[&k1, &k2, &k3, &k4, &val],
        schema.clone(),
    ) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_deep_sort_keys",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort by first 4 columns (k1, k2, k3, k4)
    let sorted = match ctx.provider.sort(&buffer, &[0, 1, 2, 3]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_deep_sort_keys",
                start.elapsed(),
                format!("Sort with 4 key columns failed: {}", e),
            )
        }
    };

    // Verify row count
    if sorted.num_rows != SIZE as u64 {
        return TestResult::error(
            "test_deep_sort_keys",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected {}",
                sorted.num_rows, SIZE
            ),
        );
    }

    // Download and verify sorted order
    let sorted_k1 = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_deep_sort_keys",
                start.elapsed(),
                format!("Failed to download k1: {}", e),
            )
        }
    };

    let sorted_k2 = match ctx.provider.download_column_u32(&sorted, 1) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_deep_sort_keys",
                start.elapsed(),
                format!("Failed to download k2: {}", e),
            )
        }
    };

    let sorted_val = match ctx.provider.download_column_u32(&sorted, 4) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_deep_sort_keys",
                start.elapsed(),
                format!("Failed to download val: {}", e),
            )
        }
    };

    // Verify lexicographic ordering of keys
    for i in 1..SIZE {
        let prev = (sorted_k1[i - 1], sorted_k2[i - 1]);
        let curr = (sorted_k1[i], sorted_k2[i]);

        if curr < prev {
            return TestResult::error(
                "test_deep_sort_keys",
                start.elapsed(),
                format!(
                    "Sort order incorrect at {}: ({},{}) > ({},{})",
                    i, prev.0, prev.1, curr.0, curr.1
                ),
            );
        }
    }

    // Verify all original values are present
    let mut val_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for &v in &sorted_val {
        val_set.insert(v);
    }

    if val_set.len() != SIZE {
        return TestResult::error(
            "test_deep_sort_keys",
            start.elapsed(),
            format!(
                "Lost values during sort: {} unique values, expected {}",
                val_set.len(), SIZE
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_deep_sort_keys",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_deep_sort_keys", start.elapsed())
}

/// Test 2: Run many operations back-to-back to stress stack.
///
/// Tests that multiple sequential operations don't cause stack overflow
/// or local memory exhaustion.
fn test_repeated_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 5000;
    const ITERATIONS: usize = 20;

    // Create initial buffer
    let data: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_repeated_operations",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Run many sort operations back-to-back
    for iter in 0..ITERATIONS {
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_operations",
                    start.elapsed(),
                    format!("Sort failed at iteration {}: {}", iter, e),
                )
            }
        };

        // Verify row count
        if sorted.num_rows != SIZE as u64 {
            return TestResult::error(
                "test_repeated_operations",
                start.elapsed(),
                format!(
                    "Iteration {}: sort returned {} rows, expected {}",
                    iter, sorted.num_rows, SIZE
                ),
            );
        }

        // Every 5th iteration, verify data correctness
        if iter % 5 == 0 {
            let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
                Ok(d) => d,
                Err(e) => {
                    return TestResult::error(
                        "test_repeated_operations",
                        start.elapsed(),
                        format!("Iteration {}: failed to download: {}", iter, e),
                    )
                }
            };

            if sorted_data != data {
                return TestResult::error(
                    "test_repeated_operations",
                    start.elapsed(),
                    format!("Iteration {}: data corrupted", iter),
                );
            }
        }
    }

    // Run filter operations back-to-back
    for iter in 0..ITERATIONS {
        let mask: Vec<u8> = (0..SIZE).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();

        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_operations",
                    start.elapsed(),
                    format!("Filter failed at iteration {}: {}", iter, e),
                )
            }
        };

        let expected_count = (SIZE + 1) / 2;
        if filtered.num_rows != expected_count as u64 {
            return TestResult::error(
                "test_repeated_operations",
                start.elapsed(),
                format!(
                    "Filter iteration {}: returned {} rows, expected {}",
                    iter, filtered.num_rows, expected_count
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_repeated_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_repeated_operations", start.elapsed())
}

/// Test 3: Mix small and large operations.
///
/// Tests that the GPU correctly handles varying workload sizes that
/// may have different register pressure.
fn test_variable_workload(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Alternating small and large sizes
    let sizes: Vec<usize> = vec![
        10,      // Tiny
        100000,  // Large
        100,     // Small
        50000,   // Medium-large
        50,      // Tiny
        200000,  // Large
        1000,    // Medium
        10000,   // Medium
    ];

    for (i, &size) in sizes.iter().enumerate() {
        // Create data in reverse order
        let data: Vec<u32> = (0..size as u32).rev().collect();

        let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_variable_workload",
                    start.elapsed(),
                    format!("Test {}: failed to create buffer of size {}: {}", i, size, e),
                )
            }
        };

        // Sort the buffer
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_variable_workload",
                    start.elapsed(),
                    format!("Test {}: sort of size {} failed: {}", i, size, e),
                )
            }
        };

        // Verify row count
        if sorted.num_rows != size as u64 {
            return TestResult::error(
                "test_variable_workload",
                start.elapsed(),
                format!(
                    "Test {}: size {} returned {} rows",
                    i, size, sorted.num_rows
                ),
            );
        }

        // Verify first and last elements
        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_variable_workload",
                    start.elapsed(),
                    format!("Test {}: failed to download: {}", i, e),
                )
            }
        };

        if sorted_data[0] != 0 {
            return TestResult::error(
                "test_variable_workload",
                start.elapsed(),
                format!(
                    "Test {}: first element is {}, expected 0",
                    i, sorted_data[0]
                ),
            );
        }

        if sorted_data[size - 1] != (size - 1) as u32 {
            return TestResult::error(
                "test_variable_workload",
                start.elapsed(),
                format!(
                    "Test {}: last element is {}, expected {}",
                    i,
                    sorted_data[size - 1],
                    size - 1
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_variable_workload",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_variable_workload", start.elapsed())
}

/// Test 4: Apply multiple filters in sequence.
///
/// Tests that chained filter operations work correctly and don't cause
/// stack or local memory issues.
fn test_complex_filter_chains(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 100000;

    // Create sequential data
    let data: Vec<u32> = (0..SIZE as u32).collect();

    let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_complex_filter_chains",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Filter 1: Keep even indices (50000 remaining)
    let mask1: Vec<u8> = (0..SIZE).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let filtered1 = match ctx.provider.filter_by_mask(&buffer, &mask1) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_complex_filter_chains",
                start.elapsed(),
                format!("Filter 1 failed: {}", e),
            )
        }
    };

    let expected1 = (SIZE + 1) / 2;
    if filtered1.num_rows != expected1 as u64 {
        return TestResult::error(
            "test_complex_filter_chains",
            start.elapsed(),
            format!(
                "Filter 1: returned {} rows, expected {}",
                filtered1.num_rows, expected1
            ),
        );
    }

    // Filter 2: From filter1 result, keep every 4th (12500 remaining)
    let mask2: Vec<u8> = (0..expected1).map(|i| if i % 4 == 0 { 1 } else { 0 }).collect();
    let filtered2 = match ctx.provider.filter_by_mask(&filtered1, &mask2) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_complex_filter_chains",
                start.elapsed(),
                format!("Filter 2 failed: {}", e),
            )
        }
    };

    let expected2 = (expected1 + 3) / 4;
    if filtered2.num_rows != expected2 as u64 {
        return TestResult::error(
            "test_complex_filter_chains",
            start.elapsed(),
            format!(
                "Filter 2: returned {} rows, expected {}",
                filtered2.num_rows, expected2
            ),
        );
    }

    // Filter 3: From filter2 result, keep every 5th (2500 remaining)
    let mask3: Vec<u8> = (0..expected2).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();
    let filtered3 = match ctx.provider.filter_by_mask(&filtered2, &mask3) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_complex_filter_chains",
                start.elapsed(),
                format!("Filter 3 failed: {}", e),
            )
        }
    };

    let expected3 = (expected2 + 4) / 5;
    if filtered3.num_rows != expected3 as u64 {
        return TestResult::error(
            "test_complex_filter_chains",
            start.elapsed(),
            format!(
                "Filter 3: returned {} rows, expected {}",
                filtered3.num_rows, expected3
            ),
        );
    }

    // Verify final values are correct
    // Original indices that survive: 0, 8*20=40, 8*40=80, ...
    // After filter1: 0, 2, 4, 6, 8, ... (even indices)
    // After filter2 (every 4th): 0, 8, 16, 24, ... (indices 0, 4, 8, 12 from even = 0, 8, 16, 24)
    // After filter3 (every 5th): 0, 40, 80, 120, ... (indices 0, 5, 10, 15 from above = 0, 40, 80, 120)
    let final_data = match ctx.provider.download_column_u32(&filtered3, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_complex_filter_chains",
                start.elapsed(),
                format!("Failed to download final result: {}", e),
            )
        }
    };

    // Check first few values
    // filter1 keeps: 0, 2, 4, 6, 8, 10, ...
    // filter2 keeps indices 0, 4, 8, 12 from filter1 result = values 0, 8, 16, 24, ...
    // filter3 keeps indices 0, 5, 10, 15 from filter2 result = values 0, 40, 80, 120, ...
    let expected_first_values: Vec<u32> = (0..5).map(|i| (i * 40) as u32).collect();

    for (i, &expected) in expected_first_values.iter().enumerate() {
        if i >= final_data.len() {
            break;
        }
        if final_data[i] != expected {
            return TestResult::error(
                "test_complex_filter_chains",
                start.elapsed(),
                format!(
                    "Final result[{}] = {}, expected {}",
                    i, final_data[i], expected
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_complex_filter_chains",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_complex_filter_chains", start.elapsed())
}

/// Test 5: Many small operations in rapid succession.
///
/// Tests local memory handling when many small kernels are launched rapidly,
/// potentially causing frequent allocation/deallocation of local memory.
fn test_local_memory_stress(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const ITERATIONS: usize = 50;
    const SIZES: [usize; 5] = [64, 128, 256, 512, 1024];

    for iter in 0..ITERATIONS {
        for &size in &SIZES {
            // Create data
            let data: Vec<u32> = (0..size as u32).rev().collect();

            let buffer = match ctx.provider.create_buffer_from_u32_slice(&data, schema.clone()) {
                Ok(buf) => buf,
                Err(e) => {
                    return TestResult::error(
                        "test_local_memory_stress",
                        start.elapsed(),
                        format!("Iter {}, size {}: buffer creation failed: {}", iter, size, e),
                    )
                }
            };

            // Sort
            let sorted = match ctx.provider.sort(&buffer, &[0]) {
                Ok(s) => s,
                Err(e) => {
                    return TestResult::error(
                        "test_local_memory_stress",
                        start.elapsed(),
                        format!("Iter {}, size {}: sort failed: {}", iter, size, e),
                    )
                }
            };

            // Quick verify
            if sorted.num_rows != size as u64 {
                return TestResult::error(
                    "test_local_memory_stress",
                    start.elapsed(),
                    format!(
                        "Iter {}, size {}: wrong row count {}",
                        iter, size, sorted.num_rows
                    ),
                );
            }

            // Filter half
            let mask: Vec<u8> = (0..size).map(|i| if i < size / 2 { 1 } else { 0 }).collect();
            let filtered = match ctx.provider.filter_by_mask(&sorted, &mask) {
                Ok(f) => f,
                Err(e) => {
                    return TestResult::error(
                        "test_local_memory_stress",
                        start.elapsed(),
                        format!("Iter {}, size {}: filter failed: {}", iter, size, e),
                    )
                }
            };

            if filtered.num_rows != (size / 2) as u64 {
                return TestResult::error(
                    "test_local_memory_stress",
                    start.elapsed(),
                    format!(
                        "Iter {}, size {}: filter returned {} rows",
                        iter, size, filtered.num_rows
                    ),
                );
            }
        }

        // Sync every 10 iterations to check for errors
        if iter % 10 == 9 {
            if let Err(e) = ctx.sync_and_check() {
                return TestResult::error(
                    "test_local_memory_stress",
                    start.elapsed(),
                    format!("Sync failed at iteration {}: {}", iter, e),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_local_memory_stress",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_local_memory_stress", start.elapsed())
}
