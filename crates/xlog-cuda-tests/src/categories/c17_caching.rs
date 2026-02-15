//! Category 17: Caching and coherence
//!
//! Tests cache behavior and coherence, including cache line access patterns,
//! cache reuse, cache thrashing scenarios, memory locality, and L2 cache effects.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c17_caching");
    let start = Instant::now();

    results.add_result(test_cache_line_access(ctx));
    results.add_result(test_cache_reuse(ctx));
    results.add_result(test_cache_thrashing(ctx));
    results.add_result(test_memory_locality(ctx));
    results.add_result(test_l2_cache_effects(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Test sizes aligned to cache lines (128 bytes).
///
/// GPU cache lines are typically 128 bytes. This test verifies operations
/// work correctly with data sizes that are aligned to cache line boundaries.
fn test_cache_line_access(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Cache line is 128 bytes = 32 u32 values
    const CACHE_LINE_U32S: usize = 32;

    // Test various cache line aligned sizes
    let test_sizes = [
        CACHE_LINE_U32S,       // 1 cache line
        CACHE_LINE_U32S * 2,   // 2 cache lines
        CACHE_LINE_U32S * 4,   // 4 cache lines
        CACHE_LINE_U32S * 16,  // 16 cache lines
        CACHE_LINE_U32S * 64,  // 64 cache lines
        CACHE_LINE_U32S * 256, // 256 cache lines
    ];

    for &size in &test_sizes {
        // Create data aligned to cache line size
        let data: Vec<u32> = (0..size)
            .map(|i| ((i * 1103515245 + 12345) % 1000000) as u32)
            .collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!("Failed to create buffer of size {}: {}", size, e),
                )
            }
        };

        // Sort operation to exercise cache
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!("Sort failed for size {}: {}", size, e),
                )
            }
        };

        // Verify sort correctness
        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!("Download failed for size {}: {}", size, e),
                )
            }
        };

        if sorted_data.len() != size {
            return TestResult::error(
                "test_cache_line_access",
                start.elapsed(),
                format!(
                    "Size {}: expected {} rows, got {}",
                    size,
                    size,
                    sorted_data.len()
                ),
            );
        }

        for i in 1..sorted_data.len() {
            if sorted_data[i] < sorted_data[i - 1] {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!(
                        "Size {}: sort incorrect at index {}: {} < {}",
                        size,
                        i,
                        sorted_data[i],
                        sorted_data[i - 1]
                    ),
                );
            }
        }

        // Also test filter with cache-aligned data
        let mask: Vec<u8> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!("Filter failed for size {}: {}", size, e),
                )
            }
        };

        let expected_count = (size + 1) / 2;
        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_cache_line_access",
                start.elapsed(),
                format!(
                    "Size {}: filter expected {} rows, got {}",
                    size,
                    expected_count,
                    ctx.device_row_count(&filtered)
                ),
            );
        }
    }

    // Test non-aligned sizes (to verify no issues with non-aligned access)
    let non_aligned_sizes = [
        CACHE_LINE_U32S + 1,
        CACHE_LINE_U32S * 2 - 1,
        CACHE_LINE_U32S * 4 + 7,
        CACHE_LINE_U32S * 10 + 13,
    ];

    for &size in &non_aligned_sizes {
        let data: Vec<u32> = (0..size).map(|i| ((i * 31337) % 100000) as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_u32_slice(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!(
                        "Failed to create non-aligned buffer of size {}: {}",
                        size, e
                    ),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!("Sort failed for non-aligned size {}: {}", size, e),
                )
            }
        };

        let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!("Download failed for non-aligned size {}: {}", size, e),
                )
            }
        };

        for i in 1..sorted_data.len() {
            if sorted_data[i] < sorted_data[i - 1] {
                return TestResult::error(
                    "test_cache_line_access",
                    start.elapsed(),
                    format!("Non-aligned size {}: sort incorrect at index {}", size, i),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cache_line_access",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_cache_line_access", start.elapsed())
}

/// Test 2: Run same operation multiple times to exercise cache reuse.
///
/// Tests cache efficiency by running the same operation repeatedly on the
/// same data, which should benefit from cache warm-up effects.
fn test_cache_reuse(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 10000;
    const ITERATIONS: usize = 10;

    // Create data that should fit in cache after first access
    let data: Vec<u32> = (0..SIZE).map(|i| ((i * 17 + 13) % 10000) as u32).collect();

    let buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_cache_reuse",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Run sort multiple times on same data - should benefit from cache
    let mut first_result: Option<Vec<u32>> = None;

    for i in 0..ITERATIONS {
        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_cache_reuse",
                    start.elapsed(),
                    format!("Iteration {}: sort failed: {}", i, e),
                )
            }
        };

        let result = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_cache_reuse",
                    start.elapsed(),
                    format!("Iteration {}: download failed: {}", i, e),
                )
            }
        };

        // Verify correctness
        for j in 1..result.len() {
            if result[j] < result[j - 1] {
                return TestResult::error(
                    "test_cache_reuse",
                    start.elapsed(),
                    format!("Iteration {}: sort incorrect at index {}", i, j),
                );
            }
        }

        // Verify consistency across iterations
        match &first_result {
            Some(first) => {
                if result != *first {
                    return TestResult::error(
                        "test_cache_reuse",
                        start.elapsed(),
                        format!("Iteration {}: result differs from first iteration", i),
                    );
                }
            }
            None => {
                first_result = Some(result);
            }
        }
    }

    // Test filter cache reuse
    let mask: Vec<u8> = (0..SIZE).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let expected_count = (SIZE + 2) / 3;

    for i in 0..ITERATIONS {
        let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
            Ok(f) => f,
            Err(e) => {
                return TestResult::error(
                    "test_cache_reuse",
                    start.elapsed(),
                    format!("Filter iteration {}: failed: {}", i, e),
                )
            }
        };

        if ctx.device_row_count(&filtered) != expected_count as u64 {
            return TestResult::error(
                "test_cache_reuse",
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

    // Test dedup cache reuse with duplicates
    let schema2 = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("val".to_string(), ScalarType::U32),
    ]);

    let keys: Vec<u32> = (0..SIZE).map(|i| (i % 1000) as u32).collect();
    let vals: Vec<u32> = (0..SIZE as u32).collect();

    let buffer2 = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&keys, &vals], schema2)
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_cache_reuse",
                start.elapsed(),
                format!("Failed to create buffer2: {}", e),
            )
        }
    };

    for i in 0..ITERATIONS {
        let deduped = match ctx.provider.dedup(&buffer2, &[0]) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_cache_reuse",
                    start.elapsed(),
                    format!("Dedup iteration {}: failed: {}", i, e),
                )
            }
        };

        if ctx.device_row_count(&deduped) != 1000 {
            return TestResult::error(
                "test_cache_reuse",
                start.elapsed(),
                format!(
                    "Dedup iteration {}: expected 1000 rows, got {}",
                    i,
                    ctx.device_row_count(&deduped)
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cache_reuse",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_cache_reuse", start.elapsed())
}

/// Test 3: Large data that exceeds cache capacity to test cache thrashing.
///
/// Tests behavior when working set exceeds cache capacity, which causes
/// cache thrashing and higher memory bandwidth requirements.
fn test_cache_thrashing(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // L2 cache is typically 2-6MB on modern GPUs
    // 1M u32s = 4MB, 5M u32s = 20MB (definitely exceeds L2)
    const LARGE_SIZE: usize = 5_000_000;
    const MEDIUM_SIZE: usize = 1_000_000;

    // Test with large data that exceeds L2 cache
    let large_data: Vec<u32> = (0..LARGE_SIZE)
        .map(|i| ((i * 1103515245 + 12345) % 10000000) as u32)
        .collect();

    let large_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&large_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Failed to create large buffer: {}", e),
            )
        }
    };

    // Sort large data (will cause cache thrashing)
    let sorted = match ctx.provider.sort(&large_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Large sort failed: {}", e),
            )
        }
    };

    // Verify correctness even with cache thrashing
    let sorted_data = match ctx.provider.download_column_u32(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Large download failed: {}", e),
            )
        }
    };

    if sorted_data.len() != LARGE_SIZE {
        return TestResult::error(
            "test_cache_thrashing",
            start.elapsed(),
            format!(
                "Large sort: expected {} rows, got {}",
                LARGE_SIZE,
                sorted_data.len()
            ),
        );
    }

    // Spot check sorted order (checking every element is slow)
    for i in (1..sorted_data.len()).step_by(10000) {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!(
                    "Large sort incorrect at index {}: {} < {}",
                    i,
                    sorted_data[i],
                    sorted_data[i - 1]
                ),
            );
        }
    }

    // Also check first and last segments thoroughly
    for i in 1..1000.min(sorted_data.len()) {
        if sorted_data[i] < sorted_data[i - 1] {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Large sort incorrect at start index {}", i),
            );
        }
    }

    if sorted_data.len() > 1000 {
        for i in (sorted_data.len() - 999)..sorted_data.len() {
            if sorted_data[i] < sorted_data[i - 1] {
                return TestResult::error(
                    "test_cache_thrashing",
                    start.elapsed(),
                    format!("Large sort incorrect at end index {}", i),
                );
            }
        }
    }

    // Test medium size that may partially fit in L2
    let medium_data: Vec<u32> = (0..MEDIUM_SIZE)
        .map(|i| ((i * 31337) % 1000000) as u32)
        .collect();

    let medium_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&medium_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Failed to create medium buffer: {}", e),
            )
        }
    };

    let sorted_medium = match ctx.provider.sort(&medium_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Medium sort failed: {}", e),
            )
        }
    };

    let medium_result = match ctx.provider.download_column_u32(&sorted_medium, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Medium download failed: {}", e),
            )
        }
    };

    for i in (1..medium_result.len()).step_by(1000) {
        if medium_result[i] < medium_result[i - 1] {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Medium sort incorrect at index {}", i),
            );
        }
    }

    // Filter on large data
    let large_mask: Vec<u8> = (0..LARGE_SIZE)
        .map(|i| if i % 4 == 0 { 1 } else { 0 })
        .collect();
    let filtered = match ctx.provider.filter_by_mask(&large_buffer, &large_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_cache_thrashing",
                start.elapsed(),
                format!("Large filter failed: {}", e),
            )
        }
    };

    let expected_count = (LARGE_SIZE + 3) / 4;
    if ctx.device_row_count(&filtered) != expected_count as u64 {
        return TestResult::error(
            "test_cache_thrashing",
            start.elapsed(),
            format!(
                "Large filter: expected {} rows, got {}",
                expected_count,
                ctx.device_row_count(&filtered)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cache_thrashing",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_cache_thrashing", start.elapsed())
}

/// Test 4: Test operations with good and bad memory locality.
///
/// Tests operations with sequential access patterns (good locality) versus
/// random access patterns (bad locality) to verify correctness in both cases.
fn test_memory_locality(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 100000;

    // Good locality: sequential data
    let sequential_data: Vec<u32> = (0..SIZE as u32).collect();

    let seq_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&sequential_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Failed to create sequential buffer: {}", e),
            )
        }
    };

    // Sort sequential data (already sorted - best case)
    let sorted_seq = match ctx.provider.sort(&seq_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Sort sequential failed: {}", e),
            )
        }
    };

    let seq_result = match ctx.provider.download_column_u32(&sorted_seq, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Download sequential failed: {}", e),
            )
        }
    };

    // Verify sorted
    for i in 0..seq_result.len() {
        if seq_result[i] != i as u32 {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!(
                    "Sequential sort incorrect at index {}: expected {}, got {}",
                    i, i, seq_result[i]
                ),
            );
        }
    }

    // Bad locality: reverse sorted data
    let reverse_data: Vec<u32> = (0..SIZE as u32).rev().collect();

    let rev_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&reverse_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Failed to create reverse buffer: {}", e),
            )
        }
    };

    let sorted_rev = match ctx.provider.sort(&rev_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Sort reverse failed: {}", e),
            )
        }
    };

    let rev_result = match ctx.provider.download_column_u32(&sorted_rev, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Download reverse failed: {}", e),
            )
        }
    };

    // Verify sorted
    for i in 0..rev_result.len() {
        if rev_result[i] != i as u32 {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!(
                    "Reverse sort incorrect at index {}: expected {}, got {}",
                    i, i, rev_result[i]
                ),
            );
        }
    }

    // Worst locality: random access pattern
    let random_data: Vec<u32> = (0..SIZE)
        .map(|i| ((i * 1103515245 + 12345) % SIZE) as u32)
        .collect();

    let rand_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&random_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Failed to create random buffer: {}", e),
            )
        }
    };

    let sorted_rand = match ctx.provider.sort(&rand_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Sort random failed: {}", e),
            )
        }
    };

    let rand_result = match ctx.provider.download_column_u32(&sorted_rand, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Download random failed: {}", e),
            )
        }
    };

    // Verify sorted order
    for i in 1..rand_result.len() {
        if rand_result[i] < rand_result[i - 1] {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!(
                    "Random sort incorrect at index {}: {} < {}",
                    i,
                    rand_result[i],
                    rand_result[i - 1]
                ),
            );
        }
    }

    // Test filter with different locality patterns
    // Sequential mask pattern (good locality)
    let seq_mask: Vec<u8> = (0..SIZE)
        .map(|i| if i < SIZE / 2 { 1 } else { 0 })
        .collect();
    let filtered_seq = match ctx.provider.filter_by_mask(&seq_buffer, &seq_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Sequential filter failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&filtered_seq) != (SIZE / 2) as u64 {
        return TestResult::error(
            "test_memory_locality",
            start.elapsed(),
            format!(
                "Sequential filter: expected {} rows, got {}",
                SIZE / 2,
                ctx.device_row_count(&filtered_seq)
            ),
        );
    }

    // Alternating mask pattern (stride access)
    let alt_mask: Vec<u8> = (0..SIZE).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let filtered_alt = match ctx.provider.filter_by_mask(&seq_buffer, &alt_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Alternating filter failed: {}", e),
            )
        }
    };

    let expected_alt = (SIZE + 1) / 2;
    if ctx.device_row_count(&filtered_alt) != expected_alt as u64 {
        return TestResult::error(
            "test_memory_locality",
            start.elapsed(),
            format!(
                "Alternating filter: expected {} rows, got {}",
                expected_alt,
                ctx.device_row_count(&filtered_alt)
            ),
        );
    }

    // Sparse mask (bad locality for output)
    let sparse_mask: Vec<u8> = (0..SIZE)
        .map(|i| if i % 100 == 0 { 1 } else { 0 })
        .collect();
    let filtered_sparse = match ctx.provider.filter_by_mask(&seq_buffer, &sparse_mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_memory_locality",
                start.elapsed(),
                format!("Sparse filter failed: {}", e),
            )
        }
    };

    let expected_sparse = (SIZE + 99) / 100;
    if ctx.device_row_count(&filtered_sparse) != expected_sparse as u64 {
        return TestResult::error(
            "test_memory_locality",
            start.elapsed(),
            format!(
                "Sparse filter: expected {} rows, got {}",
                expected_sparse,
                ctx.device_row_count(&filtered_sparse)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_memory_locality",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_memory_locality", start.elapsed())
}

/// Test 5: Test sizes that fit vs overflow L2 cache.
///
/// Tests operations with data sizes that fit within L2 cache (fast) versus
/// sizes that overflow L2 cache (requires main memory access).
fn test_l2_cache_effects(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Typical L2 cache sizes: 2-6MB on modern GPUs
    // Small: definitely fits in L2 (256KB = 64K u32s)
    // Medium: might fit in L2 (2MB = 512K u32s)
    // Large: definitely exceeds L2 (20MB = 5M u32s)

    const SMALL_SIZE: usize = 64_000; // ~256KB
    const MEDIUM_SIZE: usize = 512_000; // ~2MB
    const LARGE_SIZE: usize = 2_000_000; // ~8MB

    // Test small (L2 resident)
    let small_data: Vec<u32> = (0..SMALL_SIZE)
        .map(|i| ((i * 17 + 13) % SMALL_SIZE) as u32)
        .collect();

    let small_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&small_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Failed to create small buffer: {}", e),
            )
        }
    };

    // Run multiple times to benefit from L2 caching
    for i in 0..3 {
        let sorted = match ctx.provider.sort(&small_buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_l2_cache_effects",
                    start.elapsed(),
                    format!("Small sort iteration {} failed: {}", i, e),
                )
            }
        };

        let result = match ctx.provider.download_column_u32(&sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_l2_cache_effects",
                    start.elapsed(),
                    format!("Small download iteration {} failed: {}", i, e),
                )
            }
        };

        for j in 1..result.len() {
            if result[j] < result[j - 1] {
                return TestResult::error(
                    "test_l2_cache_effects",
                    start.elapsed(),
                    format!("Small sort iteration {} incorrect at index {}", i, j),
                );
            }
        }
    }

    // Test medium (borderline L2)
    let medium_data: Vec<u32> = (0..MEDIUM_SIZE)
        .map(|i| ((i * 31337) % MEDIUM_SIZE) as u32)
        .collect();

    let medium_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&medium_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Failed to create medium buffer: {}", e),
            )
        }
    };

    let sorted_medium = match ctx.provider.sort(&medium_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Medium sort failed: {}", e),
            )
        }
    };

    let medium_result = match ctx.provider.download_column_u32(&sorted_medium, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Medium download failed: {}", e),
            )
        }
    };

    for i in (1..medium_result.len()).step_by(1000) {
        if medium_result[i] < medium_result[i - 1] {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Medium sort incorrect at index {}", i),
            );
        }
    }

    // Test large (exceeds L2)
    let large_data: Vec<u32> = (0..LARGE_SIZE)
        .map(|i| ((i * 1103515245 + 12345) % LARGE_SIZE) as u32)
        .collect();

    let large_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&large_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Failed to create large buffer: {}", e),
            )
        }
    };

    let sorted_large = match ctx.provider.sort(&large_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Large sort failed: {}", e),
            )
        }
    };

    let large_result = match ctx.provider.download_column_u32(&sorted_large, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Large download failed: {}", e),
            )
        }
    };

    // Spot check large result
    for i in (1..large_result.len()).step_by(10000) {
        if large_result[i] < large_result[i - 1] {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Large sort incorrect at index {}", i),
            );
        }
    }

    // Test interleaved operations at different cache levels
    // This exercises cache replacement policies
    let small2_data: Vec<u32> = (0..SMALL_SIZE).map(|i| (i * 3) as u32).collect();
    let small2_buffer = match ctx
        .provider
        .create_buffer_from_u32_slice(&small2_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!("Failed to create small2 buffer: {}", e),
            )
        }
    };

    // Interleave operations on small and medium buffers
    for i in 0..3 {
        // Small operation
        let small_sorted = match ctx.provider.sort(&small2_buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_l2_cache_effects",
                    start.elapsed(),
                    format!("Interleaved small sort {} failed: {}", i, e),
                )
            }
        };

        // Medium operation (may evict small from L2)
        let medium_sorted = match ctx.provider.sort(&medium_buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_l2_cache_effects",
                    start.elapsed(),
                    format!("Interleaved medium sort {} failed: {}", i, e),
                )
            }
        };

        // Verify both completed correctly
        if ctx.device_row_count(&small_sorted) != SMALL_SIZE as u64 {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!(
                    "Interleaved {}: small has {} rows, expected {}",
                    i,
                    ctx.device_row_count(&small_sorted),
                    SMALL_SIZE
                ),
            );
        }

        if ctx.device_row_count(&medium_sorted) != MEDIUM_SIZE as u64 {
            return TestResult::error(
                "test_l2_cache_effects",
                start.elapsed(),
                format!(
                    "Interleaved {}: medium has {} rows, expected {}",
                    i,
                    ctx.device_row_count(&medium_sorted),
                    MEDIUM_SIZE
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_l2_cache_effects",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_l2_cache_effects", start.elapsed())
}
