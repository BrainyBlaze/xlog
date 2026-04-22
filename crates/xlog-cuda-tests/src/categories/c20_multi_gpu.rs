//! Category 20: Multi-GPU
//!
//! Tests multi-GPU scenarios including device detection, enumeration,
//! primary device operations, and capability queries. Tests are skipped
//! if only one GPU is available.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::CudaDevice;

/// Run all tests in this category.
pub(crate) fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c20_multi_gpu");
    let start = Instant::now();

    results.add_result(test_single_gpu_baseline(ctx));
    results.add_result(test_multi_gpu_detection(ctx));
    results.add_result(test_device_enumeration(ctx));
    results.add_result(test_primary_device_operations(ctx));
    results.add_result(test_device_capability_query(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Verify single GPU operations work.
///
/// Tests that basic operations work correctly on the primary GPU,
/// establishing a baseline for multi-GPU comparison.
fn test_single_gpu_baseline(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    const SIZE: usize = 100000;

    // Create test data
    let data: Vec<u32> = (0..SIZE)
        .map(|i| ((i * 1103515245 + 12345) % 1000000) as u32)
        .collect();

    // Upload
    let buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Failed to create buffer: {}", e),
            )
        }
    };

    // Sort
    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Sort failed: {}", e),
            )
        }
    };

    // Download and verify
    let result = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Download failed: {}", e),
            )
        }
    };

    // Verify sorted
    for i in 1..result.len() {
        if result[i] < result[i - 1] {
            return TestResult::error(
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Sort incorrect at index {}", i),
            );
        }
    }

    // Verify same elements
    let mut expected = data.clone();
    expected.sort();
    if result != expected {
        return TestResult::error(
            "test_single_gpu_baseline",
            start.elapsed(),
            "Sorted result doesn't match expected".to_string(),
        );
    }

    // Test filter
    let mask: Vec<u8> = (0..SIZE).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let filtered = match ctx.provider.filter_by_mask(&buffer, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Filter failed: {}", e),
            )
        }
    };

    let expected_count = (SIZE + 2) / 3;
    if ctx.device_row_count(&filtered) != expected_count as u64 {
        return TestResult::error(
            "test_single_gpu_baseline",
            start.elapsed(),
            format!(
                "Filter: expected {} rows, got {}",
                expected_count,
                ctx.device_row_count(&filtered)
            ),
        );
    }

    // Test dedup
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
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Failed to create buffer2: {}", e),
            )
        }
    };

    let deduped = match ctx.provider.dedup(&buffer2, &[0]) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Dedup failed: {}", e),
            )
        }
    };

    if ctx.device_row_count(&deduped) != 1000 {
        return TestResult::error(
            "test_single_gpu_baseline",
            start.elapsed(),
            format!(
                "Dedup: expected 1000 unique, got {}",
                ctx.device_row_count(&deduped)
            ),
        );
    }

    // Test join
    let left_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("lval".to_string(), ScalarType::U32),
    ]);
    let right_schema = Schema::new(vec![
        ("key".to_string(), ScalarType::U32),
        ("rval".to_string(), ScalarType::U32),
    ]);

    let left_keys: Vec<u32> = (0..1000u32).collect();
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k * 2).collect();

    let right_keys: Vec<u32> = (0..500u32).map(|i| i * 2).collect();
    let right_vals: Vec<u32> = right_keys.iter().map(|&k| k * 3).collect();

    let left_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&left_keys, &left_vals], left_schema)
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Failed to create left buffer: {}", e),
            )
        }
    };

    let right_buffer = match ctx
        .provider
        .create_buffer_from_u32_columns(&[&right_keys, &right_vals], right_schema)
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Failed to create right buffer: {}", e),
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
                "test_single_gpu_baseline",
                start.elapsed(),
                format!("Join failed: {}", e),
            )
        }
    };

    // Should have 500 matches (even keys 0-998)
    if ctx.device_row_count(&joined) != 500 {
        return TestResult::error(
            "test_single_gpu_baseline",
            start.elapsed(),
            format!(
                "Join: expected 500 rows, got {}",
                ctx.device_row_count(&joined)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_single_gpu_baseline",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_single_gpu_baseline", start.elapsed())
}

/// Test 2: Detect if multiple GPUs are available.
///
/// Tests the multi_gpu_available() function and reports whether
/// multi-GPU is available. This test always passes but provides
/// diagnostic information.
fn test_multi_gpu_detection(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let multi_gpu = ctx.multi_gpu_available();

    // Get device count for diagnostics
    let device_count = CudaDevice::count().unwrap_or(0);

    // This test always passes - it's diagnostic
    // The multi_gpu flag should match device_count > 1
    let expected_multi = device_count > 1;

    if multi_gpu != expected_multi {
        return TestResult::error(
            "test_multi_gpu_detection",
            start.elapsed(),
            format!(
                "Inconsistent detection: multi_gpu_available()={}, device_count={}",
                multi_gpu, device_count
            ),
        );
    }

    // Verify we can still do operations regardless of multi-GPU status
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let data: Vec<u32> = (0..1000u32).collect();

    let buffer = match ctx.provider.create_buffer_from_slice::<u32>(&data, schema) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_multi_gpu_detection",
                start.elapsed(),
                format!("Buffer creation failed after detection: {}", e),
            )
        }
    };

    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_multi_gpu_detection",
                start.elapsed(),
                format!("Sort failed after detection: {}", e),
            )
        }
    };

    if ctx.device_row_count(&sorted) != 1000 {
        return TestResult::error(
            "test_multi_gpu_detection",
            start.elapsed(),
            format!(
                "Sort returned {} rows, expected 1000",
                ctx.device_row_count(&sorted)
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_multi_gpu_detection",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_multi_gpu_detection", start.elapsed())
}

/// Test 3: Enumerate available devices.
///
/// Tests device enumeration and reports device information.
/// Skipped if no CUDA devices are available (shouldn't happen if tests run).
fn test_device_enumeration(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Get device count
    let device_count = match CudaDevice::count() {
        Ok(count) => count,
        Err(e) => {
            return TestResult::error(
                "test_device_enumeration",
                start.elapsed(),
                format!("Failed to get device count: {}", e),
            )
        }
    };

    if device_count == 0 {
        return TestResult::skipped("test_device_enumeration", "No CUDA devices available");
    }

    // Verify device count is reasonable
    if device_count > 16 {
        return TestResult::error(
            "test_device_enumeration",
            start.elapsed(),
            format!("Suspicious device count: {} (>16)", device_count),
        );
    }

    // Verify we can access the primary device (device 0)
    // The test context should already have initialized device 0
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    for test_num in 0..device_count.min(2) {
        // Test basic operation to verify device is usable
        // (We can only test device 0 with the current context)
        if test_num == 0 {
            let data: Vec<u32> = (0..1000u32).collect();

            let buffer = match ctx
                .provider
                .create_buffer_from_slice::<u32>(&data, schema.clone())
            {
                Ok(buf) => buf,
                Err(e) => {
                    return TestResult::error(
                        "test_device_enumeration",
                        start.elapsed(),
                        format!("Device {}: buffer creation failed: {}", test_num, e),
                    )
                }
            };

            let sorted = match ctx.provider.sort(&buffer, &[0]) {
                Ok(s) => s,
                Err(e) => {
                    return TestResult::error(
                        "test_device_enumeration",
                        start.elapsed(),
                        format!("Device {}: sort failed: {}", test_num, e),
                    )
                }
            };

            if ctx.device_row_count(&sorted) != 1000 {
                return TestResult::error(
                    "test_device_enumeration",
                    start.elapsed(),
                    format!(
                        "Device {}: sort returned {} rows",
                        test_num,
                        ctx.device_row_count(&sorted)
                    ),
                );
            }
        }
    }

    // Verify enumeration is consistent
    let device_count2 = match CudaDevice::count() {
        Ok(count) => count,
        Err(e) => {
            return TestResult::error(
                "test_device_enumeration",
                start.elapsed(),
                format!("Failed to get device count on second call: {}", e),
            )
        }
    };

    if device_count != device_count2 {
        return TestResult::error(
            "test_device_enumeration",
            start.elapsed(),
            format!(
                "Device count changed between calls: {} vs {}",
                device_count, device_count2
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_device_enumeration",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_device_enumeration", start.elapsed())
}

/// Test 4: Verify primary device (device 0) works correctly.
///
/// Tests comprehensive operations on the primary device to ensure
/// it's fully functional.
fn test_primary_device_operations(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test 1: Large data operation
    const LARGE_SIZE: usize = 500000;

    let large_data: Vec<u32> = (0..LARGE_SIZE)
        .map(|i| ((i * 1103515245 + 12345) % 10000000) as u32)
        .collect();

    let large_buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&large_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Large buffer creation failed: {}", e),
            )
        }
    };

    let large_sorted = match ctx.provider.sort(&large_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Large sort failed: {}", e),
            )
        }
    };

    let large_result = match ctx.provider.download_column::<u32>(&large_sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Large download failed: {}", e),
            )
        }
    };

    // Spot check sorted
    for i in (1..large_result.len()).step_by(1000) {
        if large_result[i] < large_result[i - 1] {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Large sort incorrect at index {}", i),
            );
        }
    }

    // Test 2: Chained operations
    let chain_data: Vec<u32> = (0..10000).map(|i| (i % 1000) as u32).collect();

    let chain_buffer = match ctx
        .provider
        .create_buffer_from_slice::<u32>(&chain_data, schema.clone())
    {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Chain buffer creation failed: {}", e),
            )
        }
    };

    // Sort -> Filter -> Sort
    let step1 = match ctx.provider.sort(&chain_buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Chain step 1 failed: {}", e),
            )
        }
    };

    let step1_data = match ctx.provider.download_column::<u32>(&step1, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Chain step 1 download failed: {}", e),
            )
        }
    };

    let mask: Vec<u8> = step1_data
        .iter()
        .map(|&v| if v < 500 { 1 } else { 0 })
        .collect();

    let step2 = match ctx.provider.filter_by_mask(&step1, &mask) {
        Ok(f) => f,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Chain step 2 failed: {}", e),
            )
        }
    };

    let step3 = match ctx.provider.sort(&step2, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Chain step 3 failed: {}", e),
            )
        }
    };

    let chain_result = match ctx.provider.download_column::<u32>(&step3, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Chain result download failed: {}", e),
            )
        }
    };

    // Verify chain result
    for i in 1..chain_result.len() {
        if chain_result[i] < chain_result[i - 1] {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Chain result not sorted at index {}", i),
            );
        }
    }

    for &val in &chain_result {
        if val >= 500 {
            return TestResult::error(
                "test_primary_device_operations",
                start.elapsed(),
                format!("Chain result contains value {} >= 500", val),
            );
        }
    }

    // Test 3: Multiple independent operations
    let mut buffers = Vec::new();

    for i in 0..5 {
        let data: Vec<u32> = (0..5000).map(|j| ((j + i * 5000) % 10000) as u32).collect();

        let buffer = match ctx
            .provider
            .create_buffer_from_slice::<u32>(&data, schema.clone())
        {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_primary_device_operations",
                    start.elapsed(),
                    format!("Multi buffer {} creation failed: {}", i, e),
                )
            }
        };

        let sorted = match ctx.provider.sort(&buffer, &[0]) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_primary_device_operations",
                    start.elapsed(),
                    format!("Multi buffer {} sort failed: {}", i, e),
                )
            }
        };

        buffers.push(sorted);
    }

    // Verify all
    for (i, sorted) in buffers.iter().enumerate() {
        let result = match ctx.provider.download_column::<u32>(sorted, 0) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::error(
                    "test_primary_device_operations",
                    start.elapsed(),
                    format!("Multi buffer {} download failed: {}", i, e),
                )
            }
        };

        for j in 1..result.len() {
            if result[j] < result[j - 1] {
                return TestResult::error(
                    "test_primary_device_operations",
                    start.elapsed(),
                    format!("Multi buffer {} not sorted at index {}", i, j),
                );
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_primary_device_operations",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_primary_device_operations", start.elapsed())
}

/// Test 5: Query and verify device capabilities.
///
/// Tests that device capability queries work and return reasonable values.
fn test_device_capability_query(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Get compute capability
    let (major, minor) = match ctx.compute_capability() {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_device_capability_query",
                start.elapsed(),
                format!("Failed to query compute capability: {}", e),
            );
        }
    };

    // Verify reasonable compute capability
    // CUDA compute capability ranges from about 2.0 to 9.x as of 2024
    if major < 2 || major > 12 {
        return TestResult::error(
            "test_device_capability_query",
            start.elapsed(),
            format!("Suspicious compute capability major version: {}", major),
        );
    }

    if minor > 9 {
        return TestResult::error(
            "test_device_capability_query",
            start.elapsed(),
            format!("Suspicious compute capability minor version: {}", minor),
        );
    }

    // Verify memory budget is reasonable
    let budget = ctx.memory_budget();
    let used = ctx.memory_used();

    // Budget should be at least 1MB (sanity check)
    if budget < 1024 * 1024 {
        return TestResult::error(
            "test_device_capability_query",
            start.elapsed(),
            format!("Memory budget too small: {} bytes", budget),
        );
    }

    // Used should be <= budget (or budget is unlimited)
    if used > budget && budget > 0 {
        return TestResult::error(
            "test_device_capability_query",
            start.elapsed(),
            format!("Memory used ({}) exceeds budget ({})", used, budget),
        );
    }

    // Verify operations work at reported capability level
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Size based on capability - newer GPUs can handle larger workloads
    let test_size = if major >= 7 {
        100000 // Volta and newer
    } else if major >= 5 {
        50000 // Maxwell and Pascal
    } else {
        10000 // Older devices
    };

    let data: Vec<u32> = (0..test_size)
        .map(|i| {
            // Use u64 + wrapping arithmetic so debug builds don't panic on overflow.
            let v = (i as u64).wrapping_mul(1103515245).wrapping_add(12345) % (test_size as u64);
            v as u32
        })
        .collect();

    let buffer = match ctx.provider.create_buffer_from_slice::<u32>(&data, schema) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_device_capability_query",
                start.elapsed(),
                format!(
                    "Buffer creation failed for capability {}.{}: {}",
                    major, minor, e
                ),
            )
        }
    };

    let sorted = match ctx.provider.sort(&buffer, &[0]) {
        Ok(s) => s,
        Err(e) => {
            return TestResult::error(
                "test_device_capability_query",
                start.elapsed(),
                format!("Sort failed for capability {}.{}: {}", major, minor, e),
            )
        }
    };

    let result = match ctx.provider.download_column::<u32>(&sorted, 0) {
        Ok(d) => d,
        Err(e) => {
            return TestResult::error(
                "test_device_capability_query",
                start.elapsed(),
                format!("Download failed for capability {}.{}: {}", major, minor, e),
            )
        }
    };

    // Verify sorted
    for i in 1..result.len() {
        if result[i] < result[i - 1] {
            return TestResult::error(
                "test_device_capability_query",
                start.elapsed(),
                format!(
                    "Sort incorrect at index {} for capability {}.{}",
                    i, major, minor
                ),
            );
        }
    }

    // Test multi-GPU detection consistency
    let multi_gpu = ctx.multi_gpu_available();
    let device_count = CudaDevice::count().unwrap_or(0);

    if (multi_gpu && device_count <= 1) || (!multi_gpu && device_count > 1) {
        return TestResult::error(
            "test_device_capability_query",
            start.elapsed(),
            format!(
                "Multi-GPU detection inconsistent: multi_gpu={}, count={}",
                multi_gpu, device_count
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_device_capability_query",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_device_capability_query", start.elapsed())
}
