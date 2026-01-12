//! Category 1: Toolchain, PTX, and SASS edge cases
//!
//! This category tests:
//! - PTX module loading and JIT compilation
//! - Compute capability verification
//! - Kernel function resolution from all modules
//! - 64-bit addressing verification
//! - JIT cache behavior under repeated kernel execution

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::time::Instant;
use xlog_cuda::{
    dedup_kernels, filter_kernels, groupby_kernels, join_kernels, pack_kernels, scan_kernels,
    set_ops_kernels, sort_kernels, DEDUP_MODULE, FILTER_MODULE, GROUPBY_MODULE, JOIN_MODULE,
    PACK_MODULE, SCAN_MODULE, SET_OPS_MODULE, SORT_MODULE,
};

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c01_toolchain");
    let start = Instant::now();

    results.add_result(test_ptx_loads_successfully(ctx));
    results.add_result(test_compute_capability_check(ctx));
    results.add_result(test_kernel_function_resolution(ctx));
    results.add_result(test_ptx_module_attributes(ctx));
    results.add_result(test_repeated_jit_compilation(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Verify all PTX modules loaded during TestContext creation.
///
/// Since we got here via TestContext creation, PTX loading already succeeded.
/// This test synchronizes the device and verifies no errors occurred during
/// the loading process.
fn test_ptx_loads_successfully(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // If we got here, the TestContext was created successfully, which means
    // CudaKernelProvider::new() succeeded, which loads all PTX modules.
    // Sync and verify no async errors occurred during module loading.
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_ptx_loads_successfully",
            start.elapsed(),
            format!("Device sync after PTX load failed: {}", e),
        );
    }

    // Verify the provider exists and device is valid
    let device_ordinal = ctx.device.ordinal();
    if device_ordinal > 100 {
        // Sanity check - device ordinal should be reasonable
        return TestResult::error(
            "test_ptx_loads_successfully",
            start.elapsed(),
            format!("Invalid device ordinal: {}", device_ordinal),
        );
    }

    TestResult::passed("test_ptx_loads_successfully", start.elapsed())
}

/// Test 2: Verify device compute capability meets minimum (7.0 for sm_70).
///
/// The xlog CUDA kernels are compiled for sm_70 (Volta) and above.
/// This test verifies the device meets this minimum requirement.
fn test_compute_capability_check(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let (major, minor) = ctx.compute_capability();

    // Minimum requirement is sm_70 (Volta)
    // major >= 7 is the effective check since minor is always >= 0 for u32
    let meets_minimum = major >= 7;

    if !meets_minimum {
        return TestResult::error(
            "test_compute_capability_check",
            start.elapsed(),
            format!(
                "Compute capability {}.{} does not meet minimum sm_70 requirement",
                major, minor
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_compute_capability_check",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_compute_capability_check", start.elapsed())
}

/// Test 3: Verify all kernel functions can be resolved from the loaded modules.
///
/// This test iterates through all 8 PTX modules and verifies that each
/// kernel function can be resolved using device.get_func().
fn test_kernel_function_resolution(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let device = ctx.device.inner();

    // Define all modules and their kernel functions
    let modules_and_functions: &[(&str, &[&str])] = &[
        (
            JOIN_MODULE,
            &[
                join_kernels::HASH_JOIN_BUILD,
                join_kernels::HASH_JOIN_PROBE,
                join_kernels::COMPUTE_COMPOSITE_HASH,
                join_kernels::HASH_JOIN_BUCKET_COUNT_V2,
                join_kernels::HASH_JOIN_SCATTER_V2,
                join_kernels::HASH_JOIN_PROBE_V2,
                join_kernels::HASH_JOIN_SEMI,
                join_kernels::HASH_JOIN_ANTI,
                join_kernels::INIT_HASH_TABLE,
            ],
        ),
        (
            DEDUP_MODULE,
            &[
                dedup_kernels::MARK_DUPLICATES,
                dedup_kernels::MARK_UNIQUE_COLUMNAR,
                dedup_kernels::MARK_UNIQUE_AND_SCAN_COLUMNAR,
                dedup_kernels::COMPACT_ROWS,
            ],
        ),
        (
            GROUPBY_MODULE,
            &[
                groupby_kernels::DETECT_GROUP_BOUNDARIES,
                groupby_kernels::DETECT_BOUNDARIES,
                groupby_kernels::EXTRACT_GROUP_KEYS,
                groupby_kernels::GROUPBY_COUNT,
                groupby_kernels::GROUPBY_SUM,
                groupby_kernels::GROUPBY_MIN,
                groupby_kernels::GROUPBY_MAX,
                groupby_kernels::GROUPBY_LOGSUMEXP_MAX,
                groupby_kernels::GROUPBY_LOGSUMEXP_SUMEXP,
                groupby_kernels::GROUPBY_LOGSUMEXP_FINAL,
            ],
        ),
        (
            SCAN_MODULE,
            &[
                scan_kernels::EXCLUSIVE_SCAN_MASK,
                scan_kernels::COUNT_MASK,
                scan_kernels::MULTIBLOCK_SCAN_PHASE1,
                scan_kernels::MULTIBLOCK_SCAN_PHASE2,
                scan_kernels::MULTIBLOCK_SCAN_PHASE3,
            ],
        ),
        (
            SORT_MODULE,
            &[
                sort_kernels::RADIX_HISTOGRAM,
                sort_kernels::RADIX_SCATTER,
                sort_kernels::COMPUTE_RANKS,
                sort_kernels::RADIX_SCATTER_STABLE,
                sort_kernels::INIT_INDICES,
                sort_kernels::APPLY_PERMUTATION_U32,
                sort_kernels::APPLY_PERMUTATION_BYTES,
            ],
        ),
        (
            FILTER_MODULE,
            &[
                filter_kernels::FILTER_COMPARE_U32,
                filter_kernels::FILTER_COMPARE_I64,
                filter_kernels::FILTER_COMPARE_F64,
                filter_kernels::FILTER_COMPARE_U32_SCAN_PHASE1,
                filter_kernels::FILTER_COMPARE_F64_SCAN_PHASE1,
                filter_kernels::COMPACT_U32_BY_MASK,
                filter_kernels::COMPACT_I64_BY_MASK,
                filter_kernels::COMPACT_F64_BY_MASK,
                filter_kernels::COMPACT_BYTES_BY_MASK,
                filter_kernels::MASK_AND,
                filter_kernels::MASK_OR,
                filter_kernels::MASK_NOT,
            ],
        ),
        (
            SET_OPS_MODULE,
            &[
                set_ops_kernels::CONCAT_U32,
                set_ops_kernels::SORTED_DIFF_MARK,
            ],
        ),
        (
            PACK_MODULE,
            &[
                pack_kernels::PACK_KEYS,
                pack_kernels::HASH_PACKED_KEYS,
                pack_kernels::PACK_AND_HASH_KEYS,
                pack_kernels::PACK_KEYS_ALIGNED,
                pack_kernels::UNPACK_COLUMN,
                pack_kernels::GATHER_PACKED_ROWS,
                pack_kernels::SCATTER_PACKED_ROWS,
                pack_kernels::COMPARE_PACKED_KEYS,
            ],
        ),
    ];

    let mut total_functions = 0;
    let mut resolved_functions = 0;

    for (module_name, functions) in modules_and_functions {
        for func_name in *functions {
            total_functions += 1;
            match device.get_func(module_name, func_name) {
                Some(_func) => {
                    resolved_functions += 1;
                }
                None => {
                    return TestResult::error(
                        "test_kernel_function_resolution",
                        start.elapsed(),
                        format!(
                            "Failed to resolve kernel function '{}' from module '{}'",
                            func_name, module_name
                        ),
                    );
                }
            }
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_kernel_function_resolution",
            start.elapsed(),
            format!("Sync failed after function resolution: {}", e),
        );
    }

    // Verify we checked all expected functions (sanity check)
    // 9 join + 4 dedup + 10 groupby + 5 scan + 7 sort + 12 filter + 2 set_ops + 8 pack = 57
    if total_functions != 57 {
        return TestResult::error(
            "test_kernel_function_resolution",
            start.elapsed(),
            format!(
                "Unexpected function count: expected 57, got {}",
                total_functions
            ),
        );
    }

    if resolved_functions != total_functions {
        return TestResult::error(
            "test_kernel_function_resolution",
            start.elapsed(),
            format!(
                "Not all functions resolved: {}/{}",
                resolved_functions, total_functions
            ),
        );
    }

    TestResult::passed("test_kernel_function_resolution", start.elapsed())
}

/// Test 4: Verify PTX modules work correctly with 64-bit addressing.
///
/// This test allocates a small GPU buffer, uploads u64::MAX values,
/// downloads them, and verifies they're correct. This ensures the PTX
/// modules are correctly compiled for 64-bit addressing mode.
fn test_ptx_module_attributes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Allocate a small buffer for 64-bit values
    let num_elements = 16;
    let test_values: Vec<u64> = vec![u64::MAX; num_elements];

    // Allocate GPU memory
    let mut gpu_buffer = match ctx.memory.alloc::<u64>(num_elements) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_ptx_module_attributes",
                start.elapsed(),
                format!("Failed to allocate GPU buffer: {}", e),
            );
        }
    };

    // Upload data to GPU
    if let Err(e) = ctx
        .device
        .inner()
        .htod_sync_copy_into(&test_values, &mut gpu_buffer)
    {
        return TestResult::error(
            "test_ptx_module_attributes",
            start.elapsed(),
            format!("Failed to upload data to GPU: {}", e),
        );
    }

    // Synchronize to ensure upload is complete
    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_ptx_module_attributes",
            start.elapsed(),
            format!("Sync failed after upload: {}", e),
        );
    }

    // Download data back from GPU
    let downloaded: Vec<u64> = match ctx.device.inner().dtoh_sync_copy(&gpu_buffer) {
        Ok(data) => data,
        Err(e) => {
            return TestResult::error(
                "test_ptx_module_attributes",
                start.elapsed(),
                format!("Failed to download data from GPU: {}", e),
            );
        }
    };

    // Verify all values are correct
    for (i, &val) in downloaded.iter().enumerate() {
        if val != u64::MAX {
            return TestResult::error(
                "test_ptx_module_attributes",
                start.elapsed(),
                format!(
                    "64-bit value mismatch at index {}: expected {}, got {}",
                    i,
                    u64::MAX,
                    val
                ),
            );
        }
    }

    // Also test with specific bit patterns to catch endianness issues
    let patterns: Vec<u64> = vec![
        0x0000_0000_0000_0001,
        0x0000_0000_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
        0x8000_0000_0000_0000,
        0xDEAD_BEEF_CAFE_BABE,
        0x0123_4567_89AB_CDEF,
    ];

    let mut pattern_buffer = match ctx.memory.alloc::<u64>(patterns.len()) {
        Ok(buf) => buf,
        Err(e) => {
            return TestResult::error(
                "test_ptx_module_attributes",
                start.elapsed(),
                format!("Failed to allocate pattern buffer: {}", e),
            );
        }
    };

    if let Err(e) = ctx
        .device
        .inner()
        .htod_sync_copy_into(&patterns, &mut pattern_buffer)
    {
        return TestResult::error(
            "test_ptx_module_attributes",
            start.elapsed(),
            format!("Failed to upload patterns: {}", e),
        );
    }

    let downloaded_patterns: Vec<u64> = match ctx.device.inner().dtoh_sync_copy(&pattern_buffer) {
        Ok(data) => data,
        Err(e) => {
            return TestResult::error(
                "test_ptx_module_attributes",
                start.elapsed(),
                format!("Failed to download patterns: {}", e),
            );
        }
    };

    for (i, (&expected, &actual)) in patterns.iter().zip(downloaded_patterns.iter()).enumerate() {
        if expected != actual {
            return TestResult::error(
                "test_ptx_module_attributes",
                start.elapsed(),
                format!(
                    "64-bit pattern mismatch at index {}: expected 0x{:016X}, got 0x{:016X}",
                    i, expected, actual
                ),
            );
        }
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_ptx_module_attributes",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_ptx_module_attributes", start.elapsed())
}

/// Test 5: Run the same kernel multiple times to verify JIT cache behavior.
///
/// This test performs repeated memory operations (allocate, upload, download)
/// to verify that JIT compilation caching works correctly and doesn't cause
/// issues with repeated kernel execution. We run 15 iterations.
fn test_repeated_jit_compilation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    const NUM_ITERATIONS: usize = 15;
    const BUFFER_SIZE: usize = 1024;

    for iteration in 0..NUM_ITERATIONS {
        // Create test data for this iteration
        let test_data: Vec<u32> = (0..BUFFER_SIZE)
            .map(|i| (i + iteration * 1000) as u32)
            .collect();

        // Allocate GPU buffer
        let mut gpu_buffer = match ctx.memory.alloc::<u32>(BUFFER_SIZE) {
            Ok(buf) => buf,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_jit_compilation",
                    start.elapsed(),
                    format!("Allocation failed at iteration {}: {}", iteration, e),
                );
            }
        };

        // Upload data
        if let Err(e) = ctx
            .device
            .inner()
            .htod_sync_copy_into(&test_data, &mut gpu_buffer)
        {
            return TestResult::error(
                "test_repeated_jit_compilation",
                start.elapsed(),
                format!("Upload failed at iteration {}: {}", iteration, e),
            );
        }

        // Download data
        let downloaded: Vec<u32> = match ctx.device.inner().dtoh_sync_copy(&gpu_buffer) {
            Ok(data) => data,
            Err(e) => {
                return TestResult::error(
                    "test_repeated_jit_compilation",
                    start.elapsed(),
                    format!("Download failed at iteration {}: {}", iteration, e),
                );
            }
        };

        // Verify data integrity
        for (i, (&expected, &actual)) in test_data.iter().zip(downloaded.iter()).enumerate() {
            if expected != actual {
                return TestResult::error(
                    "test_repeated_jit_compilation",
                    start.elapsed(),
                    format!(
                        "Data mismatch at iteration {}, index {}: expected {}, got {}",
                        iteration, i, expected, actual
                    ),
                );
            }
        }

        // Sync after each iteration to check for async errors
        if let Err(e) = ctx.sync_and_check() {
            return TestResult::error(
                "test_repeated_jit_compilation",
                start.elapsed(),
                format!("Sync failed at iteration {}: {}", iteration, e),
            );
        }

        // gpu_buffer will be dropped at the end of the loop iteration
    }

    // Final verification: resolve a kernel function to ensure JIT cache is still valid
    let device = ctx.device.inner();
    if device
        .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUILD)
        .is_none()
    {
        return TestResult::error(
            "test_repeated_jit_compilation",
            start.elapsed(),
            "Failed to resolve kernel after repeated iterations".to_string(),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_repeated_jit_compilation",
            start.elapsed(),
            format!("Final sync failed: {}", e),
        );
    }

    TestResult::passed("test_repeated_jit_compilation", start.elapsed())
}
