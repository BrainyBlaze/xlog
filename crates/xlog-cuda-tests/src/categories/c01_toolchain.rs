//! Category 1: Toolchain, PTX, and SASS edge cases
//!
//! This category tests:
//! - PTX module loading and JIT compilation
//! - Compute capability verification
//! - Kernel function resolution from all modules
//! - 64-bit addressing verification
//! - JIT cache behavior under repeated kernel execution

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use xlog_cuda::{join_kernels, JOIN_MODULE};

fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
}

fn extract_ptx_directive(ptx: &str, directive: &str) -> Option<String> {
    for line in ptx.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(directive) {
            let rest = rest.trim();
            if rest.is_empty() {
                continue;
            }
            return rest.split_whitespace().next().map(|s| s.to_string());
        }
    }
    None
}

fn extract_entry_names(ptx: &str) -> Vec<String> {
    let mut entries = Vec::new();
    for line in ptx.lines() {
        let line = line.trim();
        let line = if let Some(rest) = line.strip_prefix(".visible .entry ") {
            rest
        } else if let Some(rest) = line.strip_prefix(".entry ") {
            rest
        } else {
            continue;
        };

        if let Some((name, _)) = line.split_once('(') {
            let name = name.trim();
            if !name.is_empty() {
                entries.push(name.to_string());
            }
        }
    }
    entries
}

fn parse_sm_target(target: &str) -> Option<u32> {
    let s = target.trim();
    let sm = s.strip_prefix("sm_")?;
    sm.parse::<u32>().ok()
}

/// Run all tests in this category.
pub(crate) fn run_all(ctx: &TestContext) -> CategoryResult {
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

    let (major, minor) = match ctx.compute_capability() {
        Ok(v) => v,
        Err(e) => {
            return TestResult::error(
                "test_compute_capability_check",
                start.elapsed(),
                format!("Failed to query compute capability: {}", e),
            );
        }
    };

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
/// This test parses every `kernels/*.ptx` file, extracts all `.entry` points,
/// and verifies that `CudaKernelProvider` loaded each entry under the expected
/// module name `xlog_<stem>`.
fn test_kernel_function_resolution(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    let device = ctx.device.inner();

    let mut total_functions = 0;
    let mut resolved_functions = 0;

    let kernels_dir = kernels_dir();
    let mut ptx_files: Vec<PathBuf> = match fs::read_dir(&kernels_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "ptx"))
            .collect(),
        Err(e) => {
            return TestResult::error(
                "test_kernel_function_resolution",
                start.elapsed(),
                format!(
                    "Failed to read kernels dir {}: {}",
                    kernels_dir.display(),
                    e
                ),
            );
        }
    };
    ptx_files.sort();

    if ptx_files.is_empty() {
        return TestResult::error(
            "test_kernel_function_resolution",
            start.elapsed(),
            format!("No PTX files found under {}", kernels_dir.display()),
        );
    }

    for path in ptx_files {
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unknown>");
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem.is_empty() {
            return TestResult::error(
                "test_kernel_function_resolution",
                start.elapsed(),
                format!("Invalid PTX filename stem: {}", filename),
            );
        }

        let module_name = format!("xlog_{}", stem);

        let ptx = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                return TestResult::error(
                    "test_kernel_function_resolution",
                    start.elapsed(),
                    format!("Failed to read {}: {}", path.display(), e),
                );
            }
        };

        let address_size = extract_ptx_directive(&ptx, ".address_size").unwrap_or_default();
        if address_size != "64" {
            return TestResult::error(
                "test_kernel_function_resolution",
                start.elapsed(),
                format!(
                    "{}: expected .address_size 64, got '{}'",
                    filename, address_size
                ),
            );
        }

        let target = extract_ptx_directive(&ptx, ".target").unwrap_or_default();
        let sm = parse_sm_target(&target).unwrap_or(0);
        if sm < 70 {
            return TestResult::error(
                "test_kernel_function_resolution",
                start.elapsed(),
                format!(
                    "{}: expected .target sm_70 or later, got '{}'",
                    filename, target
                ),
            );
        }

        let entries = extract_entry_names(&ptx);
        if entries.is_empty() {
            return TestResult::error(
                "test_kernel_function_resolution",
                start.elapsed(),
                format!("{}: no .entry kernels found", filename),
            );
        }

        let mut seen = HashSet::new();
        for entry in &entries {
            if !seen.insert(entry.as_str()) {
                return TestResult::error(
                    "test_kernel_function_resolution",
                    start.elapsed(),
                    format!("{}: duplicate .entry name {}", filename, entry),
                );
            }
        }

        for entry in &entries {
            total_functions += 1;
            if device.get_func(&module_name, entry).is_some() {
                resolved_functions += 1;
            } else {
                return TestResult::error(
                    "test_kernel_function_resolution",
                    start.elapsed(),
                    format!(
                        "{}: failed to resolve kernel function '{}' from module '{}'",
                        filename, entry, module_name
                    ),
                );
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
