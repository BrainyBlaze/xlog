//! Run individual category tests in isolation.
//!
//! Run with: cargo test -p xlog-cuda-tests --test category_isolation -- c01
//!
//! Available categories: c01-c24
//!
//! Examples:
//!   cargo test -p xlog-cuda-tests --test category_isolation c01 --release -- --nocapture
//!   cargo test -p xlog-cuda-tests --test category_isolation c15 --release -- --nocapture

use std::time::Instant;
use xlog_cuda_tests::categories;
use xlog_cuda_tests::harness::{CategoryResult, TestContext};

/// Helper function to run a single category and report results.
fn run_category<F>(category_name: &str, runner: F)
where
    F: FnOnce(&TestContext) -> CategoryResult,
{
    println!("\n========================================");
    println!("Running Category: {}", category_name);
    println!("========================================\n");

    // Create test context - fail gracefully if no CUDA device
    let ctx = match TestContext::new() {
        Ok(ctx) => {
            println!("CUDA device initialized successfully");
            println!(
                "Memory budget: {} MB",
                ctx.memory_budget() / (1024 * 1024)
            );
            println!(
                "Compute capability: {}.{}",
                ctx.compute_capability().0,
                ctx.compute_capability().1
            );
            println!();
            ctx
        }
        Err(e) => {
            eprintln!("Failed to create test context: {}", e);
            eprintln!("Skipping {} - no CUDA device available", category_name);
            return;
        }
    };

    let start = Instant::now();
    let result = runner(&ctx);
    let duration = start.elapsed();

    // Print individual test results
    println!("\n--- Test Results ---");
    for test in &result.tests {
        let status_str = if test.status.is_passed() {
            "PASS"
        } else if test.status.is_failed() {
            "FAIL"
        } else {
            "SKIP"
        };
        println!(
            "[{}] {} ({:.3}s)",
            status_str,
            test.name,
            test.duration.as_secs_f64()
        );

        // Print diagnostic for failed tests
        if let Some(diag) = &test.diagnostic {
            println!("       Error: {}", diag.error_message);
        }
        if let xlog_cuda_tests::TestStatus::Error { message } = &test.status {
            println!("       Error: {}", message);
        }
    }

    // Print summary
    println!("\n--- Summary ---");
    println!("Category: {}", result.name);
    println!("Duration: {:.2}s", duration.as_secs_f64());
    println!(
        "Passed: {}/{}",
        result.passed_count(),
        result.total_count()
    );
    println!("Failed: {}", result.failed_count());
    println!("Skipped: {}", result.skipped_count());
    println!("========================================\n");

    // Assert all tests passed
    assert!(
        result.all_passed(),
        "Category {} failed: {}/{} tests passed",
        category_name,
        result.passed_count(),
        result.total_count()
    );
}

#[test]
fn c01_toolchain() {
    run_category("c01_toolchain", categories::c01_toolchain::run_all);
}

#[test]
fn c02_launch_config() {
    run_category("c02_launch_config", categories::c02_launch_config::run_all);
}

#[test]
fn c03_pointer_bounds() {
    run_category("c03_pointer_bounds", categories::c03_pointer_bounds::run_all);
}

#[test]
fn c04_address_space() {
    run_category("c04_address_space", categories::c04_address_space::run_all);
}

#[test]
fn c05_global_memory() {
    run_category("c05_global_memory", categories::c05_global_memory::run_all);
}

#[test]
fn c06_shared_memory() {
    run_category("c06_shared_memory", categories::c06_shared_memory::run_all);
}

#[test]
fn c07_local_memory() {
    run_category("c07_local_memory", categories::c07_local_memory::run_all);
}

#[test]
fn c08_synchronization() {
    run_category(
        "c08_synchronization",
        categories::c08_synchronization::run_all,
    );
}

#[test]
fn c09_warp_level() {
    run_category("c09_warp_level", categories::c09_warp_level::run_all);
}

#[test]
fn c10_block_grid() {
    run_category("c10_block_grid", categories::c10_block_grid::run_all);
}

#[test]
fn c11_control_flow() {
    run_category("c11_control_flow", categories::c11_control_flow::run_all);
}

#[test]
fn c12_atomics() {
    run_category("c12_atomics", categories::c12_atomics::run_all);
}

#[test]
fn c13_floating_point() {
    run_category(
        "c13_floating_point",
        categories::c13_floating_point::run_all,
    );
}

#[test]
fn c14_integer() {
    run_category("c14_integer", categories::c14_integer::run_all);
}

#[test]
fn c15_determinism() {
    run_category("c15_determinism", categories::c15_determinism::run_all);
}

#[test]
fn c16_async_pipeline() {
    run_category(
        "c16_async_pipeline",
        categories::c16_async_pipeline::run_all,
    );
}

#[test]
fn c17_caching() {
    run_category("c17_caching", categories::c17_caching::run_all);
}

#[test]
fn c18_host_device() {
    run_category("c18_host_device", categories::c18_host_device::run_all);
}

#[test]
fn c19_multi_stream() {
    run_category("c19_multi_stream", categories::c19_multi_stream::run_all);
}

#[test]
fn c20_multi_gpu() {
    run_category("c20_multi_gpu", categories::c20_multi_gpu::run_all);
}

#[test]
fn c21_hardware() {
    run_category("c21_hardware", categories::c21_hardware::run_all);
}

#[test]
fn c22_algorithms() {
    run_category("c22_algorithms", categories::c22_algorithms::run_all);
}

#[test]
fn c23_blind_spots() {
    run_category("c23_blind_spots", categories::c23_blind_spots::run_all);
}

#[test]
fn c24_edge_matrix() {
    run_category("c24_edge_matrix", categories::c24_edge_matrix::run_all);
}
