//! Quick smoke test for CI validation.
//!
//! Run with: cargo test -p xlog-cuda-tests --test quick_smoke --release -- --nocapture
//!
//! Expected runtime: sub-second to seconds (GPU-dependent)
//!
//! This runs a subset of key tests for quick validation:
//! - c01_toolchain: infrastructure check
//! - c02_launch_config: basic operations
//! - c04_address_space: type coverage
//! - c08_synchronization: correctness
//! - c11_control_flow: filter operations
//! - c15_determinism: reproducibility

use xlog_cuda_tests::categories;
use xlog_cuda_tests::harness::TestContext;
use xlog_cuda_tests::CertificationResults;

#[test]
fn run_quick_smoke() {
    println!("\n========================================");
    println!("CUDA Kernel Quick Smoke Test");
    println!("========================================\n");

    // Create test context - fail gracefully if no CUDA device
    let ctx = match TestContext::new() {
        Ok(ctx) => {
            println!("CUDA device initialized successfully");
            println!("Memory budget: {} MB", ctx.memory_budget() / (1024 * 1024));
            match ctx.compute_capability() {
                Ok((major, minor)) => println!("Compute capability: {}.{}", major, minor),
                Err(e) => println!("Compute capability: <unavailable> ({})", e),
            }
            println!();
            ctx
        }
        Err(e) => {
            eprintln!("Failed to create test context: {}", e);
            eprintln!("Skipping smoke test - no CUDA device available");
            return;
        }
    };

    let mut results = CertificationResults::new();

    // Run subset of key categories for quick validation
    println!("Running C01: Toolchain (infrastructure check)...");
    results.add_category(categories::c01_toolchain::run_all(&ctx));

    println!("Running C02: Launch Config (basic operations)...");
    results.add_category(categories::c02_launch_config::run_all(&ctx));

    println!("Running C04: Address Space (type coverage)...");
    results.add_category(categories::c04_address_space::run_all(&ctx));

    println!("Running C08: Synchronization (correctness)...");
    results.add_category(categories::c08_synchronization::run_all(&ctx));

    println!("Running C11: Control Flow (filter operations)...");
    results.add_category(categories::c11_control_flow::run_all(&ctx));

    println!("Running C15: Determinism (reproducibility)...");
    results.add_category(categories::c15_determinism::run_all(&ctx));

    // Finalize and print results
    results.finalize();
    results.print_summary();

    // Print detailed failure report if any failures occurred
    if !results.all_passed() {
        let failure_report = results.failure_report();
        if !failure_report.is_empty() {
            println!("\n========== FAILURE DETAILS ==========");
            println!("{}", failure_report);
        }
    }

    // Assert all tests passed
    assert!(
        results.all_passed(),
        "Smoke test failed: {}/{} tests passed ({} failed, {} skipped)",
        results.total_passed(),
        results.total_tests(),
        results.total_failed(),
        results.total_skipped()
    );
}
