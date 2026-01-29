//! Full CUDA kernel certification suite.
//!
//! Run with: cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture
//!
//! Expected runtime: seconds to minutes (GPU-dependent; dominated by C21 hardware stress tests)

use xlog_cuda_tests::categories;
use xlog_cuda_tests::harness::TestContext;
use xlog_cuda_tests::CertificationResults;

#[test]
fn run_full_certification() {
    println!("\n========================================");
    println!("CUDA Kernel Full Certification Suite");
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
            eprintln!("Skipping certification suite - no CUDA device available");
            return;
        }
    };

    let mut results = CertificationResults::new();

    // Run all 33 categories sequentially (C01-C25 + G01-G08)
    println!("Running C01: Toolchain...");
    results.add_category(categories::c01_toolchain::run_all(&ctx));

    println!("Running C02: Launch Config...");
    results.add_category(categories::c02_launch_config::run_all(&ctx));

    println!("Running C03: Pointer Bounds...");
    results.add_category(categories::c03_pointer_bounds::run_all(&ctx));

    println!("Running C04: Address Space...");
    results.add_category(categories::c04_address_space::run_all(&ctx));

    println!("Running C05: Global Memory...");
    results.add_category(categories::c05_global_memory::run_all(&ctx));

    println!("Running C06: Shared Memory...");
    results.add_category(categories::c06_shared_memory::run_all(&ctx));

    println!("Running C07: Local Memory...");
    results.add_category(categories::c07_local_memory::run_all(&ctx));

    println!("Running C08: Synchronization...");
    results.add_category(categories::c08_synchronization::run_all(&ctx));

    println!("Running C09: Warp Level...");
    results.add_category(categories::c09_warp_level::run_all(&ctx));

    println!("Running C10: Block Grid...");
    results.add_category(categories::c10_block_grid::run_all(&ctx));

    println!("Running C11: Control Flow...");
    results.add_category(categories::c11_control_flow::run_all(&ctx));

    println!("Running C12: Atomics...");
    results.add_category(categories::c12_atomics::run_all(&ctx));

    println!("Running C13: Floating Point...");
    results.add_category(categories::c13_floating_point::run_all(&ctx));

    println!("Running C14: Integer...");
    results.add_category(categories::c14_integer::run_all(&ctx));

    println!("Running C15: Determinism...");
    results.add_category(categories::c15_determinism::run_all(&ctx));

    println!("Running C16: Async Pipeline...");
    results.add_category(categories::c16_async_pipeline::run_all(&ctx));

    println!("Running C17: Caching...");
    results.add_category(categories::c17_caching::run_all(&ctx));

    println!("Running C18: Host Device...");
    results.add_category(categories::c18_host_device::run_all(&ctx));

    println!("Running C19: Multi Stream...");
    results.add_category(categories::c19_multi_stream::run_all(&ctx));

    println!("Running C20: Multi GPU...");
    results.add_category(categories::c20_multi_gpu::run_all(&ctx));

    println!("Running C21: Hardware...");
    results.add_category(categories::c21_hardware::run_all(&ctx));

    println!("Running C22: Algorithms...");
    results.add_category(categories::c22_algorithms::run_all(&ctx));

    println!("Running C23: Blind Spots...");
    results.add_category(categories::c23_blind_spots::run_all(&ctx));

    println!("Running C24: Edge Matrix...");
    results.add_category(categories::c24_edge_matrix::run_all(&ctx));

    println!("Running C25: Float Filter...");
    results.add_category(categories::c25_float_filter::run_all(&ctx));

    println!("Running G01: Circuit Forward...");
    results.add_category(categories::g01_circuit_forward::run_all(&ctx));

    println!("Running G02: Circuit Backward...");
    results.add_category(categories::g02_circuit_backward::run_all(&ctx));

    println!("Running G03: Weight Injection...");
    results.add_category(categories::g03_weight_injection::run_all(&ctx));

    println!("Running G04: Transfer Efficiency...");
    results.add_category(categories::g04_transfer_efficiency::run_all(&ctx));

    println!("Running G05: Circuit Cache...");
    results.add_category(categories::g05_circuit_cache::run_all(&ctx));

    println!("Running G06: PTX Robustness...");
    results.add_category(categories::g06_ptx_robustness::run_all(&ctx));

    println!("Running G07: SAT/CDCL...");
    results.add_category(categories::g07_sat_cdcl::run_all(&ctx));

    println!("Running G08: Device Counts...");
    results.add_category(categories::g08_device_counts::run_all(&ctx));

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
        "Certification failed: {}/{} tests passed ({} failed, {} skipped)",
        results.total_passed(),
        results.total_tests(),
        results.total_failed(),
        results.total_skipped()
    );
}
