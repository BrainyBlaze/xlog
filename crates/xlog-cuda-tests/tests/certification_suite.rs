//! Full CUDA kernel certification suite.
//!
//! Run with: cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture
//!
//! Expected runtime: seconds to minutes (GPU-dependent; dominated by hardware reliability stress tests)

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
            // Surface which allocator backend the context is
            // running on so a cert report makes the runtime
            // path unambiguous. The selection is driven by
            // `XLOG_USE_DEVICE_RUNTIME` at process start.
            let backend = if ctx.uses_device_runtime() {
                "device-runtime (AsyncCudaResource → LoggingResource → GlobalDeviceBudget)"
            } else {
                "legacy (cudarc-backed GpuMemoryManager::new)"
            };
            println!("Allocator backend: {}", backend);
            // List explicitly-set recorded-op env flags so the
            // report shows the dispatch surface the categories
            // will actually exercise. `XLOG_USE_RECORDED_CSM` is
            // included so the cert evidence is unambiguous about
            // CSM selection — even though `XLOG_USE_RECORDED_OPS`
            // implies CSM, the explicit flag's presence shows up
            // separately so a "runtime+recorded+CSM" run is
            // visibly distinct from a "runtime+recorded" run.
            let env_flag = |var: &str| {
                std::env::var(var)
                    .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True"))
                    .unwrap_or(false)
            };
            let explicit_flags: Vec<&str> = [
                ("XLOG_USE_RECORDED_OPS", "all"),
                ("XLOG_USE_RECORDED_FILTERS", "filters"),
                ("XLOG_USE_RECORDED_SORT", "sort"),
                ("XLOG_USE_RECORDED_DEDUP", "dedup"),
                ("XLOG_USE_RECORDED_GROUPBY", "groupby"),
                ("XLOG_USE_RECORDED_HASH_JOIN", "hash_join"),
                ("XLOG_USE_RECORDED_CSM", "csm"),
            ]
            .iter()
            .filter_map(|(var, label)| if env_flag(var) { Some(*label) } else { None })
            .collect();
            if explicit_flags.is_empty() {
                println!("Recorded-op dispatch (explicit): <none>");
            } else {
                println!(
                    "Recorded-op dispatch (explicit): {}",
                    explicit_flags.join(", ")
                );
            }
            // Synthesize a single cert-mode label keyed off the
            // EXPLICIT recorded-op env flags (not the implied
            // umbrella unlock). The three intended modes are:
            //
            //   * legacy/default          — no XLOG_USE_DEVICE_RUNTIME
            //   * runtime+recorded        — XLOG_USE_DEVICE_RUNTIME=1
            //                               + at least one recorded-op
            //                               flag (umbrella or specific)
            //                               but no explicit
            //                               XLOG_USE_RECORDED_CSM=1
            //   * runtime+recorded+CSM    — same as above PLUS the
            //                               explicit
            //                               XLOG_USE_RECORDED_CSM=1
            //
            // CSM is also implicitly active in the dispatch when
            // only the umbrella `XLOG_USE_RECORDED_OPS=1` is set
            // (see `CudaKernelProvider::use_recorded_csm_env`), but
            // the cert label keys off the EXPLICIT flag so the
            // evidence trail is unambiguous: a "runtime+recorded+CSM"
            // run is the one where the operator deliberately set
            // `XLOG_USE_RECORDED_CSM=1`. Set the explicit flag to
            // emit unambiguous CSM-mode evidence.
            let any_recorded = env_flag("XLOG_USE_RECORDED_OPS")
                || env_flag("XLOG_USE_RECORDED_FILTERS")
                || env_flag("XLOG_USE_RECORDED_SORT")
                || env_flag("XLOG_USE_RECORDED_DEDUP")
                || env_flag("XLOG_USE_RECORDED_GROUPBY")
                || env_flag("XLOG_USE_RECORDED_HASH_JOIN");
            let csm_explicit = env_flag("XLOG_USE_RECORDED_CSM");
            let cert_mode = match (ctx.uses_device_runtime(), any_recorded, csm_explicit) {
                (false, false, false) => "legacy/default",
                (true, true, true) => "runtime+recorded+CSM",
                (true, true, false) => "runtime+recorded",
                (true, false, false) => "runtime (no recorded-ops)",
                (true, false, true) => "runtime+CSM (no other recorded-ops)",
                (false, true, true) => "recorded+CSM (no device-runtime)",
                (false, true, false) => "recorded (no device-runtime)",
                (false, false, true) => "CSM-only (no device-runtime, no other recorded-ops)",
            };
            println!("Cert mode: {}", cert_mode);
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

    // Reap pending async frees between categories when running
    // against the device-runtime backend. Each category
    // allocates many short-lived buffers; without periodic
    // reap, the `GlobalDeviceBudget` reservation accumulates
    // because `cuMemFreeAsync` only releases real GPU memory
    // after stream completion. No-op on the legacy backend.
    let reap = || ctx.reap_pending();

    // Run all 33 categories sequentially across core CUDA infrastructure and GPU-specific coverage.
    println!("Running toolchain/PTX/SASS validation...");
    results.add_category(categories::c01_toolchain::run_all(&ctx));
    reap();

    println!("Running launch-configuration validation...");
    results.add_category(categories::c02_launch_config::run_all(&ctx));
    reap();

    println!("Running pointer, indexing, and bounds validation...");
    results.add_category(categories::c03_pointer_bounds::run_all(&ctx));
    reap();

    println!("Running address-space validation...");
    results.add_category(categories::c04_address_space::run_all(&ctx));
    reap();

    println!("Running global-memory hazard validation...");
    results.add_category(categories::c05_global_memory::run_all(&ctx));
    reap();

    println!("Running shared-memory validation...");
    results.add_category(categories::c06_shared_memory::run_all(&ctx));
    reap();

    println!("Running local-memory and stack validation...");
    results.add_category(categories::c07_local_memory::run_all(&ctx));
    reap();

    println!("Running synchronization and ordering validation...");
    results.add_category(categories::c08_synchronization::run_all(&ctx));
    reap();

    println!("Running warp-level execution validation...");
    results.add_category(categories::c09_warp_level::run_all(&ctx));
    reap();

    println!("Running block/grid coordination validation...");
    results.add_category(categories::c10_block_grid::run_all(&ctx));
    reap();

    println!("Running control-flow and predication validation...");
    results.add_category(categories::c11_control_flow::run_all(&ctx));
    reap();

    println!("Running atomic-operation validation...");
    results.add_category(categories::c12_atomics::run_all(&ctx));
    reap();

    println!("Running floating-point validation...");
    results.add_category(categories::c13_floating_point::run_all(&ctx));
    reap();

    println!("Running integer edge-case validation...");
    results.add_category(categories::c14_integer::run_all(&ctx));
    reap();

    println!("Running determinism validation...");
    results.add_category(categories::c15_determinism::run_all(&ctx));
    reap();

    println!("Running async-pipeline validation...");
    results.add_category(categories::c16_async_pipeline::run_all(&ctx));
    reap();

    println!("Running caching and coherence validation...");
    results.add_category(categories::c17_caching::run_all(&ctx));
    reap();

    println!("Running host-device integration validation...");
    results.add_category(categories::c18_host_device::run_all(&ctx));
    reap();

    println!("Running multi-stream concurrency validation...");
    results.add_category(categories::c19_multi_stream::run_all(&ctx));
    reap();

    println!("Running multi-GPU validation...");
    results.add_category(categories::c20_multi_gpu::run_all(&ctx));
    reap();

    println!("Running hardware-reliability validation...");
    results.add_category(categories::c21_hardware::run_all(&ctx));
    reap();

    println!("Running algorithm-specific validation...");
    results.add_category(categories::c22_algorithms::run_all(&ctx));
    reap();

    println!("Running testing blind-spot validation...");
    results.add_category(categories::c23_blind_spots::run_all(&ctx));
    reap();

    println!("Running edge-case matrix validation...");
    results.add_category(categories::c24_edge_matrix::run_all(&ctx));
    reap();

    println!("Running float-filter validation...");
    results.add_category(categories::c25_float_filter::run_all(&ctx));
    reap();

    println!("Running circuit forward-kernel validation...");
    results.add_category(categories::g01_circuit_forward::run_all(&ctx));
    reap();

    println!("Running circuit backward-kernel validation...");
    results.add_category(categories::g02_circuit_backward::run_all(&ctx));
    reap();

    println!("Running GPU weight-injection validation...");
    results.add_category(categories::g03_weight_injection::run_all(&ctx));
    reap();

    println!("Running transfer-efficiency validation...");
    results.add_category(categories::g04_transfer_efficiency::run_all(&ctx));
    reap();

    println!("Running circuit-cache validation...");
    results.add_category(categories::g05_circuit_cache::run_all(&ctx));
    reap();

    println!("Running PTX robustness validation...");
    results.add_category(categories::g06_ptx_robustness::run_all(&ctx));
    reap();

    println!("Running GPU CDCL SAT/UNSAT verifier validation...");
    results.add_category(categories::g07_sat_cdcl::run_all(&ctx));
    reap();

    println!("Running device-resident row-count validation...");
    results.add_category(categories::g08_device_counts::run_all(&ctx));
    reap();

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
