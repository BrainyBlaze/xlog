//! Certification-suite CUDA memory snapshots for the peak VRAM gate.

use cudarc::driver::result::mem_get_info;
use xlog_cuda_tests::harness::{CategoryResult, TestContext};
use xlog_cuda_tests::{categories, CertificationResults};

const VRAM_GATE_BYTES: u64 = 38 * 1024 * 1024 * 1024;

#[derive(Debug)]
struct VramTracker {
    baseline_free: u64,
    total: u64,
    min_free: u64,
    min_label: &'static str,
}

impl VramTracker {
    fn new() -> Self {
        let (free, total) = mem_get_info().expect("cudaMemGetInfo before cert suite");
        Self {
            baseline_free: free as u64,
            total: total as u64,
            min_free: free as u64,
            min_label: "start",
        }
    }

    fn sample(&mut self, label: &'static str) {
        let (free, total) = mem_get_info().expect("cudaMemGetInfo during cert suite");
        let free = free as u64;
        let total = total as u64;
        assert_eq!(self.total, total, "CUDA total memory changed during cert");
        if free < self.min_free {
            self.min_free = free;
            self.min_label = label;
        }
        println!(
            "CERTIFICATION_VRAM_SNAPSHOT label={} free_bytes={} total_bytes={} delta_bytes={} gate_bytes={}",
            label,
            free,
            total,
            self.baseline_free.saturating_sub(free),
            VRAM_GATE_BYTES
        );
    }

    fn peak_delta_bytes(&self) -> u64 {
        self.baseline_free.saturating_sub(self.min_free)
    }
}

fn run_category(
    ctx: &TestContext,
    results: &mut CertificationResults,
    tracker: &mut VramTracker,
    label: &'static str,
    run: fn(&TestContext) -> CategoryResult,
) {
    tracker.sample(label);
    let result = run(ctx);
    tracker.sample(label);
    ctx.reap_pending();
    ctx.sync_and_check()
        .expect("sync after certification category");
    tracker.sample(label);
    results.add_category(result);
}

#[test]
fn certification_suite_vram_snapshot_under_gate() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping certification-suite VRAM snapshot: {e}");
            return;
        }
    };

    let mut tracker = VramTracker::new();
    let mut results = CertificationResults::new();

    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "toolchain_ptx_sass",
        categories::c01_toolchain::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "launch_configuration",
        categories::c02_launch_config::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "pointer_indexing_bounds",
        categories::c03_pointer_bounds::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "address_space",
        categories::c04_address_space::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "global_memory_hazards",
        categories::c05_global_memory::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "shared_memory",
        categories::c06_shared_memory::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "local_memory_stack",
        categories::c07_local_memory::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "synchronization_ordering",
        categories::c08_synchronization::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "warp_level_execution",
        categories::c09_warp_level::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "block_grid_coordination",
        categories::c10_block_grid::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "control_flow_predication",
        categories::c11_control_flow::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "atomic_operations",
        categories::c12_atomics::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "floating_point",
        categories::c13_floating_point::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "integer_edge_cases",
        categories::c14_integer::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "determinism",
        categories::c15_determinism::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "async_pipeline",
        categories::c16_async_pipeline::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "caching_coherence",
        categories::c17_caching::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "host_device_integration",
        categories::c18_host_device::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "multi_stream_concurrency",
        categories::c19_multi_stream::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "multi_gpu",
        categories::c20_multi_gpu::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "hardware_reliability",
        categories::c21_hardware::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "algorithm_specific",
        categories::c22_algorithms::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "testing_blind_spots",
        categories::c23_blind_spots::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "edge_case_matrix",
        categories::c24_edge_matrix::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "float_filter",
        categories::c25_float_filter::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "circuit_forward_kernel",
        categories::g01_circuit_forward::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "circuit_backward_kernel",
        categories::g02_circuit_backward::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "gpu_weight_injection",
        categories::g03_weight_injection::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "transfer_efficiency",
        categories::g04_transfer_efficiency::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "circuit_cache",
        categories::g05_circuit_cache::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "ptx_robustness",
        categories::g06_ptx_robustness::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "gpu_cdcl_sat_unsat_verifier",
        categories::g07_sat_cdcl::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "device_resident_row_counts",
        categories::g08_device_counts::run_all,
    );

    results.finalize();
    results.print_summary();
    assert!(
        results.all_passed(),
        "certification categories failed during certification-suite VRAM snapshot"
    );

    let peak_delta = tracker.peak_delta_bytes();
    println!(
        "CERTIFICATION_VRAM_SNAPSHOT_PEAK label={} peak_delta_bytes={} gate_bytes={} total_bytes={}",
        tracker.min_label, peak_delta, VRAM_GATE_BYTES, tracker.total
    );
    assert!(
        peak_delta <= VRAM_GATE_BYTES,
        "cert VRAM peak delta {} exceeds gate {}",
        peak_delta,
        VRAM_GATE_BYTES
    );
}
