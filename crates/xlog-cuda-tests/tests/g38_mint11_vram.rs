//! Goal-038 M_INT.11 CUDA memory snapshots for the certification suite.

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
            "G38_MINT11_CERT_VRAM label={} free_bytes={} total_bytes={} delta_bytes={} gate_bytes={}",
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
fn g38_mint11_cert_suite_vram_under_gate() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping G38 M_INT.11 cert VRAM snapshot: {e}");
            return;
        }
    };

    let mut tracker = VramTracker::new();
    let mut results = CertificationResults::new();

    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C01_toolchain",
        categories::c01_toolchain::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C02_launch_config",
        categories::c02_launch_config::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C03_pointer_bounds",
        categories::c03_pointer_bounds::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C04_address_space",
        categories::c04_address_space::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C05_global_memory",
        categories::c05_global_memory::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C06_shared_memory",
        categories::c06_shared_memory::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C07_local_memory",
        categories::c07_local_memory::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C08_synchronization",
        categories::c08_synchronization::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C09_warp_level",
        categories::c09_warp_level::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C10_block_grid",
        categories::c10_block_grid::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C11_control_flow",
        categories::c11_control_flow::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C12_atomics",
        categories::c12_atomics::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C13_floating_point",
        categories::c13_floating_point::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C14_integer",
        categories::c14_integer::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C15_determinism",
        categories::c15_determinism::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C16_async_pipeline",
        categories::c16_async_pipeline::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C17_caching",
        categories::c17_caching::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C18_host_device",
        categories::c18_host_device::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C19_multi_stream",
        categories::c19_multi_stream::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C20_multi_gpu",
        categories::c20_multi_gpu::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C21_hardware",
        categories::c21_hardware::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C22_algorithms",
        categories::c22_algorithms::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C23_blind_spots",
        categories::c23_blind_spots::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C24_edge_matrix",
        categories::c24_edge_matrix::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "C25_float_filter",
        categories::c25_float_filter::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "G01_circuit_forward",
        categories::g01_circuit_forward::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "G02_circuit_backward",
        categories::g02_circuit_backward::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "G03_weight_injection",
        categories::g03_weight_injection::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "G04_transfer_efficiency",
        categories::g04_transfer_efficiency::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "G05_circuit_cache",
        categories::g05_circuit_cache::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "G06_ptx_robustness",
        categories::g06_ptx_robustness::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "G07_sat_cdcl",
        categories::g07_sat_cdcl::run_all,
    );
    run_category(
        &ctx,
        &mut results,
        &mut tracker,
        "G08_device_counts",
        categories::g08_device_counts::run_all,
    );

    results.finalize();
    results.print_summary();
    assert!(
        results.all_passed(),
        "certification categories failed during M_INT.11 VRAM snapshot"
    );

    let peak_delta = tracker.peak_delta_bytes();
    println!(
        "G38_MINT11_CERT_VRAM_PEAK label={} peak_delta_bytes={} gate_bytes={} total_bytes={}",
        tracker.min_label, peak_delta, VRAM_GATE_BYTES, tracker.total
    );
    assert!(
        peak_delta <= VRAM_GATE_BYTES,
        "cert VRAM peak delta {} exceeds gate {}",
        peak_delta,
        VRAM_GATE_BYTES
    );
}
