use assert_cmd::cargo::cargo_bin_cmd;
use cudarc::driver::result::mem_get_info;
use std::path::Path;
use xlog_cuda::CudaDevice;

#[test]
fn test_xlog_run_basic() {
    // cudarc panics on init when CUDA driver/runtime is unavailable; use xlog-cuda's safe wrapper.
    // Keep _device alive so the CUDA context survives through mem_get_info().
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    // CUDA context is alive via _device — memory query failure is now unexpected.
    let (_free, total) = mem_get_info()
        .expect("mem_get_info should succeed while CudaDevice is alive");

    let total_mb = total / (1024 * 1024);
    if total_mb < 16_384 {
        println!(
            "SKIPPED: GPU memory {} MB < required 16384 MB",
            total_mb
        );
        return;
    }

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let program = repo_root.join("examples/xlog/00-basics/01_tc_reachability.xlog");

    let mut cmd = cargo_bin_cmd!("xlog");
    cmd.args([
        "run",
        program.to_str().expect("valid path"),
        "--memory-mb",
        "16384",
    ]);
    cmd.assert().success();
}
