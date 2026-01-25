use assert_cmd::cargo::cargo_bin_cmd;
use std::path::Path;
use xlog_cuda::CudaDevice;

#[test]
fn test_xlog_run_basic() {
    // cudarc panics on init when CUDA driver/runtime is unavailable; use xlog-cuda's safe wrapper.
    if CudaDevice::new(0).is_err() {
        eprintln!("Skipping test: CUDA runtime unavailable");
        return;
    }

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let program = repo_root.join("examples/xlog/00-basics/01_tc_reachability.xlog");

    let mut cmd = cargo_bin_cmd!("xlog");
    cmd.args(["run", program.to_str().expect("valid path")]);
    cmd.assert().success();
}
