use assert_cmd::cargo::cargo_bin_cmd;
use cudarc::driver::CudaDevice;
use std::path::Path;

#[test]
fn test_xlog_run_basic() {
    if CudaDevice::count().unwrap_or(0) == 0 {
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
