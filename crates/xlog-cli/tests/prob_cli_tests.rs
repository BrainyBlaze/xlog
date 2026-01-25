use assert_cmd::cargo::cargo_bin_cmd;
use std::path::Path;
use xlog_cuda::CudaDevice;

#[test]
fn test_xlog_prob_exact_and_mc() {
    // cudarc panics on init when CUDA driver/runtime is unavailable; use xlog-cuda's safe wrapper.
    if CudaDevice::new(0).is_err() {
        eprintln!("Skipping test: CUDA runtime unavailable");
        return;
    }

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let exact_program = repo_root.join("examples/prob/01-wet-conditioning.xlog");
    let mc_program = repo_root.join("examples/prob/04-nonmonotone-mc.xlog");

    let mut cmd = cargo_bin_cmd!("xlog");
    cmd.args([
        "prob",
        exact_program.to_str().expect("valid path"),
        "--prob-engine",
        "exact_ddnnf",
    ]);
    cmd.assert().success();

    let mut cmd = cargo_bin_cmd!("xlog");
    cmd.args([
        "prob",
        mc_program.to_str().expect("valid path"),
        "--prob-engine",
        "mc",
        "--samples",
        "1000",
        "--seed",
        "42",
    ]);
    cmd.assert().success();
}
