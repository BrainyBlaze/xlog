use assert_cmd::Command;
use cudarc::driver::CudaDevice;
use std::path::Path;

#[test]
fn test_xlog_prob_exact_and_mc() {
    if CudaDevice::count().unwrap_or(0) == 0 {
        return;
    }

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let exact_program = repo_root.join("examples/prob/01-wet-conditioning.xlog");
    let mc_program = repo_root.join("examples/prob/04-nonmonotone-mc.xlog");

    let mut cmd = Command::cargo_bin("xlog").unwrap();
    cmd.args([
        "prob",
        exact_program.to_str().expect("valid path"),
        "--prob-engine",
        "exact_ddnnf",
    ]);
    cmd.assert().success();

    let mut cmd = Command::cargo_bin("xlog").unwrap();
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
