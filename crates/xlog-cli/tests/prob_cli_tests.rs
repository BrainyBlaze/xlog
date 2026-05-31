#![cfg(feature = "host-io")]

use assert_cmd::Command;
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
    // GPU-resident MC over a recursive program (supported fragment). The prior
    // `04-nonmonotone-mc.xlog` used negation, which the resident engine now
    // rejects fail-closed (no host-orchestrated fallback).
    let mc_program = repo_root.join("examples/prob/04-recursive-mc.xlog");

    // Use Command::cargo_bin which resolves via CARGO_BIN_EXE_xlog,
    // inheriting the same feature flags (including host-io) from the test build.
    let mut cmd = Command::cargo_bin("xlog").expect("xlog binary");
    cmd.args([
        "prob",
        exact_program.to_str().expect("valid path"),
        "--prob-engine",
        "exact_ddnnf",
    ]);
    cmd.assert().success();

    let mut cmd = Command::cargo_bin("xlog").expect("xlog binary");
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

#[test]
fn test_xlog_prob_mc_pragmas_json_and_cli_overrides() {
    if CudaDevice::new(0).is_err() {
        eprintln!("Skipping test: CUDA runtime unavailable");
        return;
    }

    let program =
        std::env::temp_dir().join(format!("xlog_v085_approx_cli_{}.xlog", std::process::id()));
    std::fs::write(
        &program,
        r#"
#pragma prob_engine = mc
#pragma prob_samples = 8
#pragma prob_seed = 1
#pragma prob_confidence = 0.80
#pragma prob_method = rejection
0.5::rain().
query(rain()).
"#,
    )
    .expect("write approximate CLI fixture");

    let mut cmd = Command::cargo_bin("xlog").expect("xlog binary");
    let output = cmd
        .args([
            "prob",
            program.to_str().expect("valid path"),
            "--samples",
            "16",
            "--seed",
            "2",
            "--confidence",
            "0.90",
            "--prob-method",
            "rejection",
            "--output",
            "json",
        ])
        .output()
        .expect("run xlog prob json");
    assert!(
        output.status.success(),
        "xlog prob failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("\"engine\": \"mc\""), "{stdout}");
    assert!(stdout.contains("\"total_samples\": 16"), "{stdout}");
    assert!(stdout.contains("\"seed\": 2"), "{stdout}");
    assert!(stdout.contains("\"confidence\": 0.9"), "{stdout}");
    assert!(
        stdout.contains("\"sampling_method\": \"rejection\""),
        "{stdout}"
    );
    assert!(stdout.contains("\"stderr\""), "{stdout}");
    assert!(stdout.contains("\"ci_low\""), "{stdout}");
    assert!(stdout.contains("\"evidence_samples\""), "{stdout}");
}
