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
    let (_free, total) =
        mem_get_info().expect("mem_get_info should succeed while CudaDevice is alive");

    let total_mb = total / (1024 * 1024);
    if total_mb < 16_384 {
        println!("SKIPPED: GPU memory {} MB < required 16384 MB", total_mb);
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

#[test]
fn test_xlog_run_epistemic_examples() {
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let examples = [
        ("01-eir-boundary.xlog", "believed", "| 1  |"),
        ("02-g91-compatibility.xlog", "accepted", "rows: 1"),
        ("03-faeel-default.xlog", "accepted", "rows: 1"),
        ("04-gpt-candidate-filter.xlog", "accepted", "rows: 1"),
        ("05-splitting.xlog", "left", "rows: 1"),
        ("05-splitting.xlog", "right", "rows: 1"),
        // v0.9.1 epistemic executor showcase (EGB-01/02/04/05/06/07), each validated
        // through the production `xlog run` path with a deterministic output marker.
        ("06-eir-candidate-enumeration.xlog", "believed", "| 3  |"),
        ("07-tuple-key-membership.xlog", "matched", "| 3  | 3  |"),
        ("08-repeated-variable.xlog", "reflexive", "| 3  |"),
        ("09-joint-multi-epistemic.xlog", "both_known", "| 1  |"),
        ("10-epistemic-constraint.xlog", "accepted", "rows: 0"),
        ("11-faeel-foundedness.xlog", "founded", "rows: 1"),
        ("12-bound-variable-splitting.xlog", "both_known", "| 1  |"),
        ("12-bound-variable-splitting.xlog", "safe_alt", "| 2  |"),
    ];

    for (example, expected_relation, expected_value) in examples {
        let program = repo_root.join("examples/epistemic").join(example);
        let output = cargo_bin_cmd!("xlog")
            .args([
                "run",
                program.to_str().expect("valid path"),
                "--memory-mb",
                "1024",
            ])
            .output()
            .expect("run xlog binary");
        assert!(
            output.status.success(),
            "{} failed:\nstdout:\n{}\nstderr:\n{}",
            example,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(expected_relation),
            "{} did not emit relation {}:\n{}",
            example,
            expected_relation,
            stdout
        );
        assert!(
            stdout.contains(expected_value),
            "{} did not emit expected value marker {}:\n{}",
            example,
            expected_value,
            stdout
        );
    }
}

#[test]
fn test_xlog_run_nested_modal_reports_typed_epistemic_diagnostic() {
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let program = repo_root
        .join("examples/epistemic")
        .join("13-nested-modal-rejected.xlog");
    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog binary");
    assert!(
        !output.status.success(),
        "nested modal example must fail closed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("UnsupportedEpistemicConstruct"), "{stderr}");
    assert!(stderr.contains("nested epistemic literal"), "{stderr}");
    assert!(stderr.contains("know possible p()"), "{stderr}");
}
