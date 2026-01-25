use std::path::Path;
use std::process::Command;

#[test]
fn test_bundled_d4_is_built_and_runnable() {
    let d4_path = option_env!("XLOG_PROB_D4_PATH")
        .expect("XLOG_PROB_D4_PATH not set: xlog-prob build script must build vendored D4");
    assert!(
        Path::new(d4_path).exists(),
        "vendored D4 binary missing at {}",
        d4_path
    );

    let output = Command::new(d4_path)
        .arg("--help")
        .output()
        .expect("failed to execute bundled d4 binary");
    assert!(
        output.status.success(),
        "bundled d4 failed (exit={})\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}
