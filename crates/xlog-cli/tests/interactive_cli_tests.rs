use assert_cmd::Command;

#[test]
fn test_xlog_repl_parses_multiline_session_without_gpu() {
    let output = Command::cargo_bin("xlog")
        .expect("xlog binary")
        .arg("repl")
        .write_stdin(
            r#"
edge(1, 2).
reach(X, Y) :- edge(X, Y).
?- reach(1, 2).
"#,
        )
        .output()
        .expect("run xlog repl");
    assert!(
        output.status.success(),
        "xlog repl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("repl:"), "{stdout}");
    assert!(stdout.contains("statements=3"), "{stdout}");
    assert!(stdout.contains("rules=2"), "{stdout}");
    assert!(stdout.contains("queries=1"), "{stdout}");
}

#[test]
fn test_xlog_watch_once_explain_smoke() {
    let program = std::env::temp_dir().join(format!("xlog_watch_once_{}.xlog", std::process::id()));
    std::fs::write(
        &program,
        r#"
#pragma magic_sets = auto
edge(1, 2).
reach(X, Y) :- edge(X, Y).
?- reach(1, 2).
"#,
    )
    .expect("write watch fixture");

    let output = Command::cargo_bin("xlog")
        .expect("xlog binary")
        .args([
            "watch",
            "--once",
            "--explain",
            program.to_str().expect("valid path"),
        ])
        .output()
        .expect("run xlog watch");
    assert!(
        output.status.success(),
        "xlog watch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("watch:"), "{stdout}");
    assert!(stdout.contains("magic_sets:"), "{stdout}");
}
