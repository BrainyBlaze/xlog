use assert_cmd::Command;
use tempfile::TempDir;

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
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("watch_once.xlog");
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

#[test]
fn test_xlog_watch_once_explain_compiles_arithmetic_udf() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("arithmetic_udf.xlog");
    std::fs::write(
        &program,
        r#"
pred input(i64).
pred answer(i64).
func double(X) = X * 2.
input(1).
answer(Y) :- input(X), Y is double(X).
?- answer(Y).
"#,
    )
    .expect("write watch UDF fixture");

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
    assert!(stdout.contains("rir:\n  status: ok"), "{stdout}");
    assert!(stdout.contains("optimizer:\n  status: ok"), "{stdout}");
}

#[test]
fn test_xlog_watch_once_explain_compiles_sibling_imported_udf() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("arithmetic.xlog"),
        "pred input(i64).\nfunc double(X) = X * 2.\ninput(1).\n",
    )
    .expect("write arithmetic module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "use arithmetic.\npred answer(i64).\nanswer(Y) :- input(X), Y is double(X).\n?- answer(Y).\n",
    )
    .expect("write main program");

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
    assert!(stdout.contains("ast:\n  rules: 2"), "{stdout}");
    assert!(stdout.contains("rir:\n  status: ok"), "{stdout}");
    assert!(stdout.contains("optimizer:\n  status: ok"), "{stdout}");
    assert!(
        !stdout.contains("must be inlined before lowering"),
        "{stdout}"
    );
}

#[test]
fn test_xlog_watch_once_explain_resolves_udf_from_module_path() {
    let fixture = TempDir::new().expect("create fixture directory");
    let modules = fixture.path().join("modules");
    std::fs::create_dir(&modules).expect("create module directory");
    std::fs::write(
        modules.join("arithmetic.xlog"),
        "pred input(i64).\nfunc double(X) = X * 2.\ninput(1).\n",
    )
    .expect("write arithmetic module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "use arithmetic.\npred answer(i64).\nanswer(Y) :- input(X), Y is double(X).\n?- answer(Y).\n",
    )
    .expect("write main program");

    let output = Command::cargo_bin("xlog")
        .expect("xlog binary")
        .args([
            "watch",
            "--once",
            "--explain",
            "--module-path",
            modules.to_str().expect("valid module path"),
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
    assert!(stdout.contains("ast:\n  rules: 2"), "{stdout}");
    assert!(stdout.contains("rir:\n  status: ok"), "{stdout}");
    assert!(stdout.contains("optimizer:\n  status: ok"), "{stdout}");
}
