use assert_cmd::cargo::cargo_bin_cmd;
use tempfile::TempDir;

#[test]
fn xlog_extract_emits_the_resolved_executable_program() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("support.xlog"),
        "pred support(symbol).\nsupport(ok).\n",
    )
    .expect("write support module");
    let entry = fixture.path().join("main.xlog");
    std::fs::write(
        &entry,
        "use support.\npred answer(symbol).\nanswer(X) :- support(X).\n?- answer(X).\n",
    )
    .expect("write entry module");

    let first = cargo_bin_cmd!("xlog")
        .args([
            "extract",
            "--source-root",
            fixture.path().to_str().expect("UTF-8 source root"),
            entry.to_str().expect("UTF-8 entry path"),
        ])
        .output()
        .expect("run xlog extract");
    assert!(
        first.status.success(),
        "xlog extract failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let second = cargo_bin_cmd!("xlog")
        .args([
            "extract",
            "--source-root",
            fixture.path().to_str().expect("UTF-8 source root"),
            entry.to_str().expect("UTF-8 entry path"),
        ])
        .output()
        .expect("rerun xlog extract");
    assert_eq!(first.stdout, second.stdout);

    let stdout = String::from_utf8(first.stdout).expect("UTF-8 extraction output");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON extraction");
    assert_eq!(
        payload["schema_version"],
        "xlog.resolved-program-extraction.v1"
    );
    assert_eq!(
        payload["source_manifest"]["modules"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        payload["source_manifest"]["imports"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        payload["executable_program"]["rules"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        payload["executable_program"]["queries"][0]["goal"]["relation_id"],
        "relation:answer/1"
    );
    assert!(!stdout.contains(fixture.path().to_str().unwrap()));
}

#[test]
fn xlog_extract_rejects_a_resolved_source_outside_the_declared_root() {
    let fixture = TempDir::new().expect("create fixture directory");
    let external = TempDir::new().expect("create external module directory");
    std::fs::write(external.path().join("support.xlog"), "support(ok).\n")
        .expect("write external support module");
    let entry = fixture.path().join("main.xlog");
    std::fs::write(&entry, "use support.\n").expect("write entry module");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "extract",
            "--source-root",
            fixture.path().to_str().expect("UTF-8 source root"),
            "--module-path",
            external.path().to_str().expect("UTF-8 module path"),
            entry.to_str().expect("UTF-8 entry path"),
        ])
        .output()
        .expect("run xlog extract");

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("outside source root"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
