use assert_cmd::cargo::cargo_bin_cmd;
use std::path::Path;

#[test]
fn test_xlog_explain_magic_sets_text() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let program = repo_root.join("examples/v085-language/magic_sets/reach_bound.xlog");

    let output = cargo_bin_cmd!("xlog")
        .args(["explain", program.to_str().expect("valid path")])
        .output()
        .expect("run xlog explain");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("status: applied"), "{stdout}");
    assert!(stdout.contains("reach/bf"), "{stdout}");
    assert!(stdout.contains("__xlog_magic_reach_bf"), "{stdout}");
}

#[test]
fn test_xlog_explain_json_reports_aggregate_lifting() {
    let program = std::env::temp_dir().join(format!(
        "xlog_aggregate_lift_explain_{}.xlog",
        std::process::id()
    ));
    std::fs::write(
        &program,
        r#"
0.5::edge(1, 2).
0.25::edge(1, 3).
out_degree(X, count(Y)) :- edge(X, Y).
query(out_degree(1, 2)).
"#,
    )
    .expect("write aggregate lift explain fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("\"aggregate_lifting\""), "{stdout}");
    assert!(stdout.contains("\"predicate\": \"out_degree\""), "{stdout}");
    assert!(stdout.contains("\"operator\": \"count\""), "{stdout}");
    assert!(stdout.contains("\"status\": \"fired\""), "{stdout}");
}
