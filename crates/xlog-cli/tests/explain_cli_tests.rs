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
    assert!(stdout.contains("\"parse\""), "{stdout}");
    assert!(stdout.contains("\"ast\""), "{stdout}");
    assert!(stdout.contains("\"stratification\""), "{stdout}");
    assert!(stdout.contains("\"rir\""), "{stdout}");
    assert!(stdout.contains("\"optimizer\""), "{stdout}");
    assert!(stdout.contains("\"wcoj\""), "{stdout}");
    assert!(stdout.contains("\"probability\""), "{stdout}");
}

#[test]
fn test_xlog_explain_json_reports_rule_provenance_for_source_and_generated_rules() {
    let program = std::env::temp_dir().join(format!(
        "xlog_rule_provenance_explain_{}.xlog",
        std::process::id()
    ));
    std::fs::write(
        &program,
        r#"
#pragma magic_sets=on
pred edge(i32, i32).
pred reach(i32, i32).
edge(1, 2).
edge(2, 3).
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).
?- reach(1, N).
"#,
    )
    .expect("write rule provenance fixture");
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
    assert!(stdout.contains("\"rule_provenance\""), "{stdout}");
    assert!(stdout.contains("\"rule_id\""), "{stdout}");
    assert!(stdout.contains("\"source_kind\": \"source\""), "{stdout}");
    assert!(
        stdout.contains("\"source_kind\": \"generated\""),
        "{stdout}"
    );
    assert!(stdout.contains("__xlog_magic_reach_bf"), "{stdout}");
    assert!(stdout.contains("\"generation_trace_hash\""), "{stdout}");
    assert!(stdout.contains("\"support_relation_ids\""), "{stdout}");
}

#[test]
fn test_xlog_explain_json_reports_contradiction_query_trace() {
    let program = std::env::temp_dir().join(format!(
        "xlog_contradiction_trace_explain_{}.xlog",
        std::process::id()
    ));
    std::fs::write(
        &program,
        r#"
state(1, 10).
state(1, 11).
incompatible(10, 11).
contradiction(E) :- state(E, A), state(E, B), incompatible(A, B).
?- contradiction(E).
"#,
    )
    .expect("write contradiction trace fixture");
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
    assert!(stdout.contains("\"proof_traces\""), "{stdout}");
    assert!(
        stdout.contains("\"query\": \"contradiction(E)\""),
        "{stdout}"
    );
    assert!(stdout.contains("\"rule_ids\""), "{stdout}");
    assert!(stdout.contains("\"source_facts\""), "{stdout}");
    assert!(stdout.contains("state(1, 10)"), "{stdout}");
    assert!(stdout.contains("incompatible(10, 11)"), "{stdout}");
}
