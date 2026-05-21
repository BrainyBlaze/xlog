use assert_cmd::cargo::cargo_bin_cmd;

#[test]
fn xlog_genrule_002_explain_json_reports_row_decisions_and_thresholds() {
    let program =
        std::env::temp_dir().join(format!("xlog_genrule_002_{}.xlog", std::process::id()));
    std::fs::write(
        &program,
        r#"
pred generated_candidate(symbol, i32, i32).
pred generated_accept(symbol).

generated_candidate("accepted", 4, 0).
generated_candidate("low_support", 1, 0).
generated_candidate("label_leak", 5, 1).

generated_accept(Name) :- generated_candidate(Name, Support, Leak), Support >= 3, Leak == 0.
?- generated_accept(Name).
"#,
    )
    .expect("write generated-rule fixture");

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
    assert!(
        stdout.contains("\"generated_rule_diagnostics\""),
        "{stdout}"
    );
    assert!(stdout.contains("\"row_decisions\""), "{stdout}");
    assert!(stdout.contains("\"row_key\": \"accepted\""), "{stdout}");
    assert!(stdout.contains("\"accepted\": true"), "{stdout}");
    assert!(stdout.contains("\"row_key\": \"low_support\""), "{stdout}");
    assert!(stdout.contains("\"accepted\": false"), "{stdout}");
    assert!(stdout.contains("\"failed_predicates\""), "{stdout}");
    assert!(stdout.contains("\"threshold_comparisons\""), "{stdout}");
    assert!(stdout.contains("\"aggregate_inputs\""), "{stdout}");
}
