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

#[test]
fn xlog_genrule_002_explain_json_reports_bfo_external_candidate_rows() {
    let dir = std::env::temp_dir().join(format!("xlog_genrule_002_bfo_{}", std::process::id()));
    std::fs::create_dir_all(dir.join("xlog")).expect("create fixture dir");
    let program = dir.join("xlog/generated_hf_discovery_rules.xlog");
    std::fs::write(
        &program,
        r#"
pred hf_candidate_input(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).
pred xlog_accepted_candidate(i64, i64, i64, i64, i64).
pred xlog_rejected_candidate(i64, u32).

xlog_accepted_candidate(Hypothesis, Rank, NeuralMilli, SymbolicMilli, GraphMilli) :-
    hf_candidate_input(Hypothesis, Rank, 1, 1, 1, 0, NeuralMilli, SymbolicMilli, GraphMilli, SupportCount, ResistanceRatioMilli, SeedContext),
    NeuralMilli >= 620,
    SymbolicMilli >= 540,
    GraphMilli >= 480,
    SupportCount >= 80,
    ResistanceRatioMilli >= 50.

xlog_rejected_candidate(Hypothesis, 4) :-
    hf_candidate_input(Hypothesis, Rank, 1, 1, 1, 0, NeuralMilli, SymbolicMilli, GraphMilli, SupportCount, ResistanceRatioMilli, SeedContext),
    SupportCount < 80.

?- xlog_accepted_candidate(Hypothesis, Rank, NeuralMilli, SymbolicMilli, GraphMilli).
?- xlog_rejected_candidate(Hypothesis, Reason).
"#,
    )
    .expect("write BFO generated-rule fixture");
    std::fs::write(
        dir.join("xlog/hf_candidate_relation.json"),
        r#"{
  "rows": [
    {
      "hypothesis_num": 36,
      "preliminary_rank": 1,
      "derived_from_huggingface": 1,
      "bfo_valid": 1,
      "novelty_valid": 1,
      "neural_contradicts": 0,
      "neural_milli": 936,
      "symbolic_milli": 833,
      "graph_milli": 734,
      "support_count": 2013,
      "resistance_ratio_milli": 820,
      "seed_context": 0
    },
    {
      "hypothesis_num": 1011,
      "preliminary_rank": 1011,
      "derived_from_huggingface": 1,
      "bfo_valid": 1,
      "novelty_valid": 1,
      "neural_contradicts": 0,
      "neural_milli": 884,
      "symbolic_milli": 670,
      "graph_milli": 590,
      "support_count": 79,
      "resistance_ratio_milli": 49,
      "seed_context": 0
    }
  ]
}
"#,
    )
    .expect("write BFO candidate relation fixture");
    let relation_path = dir.join("xlog/hf_candidate_relation.json");
    std::fs::write(
        dir.join("xlog_hypothesis_execution.json"),
        format!(
            r#"{{
  "relation_input_path": "{}",
  "relation_input_columns": [
    "hypothesis_num",
    "preliminary_rank",
    "derived_from_huggingface",
    "bfo_valid",
    "novelty_valid",
    "neural_contradicts",
    "neural_milli",
    "symbolic_milli",
    "graph_milli",
    "support_count",
    "resistance_ratio_milli",
    "seed_context"
  ]
}}
"#,
            relation_path.display()
        ),
    )
    .expect("write BFO execution manifest");

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
    assert!(
        stdout.contains("\"source_relation\": \"hf_candidate_input\""),
        "{stdout}"
    );
    assert!(stdout.contains("\"row_key\": \"36\""), "{stdout}");
    assert!(stdout.contains("\"accepted\": true"), "{stdout}");
    assert!(stdout.contains("\"row_key\": \"1011\""), "{stdout}");
    assert!(stdout.contains("\"accepted\": false"), "{stdout}");
    assert!(stdout.contains("SupportCount >= 80"), "{stdout}");
    assert!(stdout.contains("\"aggregate_inputs\""), "{stdout}");
}
