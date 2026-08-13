use assert_cmd::cargo::cargo_bin_cmd;
use tempfile::TempDir;
use xlog_cuda::CudaDevice;

const ARITHMETIC_UDF_PROGRAM: &str = r#"
pred input(i64).
pred answer(i64).
func double(X) = X * 2.
input(1).
answer(Y) :- input(X), Y is double(X).
?- answer(Y).
"#;

fn finite_udf_chain_program(call_depth: usize, configured_limit: Option<u32>) -> String {
    let mut source = String::new();
    if let Some(limit) = configured_limit {
        source.push_str(&format!("#pragma max_recursion_depth = {limit}\n"));
    }
    source.push_str("pred input(i64).\npred answer(i64).\nfunc f0(X) = X.\n");
    for index in 1..=call_depth {
        source.push_str(&format!("func f{index}(X) = f{}(X).\n", index - 1));
    }
    source.push_str(&format!(
        "input(7).\nanswer(Y) :- input(X), Y is f{call_depth}(X).\n?- answer(Y).\n"
    ));
    source
}

fn nested_udf_chain_program(call_depth: usize, body: impl Fn(usize) -> String) -> String {
    let mut source = String::from("pred input(i64).\npred answer(i64).\nfunc f0(X) = X.\n");
    for index in 1..=call_depth {
        source.push_str(&format!("func f{index}(X) = {}.\n", body(index)));
    }
    source.push_str(&format!(
        "input(7).\nanswer(Y) :- input(X), Y is f{call_depth}(X).\n?- answer(Y).\n"
    ));
    source
}

fn predicate_udf_arithmetic_chain_program(call_depth: usize) -> String {
    let mut source = String::from(
        "pred parent(u32, u32).\npred answer(u32).\nparent(1, 2).\n\
         func lookup0(X) = Parent :- parent(X, Parent).\n",
    );
    for index in 1..=call_depth {
        source.push_str(&format!(
            "func lookup{index}(X) = Result :- Result is lookup{}(X) + cast(0, u32).\n",
            index - 1
        ));
    }
    source.push_str(&format!(
        "answer(Y) :- Y is lookup{call_depth}(1).\n?- answer(Y).\n"
    ));
    source
}

fn assert_deep_udf_chain_runs(source: String, expected_header: &str, expected: i64, cse: bool) {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("deep_runtime_udf_chain.xlog");
    std::fs::write(&program, source).expect("write deep runtime UDF fixture");

    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED deep runtime UDF chain: CUDA unavailable: {error}");
            return;
        }
    };

    let mut command = cargo_bin_cmd!("xlog");
    command.args([
        "run",
        program.to_str().expect("valid program path"),
        "--output",
        "csv",
    ]);
    if cse {
        command.env("XLOG_CSE", "1");
    } else {
        command.env_remove("XLOG_CSE");
    }
    let output = command.output().expect("run deep UDF chain");
    assert!(
        output.status.success(),
        "xlog run failed with {status}: {}",
        String::from_utf8_lossy(&output.stderr),
        status = output.status
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(
        stdout,
        format!("__xlog_query_0\n{expected_header}\n{expected}\n\n")
    );
}

fn assert_analysis_not_available(
    payload: &serde_json::Value,
    section: &str,
    expected_reason: &str,
    stdout: &str,
) {
    assert_eq!(payload[section]["status"], "not_available", "{stdout}");
    assert!(
        payload[section]["reason"]
            .as_str()
            .unwrap_or_default()
            .contains(expected_reason),
        "{stdout}"
    );
}

fn explain_json(program: &std::path::Path) -> (serde_json::Value, String) {
    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload = serde_json::from_str(&stdout).expect("valid explain json");
    (payload, stdout)
}

fn assert_default_udf_cycle_reports_e0504(source: &str, function_name: &str) {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("recursive_udf.xlog");
    std::fs::write(&program, source).expect("write recursive UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain did not emit its diagnostic report: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    for section in ["rir", "optimizer", "wcoj"] {
        assert_analysis_not_available(&payload, section, "error[E0504]", &stdout);
    }
    let rir_reason = payload["rir"]["reason"]
        .as_str()
        .expect("RIR unavailable reason");
    assert!(
        rir_reason.contains("maximum recursion depth (1000) exceeded"),
        "{stdout}"
    );
    assert!(
        rir_reason.contains(&format!("function `{function_name}`")),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_magic_sets_text() {
    let program = std::env::temp_dir().join(format!(
        "xlog_magic_sets_explain_{}.xlog",
        std::process::id()
    ));
    std::fs::write(
        &program,
        r#"
#pragma magic_sets = on

pred edge(src: u32, dst: u32).
pred reach(src: u32, dst: u32).

edge(1, 2).
edge(2, 3).
edge(10, 11).
edge(11, 12).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(1, Y).
"#,
    )
    .expect("write magic sets explain fixture");

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
fn test_xlog_run_and_explain_compile_the_same_arithmetic_udf() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("arithmetic_udf.xlog");
    std::fs::write(&program, ARITHMETIC_UDF_PROGRAM).expect("write arithmetic UDF fixture");

    let explain_output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        explain_output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&explain_output.stderr)
    );
    let explain_stdout = String::from_utf8(explain_output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value =
        serde_json::from_str(&explain_stdout).expect("valid explain json");
    assert_eq!(payload["rir"]["status"], "ok", "{explain_stdout}");
    assert_eq!(payload["optimizer"]["status"], "ok", "{explain_stdout}");

    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED runtime parity: CUDA unavailable; explain parity passed: {error}");
            return;
        }
    };

    let run_output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--output",
            "csv",
        ])
        .output()
        .expect("run xlog arithmetic UDF");
    assert!(
        run_output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );
    let run_stdout = String::from_utf8(run_output.stdout).expect("utf8 stdout");
    assert_eq!(run_stdout, "__xlog_query_0\ncomputed_1\n2\n\n");
}

#[test]
fn test_xlog_explain_uses_documented_default_for_deep_finite_udf_chain() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("deep_finite_udf_chain.xlog");
    std::fs::write(&program, finite_udf_chain_program(999, None))
        .expect("write deep finite UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(payload["optimizer"]["status"], "ok", "{stdout}");
}

#[test]
fn test_xlog_explain_preserves_explicit_udf_recursion_limit() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("configured_udf_limit.xlog");
    std::fs::write(&program, finite_udf_chain_program(5, Some(5)))
        .expect("write configured UDF limit fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain did not emit its diagnostic report: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    for section in ["rir", "optimizer", "wcoj"] {
        assert_analysis_not_available(
            &payload,
            section,
            "maximum recursion depth (5) exceeded",
            &stdout,
        );
    }
}

#[test]
fn test_xlog_explain_reports_direct_udf_cycle_without_aborting() {
    assert_default_udf_cycle_reports_e0504(
        "func repeat(X) = repeat(X).\nanswer(Y) :- Y is repeat(1).\n?- answer(Y).\n",
        "repeat",
    );
}

#[test]
fn test_xlog_explain_reports_conditional_udf_cycle_without_aborting() {
    assert_default_udf_cycle_reports_e0504(
        "func countdown(X) = if X <= 0 then 0 else countdown(X - 1).\n\
         answer(Y) :- Y is countdown(1).\n?- answer(Y).\n",
        "countdown",
    );
}

#[test]
fn test_xlog_run_evaluates_default_depth_non_tail_udf_chain_without_aborting() {
    assert_deep_udf_chain_runs(
        nested_udf_chain_program(999, |index| format!("f{}(X) + 1", index - 1)),
        "computed_1",
        1006,
        false,
    );
}

#[test]
fn test_xlog_run_evaluates_default_depth_nested_argument_udf_chain_without_aborting() {
    assert_deep_udf_chain_runs(
        nested_udf_chain_program(999, |index| format!("f{}(X + 1)", index - 1)),
        "computed_1",
        1006,
        false,
    );
}

#[test]
fn test_xlog_run_evaluates_default_depth_conditional_udf_chain_without_aborting() {
    assert_deep_udf_chain_runs(
        nested_udf_chain_program(999, |index| {
            format!("if X >= 0 then f{}(X) else X", index - 1)
        }),
        "computed_1",
        7,
        false,
    );
}

#[test]
fn test_xlog_run_evaluates_deep_udf_call_in_conditional_condition_without_aborting() {
    let mut source = String::from("pred input(i64).\npred answer(i64).\nfunc f0(X) = X.\n");
    for index in 1..=998 {
        source.push_str(&format!("func f{index}(X) = f{}(X) + 1.\n", index - 1));
    }
    source.push_str(
        "func choose(X) = if f998(X) >= 0 then X else 0.\n\
         input(7).\nanswer(Y) :- input(X), Y is choose(X).\n?- answer(Y).\n",
    );
    assert_deep_udf_chain_runs(source, "computed_1", 7, false);
}

#[test]
fn test_xlog_run_evaluates_default_depth_udf_chain_with_cse_without_aborting() {
    assert_deep_udf_chain_runs(
        nested_udf_chain_program(999, |index| format!("f{}(X) + 1", index - 1)),
        "computed_1",
        1006,
        true,
    );
}

#[test]
fn test_xlog_run_evaluates_default_depth_predicate_udf_chain_without_aborting() {
    let mut source = String::from(
        "pred parent(u32, u32).\npred answer(u32).\nparent(1, 2).\n\
         func lookup0(X) = Parent :- parent(X, Parent).\n",
    );
    for index in 1..=999 {
        source.push_str(&format!(
            "func lookup{index}(X) = Result :- Result is lookup{}(X).\n",
            index - 1
        ));
    }
    source.push_str("answer(Y) :- Y is lookup999(1).\n?- answer(Y).\n");
    assert_deep_udf_chain_runs(source, "computed_2", 2, false);
}

#[test]
fn test_xlog_run_evaluates_default_depth_predicate_udf_arithmetic_chain_without_aborting() {
    assert_deep_udf_chain_runs(
        predicate_udf_arithmetic_chain_program(999),
        "computed_1001",
        2,
        false,
    );
}

#[test]
fn test_xlog_run_evaluates_default_depth_predicate_udf_arithmetic_chain_with_cse() {
    assert_deep_udf_chain_runs(
        predicate_udf_arithmetic_chain_program(999),
        "computed_1001",
        2,
        true,
    );
}

#[test]
fn test_xlog_explain_preserves_udf_normalization_error_when_magic_sets_are_forced() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("recursive_udf_with_magic_sets.xlog");
    std::fs::write(
        &program,
        "#pragma magic_sets = on\n\
         func repeat(X) = repeat(X).\n\
         answer(Y) :- Y is repeat(1).\n\
         ?- answer(Y).\n",
    )
    .expect("write recursive UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain did not emit its diagnostic report: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    for section in ["rir", "optimizer", "wcoj"] {
        assert_analysis_not_available(&payload, section, "error[E0504]", &stdout);
    }
    assert_eq!(
        payload["stratification"]["status"], "not_available",
        "{stdout}"
    );
    assert!(
        payload["stratification"]["reason"]
            .as_str()
            .expect("stratification reason")
            .contains("error[E0504]"),
        "{stdout}"
    );
    assert_eq!(payload["magic_sets"]["status"], "declined", "{stdout}");
    assert!(
        payload["magic_sets"]["declined_reasons"]
            .as_array()
            .expect("magic-set declined reasons")
            .iter()
            .any(|reason| reason
                .as_str()
                .is_some_and(|reason| reason.contains("error[E0504]"))),
        "{stdout}"
    );
    for section in ["eir", "gpu_plan", "executable_plan"] {
        assert_eq!(
            payload["epistemic"][section]["status"], "not_available",
            "{stdout}"
        );
        assert!(
            payload["epistemic"][section]["reason"]
                .as_str()
                .expect("epistemic reason")
                .contains("error[E0504]"),
            "{stdout}"
        );
    }
    assert_eq!(payload["ast"]["rules"], 1, "{stdout}");
    assert_eq!(
        payload["rule_provenance"][0]["source_kind"], "source",
        "{stdout}"
    );
    assert_eq!(payload["proof_traces"][0]["query"], "answer(Y)", "{stdout}");

    let text_output = cargo_bin_cmd!("xlog")
        .args(["explain", program.to_str().expect("valid program path")])
        .output()
        .expect("run xlog explain text");
    assert!(
        text_output.status.success(),
        "xlog explain did not emit its text diagnostic report: {}",
        String::from_utf8_lossy(&text_output.stderr)
    );
    let text = String::from_utf8(text_output.stdout).expect("utf8 text output");
    for section in ["eir_reason", "gpu_plan_reason", "executable_plan_reason"] {
        assert!(
            text.contains(&format!("{section}: execution normalization failed")),
            "{text}"
        );
    }
    assert!(text.contains("error[E0504]"), "{text}");
}

#[test]
fn test_xlog_explain_magic_dot_preserves_normalization_failure_reason() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("recursive_udf_magic_graph.xlog");
    std::fs::write(
        &program,
        "#pragma magic_sets = on\n\
         func repeat(X) = repeat(X).\n\
         answer(Y) :- Y is repeat(1).\n\
         ?- answer(Y).\n",
    )
    .expect("write recursive magic-set graph fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "dot",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain dot");
    assert!(
        output.status.success(),
        "xlog explain did not emit its diagnostic graph: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("status: declined"), "{stdout}");
    assert!(stdout.contains("error[E0504]"), "{stdout}");
    assert!(
        stdout.contains("maximum recursion depth (1000) exceeded"),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_withholds_generated_row_decisions_after_normalization_failure() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("recursive_udf_with_generated_rule.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, i32, i32).\n\
         pred generated_accept(symbol).\n\
         generated_candidate(\"accepted\", 4, 0).\n\
         func repeat(X) = repeat(X).\n\
         generated_accept(Name) :- generated_candidate(Name, Support, Leak), Value is repeat(Support), Support >= 3, Leak == 0.\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write recursive generated-rule fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain did not emit its diagnostic report: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    for section in ["rir", "optimizer", "wcoj"] {
        assert_analysis_not_available(&payload, section, "error[E0504]", &stdout);
    }
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "not_available",
        "{stdout}"
    );
    assert!(
        payload["generated_rule_diagnostics_reason"]
            .as_str()
            .expect("generated-rule diagnostics reason")
            .contains("error[E0504]"),
        "{stdout}"
    );
    assert_eq!(
        payload["generated_rule_diagnostics"],
        serde_json::json!([]),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_keeps_generated_row_diagnostics_source_formatted_after_normalization() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("generated_rule_with_predicate_udf.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, i32).\n\
         pred parent(i32, i32).\n\
         pred generated_accept(symbol).\n\
         generated_candidate(\"accepted\", 1).\n\
         parent(1, 2).\n\
         func get_parent(Child) = Parent :- parent(Child, Parent).\n\
         generated_accept(Name) :- generated_candidate(Name, Child), Parent is get_parent(Child).\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write generated-rule predicate UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "ok",
        "{stdout}"
    );
    assert!(
        payload["generated_rule_diagnostics_reason"].is_null(),
        "{stdout}"
    );
    assert_eq!(
        payload["generated_rule_diagnostics"][0]["row_decisions"][0]["accepted"], true,
        "{stdout}"
    );
    assert!(
        !stdout.to_ascii_lowercase().contains("__xlog_function"),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_rejects_generated_row_without_predicate_udf_support() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("generated_rule_without_predicate_support.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, i32).\n\
         pred parent(i32, i32).\n\
         pred generated_accept(symbol).\n\
         generated_candidate(\"unsupported\", 1).\n\
         func get_parent(Child) = Parent :- parent(Child, Parent).\n\
         generated_accept(Name) :- generated_candidate(Name, Child), Parent is get_parent(Child).\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write unsupported generated-rule predicate UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    let decision = &payload["generated_rule_diagnostics"][0]["row_decisions"][0];
    assert_eq!(decision["accepted"], false, "{stdout}");
    assert_eq!(
        decision["failed_predicates"],
        serde_json::json!(["parent(Child, Parent)"]),
        "{stdout}"
    );
    assert!(
        !stdout.to_ascii_lowercase().contains("__xlog_function"),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_evaluates_normalized_arithmetic_for_generated_rows() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("generated_rule_with_arithmetic_udf.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, i64).\n\
         pred generated_accept(symbol).\n\
         generated_candidate(\"accepted\", 2).\n\
         func double(X) = X * 2.\n\
         generated_accept(Name) :- generated_candidate(Name, X), Y is double(X), X >= 2.\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write generated-rule arithmetic UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    let decision = &payload["generated_rule_diagnostics"][0]["row_decisions"][0];
    assert_eq!(decision["accepted"], true, "{stdout}");
    assert_eq!(
        decision["threshold_comparisons"][0]["left_value"], "2",
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_evaluates_runtime_compatible_boolean_casts() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("generated_rule_boolean_cast.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, bool).\n\
         pred generated_accept(symbol, f64).\n\
         generated_candidate(\"accepted\", true).\n\
         generated_accept(Name, Value) :- generated_candidate(Name, Enabled), Value is cast(Enabled, f64).\n\
         ?- generated_accept(Name, Value).\n",
    )
    .expect("write generated-rule boolean-cast fixture");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "ok",
        "{stdout}"
    );
    assert_eq!(
        payload["generated_rule_diagnostics"][0]["row_decisions"][0]["accepted"], true,
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_withholds_generated_rows_after_arithmetic_type_error() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("generated_rule_with_arithmetic_type_error.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, i64).\n\
         pred generated_accept(symbol).\n\
         generated_candidate(\"candidate\", 2).\n\
         generated_accept(Name) :- generated_candidate(Name, X), Y is X + cast(1, i32).\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write type-invalid generated-rule fixture");

    let (payload, stdout) = explain_json(&program);
    assert!(
        payload["rir"]["status"]
            .as_str()
            .unwrap_or_default()
            .starts_with("error:"),
        "{stdout}"
    );
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "not_available",
        "{stdout}"
    );
    assert!(
        payload["generated_rule_diagnostics_reason"]
            .as_str()
            .unwrap_or_default()
            .contains("Type mismatch in arithmetic"),
        "{stdout}"
    );
    assert_eq!(
        payload["generated_rule_diagnostics"],
        serde_json::json!([]),
        "{stdout}"
    );
    assert_eq!(payload["wcoj"]["status"], "not_available", "{stdout}");
    assert!(
        payload["wcoj"]["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("RIR compilation failed"),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_preserves_declared_numeric_widths_in_generated_rows() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("generated_rule_numeric_widths.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate_unsigned(symbol, u32).\n\
         pred generated_candidate_signed(symbol, i32).\n\
         pred generated_candidate_float(symbol, f64).\n\
         pred generated_accept_unsigned(symbol).\n\
         pred generated_accept_signed(symbol).\n\
         pred generated_accept_float(symbol).\n\
         generated_candidate_unsigned(\"unsigned\", 2147483648).\n\
         generated_candidate_signed(\"signed\", 1).\n\
         generated_candidate_float(\"float\", 1.5).\n\
         func twice_unsigned(X) = X + X.\n\
         func increment_signed(X) = X + cast(1, i32).\n\
         generated_accept_unsigned(Name) :- generated_candidate_unsigned(Name, X), Y is twice_unsigned(X), X == 2147483648.\n\
         generated_accept_signed(Name) :- generated_candidate_signed(Name, X), Y is increment_signed(X), X == 1.\n\
         generated_accept_float(Name) :- generated_candidate_float(Name, X), X >= 1.0.\n\
         ?- generated_accept_unsigned(Name).\n",
    )
    .expect("write typed generated-rule fixture");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "ok",
        "{stdout}"
    );
    let diagnostics = payload["generated_rule_diagnostics"]
        .as_array()
        .expect("generated diagnostics array");
    for source_relation in [
        "generated_candidate_unsigned",
        "generated_candidate_signed",
        "generated_candidate_float",
    ] {
        let diagnostic = diagnostics
            .iter()
            .find(|entry| entry["source_relation"] == source_relation)
            .unwrap_or_else(|| panic!("missing {source_relation} diagnostic: {stdout}"));
        assert_eq!(diagnostic["row_decisions"][0]["accepted"], true, "{stdout}");
    }
}

#[test]
fn test_xlog_explain_reads_external_float_rows_for_generated_diagnostics() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("generated_external_float.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(name: symbol, score: f64).\n\
         pred generated_unsigned(name: symbol, id: u64).\n\
         pred generated_accept(symbol).\n\
         pred generated_accept_unsigned(symbol).\n\
         generated_accept(Name) :- generated_candidate(Name, Score), Score >= 1.0.\n\
         generated_accept_unsigned(Name) :- generated_unsigned(Name, Id).\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write external generated-rule fixture");
    std::fs::write(
        fixture.path().join("generated_candidate.json"),
        r#"{"rows":[{"name":"line\nbreak","score":1.5}]}"#,
    )
    .expect("write external float rows");
    std::fs::write(
        fixture.path().join("generated_unsigned.json"),
        r#"{"rows":[{"name":"wide","id":18446744073709551615}]}"#,
    )
    .expect("write external u64 rows");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "ok",
        "{stdout}"
    );
    let diagnostics = payload["generated_rule_diagnostics"]
        .as_array()
        .expect("generated diagnostics");
    let float_diagnostic = diagnostics
        .iter()
        .find(|entry| entry["source_relation"] == "generated_candidate")
        .expect("float diagnostic");
    assert_eq!(
        float_diagnostic["row_decisions"][0]["accepted"], true,
        "{stdout}"
    );
    assert_eq!(
        float_diagnostic["row_decisions"][0]["row_key"], "line\nbreak",
        "{stdout}"
    );
    assert_eq!(
        float_diagnostic["row_decisions"][0]["threshold_comparisons"][0]["left_value"], "1.5",
        "{stdout}"
    );
    let unsigned_diagnostic = diagnostics
        .iter()
        .find(|entry| entry["source_relation"] == "generated_unsigned")
        .expect("u64 diagnostic");
    assert_eq!(
        unsigned_diagnostic["row_decisions"][0]["accepted"], true,
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_does_not_apply_candidate_manifest_rows_to_support_relations() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("generated_manifest_candidate.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(name: symbol, id: i32).\n\
         pred generated_allowed(name: symbol, id: i32).\n\
         pred generated_accept(symbol).\n\
         generated_accept(Name) :- generated_candidate(Name, Id), generated_allowed(Name, Id).\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write candidate-manifest fixture");
    std::fs::write(
        fixture.path().join("candidate_rows.json"),
        r#"{"rows":[{"name":"candidate","id":1}]}"#,
    )
    .expect("write candidate rows");
    std::fs::write(
        fixture.path().join("xlog_hypothesis_execution.json"),
        r#"{"relation_input_path":"candidate_rows.json","relation_input_columns":["name","id"]}"#,
    )
    .expect("write candidate manifest");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "ok",
        "{stdout}"
    );
    let decision = &payload["generated_rule_diagnostics"][0]["row_decisions"][0];
    assert_eq!(decision["row_key"], "candidate", "{stdout}");
    assert_eq!(decision["accepted"], false, "{stdout}");
    assert_eq!(
        decision["failed_predicates"],
        serde_json::json!(["generated_allowed(Name, Id)"]),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_marks_multi_source_candidate_manifest_unavailable() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("ambiguous_candidate_manifest.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate_alpha(name: symbol, id: i32).\n\
         pred generated_candidate_beta(name: symbol, id: i32).\n\
         pred generated_accept_alpha(symbol).\n\
         pred generated_accept_beta(symbol).\n\
         generated_accept_alpha(Name) :- generated_candidate_alpha(Name, Id).\n\
         generated_accept_beta(Name) :- generated_candidate_beta(Name, Id).\n\
         ?- generated_accept_alpha(Name).\n",
    )
    .expect("write ambiguous candidate-manifest fixture");
    std::fs::write(
        fixture.path().join("candidate_rows.json"),
        r#"{"rows":[{"name":"candidate","id":1}]}"#,
    )
    .expect("write candidate rows");
    std::fs::write(
        fixture.path().join("xlog_hypothesis_execution.json"),
        r#"{"relation_input_path":"candidate_rows.json","relation_input_columns":["name","id"]}"#,
    )
    .expect("write ambiguous candidate manifest");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "not_available",
        "{stdout}"
    );
    assert!(
        payload["generated_rule_diagnostics_reason"]
            .as_str()
            .unwrap_or_default()
            .contains("manifest is ambiguous"),
        "{stdout}"
    );
    assert_eq!(
        payload["generated_rule_diagnostics"],
        serde_json::json!([]),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_marks_derived_generated_support_unavailable() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("generated_rule_with_derived_support.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, i32).\n\
         pred base(i32).\n\
         pred allowed(i32).\n\
         pred generated_accept(symbol).\n\
         generated_candidate(\"accepted\", 1).\n\
         base(1).\n\
         allowed(X) :- base(X).\n\
         func visible(X) = Y :- allowed(X), Y is X.\n\
         generated_accept(Name) :- generated_candidate(Name, X), Y is visible(X).\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write derived-support fixture");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "not_available",
        "{stdout}"
    );
    assert!(
        payload["generated_rule_diagnostics_reason"]
            .as_str()
            .unwrap_or_default()
            .contains("materialized rows for derived predicate 'allowed'"),
        "{stdout}"
    );
    assert_eq!(
        payload["generated_rule_diagnostics"],
        serde_json::json!([]),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_marks_probabilistic_generated_support_unavailable() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("generated_rule_with_probabilistic_support.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, i32).\n\
         pred maybe_parent(i32).\n\
         pred generated_accept(symbol).\n\
         generated_candidate(\"candidate\", 1).\n\
         0.5::maybe_parent(1).\n\
         generated_accept(Name) :- generated_candidate(Name, X), maybe_parent(X).\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write probabilistic-support fixture");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "not_available",
        "{stdout}"
    );
    assert!(
        payload["generated_rule_diagnostics_reason"]
            .as_str()
            .unwrap_or_default()
            .contains("probabilistic predicate 'maybe_parent'"),
        "{stdout}"
    );
    assert_eq!(
        payload["generated_rule_diagnostics"],
        serde_json::json!([]),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_backtracks_to_a_valid_predicate_udf_witness() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("generated_rule_multiple_witnesses.xlog");
    std::fs::write(
        &program,
        "pred generated_candidate(symbol, i32).\n\
         pred parent(i32, i32).\n\
         pred generated_accept(symbol).\n\
         generated_candidate(\"accepted\", 1).\n\
         generated_candidate(\"rejected\", 2).\n\
         parent(1, 1).\n\
         parent(1, 3).\n\
         parent(2, 1).\n\
         parent(2, 2).\n\
         func qualifying_parent(Child) = Parent :- parent(Child, Parent), Parent >= 3.\n\
         generated_accept(Name) :- generated_candidate(Name, Child), Parent is qualifying_parent(Child).\n\
         ?- generated_accept(Name).\n",
    )
    .expect("write multiple-witness fixture");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "ok",
        "{stdout}"
    );
    let decisions = payload["generated_rule_diagnostics"][0]["row_decisions"]
        .as_array()
        .expect("row decisions");
    let accepted = decisions
        .iter()
        .find(|decision| decision["row_key"] == "accepted")
        .expect("accepted row decision");
    assert_eq!(accepted["accepted"], true, "{stdout}");
    let rejected = decisions
        .iter()
        .find(|decision| decision["row_key"] == "rejected")
        .expect("rejected row decision");
    assert_eq!(rejected["accepted"], false, "{stdout}");
    assert!(!rejected["failed_predicates"]
        .as_array()
        .expect("failed predicates")
        .is_empty());
}

#[test]
fn test_xlog_explain_source_formats_predicate_udf_rejected_alternatives() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("predicate_udf_rejected_alternative.xlog");
    std::fs::write(
        &program,
        "pred candidate(i32, i32).\n\
         pred blocked(i32).\n\
         pred forbidden(i32).\n\
         pred answer(i32).\n\
         candidate(1, 2).\n\
         func visible(X) = Y :- candidate(X, Y), not blocked(Y).\n\
         answer(Y) :- Y is visible(1), not forbidden(Y).\n\
         ?- answer(Y).\n",
    )
    .expect("write predicate UDF rejection fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    let rejected = payload["proof_traces"][0]["rejected_alternatives"]
        .as_array()
        .expect("rejected alternatives array");
    assert!(
        rejected
            .iter()
            .any(|entry| entry.as_str() == Some("not blocked(Y)")),
        "{stdout}"
    );
    assert!(
        rejected
            .iter()
            .any(|entry| entry.as_str() == Some("not forbidden(Y)")),
        "{stdout}"
    );
    assert!(
        !stdout.contains("__XLOG_FUNCTION"),
        "internal function variable leaked into source diagnostics: {stdout}"
    );
}

#[test]
fn test_xlog_explain_does_not_remap_generated_variable_text_inside_strings() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("quoted_generated_variable_text.xlog");
    let generated_variable_text = "__XLOG_FUNCTION_VISIBLE_Result_0";
    std::fs::write(
        &program,
        format!(
            "pred generated_candidate(symbol, i32).\n\
             pred candidate_result(i32, i32).\n\
             pred blocked(symbol).\n\
             pred generated_accept(symbol).\n\
             generated_candidate(\"rejected\", 1).\n\
             candidate_result(1, 2).\n\
             blocked(\"{generated_variable_text}\").\n\
             func visible(X) = Result :- candidate_result(X, Result), not blocked(\"{generated_variable_text}\").\n\
             generated_accept(Name) :- generated_candidate(Name, X), Result is visible(X).\n\
             ?- generated_accept(Name).\n"
        ),
    )
    .expect("write quoted-token generated-rule fixture");

    let (payload, stdout) = explain_json(&program);
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(
        payload["generated_rule_diagnostics_status"], "ok",
        "{stdout}"
    );
    let failed = &payload["generated_rule_diagnostics"][0]["row_decisions"][0]["failed_predicates"];
    assert_eq!(
        failed,
        &serde_json::json!([format!("not blocked(\"{generated_variable_text}\")")]),
        "{stdout}"
    );
    assert!(!stdout.contains("not blocked(Result)"), "{stdout}");
}

#[test]
fn test_xlog_explain_preserves_udf_normalization_error_during_aggregate_analysis() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("recursive_udf_with_probabilistic_aggregate.xlog");
    std::fs::write(
        &program,
        "pred edge(u64, u64).\n\
         pred scaled(u64, u64).\n\
         pred total(u64, u64).\n\
         0.5::edge(1, 2).\n\
         func repeat(X) = repeat(X).\n\
         scaled(X, Z) :- edge(X, Y), Z is repeat(Y).\n\
         total(X, sum(Z)) :- scaled(X, Z).\n\
         query(total(1, 2)).\n",
    )
    .expect("write recursive aggregate UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain did not emit its diagnostic report: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    for section in ["rir", "optimizer", "wcoj"] {
        assert_analysis_not_available(&payload, section, "error[E0504]", &stdout);
    }
    assert_eq!(
        payload["probability"]["aggregate_lifting_status"], "not_available",
        "{stdout}"
    );
    assert!(
        payload["probability"]["aggregate_lifting_reason"]
            .as_str()
            .expect("aggregate lifting reason")
            .contains("error[E0504]"),
        "{stdout}"
    );
    assert_eq!(
        payload["aggregate_lifting"]
            .as_array()
            .expect("aggregate lifting array")
            .len(),
        0,
        "{stdout}"
    );
    assert!(
        !stdout.contains("must be expanded before provenance extraction"),
        "{stdout}"
    );
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
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(
        payload["epistemic"]["eir"]["status"], "not_applicable",
        "{stdout}"
    );
    assert_eq!(
        payload["epistemic"]["gpu_plan"]["status"], "not_applicable",
        "{stdout}"
    );
    assert_eq!(
        payload["epistemic"]["executable_plan"]["status"], "not_applicable",
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_normalizes_udfs_before_probabilistic_aggregate_analysis() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("probabilistic_aggregate_udf.xlog");
    std::fs::write(
        &program,
        r#"
pred edge(u64, u64).
pred scaled(u64, u64).
pred total(u64, u64).
0.5::edge(1, 2).
0.25::edge(1, 3).
func double(X) = X * cast(2, u64).
scaled(X, Z) :- edge(X, Y), Z is double(Y).
total(X, sum(Z)) :- scaled(X, Z).
query(total(1, 10)).
"#,
    )
    .expect("write probabilistic aggregate UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(payload["optimizer"]["status"], "ok", "{stdout}");
    let aggregate_lifting = payload["aggregate_lifting"]
        .as_array()
        .expect("aggregate lifting array");
    assert!(
        aggregate_lifting.iter().any(|entry| {
            entry["predicate"] == "total"
                && entry["operator"] == "sum"
                && entry["status"] == "fired"
        }),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_enriches_source_diagnostics_from_predicate_udf_expansion() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("predicate_udf_provenance.xlog");
    std::fs::write(
        &program,
        "pred parent(u32, u32).\n\
         pred child(u32).\n\
         func get_parent(Child) = Parent :- parent(Child, Parent).\n\
         parent(1, 2).\n\
         child(Parent) :- Parent is get_parent(1).\n\
         ?- child(Parent).\n",
    )
    .expect("write predicate UDF fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    let child_rule = payload["rule_provenance"]
        .as_array()
        .expect("rule provenance array")
        .iter()
        .find(|entry| entry["source_kind"] == "source" && entry["head"] == "child(Parent)")
        .expect("source child rule");
    assert!(
        child_rule["support_relation_ids"]
            .as_array()
            .expect("support relations")
            .iter()
            .any(|relation| relation == "parent"),
        "{stdout}"
    );
    assert!(
        payload["proof_traces"][0]["source_facts"]
            .as_array()
            .expect("source facts")
            .iter()
            .any(|fact| fact
                .as_str()
                .is_some_and(|fact| fact.starts_with("parent(1, 2)"))),
        "{stdout}"
    );
    assert!(!stdout.contains("__XLOG_FUNCTION"), "{stdout}");
    assert!(!stdout.contains("__xlog_list_"), "{stdout}");
    assert!(!stdout.contains("__xlog_meta_"), "{stdout}");
}

#[test]
fn test_xlog_run_and_explain_resolve_whitespace_separated_udf_import() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("arithmetic.xlog"),
        "pred input(i64).\nfunc double(X) = X * 2.\ninput(1).\n",
    )
    .expect("write arithmetic module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "use\narithmetic.\npred answer(i64).\nanswer(Y) :- input(X), Y is double(X).\n?- answer(Y).\n",
    )
    .expect("write main program");

    let explain_output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        explain_output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&explain_output.stderr)
    );
    let explain_stdout = String::from_utf8(explain_output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value =
        serde_json::from_str(&explain_stdout).expect("valid explain json");
    assert_eq!(payload["rir"]["status"], "ok", "{explain_stdout}");
    assert_eq!(payload["optimizer"]["status"], "ok", "{explain_stdout}");

    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED runtime import parity: CUDA unavailable: {error}");
            return;
        }
    };

    let run_output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--output",
            "csv",
        ])
        .output()
        .expect("run xlog imported arithmetic UDF");
    assert!(
        run_output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );
    let run_stdout = String::from_utf8(run_output.stdout).expect("utf8 stdout");
    assert_eq!(run_stdout, "__xlog_query_0\ncomputed_1\n2\n\n");
}

#[test]
fn test_xlog_explain_keeps_normalized_helpers_out_of_source_provenance() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("list_member.xlog");
    std::fs::write(&program, "ok(X) :- member(X, [1, 2]).\n?- ok(X).\n")
        .expect("write list member fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(payload["ast"]["rules"], 1, "{stdout}");
    let source_rules = payload["rule_provenance"]
        .as_array()
        .expect("rule provenance array")
        .iter()
        .filter(|entry| entry["source_kind"] == "source")
        .collect::<Vec<_>>();
    assert_eq!(source_rules.len(), 1, "{stdout}");
    assert_eq!(source_rules[0]["head"], "ok(X)", "{stdout}");
    assert!(!stdout.contains("__xlog_list_"), "{stdout}");
}

#[test]
fn test_xlog_explain_reports_execution_normalization_errors_in_json() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("invalid_meta.xlog");
    std::fs::write(&program, "bad(F, A) :- functor(T, F, A).\n?- bad(F, A).\n")
        .expect("write invalid meta fixture");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json");
    assert!(
        output.status.success(),
        "xlog explain did not emit its diagnostic report: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    for section in ["rir", "optimizer", "wcoj"] {
        assert_analysis_not_available(&payload, section, "meta normalization error", &stdout);
    }
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

    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    let generated_reach_rule_ids = payload["rule_provenance"]
        .as_array()
        .expect("rule provenance array")
        .iter()
        .filter(|entry| entry["source_kind"] == "generated")
        .filter(|entry| {
            entry["head"]
                .as_str()
                .is_some_and(|head| head.starts_with("reach("))
        })
        .map(|entry| entry["rule_id"].as_str().expect("generated rule id"))
        .collect::<Vec<_>>();
    assert!(!generated_reach_rule_ids.is_empty(), "{stdout}");
    let proof_rule_ids = payload["proof_traces"][0]["rule_ids"]
        .as_array()
        .expect("proof rule ids");
    assert!(
        generated_reach_rule_ids.iter().all(|generated_id| {
            proof_rule_ids
                .iter()
                .any(|proof_id| proof_id.as_str() == Some(generated_id))
        }),
        "{stdout}"
    );
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
holds(a).
not_holds(a).
contradiction(X) :- holds(X), not_holds(X).
?- contradiction(X).
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
        stdout.contains("\"query\": \"contradiction(X)\""),
        "{stdout}"
    );
    assert!(stdout.contains("\"rule_ids\""), "{stdout}");
    assert!(stdout.contains("\"source_facts\""), "{stdout}");
    assert!(stdout.contains("holds(a)"), "{stdout}");
    assert!(stdout.contains("not_holds(a)"), "{stdout}");
}

#[test]
fn test_xlog_explain_json_resolves_module_path_imports_and_reports_epistemic_plan() {
    let root =
        std::env::temp_dir().join(format!("xlog_explain_module_path_{}", std::process::id()));
    let modules = root.join("modules");
    std::fs::create_dir_all(&modules).expect("create module dir");
    let module = modules.join("support.xlog");
    std::fs::write(
        &module,
        r#"
pred support(u32).
support(1).
"#,
    )
    .expect("write support module");
    let program = root.join("main.xlog");
    std::fs::write(
        &program,
        r#"
#pragma epistemic_mode = faeel
use support.
pred gated(u32).
gated(X) :- know know support(X).
?- gated(X).
"#,
    )
    .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            "--module-path",
            modules.to_str().expect("valid module path"),
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain json with module path");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(payload["ast"]["rules"], 2, "{stdout}");
    assert!(stdout.contains("\"epistemic\""), "{stdout}");
    assert!(stdout.contains("\"eir\""), "{stdout}");
    assert!(stdout.contains("\"gpu_plan\""), "{stdout}");
    assert!(stdout.contains("\"executable_plan\""), "{stdout}");
    assert!(
        stdout.contains("\"status\":\"ok\"") || stdout.contains("\"status\": \"ok\""),
        "{stdout}"
    );
    assert!(stdout.contains("\"epistemic_literal_count\""), "{stdout}");
    assert_eq!(payload["epistemic"]["gpu_plan"]["execution_backend"], "gpu");
    assert_eq!(
        payload["epistemic"]["gpu_plan"]["fallback_policy"],
        "reject_unsupported"
    );
    assert_eq!(
        payload["epistemic"]["executable_plan"]["execution_backend"],
        "gpu"
    );
    assert_eq!(
        payload["epistemic"]["executable_plan"]["fallback_policy"],
        "reject_unsupported"
    );
    for legacy_key in [
        "cpu_fallbacks",
        "cpu_fallbacks_zero",
        "cpu_fallback_is_zero",
        "cpu_fallback_total_zero",
    ] {
        assert!(!stdout.contains(legacy_key), "{legacy_key}: {stdout}");
    }
    assert!(
        stdout.contains("\"predicate\":\"support\"")
            || stdout.contains("\"predicate\": \"support\""),
        "{stdout}"
    );
}

#[test]
fn test_xlog_explain_json_compiles_an_imported_arithmetic_udf() {
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

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain with imported UDF");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(payload["optimizer"]["status"], "ok", "{stdout}");
}

#[test]
fn test_xlog_explain_unions_compatible_predicates_from_separate_modules() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("first.xlog"),
        "pred shared(symbol).\nshared(from_first).\n",
    )
    .expect("write first module");
    std::fs::write(
        fixture.path().join("second.xlog"),
        "pred shared(symbol).\nshared(from_second).\n",
    )
    .expect("write second module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(&program, "use first.\nuse second.\n?- shared(X).\n")
        .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain with compatible predicate contributions");

    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let payload: serde_json::Value = serde_json::from_str(&stdout).expect("valid explain json");
    assert_eq!(payload["ast"]["rules"], 2, "{stdout}");
    assert_eq!(payload["stratification"]["status"], "ok", "{stdout}");
    assert_eq!(payload["rir"]["status"], "ok", "{stdout}");
    assert_eq!(payload["optimizer"]["status"], "ok", "{stdout}");
}

#[test]
fn test_xlog_explain_rejects_incompatible_undeclared_predicate_schemas() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(fixture.path().join("first.xlog"), "shared(1).\n").expect("write first module");
    std::fs::write(fixture.path().join("second.xlog"), "shared(from_second).\n")
        .expect("write second module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(&program, "use first.\nuse second.\n?- shared(X).\n")
        .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain with incompatible predicate contributions");

    assert!(
        !output.status.success(),
        "xlog explain unexpectedly succeeded"
    );
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(stderr.contains("error[E0412]"), "{stderr}");
    assert!(stderr.contains("shared/1"), "{stderr}");
    assert!(stderr.contains("first"), "{stderr}");
    assert!(stderr.contains("second"), "{stderr}");
    assert!(!stderr.contains("Type mismatch in fact"), "{stderr}");
    assert!(!stderr.contains("Symbol("), "{stderr}");
}

#[test]
fn test_xlog_explain_attributes_inferred_expression_schema_conflicts() {
    let cases = [
        (
            "arithmetic",
            "shared(X) :- X is cast(1, u32).\n",
            "shared(X) :- X is cast(1, u64).\n",
            "u32",
            "u64",
        ),
        (
            "aggregate",
            "first_source(1).\nshared(min(X)) :- first_source(X).\n",
            "second_source(1.0).\nshared(logsumexp(X)) :- second_source(X).\n",
            "u32",
            "f64",
        ),
    ];

    for (case, first_source, second_source, first_type, second_type) in cases {
        let fixture = TempDir::new().expect("create fixture directory");
        std::fs::write(fixture.path().join("first.xlog"), first_source)
            .expect("write first module");
        std::fs::write(fixture.path().join("second.xlog"), second_source)
            .expect("write second module");
        let program = fixture.path().join("main.xlog");
        std::fs::write(&program, "use first.\nuse second.\n?- shared(X).\n")
            .expect("write main program");

        let output = cargo_bin_cmd!("xlog")
            .args([
                "explain",
                "--format",
                "json",
                "--module-path",
                fixture.path().to_str().expect("valid module path"),
                program.to_str().expect("valid program path"),
            ])
            .output()
            .expect("run xlog explain with incompatible expression-derived schemas");

        assert!(
            !output.status.success(),
            "xlog explain unexpectedly succeeded for {case} inference"
        );
        let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
        assert!(stderr.contains("error[E0412]"), "{case}: {stderr}");
        assert!(stderr.contains("shared/1"), "{case}: {stderr}");
        assert!(stderr.contains("module `first`"), "{case}: {stderr}");
        assert!(stderr.contains("module `second`"), "{case}: {stderr}");
        assert!(stderr.contains(first_type), "{case}: {stderr}");
        assert!(stderr.contains(second_type), "{case}: {stderr}");
    }
}

#[test]
fn test_xlog_explain_retains_schema_evidence_beside_user_functions() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("first.xlog"),
        "func first_value(X) = cast(X, u32).\nshared(1, X) :- X is first_value(1).\n",
    )
    .expect("write first module");
    std::fs::write(
        fixture.path().join("second.xlog"),
        "func second_value(X) = cast(X, u32).\nshared(from_second, X) :- X is second_value(1).\n",
    )
    .expect("write second module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(&program, "use first.\nuse second.\n?- shared(X, Y).\n")
        .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain with mixed inferred schema evidence");

    assert!(
        !output.status.success(),
        "xlog explain unexpectedly accepted incompatible independent schema evidence"
    );
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(stderr.contains("error[E0412]"), "{stderr}");
    assert!(stderr.contains("shared/2"), "{stderr}");
    assert!(stderr.contains("module `first`"), "{stderr}");
    assert!(stderr.contains("module `second`"), "{stderr}");
    assert!(stderr.contains("u32"), "{stderr}");
    assert!(stderr.contains("symbol"), "{stderr}");
}

#[test]
fn test_xlog_explain_attributes_an_entry_schema_conflict() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(fixture.path().join("library.xlog"), "shared(1).\n")
        .expect("write library module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "use library.\nshared(from_entry).\n?- shared(X).\n",
    )
    .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--format",
            "json",
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain with an incompatible entry contribution");

    assert!(
        !output.status.success(),
        "xlog explain unexpectedly succeeded"
    );
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(stderr.contains("error[E0412]"), "{stderr}");
    assert!(stderr.contains("shared/1"), "{stderr}");
    assert!(stderr.contains("module `library`"), "{stderr}");
    assert!(stderr.contains("module `main`"), "{stderr}");
}

#[test]
fn test_xlog_explain_warns_on_ignored_imported_module_pragma() {
    let root =
        std::env::temp_dir().join(format!("xlog_explain_import_pragma_{}", std::process::id()));
    let modules = root.join("modules");
    std::fs::create_dir_all(&modules).expect("create module dir");
    let module = modules.join("corpus.xlog");
    std::fs::write(
        &module,
        r#"
#pragma magic_sets = auto
pred edge(u32, u32).
edge(1, 2).
"#,
    )
    .expect("write corpus module");
    let program = root.join("main.xlog");
    std::fs::write(
        &program,
        r#"
#pragma magic_sets = on
use corpus.
pred reach(u32, u32).
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).
?- reach(1, Y).
"#,
    )
    .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "explain",
            "--module-path",
            modules.to_str().expect("valid module path"),
            program.to_str().expect("valid program path"),
        ])
        .output()
        .expect("run xlog explain with imported-module pragma");
    assert!(
        output.status.success(),
        "xlog explain failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains(
            "warning[W0510]: `#pragma magic_sets` in imported module `corpus` is ignored"
        ),
        "{stderr}"
    );
    assert!(
        stderr.contains("pragmas apply only when declared in the entry file"),
        "{stderr}"
    );
    // The entry file's own pragma is authoritative and must not warn.
    assert_eq!(stderr.matches("warning[W0510]").count(), 1, "{stderr}");
}
