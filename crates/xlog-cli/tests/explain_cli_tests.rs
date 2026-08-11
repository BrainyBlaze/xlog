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
    assert!(
        payload["rir"]["status"]
            .as_str()
            .expect("RIR status string")
            .contains("meta normalization error"),
        "{stdout}"
    );
    assert_eq!(payload["optimizer"]["status"], "not_available", "{stdout}");
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
