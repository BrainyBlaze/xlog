#![cfg(feature = "host-io")]

use assert_cmd::Command;
use std::path::Path;
use tempfile::TempDir;
use xlog_cuda::CudaDevice;

fn cuda_available() -> bool {
    match CudaDevice::new(0) {
        Ok(_) => true,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            false
        }
    }
}

fn write_import_fixture(root: &Path) -> std::path::PathBuf {
    let module_root = root.join("modules");
    let entry_root = root.join("entry");
    std::fs::create_dir_all(&module_root).expect("create module directory");
    std::fs::create_dir_all(&entry_root).expect("create entry directory");
    std::fs::write(
        module_root.join("direct_rules.xlog"),
        "pred source_anchor(symbol).\npred direct_ready(symbol).\n\
         direct_ready(Claim) :- source_anchor(Claim).\n",
    )
    .expect("write directly imported module");
    std::fs::write(
        module_root.join("base_rules.xlog"),
        "pred source_anchor(symbol).\npred source_ready(symbol).\n\
         pred imported_marker(symbol).\nimported_marker(claim).\n\
         source_ready(Claim) :- source_anchor(Claim).\n",
    )
    .expect("write leaf module");
    std::fs::write(
        module_root.join("transitive_rules.xlog"),
        "use base_rules.\npred transitive_ready(symbol).\n\
         transitive_ready(Claim) :- source_ready(Claim).\n",
    )
    .expect("write intermediate module");
    let program = entry_root.join("main.datalog");
    std::fs::write(
        &program,
        "use direct_rules.\nuse transitive_rules.\n0.97::source_anchor(claim).\n\
         query(source_anchor(claim)).\nquery(direct_ready(claim)).\n\
         query(transitive_ready(claim)).\nquery(imported_marker(claim)).\n",
    )
    .expect("write probabilistic entry program");
    program
}

fn run_prob(
    program: &Path,
    module_path: Option<&Path>,
    arguments: &[&str],
) -> std::process::Output {
    let mut command = Command::cargo_bin("xlog").expect("xlog binary");
    command.arg("prob").arg(program);
    if let Some(module_path) = module_path {
        command.arg("--module-path").arg(module_path);
    }
    command.args(arguments).output().expect("run xlog prob")
}

fn run_prob_success(
    program: &Path,
    module_path: Option<&Path>,
    arguments: &[&str],
) -> (String, String) {
    let output = run_prob(program, module_path, arguments);
    assert!(
        output.status.success(),
        "xlog prob failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    (stdout, stderr)
}

fn run_prob_json(
    program: &Path,
    module_path: Option<&Path>,
    arguments: &[&str],
) -> (serde_json::Value, String) {
    let (stdout, stderr) = run_prob_success(program, module_path, arguments);
    let payload = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("invalid probability JSON: {error}\nstdout:\n{stdout}"));
    (payload, stderr)
}

fn json_probability(payload: &serde_json::Value, atom: &str) -> f64 {
    payload["queries"]
        .as_array()
        .expect("queries array")
        .iter()
        .find(|query| query["atom"].as_str() == Some(atom))
        .unwrap_or_else(|| panic!("missing query {atom} in {payload}"))["prob"]
        .as_f64()
        .expect("numeric query probability")
}

#[test]
fn test_xlog_prob_exact_and_mc() {
    if !cuda_available() {
        return;
    }

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let exact_program = repo_root.join("examples/prob/01-wet-conditioning.xlog");
    // GPU-resident MC over a recursive program (supported fragment). The prior
    // `04-nonmonotone-mc.xlog` used negation, which the resident engine now
    // rejects fail-closed (no host-orchestrated fallback).
    let mc_program = repo_root.join("examples/prob/04-recursive-mc.xlog");

    // Use Command::cargo_bin which resolves via CARGO_BIN_EXE_xlog,
    // inheriting the same feature flags (including host-io) from the test build.
    let mut cmd = Command::cargo_bin("xlog").expect("xlog binary");
    cmd.args([
        "prob",
        exact_program.to_str().expect("valid path"),
        "--prob-engine",
        "exact_ddnnf",
    ]);
    cmd.assert().success();

    let mut cmd = Command::cargo_bin("xlog").expect("xlog binary");
    cmd.args([
        "prob",
        mc_program.to_str().expect("valid path"),
        "--prob-engine",
        "mc",
        "--samples",
        "1000",
        "--seed",
        "42",
    ]);
    cmd.assert().success();
}

#[test]
fn test_xlog_prob_exact_evaluates_direct_and_transitive_imports() {
    if !cuda_available() {
        return;
    }

    let fixture = TempDir::new().expect("create fixture directory");
    let program = write_import_fixture(fixture.path());
    let module_path = fixture.path().join("modules");
    let (payload, _) = run_prob_json(
        &program,
        Some(&module_path),
        &["--prob-engine", "exact_ddnnf", "--output", "json"],
    );
    let source_probability = json_probability(&payload, "source_anchor(claim)");

    assert_eq!(payload["engine"], "exact_ddnnf");
    assert!((source_probability - 0.97).abs() < 1e-12, "{payload}");
    assert!(
        (json_probability(&payload, "direct_ready(claim)") - source_probability).abs() < 1e-12,
        "{payload}"
    );
    assert!(
        (json_probability(&payload, "transitive_ready(claim)") - source_probability).abs() < 1e-12,
        "{payload}"
    );
    assert_eq!(
        json_probability(&payload, "imported_marker(claim)"),
        1.0,
        "{payload}"
    );
}

#[test]
fn test_xlog_prob_mc_evaluates_direct_and_transitive_imports() {
    if !cuda_available() {
        return;
    }

    let fixture = TempDir::new().expect("create fixture directory");
    let program = write_import_fixture(fixture.path());
    let module_path = fixture.path().join("modules");
    let (payload, _) = run_prob_json(
        &program,
        Some(&module_path),
        &[
            "--prob-engine",
            "mc",
            "--samples",
            "2000",
            "--seed",
            "123",
            "--output",
            "json",
        ],
    );
    let source_probability = json_probability(&payload, "source_anchor(claim)");

    assert_eq!(payload["engine"], "mc");
    assert_eq!(payload["mc_engine"], "gpu-resident");
    assert_eq!(payload["total_samples"], 2000);
    assert_eq!(payload["seed"], 123);
    assert!((0.94..=0.99).contains(&source_probability), "{payload}");
    assert_eq!(
        json_probability(&payload, "direct_ready(claim)"),
        source_probability,
        "{payload}"
    );
    assert_eq!(
        json_probability(&payload, "transitive_ready(claim)"),
        source_probability,
        "{payload}"
    );
    assert_eq!(
        json_probability(&payload, "imported_marker(claim)"),
        1.0,
        "{payload}"
    );
}

#[test]
fn test_xlog_prob_fails_when_transitive_imported_module_is_missing() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("wrapper.xlog"),
        "use unavailable_rules.\npred wrapper_ready(symbol).\n",
    )
    .expect("write wrapper module");
    let program = fixture.path().join("main.datalog");
    std::fs::write(
        &program,
        "use wrapper.\n0.97::source_anchor(claim).\n\
         query(source_anchor(claim)).\n",
    )
    .expect("write probabilistic entry program");

    let output = run_prob(&program, None, &[]);

    assert!(!output.status.success(), "xlog prob unexpectedly succeeded");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(stderr.contains("Module resolution failed"), "{stderr}");
    assert!(
        stderr.contains("error[E0400]: module not found: `unavailable_rules`"),
        "{stderr}"
    );
    assert!(!stderr.contains("module not found: `main`"), "{stderr}");
}

#[test]
fn test_xlog_prob_rejects_imported_export_with_hidden_support() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("library.xlog"),
        "private pred hidden(symbol).\npred visible(symbol).\n\
         hidden(claim).\nvisible(Claim) :- hidden(Claim).\n",
    )
    .expect("write imported module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "use library.\n0.9::control(claim).\nquery(visible(claim)).\n\
         query(control(claim)).\n",
    )
    .expect("write probabilistic entry program");

    let output = run_prob(
        &program,
        None,
        &["--prob-engine", "exact_ddnnf", "--output", "json"],
    );

    assert!(!output.status.success(), "xlog prob unexpectedly succeeded");
    assert!(output.stdout.is_empty(), "unexpected stdout: {output:?}");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(stderr.contains("Module resolution failed"), "{stderr}");
    assert!(stderr.contains("error[E0406]"), "{stderr}");
    assert!(stderr.contains("`visible`"), "{stderr}");
    assert!(stderr.contains("`hidden`"), "{stderr}");
    assert!(stderr.contains("`library`"), "{stderr}");
}

#[test]
fn test_xlog_prob_rejects_transitively_imported_program_level_constructs() {
    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("probabilistic_support.xlog"),
        "pred imported_coin(symbol).\npred left(symbol).\npred right(symbol).\n\
         0.5::imported_coin(claim).\n\
         0.4::left(claim); 0.6::right(claim).\n\
         evidence(imported_coin(claim), true).\n\
         :- imported_coin(claim), not left(claim).\n",
    )
    .expect("write probabilistic support module");
    std::fs::write(
        fixture.path().join("wrapper.xlog"),
        "use probabilistic_support.\npred passthrough(symbol).\n\
         passthrough(Claim) :- control(Claim).\n",
    )
    .expect("write wrapper module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "use wrapper.\n0.9::control(claim).\nquery(control(claim)).\n",
    )
    .expect("write probabilistic entry program");

    for engine in ["exact_ddnnf", "mc"] {
        let output = run_prob(&program, None, &["--prob-engine", engine]);

        assert!(
            !output.status.success(),
            "xlog prob unexpectedly succeeded for {engine}"
        );
        assert!(output.stdout.is_empty(), "unexpected stdout: {output:?}");
        let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
        assert!(stderr.contains("Module resolution failed"), "{stderr}");
        assert!(stderr.contains("error[E0405]"), "{stderr}");
        assert!(stderr.contains("`probabilistic_support`"), "{stderr}");
        assert!(
            stderr.contains(
                "annotated disjunctions, evidence statements, integrity constraints, probabilistic facts"
            ),
            "{stderr}"
        );
    }
}

#[test]
fn test_xlog_prob_mc_pragmas_json_and_cli_overrides() {
    if !cuda_available() {
        return;
    }

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("mc_overrides.xlog");
    std::fs::write(
        &program,
        r#"
#pragma prob_engine = mc
#pragma prob_samples = 8
#pragma prob_seed = 1
#pragma prob_confidence = 0.80
#pragma prob_method = rejection
0.5::rain().
query(rain()).
"#,
    )
    .expect("write approximate CLI fixture");

    let mut cmd = Command::cargo_bin("xlog").expect("xlog binary");
    let output = cmd
        .args([
            "prob",
            program.to_str().expect("valid path"),
            "--samples",
            "16",
            "--seed",
            "2",
            "--confidence",
            "0.90",
            "--prob-method",
            "rejection",
            "--output",
            "json",
        ])
        .output()
        .expect("run xlog prob json");
    assert!(
        output.status.success(),
        "xlog prob failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("\"engine\": \"mc\""), "{stdout}");
    assert!(stdout.contains("\"total_samples\": 16"), "{stdout}");
    assert!(stdout.contains("\"seed\": 2"), "{stdout}");
    assert!(stdout.contains("\"confidence\": 0.9"), "{stdout}");
    assert!(
        stdout.contains("\"sampling_method\": \"rejection\""),
        "{stdout}"
    );
    assert!(stdout.contains("\"stderr\""), "{stdout}");
    assert!(stdout.contains("\"ci_low\""), "{stdout}");
    assert!(stdout.contains("\"evidence_samples\""), "{stdout}");
}

#[test]
fn test_xlog_prob_warns_on_ignored_imported_module_pragma_without_module_path() {
    if !cuda_available() {
        return;
    }

    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("library.xlog"),
        "#pragma prob_engine = exact_ddnnf\n#pragma prob_seed = 9\n\
         pred source_anchor(symbol).\npred derived_ready(symbol).\n\
         derived_ready(Claim) :- source_anchor(Claim).\n",
    )
    .expect("write imported module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "#pragma prob_engine = mc\n#pragma prob_samples = 64\n#pragma prob_seed = 123\n\
         use library.\n0.97::source_anchor(claim).\nquery(source_anchor(claim)).\n\
         query(derived_ready(claim)).\n",
    )
    .expect("write main program");

    let (payload, stderr) = run_prob_json(&program, None, &["--output", "json"]);
    assert_eq!(payload["engine"], "mc");
    assert_eq!(payload["total_samples"], 64);
    assert_eq!(payload["seed"], 123);
    assert_eq!(
        json_probability(&payload, "derived_ready(claim)"),
        json_probability(&payload, "source_anchor(claim)"),
        "{payload}"
    );
    assert!(
        stderr.contains(
            "warning[W0510]: `#pragma prob_engine` in imported module `library` is ignored"
        ),
        "{stderr}"
    );
    assert!(
        stderr.contains(
            "warning[W0510]: `#pragma prob_seed` in imported module `library` is ignored"
        ),
        "{stderr}"
    );
    assert_eq!(stderr.matches("warning[W0510]").count(), 2, "{stderr}");
    assert_eq!(
        stderr
            .matches("pragmas apply only when declared in the entry file")
            .count(),
        2,
        "{stderr}"
    );
}
