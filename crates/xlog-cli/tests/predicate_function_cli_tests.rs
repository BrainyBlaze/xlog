use assert_cmd::cargo::cargo_bin_cmd;
use std::collections::BTreeSet;
use tempfile::TempDir;
use xlog_cuda::CudaDevice;

fn cuda_device_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(device) => Some(device),
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            None
        }
    }
}

#[test]
fn predicate_function_returns_every_relation_value_through_xlog_run() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("predicate_function.xlog");
    std::fs::write(
        &program,
        "pred child(u32).\n\
         pred parent(u32, u32).\n\
         pred answer(u32, u32).\n\
         func get_parent(Child) = P :- parent(Child, P).\n\
         child(1).\n\
         child(2).\n\
         child(3).\n\
         parent(1, 2).\n\
         parent(1, 3).\n\
         parent(2, 4).\n\
         answer(Child, P) :- child(Child), P is get_parent(Child).\n\
         ?- answer(Child, P).\n",
    )
    .expect("write predicate-function program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run predicate-function program");

    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("__xlog_query_0"), "{stdout}");
    let rows: Vec<(u32, u32)> = stdout
        .lines()
        .filter_map(|line| {
            let mut fields = line.split('|').skip(1).map(str::trim);
            Some((fields.next()?.parse().ok()?, fields.next()?.parse().ok()?))
        })
        .collect();
    assert_eq!(rows.len(), 3, "{stdout}");
    assert_eq!(
        rows.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([(1, 2), (1, 3), (2, 4)]),
        "{stdout}"
    );
}

#[test]
fn predicate_function_result_keeps_the_existing_csv_header() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("predicate_function_header.xlog");
    std::fs::write(
        &program,
        "pred parent(u32, u32).\n\
         pred answer(u32).\n\
         func get_parent(Child) = Parent :- parent(Child, Parent).\n\
         parent(1, 2).\n\
         answer(Parent) :- Parent is get_parent(1).\n\
         ?- answer(Parent).\n",
    )
    .expect("write predicate-function header program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--output",
            "csv",
        ])
        .output()
        .expect("run predicate-function header program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(stdout, "__xlog_query_0\ncomputed_2\n2\n\n");
}

#[test]
fn predicate_function_wrapper_omits_the_redundant_result_column() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("predicate_function_wrapper.xlog");
    std::fs::write(
        &program,
        "pred parent(u32, u32).\n\
         pred answer(u32).\n\
         func get_parent(Child) = Parent :- parent(Child, Parent).\n\
         func lookup(Child) = Result :- Result is get_parent(Child).\n\
         parent(1, 2).\n\
         answer(Result) :- Result is lookup(1).\n\
         ?- answer(Result).\n",
    )
    .expect("write predicate-function wrapper program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--output",
            "csv",
        ])
        .output()
        .expect("run predicate-function wrapper program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(stdout, "__xlog_query_0\ncomputed_2\n2\n\n");
}

#[test]
fn violated_predicate_function_constraint_reports_the_authored_body() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("predicate_function_constraint.xlog");
    std::fs::write(
        &program,
        "pred parent(u32, u32).\n\
         func get_parent(Child) = Parent :- parent(Child, Parent).\n\
         parent(1, 2).\n\
         :- parent(9, Missing).\n\
         :- Parent is get_parent(1).\n",
    )
    .expect("write predicate-function constraint program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run predicate-function constraint program");

    assert!(!output.status.success(), "violated constraint must fail");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("Constraint 1 violated: :- Parent is get_parent(1)."),
        "{stderr}"
    );
    assert!(!stderr.contains("__XLOG_FUNCTION"), "{stderr}");
    assert!(!stderr.contains("Variable("), "{stderr}");
}

#[test]
fn epistemic_predicate_function_constraint_reports_the_authored_body() {
    let Some(_device) = cuda_device_or_skip() else {
        return;
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("epistemic_predicate_function_constraint.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred parent(u32, u32).\n\
         pred visible(u32, u32).\n\
         func get_parent(Child) = Parent :- parent(Child, Parent).\n\
         parent(1, 2).\n\
         visible(Child, Parent) :-\n\
           parent(Child, Parent), know parent(Child, Parent).\n\
         :- parent(9, Missing).\n\
         :- Parent is get_parent(1).\n\
         ?- visible(Child, Parent).\n",
    )
    .expect("write epistemic predicate-function constraint program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run epistemic predicate-function constraint program");

    assert!(!output.status.success(), "violated constraint must fail");
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("Constraint 1 violated: :- Parent is get_parent(1)."),
        "{stderr}"
    );
    assert!(!stderr.contains("__XLOG_FUNCTION"), "{stderr}");
    assert!(!stderr.contains("Variable("), "{stderr}");
}
