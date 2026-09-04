use assert_cmd::cargo::cargo_bin_cmd;
use serde_json::Value;
use tempfile::TempDir;
use xlog_cuda::CudaDevice;

#[test]
fn deterministic_json_run_exports_requested_materialized_relations() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED materialized run JSON: CUDA unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("materialized_run.xlog");
    std::fs::write(
        &program,
        r#"
pred seed(symbol).
pred reached(symbol).
seed(alpha).
reached(X) :- seed(X).
?- reached(X).
"#,
    )
    .expect("write program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--output",
            "json",
            "--materialize-relation",
            "reached",
            "--materialize-relation",
            "seed",
            "--materialize-relation",
            "seed",
        ])
        .output()
        .expect("run xlog materialized JSON export");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).expect("valid run JSON");

    assert_eq!(payload["schema_version"], "xlog.deterministic-run.v1");
    assert_eq!(payload["fixpoint_state"], "complete");
    assert_eq!(payload["queries"][0]["relation_name"], "__xlog_query_0");
    assert_eq!(payload["queries"][0]["columns"], serde_json::json!(["X"]));
    assert_eq!(
        payload["queries"][0]["scalar_types"],
        serde_json::json!(["symbol"])
    );
    assert_eq!(
        payload["queries"][0]["rows"],
        serde_json::json!([["alpha"]])
    );
    assert_eq!(
        payload["materialized_relations"],
        serde_json::json!([
            {
                "relation_name": "reached",
                "scalar_types": ["symbol"],
                "rows": [["alpha"]]
            },
            {
                "relation_name": "seed",
                "scalar_types": ["symbol"],
                "rows": [["alpha"]]
            }
        ])
    );
}

#[test]
fn recursive_deterministic_json_stats_count_serialized_query_rows() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED recursive deterministic JSON stats: CUDA unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("recursive_stats.xlog");
    std::fs::write(
        &program,
        r#"
pred edge(symbol, symbol).
pred reachable(symbol).
pred answer(symbol).
pred audit(symbol).
edge(seed, step).
reachable(seed).
reachable(To) :- reachable(From), edge(From, To).
answer(yes) :- reachable(step).
audit(ok).
?- answer(Value).
?- audit(Value).
"#,
    )
    .expect("write program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--output",
            "json",
            "--materialize-relation",
            "answer",
            "--stats",
            "--stats-format",
            "json",
        ])
        .output()
        .expect("run recursive deterministic JSON export");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).expect("valid run JSON");
    let stats_line = String::from_utf8(output.stderr)
        .expect("UTF-8 stats")
        .lines()
        .next()
        .expect("stats line")
        .to_owned();
    let stats: Value = serde_json::from_str(&stats_line).expect("valid stats JSON");
    let serialized_query_rows: u64 = payload["queries"]
        .as_array()
        .expect("query array")
        .iter()
        .map(|query| query["rows"].as_array().expect("query rows").len() as u64)
        .sum();

    assert_eq!(serialized_query_rows, 2);
    assert_eq!(stats["output_rows"].as_u64(), Some(serialized_query_rows));
}

#[test]
fn deterministic_json_run_preserves_every_scalar_type() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED typed materialized run JSON: CUDA unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("typed_materialized_run.xlog");
    std::fs::write(
        &program,
        r#"
pred typed(u32, u64, i32, i64, f32, f64, bool, symbol).
typed(7, 8, -9, -10, 1.5, -2.25, true, alpha).
?- typed(A, B, C, D, E, F, G, H).
"#,
    )
    .expect("write typed program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--output",
            "json",
            "--materialize-relation",
            "typed",
        ])
        .output()
        .expect("run typed XLOG materialized JSON export");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&output.stdout).expect("valid run JSON");

    let expected_types =
        serde_json::json!(["u32", "u64", "i32", "i64", "f32", "f64", "bool", "symbol"]);
    let expected_rows = serde_json::json!([[7, 8, -9, -10, 1.5, -2.25, true, "alpha"]]);
    assert_eq!(payload["queries"][0]["scalar_types"], expected_types);
    assert_eq!(payload["queries"][0]["rows"], expected_rows);
    assert_eq!(
        payload["materialized_relations"][0]["scalar_types"],
        expected_types
    );
    assert_eq!(payload["materialized_relations"][0]["rows"], expected_rows);
}

#[test]
fn materialized_relation_requires_json_output_before_cuda_initialization() {
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("invalid_materialized_output.xlog");
    std::fs::write(&program, "pred seed(u32). seed(1). ?- seed(X).").expect("write program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--materialize-relation",
            "seed",
        ])
        .output()
        .expect("run invalid materialized output request");

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("--materialize-relation requires --output json"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn deterministic_json_run_rejects_unknown_materialized_relation() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED unknown relation JSON: CUDA unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("unknown_materialized_relation.xlog");
    std::fs::write(&program, "pred seed(u32). seed(1). ?- seed(X).").expect("write program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--output",
            "json",
            "--materialize-relation",
            "missing",
        ])
        .output()
        .expect("run unknown relation request");

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(
            "requested materialized relation 'missing' is not present in the completed runtime store"
        ),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
