use assert_cmd::cargo::cargo_bin_cmd;
use std::collections::BTreeSet;
use tempfile::TempDir;
use xlog_cuda::CudaDevice;

#[test]
fn predicate_function_returns_every_relation_value_through_xlog_run() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
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
