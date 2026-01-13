use std::fs;
use std::path::PathBuf;

use xlog_prob::kc::d4::D4Compiler;
use xlog_prob::kc::ddnnf::DecisionDnnf;
use xlog_prob::xgcf::Xgcf;

fn make_temp_dir(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("{}-{}-{}", prefix, pid, nanos));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_bundled_d4_compiles_real_ddnnf() {
    let dir = make_temp_dir("xlog-real-d4");

    let cnf_path = dir.join("in.cnf");
    fs::write(&cnf_path, "c test\np cnf 1 1\n1 0\n").unwrap();
    let out_path = dir.join("out.nnf");

    let compiler = D4Compiler::bundled().unwrap();
    compiler.compile_ddnnf(&cnf_path, &out_path).unwrap();

    let nnf = fs::read_to_string(&out_path).unwrap();
    let ddnnf = DecisionDnnf::parse_str(&nnf).unwrap();
    let xgcf = Xgcf::from_ddnnf(&ddnnf).unwrap();

    let p = 0.25_f64;
    let log_wmc = xgcf
        .eval_log_wmc(|var| match var {
            1 => (p.ln(), (1.0 - p).ln()),
            _ => panic!("unexpected var {}", var),
        })
        .unwrap();
    assert!((log_wmc - p.ln()).abs() < 1e-9, "log_wmc={}", log_wmc);

    fs::remove_dir_all(&dir).ok();
}

