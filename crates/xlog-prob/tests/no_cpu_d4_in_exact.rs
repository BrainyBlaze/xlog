use std::path::PathBuf;

#[test]
fn exact_path_uses_gpu_only() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("exact.rs");

    let text = std::fs::read_to_string(&path).expect("read exact.rs");
    assert!(!text.contains("D4Compiler"));
    assert!(!text.contains("TempDirGuard"));
    assert!(!text.contains("in.cnf"));
    assert!(!text.contains("out.nnf"));
}
