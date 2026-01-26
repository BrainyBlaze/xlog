use std::fs;
use std::path::PathBuf;

#[test]
fn test_build_script_includes_circuit_and_mc_sample() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let build_rs = manifest_dir.join("build.rs");
    let contents = fs::read_to_string(&build_rs).expect("read build.rs");

    assert!(
        contents.contains("\"circuit\""),
        "build.rs should list circuit kernel for PTX compilation"
    );
    assert!(
        contents.contains("\"mc_sample\""),
        "build.rs should list mc_sample kernel for PTX compilation"
    );
    assert!(
        contents.contains("\"d4\""),
        "build.rs should list d4 kernel for PTX compilation"
    );
}
