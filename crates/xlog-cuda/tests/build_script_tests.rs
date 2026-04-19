use std::fs;
use std::path::PathBuf;

#[test]
fn test_build_script_includes_circuit_and_mc_sample() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // build.rs uses include!() from kernel_manifest_data.rs — check the manifest.
    let manifest = manifest_dir.join("src/kernel_manifest_data.rs");
    let contents = fs::read_to_string(&manifest).expect("read kernel_manifest_data.rs");

    assert!(
        contents.contains("\"circuit\""),
        "kernel manifest should list circuit kernel"
    );
    assert!(
        contents.contains("\"mc_sample\""),
        "kernel manifest should list mc_sample kernel"
    );
    assert!(
        contents.contains("\"d4\""),
        "kernel manifest should list d4 kernel"
    );

    // Also verify build.rs references the manifest.
    let build_rs = manifest_dir.join("build.rs");
    let build_contents = fs::read_to_string(&build_rs).expect("read build.rs");
    assert!(
        build_contents.contains("kernel_manifest_data.rs"),
        "build.rs should include the kernel manifest"
    );
    assert!(
        build_contents.contains("manifest_dir.join(\"kernels\")"),
        "build.rs should prefer package-local kernels/ when running from a packaged crate"
    );
    assert!(
        build_contents.contains(".join(\"kernels\")"),
        "build.rs should still support the workspace-root kernels/ layout"
    );
}

#[test]
fn test_package_manifest_includes_cuda_sources() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = manifest_dir.join("Cargo.toml");
    let contents = fs::read_to_string(&cargo_toml).expect("read Cargo.toml");

    assert!(
        contents.contains("\"kernels/**\""),
        "xlog-cuda package must include CUDA sources so cargo publish --verify works"
    );
}
