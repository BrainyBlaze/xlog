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

#[test]
fn test_build_script_generates_embedded_portable_ptx_fallback() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let build_rs = manifest_dir.join("build.rs");
    let build_contents = fs::read_to_string(&build_rs).expect("read build.rs");
    assert!(
        build_contents.contains("embedded_kernel_data.rs"),
        "build.rs should generate embedded portable PTX metadata for cargo-installed binaries"
    );
    assert!(
        build_contents.contains("include_str!"),
        "embedded fallback should use include_str! so portable PTX is compiled into the binary"
    );

    let lib_rs = manifest_dir.join("src/lib.rs");
    let lib_contents = fs::read_to_string(&lib_rs).expect("read lib.rs");
    assert!(
        lib_contents.contains("embedded_kernel_data"),
        "xlog-cuda should compile the generated embedded kernel metadata into the crate"
    );
    assert!(
        lib_contents.contains("OUT_DIR"),
        "embedded kernel metadata should be included from Cargo OUT_DIR"
    );
}

#[test]
fn test_build_script_caps_wcoj_register_count() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let build_rs = manifest_dir.join("build.rs");
    let build_contents = fs::read_to_string(&build_rs).expect("read build.rs");
    assert!(
        build_contents.contains("--maxrregcount=64"),
        "build.rs should cap WCOJ ptxas register allocation for G_W64 M_W64.6"
    );
    assert!(
        build_contents.contains("name == \"wcoj\""),
        "the register cap should be scoped to the WCOJ kernel module"
    );
}
