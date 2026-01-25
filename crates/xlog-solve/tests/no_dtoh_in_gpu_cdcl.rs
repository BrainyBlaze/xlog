use std::path::PathBuf;

// Guardrail: the production GPU CDCL verifier path must not perform device->host reads.
#[test]
fn no_device_to_host_reads_in_gpu_cdcl() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("gpu_cdcl.rs");

    let text = std::fs::read_to_string(&path).expect("read gpu_cdcl.rs");

    assert!(
        !text.contains("dtoh_sync_copy_into"),
        "GpuCdclSolver must not use dtoh_sync_copy_into (found in {})",
        path.display()
    );
    assert!(
        !text.contains("copy_to_host"),
        "GpuCdclSolver must not use copy_to_host (found in {})",
        path.display()
    );
    assert!(
        !text.contains("dtoh"),
        "GpuCdclSolver must not reference dtoh transfers (found in {})",
        path.display()
    );
}
