use std::path::PathBuf;

// Guardrail: the production GPU equivalence verifier path must not perform device->host reads.
#[test]
fn no_device_to_host_reads_in_gpu_equivalence() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("compilation");
    path.push("validation.rs");

    let text = std::fs::read_to_string(&path).expect("read compilation/validation.rs");

    assert!(
        !text.contains("dtoh_sync_copy_into"),
        "GPU equivalence validation must not use dtoh_sync_copy_into (found in {})",
        path.display()
    );
    assert!(
        !text.contains("copy_to_host"),
        "GPU equivalence validation must not use copy_to_host (found in {})",
        path.display()
    );
    assert!(
        !text.contains("dtoh"),
        "GPU equivalence validation must not reference dtoh transfers (found in {})",
        path.display()
    );
}
