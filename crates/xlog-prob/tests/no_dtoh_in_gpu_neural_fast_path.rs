use std::path::PathBuf;

// Guardrail: the production GPU neural fast-path implementation must not rely on device->host reads
// or Python-side CPU tensor extraction helpers.
#[test]
fn no_device_to_host_reads_in_gpu_neural_fast_path() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("neural_fast_path.rs");

    let text = std::fs::read_to_string(&path).expect("read neural_fast_path.rs");

    assert!(
        !text.contains("dtoh_sync_copy_into"),
        "GPU neural fast-path must not use dtoh_sync_copy_into (found in {})",
        path.display()
    );
    assert!(
        !text.contains("copy_to_host"),
        "GPU neural fast-path must not use copy_to_host (found in {})",
        path.display()
    );
    assert!(
        !text.contains("dtoh"),
        "GPU neural fast-path must not reference dtoh transfers (found in {})",
        path.display()
    );
    assert!(
        !text.contains("tolist"),
        "GPU neural fast-path must not reference .tolist() (found in {})",
        path.display()
    );
}

