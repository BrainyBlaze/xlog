use std::fs;
use std::path::Path;

#[test]
fn no_device_to_host_reads_in_exact_gpu_module() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(manifest_dir).join("src").join("exact_gpu.rs");
    let text = fs::read_to_string(path).expect("read exact_gpu.rs");
    assert!(
        !text.contains("dtoh_sync_copy_into"),
        "exact_gpu.rs must not use dtoh_sync_copy_into"
    );
    assert!(
        !text.contains("copy_to_host"),
        "exact_gpu.rs must not use copy_to_host"
    );
    assert!(
        !text.contains("dtoh"),
        "exact_gpu.rs must not reference dtoh transfers"
    );
}
