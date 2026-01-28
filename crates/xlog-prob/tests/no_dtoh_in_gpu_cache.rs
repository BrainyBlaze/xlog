use std::path::PathBuf;

#[path = "utils/dtoh.rs"]
mod dtoh_utils;

// Guardrail: GPU cache modules must not perform device->host reads.
#[test]
fn no_device_to_host_reads_in_gpu_cache() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("compilation");
    path.push("gpu_cache.rs");

    dtoh_utils::assert_no_dtoh_in_file(&path);
}
