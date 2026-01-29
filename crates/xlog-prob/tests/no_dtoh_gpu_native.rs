use std::path::PathBuf;
#[test]
fn gpu_d4_compile_path_has_no_dtoh_calls() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("compilation");
    path.push("gpu_d4.rs");

    let text = std::fs::read_to_string(&path).expect("read gpu_d4.rs");
    assert!(
        !text.contains("dtoh_sync_copy_into"),
        "gpu_d4.rs must not perform device->host reads"
    );
}

#[test]
fn mc_gpu_device_eval_avoids_host_query_truth() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("mc.rs");

    let text = std::fs::read_to_string(&path).expect("read mc.rs");
    let body = text
        .split("fn evaluate_gpu_device")
        .nth(1)
        .expect("evaluate_gpu_device not found")
        .to_string();

    assert!(
        !body.contains("query_truth"),
        "evaluate_gpu_device must not build host query_truth arrays"
    );
    assert!(
        !body.contains("htod_sync_copy_into"),
        "evaluate_gpu_device must not upload host query_truth arrays"
    );
}
