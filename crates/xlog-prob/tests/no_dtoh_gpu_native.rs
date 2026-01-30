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

#[test]
fn mc_gpu_path_avoids_host_sampling() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("mc.rs");

    let text = std::fs::read_to_string(&path).expect("read mc.rs");
    assert!(
        !text.contains("sample_bernoulli_matrix("),
        "mc.rs still calls host sample_bernoulli_matrix (GPU path must avoid host sampling)"
    );
}

#[test]
fn smoothing_no_dtoh_calls_in_source() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("gpu.rs");

    let text = std::fs::read_to_string(&path).expect("read gpu.rs");
    let body = text
        .split("fn smooth_random_vars_device")
        .nth(1)
        .expect("smooth_random_vars_device not found");
    let body = body
        .split("pub fn upload")
        .next()
        .unwrap_or(body)
        .to_string();
    assert!(
        !body.contains("wrap_counts_host"),
        "gpu.rs still uses host wrap_counts for smoothing"
    );
    assert!(
        !body.contains("Failed to read smoothed edges"),
        "gpu.rs still reads smoothed edges on host"
    );
    assert!(
        !body.contains("dtoh_sync_copy_into"),
        "gpu.rs still performs device->host reads during smoothing"
    );
}

#[test]
fn random_var_collection_no_dtoh_in_source() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("exact.rs");

    let text = std::fs::read_to_string(&path).expect("read exact.rs");
    assert!(
        !text.contains("collect_random_vars_host"),
        "host random-var collection still present"
    );
}

#[test]
fn host_io_feature_required_for_dtoh_apis() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("gpu.rs");

    let text = std::fs::read_to_string(&path).expect("read gpu.rs");
    assert!(
        text.contains("cfg(feature = \"host-io\")"),
        "host-io feature gates missing for host DTOH APIs"
    );
}
