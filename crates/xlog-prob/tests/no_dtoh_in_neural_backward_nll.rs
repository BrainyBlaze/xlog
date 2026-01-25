use std::path::PathBuf;

// Guardrail: the GPU neural fast-path NLL backward must not rely on device->host reads.
// We enforce this via a focused source scan over the neural fast-path function body.
#[test]
fn no_device_to_host_reads_in_neural_backward_nll() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("exact.rs");

    let text = std::fs::read_to_string(&path).expect("read exact.rs");

    let start = text
        .find("fn neural_backward_nll_buffers_inner")
        .expect("find neural_backward_nll_buffers_inner");
    let end = text[start..]
        .find("pub fn evaluate_gpu_with_grads")
        .map(|off| start + off)
        .expect("find evaluate_gpu_with_grads after neural_backward_nll_buffers_inner");

    let body = &text[start..end];

    assert!(
        !body.contains("dtoh_sync_copy_into"),
        "neural_backward_nll_buffers_inner must not call dtoh_sync_copy_into (found in {})",
        path.display()
    );
    assert!(
        !body.contains("copy_to_host"),
        "neural_backward_nll_buffers_inner must not call copy_to_host (found in {})",
        path.display()
    );
    assert!(
        !body.contains("dtoh"),
        "neural_backward_nll_buffers_inner must not reference dtoh transfers (found in {})",
        path.display()
    );
}
