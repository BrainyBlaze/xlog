use std::path::PathBuf;

#[test]
fn no_device_synchronize_in_neural_backward_inner() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("exact.rs");

    let text = std::fs::read_to_string(&path).expect("read exact.rs");

    // Find the function
    let fn_start = text
        .find("fn neural_backward_nll_buffers_inner")
        .expect("function must exist");
    let fn_body = &text[fn_start..];
    // Find the next top-level fn boundary
    let fn_end = fn_body[1..]
        .find("\n    pub fn ")
        .or_else(|| fn_body[1..].find("\n    #[cfg"))
        .unwrap_or(fn_body.len() - 1);
    let fn_text = &fn_body[..fn_end];

    assert!(
        !fn_text.contains(".synchronize()"),
        "neural_backward_nll_buffers_inner must not call device().synchronize(). \
         The fused backward kernel handles synchronization internally, and callers \
         handle sync at batch boundaries via .item()."
    );
}
