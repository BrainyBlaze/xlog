//! Tests for error conversion From impls.
//!
//! These live in an integration test file (not lib tests) because
//! xlog-neural's Cargo.toml has `test = false` for PyO3 compatibility.
//! This file has NO required-features, so it runs under the default
//! `cargo test --workspace` invocation.

use xlog_core::XlogError;
use xlog_neural::NeuralError;
use xlog_neural::TensorSourceError;

#[test]
fn test_neural_error_into_xlog() {
    let err = NeuralError::NetworkNotFound("mnist".to_string());
    let xlog_err: XlogError = err.into();
    let msg = xlog_err.to_string();
    assert!(msg.contains("mnist"), "Expected 'mnist' in: {msg}");
    assert!(
        msg.contains("Network not found"),
        "Expected 'Network not found' in: {msg}"
    );
}

#[test]
fn test_neural_error_pytorch_into_xlog() {
    let err = NeuralError::PyTorchError("CUDA OOM".to_string());
    let xlog_err: XlogError = err.into();
    let msg = xlog_err.to_string();
    assert!(msg.contains("CUDA OOM"), "Expected 'CUDA OOM' in: {msg}");
}

#[test]
fn test_tensor_source_error_into_xlog() {
    let err = TensorSourceError::NotFound("train".to_string());
    let xlog_err: XlogError = err.into();
    let msg = xlog_err.to_string();
    assert!(msg.contains("train"), "Expected 'train' in: {msg}");
}

#[test]
fn test_tensor_source_no_active_into_xlog() {
    let err = TensorSourceError::NoActive;
    let xlog_err: XlogError = err.into();
    let msg = xlog_err.to_string();
    assert!(msg.contains("No active"), "Expected 'No active' in: {msg}");
}
