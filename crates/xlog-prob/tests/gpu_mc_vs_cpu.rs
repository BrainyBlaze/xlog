#![cfg(feature = "host-io")]
//! CLASSIFICATION: CPU **oracle-only** MC tests — NOT GPU-native acceptance.
//!
//! These tests call `McProgram::evaluate_cpu`, which downloads the sampled-bit
//! matrix to the host and evaluates worlds on the CPU. They validate that the
//! GPU device counts agree with a deterministic, seed-matched CPU oracle. They
//! are excluded from the zero-host / GPU-native acceptance matrix; the
//! authoritative GPU-native gates live in `tests/mc_gpu_native.rs` and
//! `tests/gpu_mc_device_counts.rs`.

use xlog_cuda::CudaDevice;
use xlog_prob::mc::{McEvalConfig, McProgram};

fn has_cuda_device() -> bool {
    CudaDevice::new(0).is_ok()
}

fn mc_config(samples: usize, seed: u64, max_nonmonotone_iterations: usize) -> McEvalConfig {
    let mut config = McEvalConfig::default();
    config.samples = samples;
    config.seed = seed;
    config.confidence = 0.95;
    config.max_nonmonotone_iterations = max_nonmonotone_iterations;
    config
}

#[test]
fn gpu_mc_matches_cpu_on_small_program() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    let prog = McProgram::compile_source("0.3::coin(1). query(coin(1)).").unwrap();
    let cfg = mc_config(20_000, 42, 128);

    let cpu = prog.evaluate_cpu(cfg.clone()).unwrap();
    let gpu = prog.evaluate_gpu(cfg).unwrap();

    let cpu_p = cpu.query_estimates[0].prob;
    let gpu_p = gpu.query_estimates[0].prob;
    assert!(
        (cpu_p - gpu_p).abs() < 0.02,
        "cpu_p={} gpu_p={}",
        cpu_p,
        gpu_p
    );
}
