#![cfg(feature = "host-io")]

use xlog_cuda::CudaDevice;
use xlog_prob::mc::{McEvalConfig, McProgram};

fn has_cuda_device() -> bool {
    CudaDevice::new(0).is_ok()
}

#[test]
fn gpu_mc_matches_cpu_on_small_program() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    let prog = McProgram::compile_source("0.3::coin(1). query(coin(1)).").unwrap();
    let cfg = McEvalConfig {
        samples: 20_000,
        seed: 42,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };

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
