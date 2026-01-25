#![allow(clippy::arc_with_non_send_sync)]
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::gpu::GpuXgcf;
use xlog_prob::kc::ddnnf::DecisionDnnf;
use xlog_prob::xgcf::Xgcf;

fn try_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("Skipping test: failed to create CUDA kernel provider: {}", e);
            None
        }
    }
}

#[test]
fn test_gpu_xgcf_forward_matches_cpu() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };
    // Formula: x1 OR x2, represented as a decision on x1, then x2.
    let nnf = r#"
o 1 0
o 2 0
t 3 0
f 4 0
1 3 1 0
1 2 -1 0
2 3 2 0
2 4 -2 0
"#;
    let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
    let xgcf = Xgcf::from_ddnnf(&ddnnf).unwrap();

    let p1 = 0.7_f64;
    let p2 = 0.2_f64;
    let weights: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (p1.ln(), (1.0 - p1).ln()),
        (p2.ln(), (1.0 - p2).ln()),
    ];

    let cpu = xgcf.eval_log_wmc(|var| weights[var as usize]).unwrap();

    let mut gpu_xgcf = GpuXgcf::upload(&provider, &xgcf).unwrap();
    let gpu = gpu_xgcf.eval_log_wmc(&provider, &weights).unwrap();

    assert!((cpu - gpu).abs() < 1e-9, "cpu={} gpu={}", cpu, gpu);
}

#[test]
fn test_gpu_xgcf_backward_gradients_match_cpu() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };
    // Formula: x1 OR x2, represented as a decision on x1, then x2.
    let nnf = r#"
o 1 0
o 2 0
t 3 0
f 4 0
1 3 1 0
1 2 -1 0
2 3 2 0
2 4 -2 0
"#;
    let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
    let xgcf = Xgcf::from_ddnnf(&ddnnf).unwrap();

    let p1 = 0.7_f64;
    let p2 = 0.2_f64;
    let weights: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (p1.ln(), (1.0 - p1).ln()),
        (p2.ln(), (1.0 - p2).ln()),
    ];

    let (cpu_log_z, cpu_grad_true, cpu_grad_false) = xgcf.eval_log_wmc_and_grads(&weights).unwrap();

    let mut gpu_xgcf = GpuXgcf::upload(&provider, &xgcf).unwrap();
    let (gpu_log_z, gpu_grad_true, gpu_grad_false) =
        gpu_xgcf.eval_log_wmc_and_grads(&provider, &weights).unwrap();

    assert!(
        (cpu_log_z - gpu_log_z).abs() < 1e-9,
        "cpu_log_z={} gpu_log_z={}",
        cpu_log_z,
        gpu_log_z
    );

    assert_eq!(cpu_grad_true.len(), gpu_grad_true.len());
    assert_eq!(cpu_grad_false.len(), gpu_grad_false.len());
    for i in 0..cpu_grad_true.len() {
        let dt = (cpu_grad_true[i] - gpu_grad_true[i]).abs();
        let df = (cpu_grad_false[i] - gpu_grad_false[i]).abs();
        assert!(
            dt < 1e-9 && df < 1e-9,
            "var={} cpu_t={} gpu_t={} cpu_f={} gpu_f={}",
            i,
            cpu_grad_true[i],
            gpu_grad_true[i],
            cpu_grad_false[i],
            gpu_grad_false[i]
        );
    }
}
