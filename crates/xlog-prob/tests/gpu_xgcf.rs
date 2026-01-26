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
            eprintln!(
                "Skipping test: failed to create CUDA kernel provider: {}",
                e
            );
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
    let (gpu_log_z, gpu_grad_true, gpu_grad_false) = gpu_xgcf
        .eval_log_wmc_and_grads(&provider, &weights)
        .unwrap();

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

#[test]
fn test_gpu_xgcf_eval_grads_inplace_matches_cpu() {
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
    gpu_xgcf.set_base_weights(&provider, &weights).unwrap();
    gpu_xgcf.eval_grads_inplace(&provider).unwrap();

    // Download root value + gradients for verification (tests may read back).
    let device = provider.device().inner();

    let root_idx = gpu_xgcf.root() as usize;
    let root_view = gpu_xgcf.values().slice(root_idx..(root_idx + 1));
    let mut root_host = [0.0_f64];
    device
        .dtoh_sync_copy_into(&root_view, &mut root_host)
        .unwrap();
    let gpu_log_z = root_host[0];

    let mut gpu_grad_true = vec![0.0_f64; cpu_grad_true.len()];
    let mut gpu_grad_false = vec![0.0_f64; cpu_grad_false.len()];
    device
        .dtoh_sync_copy_into(gpu_xgcf.grad_true(), &mut gpu_grad_true)
        .unwrap();
    device
        .dtoh_sync_copy_into(gpu_xgcf.grad_false(), &mut gpu_grad_false)
        .unwrap();

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

#[test]
fn test_gpu_xgcf_smoothing_matches_cpu_gradients() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };

    // Unsmooth circuit: OR(x1, x2) where each branch mentions a different random var.
    let xgcf = Xgcf {
        node_type: vec![
            xlog_prob::xgcf::XgcfNodeType::Const0,
            xlog_prob::xgcf::XgcfNodeType::Const1,
            xlog_prob::xgcf::XgcfNodeType::Lit,
            xlog_prob::xgcf::XgcfNodeType::Lit,
            xlog_prob::xgcf::XgcfNodeType::Or,
        ],
        child_offsets: vec![0, 0, 0, 0, 0, 2],
        child_indices: vec![2, 3],
        lit: vec![0, 0, 1, 2, 0],
        decision_var: vec![0, 0, 0, 0, 0],
        decision_child_false: vec![0, 0, 0, 0, 0],
        decision_child_true: vec![0, 0, 0, 0, 0],
        roots: vec![4],
        level_offsets: vec![0, 4, 5],
        level_nodes: vec![0, 1, 2, 3, 4],
    };

    let is_random_var = vec![false, true, true];
    let smoothed = xgcf.smooth_random_vars(&is_random_var).unwrap();

    let p1 = 0.7_f64;
    let p2 = 0.2_f64;
    let weights: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (p1.ln(), (1.0 - p1).ln()),
        (p2.ln(), (1.0 - p2).ln()),
    ];
    let (cpu_log_z, cpu_grad_true, cpu_grad_false) =
        smoothed.eval_log_wmc_and_grads(&weights).unwrap();

    let gpu_xgcf = GpuXgcf::upload(&provider, &xgcf).unwrap();
    let random_vars = vec![1u32, 2u32];
    let mut gpu_smoothed = gpu_xgcf
        .smooth_random_vars_device(&provider, &random_vars, 64, 256)
        .unwrap();
    let (gpu_log_z, gpu_grad_true, gpu_grad_false) = gpu_smoothed
        .eval_log_wmc_and_grads(&provider, &weights)
        .unwrap();

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
            "var={} cpu_t={} gpu_t={} cpu_f={} gpu_f={} ",
            i,
            cpu_grad_true[i],
            gpu_grad_true[i],
            cpu_grad_false[i],
            gpu_grad_false[i]
        );
    }
}
