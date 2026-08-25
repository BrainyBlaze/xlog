#![allow(clippy::arc_with_non_send_sync)]
#![cfg(feature = "host-io")]

use std::sync::Arc;

use xlog_core::{MemoryBudget, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_prob::compilation::{gpu_d4::compute_free_var_mask_gpu, DeviceRandomVarList};
use xlog_prob::gpu::{GpuCircuitBuilder, GpuCircuitLayout, GpuXgcf};
use xlog_prob::kc::ddnnf::DecisionDnnf;
use xlog_prob::xgcf::{Xgcf, XgcfNodeType};
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

fn try_provider() -> Option<CudaKernelProvider> {
    match xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(1024 * 1024 * 1024))
        .build()
    {
        Ok(provider) => Some(provider),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            None
        }
    }
}

fn build_device_lit_circuit(
    provider: &Arc<CudaKernelProvider>,
    lit_var: u32,
    max_var: u32,
) -> GpuXgcf {
    let device = provider.device().inner();
    let memory = provider.memory();

    let node_type = [
        XgcfNodeType::Const0 as u8,
        XgcfNodeType::Const1 as u8,
        XgcfNodeType::Lit as u8,
    ];
    let child_offsets = [0u32, 0u32, 0u32, 0u32];
    let lit = [0i32, 0i32, lit_var as i32];
    let decision_var = [0u32, 0u32, 0u32];
    let decision_child_false = [0u32, 0u32, 0u32];
    let decision_child_true = [0u32, 0u32, 0u32];
    let level_nodes = [0u32, 1u32, 2u32];
    let level_offsets = [0u32, 2u32, 3u32];

    let mut d_node_type = memory.alloc::<u8>(node_type.len()).unwrap();
    device
        .htod_sync_copy_into(&node_type, &mut d_node_type)
        .unwrap();

    let mut d_child_offsets = memory.alloc::<u32>(child_offsets.len()).unwrap();
    device
        .htod_sync_copy_into(&child_offsets, &mut d_child_offsets)
        .unwrap();

    let d_child_indices = memory.alloc::<u32>(0).unwrap();

    let mut d_lit = memory.alloc::<i32>(lit.len()).unwrap();
    device.htod_sync_copy_into(&lit, &mut d_lit).unwrap();

    let mut d_decision_var = memory.alloc::<u32>(decision_var.len()).unwrap();
    device
        .htod_sync_copy_into(&decision_var, &mut d_decision_var)
        .unwrap();

    let mut d_decision_child_false = memory.alloc::<u32>(decision_child_false.len()).unwrap();
    device
        .htod_sync_copy_into(&decision_child_false, &mut d_decision_child_false)
        .unwrap();

    let mut d_decision_child_true = memory.alloc::<u32>(decision_child_true.len()).unwrap();
    device
        .htod_sync_copy_into(&decision_child_true, &mut d_decision_child_true)
        .unwrap();

    let mut d_level_nodes = memory.alloc::<u32>(level_nodes.len()).unwrap();
    device
        .htod_sync_copy_into(&level_nodes, &mut d_level_nodes)
        .unwrap();

    let mut d_level_offsets = memory.alloc::<u32>(level_offsets.len()).unwrap();
    device
        .htod_sync_copy_into(&level_offsets, &mut d_level_offsets)
        .unwrap();

    let builder = GpuCircuitBuilder {
        node_type: d_node_type,
        child_offsets: d_child_offsets,
        child_indices: d_child_indices,
        lit: d_lit,
        decision_var: d_decision_var,
        decision_child_false: d_decision_child_false,
        decision_child_true: d_decision_child_true,
    };

    let layout = GpuCircuitLayout {
        num_nodes: node_type.len() as u32,
        num_edges: 0,
        num_levels: 2,
        level_offsets: d_level_offsets,
        level_nodes: d_level_nodes,
        root: 2,
        max_var,
        num_nodes_device: None,
        num_edges_device: None,
    };

    GpuXgcf::from_device(builder, layout, provider).expect("GpuXgcf from_device")
}

fn binary_circuit(node_type: XgcfNodeType) -> Xgcf {
    assert!(matches!(node_type, XgcfNodeType::And | XgcfNodeType::Or));
    Xgcf {
        node_type: vec![XgcfNodeType::Lit, XgcfNodeType::Lit, node_type],
        child_offsets: vec![0, 0, 0, 2],
        child_indices: vec![0, 1],
        lit: vec![1, 2, 0],
        decision_var: vec![0, 0, 0],
        decision_child_false: vec![0, 0, 0],
        decision_child_true: vec![0, 0, 0],
        roots: vec![2],
        level_offsets: vec![0, 2, 3],
        level_nodes: vec![0, 1, 2],
    }
}

fn edge_policy_provider() -> Option<Arc<CudaKernelProvider>> {
    match try_provider() {
        Some(provider) => Some(Arc::new(provider)),
        None if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but the XGCF edge-policy provider was unavailable")
        }
        None => None,
    }
}

fn assert_nan_compilation_error(error: &XlogError) {
    assert!(
        matches!(error, XlogError::Compilation(message) if message.contains("NaN")),
        "unexpected error: {error}"
    );
}

#[test]
fn test_gpu_xgcf_positive_infinity_value_parity() {
    let Some(provider) = edge_policy_provider() else {
        return;
    };

    let or_circuit = binary_circuit(XgcfNodeType::Or);
    let or_weights = vec![(0.0, 0.0), (f64::INFINITY, 0.0), (-2.0, 0.0)];
    let cpu_or = or_circuit
        .eval_log_wmc(|var| or_weights[var as usize])
        .unwrap();
    let gpu_or = GpuXgcf::upload(&provider, &or_circuit)
        .unwrap()
        .eval_log_wmc(&provider, &or_weights)
        .unwrap();

    let nnf = r#"
o 1 0
t 2 0
f 3 0
1 2 1 0
1 3 -1 0
"#;
    let decision_circuit = Xgcf::from_ddnnf(&DecisionDnnf::parse_str(nnf).unwrap()).unwrap();
    let decision_weights = vec![(0.0, 0.0), (f64::INFINITY, -1.0)];
    let cpu_decision = decision_circuit
        .eval_log_wmc(|var| decision_weights[var as usize])
        .unwrap();
    let gpu_decision = GpuXgcf::upload(&provider, &decision_circuit)
        .unwrap()
        .eval_log_wmc(&provider, &decision_weights)
        .unwrap();

    for (name, cpu, gpu) in [
        ("OR", cpu_or, gpu_or),
        ("DECISION", cpu_decision, gpu_decision),
    ] {
        assert!(
            cpu.is_infinite() && cpu.is_sign_positive(),
            "{name}: CPU expected +inf, got {cpu}"
        );
        assert_eq!(gpu, cpu, "{name}: CPU/GPU +inf policy diverged");
    }
}

#[test]
fn test_gpu_xgcf_nan_weight_error_parity() {
    let Some(provider) = edge_policy_provider() else {
        return;
    };
    let circuit = binary_circuit(XgcfNodeType::Or);
    let weights = vec![(0.0, 0.0), (f64::NAN, 0.0), (-2.0, 0.0)];

    let cpu_error = circuit
        .eval_log_wmc(|var| weights[var as usize])
        .expect_err("CPU must reject NaN log weights");
    let gpu_error = GpuXgcf::upload(&provider, &circuit)
        .unwrap()
        .eval_log_wmc(&provider, &weights)
        .expect_err("host GPU API must reject NaN log weights");

    assert_nan_compilation_error(&cpu_error);
    assert_nan_compilation_error(&gpu_error);
}

#[test]
fn test_gpu_xgcf_computed_nan_error_parity() {
    let Some(provider) = edge_policy_provider() else {
        return;
    };
    let circuit = binary_circuit(XgcfNodeType::And);
    let weights = vec![(0.0, 0.0), (f64::INFINITY, 0.0), (f64::NEG_INFINITY, 0.0)];

    let cpu_error = circuit
        .eval_log_wmc(|var| weights[var as usize])
        .expect_err("CPU must reject computed +inf + -inf NaN");
    let gpu_error = GpuXgcf::upload(&provider, &circuit)
        .unwrap()
        .eval_log_wmc(&provider, &weights)
        .expect_err("host GPU API must reject computed +inf + -inf NaN");

    assert_nan_compilation_error(&cpu_error);
    assert_nan_compilation_error(&gpu_error);
}

#[test]
fn test_gpu_xgcf_normalized_positive_infinity_gradient_error_parity() {
    let Some(provider) = edge_policy_provider() else {
        return;
    };
    let decision_nnf = r#"
o 1 0
t 2 0
f 3 0
1 2 1 0
1 3 -1 0
"#;
    let cases = [
        (
            "OR",
            binary_circuit(XgcfNodeType::Or),
            vec![(0.0, 0.0), (f64::INFINITY, 0.0), (-2.0, 0.0)],
        ),
        (
            "DECISION",
            Xgcf::from_ddnnf(&DecisionDnnf::parse_str(decision_nnf).unwrap()).unwrap(),
            vec![(0.0, 0.0), (f64::INFINITY, -1.0)],
        ),
    ];

    for (shape, circuit, weights) in cases {
        let cpu_error = circuit
            .eval_log_wmc_and_grads(&weights)
            .expect_err("CPU normalization must reject non-finite gradients");
        let gpu_error = GpuXgcf::upload(&provider, &circuit)
            .unwrap()
            .eval_log_wmc_and_grads(&provider, &weights)
            .expect_err("GPU normalization must reject non-finite gradients");

        for (backend, error) in [("CPU", cpu_error), ("GPU", gpu_error)] {
            assert!(
                matches!(&error, XlogError::Compilation(message) if message.contains("gradient") && message.contains("non-finite")),
                "{shape} {backend}: unexpected error: {error}"
            );
        }
    }
}

#[test]
fn test_gpu_xgcf_deterministic_gradient_allows_positive_infinity() {
    let Some(provider) = edge_policy_provider() else {
        return;
    };
    let circuit = Xgcf {
        node_type: vec![XgcfNodeType::Lit],
        child_offsets: vec![0, 0],
        child_indices: vec![],
        lit: vec![3],
        decision_var: vec![0],
        decision_child_false: vec![0],
        decision_child_true: vec![0],
        roots: vec![0],
        level_offsets: vec![0, 1],
        level_nodes: vec![0],
    };
    let weights = vec![
        (0.0, 0.0),
        (f64::INFINITY, f64::INFINITY),
        (f64::INFINITY, f64::INFINITY),
        (f64::INFINITY, f64::INFINITY),
    ];

    let cpu = circuit
        .eval_log_wmc_and_grads(&weights)
        .expect("CPU deterministic gradients must allow finite-output +inf cases");
    let gpu = GpuXgcf::upload(&provider, &circuit)
        .unwrap()
        .eval_log_wmc_and_grads(&provider, &weights)
        .expect("GPU deterministic gradients must allow finite-output +inf cases");

    assert!(cpu.0.is_infinite() && cpu.0.is_sign_positive());
    assert_eq!(cpu.1, vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(cpu.2, vec![0.0; 4]);
    assert_eq!(gpu, cpu);
}

#[test]
fn test_gpu_xgcf_reserved_weight_slot_zero_is_ignored() {
    let Some(provider) = edge_policy_provider() else {
        return;
    };
    let circuit = binary_circuit(XgcfNodeType::Or);
    let baseline_weights = vec![(0.0, 0.0), (-0.25, -0.75), (-1.0, -0.5)];
    let reserved_non_finite = vec![
        (f64::NAN, f64::INFINITY),
        baseline_weights[1],
        baseline_weights[2],
    ];

    let cpu_value = circuit
        .eval_log_wmc(|var| reserved_non_finite[var as usize])
        .unwrap();
    let cpu_grads = circuit
        .eval_log_wmc_and_grads(&reserved_non_finite)
        .expect("CPU must ignore reserved slot 0");

    let mut gpu = GpuXgcf::upload(&provider, &circuit).unwrap();
    let baseline_gpu_value = gpu.eval_log_wmc(&provider, &baseline_weights).unwrap();
    let baseline_gpu_grads = gpu
        .eval_log_wmc_and_grads(&provider, &baseline_weights)
        .unwrap();
    let reserved_gpu_value = gpu
        .eval_log_wmc(&provider, &reserved_non_finite)
        .expect("GPU value path must ignore reserved slot 0");
    let reserved_gpu_grads = gpu
        .eval_log_wmc_and_grads(&provider, &reserved_non_finite)
        .expect("GPU gradient path must ignore reserved slot 0");

    assert_eq!(cpu_value, baseline_gpu_value);
    assert_eq!(reserved_gpu_value, baseline_gpu_value);
    assert_eq!(cpu_grads, baseline_gpu_grads);
    assert_eq!(reserved_gpu_grads, baseline_gpu_grads);
}

#[test]
fn test_gpu_xgcf_device_only_nan_sentinel() {
    let Some(provider) = edge_policy_provider() else {
        return;
    };
    let circuit = binary_circuit(XgcfNodeType::Or);
    let mut gpu = GpuXgcf::upload(&provider, &circuit).unwrap();
    let host_true = [0.0, f64::NAN, -2.0];
    let host_false = [0.0; 3];
    {
        let (device_true, device_false) = gpu.var_log_weights_mut();
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&host_true, device_true)
            .unwrap();
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&host_false, device_false)
            .unwrap();
    }

    let mut output = provider.memory().alloc::<f64>(1).unwrap();
    gpu.eval_log_wmc_device_inplace(&provider, &mut output)
        .unwrap();
    let mut host_output = [0.0];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&output, &mut host_output)
        .unwrap();
    assert!(
        host_output[0].is_nan(),
        "device-only API must emit a NaN sentinel"
    );
}

#[test]
fn test_gpu_xgcf_forward_matches_cpu() {
    let provider = match try_provider() {
        Some(p) => Arc::new(p),
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
        Some(p) => Arc::new(p),
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
    let random_vars_device = DeviceRandomVarList::from_host(&provider, &random_vars).unwrap();
    let mut gpu_smoothed = gpu_xgcf
        .smooth_random_vars_device(
            &provider,
            random_vars_device.list(),
            random_vars_device.count(),
            64,
            256,
        )
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

#[test]
fn test_gpu_free_var_mask_matches_cpu() {
    let provider = match try_provider() {
        Some(p) => Arc::new(p),
        None => return,
    };

    // CNF with 2 vars where only var1 appears in clauses (var2 is free).
    let instance = SolveInstance::new(2, vec![Clause::new(vec![Literal::positive(0)])]);
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    // Circuit uses only var1 (DIMACS 1).
    let xgcf = Xgcf {
        node_type: vec![
            XgcfNodeType::Const0,
            XgcfNodeType::Const1,
            XgcfNodeType::Lit,
        ],
        child_offsets: vec![0, 0, 0, 0],
        child_indices: vec![],
        lit: vec![0, 0, 1],
        decision_var: vec![0, 0, 0],
        decision_child_false: vec![0, 0, 0],
        decision_child_true: vec![0, 0, 0],
        roots: vec![2],
        level_offsets: vec![0, 2, 3],
        level_nodes: vec![0, 1, 2],
    };
    let gpu_xgcf = GpuXgcf::upload(&provider, &xgcf).expect("GpuXgcf upload");

    let free_mask =
        compute_free_var_mask_gpu(&cnf, &gpu_xgcf, &provider).expect("compute_free_var_mask_gpu");

    let mut host_mask = vec![0u8; (cnf.var_cap as usize) + 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&free_mask, &mut host_mask)
        .expect("free_var_mask dtoh");

    let mut vars_in_clauses = vec![false; host_mask.len()];
    for clause in &instance.clauses {
        for lit in &clause.literals {
            let var = lit.to_dimacs().unsigned_abs() as usize;
            vars_in_clauses[var] = true;
        }
    }

    let mut vars_in_circuit = vec![false; host_mask.len()];
    for (idx, ty) in xgcf.node_type.iter().enumerate() {
        match ty {
            XgcfNodeType::Lit => {
                let var = xgcf.lit[idx].unsigned_abs() as usize;
                vars_in_circuit[var] = true;
            }
            XgcfNodeType::Decision => {
                let var = xgcf.decision_var[idx] as usize;
                vars_in_circuit[var] = true;
            }
            _ => {}
        }
    }

    let mut expected = vec![0u8; host_mask.len()];
    for var in 1..=instance.num_vars as usize {
        if !vars_in_clauses[var] && !vars_in_circuit[var] {
            expected[var] = 1u8;
        }
    }

    assert_eq!(host_mask, expected);
}

#[test]
fn test_gpu_free_var_correction_matches_cpu() {
    let provider = match try_provider() {
        Some(p) => Arc::new(p),
        None => return,
    };

    // CNF with 2 vars where only var1 appears in clauses (var2 is free).
    let instance = SolveInstance::new(2, vec![Clause::new(vec![Literal::positive(0)])]);
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    // Host circuit (var1 only) for CPU baseline.
    let xgcf = Xgcf {
        node_type: vec![
            XgcfNodeType::Const0,
            XgcfNodeType::Const1,
            XgcfNodeType::Lit,
        ],
        child_offsets: vec![0, 0, 0, 0],
        child_indices: vec![],
        lit: vec![0, 0, 1],
        decision_var: vec![0, 0, 0],
        decision_child_false: vec![0, 0, 0],
        decision_child_true: vec![0, 0, 0],
        roots: vec![2],
        level_offsets: vec![0, 2, 3],
        level_nodes: vec![0, 1, 2],
    };

    let p1 = 0.7_f64;
    let p2 = 0.2_f64;
    let weights: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (p1.ln(), (1.0 - p1).ln()),
        (p2.ln(), (1.0 - p2).ln()),
    ];

    let (base_log_z, base_grad_true, base_grad_false) =
        xgcf.eval_log_wmc_and_grads(&weights).unwrap();

    let logsumexp2_with_grads = |t: f64, f: f64| -> (f64, f64, f64) {
        let m = if t > f { t } else { f };
        if m.is_infinite() && m.is_sign_negative() {
            return (m, 0.0, 0.0);
        }
        let et = (t - m).exp();
        let ef = (f - m).exp();
        let sum = et + ef;
        let log_z = m + sum.ln();
        let pt = et / sum;
        let pf = ef / sum;
        (log_z, pt, pf)
    };

    let (free_log_z, free_pt, free_pf) = logsumexp2_with_grads(weights[2].0, weights[2].1);

    let mut expected_grad_true = vec![0.0_f64; weights.len()];
    let mut expected_grad_false = vec![0.0_f64; weights.len()];
    expected_grad_true[..base_grad_true.len()].copy_from_slice(&base_grad_true);
    expected_grad_false[..base_grad_false.len()].copy_from_slice(&base_grad_false);
    expected_grad_true[2] += free_pt;
    expected_grad_false[2] += free_pf;
    let expected_log_z = base_log_z + free_log_z;

    let mut gpu_xgcf = build_device_lit_circuit(&provider, 1, cnf.var_cap);
    let free_mask =
        compute_free_var_mask_gpu(&cnf, &gpu_xgcf, &provider).expect("compute_free_var_mask_gpu");
    gpu_xgcf
        .set_free_var_mask_device(free_mask)
        .expect("set_free_var_mask_device");

    let (gpu_log_z, gpu_grad_true, gpu_grad_false) = gpu_xgcf
        .eval_log_wmc_and_grads(&provider, &weights)
        .unwrap();

    assert!(
        (gpu_log_z - expected_log_z).abs() < 1e-9,
        "expected_log_z={} gpu_log_z={}",
        expected_log_z,
        gpu_log_z
    );
    assert_eq!(gpu_grad_true.len(), expected_grad_true.len());
    assert_eq!(gpu_grad_false.len(), expected_grad_false.len());
    for i in 0..expected_grad_true.len() {
        let dt = (expected_grad_true[i] - gpu_grad_true[i]).abs();
        let df = (expected_grad_false[i] - gpu_grad_false[i]).abs();
        assert!(
            dt < 1e-9 && df < 1e-9,
            "var={} exp_t={} gpu_t={} exp_f={} gpu_f={}",
            i,
            expected_grad_true[i],
            gpu_grad_true[i],
            expected_grad_false[i],
            gpu_grad_false[i]
        );
    }
}

#[test]
fn test_gpu_xgcf_device_logz_into_matches_cpu() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };
    let memory = provider.memory();
    let device = provider.device().inner();

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
    let mut out_log_z = memory.alloc::<f64>(1).unwrap();
    gpu_xgcf
        .eval_log_wmc_device_into(&provider, &weights, &mut out_log_z)
        .unwrap();

    let mut host = [0.0_f64];
    device.dtoh_sync_copy_into(&out_log_z, &mut host).unwrap();

    assert!((cpu - host[0]).abs() < 1e-9, "cpu={} gpu={}", cpu, host[0]);
}

#[test]
fn test_gpu_xgcf_device_logz_alloc_matches_cpu() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };
    let device = provider.device().inner();

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
    let out_log_z = gpu_xgcf.eval_log_wmc_device(&provider, &weights).unwrap();

    let mut host = [0.0_f64];
    device.dtoh_sync_copy_into(&out_log_z, &mut host).unwrap();

    assert!((cpu - host[0]).abs() < 1e-9, "cpu={} gpu={}", cpu, host[0]);
}
