#![allow(clippy::arc_with_non_send_sync)]
#![cfg(feature = "host-io")]

use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::neural_fast_path::GpuWeightSlots;
use xlog_prob::neural_fast_path::NeuralFastPathConfig;

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
fn test_gpu_weight_slots_upload_roundtrips() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };

    let groups: Vec<Vec<u32>> = vec![vec![10, 11, 12], vec![20, 21, 22, 23]];
    let slots = GpuWeightSlots::upload(&provider, &groups).unwrap();

    assert_eq!(slots.num_groups(), 2);
    assert_eq!(slots.total_slots(), 7);

    let device = provider.device().inner();

    let mut offsets_host = vec![0u32; 3];
    device
        .dtoh_sync_copy_into(slots.group_offsets(), &mut offsets_host)
        .unwrap();
    assert_eq!(offsets_host, vec![0, 3, 7]);

    let mut vars_host = vec![0u32; 7];
    device
        .dtoh_sync_copy_into(slots.slot_cnf_var(), &mut vars_host)
        .unwrap();
    assert_eq!(vars_host, vec![10, 11, 12, 20, 21, 22, 23]);
}

#[test]
fn test_neural_backward_nll_matches_analytic_single_outcome() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };

    // Single annotated disjunction with 3 labels (plus implicit none).
    // Query a single outcome; analytic gradient in probability-space is simple and
    // exercises the AD conditional-chain Jacobian.
    let source = r#"
0.33::pred(0, 0); 0.33::pred(0, 1); 0.33::pred(0, 2).
query(pred(0, 0)).
query(pred(0, 1)).
query(pred(0, 2)).
"#;

    let cfg = GpuConfig {
        device_ordinal: 0,
        memory_bytes: 1024 * 1024 * 1024,
        ..Default::default()
    };
    let program = ExactDdnnfProgram::compile_source_with_gpu(source, cfg).unwrap();

    let vars = program.random_var_indices();
    assert_eq!(vars.len(), 3, "expected exactly 3 AD chain vars");

    let groups: Vec<Vec<u32>> = vec![vars];
    let slots = GpuWeightSlots::upload(&provider, &groups).unwrap();

    // Softmax probabilities (sum to 1).
    let p: [f32; 3] = [0.2, 0.3, 0.5];

    let schema = xlog_core::Schema::new(vec![("col0".to_string(), xlog_core::ScalarType::F32)]);
    let prob_buf = provider
        .create_buffer_from_slice::<f32>(&p, schema.clone())
        .unwrap();
    let grad_buf = provider
        .create_buffer_from_slice::<f32>(&[0.0f32; 3], schema)
        .unwrap();

    let cfg = NeuralFastPathConfig {
        eps: 1e-7,
        min_p: 1e-12,
        ..Default::default()
    };

    let probs = vec![prob_buf];
    let mut grads = vec![grad_buf];
    // Query index 1 corresponds to pred(0, 1) in the source above.
    program
        .neural_backward_nll_buffers(&slots, 1, &probs, &mut grads, cfg)
        .unwrap();

    // Download gradients (tests may read back).
    let mut out_bytes = vec![0u8; 3 * 4];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(grads[0].column(0).unwrap(), &mut out_bytes)
        .unwrap();
    let out: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Analytic: L = -log((1-eps)*p1) => dL/dp1 = -1/p1; others 0.
    let expected = vec![0.0f32, (-1.0f32 / p[1]), 0.0f32];
    for i in 0..3 {
        let err = (out[i] - expected[i]).abs();
        assert!(
            err < 1e-3,
            "grad[{}] expected {} got {} (err={})",
            i,
            expected[i],
            out[i],
            err
        );
    }
}

#[test]
fn test_neural_backward_nll_device_loss_matches_analytic_single_outcome() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };

    let source = r#"
0.33::pred(0, 0); 0.33::pred(0, 1); 0.33::pred(0, 2).
query(pred(0, 0)).
query(pred(0, 1)).
query(pred(0, 2)).
"#;

    let cfg = GpuConfig {
        device_ordinal: 0,
        memory_bytes: 1024 * 1024 * 1024,
        ..Default::default()
    };
    let program = ExactDdnnfProgram::compile_source_with_gpu(source, cfg).unwrap();

    let vars = program.random_var_indices();
    assert_eq!(vars.len(), 3, "expected exactly 3 AD chain vars");

    let groups: Vec<Vec<u32>> = vec![vars];
    let slots = GpuWeightSlots::upload(&provider, &groups).unwrap();

    let p: [f32; 3] = [0.2, 0.3, 0.5];

    let schema = xlog_core::Schema::new(vec![("col0".to_string(), xlog_core::ScalarType::F32)]);
    let prob_buf = provider
        .create_buffer_from_slice::<f32>(&p, schema.clone())
        .unwrap();
    let grad_buf = provider
        .create_buffer_from_slice::<f32>(&[0.0f32; 3], schema)
        .unwrap();

    let cfg = NeuralFastPathConfig {
        eps: 1e-7,
        min_p: 1e-12,
        ..Default::default()
    };

    let probs = vec![prob_buf];
    let mut grads = vec![grad_buf];

    // Query index 1 corresponds to pred(0, 1) in the source above.
    let loss_dev = program
        .neural_backward_nll_buffers_with_device_loss(&slots, 1, &probs, &mut grads, cfg, true)
        .unwrap();

    let mut host = [0.0_f64];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&loss_dev, &mut host)
        .unwrap();
    let loss = host[0];

    let expected = -((1.0 - cfg.eps) * (p[1] as f64)).ln();
    let err = (loss - expected).abs();
    assert!(
        err < 1e-6,
        "loss expected {} got {} (err={})",
        expected,
        loss,
        err
    );
}

#[test]
fn test_neural_backward_nll_device_loss_expected_false_matches_analytic_single_outcome() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };

    let source = r#"
0.33::pred(0, 0); 0.33::pred(0, 1); 0.33::pred(0, 2).
query(pred(0, 0)).
query(pred(0, 1)).
query(pred(0, 2)).
"#;

    let cfg = GpuConfig {
        device_ordinal: 0,
        memory_bytes: 1024 * 1024 * 1024,
        ..Default::default()
    };
    let program = ExactDdnnfProgram::compile_source_with_gpu(source, cfg).unwrap();

    let vars = program.random_var_indices();
    assert_eq!(vars.len(), 3, "expected exactly 3 AD chain vars");

    let groups: Vec<Vec<u32>> = vec![vars];
    let slots = GpuWeightSlots::upload(&provider, &groups).unwrap();

    let p: [f32; 3] = [0.2, 0.3, 0.5];

    let schema = xlog_core::Schema::new(vec![("col0".to_string(), xlog_core::ScalarType::F32)]);
    let prob_buf = provider
        .create_buffer_from_slice::<f32>(&p, schema.clone())
        .unwrap();
    let grad_buf = provider
        .create_buffer_from_slice::<f32>(&[0.0f32; 3], schema)
        .unwrap();

    let cfg = NeuralFastPathConfig {
        eps: 1e-7,
        min_p: 1e-12,
        ..Default::default()
    };

    let probs = vec![prob_buf];
    let mut grads = vec![grad_buf];

    // Query index 1 corresponds to pred(0, 1) in the source above.
    let loss_dev = program
        .neural_backward_nll_buffers_with_device_loss(&slots, 1, &probs, &mut grads, cfg, false)
        .unwrap();

    let mut host = [0.0_f64];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&loss_dev, &mut host)
        .unwrap();
    let loss = host[0];

    let expected = -(1.0 - (1.0 - cfg.eps) * (p[1] as f64)).ln();
    let err = (loss - expected).abs();
    assert!(
        err < 1e-6,
        "expected_false loss expected {} got {} (err={})",
        expected,
        loss,
        err
    );
}
