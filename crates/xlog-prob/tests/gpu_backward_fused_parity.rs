//! Parity test: verify that eval_grads_inplace_fused produces identical gradients
//! to the original per-level eval_grads_inplace.
//!
//! Uses a circuit with all node types (Const0, Const1, Lit, And, Or, Decision) to
//! exercise every backward code path. Both methods run on the same cached circuit
//! with the same weights, then grad_true and grad_false are downloaded and compared.
//!
//! This is the primary correctness safety net for the fused backward kernel
//! (xgcf_backward_all_levels_cached).

#![allow(clippy::arc_with_non_send_sync)]

use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheConfig};
use xlog_prob::gpu::GpuXgcf;
use xlog_prob::xgcf::{Xgcf, XgcfNodeType};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA device unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1 << 30),
    ));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: kernel provider failed: {}", e);
            None
        }
    }
}

/// Build a circuit that exercises all backward code paths:
///
/// ```text
///   node 0: Const0            (level 0)
///   node 1: Const1            (level 0)
///   node 2: Lit(+1)  (var 1)  (level 0)
///   node 3: Lit(-2)  (var 2)  (level 0)
///   node 4: And(2, 3)         (level 1)  — tests AND backward
///   node 5: Or(1, 4)          (level 2)  — tests OR backward
///   node 6: Decision(var=3, false→0, true→5)  (level 3) — tests DECISION backward
/// ```
///
/// Variables: var 1 (p=0.3), var 2 (p=0.7), var 3 (p=0.5, decision)
/// Root: node 6
fn build_test_circuit() -> Xgcf {
    use XgcfNodeType::*;
    Xgcf {
        node_type: vec![Const0, Const1, Lit, Lit, And, Or, Decision],
        // CSR: child_offsets[node] .. child_offsets[node+1] = range in child_indices
        child_offsets: vec![
            0, // node 0 (Const0): no children
            0, // node 1 (Const1): no children
            0, // node 2 (Lit): no children
            0, // node 3 (Lit): no children
            0, // node 4 (And): children at indices 0..2
            2, // node 5 (Or): children at indices 2..4
            4, // node 6 (Decision): no CSR children (uses decision_child_*)
            4, // sentinel
        ],
        child_indices: vec![
            2, 3, // And(2,3) children
            1, 4, // Or(1,4) children
        ],
        lit: vec![
            0,  // node 0: not a Lit
            0,  // node 1: not a Lit
            1,  // node 2: positive literal for var 1
            -2, // node 3: negative literal for var 2
            0,  // node 4: not a Lit
            0,  // node 5: not a Lit
            0,  // node 6: not a Lit
        ],
        decision_var: vec![0, 0, 0, 0, 0, 0, 3],
        decision_child_false: vec![0, 0, 0, 0, 0, 0, 0], // false→node 0 (Const0)
        decision_child_true: vec![0, 0, 0, 0, 0, 0, 5],  // true→node 5 (Or)
        roots: vec![6],
        // 4 levels: [0,1,2,3 at level 0], [4 at level 1], [5 at level 2], [6 at level 3]
        level_offsets: vec![0, 4, 5, 6, 7],
        level_nodes: vec![0, 1, 2, 3, 4, 5, 6],
    }
}

/// Download the first `n` elements from a device slice.
/// The device slice may be larger (e.g., multi-slot cache layout), so we download
/// the full buffer and truncate to the requested length.
fn download_f64(
    provider: &CudaKernelProvider,
    src: &xlog_cuda::memory::TrackedCudaSlice<f64>,
    n: usize,
) -> Vec<f64> {
    let device_len = src.len();
    let mut host = vec![0.0f64; device_len];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(src, &mut host)
        .expect("dtoh copy");
    host.truncate(n);
    host
}

#[test]
fn fused_backward_matches_per_level() {
    let provider = match try_provider() {
        Some(p) => p,
        None => return,
    };

    let circuit = build_test_circuit();
    let mut direct = GpuXgcf::upload(&provider, &circuit).expect("upload");

    // Set non-trivial weights: var 1 = 0.3, var 2 = 0.7, var 3 = 0.5
    let max_var = direct.max_var() as usize;
    let mut weights = vec![(0.0f64, 0.0f64); max_var + 1];
    weights[1] = (0.3f64.ln(), (1.0 - 0.3f64).ln()); // var 1
    weights[2] = (0.7f64.ln(), (1.0 - 0.7f64).ln()); // var 2
    weights[3] = (0.5f64.ln(), (1.0 - 0.5f64).ln()); // var 3
    direct
        .set_base_weights(&provider, &weights)
        .expect("set weights");

    let var_cap = (max_var + 1) as u32;
    let config = {
        let mut config = GpuCircuitCacheConfig::default();
        config.num_slots = 2;
        config.table_size = 4;
        config.node_cap = 16;
        config.edge_cap = 16;
        config.level_cap = 16;
        config.var_cap = var_cap;
        config
    };

    let mut cache = GpuCircuitCache::new(&provider, config).expect("cache");

    // Store same circuit into slot 0 for per-level backward
    let mut handle0 = cache.claim_slot(0x1111u64).expect("claim slot 0");
    cache
        .store_from_xgcf(&mut handle0, &direct)
        .expect("store slot 0");

    // Run per-level backward
    cache
        .eval_grads_inplace(&handle0)
        .expect("per-level backward");
    provider.device().inner().synchronize().unwrap();

    // Download per-level results
    let gt_per_level = download_f64(&provider, cache.grad_true(), var_cap as usize + 1);
    let gf_per_level = download_f64(&provider, cache.grad_false(), var_cap as usize + 1);

    // Reset: re-store into same slot to clear gradients
    cache
        .store_from_xgcf(&mut handle0, &direct)
        .expect("re-store slot 0");

    // Run fused backward
    cache
        .eval_grads_inplace_fused(&handle0)
        .expect("fused backward");
    provider.device().inner().synchronize().unwrap();

    // Download fused results
    let gt_fused = download_f64(&provider, cache.grad_true(), var_cap as usize + 1);
    let gf_fused = download_f64(&provider, cache.grad_false(), var_cap as usize + 1);

    // Compare bit-for-bit (both methods use the same atomicAdd ordering within a block,
    // and the circuit is small enough that thread scheduling is deterministic).
    for i in 0..=max_var {
        let diff_true = (gt_per_level[i] - gt_fused[i]).abs();
        let diff_false = (gf_per_level[i] - gf_fused[i]).abs();
        // Allow small epsilon for atomicAdd ordering differences
        let eps = 1e-12;
        assert!(
            diff_true < eps,
            "grad_true[{}] mismatch: per_level={:.15e}, fused={:.15e}, diff={:.15e}",
            i,
            gt_per_level[i],
            gt_fused[i],
            diff_true
        );
        assert!(
            diff_false < eps,
            "grad_false[{}] mismatch: per_level={:.15e}, fused={:.15e}, diff={:.15e}",
            i,
            gf_per_level[i],
            gf_fused[i],
            diff_false
        );
    }

    // Sanity: at least some gradients should be non-zero
    let any_nonzero = gt_fused.iter().chain(gf_fused.iter()).any(|&v| v != 0.0);
    assert!(
        any_nonzero,
        "All gradients are zero — circuit evaluation likely failed"
    );
}
