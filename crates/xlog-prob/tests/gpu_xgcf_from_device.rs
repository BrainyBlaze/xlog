use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

use xlog_prob::compilation::{validate_equivalence_gpu, GpuEquivalenceConfig};
use xlog_prob::gpu::{GpuCircuitBuilder, GpuCircuitLayout, GpuXgcf};
use xlog_prob::xgcf::XgcfNodeType;

use xlog_solve::{GpuCnf, SolveInstance};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GiB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
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
fn gpu_xgcf_from_device_const1_matches_true_cnf() {
    let Some(provider) = try_provider() else {
        return;
    };
    let device = provider.device().inner();
    let memory = provider.memory();

    // φ = TRUE (CNF with 1 var and 0 clauses).
    let instance = SolveInstance::new(1, vec![]);
    let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    // Circuit C = Const1 (root=0).
    let mut d_node_type = memory.alloc::<u8>(1).unwrap();
    device
        .htod_sync_copy_into(&[XgcfNodeType::Const1 as u8], &mut d_node_type)
        .unwrap();

    let mut d_child_offsets = memory.alloc::<u32>(2).unwrap();
    device.htod_sync_copy_into(&[0u32, 0u32], &mut d_child_offsets).unwrap();

    let d_child_indices = memory.alloc::<u32>(0).unwrap();

    let mut d_lit = memory.alloc::<i32>(1).unwrap();
    device.htod_sync_copy_into(&[0i32], &mut d_lit).unwrap();

    let mut d_decision_var = memory.alloc::<u32>(1).unwrap();
    device.htod_sync_copy_into(&[0u32], &mut d_decision_var).unwrap();

    let mut d_decision_child_false = memory.alloc::<u32>(1).unwrap();
    device
        .htod_sync_copy_into(&[0u32], &mut d_decision_child_false)
        .unwrap();

    let mut d_decision_child_true = memory.alloc::<u32>(1).unwrap();
    device
        .htod_sync_copy_into(&[0u32], &mut d_decision_child_true)
        .unwrap();

    let mut d_level_nodes = memory.alloc::<u32>(1).unwrap();
    device.htod_sync_copy_into(&[0u32], &mut d_level_nodes).unwrap();

    let mut d_level_offsets = memory.alloc::<u32>(2).unwrap();
    device
        .htod_sync_copy_into(&[0u32, 1u32], &mut d_level_offsets)
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
        num_nodes: 1,
        num_levels: 1,
        level_offsets: d_level_offsets,
        level_nodes: d_level_nodes,
        root: 0,
        max_var: 0,
    };

    let circuit = GpuXgcf::from_device(builder, layout, &provider).expect("GpuXgcf from_device");

    validate_equivalence_gpu(&phi, &circuit, &provider, GpuEquivalenceConfig::default())
        .expect("equivalence should hold");

    // Device-only layout should still be evaluatable (no host level_offsets required).
    let weights: Vec<(f64, f64)> = vec![(0.0, 0.0)]; // max_var=0 => len=1
    let mut circuit_eval = circuit;
    let log_z = circuit_eval
        .eval_log_wmc(&provider, &weights)
        .expect("eval_log_wmc should succeed for from_device circuits");
    assert!((log_z - 0.0).abs() < 1e-12, "expected logZ=0 for Const1");
}
