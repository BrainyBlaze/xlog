use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_pir::{GpuPirGraph, PIR_AND, PIR_LIT, PIR_NEG_LIT, PIR_OR};
use xlog_prob::pir::{LeafId, PirGraph};

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
fn gpu_pir_layout_matches_cpu_nodes() {
    let Some(provider) = try_provider() else {
        return;
    };

    let mut pir = PirGraph::new();
    let a = pir.lit(LeafId::new(0));
    let b = pir.neg_lit(LeafId::new(1));
    let root = pir.and(vec![a, b]);
    let _ = root;

    let gpu = GpuPirGraph::from_host(&pir, &provider).expect("from_host");

    let device = provider.device().inner();

    let mut node_type = vec![0u8; gpu.node_type.len()];
    device
        .dtoh_sync_copy_into(&gpu.node_type, &mut node_type)
        .unwrap();

    assert_eq!(node_type.len(), 3);
    assert_eq!(node_type[0], PIR_LIT);
    assert_eq!(node_type[1], PIR_NEG_LIT);
    assert_eq!(node_type[2], PIR_AND);

    let mut child_offsets = vec![0u32; gpu.child_offsets.len()];
    device
        .dtoh_sync_copy_into(&gpu.child_offsets, &mut child_offsets)
        .unwrap();
    assert_eq!(child_offsets.len(), 4);
    assert_eq!(child_offsets, vec![0, 0, 0, 2]);

    let mut children = vec![0u32; gpu.children.len()];
    device
        .dtoh_sync_copy_into(&gpu.children, &mut children)
        .unwrap();
    assert_eq!(children, vec![0, 1]);

    let mut leaf_id = vec![0u32; gpu.leaf_id.len()];
    device
        .dtoh_sync_copy_into(&gpu.leaf_id, &mut leaf_id)
        .unwrap();
    assert_eq!(leaf_id, vec![0, 1, 0]);

    let mut decision_var = vec![0u32; gpu.decision_var.len()];
    device
        .dtoh_sync_copy_into(&gpu.decision_var, &mut decision_var)
        .unwrap();
    assert!(decision_var.iter().all(|&v| v == 0));

    let mut decision_false = vec![0u32; gpu.decision_child_false.len()];
    device
        .dtoh_sync_copy_into(&gpu.decision_child_false, &mut decision_false)
        .unwrap();
    assert!(decision_false.iter().all(|&v| v == 0));

    let mut decision_true = vec![0u32; gpu.decision_child_true.len()];
    device
        .dtoh_sync_copy_into(&gpu.decision_child_true, &mut decision_true)
        .unwrap();
    assert!(decision_true.iter().all(|&v| v == 0));

    assert_ne!(PIR_OR, PIR_LIT);
}
