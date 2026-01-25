#![allow(clippy::arc_with_non_send_sync)]
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::neural_fast_path::GpuWeightSlots;

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

