use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheConfig};
use xlog_prob::gpu::GpuXgcf;
use xlog_prob::xgcf::{Xgcf, XgcfNodeType};

fn read_u32_at(
    provider: &Arc<CudaKernelProvider>,
    slice: &xlog_cuda::memory::TrackedCudaSlice<u32>,
    idx: usize,
) -> u32 {
    let view = slice.slice(idx..idx + 1);
    let mut host = [0u32; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&view, &mut host)
        .expect("dtoh u32");
    host[0]
}

fn read_slot(
    provider: &Arc<CudaKernelProvider>,
    handle: &xlog_prob::compilation::gpu_cache::GpuCircuitCacheHandle,
) -> usize {
    let mut host = [0u32; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(handle.slot_device(), &mut host)
        .expect("dtoh slot");
    host[0] as usize
}

#[test]
fn cache_store_writes_metadata() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1 << 30),
    ));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).expect("provider"));

    let circuit = Xgcf {
        node_type: vec![XgcfNodeType::Lit],
        child_offsets: vec![0, 0],
        child_indices: vec![],
        lit: vec![1],
        decision_var: vec![0],
        decision_child_false: vec![0],
        decision_child_true: vec![0],
        roots: vec![0],
        level_offsets: vec![0, 1],
        level_nodes: vec![0],
    };
    let mut direct = GpuXgcf::upload(&provider, &circuit).expect("upload");

    let weights_len = direct.max_var() as usize + 1;
    let weights = vec![(0.0f64, 0.0f64); weights_len];
    direct
        .set_base_weights(&provider, &weights)
        .expect("set weights");

    let config = GpuCircuitCacheConfig {
        num_slots: 1,
        table_size: 4,
        node_cap: 16,
        edge_cap: 32,
        level_cap: 16,
        var_cap: 16,
    };
    let mut cache = GpuCircuitCache::new(&provider, config).expect("cache");
    let mut handle = cache.claim_slot(0xabcdu64).expect("claim");
    cache.store_from_xgcf(&mut handle, &direct).expect("store");

    let slot = read_slot(&provider, &handle);
    let num_nodes = read_u32_at(&provider, cache.meta_num_nodes_device(), slot);
    let num_levels = read_u32_at(&provider, cache.meta_num_levels_device(), slot);
    let max_var = read_u32_at(&provider, cache.meta_max_var_device(), slot);

    assert!(num_nodes > 0);
    assert!(num_levels > 0);
    assert!(max_var > 0);
}
