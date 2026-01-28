use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheConfig};
use xlog_prob::gpu::GpuXgcf;
use xlog_prob::xgcf::{Xgcf, XgcfNodeType};

#[test]
fn cache_store_writes_metadata() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
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

    let meta = cache.read_meta_host(&handle).expect("meta");
    assert!(meta.num_nodes > 0);
    assert!(meta.num_levels > 0);
    assert!(meta.max_var > 0);
}
