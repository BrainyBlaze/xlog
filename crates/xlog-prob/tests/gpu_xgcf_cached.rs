use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheConfig};
use xlog_prob::gpu::GpuXgcf;
use xlog_prob::xgcf::{Xgcf, XgcfNodeType};

#[test]
fn cached_eval_matches_direct_eval() {
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

    let mut handle = cache.claim_slot(0xabcdefu64).expect("claim");
    cache.store_from_xgcf(&mut handle, &direct).expect("store");

    let mut out_direct = provider.memory().alloc::<f64>(1).unwrap();
    let mut out_cached = provider.memory().alloc::<f64>(1).unwrap();

    direct
        .eval_log_wmc_device_inplace(&provider, &mut out_direct)
        .expect("direct eval");
    cache
        .eval_log_wmc_device_inplace(&handle, &mut out_cached)
        .expect("cached eval");

    let mut hd = vec![0.0f64; 1];
    let mut hc = vec![0.0f64; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&out_direct, &mut hd)
        .unwrap();
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&out_cached, &mut hc)
        .unwrap();

    assert_eq!(hd[0].to_bits(), hc[0].to_bits());
}
