use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheConfig};
use xlog_prob::compilation::{
    compile_gpu_d4_and_verify_cached, DeviceRandomVarList, GpuCompileConfig,
};
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

#[test]
fn gpu_cache_compile_reuses_slot() {
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

    let clauses = vec![Clause::new(vec![Literal::positive(0)])];
    let instance = SolveInstance::new(1, clauses);
    let cnf = GpuCnf::from_host(&instance, &provider).unwrap();

    let compile_config = GpuCompileConfig {
        frontier_depth: 0,
        max_frontier_items: 8,
        max_depth: 32,
        smooth_node_cap: 1024,
        smooth_edge_cap: 4096,
        cdcl_restart_interval: 32,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
        incremental_verify: false,
        ..Default::default()
    };
    let level_cap = u32::from(compile_config.max_depth)
        .checked_mul(2)
        .and_then(|v| v.checked_add(8))
        .expect("level_cap overflow");
    let config = GpuCircuitCacheConfig {
        num_slots: 1,
        table_size: 4,
        node_cap: compile_config.smooth_node_cap,
        edge_cap: compile_config.smooth_edge_cap,
        level_cap,
        var_cap: cnf.var_cap,
        ..Default::default()
    };
    let mut cache = GpuCircuitCache::new(&provider, config).unwrap();

    let random_vars =
        DeviceRandomVarList::from_host(provider.as_ref(), &[]).expect("random vars upload");
    let (h1, _) = compile_gpu_d4_and_verify_cached(
        &cnf,
        &cnf.num_vars,
        &provider,
        &compile_config,
        &mut cache,
        &random_vars,
        None, // no PIR available, skip disk cache
    )
    .expect("compile 1");
    let (h2, _) = compile_gpu_d4_and_verify_cached(
        &cnf,
        &cnf.num_vars,
        &provider,
        &compile_config,
        &mut cache,
        &random_vars,
        None,
    )
    .expect("compile 2");

    let mut slot1 = vec![0u32; 1];
    let mut slot2 = vec![0u32; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(h1.slot_device(), &mut slot1)
        .unwrap();
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(h2.slot_device(), &mut slot2)
        .unwrap();

    assert_eq!(slot1[0], slot2[0]);
}
