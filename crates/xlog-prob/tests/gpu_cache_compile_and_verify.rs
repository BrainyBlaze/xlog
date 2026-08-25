use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_prob::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheConfig};
use xlog_prob::compilation::{
    compile_gpu_d4_and_verify_cached, DeviceRandomVarList, GpuCompileConfig,
};
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

fn provider_or_skip() -> Option<Arc<xlog_cuda::CudaKernelProvider>> {
    match xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(1 << 30)).build() {
        Ok(provider) => Some(Arc::new(provider)),
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            None
        }
    }
}

fn compile_config() -> GpuCompileConfig {
    GpuCompileConfig {
        frontier_depth: 0,
        max_frontier_items: 8,
        max_depth: 32,
        smooth_node_cap: 1024,
        smooth_edge_cap: 4096,
        cdcl_restart_interval: 32,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
        incremental_verify: false,
    }
}

fn cache_config(compile: &GpuCompileConfig, cnf: &GpuCnf) -> GpuCircuitCacheConfig {
    let level_cap = u32::from(compile.max_depth)
        .checked_mul(2)
        .and_then(|value| value.checked_add(8))
        .expect("level_cap overflow");
    let mut config = GpuCircuitCacheConfig::default();
    config.num_slots = 1;
    config.table_size = 4;
    config.node_cap = compile.smooth_node_cap;
    config.edge_cap = compile.smooth_edge_cap;
    config.level_cap = level_cap;
    config.var_cap = cnf.var_cap;
    config
}

#[test]
fn gpu_cache_compile_reuses_slot() {
    let Some(provider) = provider_or_skip() else {
        return;
    };

    let clauses = vec![Clause::new(vec![Literal::positive(0)])];
    let instance = SolveInstance::new(1, clauses);
    let cnf = GpuCnf::from_host(&instance, &provider).unwrap();

    let compile_config = compile_config();
    let config = cache_config(&compile_config, &cnf);
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

#[test]
fn smoothing_changes_the_exact_circuit_certified_for_cache_storage() {
    let Some(provider) = provider_or_skip() else {
        return;
    };

    // x OR y has a decision branch that omits y. Smoothing both random
    // variables must add a y OR !y tautology to that branch.
    let instance = SolveInstance::new(
        2,
        vec![Clause::new(vec![
            Literal::positive(0),
            Literal::positive(1),
        ])],
    );
    let cnf = GpuCnf::from_host(&instance, &provider).expect("CNF upload");
    let compile = compile_config();

    let mut base_compile = compile;
    base_compile.smooth_node_cap = base_compile
        .smooth_node_cap
        .checked_sub(4)
        .expect("smoothing headroom");
    let base = xlog_prob::compilation::compile_gpu_d4_and_verify(
        &cnf,
        &cnf.num_vars,
        &provider,
        &base_compile,
    )
    .expect("base circuit certification");
    let mut base_nodes = [0_u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(base.num_nodes_device(), &mut base_nodes)
        .expect("base node count");

    let random_vars =
        DeviceRandomVarList::from_host(provider.as_ref(), &[1, 2]).expect("random vars upload");
    let mut cache = GpuCircuitCache::new(&provider, cache_config(&compile, &cnf)).expect("cache");
    let (_handle, _) = compile_gpu_d4_and_verify_cached(
        &cnf,
        &cnf.num_vars,
        &provider,
        &compile,
        &mut cache,
        &random_vars,
        None,
    )
    .expect("smoothed final-circuit certification");

    let mut cached_nodes = [0_u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(cache.meta_num_nodes_device(), &mut cached_nodes)
        .expect("cached node count");
    assert!(
        cached_nodes[0] > base_nodes[0],
        "fixture must require smoothing: base={} cached={}",
        base_nodes[0],
        cached_nodes[0]
    );
}
