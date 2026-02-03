use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{
    compile_gpu_d4_and_verify, encode_cnf_gpu, DeviceRandomVarList, GpuCompileConfig,
    GpuPirGraph, GpuPirRoots,
};
use xlog_prob::compilation::gpu_d4::compute_free_var_mask_gpu;
use xlog_prob::provenance::extract_from_source;

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: failed to create provider: {}", e);
            None
        }
    }
}

#[test]
fn gpu_d4_circuit_mentions_sprinkler_var() {
    let Some(provider) = try_provider() else {
        return;
    };

    let source = r#"
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
query(rain()).
query(sprinkler()).
"#;

    let provenance = extract_from_source(source).unwrap();
    let wet = provenance.query_formula("wet", &[]).unwrap();
    let rain = provenance.query_formula("rain", &[]).unwrap();
    let sprinkler = provenance.query_formula("sprinkler", &[]).unwrap();

    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[wet, rain, sprinkler], &provider).unwrap();
    let encoding = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    let config = GpuCompileConfig {
        frontier_depth: 6,
        max_frontier_items: 128,
        max_depth: 128,
        smooth_node_cap: 2048,
        smooth_edge_cap: 8192,
        cdcl_restart_interval: 64,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
    };

    // Collect the random-var list for smoothing (device-side compaction is tested elsewhere).
    // This test uses host reads only for assertions.
    let mut leaf_vars = vec![0u32; encoding.vars.leaf_var.len()];
    let mut choice_vars = vec![0u32; encoding.vars.choice_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.leaf_var, &mut leaf_vars)
        .unwrap();
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.choice_var, &mut choice_vars)
        .unwrap();
    let mut random_vars: Vec<u32> = Vec::new();
    for v in leaf_vars.into_iter().chain(choice_vars.into_iter()) {
        if v != 0 {
            random_vars.push(v);
        }
    }
    random_vars.sort_unstable();
    let random_vars_device =
        DeviceRandomVarList::from_host(provider.as_ref(), &random_vars).unwrap();

    // Reserve the same smoothing headroom as the production cached path.
    let mut base_config = config;
    if !random_vars_device.is_empty() {
        let headroom = 2u32.checked_add(random_vars_device.count()).unwrap();
        base_config.smooth_node_cap = base_config.smooth_node_cap.checked_sub(headroom).unwrap();
    }

    let mut circuit =
        compile_gpu_d4_and_verify(&encoding.cnf, &encoding.decision_var_limit, &provider, &base_config)
            .unwrap();
    if !random_vars_device.is_empty() {
        circuit = circuit
            .smooth_random_vars_device(
                &provider,
                random_vars_device.list(),
                random_vars_device.count(),
                config.smooth_node_cap,
                config.smooth_edge_cap,
            )
            .unwrap();
    }

    let mut node_vars = vec![0u32; encoding.vars.node_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.node_var, &mut node_vars)
        .unwrap();
    let sprinkler_var = node_vars[sprinkler.as_u32() as usize];

    let mut lits = vec![0i32; circuit.lit().len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(circuit.lit(), &mut lits)
        .unwrap();

    assert!(
        lits.iter().any(|lit| lit.unsigned_abs() == sprinkler_var),
        "expected circuit to mention sprinkler var"
    );

    let mut decision_vars = vec![0u32; circuit.decision_var().len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(circuit.decision_var(), &mut decision_vars)
        .unwrap();
    assert!(
        decision_vars.iter().any(|&v| v == sprinkler_var),
        "expected circuit to branch on sprinkler var"
    );

    let free_mask = compute_free_var_mask_gpu(&encoding.cnf, &circuit, &provider).unwrap();
    let mut mask_host = vec![0u8; free_mask.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&free_mask, &mut mask_host)
        .unwrap();

    assert_eq!(mask_host[sprinkler_var as usize], 0, "sprinkler var marked free");
}
