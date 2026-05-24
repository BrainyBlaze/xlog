use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheConfig};
use xlog_prob::compilation::gpu_d4::compute_free_var_mask_gpu;
use xlog_prob::compilation::{
    compile_gpu_d4_and_verify, encode_cnf_gpu, map_nodes_to_vars_gpu, DeviceRandomVarList,
    GpuCompileConfig, GpuPirGraph, GpuPirRoots,
};
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
        incremental_verify: false,
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
    for v in leaf_vars.into_iter().chain(choice_vars) {
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

    let mut circuit = compile_gpu_d4_and_verify(
        &encoding.cnf,
        &encoding.decision_var_limit,
        &provider,
        &base_config,
    )
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
        decision_vars.contains(&sprinkler_var),
        "expected circuit to branch on sprinkler var"
    );

    let free_mask = compute_free_var_mask_gpu(&encoding.cnf, &circuit, &provider).unwrap();
    let mut mask_host = vec![0u8; free_mask.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&free_mask, &mut mask_host)
        .unwrap();

    assert_eq!(
        mask_host[sprinkler_var as usize], 0,
        "sprinkler var marked free"
    );
}

#[test]
fn gpu_d4_circuit_mentions_derived_query_var_dry() {
    let Some(provider) = try_provider() else {
        return;
    };

    let source = r#"
0.3::rain().
dry() :- not rain().
query(dry()).
"#;

    let provenance = extract_from_source(source).unwrap();
    let dry = provenance.query_formula("dry", &[]).unwrap();

    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[dry], &provider).unwrap();
    let encoding = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    // Map dry's PIR node id to its CNF var id and ensure the var is actually part of the CNF.
    let mut node_vars = vec![0u32; encoding.vars.node_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.node_var, &mut node_vars)
        .unwrap();
    let dry_var = node_vars[dry.as_u32() as usize];

    // The `map_nodes_to_vars_gpu` helper is used by the production exact engine to map queries.
    // Ensure it agrees with the raw node_var table.
    let mut node_ids_device = provider.memory().alloc::<u32>(1).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[dry.as_u32()], &mut node_ids_device)
        .unwrap();
    let mapped = map_nodes_to_vars_gpu(
        &encoding.vars.node_var,
        &node_ids_device,
        encoding.vars.max_var,
        &provider,
    )
    .unwrap();
    let mut mapped_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&mapped, &mut mapped_host)
        .unwrap();
    assert_eq!(
        mapped_host[0], dry_var,
        "map_nodes_to_vars_gpu returned {}, expected {} for dry()",
        mapped_host[0], dry_var
    );

    let mut cnf_num_vars = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.cnf.num_vars, &mut cnf_num_vars)
        .unwrap();
    assert!(
        dry_var >= 1 && dry_var <= cnf_num_vars[0],
        "dry var {} out of CNF bounds (num_vars={})",
        dry_var,
        cnf_num_vars[0]
    );

    let mut num_lits = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.cnf.num_lits, &mut num_lits)
        .unwrap();
    let lits_len = num_lits[0] as usize;
    let mut cnf_lits = vec![0i32; lits_len];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.cnf.literals.slice(0..lits_len), &mut cnf_lits)
        .unwrap();
    assert!(
        cnf_lits.iter().any(|lit| lit.unsigned_abs() == dry_var),
        "expected CNF to mention dry var {}, but it does not appear in literals",
        dry_var
    );

    let config = GpuCompileConfig {
        frontier_depth: 6,
        max_frontier_items: 128,
        max_depth: 128,
        smooth_node_cap: 2048,
        smooth_edge_cap: 8192,
        cdcl_restart_interval: 64,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
        incremental_verify: false,
    };

    // Smoothing list: probabilistic vars only (same as production).
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
    for v in leaf_vars.into_iter().chain(choice_vars) {
        if v != 0 {
            random_vars.push(v);
        }
    }
    random_vars.sort_unstable();
    let random_vars_device =
        DeviceRandomVarList::from_host(provider.as_ref(), &random_vars).unwrap();

    let mut base_config = config;
    if !random_vars_device.is_empty() {
        let headroom = 2u32.checked_add(random_vars_device.count()).unwrap();
        base_config.smooth_node_cap = base_config.smooth_node_cap.checked_sub(headroom).unwrap();
    }

    let mut circuit = compile_gpu_d4_and_verify(
        &encoding.cnf,
        &encoding.decision_var_limit,
        &provider,
        &base_config,
    )
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

    let mut lits = vec![0i32; circuit.lit().len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(circuit.lit(), &mut lits)
        .unwrap();
    let mut decision_vars = vec![0u32; circuit.decision_var().len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(circuit.decision_var(), &mut decision_vars)
        .unwrap();

    assert!(
        lits.iter().any(|lit| lit.unsigned_abs() == dry_var) || decision_vars.contains(&dry_var),
        "expected circuit to mention derived query var dry (var={})",
        dry_var
    );
}

#[test]
fn gpu_d4_exact_probability_of_dry_is_0_7() {
    let Some(provider) = try_provider() else {
        return;
    };

    let source = r#"
0.3::rain().
dry() :- not rain().
query(dry()).
"#;

    let provenance = extract_from_source(source).unwrap();
    let dry = provenance.query_formula("dry", &[]).unwrap();

    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[dry], &provider).unwrap();
    let encoding = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    // Identify the CNF vars we care about.
    let mut node_vars = vec![0u32; encoding.vars.node_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.node_var, &mut node_vars)
        .unwrap();
    let dry_var = node_vars[dry.as_u32() as usize];

    let mut leaf_vars = vec![0u32; encoding.vars.leaf_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.leaf_var, &mut leaf_vars)
        .unwrap();
    let rain_var = leaf_vars
        .into_iter()
        .find(|&v| v != 0)
        .expect("expected at least one leaf var for rain()");

    let config = GpuCompileConfig {
        frontier_depth: 6,
        max_frontier_items: 128,
        max_depth: 128,
        smooth_node_cap: 2048,
        smooth_edge_cap: 8192,
        cdcl_restart_interval: 64,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
        incremental_verify: false,
    };

    // Smoothing list: probabilistic vars only (same as production).
    let mut choice_vars = vec![0u32; encoding.vars.choice_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.choice_var, &mut choice_vars)
        .unwrap();
    let mut random_vars: Vec<u32> = Vec::new();
    random_vars.push(rain_var);
    for v in choice_vars {
        if v != 0 {
            random_vars.push(v);
        }
    }
    random_vars.sort_unstable();
    random_vars.dedup();
    let random_vars_device =
        DeviceRandomVarList::from_host(provider.as_ref(), &random_vars).unwrap();

    let mut base_config = config;
    if !random_vars_device.is_empty() {
        let headroom = 2u32.checked_add(random_vars_device.count()).unwrap();
        base_config.smooth_node_cap = base_config.smooth_node_cap.checked_sub(headroom).unwrap();
    }

    let mut circuit = compile_gpu_d4_and_verify(
        &encoding.cnf,
        &encoding.decision_var_limit,
        &provider,
        &base_config,
    )
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

    // Weight table: default log(1)=0 everywhere, but rain has log-prob weights.
    let max_var = encoding.vars.max_var as usize;
    let mut weights = vec![(0.0f64, 0.0f64); max_var + 1];
    weights[rain_var as usize] = (0.3f64.ln(), 0.7f64.ln());

    let mut out_log_z = provider.memory().alloc::<f64>(1).unwrap();
    circuit
        .eval_log_wmc_device_into(provider.as_ref(), &weights, &mut out_log_z)
        .unwrap();
    let mut host = [0.0f64];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&out_log_z, &mut host)
        .unwrap();
    let log_z_e = host[0];

    // Force dry() to true via weights (log_false=-inf) and recompute.
    weights[dry_var as usize].1 = f64::NEG_INFINITY;
    circuit
        .eval_log_wmc_device_into(provider.as_ref(), &weights, &mut out_log_z)
        .unwrap();
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&out_log_z, &mut host)
        .unwrap();
    let log_z_eq = host[0];

    let p_dry = (log_z_eq - log_z_e).exp();
    assert!(
        (p_dry - 0.7).abs() < 1e-6,
        "P(dry) should be 0.7, got {} (log_z_e={}, log_z_eq={})",
        p_dry,
        log_z_e,
        log_z_eq
    );
}

#[test]
fn gpu_cache_exact_probability_of_dry_is_0_7() {
    let Some(provider) = try_provider() else {
        return;
    };

    let source = r#"
0.3::rain().
dry() :- not rain().
query(dry()).
"#;

    let provenance = extract_from_source(source).unwrap();
    let dry = provenance.query_formula("dry", &[]).unwrap();

    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[dry], &provider).unwrap();
    let encoding = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    // Identify the CNF vars we care about.
    let mut node_vars = vec![0u32; encoding.vars.node_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.node_var, &mut node_vars)
        .unwrap();
    let dry_var = node_vars[dry.as_u32() as usize];

    let mut leaf_vars = vec![0u32; encoding.vars.leaf_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.leaf_var, &mut leaf_vars)
        .unwrap();
    let rain_var = leaf_vars
        .into_iter()
        .find(|&v| v != 0)
        .expect("expected at least one leaf var for rain()");

    let compile = GpuCompileConfig {
        frontier_depth: 6,
        max_frontier_items: 128,
        max_depth: 128,
        smooth_node_cap: 2048,
        smooth_edge_cap: 8192,
        cdcl_restart_interval: 64,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
        incremental_verify: false,
    };

    // Smoothing list: probabilistic vars only (same as production).
    let mut choice_vars = vec![0u32; encoding.vars.choice_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.choice_var, &mut choice_vars)
        .unwrap();
    let mut random_vars: Vec<u32> = Vec::new();
    random_vars.push(rain_var);
    for v in choice_vars {
        if v != 0 {
            random_vars.push(v);
        }
    }
    random_vars.sort_unstable();
    random_vars.dedup();
    let random_vars_device =
        DeviceRandomVarList::from_host(provider.as_ref(), &random_vars).unwrap();

    let mut base_config = compile;
    if !random_vars_device.is_empty() {
        let headroom = 2u32.checked_add(random_vars_device.count()).unwrap();
        base_config.smooth_node_cap = base_config.smooth_node_cap.checked_sub(headroom).unwrap();
    }

    let mut circuit = compile_gpu_d4_and_verify(
        &encoding.cnf,
        &encoding.decision_var_limit,
        &provider,
        &base_config,
    )
    .unwrap();
    if !random_vars_device.is_empty() {
        circuit = circuit
            .smooth_random_vars_device(
                &provider,
                random_vars_device.list(),
                random_vars_device.count(),
                compile.smooth_node_cap,
                compile.smooth_edge_cap,
            )
            .unwrap();
    }

    // Cache configured like the exact engine: single-slot, store circuit + free-var mask + weights.
    let cache_config = {
        let mut config = GpuCircuitCacheConfig::default();
        config.num_slots = 1;
        config.table_size = 1;
        config.node_cap = compile.smooth_node_cap;
        config.edge_cap = compile.smooth_edge_cap;
        config.level_cap = compile.smooth_node_cap;
        config.var_cap = encoding.cnf.var_cap;
        config
    };
    let mut cache = GpuCircuitCache::new(&provider, cache_config).unwrap();
    let mut handle = cache.claim_slot(0x1234u64).unwrap();
    // Sanity-check cache lookup results: a single-slot cache must return slot 0 and a miss
    // on first insert.
    let mut slot_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(handle.slot_device(), &mut slot_host)
        .unwrap();
    assert_eq!(
        slot_host[0], 0,
        "expected single-slot cache to allocate slot 0, got {}",
        slot_host[0]
    );
    let mut compile_needed_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(handle.compile_needed_device(), &mut compile_needed_host)
        .unwrap();
    assert_eq!(
        compile_needed_host[0], 1,
        "expected compile_needed=1 on first insert, got {}",
        compile_needed_host[0]
    );
    cache.store_from_xgcf(&mut handle, &circuit).unwrap();
    // Cached evaluation relies on device-resident meta arrays. Ensure they were written.
    let mut meta_num_levels_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(cache.meta_num_levels_device(), &mut meta_num_levels_host)
        .unwrap();
    assert_eq!(
        meta_num_levels_host[0],
        circuit.num_levels(),
        "meta_num_levels mismatch after cache store"
    );
    let mut meta_root_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(cache.meta_root_device(), &mut meta_root_host)
        .unwrap();
    assert_eq!(
        meta_root_host[0],
        circuit.root(),
        "meta_root mismatch after cache store"
    );
    let mut meta_max_var_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(cache.meta_max_var_device(), &mut meta_max_var_host)
        .unwrap();
    assert_eq!(
        meta_max_var_host[0],
        circuit.max_var(),
        "meta_max_var mismatch after cache store"
    );
    let free_mask = compute_free_var_mask_gpu(&encoding.cnf, &circuit, &provider).unwrap();
    cache.store_free_var_mask(&handle, &free_mask).unwrap();

    // Upload weights for cache storage.
    let weights_len = (encoding.vars.max_var as usize) + 1;
    let mut host_true = vec![0.0f64; weights_len];
    let mut host_false = vec![0.0f64; weights_len];
    host_true[rain_var as usize] = 0.3f64.ln();
    host_false[rain_var as usize] = 0.7f64.ln();

    let mut w_true = provider.memory().alloc::<f64>(weights_len).unwrap();
    let mut w_false = provider.memory().alloc::<f64>(weights_len).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&host_true, &mut w_true)
        .unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&host_false, &mut w_false)
        .unwrap();
    cache.overwrite_weights(&handle, &w_true, &w_false).unwrap();
    {
        let (_, cache_false) = cache.var_log_weights_mut();
        let mut false_host = vec![0.0f64; weights_len];
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&cache_false.slice(0..weights_len), &mut false_host)
            .unwrap();
        assert!(
            (false_host[rain_var as usize] - 0.7f64.ln()).abs() < 1e-12,
            "cache weights did not store rain false weight correctly"
        );
    }

    let mut out_log_z = provider.memory().alloc::<f64>(1).unwrap();
    cache
        .eval_log_wmc_device_inplace(&handle, &mut out_log_z)
        .unwrap();
    let mut host = [0.0f64];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&out_log_z, &mut host)
        .unwrap();
    let log_z_e = host[0];

    // Force dry() to true via weights and recompute.
    host_false[dry_var as usize] = f64::NEG_INFINITY;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&host_false, &mut w_false)
        .unwrap();
    cache.overwrite_weights(&handle, &w_true, &w_false).unwrap();
    {
        let (_, cache_false) = cache.var_log_weights_mut();
        let mut false_host = vec![0.0f64; weights_len];
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&cache_false.slice(0..weights_len), &mut false_host)
            .unwrap();
        assert!(
            false_host[dry_var as usize].is_infinite()
                && false_host[dry_var as usize].is_sign_negative(),
            "cache weights did not store dry false weight correctly"
        );
    }
    cache
        .eval_log_wmc_device_inplace(&handle, &mut out_log_z)
        .unwrap();
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&out_log_z, &mut host)
        .unwrap();
    let log_z_eq = host[0];

    let p_dry = (log_z_eq - log_z_e).exp();
    assert!(
        (p_dry - 0.7).abs() < 1e-6,
        "P(dry) should be 0.7, got {} (log_z_e={}, log_z_eq={})",
        p_dry,
        log_z_e,
        log_z_eq
    );
}
