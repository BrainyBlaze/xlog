use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{
    build_evidence_by_var_gpu, encode_cnf_gpu, map_nodes_to_vars_gpu, GpuPirGraph, GpuPirRoots,
};
use xlog_prob::pir::PirNode;
use xlog_prob::provenance::extract_from_source;
use xlog_prob::compilation::build_weights_gpu;

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
fn gpu_query_var_mapping_wet_sprinkler() {
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

    let host_nodes = [sprinkler.as_u32(), wet.as_u32(), rain.as_u32()];
    let mut d_nodes = provider
        .memory()
        .alloc::<u32>(host_nodes.len())
        .unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&host_nodes, &mut d_nodes)
        .unwrap();

    let vars = map_nodes_to_vars_gpu(
        &encoding.vars.node_var,
        &d_nodes,
        encoding.vars.max_var,
        &provider,
    )
    .unwrap();
    let mut host_vars = vec![0u32; host_nodes.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&vars, &mut host_vars)
        .unwrap();

    let mut leaf_vars = vec![0u32; encoding.vars.leaf_var.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.vars.leaf_var, &mut leaf_vars)
        .unwrap();

    let wet_var = host_vars[1];
    assert!(wet_var != 0, "wet var missing");

    let PirNode::Lit { leaf: rain_leaf } = provenance.pir.node(rain).unwrap() else {
        panic!("rain formula is not a leaf literal");
    };
    let PirNode::Lit {
        leaf: sprinkler_leaf,
    } = provenance.pir.node(sprinkler).unwrap() else {
        panic!("sprinkler formula is not a leaf literal");
    };

    let rain_var = leaf_vars[rain_leaf.as_u32() as usize];
    let sprinkler_var = leaf_vars[sprinkler_leaf.as_u32() as usize];
    assert_eq!(host_vars[2], rain_var);
    assert_eq!(host_vars[0], sprinkler_var);
    assert_ne!(wet_var, sprinkler_var);
    assert_ne!(wet_var, rain_var);
}

#[test]
fn gpu_evidence_by_var_marks_wet_only() {
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

    let host_nodes = [wet.as_u32()];
    let mut d_nodes = provider
        .memory()
        .alloc::<u32>(host_nodes.len())
        .unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&host_nodes, &mut d_nodes)
        .unwrap();
    let host_vals = [1u8];
    let mut d_vals = provider.memory().alloc::<u8>(host_vals.len()).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&host_vals, &mut d_vals)
        .unwrap();

    let vars = map_nodes_to_vars_gpu(
        &encoding.vars.node_var,
        &d_nodes,
        encoding.vars.max_var,
        &provider,
    )
    .unwrap();
    let mut host_vars = vec![0u32; host_nodes.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&vars, &mut host_vars)
        .unwrap();
    let wet_var = host_vars[0];
    assert!(wet_var != 0, "wet var missing");

    let evidence = build_evidence_by_var_gpu(
        &encoding.vars.node_var,
        &d_nodes,
        &d_vals,
        encoding.vars.max_var,
        &provider,
    )
    .unwrap();
    let mut evidence_host = vec![0u8; evidence.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&evidence, &mut evidence_host)
        .unwrap();

    assert_eq!(evidence_host[wet_var as usize], 1);
    assert_eq!(evidence_host[1], 0);
    assert_eq!(evidence_host[2], 0);
}

#[test]
fn gpu_weights_respect_wet_evidence() {
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

    let max_leaf = provenance
        .leaf_probs
        .keys()
        .map(|leaf| leaf.as_u32())
        .max()
        .unwrap_or(0);
    let leaf_len = max_leaf as usize + 1;
    let mut leaf_probs = vec![0.0f64; leaf_len];
    for (leaf, p) in &provenance.leaf_probs {
        leaf_probs[leaf.as_u32() as usize] = *p;
    }

    let choice_true: Vec<f64> = Vec::new();
    let choice_false: Vec<f64> = Vec::new();

    let mut d_leaf = provider.memory().alloc::<f64>(leaf_probs.len()).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&leaf_probs, &mut d_leaf)
        .unwrap();
    let mut d_choice_true = provider.memory().alloc::<f64>(choice_true.len()).unwrap();
    let mut d_choice_false = provider.memory().alloc::<f64>(choice_false.len()).unwrap();
    if !choice_true.is_empty() {
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&choice_true, &mut d_choice_true)
            .unwrap();
    }
    if !choice_false.is_empty() {
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&choice_false, &mut d_choice_false)
            .unwrap();
    }

    let mut d_nodes = provider.memory().alloc::<u32>(1).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[wet.as_u32()], &mut d_nodes)
        .unwrap();
    let mut d_vals = provider.memory().alloc::<u8>(1).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u8], &mut d_vals)
        .unwrap();

    let evidence_by_var = build_evidence_by_var_gpu(
        &encoding.vars.node_var,
        &d_nodes,
        &d_vals,
        encoding.vars.max_var,
        &provider,
    )
    .unwrap();

    let weights = build_weights_gpu(
        &encoding.vars,
        &d_leaf,
        &d_choice_true,
        &d_choice_false,
        &evidence_by_var,
        &provider,
    )
    .unwrap();

    let mut log_true = vec![0.0f64; weights.log_true.len()];
    let mut log_false = vec![0.0f64; weights.log_false.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&weights.log_true, &mut log_true)
        .unwrap();
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&weights.log_false, &mut log_false)
        .unwrap();

    let rain_var = 1usize;
    let sprinkler_var = 2usize;
    let wet_var = 3usize;

    assert!((log_true[rain_var] - 0.7_f64.ln()).abs() < 1e-9);
    assert!((log_false[rain_var] - 0.3_f64.ln()).abs() < 1e-9);
    assert!((log_true[sprinkler_var] - 0.2_f64.ln()).abs() < 1e-9);
    assert!((log_false[sprinkler_var] - 0.8_f64.ln()).abs() < 1e-9);
    assert_eq!(log_true[wet_var], 0.0);
    assert!(log_false[wet_var].is_infinite() && log_false[wet_var].is_sign_negative());
}
