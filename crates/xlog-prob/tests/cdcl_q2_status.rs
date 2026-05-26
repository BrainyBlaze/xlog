use std::collections::HashSet;
use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{
    build_equivalence_queries_gpu, encode_cnf_gpu, gpu_d4, GpuCompileConfig, GpuPirGraph,
    GpuPirRoots,
};
use xlog_prob::provenance::extract_from_source;
use xlog_solve::{GpuCdclConfig, GpuCdclSolver};

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

fn template_source_addition(num_labels: usize) -> String {
    assert!(num_labels > 0);

    let p = 1.0f64 / (num_labels as f64);
    let scale = 0.9999999;
    let normalized_p = (p * scale).max(1e-10);

    let mut source = String::new();
    for input_pos in 0..2usize {
        for label_idx in 0..num_labels {
            if label_idx > 0 {
                source.push_str("; ");
            }
            source.push_str(&format!(
                "{:.10}::digit({}, {})",
                normalized_p, input_pos, label_idx
            ));
        }
        source.push_str(".\n");
    }

    source.push_str("addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.\n");

    let max_sum = 2usize * (num_labels.saturating_sub(1));
    for sum in 0..=max_sum {
        source.push_str(&format!("query(addition(0, 1, {})).\n", sum));
    }

    source
}

fn cdcl_config_from_learned_bytes(
    cdcl_learned_bytes: u64,
    restart_base: u32,
    avg_len: u64,
) -> GpuCdclConfig {
    // Must match `cdcl_config_from_compile` sizing heuristics in `crates/xlog-prob/src/compilation/mod.rs`.
    const META_BYTES_PER_CLAUSE: u64 = 24;
    let proof_bytes_per_clause = 8 + (8 * avg_len);
    let lit_bytes_per_clause = 4 * avg_len;

    let bytes_per_clause = META_BYTES_PER_CLAUSE + proof_bytes_per_clause + lit_bytes_per_clause;
    let max_clauses = (cdcl_learned_bytes / bytes_per_clause).max(1);
    let max_lits = max_clauses * avg_len;
    let max_proof_u32 = max_clauses * (2 + 2 * avg_len);

    let max_learned_clauses = u32::try_from(max_clauses).unwrap_or(u32::MAX);
    let max_learned_lits = u32::try_from(max_lits).unwrap_or(u32::MAX);
    let max_proof_u32 = u32::try_from(max_proof_u32).unwrap_or(u32::MAX);

    let reduce_interval = restart_base.saturating_mul(20);

    {
        let mut config = GpuCdclConfig::default();
        config.max_learned_clauses = max_learned_clauses;
        config.max_learned_lits = max_learned_lits;
        config.max_proof_u32 = max_proof_u32;
        config.restart_base = restart_base;
        config.reduce_interval = reduce_interval;
        config
    }
}

#[test]
fn cdcl_q2_reports_status_and_error_without_trap() {
    let Some(provider) = try_provider() else {
        return;
    };

    // Mirror the python training repro (10 labels, 2 digits).
    let source = template_source_addition(10);
    let provenance = extract_from_source(&source).expect("extract provenance");

    // Mirror `ExactDdnnfProgram::compile_provenance_with_gpu`: roots are all query formulas.
    let mut roots_set = HashSet::new();
    for atom in &provenance.queries {
        if let Some(id) = provenance.query_formula(&atom.predicate, &atom.args) {
            roots_set.insert(id);
        }
    }
    let mut roots: Vec<_> = roots_set.into_iter().collect();
    roots.sort();
    assert!(!roots.is_empty(), "expected at least one query root");

    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider).expect("upload pir");
    let gpu_roots = GpuPirRoots::from_host(&roots, &provider).expect("upload roots");
    let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, &provider).expect("encode cnf");

    // Reconstruct the compile config that `ExactDdnnfProgram` uses (for 1GiB budget).
    let memory_bytes = 1024u64 * 1024 * 1024;
    let frontier_depth: u16 = 6;
    let max_frontier_items: u32 = 4096;

    // D4 smoothing caps: base formula from `default_compile_config` minus headroom for smoothing.
    let per_item_nodes = encoding.cnf.var_cap.saturating_mul(5).max(1024);
    let frontier_cap_factor = 1u32 << (frontier_depth as u32);
    let base_node_cap = per_item_nodes.saturating_mul(frontier_cap_factor);

    // Random vars in this template: 2 * num_labels probabilistic facts.
    let random_var_count: u32 = 2 * 10;
    let headroom = 2u32 + random_var_count;
    let smooth_node_cap = base_node_cap.saturating_sub(headroom);
    let smooth_edge_cap = base_node_cap.saturating_mul(2);

    let compile = GpuCompileConfig {
        frontier_depth,
        max_frontier_items,
        max_depth: 128,
        smooth_node_cap,
        smooth_edge_cap,
        cdcl_restart_interval: 64,
        cdcl_learned_bytes: memory_bytes / 8,
        cdcl_conflict_budget: None,
        incremental_verify: false,
    };

    let mut compile_needed = provider
        .memory()
        .alloc::<u32>(1)
        .expect("alloc compile gate");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u32], &mut compile_needed)
        .expect("upload compile gate");

    let circuit = gpu_d4::compile_gpu_d4_gated(&encoding.cnf, &provider, &compile, &compile_needed)
        .expect("compile d4");

    let queries =
        build_equivalence_queries_gpu(&encoding.cnf, &circuit, &provider).expect("build queries");

    // The production pipeline currently budgets CDCL assuming small average learned clause length.
    // For this debug harness, use a larger avg_len to see whether q2 is failing due to budgeting
    // rather than solver correctness.
    let cdcl = cdcl_config_from_learned_bytes(
        compile.cdcl_learned_bytes,
        compile.cdcl_restart_interval,
        64,
    );
    eprintln!(
        "cdcl config: max_learned_clauses={} max_learned_lits={} max_proof_u32={} restart_base={} reduce_interval={}",
        cdcl.max_learned_clauses,
        cdcl.max_learned_lits,
        cdcl.max_proof_u32,
        cdcl.restart_base,
        cdcl.reduce_interval
    );
    let solver = GpuCdclSolver::new(provider.clone(), cdcl);

    let raw = solver
        .solve_raw_with_decision_ranges(
            &queries.q2,
            &encoding.decision_var_limit,
            &queries.q2_unsat_var_base,
            &encoding.cnf.num_clauses,
        )
        .expect("cdcl raw solve");

    let mut st = [0i32];
    let mut er = [0i32];
    let mut learned = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&raw.out_status, &mut st)
        .expect("dtoh status");
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&raw.out_error, &mut er)
        .expect("dtoh error");
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&raw.out_learned_count, &mut learned)
        .expect("dtoh learned_count");

    eprintln!(
        "q2 cdcl raw: status={} error={} learned_count={}",
        st[0], er[0], learned[0]
    );
    assert_eq!(
        er[0], 0,
        "expected q2 to complete without solver errors (error=0), got {}",
        er[0]
    );
    assert_eq!(
        st[0], 0,
        "expected q2 to be UNSAT (status=0), got {}",
        st[0]
    );
}
