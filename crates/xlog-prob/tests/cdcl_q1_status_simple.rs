use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{
    build_equivalence_queries_gpu, encode_cnf_gpu, gpu_d4, GpuCompileConfig, GpuPirGraph, GpuPirRoots,
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

fn default_compile_config_for_test(cnf: &xlog_solve::GpuCnf, memory_bytes: u64) -> xlog_core::Result<GpuCompileConfig> {
    // Keep in sync with `ExactDdnnfProgram::default_compile_config` (private) for parity with
    // production exact inference.
    let frontier_depth: u16 = 6;

    let var_cap = cnf.var_cap.max(1);
    let trail_bytes_per_item = (var_cap as u64)
        .checked_add(1)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>() as u64))
        .ok_or_else(|| xlog_core::XlogError::Compilation("trail size overflow".to_string()))?;
    let denom = trail_bytes_per_item.saturating_mul(8).max(1);
    let max_items_by_trail = memory_bytes / denom;
    let max_frontier_items = max_items_by_trail
        .clamp(8, 4096)
        .min(u64::from(u32::MAX)) as u32;

    let frontier_cap_factor = (1u64
        .checked_shl(frontier_depth as u32)
        .unwrap_or(u64::from(u32::MAX)))
    .min(u64::from(max_frontier_items)) as u32;

    let per_item_nodes = cnf
        .var_cap
        .checked_mul(5)
        .ok_or_else(|| xlog_core::XlogError::Compilation("smooth_node_cap overflow".to_string()))?
        .max(1024);
    let smooth_node_cap = per_item_nodes
        .checked_mul(frontier_cap_factor)
        .ok_or_else(|| xlog_core::XlogError::Compilation("smooth_node_cap overflow".to_string()))?;

    let mut smooth_edge_cap = smooth_node_cap
        .checked_mul(2)
        .ok_or_else(|| xlog_core::XlogError::Compilation("smooth_edge_cap overflow".to_string()))?;
    if smooth_edge_cap < max_frontier_items {
        smooth_edge_cap = max_frontier_items;
    }

    let mut cdcl_learned_bytes = memory_bytes / 8;
    if cdcl_learned_bytes < 4 * 1024 * 1024 {
        cdcl_learned_bytes = 4 * 1024 * 1024;
    }

    Ok(GpuCompileConfig {
        frontier_depth,
        max_frontier_items,
        max_depth: 128,
        smooth_node_cap,
        smooth_edge_cap,
        cdcl_restart_interval: 64,
        cdcl_learned_bytes,
        cdcl_conflict_budget: None,
    })
}

fn cdcl_config_from_compile(config: &GpuCompileConfig) -> xlog_core::Result<GpuCdclConfig> {
    if config.cdcl_restart_interval == 0 {
        return Err(xlog_core::XlogError::Compilation(
            "cdcl_restart_interval must be > 0".to_string(),
        ));
    }
    if config.cdcl_learned_bytes == 0 {
        return Err(xlog_core::XlogError::Compilation(
            "cdcl_learned_bytes must be > 0".to_string(),
        ));
    }

    // Deterministic sizing: assume average learned clause length = 4.
    const AVG_LEN: u64 = 4;
    const META_BYTES_PER_CLAUSE: u64 = 24;
    const PROOF_BYTES_PER_CLAUSE: u64 = 8 + (8 * AVG_LEN);
    const LIT_BYTES_PER_CLAUSE: u64 = 4 * AVG_LEN;

    let bytes_per_clause = META_BYTES_PER_CLAUSE
        .checked_add(PROOF_BYTES_PER_CLAUSE)
        .and_then(|v| v.checked_add(LIT_BYTES_PER_CLAUSE))
        .ok_or_else(|| {
            xlog_core::XlogError::Compilation("cdcl bytes per clause overflow".to_string())
        })?;

    let max_clauses = config
        .cdcl_learned_bytes
        .checked_div(bytes_per_clause)
        .ok_or_else(|| {
            xlog_core::XlogError::Compilation("cdcl_learned_bytes div overflow".to_string())
        })?;
    if max_clauses == 0 {
        return Err(xlog_core::XlogError::Compilation(
            "cdcl_learned_bytes too small for learned clause arena".to_string(),
        ));
    }

    let max_lits = max_clauses
        .checked_mul(AVG_LEN)
        .ok_or_else(|| xlog_core::XlogError::Compilation("max_learned_lits overflow".to_string()))?;
    let max_proof_u32 = max_clauses
        .checked_mul(2 + 2 * AVG_LEN)
        .ok_or_else(|| xlog_core::XlogError::Compilation("max_proof_u32 overflow".to_string()))?;

    let max_learned_clauses = u32::try_from(max_clauses).map_err(|_| {
        xlog_core::XlogError::Compilation("max_learned_clauses exceeds u32::MAX".to_string())
    })?;
    let max_learned_lits = u32::try_from(max_lits).map_err(|_| {
        xlog_core::XlogError::Compilation("max_learned_lits exceeds u32::MAX".to_string())
    })?;
    let max_proof_u32 = u32::try_from(max_proof_u32).map_err(|_| {
        xlog_core::XlogError::Compilation("max_proof_u32 exceeds u32::MAX".to_string())
    })?;

    let reduce_interval = config
        .cdcl_restart_interval
        .checked_mul(20)
        .ok_or_else(|| xlog_core::XlogError::Compilation("cdcl reduce_interval overflow".to_string()))?;

    Ok(GpuCdclConfig {
        max_learned_clauses,
        max_learned_lits,
        max_proof_u32,
        restart_base: config.cdcl_restart_interval,
        reduce_interval,
    })
}

#[test]
fn cdcl_q1_solves_unsat_without_trap_simple_wfs_asymmetric() {
    let Some(provider) = try_provider() else {
        return;
    };

    let source = r#"
p() :- not q().
q().
query(p()).
query(q()).
"#;
    let provenance = extract_from_source(source).expect("extract provenance");

    // Mirror `ExactDdnnfProgram::compile_provenance_with_gpu`: roots are all query formulas.
    let p = provenance.query_formula("p", &[]).expect("p() query formula");
    let q = provenance.query_formula("q", &[]).expect("q() query formula");
    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider).expect("upload pir");
    let gpu_roots = GpuPirRoots::from_host(&[p, q], &provider).expect("upload roots");
    let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, &provider).expect("encode cnf");

    let compile = default_compile_config_for_test(&encoding.cnf, 1024 * 1024 * 1024).expect("compile config");
    let cdcl = cdcl_config_from_compile(&compile).expect("cdcl config");

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

    let mut decision_limit_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&encoding.decision_var_limit, &mut decision_limit_host)
        .expect("read decision_var_limit");
    let mut q1_num_vars_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&queries.q1.num_vars, &mut q1_num_vars_host)
        .expect("read q1 num_vars");
    let mut q1_num_clauses_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&queries.q1.num_clauses, &mut q1_num_clauses_host)
        .expect("read q1 num_clauses");
    let mut q1_num_lits_host = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&queries.q1.num_lits, &mut q1_num_lits_host)
        .expect("read q1 num_lits");
    assert_eq!(
        decision_limit_host[0], 0,
        "expected decision_var_limit=0 for deterministic program (no leaf/choice vars), got {}",
        decision_limit_host[0]
    );
    eprintln!(
        "q1 meta: var_cap={} clause_cap={} lit_cap={} num_vars={} num_clauses={} num_lits={} decision_limit={}",
        queries.q1.var_cap,
        queries.q1.clause_cap,
        queries.q1.lit_cap,
        q1_num_vars_host[0],
        q1_num_clauses_host[0],
        q1_num_lits_host[0],
        decision_limit_host[0]
    );

    let solver = GpuCdclSolver::new(provider.clone(), cdcl);
    let raw = solver
        .solve_raw_with_branch_limit(&queries.q1, &encoding.cnf.num_vars)
        .expect("solve q1 raw");

    let mut status_host = [0i32];
    let mut error_host = [0i32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&raw.out_status, &mut status_host)
        .expect("read out_status");
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&raw.out_error, &mut error_host)
        .expect("read out_error");

    assert_eq!(error_host[0], 0, "expected q1 out_error=0, got {}", error_host[0]);
    assert_eq!(
        status_host[0], 0,
        "expected q1 UNSAT (status=0), got status={} error={}",
        status_host[0], error_host[0]
    );
}
