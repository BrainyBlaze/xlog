use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{compile_gpu_d4_and_verify, GpuCompileConfig};
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GiB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!(
                "Skipping test: failed to create CUDA kernel provider: {}",
                e
            );
            None
        }
    }
}

/// Build a shared GpuCompileConfig with the given `incremental_verify` setting.
fn make_config(incremental_verify: bool) -> GpuCompileConfig {
    GpuCompileConfig {
        frontier_depth: 0,
        max_frontier_items: 8,
        max_depth: 32,
        smooth_node_cap: 1024,
        smooth_edge_cap: 4096,
        cdcl_restart_interval: 32,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
        incremental_verify,
    }
}

/// Original smoke test: workspace path succeeds on a unit clause.
#[test]
fn gpu_workspace_verify_unit_clause() {
    let Some(provider) = try_provider() else {
        return;
    };

    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    let config = make_config(true);

    let circuit = compile_gpu_d4_and_verify(&phi, &phi.num_vars, &provider, &config)
        .expect("compile must succeed");
    assert!(circuit.num_nodes() > 0);
}

/// Parity test: default path (incremental_verify=false) and workspace path
/// (incremental_verify=true) produce identical node counts on a unit clause.
#[test]
fn gpu_workspace_verify_parity_unit_clause() {
    let Some(provider) = try_provider() else {
        return;
    };

    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    // Run default path (no workspace reuse).
    let config_default = make_config(false);
    let circuit_default =
        compile_gpu_d4_and_verify(&phi, &phi.num_vars, &provider, &config_default)
            .expect("default path must succeed");

    // Run workspace path (incremental verify).
    let config_workspace = make_config(true);
    let circuit_workspace =
        compile_gpu_d4_and_verify(&phi, &phi.num_vars, &provider, &config_workspace)
            .expect("workspace path must succeed");

    assert!(
        circuit_default.num_nodes() > 0,
        "default path produced zero nodes"
    );
    assert_eq!(
        circuit_default.num_nodes(),
        circuit_workspace.num_nodes(),
        "node count mismatch: default={} vs workspace={}",
        circuit_default.num_nodes(),
        circuit_workspace.num_nodes(),
    );

    eprintln!(
        "parity_unit_clause: both paths produced {} nodes",
        circuit_default.num_nodes()
    );
}

/// Parity test on a nontrivial multi-variable, multi-clause formula.
///
/// Formula (3 vars, 4 clauses):
///   (x0 | x1)  &  (!x0 | x2)  &  (!x1 | !x2)  &  (x0 | x2)
///
/// This exercises meaningful selector growth across the compilation/verification
/// pipeline, unlike a trivial unit clause.
#[test]
fn gpu_workspace_verify_parity_multi_clause() {
    let Some(provider) = try_provider() else {
        return;
    };

    let clauses = vec![
        Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
        Clause::new(vec![Literal::negative(0), Literal::positive(2)]),
        Clause::new(vec![Literal::negative(1), Literal::negative(2)]),
        Clause::new(vec![Literal::positive(0), Literal::positive(2)]),
    ];
    let instance = SolveInstance::new(3, clauses);
    let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    // Run default path.
    let config_default = make_config(false);
    let circuit_default =
        compile_gpu_d4_and_verify(&phi, &phi.num_vars, &provider, &config_default)
            .expect("default path must succeed on multi-clause formula");

    // Run workspace path.
    let config_workspace = make_config(true);
    let circuit_workspace =
        compile_gpu_d4_and_verify(&phi, &phi.num_vars, &provider, &config_workspace)
            .expect("workspace path must succeed on multi-clause formula");

    assert!(
        circuit_default.num_nodes() > 0,
        "default path produced zero nodes on multi-clause formula"
    );
    assert_eq!(
        circuit_default.num_nodes(),
        circuit_workspace.num_nodes(),
        "node count mismatch on multi-clause: default={} vs workspace={}",
        circuit_default.num_nodes(),
        circuit_workspace.num_nodes(),
    );

    eprintln!(
        "parity_multi_clause: both paths produced {} nodes",
        circuit_default.num_nodes()
    );
}
