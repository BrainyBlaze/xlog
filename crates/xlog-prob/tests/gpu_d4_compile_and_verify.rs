use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::CudaKernelProvider;
use xlog_prob::compilation::{compile_gpu_d4_and_verify, GpuCompileConfig};
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    match xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(1024 * 1024 * 1024))
        .build()
    {
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

#[test]
fn gpu_d4_compile_and_verify_unit_clause() {
    let Some(provider) = try_provider() else {
        return;
    };

    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    let config = GpuCompileConfig {
        frontier_depth: 0,
        max_frontier_items: 8,
        max_depth: 32,
        smooth_node_cap: 1024,
        smooth_edge_cap: 4096,
        cdcl_restart_interval: 32,
        cdcl_learned_bytes: 4 * 1024 * 1024,
        cdcl_conflict_budget: None,
        incremental_verify: false,
    };

    let circuit = compile_gpu_d4_and_verify(&phi, &phi.num_vars, &provider, &config)
        .expect("compile must succeed");
    assert!(circuit.num_nodes() > 0);
}
