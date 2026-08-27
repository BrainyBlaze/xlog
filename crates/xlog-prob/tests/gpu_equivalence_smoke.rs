use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::CudaKernelProvider;

use xlog_prob::compilation::{validate_equivalence_gpu, GpuEquivalenceConfig};
use xlog_prob::gpu::GpuXgcf;
use xlog_prob::xgcf::{Xgcf, XgcfNodeType};

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
fn gpu_equivalence_smoke_phi_is_lit() {
    let Some(provider) = try_provider() else {
        return;
    };

    // φ = (x0)
    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let phi = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    // C = x0 (single literal circuit).
    let xgcf = Xgcf {
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
    let circuit = GpuXgcf::upload(&provider, &xgcf).expect("GpuXgcf upload");

    validate_equivalence_gpu(
        &phi,
        &phi.num_vars,
        &circuit,
        &provider,
        GpuEquivalenceConfig::default(),
    )
    .expect("equivalence should hold");
}
