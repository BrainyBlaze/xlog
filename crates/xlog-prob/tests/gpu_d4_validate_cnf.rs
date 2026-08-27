use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::CudaKernelProvider;
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

use xlog_prob::compilation::gpu_d4::validate_cnf_gpu;

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
fn gpu_d4_validate_cnf_accepts_well_formed_cnf() {
    let Some(provider) = try_provider() else {
        return;
    };

    // φ = (x0) ∧ (¬x0 ∨ x1)
    let instance = SolveInstance::new(
        2,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
        ],
    );
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    validate_cnf_gpu(&cnf, &provider).expect("CNF validation should succeed");
}
