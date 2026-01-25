use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

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
            eprintln!("Skipping test: failed to create CUDA kernel provider: {}", e);
            None
        }
    }
}

#[test]
fn gpu_cnf_roundtrip_offsets_and_lits() {
    let Some(provider) = try_provider() else {
        return;
    };

    // (x0) AND (~x1 OR x2)
    let instance = SolveInstance::new(
        3,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(1), Literal::positive(2)]),
        ],
    );

    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload should succeed");

    let mut offsets_host = vec![0u32; (cnf.clause_cap as usize) + 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&cnf.clause_offsets, &mut offsets_host)
        .expect("dtoh offsets");

    let mut lits_host = vec![0i32; cnf.num_literals_cap()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&cnf.literals, &mut lits_host)
        .expect("dtoh lits");

    assert_eq!(offsets_host, vec![0, 1, 3]);
    // DIMACS is 1-based.
    assert_eq!(lits_host, vec![1, -2, 3]);
}
