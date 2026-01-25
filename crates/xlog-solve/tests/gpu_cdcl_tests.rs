use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

use xlog_solve::{Clause, GpuCdclConfig, GpuCdclSolver, GpuCnf, Literal, SolveInstance};

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

#[test]
fn gpu_cdcl_sat_unit_clause() {
    let Some(provider) = try_provider() else {
        return;
    };

    // (x0)
    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    let solver = GpuCdclSolver::new(provider.clone(), GpuCdclConfig::default());
    let assignment = solver.solve_expect_sat(&cnf).expect("solve_expect_sat");

    // Download assignment and sanity check x0=true (DIMACS var 1).
    let mut assign_host = vec![0i8; 2];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&assignment, &mut assign_host)
        .expect("dtoh assignment");
    assert_eq!(assign_host[1], 1);
}

#[test]
fn gpu_cdcl_unsat_contradictory_units() {
    let Some(provider) = try_provider() else {
        return;
    };

    // (x0) AND (~x0)
    let instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    let solver = GpuCdclSolver::new(provider.clone(), GpuCdclConfig::default());
    solver.solve_expect_unsat(&cnf).expect("solve_expect_unsat");
}
