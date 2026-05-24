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
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
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
fn gpu_cdcl_sat_and_unsat_paths_do_not_perform_data_plane_dtoh_reads() {
    let Some(provider) = try_provider() else {
        return;
    };

    let solver = GpuCdclSolver::new(provider.clone(), GpuCdclConfig::default());

    let sat_instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let sat_cnf = GpuCnf::from_host(&sat_instance, &provider).expect("upload SAT CNF");
    provider.reset_host_transfer_stats();
    provider.reset_d2h_transfer_count();
    let assignment = solver.solve_expect_sat(&sat_cnf).expect("solve SAT");
    assert_eq!(assignment.len(), 2);
    let sat_transfers = provider.host_transfer_stats();
    assert_eq!(sat_transfers.dtoh_calls, 0);
    assert_eq!(sat_transfers.dtoh_bytes, 0);
    assert_eq!(provider.d2h_transfer_count(), 0);

    let unsat_instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let unsat_cnf = GpuCnf::from_host(&unsat_instance, &provider).expect("upload UNSAT CNF");
    provider.reset_host_transfer_stats();
    provider.reset_d2h_transfer_count();
    solver.solve_expect_unsat(&unsat_cnf).expect("solve UNSAT");
    let unsat_transfers = provider.host_transfer_stats();
    assert_eq!(unsat_transfers.dtoh_calls, 0);
    assert_eq!(unsat_transfers.dtoh_bytes, 0);
    assert_eq!(provider.d2h_transfer_count(), 0);
}
