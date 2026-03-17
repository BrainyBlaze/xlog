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

/// Two consecutive solves on the same workspace both return UNSAT and reuse
/// the same device buffers (no reallocation).
#[test]
fn test_workspace_reuse_two_solves() {
    let Some(provider) = try_provider() else {
        return;
    };

    // Trivially UNSAT: (x0) AND (NOT x0)
    let instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    let config = GpuCdclConfig::default();
    let solver = GpuCdclSolver::new(provider.clone(), config);

    // Create workspace with capacity matching the CNF.
    let mut ws = solver
        .new_workspace(cnf.var_cap, cnf.clause_cap)
        .expect("new_workspace");

    // Record the device pointer of ws.assign before the first solve.
    let assign_ptr_before = ws.assign_device_ptr();

    // Allocate branch_limit = 1 as a GPU scalar.
    let branch_limit = solver_alloc_u32(&provider, 1);

    // First solve: should succeed with UNSAT.
    solver
        .solve_expect_unsat_with_branch_limit_ws(&mut ws, &cnf, &branch_limit)
        .expect("first solve should return UNSAT");

    // Verify workspace buffers were reused (same device pointer).
    let assign_ptr_after_first = ws.assign_device_ptr();
    assert_eq!(
        assign_ptr_before, assign_ptr_after_first,
        "workspace assign buffer was reallocated after first solve"
    );

    // Second solve: should also succeed with UNSAT (kernel reinitializes state).
    solver
        .solve_expect_unsat_with_branch_limit_ws(&mut ws, &cnf, &branch_limit)
        .expect("second solve should return UNSAT");

    // Verify workspace buffers still reused after second solve.
    let assign_ptr_after_second = ws.assign_device_ptr();
    assert_eq!(
        assign_ptr_before, assign_ptr_after_second,
        "workspace assign buffer was reallocated after second solve"
    );
}

/// A workspace with tiny capacity should produce a clean error when given
/// a CNF that exceeds its var_cap.
#[test]
fn test_workspace_capacity_overflow() {
    let Some(provider) = try_provider() else {
        return;
    };

    let config = GpuCdclConfig::default();
    let solver = GpuCdclSolver::new(provider.clone(), config);

    // Create workspace with tiny capacity: var_cap=1, clause_cap=1.
    let mut ws = solver.new_workspace(1, 1).expect("new_workspace");

    // Build a CNF with var_cap=10, which exceeds workspace var_cap of 1.
    // We need at least 1 clause so the CNF is valid.
    let instance = SolveInstance::new(10, vec![Clause::new(vec![Literal::positive(0)])]);
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    let branch_limit = solver_alloc_u32(&provider, 1);

    let result = solver.solve_expect_unsat_with_branch_limit_ws(&mut ws, &cnf, &branch_limit);

    assert!(
        result.is_err(),
        "should fail when CNF exceeds workspace capacity"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("exceeds workspace"),
        "error should mention 'exceeds workspace', got: {}",
        err_msg
    );
}

/// Solve with explicit decision ranges (base limit only, no extra range)
/// returns UNSAT on a trivially unsatisfiable CNF.
#[test]
fn test_workspace_decision_ranges_ws() {
    let Some(provider) = try_provider() else {
        return;
    };

    // Trivially UNSAT: (x0) AND (NOT x0)
    let instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    let config = GpuCdclConfig::default();
    let solver = GpuCdclSolver::new(provider.clone(), config);

    let mut ws = solver
        .new_workspace(cnf.var_cap, cnf.clause_cap)
        .expect("new_workspace");

    // decision_base_limit = var_cap (all vars allowed as decisions)
    let decision_base_limit = solver_alloc_u32(&provider, cnf.var_cap as u32);
    // No extra decision range.
    let decision_extra_base = solver_alloc_u32(&provider, 0);
    let decision_extra_count = solver_alloc_u32(&provider, 0);

    solver
        .solve_expect_unsat_with_decision_ranges_ws(
            &mut ws,
            &cnf,
            &decision_base_limit,
            &decision_extra_base,
            &decision_extra_count,
        )
        .expect("decision_ranges_ws should return UNSAT");
}

/// When `compile_needed == 0`, the gated workspace variant early-returns
/// without performing any CDCL work. The workspace buffers remain untouched.
#[test]
fn test_workspace_gated_ws_compile_not_needed() {
    let Some(provider) = try_provider() else {
        return;
    };

    // Trivially UNSAT: (x0) AND (NOT x0)
    let instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf upload");

    let config = GpuCdclConfig::default();
    let solver = GpuCdclSolver::new(provider.clone(), config);

    let mut ws = solver
        .new_workspace(cnf.var_cap, cnf.clause_cap)
        .expect("new_workspace");

    // compile_needed = 0 → kernel early-returns at sat.cu:1137
    let compile_needed = solver_alloc_u32(&provider, 0);
    let branch_limit = solver_alloc_u32(&provider, 1);

    solver
        .solve_expect_unsat_with_branch_limit_gated_ws(
            &mut ws,
            &cnf,
            &compile_needed,
            &branch_limit,
        )
        .expect("gated_ws with compile_needed=0 should succeed (early-return)");
}

/// Helper: upload a u32 scalar to the GPU.
fn solver_alloc_u32(
    provider: &Arc<CudaKernelProvider>,
    value: u32,
) -> xlog_cuda::memory::TrackedCudaSlice<u32> {
    let memory = provider.memory();
    let mut slot = memory.alloc::<u32>(1).expect("alloc u32 scalar");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[value], &mut slot)
        .expect("upload u32 scalar");
    slot
}
