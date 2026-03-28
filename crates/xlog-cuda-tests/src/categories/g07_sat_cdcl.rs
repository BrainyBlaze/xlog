//! Category G07: SAT/CDCL GPU verifier tests
//!
//! Validates the GPU-resident CDCL solver on deterministic, small CNFs
//! covering SAT, UNSAT, propagation, and proof checking paths.

use crate::harness::{CategoryResult, TestContext, TestResult};
use cudarc::driver::DeviceSlice;
use std::sync::Arc;
use std::time::Instant;
use xlog_cuda::CudaKernelProvider;
use xlog_solve::{Clause, GpuCdclConfig, GpuCdclSolver, GpuCnf, Literal, SolveInstance};

/// Run all tests in this category.
pub(crate) fn run_all(ctx: &TestContext) -> CategoryResult {
    let start = Instant::now();
    let mut results = CategoryResult::new("g07_sat_cdcl");

    let provider = match CudaKernelProvider::new(ctx.device.clone(), ctx.memory.clone()) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            results.add_result(TestResult::error(
                "init_cdcl_provider",
                start.elapsed(),
                format!("Failed to create CDCL provider: {}", e),
            ));
            results.set_duration(start.elapsed());
            return results;
        }
    };

    results.add_result(test_cdcl_sat_unit(ctx, &provider));
    results.add_result(test_cdcl_sat_implication_chain(ctx, &provider));
    results.add_result(test_cdcl_unsat_contradictory_units(ctx, &provider));
    results.add_result(test_cdcl_unsat_xor(ctx, &provider));

    results.set_duration(start.elapsed());
    results
}

fn test_cdcl_sat_unit(ctx: &TestContext, provider: &Arc<CudaKernelProvider>) -> TestResult {
    let start = Instant::now();

    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let cnf = match GpuCnf::from_host(&instance, provider) {
        Ok(cnf) => cnf,
        Err(e) => {
            return TestResult::error(
                "test_cdcl_sat_unit",
                start.elapsed(),
                format!("GpuCnf upload failed: {}", e),
            );
        }
    };

    let solver = GpuCdclSolver::new(provider.clone(), GpuCdclConfig::default());
    let assignment = match solver.solve_expect_sat(&cnf) {
        Ok(a) => a,
        Err(e) => {
            return TestResult::error(
                "test_cdcl_sat_unit",
                start.elapsed(),
                format!("solve_expect_sat failed: {}", e),
            );
        }
    };

    if assignment.len() != (cnf.var_cap as usize + 1) {
        return TestResult::error(
            "test_cdcl_sat_unit",
            start.elapsed(),
            format!(
                "Unexpected assignment length: got {}, expected {}",
                assignment.len(),
                cnf.var_cap as usize + 1
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cdcl_sat_unit",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_cdcl_sat_unit", start.elapsed())
}

fn test_cdcl_sat_implication_chain(
    ctx: &TestContext,
    provider: &Arc<CudaKernelProvider>,
) -> TestResult {
    let start = Instant::now();

    // (x0) AND (~x0 OR x1) AND (~x1 OR x2) AND (~x2 OR x3)
    let instance = SolveInstance::new(
        4,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
            Clause::new(vec![Literal::negative(1), Literal::positive(2)]),
            Clause::new(vec![Literal::negative(2), Literal::positive(3)]),
        ],
    );

    let cnf = match GpuCnf::from_host(&instance, provider) {
        Ok(cnf) => cnf,
        Err(e) => {
            return TestResult::error(
                "test_cdcl_sat_implication_chain",
                start.elapsed(),
                format!("GpuCnf upload failed: {}", e),
            );
        }
    };

    let solver = GpuCdclSolver::new(provider.clone(), GpuCdclConfig::default());
    let assignment = match solver.solve_expect_sat(&cnf) {
        Ok(a) => a,
        Err(e) => {
            return TestResult::error(
                "test_cdcl_sat_implication_chain",
                start.elapsed(),
                format!("solve_expect_sat failed: {}", e),
            );
        }
    };

    if assignment.len() != (cnf.var_cap as usize + 1) {
        return TestResult::error(
            "test_cdcl_sat_implication_chain",
            start.elapsed(),
            format!(
                "Unexpected assignment length: got {}, expected {}",
                assignment.len(),
                cnf.var_cap as usize + 1
            ),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cdcl_sat_implication_chain",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_cdcl_sat_implication_chain", start.elapsed())
}

fn test_cdcl_unsat_contradictory_units(
    ctx: &TestContext,
    provider: &Arc<CudaKernelProvider>,
) -> TestResult {
    let start = Instant::now();

    // (x0) AND (~x0)
    let instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );

    let cnf = match GpuCnf::from_host(&instance, provider) {
        Ok(cnf) => cnf,
        Err(e) => {
            return TestResult::error(
                "test_cdcl_unsat_contradictory_units",
                start.elapsed(),
                format!("GpuCnf upload failed: {}", e),
            );
        }
    };

    let solver = GpuCdclSolver::new(provider.clone(), GpuCdclConfig::default());
    if let Err(e) = solver.solve_expect_unsat(&cnf) {
        return TestResult::error(
            "test_cdcl_unsat_contradictory_units",
            start.elapsed(),
            format!("solve_expect_unsat failed: {}", e),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cdcl_unsat_contradictory_units",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_cdcl_unsat_contradictory_units", start.elapsed())
}

fn test_cdcl_unsat_xor(ctx: &TestContext, provider: &Arc<CudaKernelProvider>) -> TestResult {
    let start = Instant::now();

    // XOR unsat: (x0 OR x1) AND (x0 OR ~x1) AND (~x0 OR x1) AND (~x0 OR ~x1)
    let instance = SolveInstance::new(
        2,
        vec![
            Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
            Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
            Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
            Clause::new(vec![Literal::negative(0), Literal::negative(1)]),
        ],
    );

    let cnf = match GpuCnf::from_host(&instance, provider) {
        Ok(cnf) => cnf,
        Err(e) => {
            return TestResult::error(
                "test_cdcl_unsat_xor",
                start.elapsed(),
                format!("GpuCnf upload failed: {}", e),
            );
        }
    };

    let solver = GpuCdclSolver::new(provider.clone(), GpuCdclConfig::default());
    if let Err(e) = solver.solve_expect_unsat(&cnf) {
        return TestResult::error(
            "test_cdcl_unsat_xor",
            start.elapsed(),
            format!("solve_expect_unsat failed: {}", e),
        );
    }

    if let Err(e) = ctx.sync_and_check() {
        return TestResult::error(
            "test_cdcl_unsat_xor",
            start.elapsed(),
            format!("Sync failed: {}", e),
        );
    }

    TestResult::passed("test_cdcl_unsat_xor", start.elapsed())
}
