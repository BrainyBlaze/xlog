//! SAT solver services for XLOG.
#![warn(missing_docs)]
//!
//! This crate contains:
//! - a CPU Continuous Local Search (CLS) solver (heuristic, not complete), and
//! - a GPU-native CDCL solver (complete SAT/UNSAT) used as the production verifier.
//!
//! # Architecture
//!
//! - `instance`: SAT instance representation (CNF clauses, literals)
//! - `solver`: CLS solver with configurable parameters
//! - `proof`: Proof artifacts and verification
//!
//! # Usage
//!
//! ```ignore
//! use xlog_solve::{SolveInstance, Clause, Literal, Solver};
//!
//! // Create a SAT instance: (x0 OR NOT x1) AND (x1 OR x2)
//! let instance = SolveInstance::new(3, vec![
//!     Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
//!     Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
//! ]);
//!
//! // Solve
//! let solver = Solver::new_cpu();
//! let result = solver.solve(instance);
//!
//! match result.status {
//!     SolveStatus::Sat => println!("Satisfiable!"),
//!     SolveStatus::Unsat => println!("Unsatisfiable"),
//!     SolveStatus::Unknown => println!("Could not determine"),
//!     SolveStatus::Optimal(_) => println!("Found optimal"),
//! }
//! ```

#[allow(missing_docs)] // TODO(v0.6): document GPU CDCL solver methods
mod gpu_cdcl;
#[allow(missing_docs)] // TODO(v0.6): document or make pub(crate)
mod gpu_cnf;
mod instance;
mod production;
mod proof;
mod service;
mod solver;

pub use gpu_cdcl::{GpuCdclConfig, GpuCdclRawOutput, GpuCdclSolver, GpuCdclWorkspace};
pub use gpu_cnf::GpuCnf;
pub use instance::{Clause, Literal, Objective, SolveInstance};
pub use production::{
    production_capabilities, GpuSolverProductionAdapter, GpuSolverProductionBatchExecutionEvidence,
    GpuSolverProductionCapabilities, GpuSolverProductionCapabilityStatus,
    GpuSolverProductionExpectation, GpuSolverProductionLearnedClauseArenaReport,
    GpuSolverProductionLearnedClauseReuseReport, GpuSolverProductionLifecycleReport,
    GpuSolverProductionLifecycleStep, GpuSolverProductionMaxSatCandidate,
    GpuSolverProductionMaxSatReport, GpuSolverProductionMaxSatScheduleJob,
    GpuSolverProductionMaxSatScheduleReport, GpuSolverProductionMaxSatSearchCandidate,
    GpuSolverProductionMaxSatSearchStatus, GpuSolverProductionPortfolioJob,
    GpuSolverProductionPortfolioReport, GpuSolverProductionTrace,
    GpuSolverProductionWeightedMaxSatSelection,
};
pub use proof::{compute_checksum, SolveProof, SolveResult, SolveStats, SolveStatus};
pub use service::{
    LearnedClauseTransfer, SolverPortfolioStatus, SolverService, SolverServiceBudget,
    SolverServiceResult, SolverServiceStatus, SolverServiceTrace,
};
pub use solver::{Solver, SolverConfig, SolverState};
