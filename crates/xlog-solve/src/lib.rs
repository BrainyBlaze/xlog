//! SAT solver services for XLOG.
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

mod gpu_cdcl;
mod gpu_cnf;
mod instance;
mod proof;
mod solver;

pub use gpu_cdcl::{GpuCdclConfig, GpuCdclRawOutput, GpuCdclSolver};
pub use gpu_cnf::GpuCnf;
pub use instance::{Clause, Literal, Objective, SolveInstance};
pub use proof::{compute_checksum, SolveProof, SolveResult, SolveStats, SolveStatus};
pub use solver::{Solver, SolverConfig, SolverState};
