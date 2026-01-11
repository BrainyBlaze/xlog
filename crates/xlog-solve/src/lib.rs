//! GPU-native SAT/MaxSAT solver using Continuous Local Search (CLS).
//!
//! This crate implements a FastFourierSAT-style CLS solver that treats SAT as
//! continuous optimization. Variables are relaxed to [0,1] values, and gradient
//! descent with momentum finds satisfying assignments.
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

mod instance;
mod solver;
mod proof;

pub use instance::{SolveInstance, Clause, Literal, Objective};
pub use solver::{Solver, SolverConfig, SolverState};
pub use proof::{SolveProof, SolveResult, SolveStatus, SolveStats, compute_checksum};
