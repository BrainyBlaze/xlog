//! Proof artifacts and solve results (to be implemented in Task 10+).

/// Proof artifact from solving.
#[derive(Debug, Clone)]
pub struct SolveProof {
    _placeholder: (),
}

/// Result from a solve operation.
#[derive(Debug, Clone)]
pub struct SolveResult {
    _placeholder: (),
}

/// Status of a solve operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveStatus {
    /// Instance is satisfiable.
    Sat,
    /// Instance is unsatisfiable.
    Unsat,
    /// Could not determine satisfiability.
    Unknown,
    /// Found optimal solution with given value.
    Optimal(f64),
}

/// Statistics from solving.
#[derive(Debug, Clone, Default)]
pub struct SolveStats {
    _placeholder: (),
}
