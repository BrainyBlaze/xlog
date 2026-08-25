//! Error types for XLOG

use thiserror::Error;

/// Primary error type for XLOG operations
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum XlogError {
    /// Parse error from the Datalog frontend.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Stratification failed due to a cycle through negation.
    #[error("Stratification failed: cycle through negation involving {0:?}")]
    StratificationCycle(Vec<String>),

    /// A variable is not bound in any positive body literal (domain safety violation).
    #[error("Domain safety: variable {0} not bound in positive literal")]
    UnsafeVariable(String),

    /// GPU memory budget exceeded.
    #[error("Resource exhausted: {context}, estimated {estimated_bytes} bytes, budget {budget_bytes} bytes")]
    ResourceExhausted {
        /// Description of the operation that exceeded the budget.
        context: String,
        /// Estimated memory required in bytes.
        estimated_bytes: u64,
        /// Available memory budget in bytes.
        budget_bytes: u64,
    },

    /// A non-memory cardinality, dimension, or fixed-capacity limit was exceeded.
    #[error("Capacity exceeded: {context}, required {required} {unit}, limit {limit} {unit}")]
    CapacityExceeded {
        /// Operation or data structure whose capacity was exceeded.
        context: String,
        /// Minimum observed or requested capacity.
        required: u64,
        /// Configured or representable capacity.
        limit: u64,
        /// Human-readable unit shared by `required` and `limit`.
        unit: String,
    },

    /// A bounded iterative computation failed to reach its required fixpoint.
    #[error(
        "Convergence failed: {context}, {unconverged_worlds} of {total_worlds} worlds did not converge within {iteration_limit} iterations"
    )]
    ConvergenceFailure {
        /// Iterative computation that did not converge.
        context: String,
        /// Number of worlds that remained unconverged.
        unconverged_worlds: u64,
        /// Total number of evaluated worlds.
        total_worlds: u64,
        /// Maximum iterations executed per world.
        iteration_limit: u64,
    },

    /// The D4 **compile** phase declined a CNF too large to compile safely
    /// (the knowledge-compilation emit buffers are fixed-capacity; a larger
    /// instance would overrun them and fail with a context-poisoning CUDA
    /// launch error). A typed, catchable decline distinct from the verify-phase
    /// signal — "too big to compile", not "verify gave up". The caller can skip
    /// the query or fall back to an approximate engine.
    #[error("D4 compile declined: {context}: {detail}")]
    CompileCapacityExceeded {
        /// Description of the compile operation that was declined.
        context: String,
        /// Which capacity tripped and its measured/configured values.
        detail: String,
    },

    /// The GPU CDCL equivalence **verifier** declined rather than risk a
    /// CUDA launch failure that poisons the primary context: a per-verify
    /// conflict budget ran out before the search reached a definite answer
    /// (INDETERMINATE — declined fail-closed, never trusted as a proof).
    /// Distinct from [`CompileCapacityExceeded`] so the two phases stay
    /// diagnosably separate. The caller can skip the query or fall back to an
    /// approximate engine.
    #[error("D4 equivalence verify declined: {context}: {detail}")]
    VerifyBudgetExceeded {
        /// Description of the verify operation that was declined.
        context: String,
        /// Which budget tripped and its measured/configured values.
        detail: String,
    },

    /// GPU kernel launch or execution error.
    #[error("Kernel error: {0}")]
    Kernel(String),

    /// Type checking or inference error.
    #[error("Type error: {0}")]
    Type(String),

    /// Compilation pipeline error.
    #[error("Compilation error: {0}")]
    Compilation(String),

    /// Epistemic construct is known to the frontend but unsupported in this context.
    #[error("Unsupported epistemic construct: {construct} ({context})")]
    UnsupportedEpistemicConstruct {
        /// Construct that was rejected.
        construct: String,
        /// Context where the construct was rejected.
        context: String,
    },

    /// A compiler-generated integrity-constraint relation contains witness rows.
    #[error(
        "Constraint {constraint_index} violated: {relation_name} produced {witness_rows} witness row(s)"
    )]
    ConstraintViolation {
        /// Source-order constraint index encoded by the compiler-generated relation.
        constraint_index: usize,
        /// Compiler-generated relation that contains the violation witnesses.
        relation_name: String,
        /// Number of materialized violation witnesses.
        witness_rows: usize,
    },

    /// Runtime execution error.
    #[error("Execution error: {0}")]
    Execution(String),
}

impl XlogError {
    /// Create a Kernel error with structured context: "op: detail: source".
    pub fn kernel_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Kernel(format!("{op}: {detail}: {source}"))
    }

    /// Create an Execution error with structured context.
    pub fn execution_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Execution(format!("{op}: {detail}: {source}"))
    }

    /// Create a Compilation error with structured context.
    pub fn compilation_ctx(op: &str, detail: &str, source: &impl std::fmt::Display) -> Self {
        XlogError::Compilation(format!("{op}: {detail}: {source}"))
    }
}

/// Result alias using XlogError
pub type Result<T> = std::result::Result<T, XlogError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_display() {
        let err = XlogError::Parse("unexpected token".to_string());
        assert_eq!(err.to_string(), "Parse error: unexpected token");
    }

    #[test]
    fn test_stratification_cycle_display() {
        let err = XlogError::StratificationCycle(vec!["foo".to_string(), "bar".to_string()]);
        assert!(err.to_string().contains("foo"));
        assert!(err.to_string().contains("bar"));
    }

    #[test]
    fn test_resource_exhausted_display() {
        let err = XlogError::ResourceExhausted {
            context: "join operation".to_string(),
            estimated_bytes: 1024,
            budget_bytes: 512,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_capacity_exceeded_display_preserves_non_byte_units() {
        let err = XlogError::CapacityExceeded {
            context: "resident sparse rows".to_string(),
            required: 65,
            limit: 64,
            unit: "rows per world".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Capacity exceeded: resident sparse rows, required 65 rows per world, limit 64 rows per world"
        );
    }

    #[test]
    fn test_convergence_failure_display_reports_affected_worlds_and_limit() {
        let err = XlogError::ConvergenceFailure {
            context: "resident Monte Carlo fixpoint".to_string(),
            unconverged_worlds: 3,
            total_worlds: 128,
            iteration_limit: 7,
        };
        assert_eq!(
            err.to_string(),
            "Convergence failed: resident Monte Carlo fixpoint, 3 of 128 worlds did not converge within 7 iterations"
        );
    }

    #[test]
    fn test_constraint_violation_display() {
        let err = XlogError::ConstraintViolation {
            constraint_index: 3,
            relation_name: "__xlog_constraint_3".to_string(),
            witness_rows: 2,
        };
        assert_eq!(
            err.to_string(),
            "Constraint 3 violated: __xlog_constraint_3 produced 2 witness row(s)"
        );
    }

    #[test]
    fn test_kernel_ctx() {
        let err = XlogError::kernel_ctx("download_column", "dtoh copy failed", &"device error 42");
        assert_eq!(
            err.to_string(),
            "Kernel error: download_column: dtoh copy failed: device error 42"
        );
    }

    #[test]
    fn test_execution_ctx() {
        let err = XlogError::execution_ctx("execute_node", "filter failed", &"type mismatch");
        assert_eq!(
            err.to_string(),
            "Execution error: execute_node: filter failed: type mismatch"
        );
    }

    #[test]
    fn test_compilation_ctx() {
        let err = XlogError::compilation_ctx("compile_d4", "frontier overflow", &"limit 1024");
        assert_eq!(
            err.to_string(),
            "Compilation error: compile_d4: frontier overflow: limit 1024"
        );
    }
}
