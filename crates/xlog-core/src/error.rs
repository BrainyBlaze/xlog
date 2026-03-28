//! Error types for XLOG

use thiserror::Error;

/// Primary error type for XLOG operations
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum XlogError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Stratification failed: cycle through negation involving {0:?}")]
    StratificationCycle(Vec<String>),

    #[error("Domain safety: variable {0} not bound in positive literal")]
    UnsafeVariable(String),

    #[error("Resource exhausted: {context}, estimated {estimated_bytes} bytes, budget {budget_bytes} bytes")]
    ResourceExhausted {
        context: String,
        estimated_bytes: u64,
        budget_bytes: u64,
    },

    #[error("Kernel error: {0}")]
    Kernel(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

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
