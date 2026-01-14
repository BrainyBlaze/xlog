//! Test harness infrastructure for CUDA certification tests.

pub mod provider;
pub mod generators;
pub mod validators;
pub mod diagnostics;
pub mod xgcf;

pub use provider::TestContext;
pub use diagnostics::{FailureDiagnostic, CertificationResults, CategoryResult, TestResult, TestStatus};
