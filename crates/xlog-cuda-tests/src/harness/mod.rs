//! Test harness infrastructure for CUDA certification tests.

pub(crate) mod diagnostics;
pub mod generators;
pub(crate) mod provider;
pub mod validators;
pub mod xgcf;

pub use diagnostics::{
    CategoryResult, CertificationResults, FailureDiagnostic, TestResult, TestStatus,
};
pub use provider::{enforce_cuda_required, TestContext};
