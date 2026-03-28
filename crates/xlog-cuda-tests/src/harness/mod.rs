//! Test harness infrastructure for CUDA certification tests.

pub(crate) mod diagnostics;
pub(crate) mod generators;
pub(crate) mod provider;
pub(crate) mod validators;
pub(crate) mod xgcf;

pub(crate) use diagnostics::{
    CategoryResult, CertificationResults, FailureDiagnostic, TestResult, TestStatus,
};
pub(crate) use provider::TestContext;
