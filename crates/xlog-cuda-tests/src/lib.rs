//! xlog-cuda-tests: Comprehensive CUDA kernel certification suite
//!
//! # Usage
//! ```bash
//! # Full certification (seconds to minutes; GPU-dependent)
//! cargo test -p xlog-cuda-tests --test certification_suite --release
//!
//! # Quick smoke test (sub-second to seconds; GPU-dependent)
//! cargo test -p xlog-cuda-tests --test quick_smoke --release
//!
//! # Single category
//! cargo test -p xlog-cuda-tests --test category_isolation c03 --release
//! ```

pub(crate) mod categories;
pub(crate) mod harness;
pub(crate) mod properties;

pub(crate) use harness::generators::{AlignmentGen, Distribution, NumericEdges, SizeGen};
pub(crate) use harness::validators::{compare, reference};
pub(crate) use harness::{
    CategoryResult, CertificationResults, FailureDiagnostic, TestContext, TestResult, TestStatus,
};
