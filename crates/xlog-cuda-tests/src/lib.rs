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

pub mod categories;
pub mod harness;
pub mod properties;

pub use harness::generators::{AlignmentGen, Distribution, NumericEdges, SizeGen};
pub use harness::validators::{compare, reference};
pub use harness::{
    CategoryResult, CertificationResults, FailureDiagnostic, TestContext, TestResult, TestStatus,
};
