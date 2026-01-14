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

pub mod harness;
pub mod categories;

pub use harness::{TestContext, FailureDiagnostic, CertificationResults, CategoryResult, TestResult, TestStatus};
pub use harness::generators::{SizeGen, Distribution, NumericEdges, AlignmentGen};
pub use harness::validators::{reference, compare};
