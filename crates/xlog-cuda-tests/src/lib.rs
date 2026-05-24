//! xlog-cuda-tests: Comprehensive CUDA kernel certification suite
#![allow(
    clippy::collapsible_if,
    clippy::doc_lazy_continuation,
    clippy::items_after_test_module,
    clippy::manual_clamp,
    clippy::manual_contains,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::same_item_push,
    clippy::type_complexity,
    clippy::unnecessary_mut_passed,
    clippy::useless_conversion,
    clippy::useless_vec
)]
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

pub use harness::{
    CategoryResult, CertificationResults, FailureDiagnostic, TestContext, TestResult, TestStatus,
};
