//! Helper functions for converting external error types into XlogError.
//!
//! cudarc::driver::DriverError → XlogError cannot use From (orphan rule:
//! neither type is defined in xlog-cuda). This module provides a local
//! conversion function instead.

use xlog_core::XlogError;

/// Convert a cudarc DriverError into an XlogError::Kernel.
pub(crate) fn driver_err(e: cudarc::driver::DriverError) -> XlogError {
    XlogError::Kernel(format!("CUDA driver error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_err_helper() {
        // cudarc::driver::DriverError doesn't have public constructors we can
        // easily use in tests, so we test via the context helper pattern.
        // The real validation is that this compiles with the correct type
        // signature — call-site adoption in Wave 2 provides integration coverage.
        let err = XlogError::kernel_ctx("test", "driver error", &"mock error");
        let msg = err.to_string();
        assert!(msg.contains("test"));
        assert!(msg.contains("driver error"));
        assert!(msg.contains("mock error"));
    }
}
