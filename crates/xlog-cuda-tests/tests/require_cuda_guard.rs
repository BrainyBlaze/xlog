//! Release-gate hardening: when `XLOG_REQUIRE_CUDA=1`, a missing or broken
//! CUDA device must fail the test run loudly instead of letting every test
//! skip silently and pass vacuously. `scripts/validate_release_gpu.sh`
//! exports this variable so the certification gate can never be satisfied by
//! a CPU-only machine.

use std::sync::Mutex;

use xlog_core::XlogError;
use xlog_cuda_tests::harness::{enforce_cuda_required, TestContext};

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn enforce_cuda_required_panics_when_env_set() {
    let _guard = ENV_LOCK.lock().unwrap();
    let old = std::env::var("XLOG_REQUIRE_CUDA").ok();
    std::env::set_var("XLOG_REQUIRE_CUDA", "1");
    let err = XlogError::Kernel("No CUDA devices available".to_string());
    let outcome = std::panic::catch_unwind(|| enforce_cuda_required("guard-test", &err));
    match old {
        Some(value) => std::env::set_var("XLOG_REQUIRE_CUDA", value),
        None => std::env::remove_var("XLOG_REQUIRE_CUDA"),
    }
    assert!(
        outcome.is_err(),
        "XLOG_REQUIRE_CUDA=1 must turn CUDA-init failures into loud panics"
    );
}

#[test]
fn enforce_cuda_required_is_noop_when_env_unset() {
    let _guard = ENV_LOCK.lock().unwrap();
    let old = std::env::var("XLOG_REQUIRE_CUDA").ok();
    std::env::remove_var("XLOG_REQUIRE_CUDA");
    let err = XlogError::Kernel("No CUDA devices available".to_string());
    let outcome = std::panic::catch_unwind(|| enforce_cuda_required("guard-test", &err));
    match old {
        Some(value) => std::env::set_var("XLOG_REQUIRE_CUDA", value),
        None => std::env::remove_var("XLOG_REQUIRE_CUDA"),
    }
    assert!(
        outcome.is_ok(),
        "without XLOG_REQUIRE_CUDA the skip contract is unchanged"
    );
}

#[test]
fn cuda_device_must_be_available_when_required() {
    // Always-run gate: on a machine without CUDA this test FAILS (instead of
    // skipping) whenever XLOG_REQUIRE_CUDA=1 — e.g. under the release script.
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
        TestContext::new()
            .expect("XLOG_REQUIRE_CUDA=1 but CUDA init failed: refusing the vacuous skip");
    }
}
