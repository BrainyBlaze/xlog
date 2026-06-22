//! Release-gate hardening: integration tests skip individually when CUDA is
//! unavailable (`CudaDevice::new(0).ok()?` fixture patterns), which would let
//! a CPU-only machine pass the whole package vacuously. This always-running
//! guard fails the package loudly when `XLOG_REQUIRE_CUDA=1` (exported by
//! `scripts/validate_release_gpu.sh`) and no CUDA device can be initialized.

#[test]
fn cuda_device_must_be_available_when_required() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
        xlog_cuda::CudaDevice::new(0)
            .expect("XLOG_REQUIRE_CUDA=1 but CUDA init failed: refusing the vacuous skip");
    }
}
