//! First-slice migrated launch path through the v0.6 launch
//! recorder.
//!
//! Adds a single, narrow method
//! [`CudaKernelProvider::memset_recorded`] that performs an
//! async memset on a caller-supplied launch stream **and**
//! records the buffer use against the runtime via the
//! [`crate::launch::LaunchRecorder`]. This is intentionally the
//! simplest possible "real launch" path:
//!   * one buffer arg (a write),
//!   * one CUDA call (`cuMemsetD8Async`),
//!   * a launch_stream that can differ from the buffer's
//!     `alloc_stream`,
//!   * an explicit `commit` step that surfaces
//!     `ResourceError::StreamMisuse` from the runtime if the
//!     active resource cannot track cross-stream uses.
//!
//! No existing operator is modified by this slice. Existing
//! kernel launches (filter, compact, dedup, hash join, etc.)
//! continue to use raw `Vec<*mut c_void>` and remain unsafe by
//! themselves. They will be migrated in follow-up commits once
//! this minimal path is certified.

use cudarc::driver::sys;
use xlog_core::{Result, XlogError};

use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::TrackedCudaSlice;

impl super::CudaKernelProvider {
    /// Async memset of `value` into every byte of `dst` on
    /// `launch_stream`, then record the use against the
    /// runtime.
    ///
    /// Requires the provider's `GpuMemoryManager` to be built
    /// via [`crate::GpuMemoryManager::with_runtime`] (so
    /// `dst.runtime_block()` is `Some` and the runtime is
    /// reachable). On a legacy/no-runtime manager, returns
    /// [`XlogError::Kernel`].
    ///
    /// # Errors
    ///   * `XlogError::Kernel("memset_recorded requires
    ///     runtime-backed manager")` if the manager has no
    ///     runtime attached.
    ///   * `XlogError::Kernel` from
    ///     `cuMemsetD8Async`/stream-resolution failure.
    ///   * `XlogError::Kernel` wrapping any
    ///     `ResourceError::StreamMisuse` from the recorder's
    ///     commit (notably when the active resource is
    ///     `DirectCudaResource` — the trait default that
    ///     intentionally rejects `record_block_use`).
    pub fn memset_recorded(
        &self,
        dst: &mut TrackedCudaSlice<u8>,
        value: u8,
        launch_stream: StreamId,
    ) -> Result<()> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "memset_recorded requires a runtime-backed GpuMemoryManager \
                 (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let pool = runtime.stream_pool();
        let cu_stream = pool.resolve(launch_stream).ok_or_else(|| {
            XlogError::Kernel(format!(
                "memset_recorded: launch_stream StreamId({}) does not resolve",
                launch_stream.0
            ))
        })?;

        // Capture identity bits before borrowing dst into the
        // recorder. `cuMemsetD8Async` writes to `dst.device_ptr`
        // for `dst.len()` bytes (T = u8 so len == byte count).
        let dst_ptr = dst.device_ptr_value();
        let dst_len = dst.len();

        // SAFETY: dst_ptr is a live device pointer, the buffer
        // owns `dst_len` bytes (verified by the slice's len),
        // and cu_stream is a valid CUDA stream the runtime
        // owns. cuMemsetD8Async is genuinely
        // stream-asynchronous: it queues on the stream and
        // returns immediately.
        unsafe {
            let res = sys::cuMemsetD8Async(dst_ptr, value, dst_len, cu_stream.cu_stream());
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async failed: {:?}",
                    res
                )));
            }
        }

        // Record the use AFTER queuing the work on
        // launch_stream so the recorded event fires when the
        // memset completes. `commit` calls
        // runtime.record_block_use which surfaces any
        // StreamMisuse from the active resource — no silent
        // fallback.
        let mut rec = LaunchRecorder::new(launch_stream);
        rec.write(dst);
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "memset_recorded: launch recorder commit failed: {}",
                e
            ))
        })?;
        Ok(())
    }
}
