//! Recorded asynchronous memset paths through the launch recorder.
//!
//! Provides two narrow helpers. Public
//! [`CudaKernelProvider::memset_recorded`] and the public column helper
//! [`CudaKernelProvider::memset_column_recorded`] use the consuming recorder
//! enqueue path, so unsupported tracking is rejected during preparation before
//! the asynchronous memset.
//! Both helpers record the destination write on a caller-supplied launch stream,
//! which may differ from the allocation stream.

use cudarc::driver::sys;
use xlog_core::{Result, XlogError};

use crate::device_runtime::StreamId;
use crate::launch::{LaunchEnqueueError, LaunchRecorder};
use crate::memory::{CudaColumn, TrackedCudaSlice};

impl super::CudaKernelProvider {
    /// Async memset of `value` into every byte of `dst` on
    /// `launch_stream`, then record the use against the
    /// runtime.
    ///
    /// Requires the provider's `GpuMemoryManager` to carry the runtime attached
    /// by [`crate::CudaProviderBuilder`] (so `dst.runtime_block()` is `Some`).
    /// On a legacy/no-runtime manager, returns [`XlogError::Kernel`].
    ///
    /// # Errors
    ///   * `XlogError::Kernel("memset_recorded requires
    ///     runtime-backed manager")` if the manager has no
    ///     runtime attached.
    ///   * `XlogError::Kernel` from preparation, `cuMemsetD8Async`, cleanup,
    ///     or commit failure.
    pub fn memset_recorded(
        &self,
        dst: &mut TrackedCudaSlice<u8>,
        value: u8,
        launch_stream: StreamId,
    ) -> Result<()> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "memset_recorded requires a runtime-backed GpuMemoryManager \
                 (constructed by CudaProviderBuilder)"
                    .to_string(),
            )
        })?;
        let dst_ptr = dst.device_ptr_value();
        let dst_len = dst.len();

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.write(dst);

        // SAFETY: `dst_ptr` remains live for this synchronous enqueue call,
        // spans exactly `dst_len` bytes because the element type is `u8`, and
        // was registered above as a write. The closure enqueues exactly one
        // operation, immediately, and only on the supplied stream.
        let enqueued = unsafe {
            rec.enqueue(runtime, |stream| {
                let result =
                    sys::cuMemsetD8Async(dst_ptr, value, dst_len, stream.stream().cu_stream());
                if result == sys::cudaError_enum::CUDA_SUCCESS {
                    Ok(())
                } else {
                    Err(XlogError::Kernel(format!(
                        "cuMemsetD8Async failed: {result:?}"
                    )))
                }
            })
        }
        .map_err(|error| match error {
            LaunchEnqueueError::Preparation(preparation) => XlogError::Kernel(format!(
                "memset_recorded: launch recorder preparation failed: {preparation}"
            )),
            LaunchEnqueueError::PreparationAndCleanup {
                preparation,
                cleanup,
            } => XlogError::Kernel(format!(
                "memset_recorded: launch recorder preparation failed: {preparation}; \
                 launch reservation cleanup also failed: {cleanup}"
            )),
            LaunchEnqueueError::Operation(operation) => operation,
            LaunchEnqueueError::OperationAndCleanup { operation, cleanup } => XlogError::Kernel(
                format!("{operation}; launch reservation cleanup also failed: {cleanup}"),
            ),
        })?;

        enqueued.commit().map_err(|error| {
            XlogError::Kernel(format!(
                "memset_recorded: launch recorder commit failed: {error}"
            ))
        })
    }

    /// Column-level variant of [`Self::memset_recorded`] —
    /// exercises the `LaunchRecorder::write_column` path. Used
    /// by tests that prove `CudaColumn::Owned` records its
    /// runtime block automatically; strict mode rejects
    /// `CudaColumn::Dlpack` / `CudaColumn::ArrowDevice` at
    /// preflight (no CUDA work queued).
    pub fn memset_column_recorded(
        &self,
        dst: &mut CudaColumn,
        value: u8,
        launch_stream: StreamId,
    ) -> Result<()> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "memset_column_recorded requires a runtime-backed GpuMemoryManager".to_string(),
            )
        })?;
        let dst_ptr = *dst.device_ptr();
        let dst_len = <CudaColumn as cudarc::driver::DeviceSlice<u8>>::len(dst);

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.write_column(dst);
        // SAFETY: `dst_ptr` remains live for this synchronous enqueue call and
        // spans exactly `dst_len` bytes. The destination was registered above
        // as a write, and the closure uses only the supplied stream.
        let enqueued = unsafe {
            rec.enqueue(runtime, |stream| {
                let result =
                    sys::cuMemsetD8Async(dst_ptr, value, dst_len, stream.stream().cu_stream());
                if result == sys::cudaError_enum::CUDA_SUCCESS {
                    Ok(())
                } else {
                    Err(XlogError::Kernel(format!(
                        "cuMemsetD8Async (column) failed: {result:?}"
                    )))
                }
            })
        }
        .map_err(|error| match error {
            LaunchEnqueueError::Preparation(preparation) => XlogError::Kernel(format!(
                "memset_column_recorded: launch recorder preparation failed: {preparation}"
            )),
            LaunchEnqueueError::PreparationAndCleanup {
                preparation,
                cleanup,
            } => XlogError::Kernel(format!(
                "memset_column_recorded: launch recorder preparation failed: {preparation}; \
                 launch reservation cleanup also failed: {cleanup}"
            )),
            LaunchEnqueueError::Operation(operation) => operation,
            LaunchEnqueueError::OperationAndCleanup { operation, cleanup } => XlogError::Kernel(
                format!("{operation}; launch reservation cleanup also failed: {cleanup}"),
            ),
        })?;

        enqueued.commit().map_err(|e| {
            XlogError::Kernel(format!(
                "memset_column_recorded: launch recorder commit failed: {}",
                e
            ))
        })
    }
}
