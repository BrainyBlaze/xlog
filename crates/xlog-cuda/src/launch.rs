//! Launch / use recorder for runtime-backed buffers.
//!
//! Closes the production-side of the cross-stream lifetime gap
//! identified by A4 and proven on
//! `tests/test_runtime_cross_stream_use_after_free.rs`. Code that
//! enqueues kernels or copies on a `launch_stream` other than the
//! buffer's `alloc_stream` MUST tell the runtime about the use so
//! the runtime can wait for the launch to complete before its
//! `cuMemFreeAsync` runs. Without this, the CUDA mempool is free
//! to reuse the address while the cross-stream work is still in
//! flight.
//!
//! # Usage
//!
//! ```ignore
//! use xlog_cuda::launch::LaunchRecorder;
//!
//! let mut rec = LaunchRecorder::new(launch_stream);
//! rec.read(&input_a);
//! rec.read(&input_b);
//! rec.write(&mut output);
//! // ... build kernel params, launch on launch_stream ...
//! unsafe { func.launch(cfg, &mut params)?; }
//! // Now record the uses against the runtime. This MUST be
//! // called AFTER the launch is queued so the runtime's
//! // recorded events fire when the launch completes.
//! rec.commit(&runtime)?;
//! ```
//!
//! # Failure semantics
//!
//! `commit` calls
//! [`XlogDeviceRuntime::record_block_use`](crate::device_runtime::XlogDeviceRuntime::record_block_use)
//! for every recorded buffer. If the active resource does not
//! support cross-stream tracking (e.g. the runtime was built
//! around `DirectCudaResource`, the trait default), the call
//! returns
//! [`ResourceError::StreamMisuse`](crate::device_runtime::ResourceError::StreamMisuse)
//! and the recorder surfaces it to the caller. **No silent
//! fallback.** Callers must either:
//!   * Route allocations through
//!     [`GpuMemoryManager::with_runtime`](crate::GpuMemoryManager::with_runtime)
//!     against an `AsyncCudaResource`-backed runtime, or
//!   * Take explicit responsibility for cross-stream
//!     synchronization (and use a launch path that does NOT go
//!     through the recorder).
//!
//! # Legacy cudarc-backed buffers
//!
//! Buffers allocated via the legacy `GpuMemoryManager::new` path
//! return `None` from
//! [`TrackedCudaSlice::runtime_block`](crate::memory::TrackedCudaSlice::runtime_block).
//! The recorder skips those — they are not tracked by the v0.6
//! runtime and the recorder cannot attach an event. This is by
//! design for the migration window; callers that mix legacy
//! and runtime-backed buffers in the same launch should be
//! aware that legacy buffers carry no cross-stream lifetime
//! guarantee.

use crate::device_runtime::{DeviceBlock, ResourceResult, StreamId, XlogDeviceRuntime};

/// Records buffer uses for a single launch / copy on
/// `launch_stream`. Drop without calling `commit` is a
/// programmer error; the recorder logs (via `eprintln!` only in
/// debug builds) but does not panic.
pub struct LaunchRecorder<'b> {
    launch_stream: StreamId,
    /// Recorded uses, by reference. Held as `&DeviceBlock` rather
    /// than copied so the caller's borrow of the
    /// [`crate::memory::TrackedCudaSlice`] enforces that the
    /// buffer is alive at recorder-construction time.
    uses: Vec<&'b DeviceBlock>,
    committed: bool,
}

impl<'b> LaunchRecorder<'b> {
    /// Construct a recorder for a launch on `launch_stream`. The
    /// caller is responsible for passing the same `launch_stream`
    /// id used for the actual kernel/copy invocation.
    pub fn new(launch_stream: StreamId) -> Self {
        Self {
            launch_stream,
            uses: Vec::new(),
            committed: false,
        }
    }

    /// Record a buffer that the launch will READ from on
    /// `launch_stream`. The buffer's runtime block is borrowed
    /// for the recorder's lifetime; if the buffer is legacy
    /// cudarc-backed (`runtime_block() == None`) the call is a
    /// no-op (logged in debug builds).
    pub fn read<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &'b crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.maybe_push("read", slice.runtime_block(), slice.device_ptr_value())
    }

    /// Record a buffer that the launch will WRITE to. Same shape
    /// as [`read`](Self::read); the recorder does not currently
    /// distinguish between read and write at the runtime level
    /// (both attach an event), but the access mode is captured
    /// for future telemetry / dependency-graph use.
    pub fn write<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &'b crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.maybe_push("write", slice.runtime_block(), slice.device_ptr_value())
    }

    /// Record a buffer that the launch will both READ and WRITE.
    pub fn read_write<T: cudarc::driver::DeviceRepr>(
        &mut self,
        slice: &'b mut crate::memory::TrackedCudaSlice<T>,
    ) -> &mut Self {
        self.maybe_push(
            "read_write",
            slice.runtime_block(),
            slice.device_ptr_value(),
        )
    }

    fn maybe_push(
        &mut self,
        _mode: &'static str,
        block: Option<&'b DeviceBlock>,
        _raw_ptr: cudarc::driver::sys::CUdeviceptr,
    ) -> &mut Self {
        if let Some(b) = block {
            self.uses.push(b);
        }
        // Legacy slices (block == None) are intentionally
        // skipped: they have no runtime-side identity to
        // attach uses to. See module-level doc.
        self
    }

    /// Number of recorded runtime-backed buffer uses. Test/
    /// diagnostic accessor.
    pub fn recorded_count(&self) -> usize {
        self.uses.len()
    }

    /// Commit the recorded uses to the runtime. MUST be called
    /// AFTER the launch / copy has been enqueued on
    /// `launch_stream` — the runtime records its event on
    /// `launch_stream` at this point, and that event will fire
    /// when the queued work completes.
    ///
    /// Returns the first error from
    /// [`XlogDeviceRuntime::record_block_use`] (notably
    /// `StreamMisuse` when the active resource does not support
    /// cross-stream tracking). On error, any earlier successful
    /// records remain attached to their blocks — the runtime
    /// already recorded those events on `launch_stream`, and the
    /// blocks' subsequent deallocate will wait on them as
    /// designed.
    pub fn commit(mut self, runtime: &XlogDeviceRuntime) -> ResourceResult<()> {
        for block in &self.uses {
            runtime.record_block_use(block, self.launch_stream)?;
        }
        self.committed = true;
        Ok(())
    }
}

impl Drop for LaunchRecorder<'_> {
    fn drop(&mut self) {
        if !self.committed && !self.uses.is_empty() {
            // A recorder dropped without commit means the launch
            // was queued on launch_stream but the runtime was
            // never told about the uses. That is a
            // lifetime-safety bug class. We log loudly in debug
            // builds; we cannot panic here because the builder
            // may have intentionally bailed before commit (e.g.,
            // an error elsewhere prompted early return).
            #[cfg(debug_assertions)]
            eprintln!(
                "[xlog_cuda::launch] LaunchRecorder dropped without commit: \
                 {} uses on launch_stream={} were NOT recorded; \
                 cross-stream lifetime safety lost for this launch",
                self.uses.len(),
                self.launch_stream.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_runtime::{
        AsyncCudaResource, DeviceMemoryResource, ResourceError, StreamPool,
    };
    use crate::CudaDevice;
    use std::sync::Arc;
    use xlog_core::MemoryBudget;

    fn try_runtime() -> Option<(Arc<crate::CudaDevice>, Arc<XlogDeviceRuntime>, StreamId)> {
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
            async_resource,
        ));
        let launch_stream = pool.acquire().ok()?;
        Some((device, runtime, launch_stream))
    }

    #[test]
    fn empty_commit_is_ok() {
        let Some((_d, rt, ls)) = try_runtime() else {
            return;
        };
        let rec = LaunchRecorder::new(ls);
        rec.commit(&rt).expect("empty commit must be Ok");
    }

    #[test]
    fn read_write_runtime_backed_records_use() {
        let Some((device, runtime, launch_stream)) = try_runtime() else {
            return;
        };
        let manager = Arc::new(crate::GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let input = manager.alloc::<u8>(64).expect("alloc input");
        let mut output = manager.alloc::<u8>(64).expect("alloc output");

        // Sanity: both slices are runtime-backed.
        assert_eq!(input.runtime_block().unwrap().bytes, 64);
        assert_eq!(output.runtime_block().unwrap().bytes, 64);

        // Build the recorder in its own scope so its borrows on
        // input/output drop before we keep using them. Sequence:
        // read(input), read_write(output), commit, drop recorder.
        {
            let mut rec = LaunchRecorder::new(launch_stream);
            rec.read(&input);
            rec.read_write(&mut output);
            assert_eq!(rec.recorded_count(), 2);
            rec.commit(&runtime).expect("commit");
        }
    }

    #[test]
    fn legacy_slice_silently_skipped() {
        // No runtime attached — manager built via legacy `new`.
        let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
            return;
        };
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
            async_resource,
        ));
        let launch_stream = pool.acquire().expect("acquire");

        let manager = Arc::new(crate::GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
        ));
        // Legacy alloc path: no runtime block.
        let legacy = manager.alloc::<u8>(64).expect("legacy alloc");
        assert!(legacy.runtime_block().is_none());

        let mut rec = LaunchRecorder::new(launch_stream);
        rec.read(&legacy);
        assert_eq!(
            rec.recorded_count(),
            0,
            "legacy slice without runtime block must be skipped"
        );
        rec.commit(&runtime).expect("commit no-op");
    }

    #[test]
    fn commit_surfaces_stream_misuse_from_direct_resource() {
        // Build a runtime around DirectCudaResource (the trait
        // default behavior — record_block_use returns
        // StreamMisuse). LaunchRecorder::commit must surface
        // that error rather than masking it.
        use crate::device_runtime::DirectCudaResource;

        let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
            return;
        };
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let direct: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
            direct,
        ));

        // We need a runtime-backed TrackedCudaSlice (so the
        // recorder records something). The manager's
        // `with_runtime` path against this Direct-backed runtime
        // will call runtime.allocate which goes through
        // DirectCudaResource — that returns a DeviceBlock just
        // like AsyncCudaResource would. The difference shows up
        // only at record_block_use time.
        let manager = Arc::new(crate::GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let buf = manager.alloc::<u8>(64).expect("alloc");
        assert!(buf.runtime_block().is_some());

        let launch_stream = StreamId::DEFAULT;
        let mut rec = LaunchRecorder::new(launch_stream);
        rec.read(&buf);
        let err = rec.commit(&runtime);
        match err {
            Err(ResourceError::StreamMisuse(msg)) => {
                assert!(
                    msg.contains("unsupported"),
                    "expected 'unsupported' in StreamMisuse message, got {:?}",
                    msg
                );
            }
            other => panic!(
                "LaunchRecorder::commit must surface StreamMisuse from \
                 DirectCudaResource; got {:?}",
                other
            ),
        }
    }
}
