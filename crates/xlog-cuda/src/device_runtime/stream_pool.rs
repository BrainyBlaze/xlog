//! [`StreamPool`] — owned non-blocking CUDA streams indexed by
//! [`StreamId`].
//!
//! The runtime hands out stable [`StreamId`]s to callers and resolves
//! them to live `cudarc::driver::CudaStream` handles internally. The
//! pool grows on demand: `acquire` returns an existing stream if one
//! is available, otherwise creates and stores a new non-blocking
//! stream on the device.
//!
//! Streams are never returned to a free-list — they stay alive for
//! the runtime's lifetime so [`StreamId`] handles remain valid for
//! correlated allocate/launch/deallocate sequences. The pool's
//! growth is bounded by `max_streams` (defaults to a small constant
//! tuned for the v0.6 baseline; raise via the runtime config when
//! the executor needs more concurrency).

use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::CudaStream;

use super::resource::StreamId;
use crate::CudaDevice;

/// Default maximum stream count. The executor's typical concurrency
/// is 1 deterministic stream + a small handful of join/scan helpers,
/// so 16 leaves substantial headroom without burning device-state on
/// idle streams.
pub const DEFAULT_MAX_STREAMS: usize = 16;

/// Pool of owned non-blocking CUDA streams.
pub struct StreamPool {
    device: Arc<CudaDevice>,
    max_streams: usize,
    /// Stream handles indexed by [`StreamId`]. The slot at index 0 is
    /// reserved for [`StreamId::DEFAULT`] and lazily-initialized to
    /// the device's default stream so synchronous codepaths keep
    /// working unchanged during the migration window.
    streams: Mutex<Vec<Arc<CudaStream>>>,
}

impl StreamPool {
    /// Construct a pool bound to `device`, capped at `max_streams`.
    pub fn new(device: Arc<CudaDevice>, max_streams: usize) -> Self {
        Self {
            device,
            max_streams: max_streams.max(1),
            streams: Mutex::new(Vec::new()),
        }
    }

    /// Construct with the default cap [`DEFAULT_MAX_STREAMS`].
    pub fn with_defaults(device: Arc<CudaDevice>) -> Self {
        Self::new(device, DEFAULT_MAX_STREAMS)
    }

    /// Acquire a stream id. Currently always returns
    /// [`StreamId::DEFAULT`] so the migration step can wire the
    /// allocator into `CudaKernelProvider` without changing executor
    /// stream usage. The fork-and-allocate path is reserved for the
    /// AsyncCudaResource follow-up commit.
    ///
    /// This is intentionally minimal for the first cut: real
    /// per-call non-blocking streams arrive together with the async
    /// backend. Until then, the runtime exposes only the default
    /// stream slot, and stream-ordered tests use it to assert the
    /// trait contract holds even on a degenerate single-stream pool.
    pub fn acquire(&self) -> StreamId {
        drop(self.streams.lock());
        StreamId::DEFAULT
    }

    /// Borrow the live `CudaStream` for `id`. Returns `None` if `id`
    /// has never been issued by this pool. The default-stream slot
    /// resolves to the device's default stream.
    pub fn resolve(&self, id: StreamId) -> Option<Arc<CudaStream>> {
        if id == StreamId::DEFAULT {
            return Some(Arc::clone(self.device.inner().stream()));
        }
        let streams = self.streams.lock().expect("stream pool poisoned");
        let idx = id.0 as usize;
        if idx == 0 || idx > streams.len() {
            return None;
        }
        Some(Arc::clone(&streams[idx - 1]))
    }

    /// Borrow the device handle. Test helpers use this to launch
    /// kernels into the same device the pool was constructed on.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Maximum streams the pool will create on demand. Currently
    /// advisory; enforcement lands with the async backend.
    pub fn max_streams(&self) -> usize {
        self.max_streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_device() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok().map(Arc::new)
    }

    #[test]
    fn acquire_returns_default_for_now() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = StreamPool::with_defaults(device);
        assert_eq!(pool.acquire(), StreamId::DEFAULT);
    }

    #[test]
    fn resolve_default_returns_device_default_stream() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = StreamPool::with_defaults(device);
        assert!(pool.resolve(StreamId::DEFAULT).is_some());
    }

    #[test]
    fn resolve_unknown_returns_none() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = StreamPool::with_defaults(device);
        assert!(pool.resolve(StreamId(99)).is_none());
    }
}
