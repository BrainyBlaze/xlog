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

    /// Acquire a non-default stream id, growing the pool up to
    /// `max_streams`. Each call returns a distinct [`StreamId`]
    /// backed by an owned non-blocking `cudarc::driver::CudaStream`
    /// forked from the device's default stream. When the pool is at
    /// capacity, returns [`StreamId::DEFAULT`] (which resolves to
    /// the device default stream) so callers can fall back without
    /// failing.
    ///
    /// Streams are never returned to a free-list; they remain valid
    /// for the runtime's lifetime so existing [`StreamId`] handles
    /// keep resolving.
    pub fn acquire(&self) -> StreamId {
        let mut streams = self.streams.lock().expect("stream pool poisoned");
        if streams.len() >= self.max_streams {
            return StreamId::DEFAULT;
        }
        // Fork a non-blocking stream from the device's default stream.
        // On failure we fall back to the default-stream id.
        match self.device.inner().stream().fork() {
            Ok(handle) => {
                streams.push(handle);
                // Index 0 is reserved for DEFAULT; non-default
                // streams start at id 1 and correspond to
                // `streams[id - 1]`.
                StreamId(streams.len() as u32)
            }
            Err(_) => StreamId::DEFAULT,
        }
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

    /// Number of non-default streams currently in the pool.
    pub fn non_default_len(&self) -> usize {
        self.streams.lock().expect("stream pool poisoned").len()
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
    fn acquire_returns_distinct_non_default_ids() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = StreamPool::new(device, 4);
        let a = pool.acquire();
        let b = pool.acquire();
        assert_ne!(a, StreamId::DEFAULT);
        assert_ne!(b, StreamId::DEFAULT);
        assert_ne!(a, b, "consecutive acquire calls must yield distinct ids");
        assert_eq!(pool.non_default_len(), 2);
    }

    #[test]
    fn acquire_falls_back_to_default_at_capacity() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = StreamPool::new(device, 1);
        let _first = pool.acquire();
        let second = pool.acquire();
        assert_eq!(
            second,
            StreamId::DEFAULT,
            "pool must fall back to DEFAULT once max_streams is hit"
        );
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
    fn resolve_acquired_returns_owned_stream() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = StreamPool::new(device, 4);
        let id = pool.acquire();
        assert_ne!(id, StreamId::DEFAULT);
        assert!(pool.resolve(id).is_some());
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
