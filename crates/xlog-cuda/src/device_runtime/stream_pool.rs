//! [`StreamPool`] — owned non-blocking CUDA streams indexed by
//! [`StreamId`].
//!
//! The runtime hands out stable [`StreamId`]s to callers and resolves
//! them to live `cudarc::driver::CudaStream` handles internally. The
//! pool grows on demand: `acquire` returns a fresh non-blocking stream
//! up to `max_streams`. Streams are never returned to a free-list —
//! they stay alive for the runtime's lifetime so [`StreamId`] handles
//! remain valid for correlated allocate/launch/deallocate sequences.
//!
//! # Failure semantics
//!
//! `acquire` returns [`Result`]. On capacity exhaustion or
//! `cudarc::driver::CudaStream::fork` failure the call returns
//! [`StreamPoolError`] rather than silently collapsing onto the
//! default stream — that fall-back was a footgun: it broke
//! stream-ordered isolation (a "non-default" allocation could end up
//! on the legacy default stream) without surfacing the failure to
//! the caller.

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
pub const ENV_WCOJ_POOL_MB_PER_STREAM: &str = "XLOG_WCOJ_POOL_MB_PER_STREAM";
pub const DEFAULT_POOL_MB_PER_STREAM: u64 = 256;

pub fn configured_pool_mb_per_stream() -> u64 {
    std::env::var(ENV_WCOJ_POOL_MB_PER_STREAM)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|mb| *mb > 0)
        .unwrap_or(DEFAULT_POOL_MB_PER_STREAM)
}

pub fn configured_pool_bytes_per_stream() -> u64 {
    configured_pool_mb_per_stream().saturating_mul(1024 * 1024)
}

pub fn planned_pool_budget_bytes(arms: u64, streams: u64) -> u64 {
    arms.saturating_mul(streams)
        .saturating_mul(configured_pool_bytes_per_stream())
}

/// Errors returned by [`StreamPool::acquire`]. Both variants are hard
/// failures; callers must not silently substitute [`StreamId::DEFAULT`].
#[derive(Debug)]
pub enum StreamPoolError {
    /// Pool already holds `max` non-default streams. Caller should
    /// either reuse an existing acquired id or raise the pool cap via
    /// the runtime config.
    Capacity { max: usize },
    /// `CudaStream::fork` returned an error. Carries the wrapped
    /// driver message.
    ForkFailed(String),
}

impl std::fmt::Display for StreamPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity { max } => {
                write!(f, "stream pool at capacity (max={})", max)
            }
            Self::ForkFailed(msg) => {
                write!(f, "stream fork failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for StreamPoolError {}

/// Pool of owned non-blocking CUDA streams.
pub struct StreamPool {
    device: Arc<CudaDevice>,
    max_streams: usize,
    pool_bytes_per_stream: u64,
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
            pool_bytes_per_stream: configured_pool_bytes_per_stream(),
            streams: Mutex::new(Vec::new()),
        }
    }

    /// Construct with the default cap [`DEFAULT_MAX_STREAMS`].
    pub fn with_defaults(device: Arc<CudaDevice>) -> Self {
        Self::new(device, DEFAULT_MAX_STREAMS)
    }

    /// Acquire a non-default stream id, growing the pool up to
    /// `max_streams`. Each successful call returns a distinct
    /// [`StreamId`] backed by an owned non-blocking
    /// `cudarc::driver::CudaStream` forked from the device's default
    /// stream.
    ///
    /// # Errors
    ///   * [`StreamPoolError::Capacity`] if the pool already holds
    ///     `max_streams` non-default streams.
    ///   * [`StreamPoolError::ForkFailed`] if the underlying
    ///     `CudaStream::fork` call failed.
    ///
    /// Streams are never returned to a free-list; they remain valid
    /// for the runtime's lifetime so previously returned [`StreamId`]
    /// handles keep resolving.
    pub fn acquire(&self) -> Result<StreamId, StreamPoolError> {
        let mut streams = self.streams.lock().expect("stream pool poisoned");
        if streams.len() >= self.max_streams {
            return Err(StreamPoolError::Capacity {
                max: self.max_streams,
            });
        }
        match self.device.inner().stream().fork() {
            Ok(handle) => {
                streams.push(handle);
                // Index 0 is reserved for DEFAULT; non-default
                // streams start at id 1 and correspond to
                // `streams[id - 1]`.
                Ok(StreamId(streams.len() as u32))
            }
            Err(e) => Err(StreamPoolError::ForkFailed(e.to_string())),
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

    /// Maximum streams the pool will create on demand.
    pub fn max_streams(&self) -> usize {
        self.max_streams
    }

    /// Planned per-stream pool budget, in bytes, from
    /// `XLOG_WCOJ_POOL_MB_PER_STREAM` or the 256 MiB default.
    pub fn pool_bytes_per_stream(&self) -> u64 {
        self.pool_bytes_per_stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn try_device() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok().map(Arc::new)
    }

    #[test]
    fn acquire_returns_distinct_non_default_ids() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = StreamPool::new(device, 4);
        let a = pool.acquire().expect("first acquire");
        let b = pool.acquire().expect("second acquire");
        assert_ne!(a, StreamId::DEFAULT);
        assert_ne!(b, StreamId::DEFAULT);
        assert_ne!(a, b, "consecutive acquire calls must yield distinct ids");
        assert_eq!(pool.non_default_len(), 2);
    }

    #[test]
    fn acquire_returns_capacity_error_at_max() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = StreamPool::new(device, 1);
        let _first = pool.acquire().expect("first acquire under cap");
        let err = pool.acquire();
        assert!(
            matches!(err, Err(StreamPoolError::Capacity { max: 1 })),
            "expected Capacity error once max_streams hit, got {:?}",
            err
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
        let id = pool.acquire().expect("acquire");
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

    #[test]
    fn pool_mb_per_stream_env_overrides_default() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let old = std::env::var(ENV_WCOJ_POOL_MB_PER_STREAM).ok();
        std::env::set_var(ENV_WCOJ_POOL_MB_PER_STREAM, "128");
        assert_eq!(configured_pool_mb_per_stream(), 128);
        match old {
            Some(value) => std::env::set_var(ENV_WCOJ_POOL_MB_PER_STREAM, value),
            None => std::env::remove_var(ENV_WCOJ_POOL_MB_PER_STREAM),
        }
    }

    #[test]
    fn planned_pool_budget_uses_default_4_by_4_contract() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let old = std::env::var(ENV_WCOJ_POOL_MB_PER_STREAM).ok();
        std::env::remove_var(ENV_WCOJ_POOL_MB_PER_STREAM);
        assert_eq!(configured_pool_mb_per_stream(), DEFAULT_POOL_MB_PER_STREAM);
        assert_eq!(
            planned_pool_budget_bytes(4, 4),
            4_u64 * 4 * 256 * 1024 * 1024
        );
        match old {
            Some(value) => std::env::set_var(ENV_WCOJ_POOL_MB_PER_STREAM, value),
            None => std::env::remove_var(ENV_WCOJ_POOL_MB_PER_STREAM),
        }
    }
}
