//! Per-phase timing types for the WCOJ triangle dispatch.
//!
//! Compiled in only when the `wcoj-phase-timing` Cargo feature is
//! enabled. Production builds omit this module entirely so the
//! hot path has zero overhead.
//!
//! Records four GPU-stream-time phases inside
//! `wcoj_triangle_*_recorded` via CUDA events:
//!   * `count_ms`       — `wcoj_triangle_count` kernel + the
//!                        `count_buf → offsets_buf` dtod copy.
//!   * `scan_ms`        — `multiblock_scan_u32_inplace_on_stream`.
//!   * `total_ms`       — `wcoj_compute_total` kernel.
//!   * `materialize_ms` — `wcoj_triangle_materialize` kernel.
//!
//! Each value is the stream-time elapsed between two
//! `cudarc::driver::CudaEvent` records, returned as `f32`
//! milliseconds (CUDA's native event-elapsed precision). The
//! sum of the four buckets is `triangle_gpu_total_ms`; any
//! host-side overhead between phases (the scalar D2H of
//! `total_rows`, output buffer allocation, the H2D of
//! `out_d_num_rows`, recorder setup/commit) lives in the
//! caller's residual bucket.

#![cfg(feature = "wcoj-phase-timing")]

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WcojTrianglePhaseTiming {
    pub count_ms: f32,
    pub scan_ms: f32,
    pub total_ms: f32,
    pub materialize_ms: f32,
}

impl WcojTrianglePhaseTiming {
    /// Sum of the four GPU buckets. Excludes host-side
    /// overhead between phases.
    pub fn triangle_gpu_total_ms(&self) -> f32 {
        self.count_ms + self.scan_ms + self.total_ms + self.materialize_ms
    }
}
