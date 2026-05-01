//! Per-dispatch WCOJ triangle phase timings, feature-gated.
//!
//! Compiled in only when the `wcoj-phase-timing` Cargo feature
//! is enabled. The struct + executor accessor are absent in
//! production builds so the hot path has zero overhead.

#![cfg(feature = "wcoj-phase-timing")]

use xlog_cuda::wcoj_phase_timing::WcojTrianglePhaseTiming;

/// Per-dispatch breakdown of WCOJ triangle dispatch wall-clock
/// time into 11 buckets (4 GPU + 4 layout/classifier/wall +
/// 1 derived total + 1 derived residual).
///
/// All times are in milliseconds (`f32`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WcojDispatchPhaseTiming {
    /// Time spent in the GPU skew classifier
    /// (`wcoj_triangle_skew_score_*`). 0.0 in `Mode::Force`.
    pub classifier_ms: f32,
    /// Time spent in `wcoj_layout_*_recorded` for e_xy / e_yz /
    /// e_xz independently.
    pub layout_xy_ms: f32,
    /// Time spent building the e_yz sorted/deduped WCOJ layout.
    pub layout_yz_ms: f32,
    /// Time spent building the e_xz sorted/deduped WCOJ layout.
    pub layout_xz_ms: f32,
    /// Sum of the three layout calls (cached for the report).
    pub layout_total_ms: f32,
    /// Per-phase GPU breakdown of `wcoj_triangle_*_recorded`
    /// from CUDA events. The host-side overhead between the
    /// total reducer and the materialize launch (sync,
    /// dtoh_scalar, alloc, H2D, recorder setup) is NOT in any
    /// of these buckets — it lives in `residual_overhead_ms`.
    pub triangle_count_ms: f32,
    /// GPU stream time spent in the device-side prefix scan.
    pub triangle_scan_ms: f32,
    /// GPU stream time spent in the single-thread total reducer.
    pub triangle_total_ms: f32,
    /// GPU stream time spent in the materialize kernel.
    pub triangle_materialize_ms: f32,
    /// Sum of the four GPU buckets (cached for the report).
    pub triangle_gpu_total_ms: f32,
    /// Wall clock of the entire dispatch — `Instant::now()`
    /// from `try_dispatch_wcoj_triangle` entry to dispatch
    /// success.
    pub execute_plan_wall_ms: f32,
    /// `wall - classifier - layout_total - triangle_gpu_total`.
    /// Captures: validation, memory allocation, recorder
    /// preflight/commit, host syncs, `dtoh_scalar_untracked`,
    /// the metadata D2H of the classifier histogram, output
    /// buffer alloc, H2D of `out_d_num_rows`, etc.
    pub residual_overhead_ms: f32,
}

impl WcojDispatchPhaseTiming {
    pub(crate) fn new(
        classifier_ms: f32,
        layout_xy_ms: f32,
        layout_yz_ms: f32,
        layout_xz_ms: f32,
        triangle: WcojTrianglePhaseTiming,
        execute_plan_wall_ms: f32,
    ) -> Self {
        let layout_total_ms = layout_xy_ms + layout_yz_ms + layout_xz_ms;
        let triangle_gpu_total_ms = triangle.triangle_gpu_total_ms();
        let residual_overhead_ms =
            execute_plan_wall_ms - classifier_ms - layout_total_ms - triangle_gpu_total_ms;
        Self {
            classifier_ms,
            layout_xy_ms,
            layout_yz_ms,
            layout_xz_ms,
            layout_total_ms,
            triangle_count_ms: triangle.count_ms,
            triangle_scan_ms: triangle.scan_ms,
            triangle_total_ms: triangle.total_ms,
            triangle_materialize_ms: triangle.materialize_ms,
            triangle_gpu_total_ms,
            execute_plan_wall_ms,
            residual_overhead_ms,
        }
    }
}
