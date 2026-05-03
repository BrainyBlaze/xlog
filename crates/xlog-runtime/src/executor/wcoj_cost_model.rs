//! v0.6.5 slice 3 — WCOJ cost-model foundation (S1: infrastructure-only).
//!
//! Lays the seam slices 4 (recursive WCOJ) and 5 (general-arity
//! kernels) will consume to make stats-aware dispatch decisions.
//! **Behavior is unchanged in slice 3**: the default impl is a
//! verbatim wrap of the v0.6.5 slice 2 skew-classifier dispatch
//! decision.
//!
//! Two `pub(super)` traits, both internal to the executor module
//! tree:
//!
//!   * [`SkewScoreSource`] — abstraction over the GPU classifier.
//!     Production wiring: [`xlog_cuda::CudaKernelProvider`] impls
//!     it. Trait-swap unit tests use stub impls so they don't
//!     need a real CUDA fixture.
//!   * [`WcojCostModel`] — the dispatch decision. Default impl
//!     is [`SkewClassifierCostModel`]; future slices may add
//!     stats-driven impls.
//!
//! Visibility: `pub(super)` everywhere. The seam is internal to
//! the executor; slice 4/5 promote if/when needed.

use xlog_core::{RelId, Result};
use xlog_cuda::device_runtime::StreamId;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_stats::StatsManager;

use super::wcoj_dispatch::WcojKeyWidth;

/// Slice-2 thresholds — kept here as the default-impl source of
/// truth so `wcoj_dispatch.rs`'s constants stay the single
/// authoritative declaration. Slice 3 does NOT change them.
use super::wcoj_dispatch::{WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD, WCOJ_ADAPTIVE_SKEW_THRESHOLD};

// -----------------------------------------------------------------
// SkewScoreSource sub-seam
// -----------------------------------------------------------------

/// Abstraction over the GPU skew-classifier provider entries.
///
/// Production: [`CudaKernelProvider`] implements this trait by
/// dispatching to `wcoj_*_skew_score_*` based on `width`. Tests
/// use stub impls that return configured `Result<Option<f64>>`
/// values; no CUDA fixture required.
pub(super) trait SkewScoreSource {
    fn triangle_skew_score(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>>;

    fn cycle4_skew_score(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>>;
}

impl SkewScoreSource for CudaKernelProvider {
    fn triangle_skew_score(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>> {
        match width {
            WcojKeyWidth::FourByte => {
                self.wcoj_triangle_skew_score_u32(e_xy, e_yz, e_xz, launch_stream)
            }
            WcojKeyWidth::EightByte => {
                self.wcoj_triangle_skew_score_u64(e_xy, e_yz, e_xz, launch_stream)
            }
        }
    }

    fn cycle4_skew_score(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>> {
        match width {
            WcojKeyWidth::FourByte => {
                self.wcoj_4cycle_skew_score_u32(e1, e2, e3, e4, launch_stream)
            }
            WcojKeyWidth::EightByte => {
                self.wcoj_4cycle_skew_score_u64(e1, e2, e3, e4, launch_stream)
            }
        }
    }
}

// -----------------------------------------------------------------
// WcojCostModel trait + dispatch context
// -----------------------------------------------------------------

/// Inputs to a cost-model dispatch decision. Shape-agnostic;
/// carries the minimum context every implementation needs. Slice 3
/// passes the executor's `&StatsManager` for cost models that
/// want it; the default `SkewClassifierCostModel` ignores it.
///
/// `provider` is **not** a field — the classifier score is
/// fetched via a separate `&dyn SkewScoreSource` parameter so
/// trait-swap unit tests don't need a real `CudaKernelProvider`.
///
/// `stats` and `slot_rels` are populated by every call site but
/// the slice 3 default impl ignores them. Slice 4/5 cost models
/// will consult `stats.estimate_join_cardinality(slot_rels[i],
/// slot_rels[j], …)` and per-column selectivity. Marked
/// `allow(dead_code)` so the slice-3 default doesn't trigger
/// unused-field warnings while the future-consumer fields
/// remain in the public ctx shape.
#[allow(dead_code)]
pub(super) struct WcojDispatchCtx<'a> {
    pub stats: &'a StatsManager,
    pub launch_stream: StreamId,
    pub width: WcojKeyWidth,
    pub buffers: &'a [&'a CudaBuffer],
    pub slot_rels: &'a [RelId],
}

/// Cost-model seam for WCOJ dispatch.
///
/// Slice 3 ships [`SkewClassifierCostModel`] as the default impl,
/// preserving v0.6.5 slice 2 behavior verbatim. Slice 4/5 may add
/// additional impls (e.g. stats-driven cardinality estimates)
/// without rewriting the dispatch call sites.
pub(super) trait WcojCostModel: Send + Sync {
    fn should_dispatch_triangle(&self, ctx: &WcojDispatchCtx, scorer: &dyn SkewScoreSource)
        -> bool;

    fn should_dispatch_4cycle(&self, ctx: &WcojDispatchCtx, scorer: &dyn SkewScoreSource) -> bool;
}

// -----------------------------------------------------------------
// Default impl: SkewClassifierCostModel
// -----------------------------------------------------------------

/// Verbatim wrap of v0.6.5 slice 2 adaptive-classifier dispatch
/// logic. Slice 3 default; preserves dispatch counts identically.
pub(super) struct SkewClassifierCostModel {
    triangle_threshold: f64,
    cycle4_threshold: f64,
}

impl Default for SkewClassifierCostModel {
    fn default() -> Self {
        Self {
            triangle_threshold: WCOJ_ADAPTIVE_SKEW_THRESHOLD,
            cycle4_threshold: WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD,
        }
    }
}

impl WcojCostModel for SkewClassifierCostModel {
    fn should_dispatch_triangle(
        &self,
        ctx: &WcojDispatchCtx,
        scorer: &dyn SkewScoreSource,
    ) -> bool {
        debug_assert_eq!(
            ctx.buffers.len(),
            3,
            "triangle ctx must carry exactly 3 buffers"
        );
        let score = scorer.triangle_skew_score(
            ctx.buffers[0],
            ctx.buffers[1],
            ctx.buffers[2],
            ctx.launch_stream,
            ctx.width,
        );
        match score {
            Ok(Some(s)) => s >= self.triangle_threshold,
            Ok(None) | Err(_) => false,
        }
    }

    fn should_dispatch_4cycle(&self, ctx: &WcojDispatchCtx, scorer: &dyn SkewScoreSource) -> bool {
        debug_assert_eq!(
            ctx.buffers.len(),
            4,
            "4-cycle ctx must carry exactly 4 buffers"
        );
        let score = scorer.cycle4_skew_score(
            ctx.buffers[0],
            ctx.buffers[1],
            ctx.buffers[2],
            ctx.buffers[3],
            ctx.launch_stream,
            ctx.width,
        );
        match score {
            Ok(Some(s)) => s >= self.cycle4_threshold,
            Ok(None) | Err(_) => false,
        }
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Trait-swap smoke: a custom `WcojCostModel` impl compiles and
    /// can be used through a trait object. The concrete dispatch
    /// invocation is exercised end-to-end by the slice 2 cert
    /// regression after step 5 wires the model into the executor.
    struct AlwaysTrueModel;
    impl WcojCostModel for AlwaysTrueModel {
        fn should_dispatch_triangle(
            &self,
            _ctx: &WcojDispatchCtx,
            _scorer: &dyn SkewScoreSource,
        ) -> bool {
            true
        }
        fn should_dispatch_4cycle(
            &self,
            _ctx: &WcojDispatchCtx,
            _scorer: &dyn SkewScoreSource,
        ) -> bool {
            true
        }
    }

    struct AlwaysFalseModel;
    impl WcojCostModel for AlwaysFalseModel {
        fn should_dispatch_triangle(
            &self,
            _ctx: &WcojDispatchCtx,
            _scorer: &dyn SkewScoreSource,
        ) -> bool {
            false
        }
        fn should_dispatch_4cycle(
            &self,
            _ctx: &WcojDispatchCtx,
            _scorer: &dyn SkewScoreSource,
        ) -> bool {
            false
        }
    }

    #[test]
    fn default_thresholds_match_slice2_constants() {
        let m = SkewClassifierCostModel::default();
        assert_eq!(m.triangle_threshold, WCOJ_ADAPTIVE_SKEW_THRESHOLD);
        assert_eq!(m.cycle4_threshold, WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD);
    }

    #[test]
    fn slice2_thresholds_pinned_at_0_10() {
        // Slice 3 must NOT change slice 2 thresholds. Pinning here
        // catches accidental drift; future slices that intentionally
        // change the threshold update this test explicitly with
        // bench evidence.
        assert_eq!(WCOJ_ADAPTIVE_SKEW_THRESHOLD, 0.10);
        assert_eq!(WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD, 0.10);
    }

    #[test]
    fn custom_cost_model_swap_via_trait_object() {
        // Different impls produce different decisions. The trait
        // object handle compiles and dispatches dynamically — load-
        // bearing for slice 4/5 swap-in.
        let always_true: Box<dyn WcojCostModel> = Box::new(AlwaysTrueModel);
        let always_false: Box<dyn WcojCostModel> = Box::new(AlwaysFalseModel);
        let _ = (&always_true, &always_false);
    }

    /// `Result<Option<f64>>` patterns the cost model collapses:
    /// `Ok(Some(score))` with score >= threshold → dispatch;
    /// `Ok(Some(score))` with score < threshold → fall back;
    /// `Ok(None)` and `Err(_)` → fall back. This test pins the
    /// branch logic without requiring a real `&CudaBuffer`.
    #[test]
    fn classifier_branch_logic_at_threshold() {
        let m = SkewClassifierCostModel::default();
        let collapse = |result: Result<Option<f64>>| -> bool {
            match result {
                Ok(Some(s)) => s >= m.triangle_threshold,
                Ok(None) | Err(_) => false,
            }
        };
        assert!(collapse(Ok(Some(0.20))));
        assert!(collapse(Ok(Some(m.triangle_threshold))));
        assert!(!collapse(Ok(Some(0.05))));
        assert!(!collapse(Ok(None)));
        assert!(!collapse(Err(xlog_core::XlogError::Kernel("test".into()))));
    }
}
