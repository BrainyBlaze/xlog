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
//!     Production wiring: per-shape wrappers
//!     ([`TriangleScorer`], [`Cycle4Scorer`]) bind a
//!     [`xlog_cuda::CudaKernelProvider`] reference together with
//!     the dispatch-site edge buffers; the trait surface itself
//!     is buffer-free, so unit tests use a stub impl that
//!     ignores buffers entirely (no CUDA fixture required).
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
/// **Buffer-free trait surface.** The trait deliberately does
/// not carry `&CudaBuffer` params: production scorers
/// ([`TriangleScorer`], [`Cycle4Scorer`]) own their per-shape
/// buffer references in `&self`; the `WcojCostModel` only sees
/// `(launch_stream, width) → Result<Option<f64>>`. This is
/// what makes the seam unit-testable — tests construct a stub
/// scorer that returns configured scores without needing a
/// real `&CudaBuffer` (and thus no CUDA fixture).
///
/// Each scorer is shape-specific. A `TriangleScorer`'s
/// `cycle4_skew_score` (and vice versa) is unreachable in
/// practice — the cost model only invokes the matching method
/// for its dispatch shape — so the wrong-shape impls return
/// `Ok(None)` defensively. If ever called by mistake, the
/// dispatch path falls back gracefully instead of panicking
/// inside the executor's hot path.
pub(super) trait SkewScoreSource {
    fn triangle_skew_score(
        &self,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>>;

    fn cycle4_skew_score(
        &self,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>>;
}

/// Production triangle scorer: wraps the provider + the three
/// edge buffers the classifier reads. Constructed at the
/// triangle dispatch site; lives only for the duration of the
/// cost-model decision.
pub(super) struct TriangleScorer<'a> {
    pub provider: &'a CudaKernelProvider,
    pub e_xy: &'a CudaBuffer,
    pub e_yz: &'a CudaBuffer,
    pub e_xz: &'a CudaBuffer,
}

impl<'a> SkewScoreSource for TriangleScorer<'a> {
    fn triangle_skew_score(
        &self,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>> {
        match width {
            WcojKeyWidth::FourByte => self.provider.wcoj_triangle_skew_score_u32(
                self.e_xy,
                self.e_yz,
                self.e_xz,
                launch_stream,
            ),
            WcojKeyWidth::EightByte => self.provider.wcoj_triangle_skew_score_u64(
                self.e_xy,
                self.e_yz,
                self.e_xz,
                launch_stream,
            ),
        }
    }

    fn cycle4_skew_score(
        &self,
        _launch_stream: StreamId,
        _width: WcojKeyWidth,
    ) -> Result<Option<f64>> {
        // Wrong-shape call — defensively fall back. Unreachable
        // from the executor's dispatch sites (triangle invokes
        // only `should_dispatch_triangle`, which calls only
        // `triangle_skew_score`). Returning `Ok(None)` keeps the
        // misuse contained instead of panicking inside a hot
        // path or unwinding across the FFI boundary.
        Ok(None)
    }
}

/// Production 4-cycle scorer: wraps the provider + the four
/// edge buffers the classifier reads.
pub(super) struct Cycle4Scorer<'a> {
    pub provider: &'a CudaKernelProvider,
    pub e1: &'a CudaBuffer,
    pub e2: &'a CudaBuffer,
    pub e3: &'a CudaBuffer,
    pub e4: &'a CudaBuffer,
}

impl<'a> SkewScoreSource for Cycle4Scorer<'a> {
    fn triangle_skew_score(
        &self,
        _launch_stream: StreamId,
        _width: WcojKeyWidth,
    ) -> Result<Option<f64>> {
        // Wrong-shape call — see TriangleScorer::cycle4_skew_score.
        Ok(None)
    }

    fn cycle4_skew_score(
        &self,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<Option<f64>> {
        match width {
            WcojKeyWidth::FourByte => self.provider.wcoj_4cycle_skew_score_u32(
                self.e1,
                self.e2,
                self.e3,
                self.e4,
                launch_stream,
            ),
            WcojKeyWidth::EightByte => self.provider.wcoj_4cycle_skew_score_u64(
                self.e1,
                self.e2,
                self.e3,
                self.e4,
                launch_stream,
            ),
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
/// **No buffer field.** The classifier score is fetched via a
/// separate `&dyn SkewScoreSource` parameter — production
/// scorers carry their per-shape buffers internally
/// ([`TriangleScorer`] / [`Cycle4Scorer`]). This keeps the ctx
/// itself shape-agnostic and lets unit tests build it without a
/// `&CudaBuffer`.
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
            ctx.slot_rels.len(),
            3,
            "triangle ctx must carry exactly 3 slot relations"
        );
        let score = scorer.triangle_skew_score(ctx.launch_stream, ctx.width);
        match score {
            Ok(Some(s)) => s >= self.triangle_threshold,
            Ok(None) | Err(_) => false,
        }
    }

    fn should_dispatch_4cycle(&self, ctx: &WcojDispatchCtx, scorer: &dyn SkewScoreSource) -> bool {
        debug_assert_eq!(
            ctx.slot_rels.len(),
            4,
            "4-cycle ctx must carry exactly 4 slot relations"
        );
        let score = scorer.cycle4_skew_score(ctx.launch_stream, ctx.width);
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
    use std::cell::RefCell;
    use xlog_stats::StatsManager;

    // -------------------------------------------------------------
    // StubScorer: configurable SkewScoreSource for unit tests
    // -------------------------------------------------------------

    /// Stub scorer that returns configured `Result<Option<f64>>`
    /// values per shape. `RefCell<Option<…>>` lets each call site
    /// `.take()` the configured value once — pinning the cost-
    /// model's contract that it consults the scorer exactly once
    /// per `should_dispatch_*` invocation. A second call would
    /// observe `None` and panic in `expect`, which doubles as a
    /// regression check.
    struct StubScorer {
        triangle: RefCell<Option<Result<Option<f64>>>>,
        cycle4: RefCell<Option<Result<Option<f64>>>>,
    }

    impl StubScorer {
        fn with_triangle(score: Result<Option<f64>>) -> Self {
            Self {
                triangle: RefCell::new(Some(score)),
                cycle4: RefCell::new(None),
            }
        }

        fn with_cycle4(score: Result<Option<f64>>) -> Self {
            Self {
                triangle: RefCell::new(None),
                cycle4: RefCell::new(Some(score)),
            }
        }
    }

    impl SkewScoreSource for StubScorer {
        fn triangle_skew_score(
            &self,
            _launch_stream: StreamId,
            _width: WcojKeyWidth,
        ) -> Result<Option<f64>> {
            self.triangle
                .borrow_mut()
                .take()
                .expect("triangle_skew_score called without configured stub value")
        }

        fn cycle4_skew_score(
            &self,
            _launch_stream: StreamId,
            _width: WcojKeyWidth,
        ) -> Result<Option<f64>> {
            self.cycle4
                .borrow_mut()
                .take()
                .expect("cycle4_skew_score called without configured stub value")
        }
    }

    /// Helper: builds a triangle-shape `WcojDispatchCtx` for
    /// `should_dispatch_triangle` tests. `slot_rels.len() == 3`
    /// is what the cost model `debug_assert`s.
    fn triangle_ctx<'a>(stats: &'a StatsManager, slot_rels: &'a [RelId]) -> WcojDispatchCtx<'a> {
        WcojDispatchCtx {
            stats,
            launch_stream: StreamId(0),
            width: WcojKeyWidth::FourByte,
            slot_rels,
        }
    }

    /// Helper: builds a 4-cycle-shape ctx (`slot_rels.len() == 4`).
    fn cycle4_ctx<'a>(stats: &'a StatsManager, slot_rels: &'a [RelId]) -> WcojDispatchCtx<'a> {
        WcojDispatchCtx {
            stats,
            launch_stream: StreamId(0),
            width: WcojKeyWidth::FourByte,
            slot_rels,
        }
    }

    // -------------------------------------------------------------
    // Threshold pinning — slice 2 contract
    // -------------------------------------------------------------

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

    // -------------------------------------------------------------
    // Triangle: 5 score scenarios × should_dispatch_triangle
    // -------------------------------------------------------------

    #[test]
    fn triangle_dispatches_when_score_above_threshold() {
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_triangle(Ok(Some(0.20)));
        assert!(m.should_dispatch_triangle(&ctx, &scorer));
    }

    #[test]
    fn triangle_dispatches_when_score_at_threshold() {
        // Threshold check is `>=` — equality counts as dispatch.
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_triangle(Ok(Some(WCOJ_ADAPTIVE_SKEW_THRESHOLD)));
        assert!(m.should_dispatch_triangle(&ctx, &scorer));
    }

    #[test]
    fn triangle_falls_back_when_score_below_threshold() {
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_triangle(Ok(Some(0.05)));
        assert!(!m.should_dispatch_triangle(&ctx, &scorer));
    }

    #[test]
    fn triangle_falls_back_when_classifier_returns_none() {
        // `Ok(None)` is the slice 2 contract for "classifier
        // declined to score" (e.g. empty inputs, runtime issue
        // short of an error). Cost model must fall back, not
        // dispatch.
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_triangle(Ok(None));
        assert!(!m.should_dispatch_triangle(&ctx, &scorer));
    }

    #[test]
    fn triangle_falls_back_when_classifier_errors() {
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_triangle(Err(xlog_core::XlogError::Kernel("test".into())));
        assert!(!m.should_dispatch_triangle(&ctx, &scorer));
    }

    // -------------------------------------------------------------
    // 4-cycle: 5 score scenarios × should_dispatch_4cycle
    // -------------------------------------------------------------

    #[test]
    fn cycle4_dispatches_when_score_above_threshold() {
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2), RelId(3)];
        let ctx = cycle4_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_cycle4(Ok(Some(0.20)));
        assert!(m.should_dispatch_4cycle(&ctx, &scorer));
    }

    #[test]
    fn cycle4_dispatches_when_score_at_threshold() {
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2), RelId(3)];
        let ctx = cycle4_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_cycle4(Ok(Some(WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD)));
        assert!(m.should_dispatch_4cycle(&ctx, &scorer));
    }

    #[test]
    fn cycle4_falls_back_when_score_below_threshold() {
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2), RelId(3)];
        let ctx = cycle4_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_cycle4(Ok(Some(0.05)));
        assert!(!m.should_dispatch_4cycle(&ctx, &scorer));
    }

    #[test]
    fn cycle4_falls_back_when_classifier_returns_none() {
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2), RelId(3)];
        let ctx = cycle4_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_cycle4(Ok(None));
        assert!(!m.should_dispatch_4cycle(&ctx, &scorer));
    }

    #[test]
    fn cycle4_falls_back_when_classifier_errors() {
        let m = SkewClassifierCostModel::default();
        let stats = StatsManager::default();
        let slot_rels = [RelId(0), RelId(1), RelId(2), RelId(3)];
        let ctx = cycle4_ctx(&stats, &slot_rels);
        let scorer = StubScorer::with_cycle4(Err(xlog_core::XlogError::Kernel("test".into())));
        assert!(!m.should_dispatch_4cycle(&ctx, &scorer));
    }

    // -------------------------------------------------------------
    // Trait-swap smoke: dynamic dispatch through Box<dyn …>
    // -------------------------------------------------------------

    #[test]
    fn custom_cost_model_swap_via_trait_object() {
        // A custom impl reaches the dispatch decision through the
        // same `WcojCostModel` trait object the executor uses —
        // load-bearing for slice 4/5 swap-in.
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
        let stats = StatsManager::default();
        let triangle_slots = [RelId(0), RelId(1), RelId(2)];
        let cycle4_slots = [RelId(0), RelId(1), RelId(2), RelId(3)];
        let m: Box<dyn WcojCostModel> = Box::new(AlwaysTrueModel);
        let scorer = StubScorer {
            triangle: RefCell::new(None),
            cycle4: RefCell::new(None),
        };
        // The trait-object impl ignores the scorer entirely, so
        // `None` configured values never get `.take()`d — proving
        // the dispatch goes through the swapped impl, not the
        // default one.
        assert!(m.should_dispatch_triangle(&triangle_ctx(&stats, &triangle_slots), &scorer));
        assert!(m.should_dispatch_4cycle(&cycle4_ctx(&stats, &cycle4_slots), &scorer));
    }
}
