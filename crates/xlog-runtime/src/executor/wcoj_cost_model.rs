use xlog_core::{CostModelKind, RelId, RuntimeConfig};
use xlog_cuda::device_runtime::StreamId;
use xlog_stats::StatsManager;

use super::wcoj_dispatch::WcojKeyWidth;

#[allow(dead_code)]
pub(super) struct WcojDispatchCtx<'a> {
    pub stats: &'a StatsManager,
    pub launch_stream: StreamId,
    pub width: WcojKeyWidth,
    pub slot_rels: &'a [RelId],
}

pub(super) trait WcojCostModel: Send + Sync {
    fn should_dispatch_triangle(&self, ctx: &WcojDispatchCtx) -> bool;
    fn should_dispatch_4cycle(&self, ctx: &WcojDispatchCtx) -> bool;

    /// Fail-OPEN loss-region veto for the general factorized routes
    /// (D1 aggregate-fused WCOJ over a triangle; D2 Free Join). Returns
    /// `true` ONLY when the model has full cardinality stats for every
    /// slot relation AND the largest one is below the WCOJ-worthwhile
    /// threshold — i.e. the join is provably small, no intermediate can
    /// blow up, and the binary fallback wins (the measured 1.7–2.0×
    /// cost-of-generality region for Free Join, and the small-triangle
    /// region the base triangle cost model already declines).
    ///
    /// Deliberately conservative: stats absent for ANY slot, or ANY slot
    /// large → `false` (no veto). This NEVER vetoes a case with a large
    /// input (where factorized can win on a large avoided intermediate)
    /// and NEVER vetoes when stats are unavailable (e.g. recursive deltas
    /// on early iterations) — so every measured D1/D2 gate win is
    /// preserved exactly; the veto only removes provably-small losses.
    fn factorized_loss_veto(&self, ctx: &WcojDispatchCtx) -> bool {
        let _ = ctx;
        false
    }
}

pub(super) const MIN_CARDINALITY_BINARY_INTERMEDIATE: u64 = 4_096;
pub(super) const LARGE_CARDINALITY_BINARY_INTERMEDIATE: u64 = 1_000_000;

#[derive(Default)]
pub(super) struct SkewClassifierCostModel;

impl WcojCostModel for SkewClassifierCostModel {
    fn should_dispatch_triangle(&self, _ctx: &WcojDispatchCtx) -> bool {
        false
    }

    fn should_dispatch_4cycle(&self, _ctx: &WcojDispatchCtx) -> bool {
        false
    }
}

pub(super) struct CardinalityAwareCostModel {
    min_binary_intermediate: u64,
    large_binary_intermediate: u64,
}

impl Default for CardinalityAwareCostModel {
    fn default() -> Self {
        Self {
            min_binary_intermediate: MIN_CARDINALITY_BINARY_INTERMEDIATE,
            large_binary_intermediate: LARGE_CARDINALITY_BINARY_INTERMEDIATE,
        }
    }
}

impl CardinalityAwareCostModel {
    fn populated_cards(&self, ctx: &WcojDispatchCtx) -> Option<Vec<u64>> {
        ctx.slot_rels
            .iter()
            .map(|r| {
                ctx.stats
                    .get_relation_stats(*r)
                    .map(|s| s.cardinality)
                    .filter(|c| *c > 0)
            })
            .collect()
    }

    fn decide_from_cardinality(&self, binary_est: u64) -> bool {
        binary_est >= self.large_binary_intermediate || binary_est >= self.min_binary_intermediate
    }
}

impl WcojCostModel for CardinalityAwareCostModel {
    fn should_dispatch_triangle(&self, ctx: &WcojDispatchCtx) -> bool {
        debug_assert_eq!(
            ctx.slot_rels.len(),
            3,
            "triangle ctx must carry exactly 3 slot relations"
        );
        if self.populated_cards(ctx).is_none() {
            return false;
        }
        let binary_est =
            ctx.stats
                .estimate_join_cardinality(ctx.slot_rels[0], ctx.slot_rels[1], &[1], &[0]);
        self.decide_from_cardinality(binary_est)
    }

    fn should_dispatch_4cycle(&self, ctx: &WcojDispatchCtx) -> bool {
        debug_assert_eq!(
            ctx.slot_rels.len(),
            4,
            "4-cycle ctx must carry exactly 4 slot relations"
        );
        if self.populated_cards(ctx).is_none() {
            return false;
        }
        let binary_est =
            ctx.stats
                .estimate_join_cardinality(ctx.slot_rels[0], ctx.slot_rels[1], &[1], &[0]);
        self.decide_from_cardinality(binary_est)
    }

    fn factorized_loss_veto(&self, ctx: &WcojDispatchCtx) -> bool {
        // Fail-open: need every slot's cardinality to make any claim.
        let cards = match self.populated_cards(ctx) {
            Some(c) => c,
            None => return false,
        };
        // Veto only when the LARGEST input is below the WCOJ-worthwhile
        // threshold: then every join intermediate is bounded small, the
        // binary plan is cheap, and the factorized route's overhead is
        // not justified. Any large input → no veto (factorized may win
        // on the avoided large intermediate).
        cards.iter().copied().max().unwrap_or(0) < self.min_binary_intermediate
    }
}

pub(super) fn build_wcoj_cost_model(config: &RuntimeConfig) -> Box<dyn WcojCostModel> {
    match config.resolved_wcoj_cost_model() {
        CostModelKind::SkewClassifier => Box::new(SkewClassifierCostModel),
        CostModelKind::Cardinality => Box::new(CardinalityAwareCostModel::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn triangle_ctx<'a>(stats: &'a StatsManager, slot_rels: &'a [RelId; 3]) -> WcojDispatchCtx<'a> {
        WcojDispatchCtx {
            stats,
            launch_stream: StreamId::DEFAULT,
            width: WcojKeyWidth::FourByte,
            slot_rels,
        }
    }

    fn cycle4_ctx<'a>(stats: &'a StatsManager, slot_rels: &'a [RelId; 4]) -> WcojDispatchCtx<'a> {
        WcojDispatchCtx {
            stats,
            launch_stream: StreamId::DEFAULT,
            width: WcojKeyWidth::FourByte,
            slot_rels,
        }
    }

    fn stats_with_cards(cards: &[u64]) -> StatsManager {
        let mut stats = StatsManager::new();
        for (i, c) in cards.iter().enumerate() {
            let rid = RelId(i as u32);
            stats.register_relation(rid);
            stats.update_cardinality(rid, *c);
        }
        stats
    }

    fn cost_model_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct CostModelEnvSnapshot(Option<String>);

    impl CostModelEnvSnapshot {
        fn capture_and_clear() -> Self {
            let prior = std::env::var("XLOG_WCOJ_COST_MODEL").ok();
            unsafe {
                std::env::remove_var("XLOG_WCOJ_COST_MODEL");
            }
            Self(prior)
        }
    }

    impl Drop for CostModelEnvSnapshot {
        fn drop(&mut self) {
            unsafe {
                match self.0.take() {
                    Some(value) => std::env::set_var("XLOG_WCOJ_COST_MODEL", value),
                    None => std::env::remove_var("XLOG_WCOJ_COST_MODEL"),
                }
            }
        }
    }

    fn with_cost_model_env<R>(f: impl FnOnce() -> R) -> R {
        let _guard = cost_model_env_lock()
            .lock()
            .expect("cost-model env lock poisoned");
        let _snapshot = CostModelEnvSnapshot::capture_and_clear();
        f()
    }

    #[test]
    fn cardinality_thresholds_pinned_in_default() {
        let m = CardinalityAwareCostModel::default();
        assert_eq!(
            m.min_binary_intermediate,
            MIN_CARDINALITY_BINARY_INTERMEDIATE
        );
        assert_eq!(
            m.large_binary_intermediate,
            LARGE_CARDINALITY_BINARY_INTERMEDIATE
        );
    }

    #[test]
    fn triangle_declines_when_any_slot_card_missing() {
        let mut stats = StatsManager::new();
        stats.register_relation(RelId(0));
        stats.update_cardinality(RelId(0), 1000);
        stats.register_relation(RelId(1));
        stats.update_cardinality(RelId(1), 1000);
        let slots = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slots);
        let m = CardinalityAwareCostModel::default();
        assert!(!m.should_dispatch_triangle(&ctx));
    }

    #[test]
    fn triangle_dispatches_when_binary_est_above_min_threshold() {
        let stats = stats_with_cards(&[1_000, 1_000, 1_000]);
        let slots = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slots);
        let m = CardinalityAwareCostModel::default();
        assert!(m.should_dispatch_triangle(&ctx));
    }

    #[test]
    fn triangle_declines_when_binary_est_below_min_threshold() {
        let stats = stats_with_cards(&[50, 50, 50]);
        let slots = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slots);
        let m = CardinalityAwareCostModel::default();
        assert!(!m.should_dispatch_triangle(&ctx));
    }

    #[test]
    fn cycle4_dispatches_when_binary_est_above_min_threshold() {
        let stats = stats_with_cards(&[1_000, 1_000, 1_000, 1_000]);
        let slots = [RelId(0), RelId(1), RelId(2), RelId(3)];
        let ctx = cycle4_ctx(&stats, &slots);
        let m = CardinalityAwareCostModel::default();
        assert!(m.should_dispatch_4cycle(&ctx));
    }

    #[test]
    fn test_w25_default_flip_factory_uses_cardinality_default() {
        with_cost_model_env(|| {
            let stats = stats_with_cards(&[1_000, 1_000, 1_000]);
            let slots = [RelId(0), RelId(1), RelId(2)];
            let ctx = triangle_ctx(&stats, &slots);
            let model = build_wcoj_cost_model(&RuntimeConfig::default());
            assert!(
                model.should_dispatch_triangle(&ctx),
                "bare default must use CardinalityAwareCostModel"
            );
        });
    }

    #[test]
    fn factorized_veto_fires_when_all_inputs_small() {
        // All slot cardinalities below the WCOJ-worthwhile threshold →
        // provably-small join → veto (use the binary fallback).
        let stats = stats_with_cards(&[50, 50, 50]);
        let slots = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slots);
        let m = CardinalityAwareCostModel::default();
        assert!(m.factorized_loss_veto(&ctx));
    }

    #[test]
    fn factorized_veto_declines_when_any_input_large() {
        // A large input means an intermediate could blow up → factorized
        // may win → never veto (fail-open on the win side).
        let stats = stats_with_cards(&[50, MIN_CARDINALITY_BINARY_INTERMEDIATE, 50]);
        let slots = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slots);
        let m = CardinalityAwareCostModel::default();
        assert!(!m.factorized_loss_veto(&ctx));
    }

    #[test]
    fn factorized_veto_fail_open_when_any_stat_missing() {
        // Missing cardinality for any slot → cannot prove small → no veto
        // (preserves measured wins where stats are unavailable, e.g.
        // recursive deltas on early iterations).
        let mut stats = StatsManager::new();
        stats.register_relation(RelId(0));
        stats.update_cardinality(RelId(0), 50);
        stats.register_relation(RelId(1));
        stats.update_cardinality(RelId(1), 50);
        // RelId(2) has no cardinality.
        let slots = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slots);
        let m = CardinalityAwareCostModel::default();
        assert!(!m.factorized_loss_veto(&ctx));
    }

    #[test]
    fn factorized_veto_general_arity_free_join_shape() {
        // The veto is arity-agnostic (Free Join bodies are >=3 inputs).
        let stats = stats_with_cards(&[10, 10, 10, 10, 10]);
        let slots = [RelId(0), RelId(1), RelId(2), RelId(3), RelId(4)];
        let ctx = WcojDispatchCtx {
            stats: &stats,
            launch_stream: StreamId::DEFAULT,
            width: WcojKeyWidth::FourByte,
            slot_rels: &slots,
        };
        let m = CardinalityAwareCostModel::default();
        assert!(m.factorized_loss_veto(&ctx), "all-small >=3-input body must veto");

        let stats_big = stats_with_cards(&[10, 10, 2_000_000, 10, 10]);
        let ctx_big = WcojDispatchCtx {
            stats: &stats_big,
            launch_stream: StreamId::DEFAULT,
            width: WcojKeyWidth::FourByte,
            slot_rels: &slots,
        };
        assert!(
            !m.factorized_loss_veto(&ctx_big),
            "a large input must NOT veto (factorized may win)"
        );
    }

    #[test]
    fn factorized_veto_skew_classifier_never_vetoes() {
        // The stub skew model must never veto (default trait impl → false).
        let stats = stats_with_cards(&[10, 10, 10]);
        let slots = [RelId(0), RelId(1), RelId(2)];
        let ctx = triangle_ctx(&stats, &slots);
        let m = SkewClassifierCostModel;
        assert!(!m.factorized_loss_veto(&ctx));
    }

    #[test]
    fn test_w25_default_flip_factory_honors_env_skew_opt_out() {
        with_cost_model_env(|| {
            unsafe {
                std::env::set_var("XLOG_WCOJ_COST_MODEL", "skew");
            }
            let stats = stats_with_cards(&[1_000, 1_000, 1_000]);
            let slots = [RelId(0), RelId(1), RelId(2)];
            let ctx = triangle_ctx(&stats, &slots);
            let model = build_wcoj_cost_model(&RuntimeConfig::default());
            assert!(
                !model.should_dispatch_triangle(&ctx),
                "env skew opt-out must bypass cardinality dispatch"
            );
        });
    }
}
