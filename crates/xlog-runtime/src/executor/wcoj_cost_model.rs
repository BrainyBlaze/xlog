use xlog_core::{RelId, RuntimeConfig};
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
}

pub(super) const MIN_CARDINALITY_BINARY_INTERMEDIATE: u64 = 4_096;
pub(super) const LARGE_CARDINALITY_BINARY_INTERMEDIATE: u64 = 1_000_000;

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
}

pub(super) fn build_wcoj_cost_model(_config: &RuntimeConfig) -> Box<dyn WcojCostModel> {
    Box::new(CardinalityAwareCostModel::default())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
