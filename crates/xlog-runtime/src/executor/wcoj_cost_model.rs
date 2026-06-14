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

/// Tier-1.5 — Free-Join atom processing-order decision (decline-or-reorder).
///
/// The FJ engine materializes a left-deep prefix; because each probe's keys
/// must be a leading *column* prefix of its atom, FJ cannot reorder a chain
/// to start from a selective tail the way the binary fallback can. On an
/// adversarial chain this forces a large intermediate even when the result
/// is tiny (the decider's measured 3.07× peak loss). This decision lets the
/// dispatcher rescue or decline that case.
pub(super) enum FjOrderDecision {
    /// Keep the default traversal order. Returned whenever the current order
    /// is already estimated within tolerance of the binary plan (so every
    /// winning fixture is untouched) or the planner is inactive — identical
    /// to pre-Tier-1.5 behavior (fail-OPEN).
    KeepDefault,
    /// Dispatch Free Join with this atom processing order (original input
    /// indices). Prefix-key-joinable by construction.
    Reorder(Vec<usize>),
    /// No prefix-key-joinable order is estimated within tolerance of the
    /// unconstrained binary plan's peak — decline FJ so the binary fallback
    /// runs (fail-CLOSED).
    Decline,
}

/// FJ is reordered/declined only when the current order's estimated peak
/// exceeds the binary plan's by more than this ratio (6/5 = 1.2×). Integer
/// math (`5·a ≤ 6·b` ⇔ `a ≤ 1.2·b`) avoids float comparison. Mirrors the
/// strengthened `d2_skew_order_decider` acceptance gate.
const FJ_PEAK_TOLERANCE_NUM: u128 = 6;
const FJ_PEAK_TOLERANCE_DEN: u128 = 5;

/// `true` iff `peak ≤ 1.2 × baseline` (within tolerance — FJ is acceptable).
fn within_peak_tolerance(peak: u64, baseline: u64) -> bool {
    (peak as u128) * FJ_PEAK_TOLERANCE_DEN <= (baseline as u128) * FJ_PEAK_TOLERANCE_NUM
}

/// Default selectivity StatsManager applies when no cached selectivity /
/// column-distinct stats exist (10%). Reused for the statless local estimate
/// so the planner works for single-rule base-relation joins (where scans
/// never populate `StatsManager`), consistent with the authoritative path.
const DEFAULT_JOIN_SELECTIVITY_DEN: u128 = 10;

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

    /// Tier-1.5 FJ order planner (decline-or-reorder). Acts only as a safety
    /// net: when the current traversal order is already estimated within
    /// tolerance of the binary plan it returns [`FjOrderDecision::KeepDefault`]
    /// (every winning fixture untouched); only when the current order is
    /// estimated to LOSE does it search for a better prefix-key-joinable order,
    /// reordering to it ([`FjOrderDecision::Reorder`]) or declining to the
    /// binary fallback ([`FjOrderDecision::Decline`]) if none is competitive.
    ///
    /// `atom_vars[i]` = dense join-variable ids per column of input `i` (column
    /// order preserved — the prefix-key constraint keys on it); `cards[i]` =
    /// that input's ground-truth row count (from the buffer the dispatcher is
    /// about to join, NOT `StatsManager` — so it is always available even for
    /// single-rule base-relation joins); `ctx.slot_rels[i]` = its relation.
    /// All three index spaces coincide. Per-pair join estimates use
    /// `StatsManager::estimate_join_cardinality` when stats are populated.
    ///
    /// Default: [`FjOrderDecision::KeepDefault`] — only `CardinalityAware`
    /// plans (so the `SkewClassifier` opt-out also disables the reorder).
    fn plan_free_join_order(
        &self,
        ctx: &WcojDispatchCtx,
        atom_vars: &[Vec<usize>],
        cards: &[u64],
    ) -> FjOrderDecision {
        let _ = (ctx, atom_vars, cards);
        FjOrderDecision::KeepDefault
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

    fn plan_free_join_order(
        &self,
        ctx: &WcojDispatchCtx,
        atom_vars: &[Vec<usize>],
        cards: &[u64],
    ) -> FjOrderDecision {
        let n = atom_vars.len();
        // Defensive: index spaces must coincide; <3 atoms never reach here.
        if n < 3 || cards.len() != n || ctx.slot_rels.len() != n {
            return FjOrderDecision::KeepDefault;
        }
        // Binary fallback's freedom: best estimated peak over ANY connected
        // order (no prefix-key constraint). The yardstick both the current
        // and any candidate FJ order are judged against.
        let binary_peak = match best_greedy_peak(ctx, atom_vars, cards, false) {
            Some((_, p)) => p,
            // No connected order at all (degenerate) → don't intervene.
            None => return FjOrderDecision::KeepDefault,
        };
        // Safety-net gate: if the CURRENT traversal order is already within
        // tolerance, keep it — guarantees every winning fixture (whose
        // current order is good) is byte-for-byte untouched.
        let traversal: Vec<usize> = (0..n).collect();
        match eval_order_peak(ctx, atom_vars, cards, &traversal, true) {
            Some(default_peak) => {
                // Absolute floor: a small intermediate is cheap regardless of
                // order — never intervene (the Tier-1 veto's small-join
                // territory; FJ fires exactly as before). Only LARGE
                // intermediates can carry the order-loss the planner removes.
                // Also avoids spurious declines from ratio noise on tiny cards.
                if default_peak < self.min_binary_intermediate
                    || within_peak_tolerance(default_peak, binary_peak)
                {
                    return FjOrderDecision::KeepDefault;
                }
            }
            None => {
                // Traversal order is not prefix-key-joinable (FJ already
                // declines to binary). Only attempt a rescue when the join is
                // large enough to matter; otherwise keep the original behavior.
                if binary_peak < self.min_binary_intermediate {
                    return FjOrderDecision::KeepDefault;
                }
            }
        }
        // Current order loses (or is not prefix-key-valid) on a large join.
        // Try the best prefix-key-joinable order; reorder to it if competitive,
        // else decline to the binary fallback.
        match best_greedy_peak(ctx, atom_vars, cards, true) {
            Some((order, fj_peak)) if within_peak_tolerance(fj_peak, binary_peak) => {
                FjOrderDecision::Reorder(order)
            }
            _ => FjOrderDecision::Decline,
        }
    }
}

pub(super) fn build_wcoj_cost_model(config: &RuntimeConfig) -> Box<dyn WcojCostModel> {
    match config.resolved_wcoj_cost_model() {
        CostModelKind::SkewClassifier => Box::new(SkewClassifierCostModel),
        CostModelKind::Cardinality => Box::new(CardinalityAwareCostModel::default()),
    }
}

// ---------------------------------------------------------------------------
// Tier-1.5 FJ order-planning helpers (decline-or-reorder).
//
// A left-deep prefix model: starting from a leader relation, atoms are added
// one at a time; the running intermediate size is multiplied by the per-tuple
// fan-out of each added atom (independence assumption), and the planner tracks
// the peak running size. `constrained = true` enforces FJ's rule that an
// atom's already-bound vars form a leading COLUMN prefix (so it can be probed
// as a key); `constrained = false` is the binary fallback's freedom to join on
// any shared key. The constrained/unconstrained peak gap is exactly the
// order-loss the decider measured.
// ---------------------------------------------------------------------------

/// Best (lowest estimated peak) order over all leaders. Returns `(order,
/// peak)`, or `None` if no complete connected (and, when constrained,
/// prefix-key-joinable) order exists from any leader.
fn best_greedy_peak(
    ctx: &WcojDispatchCtx,
    atom_vars: &[Vec<usize>],
    cards: &[u64],
    constrained: bool,
) -> Option<(Vec<usize>, u64)> {
    let n = atom_vars.len();
    let mut best: Option<(Vec<usize>, u64)> = None;
    for leader in 0..n {
        if let Some((order, peak)) = greedy_from_leader(ctx, atom_vars, cards, constrained, leader) {
            if best.as_ref().map_or(true, |(_, bp)| peak < *bp) {
                best = Some((order, peak));
            }
        }
    }
    best
}

/// Greedy left-deep order from a fixed leader: repeatedly append the
/// connected (and, when constrained, prefix-key-joinable) not-yet-placed atom
/// that minimizes the running intermediate estimate. Returns `(order, peak)`
/// or `None` if some atom can never be appended under the constraints.
fn greedy_from_leader(
    ctx: &WcojDispatchCtx,
    atom_vars: &[Vec<usize>],
    cards: &[u64],
    constrained: bool,
    leader: usize,
) -> Option<(Vec<usize>, u64)> {
    let n = atom_vars.len();
    let mut order = Vec::with_capacity(n);
    let mut placed = vec![false; n];
    let mut bound: std::collections::HashSet<usize> = std::collections::HashSet::new();
    order.push(leader);
    placed[leader] = true;
    bound.extend(atom_vars[leader].iter().copied());
    let mut running = cards[leader].max(1);
    let mut peak = running;
    for _ in 1..n {
        let mut choice: Option<(usize, u64)> = None;
        for a in 0..n {
            if placed[a] || !is_addable(&atom_vars[a], &bound, constrained) {
                continue;
            }
            let new_running = estimate_join_onto_prefix(ctx, atom_vars, cards, &order, running, a);
            if choice.as_ref().map_or(true, |(_, c)| new_running < *c) {
                choice = Some((a, new_running));
            }
        }
        let (a, new_running) = choice?;
        order.push(a);
        placed[a] = true;
        bound.extend(atom_vars[a].iter().copied());
        running = new_running;
        peak = peak.max(running);
    }
    Some((order, peak))
}

/// Estimated peak of a SPECIFIC processing order (used to score the current
/// traversal order). Returns `None` if the order is not valid under the
/// constraints (e.g. the traversal order is not prefix-key-joinable).
fn eval_order_peak(
    ctx: &WcojDispatchCtx,
    atom_vars: &[Vec<usize>],
    cards: &[u64],
    order: &[usize],
    constrained: bool,
) -> Option<u64> {
    let mut bound: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut running = 0u64;
    let mut peak = 0u64;
    for (step, &a) in order.iter().enumerate() {
        if step == 0 {
            running = cards[a].max(1);
        } else {
            if !is_addable(&atom_vars[a], &bound, constrained) {
                return None;
            }
            running = estimate_join_onto_prefix(ctx, atom_vars, cards, &order[..step], running, a);
        }
        bound.extend(atom_vars[a].iter().copied());
        peak = peak.max(running);
    }
    Some(peak)
}

/// Whether atom `a` (its per-column var ids) can extend a prefix that has
/// already bound `bound`. Always requires connectivity (a shares ≥1 bound
/// var); when `constrained`, the bound vars must additionally be a leading
/// COLUMN prefix of `a` (FJ's probe-key rule).
fn is_addable(a_vars: &[usize], bound: &std::collections::HashSet<usize>, constrained: bool) -> bool {
    if !a_vars.iter().any(|v| bound.contains(v)) {
        return false; // disconnected — never a Cartesian step
    }
    if constrained {
        let split = a_vars.iter().take_while(|v| bound.contains(v)).count();
        // need ≥1 bound key prefix, and no bound var AFTER an unbound one.
        if split == 0 || a_vars[split..].iter().any(|v| bound.contains(v)) {
            return false;
        }
    }
    true
}

/// Estimated intermediate size after joining atom `a` onto the current
/// left-deep prefix, via the most selective already-placed partner. Uses the
/// independence model `running × (pairwise(p,a) / card(p))`.
fn estimate_join_onto_prefix(
    ctx: &WcojDispatchCtx,
    atom_vars: &[Vec<usize>],
    cards: &[u64],
    placed: &[usize],
    running: u64,
    a: usize,
) -> u64 {
    let mut best_new = u64::MAX;
    for &p in placed {
        let (p_keys, a_keys) = shared_keys(&atom_vars[p], &atom_vars[a]);
        if p_keys.is_empty() {
            continue;
        }
        let pairwise = pairwise_estimate(ctx, cards, p, a, &p_keys, &a_keys);
        let fan_out = pairwise as f64 / (cards[p].max(1)) as f64;
        let new_running = ((running as f64) * fan_out).ceil().max(1.0);
        let new_running = if new_running >= u64::MAX as f64 {
            u64::MAX
        } else {
            new_running as u64
        };
        best_new = best_new.min(new_running);
    }
    // Caller guarantees `a` shares a bound var with some placed atom.
    best_new.max(1)
}

/// Column positions of the join variables shared between `p` and `a`.
fn shared_keys(p_vars: &[usize], a_vars: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut p_keys = Vec::new();
    let mut a_keys = Vec::new();
    for (pi, pv) in p_vars.iter().enumerate() {
        if let Some(ai) = a_vars.iter().position(|av| av == pv) {
            p_keys.push(pi);
            a_keys.push(ai);
        }
    }
    (p_keys, a_keys)
}

/// Pairwise join-cardinality estimate. Prefers the authoritative
/// `StatsManager::estimate_join_cardinality` over the ACTUAL key columns when
/// both relations have populated stats (cached selectivity / column distinct
/// estimates then refine it, per @dts-dlm-main); otherwise applies the same
/// default model (10% selectivity) over the ground-truth buffer cardinalities,
/// so statless single-rule joins (where scans never populate StatsManager) are
/// still planned.
fn pairwise_estimate(
    ctx: &WcojDispatchCtx,
    cards: &[u64],
    p: usize,
    a: usize,
    p_keys: &[usize],
    a_keys: &[usize],
) -> u64 {
    let stats_present = ctx
        .stats
        .get_relation_stats(ctx.slot_rels[p])
        .map_or(false, |s| s.cardinality > 0)
        && ctx
            .stats
            .get_relation_stats(ctx.slot_rels[a])
            .map_or(false, |s| s.cardinality > 0);
    if stats_present {
        ctx.stats
            .estimate_join_cardinality(ctx.slot_rels[p], ctx.slot_rels[a], p_keys, a_keys)
    } else {
        let prod = (cards[p] as u128) * (cards[a] as u128) / DEFAULT_JOIN_SELECTIVITY_DEN;
        prod.min(u64::MAX as u128).max(1) as u64
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

    // -- Tier-1.5 FJ order planner -----------------------------------------

    fn nway_ctx<'a>(stats: &'a StatsManager, slot_rels: &'a [RelId]) -> WcojDispatchCtx<'a> {
        WcojDispatchCtx {
            stats,
            launch_stream: StreamId::DEFAULT,
            width: WcojKeyWidth::FourByte,
            slot_rels,
        }
    }

    fn slots(n: usize) -> Vec<RelId> {
        (0..n as u32).map(RelId).collect()
    }

    /// Adversarial chain q(A,E):-e1(A,B),e2(B,C),e3(C,D),e4(D,E): the ONLY
    /// prefix-key-joinable order is left-to-right (which materializes the N²
    /// prefix), while the binary plan can start from the 1-row tail. The
    /// planner must DECLINE.
    #[test]
    fn test_tier15_chain_declines() {
        let cards = [100u64, 100, 10_000, 1];
        let stats = stats_with_cards(&cards); // populated, but no cached selectivity
        let rels = slots(4);
        let ctx = nway_ctx(&stats, &rels);
        // e1=[A,B] e2=[B,C] e3=[C,D] e4=[D,E] (vars A=0..E=4)
        let atom_vars = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]];
        let model = CardinalityAwareCostModel::default();
        assert!(
            matches!(
                model.plan_free_join_order(&ctx, &atom_vars, &cards),
                FjOrderDecision::Decline
            ),
            "FJ-forced chain order blows up vs the tail-first binary plan → decline"
        );
    }

    /// Small chain e1(X,Y),e2(Y,Z),r(Z,B) (the recursive-SCC winner shape):
    /// the same chain topology as the adversarial case but with small balanced
    /// cardinalities, so the intermediate stays below the worthwhile threshold.
    /// The planner must KEEP the default order (FJ fires as before) — the
    /// order-loss only matters for large intermediates.
    #[test]
    fn test_tier15_small_chain_keeps_default() {
        let cards = [3u64, 3, 2];
        let stats = stats_with_cards(&cards);
        let rels = slots(3);
        let ctx = nway_ctx(&stats, &rels);
        let atom_vars = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let model = CardinalityAwareCostModel::default();
        assert!(
            matches!(
                model.plan_free_join_order(&ctx, &atom_vars, &cards),
                FjOrderDecision::KeepDefault
            ),
            "small chain below the worthwhile threshold: keep default, FJ fires"
        );
    }

    /// Triangle r1(A,B),r2(B,C),r3(A,C): cyclic, so FJ's prefix order is no
    /// worse than binary — the planner keeps the default order (winner
    /// untouched).
    #[test]
    fn test_tier15_triangle_keeps_default() {
        let cards = [100u64, 100, 100];
        let stats = stats_with_cards(&cards);
        let rels = slots(3);
        let ctx = nway_ctx(&stats, &rels);
        let atom_vars = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let model = CardinalityAwareCostModel::default();
        assert!(
            matches!(
                model.plan_free_join_order(&ctx, &atom_vars, &cards),
                FjOrderDecision::KeepDefault
            ),
            "symmetric triangle: default order already competitive → keep it"
        );
    }

    /// Alternating-cardinality 4-cycle: the traversal order starts on a big
    /// relation (bad prefix), but a rotation starting on a 1-row relation is
    /// both prefix-key-joinable AND competitive — the planner must REORDER to
    /// a non-identity valid permutation.
    #[test]
    fn test_tier15_cycle4_reorders() {
        let cards = [10_000u64, 1, 10_000, 1];
        let stats = stats_with_cards(&cards);
        let rels = slots(4);
        let ctx = nway_ctx(&stats, &rels);
        // r1=[A,B] r2=[B,C] r3=[C,D] r4=[D,A] (vars A=0,B=1,C=2,D=3)
        let atom_vars = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]];
        let model = CardinalityAwareCostModel::default();
        match model.plan_free_join_order(&ctx, &atom_vars, &cards) {
            FjOrderDecision::Reorder(order) => {
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, vec![0, 1, 2, 3], "order must be a permutation");
                assert_ne!(order, vec![0, 1, 2, 3], "must differ from traversal order");
                assert!(
                    eval_order_peak(&ctx, &atom_vars, &cards, &order, true).is_some(),
                    "reordered plan must be prefix-key-joinable"
                );
            }
            other => panic!("expected Reorder, got {:?}", DecisionDbg(&other)),
        }
    }

    /// The SkewClassifier opt-out disables the planner entirely (default trait
    /// impl → KeepDefault), even on the adversarial chain.
    #[test]
    fn test_tier15_skew_model_never_plans() {
        let cards = [100u64, 100, 10_000, 1];
        let stats = stats_with_cards(&cards);
        let rels = slots(4);
        let ctx = nway_ctx(&stats, &rels);
        let atom_vars = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]];
        let model = SkewClassifierCostModel;
        assert!(matches!(
            model.plan_free_join_order(&ctx, &atom_vars, &cards),
            FjOrderDecision::KeepDefault
        ));
    }

    // Minimal Debug for panic messages (FjOrderDecision is not Debug).
    struct DecisionDbg<'a>(&'a FjOrderDecision);
    impl std::fmt::Debug for DecisionDbg<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self.0 {
                FjOrderDecision::KeepDefault => write!(f, "KeepDefault"),
                FjOrderDecision::Reorder(o) => write!(f, "Reorder({o:?})"),
                FjOrderDecision::Decline => write!(f, "Decline"),
            }
        }
    }
}
