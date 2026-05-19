//! W2.1 — Variable-ordering cost model for WCOJ dispatch.
//!
//! The trait [`WcojVariableOrderingModel`] picks the kernel **leader
//! slot** (slot 0) for triangle and 4-cycle WCOJ shapes. The leader
//! is the input that drives the outer iteration; smaller leaders
//! mean fewer iterations.
//!
//! The default implementation [`LeaderCardinalityModel`] picks the
//! min-cardinality input as leader IFF the ratio
//! `min_card / default_leader_card ≤
//! config.effective_wcoj_var_ordering_threshold()`. Marginal
//! cases (above threshold) return `None` so the promoter leaves
//! `var_order = None` and slice 1/2/4/W2.2 dispatch behavior is
//! preserved.
//!
//! The trait returns only the **leader index** in the promoter's
//! canonical input order — the promoter is responsible for building
//! the full [`xlog_ir::rir::VariableOrder`] using the locked
//! permutation tables (because computing `kernel_output_cols`
//! requires the rule's head-variable order, which only the promoter
//! has).
//!
//! # Locked permutation tables (W2.1 plan §"Permutation Tables")
//!
//! Triangle (canonical inputs `[e_xy, e_yz, e_xz]`, vars `{X, Y, Z}`):
//!
//! | Leader | Slot 0 | Slot 1 | Slot 2 | Lookup col-swaps    | Kernel-direct output |
//! |--------|--------|--------|--------|---------------------|----------------------|
//! | e_xy 0 | e_xy   | e_yz   | e_xz   | none                | (X, Y, Z)            |
//! | e_yz 1 | e_yz   | e_xz↔  | e_xy↔  | e_xz, e_xy          | (Y, Z, X)            |
//! | e_xz 2 | e_xz   | e_yz↔  | e_xy   | e_yz                | (X, Z, Y)            |
//!
//! 4-cycle (canonical inputs `[e_wx, e_xy, e_yz, e_zw]`,
//! vars `{W, X, Y, Z}`) — all rotation-only, no col-swaps:
//!
//! | Leader | Slot 0 | Slot 1 | Slot 2 | Slot 3 | Kernel-direct output |
//! |--------|--------|--------|--------|--------|----------------------|
//! | e_wx 0 | e_wx   | e_xy   | e_yz   | e_zw   | (W, X, Y, Z)         |
//! | e_xy 1 | e_xy   | e_yz   | e_zw   | e_wx   | (X, Y, Z, W)         |
//! | e_yz 2 | e_yz   | e_zw   | e_wx   | e_xy   | (Y, Z, W, X)         |
//! | e_zw 3 | e_zw   | e_wx   | e_xy   | e_yz   | (Z, W, X, Y)         |

use xlog_core::RelId;
use xlog_ir::rir::{LookupPerm, ProjectExpr, VariableOrder};
use xlog_stats::StatsManager;

use crate::compiler_config::{CompilerConfig, WcojVarOrderingKind};

/// Named K-clique WCOJ-vs-hash gate parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WcojCostGateParams {
    /// WCOJ routes only when estimated WCOJ work is no greater than
    /// this multiple of estimated hash-chain work. The equality
    /// boundary keeps the gate one-sided and mirrors the planner's
    /// lower-cost winner semantics.
    pub wcoj_to_hash_cost_ratio_ceiling: f64,
}

/// Default K-clique cost-gate policy.
pub const WCOJ_COST_GATE_PARAMS: WcojCostGateParams = WcojCostGateParams {
    wcoj_to_hash_cost_ratio_ceiling: 1.0,
};

/// Returns true when the named K-clique cost-gate policy routes WCOJ.
pub fn wcoj_cost_gate_predicts_wcoj(wcoj_cost: f64, hash_cost: f64) -> bool {
    if !wcoj_cost.is_finite() || !hash_cost.is_finite() || hash_cost <= 0.0 {
        return false;
    }
    wcoj_cost / hash_cost <= WCOJ_COST_GATE_PARAMS.wcoj_to_hash_cost_ratio_ceiling
}

/// Trait that picks a leader slot for triangle / 4-cycle WCOJ.
///
/// Implementations look at relation cardinalities (or other stats)
/// and decide whether a non-default leader is worth selecting. They
/// return `None` to mean "leave default leader" (i.e., slice 1/2
/// behavior).
pub trait WcojVariableOrderingModel {
    /// Pick a leader for the triangle WCOJ shape.
    ///
    /// `rel_ids` are in **promoter canonical order**:
    /// `[e_xy, e_yz, e_xz]`. Returns leader index in `[0, 3)` or
    /// `None` if the model declines to override the default.
    fn pick_triangle_leader(
        &self,
        rel_ids: [RelId; 3],
        stats: &StatsManager,
        config: &CompilerConfig,
    ) -> Option<u8>;

    /// Pick a leader for the 4-cycle WCOJ shape.
    ///
    /// `rel_ids` are in **promoter canonical order**:
    /// `[e_wx, e_xy, e_yz, e_zw]`. Returns leader index in
    /// `[0, 4)` or `None` if the model declines to override the
    /// default.
    fn pick_4cycle_leader(
        &self,
        rel_ids: [RelId; 4],
        stats: &StatsManager,
        config: &CompilerConfig,
    ) -> Option<u8>;
}

/// Default implementation: pick the min-cardinality input as leader,
/// gated by the configurable `wcoj_var_ordering_threshold` ratio.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeaderCardinalityModel;

impl LeaderCardinalityModel {
    /// Look up cardinality from `StatsManager`. Returns `None` for
    /// unregistered or unseeded relations — the caller treats `None`
    /// as "no useful stats, decline to reorder".
    fn card_of(stats: &StatsManager, rel: RelId) -> Option<u64> {
        let s = stats.get_relation_stats(rel)?;
        // 0 cardinality is treated as "no info" rather than "empty"
        // because the promoter runs at compile time when EDBs may
        // not yet be loaded; the dispatcher's runtime layout helper
        // handles truly empty inputs.
        if s.cardinality == 0 {
            None
        } else {
            Some(s.cardinality)
        }
    }

    /// Returns the slot index whose cardinality is minimum, or
    /// `None` if any input has missing/zero stats. This is the
    /// safety floor — partial stats degrade to default-leader rather
    /// than mis-picking.
    fn argmin_card<const N: usize>(
        rel_ids: &[RelId; N],
        stats: &StatsManager,
    ) -> Option<(u8, u64)> {
        let mut best: Option<(u8, u64)> = None;
        for (idx, rel) in rel_ids.iter().enumerate() {
            let card = Self::card_of(stats, *rel)?;
            best = match best {
                None => Some((idx as u8, card)),
                Some((_, b)) if card < b => Some((idx as u8, card)),
                Some(prev) => Some(prev),
            };
        }
        best
    }

    /// Apply the `Disabled`-config short-circuit and the threshold
    /// gate. Returns `Some(leader_idx)` only when the model is
    /// confident the non-default leader is worth the layout overhead.
    fn pick_leader<const N: usize>(
        &self,
        rel_ids: [RelId; N],
        stats: &StatsManager,
        config: &CompilerConfig,
    ) -> Option<u8> {
        if config.wcoj_variable_ordering == WcojVarOrderingKind::Disabled {
            return None;
        }
        let (min_idx, min_card) = Self::argmin_card(&rel_ids, stats)?;
        // Default leader is index 0 (e_xy / e_wx). If the argmin IS
        // index 0, no reordering would happen — return None to keep
        // the IR `var_order = None` (bit-identical to slice 1/2).
        if min_idx == 0 {
            return None;
        }
        let default_card = Self::card_of(stats, rel_ids[0])?;
        // Ratio gate: require min_card / default_card ≤ threshold.
        // Use float math for clarity; cardinalities fit comfortably
        // in f64 precision for the relation sizes the promoter sees.
        let ratio = min_card as f64 / default_card as f64;
        let threshold = config.effective_wcoj_var_ordering_threshold();
        if ratio <= threshold {
            Some(min_idx)
        } else {
            None
        }
    }
}

impl WcojVariableOrderingModel for LeaderCardinalityModel {
    fn pick_triangle_leader(
        &self,
        rel_ids: [RelId; 3],
        stats: &StatsManager,
        config: &CompilerConfig,
    ) -> Option<u8> {
        self.pick_leader(rel_ids, stats, config)
    }

    fn pick_4cycle_leader(
        &self,
        rel_ids: [RelId; 4],
        stats: &StatsManager,
        config: &CompilerConfig,
    ) -> Option<u8> {
        self.pick_leader(rel_ids, stats, config)
    }
}

// ---------------------------------------------------------------
// W2.6 — HeatAwareLeaderModel
// ---------------------------------------------------------------

/// W2.6 cost model: combines cardinality, access heat, and observed
/// join selectivity into a composite score per the locked plan
/// formula. Higher score = more expensive to iterate over → demote
/// from the leader (slot 0). The model picks `argmin(score)` as
/// leader, gated by the same `effective_wcoj_var_ordering_threshold`
/// as W2.1.
///
/// **Locked formula** (W2.6 plan §"Composite score"):
/// ```text
/// score(rel) = cardinality(rel)
///            * (1.0 + 4.0 * heat(rel))
///            * Σ_{e ∈ edges(rel)} 1.0 / max(0.01, observed_sel_or_one(e))
/// ```
///
/// * `cardinality` from `StatsManager::get_relation_stats(rel).cardinality`.
/// * `heat` from `RelationStats.heat: f32` (EMA written by
///   `record_access` at `node_dispatch.rs:26`).
/// * `observed_sel_or_one(e)` from
///   `StatsManager::get_join_selectivity(rel_a, rel_b)` (W2.4
///   output via `record_join_result`, EMA-smoothed).
///   **Key-validation**: only consume a cached `JoinSelectivity`
///   when its `(left_keys, right_keys)` match the candidate edge
///   keys after `StatsManager::canonical_join_key` swap. On
///   mismatch, default to `1.0` (no observed filter assumption).
///
/// **Heat weight 4.0** locked: with W2.1 default threshold 0.5,
/// gate fires when `min/default ≤ 0.5`. With cards equal +
/// `heat = h` on the hot rel, ratio = `1 / (1 + 4h)`. For
/// `1 + 4h ≥ 2.0` → `h ≥ 0.25` (~3 `record_access` calls).
#[derive(Debug, Clone, Copy, Default)]
pub struct HeatAwareLeaderModel;

impl HeatAwareLeaderModel {
    /// Locked heat weight (W2.6 plan §"Composite score").
    pub const HEAT_WEIGHT: f32 = 4.0;

    /// Default selectivity for an edge with no observed
    /// `JoinSelectivity` record (or with mismatched keys).
    pub const NO_OBSERVED_SEL: f64 = 1.0;

    /// Floor used in `1/max(0.01, sel)` to avoid divide-by-zero
    /// on tightly observed edges.
    pub const SEL_FLOOR: f64 = 0.01;

    /// Same `card_of` semantics as `LeaderCardinalityModel`:
    /// returns None for unregistered or zero-card rels (safety
    /// floor).
    fn card_of(stats: &StatsManager, rel: RelId) -> Option<u64> {
        let s = stats.get_relation_stats(rel)?;
        if s.cardinality == 0 {
            None
        } else {
            Some(s.cardinality)
        }
    }

    /// Read heat from the cached `RelationStats`. Returns 0.0 for
    /// unregistered rels — the formula treats unobserved heat as
    /// cold.
    fn heat_of(stats: &StatsManager, rel: RelId) -> f32 {
        stats.get_relation_stats(rel).map(|s| s.heat).unwrap_or(0.0)
    }

    /// Look up the observed selectivity for an edge with explicit
    /// key validation. The candidate keys describe how slot
    /// `pair.0` joins with slot `pair.1` in the canonical kernel
    /// topology; the cache key is canonicalized via
    /// `StatsManager::canonical_join_key` (which may swap the rel
    /// order). On a hit, the stored `left_keys` / `right_keys`
    /// MUST match the candidate keys after the same swap;
    /// otherwise return `NO_OBSERVED_SEL` (the model treats the
    /// edge as having no useful filter info).
    fn observed_sel_or_one(
        stats: &StatsManager,
        rel_a: RelId,
        rel_b: RelId,
        candidate_left_keys: &[usize],
        candidate_right_keys: &[usize],
    ) -> f64 {
        let Some(js) = stats.get_join_selectivity(rel_a, rel_b) else {
            return Self::NO_OBSERVED_SEL;
        };
        // `get_join_selectivity` canonicalizes internally so
        // `js.left_rel <= js.right_rel`. The candidate keys must
        // be swapped correspondingly when the stored canonical
        // order differs from the candidate's `(rel_a, rel_b)`
        // order.
        let (expected_left_keys, expected_right_keys) = if js.left_rel == rel_a {
            (candidate_left_keys, candidate_right_keys)
        } else {
            // Stored entry has rels swapped vs candidate; swap
            // expected keys correspondingly.
            (candidate_right_keys, candidate_left_keys)
        };
        if js.left_keys.as_slice() == expected_left_keys
            && js.right_keys.as_slice() == expected_right_keys
        {
            js.selectivity
        } else {
            Self::NO_OBSERVED_SEL
        }
    }

    /// Compute the locked composite score for a single rel.
    ///
    /// `incident_edges` is a slice of `(other_rel,
    /// candidate_left_keys, candidate_right_keys)` for each edge
    /// containing `rel`. The model derives this from the canonical
    /// kernel topology (triangle: 3 edges, 4-cycle: 4 edges); the
    /// key columns come from the locked permutation tables.
    fn composite_score(
        rel: RelId,
        card: u64,
        stats: &StatsManager,
        incident_edges: &[(RelId, &[usize], &[usize])],
    ) -> f64 {
        let heat = Self::heat_of(stats, rel);
        let heat_factor = 1.0 + Self::HEAT_WEIGHT as f64 * heat as f64;
        let sel_penalty: f64 = incident_edges
            .iter()
            .map(|(other, lk, rk)| {
                let sel = Self::observed_sel_or_one(stats, rel, *other, lk, rk);
                1.0 / sel.max(Self::SEL_FLOOR)
            })
            .sum();
        card as f64 * heat_factor * sel_penalty
    }

    /// Generic argmin over composite scores with the same
    /// short-circuits as `LeaderCardinalityModel::pick_leader`:
    /// `Disabled` config → None; argmin == 0 (default leader
    /// already optimal) → None; threshold-gate via
    /// `min_score / default_score ≤ effective_threshold` else
    /// None.
    fn pick_leader_from_scores<const N: usize>(
        rel_ids: [RelId; N],
        scores: [f64; N],
        config: &CompilerConfig,
    ) -> Option<u8> {
        if config.wcoj_variable_ordering == WcojVarOrderingKind::Disabled {
            return None;
        }
        // Defensive: if any score is non-finite or zero, treat as
        // missing data → safety floor returns None.
        if scores.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return None;
        }
        let _ = rel_ids; // future-proof param; unused today.
        let (min_idx, &min_score) = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        if min_idx == 0 {
            return None;
        }
        let default_score = scores[0];
        let ratio = min_score / default_score;
        let threshold = config.effective_wcoj_var_ordering_threshold();
        if ratio <= threshold {
            Some(min_idx as u8)
        } else {
            None
        }
    }
}

impl WcojVariableOrderingModel for HeatAwareLeaderModel {
    fn pick_triangle_leader(
        &self,
        rel_ids: [RelId; 3],
        stats: &StatsManager,
        config: &CompilerConfig,
    ) -> Option<u8> {
        // Triangle canonical edges: {(0,1), (1,2), (0,2)} —
        // each rel ∈ 2 edges. Edge keys per W2.1 locked
        // permutation tables: every triangle edge joins on
        // [1]/[0] in the **canonical** layout (no swaps in the
        // canonical default-leader topology). The HeatAware
        // model's selectivity lookup uses these canonical-layout
        // keys; non-default-leader rotated keys (W2.6 step 5
        // table) are only relevant on the FEEDBACK side.
        let cards: [u64; 3] = match (
            Self::card_of(stats, rel_ids[0]),
            Self::card_of(stats, rel_ids[1]),
            Self::card_of(stats, rel_ids[2]),
        ) {
            (Some(a), Some(b), Some(c)) => [a, b, c],
            _ => return None, // safety floor on missing card
        };
        // Edges: indexed by (slot_a, slot_b) per canonical topology.
        // edge01 = rel_ids[0] ⋈ rel_ids[1] on rel_ids[0].col1 ↔ rel_ids[1].col0
        // edge12 = rel_ids[1] ⋈ rel_ids[2] on rel_ids[1].col1 ↔ rel_ids[2].col0
        // edge02 = rel_ids[0] ⋈ rel_ids[2] on rel_ids[0].col0 ↔ rel_ids[2].col0
        //   (canonical triangle slot_vars[2] = [a, c] — keys [0]/[0]).
        let scores: [f64; 3] = [
            Self::composite_score(
                rel_ids[0],
                cards[0],
                stats,
                &[(rel_ids[1], &[1], &[0]), (rel_ids[2], &[0], &[0])],
            ),
            Self::composite_score(
                rel_ids[1],
                cards[1],
                stats,
                &[(rel_ids[0], &[0], &[1]), (rel_ids[2], &[1], &[1])],
            ),
            Self::composite_score(
                rel_ids[2],
                cards[2],
                stats,
                &[(rel_ids[0], &[0], &[0]), (rel_ids[1], &[1], &[1])],
            ),
        ];
        Self::pick_leader_from_scores(rel_ids, scores, config)
    }

    fn pick_4cycle_leader(
        &self,
        rel_ids: [RelId; 4],
        stats: &StatsManager,
        config: &CompilerConfig,
    ) -> Option<u8> {
        // 4-cycle canonical edges: {(0,1), (1,2), (2,3), (3,0)} —
        // each rel ∈ 2 edges. Rotation-only (no swaps); every
        // edge joins on [1]/[0] in canonical layout.
        let cards: [u64; 4] = match (
            Self::card_of(stats, rel_ids[0]),
            Self::card_of(stats, rel_ids[1]),
            Self::card_of(stats, rel_ids[2]),
            Self::card_of(stats, rel_ids[3]),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => [a, b, c, d],
            _ => return None,
        };
        let scores: [f64; 4] = [
            Self::composite_score(
                rel_ids[0],
                cards[0],
                stats,
                &[
                    (rel_ids[1], &[1], &[0]), // edge (0,1)
                    (rel_ids[3], &[0], &[1]), // edge (3,0) — reversed: rel0.col0 ↔ rel3.col1
                ],
            ),
            Self::composite_score(
                rel_ids[1],
                cards[1],
                stats,
                &[(rel_ids[0], &[0], &[1]), (rel_ids[2], &[1], &[0])],
            ),
            Self::composite_score(
                rel_ids[2],
                cards[2],
                stats,
                &[(rel_ids[1], &[0], &[1]), (rel_ids[3], &[1], &[0])],
            ),
            Self::composite_score(
                rel_ids[3],
                cards[3],
                stats,
                &[(rel_ids[2], &[0], &[1]), (rel_ids[0], &[1], &[0])],
            ),
        ];
        Self::pick_leader_from_scores(rel_ids, scores, config)
    }
}

/// Promoter helper: triangle locked permutation table.
///
/// Returns `lookup_perms` (the 2 non-leader slot entries) for the
/// given `leader_idx ∈ [0, 3)`. The promoter combines this with its
/// own knowledge of the rule head to produce
/// `kernel_output_cols`.
///
/// **Locked semantics** per W2.1 plan §"Permutation Tables":
/// * 0 (e_xy default) — slots `[e_yz, e_xz]`, no col-swaps.
/// * 1 (e_yz)         — slots `[e_xz↔, e_xy↔]`, both col-swapped.
/// * 2 (e_xz)         — slots `[e_yz↔, e_xy]`, e_yz col-swapped only.
pub fn triangle_lookup_perms(leader_idx: u8) -> Vec<LookupPerm> {
    match leader_idx {
        0 => vec![
            // slot 1 = e_yz (input_idx 1), slot 2 = e_xz (input_idx 2)
            LookupPerm {
                input_idx: 1,
                swap_cols: false,
            },
            LookupPerm {
                input_idx: 2,
                swap_cols: false,
            },
        ],
        1 => vec![
            // slot 1 = e_xz swapped (input_idx 2), slot 2 = e_xy swapped (input_idx 0)
            LookupPerm {
                input_idx: 2,
                swap_cols: true,
            },
            LookupPerm {
                input_idx: 0,
                swap_cols: true,
            },
        ],
        2 => vec![
            // slot 1 = e_yz swapped (input_idx 1), slot 2 = e_xy as-is (input_idx 0)
            LookupPerm {
                input_idx: 1,
                swap_cols: true,
            },
            LookupPerm {
                input_idx: 0,
                swap_cols: false,
            },
        ],
        _ => panic!("triangle leader_idx must be in [0, 3): got {leader_idx}"),
    }
}

/// Promoter helper: triangle locked head-projection table.
///
/// Maps the kernel-direct output (in leader's `(a, b, c)` order)
/// back to the canonical head order `(X, Y, Z)`.
///
/// **Locked semantics** per W2.1 plan §"Permutation Tables":
/// * 0 (e_xy default) — identity `[Column(0), Column(1), Column(2)]`.
/// * 1 (e_yz)         — `[Column(2), Column(0), Column(1)]`
///                      because kernel-direct = `(Y, Z, X)`.
/// * 2 (e_xz)         — `[Column(0), Column(2), Column(1)]`
///                      because kernel-direct = `(X, Z, Y)`.
pub fn triangle_kernel_output_cols(leader_idx: u8) -> Vec<ProjectExpr> {
    match leader_idx {
        0 => vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ],
        1 => vec![
            ProjectExpr::Column(2),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ],
        2 => vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(2),
            ProjectExpr::Column(1),
        ],
        _ => panic!("triangle leader_idx must be in [0, 3): got {leader_idx}"),
    }
}

/// Promoter helper: 4-cycle locked head-projection table.
///
/// Maps the kernel-direct output (in leader's rotated cycle order)
/// back to the canonical head order `(W, X, Y, Z)`. All 4 entries
/// are pure permutations; no col-swap is involved.
pub fn cycle4_kernel_output_cols(leader_idx: u8) -> Vec<ProjectExpr> {
    match leader_idx {
        0 => vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
        ],
        1 => vec![
            // kernel-direct = (X, Y, Z, W) → head (W, X, Y, Z)
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ],
        2 => vec![
            // kernel-direct = (Y, Z, W, X) → head (W, X, Y, Z)
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ],
        3 => vec![
            // kernel-direct = (Z, W, X, Y) → head (W, X, Y, Z)
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
        ],
        _ => panic!("4-cycle leader_idx must be in [0, 4): got {leader_idx}"),
    }
}

/// Promoter helper: build a complete `VariableOrder` for a triangle
/// from the cost model's `leader_idx` decision.
pub fn build_triangle_var_order(leader_idx: u8) -> VariableOrder {
    VariableOrder::legacy(
        leader_idx,
        triangle_lookup_perms(leader_idx),
        triangle_kernel_output_cols(leader_idx),
    )
}

/// Promoter helper: build a complete `VariableOrder` for a 4-cycle
/// from the cost model's `leader_idx` decision.
pub fn build_cycle4_var_order(leader_idx: u8) -> VariableOrder {
    VariableOrder::legacy(
        leader_idx,
        cycle4_lookup_perms(leader_idx),
        cycle4_kernel_output_cols(leader_idx),
    )
}

/// Promoter helper: 4-cycle locked permutation table.
///
/// All 4 leaders are rotation-only — no col-swaps. `lookup_perms`
/// rotates the input list so the leader is at slot 0 and subsequent
/// slots follow the cycle direction.
pub fn cycle4_lookup_perms(leader_idx: u8) -> Vec<LookupPerm> {
    if leader_idx >= 4 {
        panic!("4-cycle leader_idx must be in [0, 4): got {leader_idx}");
    }
    // Rotation: slot k holds input (leader_idx + k) mod 4.
    // The leader itself (k=0) is recorded in VariableOrder::leader_idx
    // and not duplicated here, so this returns the 3 non-leader slots.
    (1..4)
        .map(|k| LookupPerm {
            input_idx: ((leader_idx as usize + k) % 4) as u8,
            swap_cols: false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! W2.1 step 3 unit tests for the cost-model trait + default
    //! impl + locked permutation tables. End-to-end leader-pick
    //! tests live in W2.1 step 7 Part A.
    use super::*;
    use xlog_core::RelId;

    fn stats_with(cards: &[(RelId, u64)]) -> StatsManager {
        let mut mgr = StatsManager::new();
        for (rel, card) in cards {
            mgr.register_relation(*rel);
            if *card > 0 {
                mgr.update_cardinality(*rel, *card);
            }
        }
        mgr
    }

    #[test]
    fn disabled_config_returns_none_even_with_smaller_input() {
        let stats = stats_with(&[(RelId(1), 1000), (RelId(2), 10), (RelId(3), 1000)]);
        let config = CompilerConfig::default(); // Disabled
        let m = LeaderCardinalityModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            None,
            "Disabled config must short-circuit before the threshold gate"
        );
    }

    #[test]
    fn missing_stats_returns_none_safety_floor() {
        // RelId(2) has no cardinality registered (registration without
        // update_cardinality leaves card=0, treated as missing).
        let stats = stats_with(&[(RelId(1), 1000), (RelId(2), 0), (RelId(3), 1000)]);
        let config = CompilerConfig {
            wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
            ..CompilerConfig::default()
        };
        let m = LeaderCardinalityModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            None,
            "missing-stats safety floor: any None card → None decision"
        );
    }

    #[test]
    fn picks_min_when_clearly_under_threshold() {
        let stats = stats_with(&[(RelId(1), 1000), (RelId(2), 100), (RelId(3), 800)]);
        let config = CompilerConfig {
            wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
            ..CompilerConfig::default()
        };
        let m = LeaderCardinalityModel;
        // ratio = 100/1000 = 0.1 ≤ 0.5 → leader = 1
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            Some(1)
        );
    }

    #[test]
    fn declines_when_min_is_default_leader() {
        // RelId(1) is the default leader and is already the smallest.
        let stats = stats_with(&[(RelId(1), 100), (RelId(2), 1000), (RelId(3), 1000)]);
        let config = CompilerConfig {
            wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
            ..CompilerConfig::default()
        };
        let m = LeaderCardinalityModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            None,
            "min == default leader → no reorder needed → None"
        );
    }

    #[test]
    fn declines_marginal_above_threshold() {
        // ratio = 700/1000 = 0.7 > 0.5 → no reorder
        let stats = stats_with(&[(RelId(1), 1000), (RelId(2), 700), (RelId(3), 900)]);
        let config = CompilerConfig {
            wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
            ..CompilerConfig::default()
        };
        let m = LeaderCardinalityModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            None,
        );
    }

    #[test]
    fn picks_4cycle_min_under_threshold() {
        let stats = stats_with(&[
            (RelId(1), 1000),
            (RelId(2), 1000),
            (RelId(3), 100),
            (RelId(4), 1000),
        ]);
        let config = CompilerConfig {
            wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
            ..CompilerConfig::default()
        };
        let m = LeaderCardinalityModel;
        assert_eq!(
            m.pick_4cycle_leader([RelId(1), RelId(2), RelId(3), RelId(4)], &stats, &config),
            Some(2)
        );
    }

    #[test]
    fn triangle_lookup_perms_default_leader_is_identity() {
        let p = triangle_lookup_perms(0);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].input_idx, 1);
        assert!(!p[0].swap_cols);
        assert_eq!(p[1].input_idx, 2);
        assert!(!p[1].swap_cols);
    }

    #[test]
    fn triangle_lookup_perms_leader_yz_double_swap() {
        let p = triangle_lookup_perms(1);
        // slot 1 = e_xz swapped, slot 2 = e_xy swapped
        assert_eq!(p[0].input_idx, 2);
        assert!(p[0].swap_cols);
        assert_eq!(p[1].input_idx, 0);
        assert!(p[1].swap_cols);
    }

    #[test]
    fn triangle_lookup_perms_leader_xz_partial_swap() {
        let p = triangle_lookup_perms(2);
        // slot 1 = e_yz swapped, slot 2 = e_xy as-is
        assert_eq!(p[0].input_idx, 1);
        assert!(p[0].swap_cols);
        assert_eq!(p[1].input_idx, 0);
        assert!(!p[1].swap_cols);
    }

    #[test]
    fn cycle4_lookup_perms_default_is_identity_rotation() {
        let p = cycle4_lookup_perms(0);
        // slots 1, 2, 3 = inputs 1, 2, 3
        assert_eq!(
            p.iter().map(|lp| lp.input_idx).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(p.iter().all(|lp| !lp.swap_cols));
    }

    #[test]
    fn cycle4_lookup_perms_leader_xy_rotates_to_yz_zw_wx() {
        let p = cycle4_lookup_perms(1);
        // leader = e_xy. slots 1, 2, 3 = e_yz (2), e_zw (3), e_wx (0).
        assert_eq!(
            p.iter().map(|lp| lp.input_idx).collect::<Vec<_>>(),
            vec![2, 3, 0]
        );
        assert!(p.iter().all(|lp| !lp.swap_cols));
    }

    #[test]
    fn cycle4_lookup_perms_leader_zw_rotates_to_wx_xy_yz() {
        let p = cycle4_lookup_perms(3);
        assert_eq!(
            p.iter().map(|lp| lp.input_idx).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(p.iter().all(|lp| !lp.swap_cols));
    }

    // ===============================================================
    // W2.6 step 7 Part A — HeatAwareLeaderModel composite-score certs
    // ===============================================================

    /// Build stats with cards + per-rel heat values (heat injected
    /// directly via `get_relation_stats_mut`; sidesteps EMA so unit
    /// tests can pin exact values).
    fn stats_with_heat(seeded: &[(RelId, u64, f32)]) -> StatsManager {
        let mut mgr = StatsManager::new();
        for (rel, card, heat) in seeded {
            mgr.register_relation(*rel);
            if *card > 0 {
                mgr.update_cardinality(*rel, *card);
            }
            if let Some(s) = mgr.get_relation_stats_mut(*rel) {
                s.heat = *heat;
            }
        }
        mgr
    }

    fn heat_aware_config() -> CompilerConfig {
        CompilerConfig {
            wcoj_variable_ordering: WcojVarOrderingKind::HeatAware,
            ..CompilerConfig::default()
        }
    }

    #[test]
    fn heat_aware_leader_picks_cold_when_hot_relation_at_default_idx() {
        // Triangle. Cards equal at 100. Hot rel at idx 0 (default
        // leader): heat = 0.5. Other rels heat = 0.
        // Expected scores (per locked formula HEAT_WEIGHT = 4):
        //   idx0: 100 * (1+4*0.5) * 2 = 100 * 3 * 2 = 600
        //   idx1: 100 * 1 * 2 = 200
        //   idx2: 100 * 1 * 2 = 200
        // argmin = idx 1 (first hit). default = 600. ratio
        // 200/600 = 0.333 ≤ 0.5 → returns Some(1).
        let stats = stats_with_heat(&[
            (RelId(1), 100, 0.5),
            (RelId(2), 100, 0.0),
            (RelId(3), 100, 0.0),
        ]);
        let config = heat_aware_config();
        let m = HeatAwareLeaderModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            Some(1)
        );
    }

    #[test]
    fn heat_aware_leader_demotes_relation_in_tight_edge() {
        // Triangle. Cards 100 each. Heat 0 each. ONE
        // JoinSelectivity record on edge (rel(1), rel(2))
        // (canonical idx 0/1) with selectivity = 0.01.
        // Expected penalties:
        //   rel0 ∈ {(0,1) tight, (0,2) default}: 100 + 1 = 101.
        //   rel1 ∈ {(0,1) tight, (1,2) default}: 101.
        //   rel2 ∈ {(1,2) default, (0,2) default}: 1 + 1 = 2.
        // Scores (heat factor 1):
        //   idx0: 100 * 101 = 10100
        //   idx1: 10100
        //   idx2: 100 * 2 = 200
        // argmin = idx 2. default = 10100. ratio 200/10100
        // ≈ 0.0198 ≤ 0.5 → returns Some(2).
        let mut stats = stats_with_heat(&[
            (RelId(1), 100, 0.0),
            (RelId(2), 100, 0.0),
            (RelId(3), 100, 0.0),
        ]);
        // Tight edge on (rel0=RelId(1), rel1=RelId(2)) with keys
        // [1]/[0] (canonical triangle slot_vars[0]=[a,b],
        // slot_vars[1]=[b,c] → join key b at slot0.col1 / slot1.col0).
        stats.set_join_selectivity(RelId(1), RelId(2), vec![1], vec![0], 0.01);
        let config = heat_aware_config();
        let m = HeatAwareLeaderModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            Some(2)
        );
    }

    #[test]
    fn heat_aware_leader_returns_none_when_heat_too_low() {
        // Cards equal. Heat at idx 0 = 0.1 (one record_access from
        // cold), others 0. Factor at idx 0 = 1.4, others 1.0.
        // Score idx0 = 100*1.4*2 = 280. idx1/idx2 = 200.
        // ratio 200/280 ≈ 0.714 > 0.5 → returns None.
        let stats = stats_with_heat(&[
            (RelId(1), 100, 0.1),
            (RelId(2), 100, 0.0),
            (RelId(3), 100, 0.0),
        ]);
        let config = heat_aware_config();
        let m = HeatAwareLeaderModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            None
        );
    }

    #[test]
    fn heat_aware_leader_disabled_short_circuit() {
        // Strong heat differential, but config = Disabled → None.
        let stats = stats_with_heat(&[
            (RelId(1), 100, 0.9),
            (RelId(2), 100, 0.0),
            (RelId(3), 100, 0.0),
        ]);
        let config = CompilerConfig::default(); // Disabled
        let m = HeatAwareLeaderModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            None
        );
    }

    #[test]
    fn heat_aware_leader_missing_card_safety_floor() {
        // RelId(2) has no card registered (zero). Cost model
        // returns None per missing-card safety floor (matches
        // LeaderCardinalityModel's behavior).
        let stats = stats_with_heat(&[
            (RelId(1), 100, 0.5),
            (RelId(2), 0, 0.0), // missing card
            (RelId(3), 100, 0.0),
        ]);
        let config = heat_aware_config();
        let m = HeatAwareLeaderModel;
        assert_eq!(
            m.pick_triangle_leader([RelId(1), RelId(2), RelId(3)], &stats, &config),
            None
        );
    }
}
