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
    VariableOrder {
        leader_idx,
        lookup_perms: triangle_lookup_perms(leader_idx),
        kernel_output_cols: triangle_kernel_output_cols(leader_idx),
    }
}

/// Promoter helper: build a complete `VariableOrder` for a 4-cycle
/// from the cost model's `leader_idx` decision.
pub fn build_cycle4_var_order(leader_idx: u8) -> VariableOrder {
    VariableOrder {
        leader_idx,
        lookup_perms: cycle4_lookup_perms(leader_idx),
        kernel_output_cols: cycle4_kernel_output_cols(leader_idx),
    }
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
}
