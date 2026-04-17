//! Deterministic per-topology top-K reduction and tie diagnostics.
//!
//! Behaviorally equivalent to the Python reference in
//! `crates/pyxlog/python/pyxlog/ilp/exact_induce.py`:
//!
//! 1. Within each topology, sort scored pairs by lexicographic key
//!    `(-positives_covered, negatives_covered, left_idx, right_idx)`.
//! 2. Filter to pairs with `positives_covered > 0`, then keep the first
//!    `k_per_topology`.
//! 3. For each kept pair:
//!    - `local_rank` = 0-indexed position within the positive-filtered list.
//!    - `next_positives_covered` / `next_negatives_covered` = the next pair
//!      in the positive-filtered list, or `(0, 0)` if there is none.
//!    - `tie_class_size` = count of pairs in the FULL sorted list (including
//!      zero-coverage) sharing the same `(positives_covered, negatives_covered)`.
//! 4. Output groups candidates by `Topology::ALL` order.

use crate::types::{ScoredCandidate, Topology};
use xlog_core::RelId;

/// One scored `(topology, left, right)` triple produced by the scoring stage.
///
/// Passed into [`reduce_per_topology`] as a flat list; grouping happens inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoredPair {
    pub topology: Topology,
    pub left_rel_idx: RelId,
    pub right_rel_idx: RelId,
    pub positives_covered: u32,
    pub negatives_covered: u32,
}

/// Reduce a flat scored-pair list to the final ordered `ScoredCandidate` list.
///
/// Matches the Python reference comparator and diagnostics bit-for-bit.
pub fn reduce_per_topology(
    scored_pairs: &[ScoredPair],
    head_rel_idx: RelId,
    k_per_topology: u32,
) -> Vec<ScoredCandidate> {
    let mut result = Vec::new();
    let k = k_per_topology as usize;

    for topology in Topology::ALL {
        // Pull this topology's pairs into a sortable vector.
        let mut this_topo: Vec<ScoredPair> = scored_pairs
            .iter()
            .copied()
            .filter(|p| p.topology == topology)
            .collect();

        // Lexicographic sort: max positives, min negatives, min L idx, min R idx.
        this_topo.sort_by(|a, b| {
            (
                std::cmp::Reverse(a.positives_covered),
                a.negatives_covered,
                a.left_rel_idx.0,
                a.right_rel_idx.0,
            )
                .cmp(&(
                    std::cmp::Reverse(b.positives_covered),
                    b.negatives_covered,
                    b.left_rel_idx.0,
                    b.right_rel_idx.0,
                ))
        });

        // Positive-coverage filter preserves sorted order.
        let positives: Vec<ScoredPair> = this_topo
            .iter()
            .copied()
            .filter(|p| p.positives_covered > 0)
            .collect();

        let kept_n = std::cmp::min(k, positives.len());
        for (rank, pair) in positives.iter().take(kept_n).enumerate() {
            // Diagnostics: next candidate in the positive-filtered list.
            let (next_pos, next_neg) = positives
                .get(rank + 1)
                .map(|nxt| (nxt.positives_covered, nxt.negatives_covered))
                .unwrap_or((0, 0));

            // Tie class counted over the FULL sorted list (including zero-coverage).
            let tie_count = this_topo
                .iter()
                .filter(|s| {
                    s.positives_covered == pair.positives_covered
                        && s.negatives_covered == pair.negatives_covered
                })
                .count() as u32;

            result.push(ScoredCandidate {
                topology,
                head_rel_idx,
                left_rel_idx: pair.left_rel_idx,
                right_rel_idx: pair.right_rel_idx,
                positives_covered: pair.positives_covered,
                negatives_covered: pair.negatives_covered,
                local_rank: rank as u32,
                next_positives_covered: next_pos,
                next_negatives_covered: next_neg,
                tie_class_size: tie_count,
            });
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(topo: Topology, l: u32, r: u32, pos: u32, neg: u32) -> ScoredPair {
        ScoredPair {
            topology: topo,
            left_rel_idx: RelId(l),
            right_rel_idx: RelId(r),
            positives_covered: pos,
            negatives_covered: neg,
        }
    }

    const HEAD: RelId = RelId(100);

    #[test]
    fn empty_input_yields_empty_output() {
        let result = reduce_per_topology(&[], HEAD, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn zero_k_yields_empty_output() {
        let pairs = vec![pair(Topology::Chain, 1, 2, 5, 0)];
        let result = reduce_per_topology(&pairs, HEAD, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn zero_coverage_pairs_are_excluded_from_kept() {
        let pairs = vec![
            pair(Topology::Chain, 1, 2, 0, 0),
            pair(Topology::Chain, 3, 4, 0, 0),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn single_positive_pair_has_tie_one_and_no_next() {
        let pairs = vec![pair(Topology::Chain, 1, 2, 5, 0)];
        let result = reduce_per_topology(&pairs, HEAD, 2);
        assert_eq!(result.len(), 1);
        let c = &result[0];
        assert_eq!(c.topology, Topology::Chain);
        assert_eq!(c.left_rel_idx, RelId(1));
        assert_eq!(c.right_rel_idx, RelId(2));
        assert_eq!(c.positives_covered, 5);
        assert_eq!(c.negatives_covered, 0);
        assert_eq!(c.local_rank, 0);
        assert_eq!(c.next_positives_covered, 0);
        assert_eq!(c.next_negatives_covered, 0);
        assert_eq!(c.tie_class_size, 1);
    }

    #[test]
    fn max_positives_wins_over_higher_negatives() {
        // pair A: pos=5, neg=2; pair B: pos=3, neg=0 → A wins despite more negatives.
        let pairs = vec![
            pair(Topology::Chain, 1, 2, 3, 0),
            pair(Topology::Chain, 3, 4, 5, 2),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].positives_covered, 5);
        assert_eq!(result[0].negatives_covered, 2);
    }

    #[test]
    fn min_negatives_breaks_positives_tie() {
        // Same positives — lower negatives wins.
        let pairs = vec![
            pair(Topology::Chain, 1, 2, 5, 3),
            pair(Topology::Chain, 3, 4, 5, 1),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].left_rel_idx, RelId(3));
        assert_eq!(result[0].negatives_covered, 1);
    }

    #[test]
    fn left_idx_breaks_pos_neg_tie() {
        // Same positives+negatives — lower left idx wins.
        let pairs = vec![
            pair(Topology::Chain, 5, 2, 3, 0),
            pair(Topology::Chain, 1, 2, 3, 0),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].left_rel_idx, RelId(1));
    }

    #[test]
    fn right_idx_breaks_all_other_ties() {
        // Same pos, neg, left — lower right idx wins.
        let pairs = vec![
            pair(Topology::Chain, 1, 5, 3, 0),
            pair(Topology::Chain, 1, 2, 3, 0),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].right_rel_idx, RelId(2));
    }

    #[test]
    fn top_k_truncation_preserves_order() {
        // Three positive pairs, K=2 → keep the top two.
        let pairs = vec![
            pair(Topology::Chain, 1, 2, 3, 0),
            pair(Topology::Chain, 1, 3, 5, 0),
            pair(Topology::Chain, 1, 4, 4, 0),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].positives_covered, 5);
        assert_eq!(result[0].local_rank, 0);
        assert_eq!(result[1].positives_covered, 4);
        assert_eq!(result[1].local_rank, 1);
    }

    #[test]
    fn next_diagnostics_point_to_rank_plus_one_in_positive_list() {
        // Three positives; K=2. Top-2 get next_* from positions 1 and 2.
        let pairs = vec![
            pair(Topology::Chain, 1, 1, 5, 0),  // rank 0
            pair(Topology::Chain, 2, 2, 4, 0),  // rank 1
            pair(Topology::Chain, 3, 3, 3, 0),  // rank 2
        ];
        let result = reduce_per_topology(&pairs, HEAD, 2);
        assert_eq!(result.len(), 2);
        // rank 0's next_* is rank 1's (pos=4, neg=0)
        assert_eq!(result[0].next_positives_covered, 4);
        assert_eq!(result[0].next_negatives_covered, 0);
        // rank 1's next_* is rank 2's (pos=3, neg=0)
        assert_eq!(result[1].next_positives_covered, 3);
        assert_eq!(result[1].next_negatives_covered, 0);
    }

    #[test]
    fn next_diagnostics_are_zero_when_no_next() {
        // Only one positive — next_* should be (0, 0).
        let pairs = vec![pair(Topology::Chain, 1, 1, 5, 0)];
        let result = reduce_per_topology(&pairs, HEAD, 2);
        assert_eq!(result[0].next_positives_covered, 0);
        assert_eq!(result[0].next_negatives_covered, 0);
    }

    #[test]
    fn tie_class_size_counts_same_pos_neg_in_full_sorted_list() {
        // Three pairs share (pos=5, neg=0); one has different (pos=3, neg=0).
        let pairs = vec![
            pair(Topology::Chain, 1, 1, 5, 0),
            pair(Topology::Chain, 2, 2, 5, 0),
            pair(Topology::Chain, 3, 3, 5, 0),
            pair(Topology::Chain, 4, 4, 3, 0),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 2);
        // Top-2 should both have tie_class_size=3 (three pairs share pos=5, neg=0).
        assert_eq!(result[0].tie_class_size, 3);
        assert_eq!(result[1].tie_class_size, 3);
    }

    #[test]
    fn topologies_output_in_all_order() {
        // One positive per topology; check output order = chain, star, fanout, fanin.
        let pairs = vec![
            pair(Topology::Fanin, 1, 1, 1, 0),
            pair(Topology::Fanout, 1, 1, 1, 0),
            pair(Topology::Star, 1, 1, 1, 0),
            pair(Topology::Chain, 1, 1, 1, 0),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 1);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].topology, Topology::Chain);
        assert_eq!(result[1].topology, Topology::Star);
        assert_eq!(result[2].topology, Topology::Fanout);
        assert_eq!(result[3].topology, Topology::Fanin);
    }

    #[test]
    fn topology_filtering_is_per_topology() {
        // Pairs from both chain and star; ranking happens independently.
        let pairs = vec![
            pair(Topology::Chain, 1, 1, 3, 0),
            pair(Topology::Chain, 2, 2, 5, 0),
            pair(Topology::Star, 3, 3, 2, 0),
            pair(Topology::Star, 4, 4, 4, 0),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 1);
        assert_eq!(result.len(), 2);
        // chain winner: pos=5
        assert_eq!(result[0].topology, Topology::Chain);
        assert_eq!(result[0].positives_covered, 5);
        // star winner: pos=4
        assert_eq!(result[1].topology, Topology::Star);
        assert_eq!(result[1].positives_covered, 4);
    }

    #[test]
    fn k_larger_than_positives_returns_all_positives() {
        let pairs = vec![
            pair(Topology::Chain, 1, 1, 5, 0),
            pair(Topology::Chain, 2, 2, 3, 0),
        ];
        let result = reduce_per_topology(&pairs, HEAD, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn head_rel_idx_propagates_to_every_candidate() {
        let pairs = vec![
            pair(Topology::Chain, 1, 1, 5, 0),
            pair(Topology::Star, 2, 2, 3, 0),
        ];
        let head = RelId(42);
        let result = reduce_per_topology(&pairs, head, 1);
        for c in &result {
            assert_eq!(c.head_rel_idx, head);
        }
    }
}
