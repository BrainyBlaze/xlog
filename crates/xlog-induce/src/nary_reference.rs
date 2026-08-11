//! Host-side reference scorer for n-ary rule patterns.
//!
//! The parity anchor for the device n-ary scoring stage: a direct,
//! obviously-correct interpretation of [`NaryRulePattern`] coverage
//! semantics over host fact tables. The device kernel, when it lands, must
//! reproduce these counts bit-for-bit on bounded inputs — exactly the role
//! the Python prototype played for the binary engine.
//!
//! Coverage semantics, stated once: an example tuple (one head assignment)
//! is covered by a pattern iff there exists an assignment of the pattern's
//! join variables such that every body atom's bound row exists in its
//! candidate relation. Head positions are fixed by the example; join
//! variables are searched. The search is a plain backtracking walk — the
//! reference optimizes for auditability, not speed.

use crate::nary::{BodyAtomPattern, NaryRulePattern, PatternVar};

/// One candidate relation's facts as host rows; row length is the arity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostRelation {
    pub rows: Vec<Vec<u64>>,
}

/// Coverage counts for one pattern over one example set pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceCoverage {
    pub positives_covered: u32,
    pub negatives_covered: u32,
}

/// Score one pattern against positive and negative example tuples.
///
/// `candidates` is indexed by the pattern's `candidate_slot` values; each
/// example tuple's length must equal the pattern's head arity. The caller
/// validates the pattern first ([`NaryRulePattern::validate`]) — this
/// interpreter assumes canonical form and panics only on indexing bugs,
/// never on data.
pub fn score_pattern_reference(
    pattern: &NaryRulePattern,
    candidates: &[HostRelation],
    positives: &[Vec<u64>],
    negatives: &[Vec<u64>],
) -> ReferenceCoverage {
    let count = |examples: &[Vec<u64>]| -> u32 {
        examples
            .iter()
            .filter(|example| covers(pattern, candidates, example))
            .count() as u32
    };
    ReferenceCoverage {
        positives_covered: count(positives),
        negatives_covered: count(negatives),
    }
}

/// Does the pattern cover one head assignment?
fn covers(pattern: &NaryRulePattern, candidates: &[HostRelation], example: &[u64]) -> bool {
    let join_count = join_variable_count(pattern);
    let mut joins: Vec<Option<u64>> = vec![None; join_count];
    satisfy(&pattern.body, candidates, example, &mut joins)
}

/// Backtracking satisfaction over the remaining body atoms.
fn satisfy(
    body: &[BodyAtomPattern],
    candidates: &[HostRelation],
    example: &[u64],
    joins: &mut Vec<Option<u64>>,
) -> bool {
    let Some((atom, rest)) = body.split_first() else {
        return true;
    };
    let relation = &candidates[atom.candidate_slot as usize];
    'rows: for row in &relation.rows {
        debug_assert_eq!(row.len(), atom.bindings.len());
        // Check the row against fixed positions, collecting the join
        // variables this row would newly bind so they can be undone.
        let mut newly_bound: Vec<u8> = Vec::new();
        for (value, binding) in row.iter().zip(&atom.bindings) {
            match *binding {
                PatternVar::Head(i) => {
                    if example[i as usize] != *value {
                        undo(joins, &newly_bound);
                        continue 'rows;
                    }
                }
                PatternVar::Join(j) => match joins[j as usize] {
                    Some(bound) => {
                        if bound != *value {
                            undo(joins, &newly_bound);
                            continue 'rows;
                        }
                    }
                    None => {
                        joins[j as usize] = Some(*value);
                        newly_bound.push(j);
                    }
                },
            }
        }
        if satisfy(rest, candidates, example, joins) {
            return true;
        }
        undo(joins, &newly_bound);
    }
    false
}

fn undo(joins: &mut [Option<u64>], newly_bound: &[u8]) {
    for j in newly_bound {
        joins[*j as usize] = None;
    }
}

fn join_variable_count(pattern: &NaryRulePattern) -> usize {
    let mut count = 0usize;
    for atom in &pattern.body {
        for binding in &atom.bindings {
            if let PatternVar::Join(j) = *binding {
                count = count.max(j as usize + 1);
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nary::canonical_binary_pattern;
    use crate::types::Topology;

    fn pairs(rows: &[(u64, u64)]) -> HostRelation {
        HostRelation {
            rows: rows.iter().map(|(a, b)| vec![*a, *b]).collect(),
        }
    }

    /// The exact hand-computed fixture the CUDA kernel test pins
    /// (`ilp_exact.rs::ilp_exact_score_matches_hand_computed_fixture`):
    /// candidates p_B={(1,2),(2,3)}, p_C={(2,4),(3,5),(4,6)}, positives
    /// {(1,4),(2,5)}, negatives {(7,8)}. Only chain(p_B, p_C) covers, and it
    /// covers both positives (joins z=2 and z=3); every other
    /// (topology, L, R) combination covers nothing, negatives included.
    #[test]
    fn reference_matches_shipped_binary_kernel_fixture() {
        let p_b = pairs(&[(1, 2), (2, 3)]);
        let p_c = pairs(&[(2, 4), (3, 5), (4, 6)]);
        let candidates = vec![p_b, p_c];
        let positives = vec![vec![1u64, 4u64], vec![2, 5]];
        let negatives = vec![vec![7u64, 8u64]];

        let mut nonzero = Vec::new();
        for topology in Topology::ALL {
            for left in 0..2u32 {
                for right in 0..2u32 {
                    let pattern = canonical_binary_pattern(topology, left, right);
                    let coverage =
                        score_pattern_reference(&pattern, &candidates, &positives, &negatives);
                    assert_eq!(
                        coverage.negatives_covered, 0,
                        "{topology:?}({left},{right}) covered a negative"
                    );
                    if coverage.positives_covered > 0 {
                        nonzero.push((topology, left, right, coverage.positives_covered));
                    }
                }
            }
        }
        assert_eq!(
            nonzero,
            vec![(Topology::Chain, 0, 1, 2)],
            "coverage disagrees with the shipped kernel fixture"
        );
    }

    /// Ternary head with a two-atom body sharing one join variable —
    /// hand-computed. H(x0,x1,x2) :- T(x0,x1,z0), P(z0,x2) with
    /// T={(1,2,9),(4,5,8)}, P={(9,3),(8,7)}: (1,2,3) covers via z0=9,
    /// (4,5,6) fails (P has (8,7) not (8,6)), (1,2,7) fails (z0=9 forces
    /// P(9,7) which is absent — the join must be consistent across atoms).
    #[test]
    fn ternary_head_join_consistency_is_enforced() {
        use crate::nary::{BodyAtomPattern, NaryRulePattern};
        use PatternVar::{Head, Join};
        let pattern = NaryRulePattern {
            head_arity: 3,
            body: vec![
                BodyAtomPattern {
                    candidate_slot: 0,
                    bindings: vec![Head(0), Head(1), Join(0)],
                },
                BodyAtomPattern {
                    candidate_slot: 1,
                    bindings: vec![Join(0), Head(2)],
                },
            ],
        };
        let ternary = HostRelation {
            rows: vec![vec![1, 2, 9], vec![4, 5, 8]],
        };
        let binary = pairs(&[(9, 3), (8, 7)]);
        let candidates = vec![ternary, binary];
        let positives = vec![vec![1u64, 2, 3], vec![4, 5, 6], vec![1, 2, 7]];
        let coverage = score_pattern_reference(&pattern, &candidates, &positives, &[]);
        assert_eq!(coverage.positives_covered, 1);
    }

    /// A join variable appearing in a single atom is a don't-care position
    /// (the Fanout/Fanin shape): any row value satisfies it.
    #[test]
    fn single_occurrence_join_is_a_dont_care_position() {
        let pattern = canonical_binary_pattern(Topology::Fanout, 0, 1);
        // L(X,Z), R(X,Y) — L's second column can be anything.
        let l = pairs(&[(1, 999)]);
        let r = pairs(&[(1, 5)]);
        let coverage = score_pattern_reference(&pattern, &[l, r], &[vec![1u64, 5u64]], &[]);
        assert_eq!(coverage.positives_covered, 1);
    }

    /// Backtracking must try later rows of an earlier atom when the first
    /// binding dead-ends: T={(1,8),(1,9)}, P={(9,2)} — z0=8 fails P, z0=9
    /// succeeds. A greedy first-row-only walk would miss the cover.
    #[test]
    fn backtracking_revisits_earlier_atom_rows() {
        use crate::nary::{BodyAtomPattern, NaryRulePattern};
        use PatternVar::{Head, Join};
        let pattern = NaryRulePattern {
            head_arity: 2,
            body: vec![
                BodyAtomPattern {
                    candidate_slot: 0,
                    bindings: vec![Head(0), Join(0)],
                },
                BodyAtomPattern {
                    candidate_slot: 1,
                    bindings: vec![Join(0), Head(1)],
                },
            ],
        };
        let t = pairs(&[(1, 8), (1, 9)]);
        let p = pairs(&[(9, 2)]);
        let coverage = score_pattern_reference(&pattern, &[t, p], &[vec![1u64, 2u64]], &[]);
        assert_eq!(coverage.positives_covered, 1);
    }
}
