//! Flat device encoding of n-ary rule patterns + the iterative scorer.
//!
//! The CUDA n-ary scoring kernel cannot walk `Vec<BodyAtomPattern>` — it
//! consumes the pattern batch as parallel flat arrays. This module owns
//! that encoding AND `score_pattern_flat`, an iterative backtracking
//! interpreter over the encoding that is the kernel's algorithm stated in
//! host Rust: same state (row cursors, join values, per-depth bound
//! masks), same order, no recursion. Its tests pin it against the
//! recursive [`crate::nary_reference`] scorer, so when the device kernel
//! reproduces this walk, agreement with the reference follows by
//! transitivity — the CUDA leg then only has to witness bit-equality on
//! the pod.
//!
//! Bounds are part of the device contract: the kernel allocates fixed
//! per-thread state, so flattening REFUSES (typed, host-side) any pattern
//! the kernel could not evaluate. Nothing out of bounds ever reaches a
//! launch.
//!
//! Binding code layout (u32): bit 31 set => join variable, clear => head
//! position; low 8 bits carry the index. The remaining bits are zero and
//! reserved.

use crate::nary::{NaryRulePattern, PatternVar};

/// Device-contract bounds for one pattern evaluation thread.
pub const NARY_MAX_BODY_ATOMS: usize = 8;
pub const NARY_MAX_JOIN_VARS: usize = 8;
pub const NARY_MAX_ATOM_ARITY: usize = 8;

const JOIN_FLAG: u32 = 1 << 31;

/// Typed refusal: the pattern batch cannot be represented on device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NaryLayoutError {
    EmptyBatch,
    EmptyBody {
        pattern: usize,
    },
    TooManyBodyAtoms {
        pattern: usize,
        atoms: usize,
    },
    AtomArityOutOfRange {
        pattern: usize,
        atom: usize,
        arity: usize,
    },
    JoinIndexOutOfRange {
        pattern: usize,
        atom: usize,
        join: u8,
    },
    HeadIndexOutOfRange {
        pattern: usize,
        atom: usize,
        head: u8,
    },
}

impl std::fmt::Display for NaryLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBatch => write!(f, "pattern batch is empty"),
            Self::EmptyBody { pattern } => {
                write!(f, "pattern {pattern} has an empty body")
            }
            Self::TooManyBodyAtoms { pattern, atoms } => write!(
                f,
                "pattern {pattern} has {atoms} body atoms; device bound is \
                 {NARY_MAX_BODY_ATOMS}"
            ),
            Self::AtomArityOutOfRange {
                pattern,
                atom,
                arity,
            } => write!(
                f,
                "pattern {pattern} atom {atom} arity {arity} outside \
                 1..={NARY_MAX_ATOM_ARITY}"
            ),
            Self::JoinIndexOutOfRange {
                pattern,
                atom,
                join,
            } => write!(
                f,
                "pattern {pattern} atom {atom} join index {join} >= device \
                 bound {NARY_MAX_JOIN_VARS}"
            ),
            Self::HeadIndexOutOfRange {
                pattern,
                atom,
                head,
            } => write!(
                f,
                "pattern {pattern} atom {atom} head index {head} >= the \
                 pattern's head arity"
            ),
        }
    }
}

impl std::error::Error for NaryLayoutError {}

/// One pattern batch as the parallel flat arrays the kernel consumes.
///
/// Per pattern `p`: head arity `head_arity[p]`, join-variable count
/// `join_count[p]`, and body atoms `body_offset[p] .. body_offset[p] +
/// body_len[p]`. Per atom `a` (flat index): candidate relation slot
/// `atom_candidate_slot[a]`, arity `atom_arity[a]`, and binding codes
/// `binding_codes[atom_binding_offset[a] .. + atom_arity[a]]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaryPatternBatchLayout {
    pub head_arity: Vec<u32>,
    pub join_count: Vec<u32>,
    pub body_offset: Vec<u32>,
    pub body_len: Vec<u32>,
    pub atom_candidate_slot: Vec<u32>,
    pub atom_arity: Vec<u32>,
    pub atom_binding_offset: Vec<u32>,
    pub binding_codes: Vec<u32>,
}

/// Encode one binding as the device u32 code.
pub fn binding_code(var: PatternVar) -> u32 {
    match var {
        PatternVar::Head(i) => u32::from(i),
        PatternVar::Join(j) => JOIN_FLAG | u32::from(j),
    }
}

/// Decode a device binding code (inverse of [`binding_code`]).
pub fn decode_binding(code: u32) -> PatternVar {
    if code & JOIN_FLAG != 0 {
        PatternVar::Join((code & 0xFF) as u8)
    } else {
        PatternVar::Head((code & 0xFF) as u8)
    }
}

/// Flatten a validated pattern batch into the device layout.
///
/// The caller has already established canonical form per pattern
/// ([`NaryRulePattern::validate`]); this function enforces only the
/// DEVICE bounds and refuses anything the kernel could not evaluate.
pub fn flatten_patterns(
    patterns: &[NaryRulePattern],
) -> Result<NaryPatternBatchLayout, NaryLayoutError> {
    if patterns.is_empty() {
        return Err(NaryLayoutError::EmptyBatch);
    }
    let mut layout = NaryPatternBatchLayout {
        head_arity: Vec::with_capacity(patterns.len()),
        join_count: Vec::with_capacity(patterns.len()),
        body_offset: Vec::with_capacity(patterns.len()),
        body_len: Vec::with_capacity(patterns.len()),
        atom_candidate_slot: Vec::new(),
        atom_arity: Vec::new(),
        atom_binding_offset: Vec::new(),
        binding_codes: Vec::new(),
    };
    for (p, pattern) in patterns.iter().enumerate() {
        if pattern.body.is_empty() {
            return Err(NaryLayoutError::EmptyBody { pattern: p });
        }
        if pattern.body.len() > NARY_MAX_BODY_ATOMS {
            return Err(NaryLayoutError::TooManyBodyAtoms {
                pattern: p,
                atoms: pattern.body.len(),
            });
        }
        let mut joins = 0u32;
        layout.head_arity.push(u32::from(pattern.head_arity));
        layout
            .body_offset
            .push(layout.atom_candidate_slot.len() as u32);
        layout.body_len.push(pattern.body.len() as u32);
        for (a, atom) in pattern.body.iter().enumerate() {
            let arity = atom.bindings.len();
            if arity == 0 || arity > NARY_MAX_ATOM_ARITY {
                return Err(NaryLayoutError::AtomArityOutOfRange {
                    pattern: p,
                    atom: a,
                    arity,
                });
            }
            layout.atom_candidate_slot.push(atom.candidate_slot);
            layout.atom_arity.push(arity as u32);
            layout
                .atom_binding_offset
                .push(layout.binding_codes.len() as u32);
            for binding in &atom.bindings {
                match *binding {
                    PatternVar::Head(i) => {
                        if u32::from(i) >= u32::from(pattern.head_arity) {
                            return Err(NaryLayoutError::HeadIndexOutOfRange {
                                pattern: p,
                                atom: a,
                                head: i,
                            });
                        }
                    }
                    PatternVar::Join(j) => {
                        if usize::from(j) >= NARY_MAX_JOIN_VARS {
                            return Err(NaryLayoutError::JoinIndexOutOfRange {
                                pattern: p,
                                atom: a,
                                join: j,
                            });
                        }
                        joins = joins.max(u32::from(j) + 1);
                    }
                }
                layout.binding_codes.push(binding_code(*binding));
            }
        }
        layout.join_count.push(joins);
    }
    Ok(layout)
}

/// One candidate relation as the flat row-major buffer the kernel reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatRelation {
    pub arity: u32,
    pub row_count: u32,
    /// Row-major values; length == `arity * row_count`.
    pub values: Vec<u64>,
}

impl FlatRelation {
    pub fn from_rows(rows: &[Vec<u64>], arity: u32) -> Self {
        let mut values = Vec::with_capacity(rows.len() * arity as usize);
        for row in rows {
            assert_eq!(row.len(), arity as usize, "row arity mismatch");
            values.extend_from_slice(row);
        }
        Self {
            arity,
            row_count: rows.len() as u32,
            values,
        }
    }
}

/// Score ONE pattern of the batch against one example tuple — the exact
/// iterative walk the device kernel runs, in host Rust.
///
/// State per depth (= body atom index): the row cursor and the bitmask of
/// join variables that depth's row newly bound. Join VALUES live in one
/// array; a value is meaningful only while its bit is set in `bound`.
/// Backtracking clears the depth's mask and advances its cursor — never
/// touching values bound by shallower depths.
pub fn score_pattern_flat(
    layout: &NaryPatternBatchLayout,
    pattern_index: usize,
    candidates: &[FlatRelation],
    example: &[u64],
) -> bool {
    let body_offset = layout.body_offset[pattern_index] as usize;
    let body_len = layout.body_len[pattern_index] as usize;
    debug_assert_eq!(example.len(), layout.head_arity[pattern_index] as usize);

    let mut joins = [0u64; NARY_MAX_JOIN_VARS];
    let mut bound: u32 = 0;
    let mut row_cursor = [0u32; NARY_MAX_BODY_ATOMS];
    let mut depth_mask = [0u32; NARY_MAX_BODY_ATOMS];

    let mut depth = 0usize;
    loop {
        if depth == body_len {
            return true;
        }
        let atom = body_offset + depth;
        let relation = &candidates[layout.atom_candidate_slot[atom] as usize];
        let arity = layout.atom_arity[atom] as usize;
        let binding_offset = layout.atom_binding_offset[atom] as usize;
        debug_assert_eq!(relation.arity as usize, arity);

        let mut descended = false;
        while row_cursor[depth] < relation.row_count {
            let row = row_cursor[depth] as usize;
            // Try to match this row; joins newly bound here are recorded
            // in `mask` so a failed row (or a failed deeper walk) can be
            // undone exactly.
            let mut mask: u32 = 0;
            let mut matched = true;
            for position in 0..arity {
                let value = relation.values[row * arity + position];
                let code = layout.binding_codes[binding_offset + position];
                if code & JOIN_FLAG != 0 {
                    let j = (code & 0xFF) as usize;
                    let bit = 1u32 << j;
                    if bound & bit != 0 {
                        if joins[j] != value {
                            matched = false;
                            break;
                        }
                    } else {
                        joins[j] = value;
                        bound |= bit;
                        mask |= bit;
                    }
                } else {
                    let head = (code & 0xFF) as usize;
                    if example[head] != value {
                        matched = false;
                        break;
                    }
                }
            }
            if matched {
                depth_mask[depth] = mask;
                depth += 1;
                if depth < body_len {
                    row_cursor[depth] = 0;
                }
                descended = true;
                break;
            }
            bound &= !mask;
            row_cursor[depth] += 1;
        }
        if descended {
            continue;
        }
        // This depth is exhausted: unwind one level and retry its next row.
        if depth == 0 {
            return false;
        }
        depth -= 1;
        bound &= !depth_mask[depth];
        row_cursor[depth] += 1;
    }
}

/// Coverage counts for one pattern over example sets, via the flat walk.
pub fn score_pattern_flat_coverage(
    layout: &NaryPatternBatchLayout,
    pattern_index: usize,
    candidates: &[FlatRelation],
    positives: &[Vec<u64>],
    negatives: &[Vec<u64>],
) -> (u32, u32) {
    let count = |examples: &[Vec<u64>]| -> u32 {
        examples
            .iter()
            .filter(|example| score_pattern_flat(layout, pattern_index, candidates, example))
            .count() as u32
    };
    (count(positives), count(negatives))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nary::{canonical_binary_pattern, BodyAtomPattern};
    use crate::nary_reference::{score_pattern_reference, HostRelation, ReferenceCoverage};
    use crate::types::Topology;

    fn flat_pairs(rows: &[(u64, u64)]) -> FlatRelation {
        FlatRelation::from_rows(
            &rows.iter().map(|(a, b)| vec![*a, *b]).collect::<Vec<_>>(),
            2,
        )
    }

    fn host_pairs(rows: &[(u64, u64)]) -> HostRelation {
        HostRelation {
            rows: rows.iter().map(|(a, b)| vec![*a, *b]).collect(),
        }
    }

    #[test]
    fn binding_code_round_trips() {
        for var in [
            PatternVar::Head(0),
            PatternVar::Head(7),
            PatternVar::Join(0),
            PatternVar::Join(7),
        ] {
            assert_eq!(decode_binding(binding_code(var)), var);
        }
    }

    /// The flat iterative walk must agree with the recursive reference on
    /// the shipped kernel fixture across ALL 16 (topology, L, R) combos.
    #[test]
    fn flat_walk_matches_reference_on_kernel_fixture() {
        let flat_candidates = vec![
            flat_pairs(&[(1, 2), (2, 3)]),
            flat_pairs(&[(2, 4), (3, 5), (4, 6)]),
        ];
        let host_candidates = vec![
            host_pairs(&[(1, 2), (2, 3)]),
            host_pairs(&[(2, 4), (3, 5), (4, 6)]),
        ];
        let positives = vec![vec![1u64, 4u64], vec![2, 5]];
        let negatives = vec![vec![7u64, 8u64]];

        let mut patterns = Vec::new();
        for topology in Topology::ALL {
            for left in 0..2u32 {
                for right in 0..2u32 {
                    patterns.push(canonical_binary_pattern(topology, left, right));
                }
            }
        }
        let layout = flatten_patterns(&patterns).expect("bounded batch flattens");
        for (index, pattern) in patterns.iter().enumerate() {
            let reference =
                score_pattern_reference(pattern, &host_candidates, &positives, &negatives);
            let (pos, neg) = score_pattern_flat_coverage(
                &layout,
                index,
                &flat_candidates,
                &positives,
                &negatives,
            );
            assert_eq!(
                ReferenceCoverage {
                    positives_covered: pos,
                    negatives_covered: neg
                },
                reference,
                "pattern {index} diverges from the recursive reference"
            );
        }
    }

    /// Ternary fixture from the reference suite: join consistency across
    /// atoms must hold in the iterative walk too.
    #[test]
    fn flat_walk_matches_reference_on_ternary_fixture() {
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
        let ternary_rows = vec![vec![1u64, 2, 9], vec![4, 5, 8]];
        let binary_rows = [(9u64, 3u64), (8, 7)];
        let flat_candidates = vec![
            FlatRelation::from_rows(&ternary_rows, 3),
            flat_pairs(&binary_rows),
        ];
        let host_candidates = vec![
            HostRelation {
                rows: ternary_rows.clone(),
            },
            host_pairs(&binary_rows),
        ];
        let positives = vec![vec![1u64, 2, 3], vec![4, 5, 6], vec![1, 2, 7]];

        let layout = flatten_patterns(std::slice::from_ref(&pattern)).unwrap();
        let reference = score_pattern_reference(&pattern, &host_candidates, &positives, &[]);
        let (pos, neg) = score_pattern_flat_coverage(&layout, 0, &flat_candidates, &positives, &[]);
        assert_eq!(pos, reference.positives_covered);
        assert_eq!(neg, reference.negatives_covered);
        assert_eq!(pos, 1);
    }

    /// Backtracking must revisit earlier atom rows after a deeper failure
    /// (the greedy-walk trap the reference suite pins).
    #[test]
    fn flat_walk_backtracks_across_atoms() {
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
        let candidates = vec![flat_pairs(&[(1, 8), (1, 9)]), flat_pairs(&[(9, 2)])];
        let layout = flatten_patterns(std::slice::from_ref(&pattern)).unwrap();
        assert!(score_pattern_flat(&layout, 0, &candidates, &[1, 2]));
        assert!(!score_pattern_flat(&layout, 0, &candidates, &[1, 3]));
    }

    /// A join variable bound then undone must not leak: after failing via
    /// row (1,8), the retry with row (1,9) starts from an unbound state.
    #[test]
    fn undo_masks_do_not_leak_bindings_within_a_row() {
        use PatternVar::Join;
        // One atom with a repeated join variable: row matches only when
        // both positions carry the same value.
        let pattern = NaryRulePattern {
            head_arity: 1,
            body: vec![BodyAtomPattern {
                candidate_slot: 0,
                bindings: vec![Join(0), Join(0)],
            }],
        };
        let candidates = vec![flat_pairs(&[(3, 4), (5, 5)])];
        let layout = flatten_patterns(std::slice::from_ref(&pattern)).unwrap();
        // (3,4) binds z=3 then fails 4 != 3; the undo must clear z so
        // (5,5) can bind and match.
        assert!(score_pattern_flat(&layout, 0, &candidates, &[0]));
    }

    #[test]
    fn device_bounds_are_refused_typed() {
        use PatternVar::{Head, Join};
        assert_eq!(flatten_patterns(&[]), Err(NaryLayoutError::EmptyBatch));

        let empty_body = NaryRulePattern {
            head_arity: 2,
            body: vec![],
        };
        assert_eq!(
            flatten_patterns(std::slice::from_ref(&empty_body)),
            Err(NaryLayoutError::EmptyBody { pattern: 0 })
        );

        let atom = BodyAtomPattern {
            candidate_slot: 0,
            bindings: vec![Head(0), Head(1)],
        };
        let too_many_atoms = NaryRulePattern {
            head_arity: 2,
            body: vec![atom.clone(); NARY_MAX_BODY_ATOMS + 1],
        };
        assert!(matches!(
            flatten_patterns(std::slice::from_ref(&too_many_atoms)),
            Err(NaryLayoutError::TooManyBodyAtoms { pattern: 0, .. })
        ));

        let wide_atom = NaryRulePattern {
            head_arity: 2,
            body: vec![BodyAtomPattern {
                candidate_slot: 0,
                bindings: vec![Head(0); NARY_MAX_ATOM_ARITY + 1],
            }],
        };
        assert!(matches!(
            flatten_patterns(std::slice::from_ref(&wide_atom)),
            Err(NaryLayoutError::AtomArityOutOfRange { .. })
        ));

        let join_out_of_range = NaryRulePattern {
            head_arity: 2,
            body: vec![BodyAtomPattern {
                candidate_slot: 0,
                bindings: vec![Head(0), Join(NARY_MAX_JOIN_VARS as u8)],
            }],
        };
        assert!(matches!(
            flatten_patterns(std::slice::from_ref(&join_out_of_range)),
            Err(NaryLayoutError::JoinIndexOutOfRange { .. })
        ));

        let head_out_of_range = NaryRulePattern {
            head_arity: 2,
            body: vec![BodyAtomPattern {
                candidate_slot: 0,
                bindings: vec![Head(2), Head(0)],
            }],
        };
        assert!(matches!(
            flatten_patterns(std::slice::from_ref(&head_out_of_range)),
            Err(NaryLayoutError::HeadIndexOutOfRange { .. })
        ));
    }

    /// Offsets must tile the flat arrays exactly (no gaps, no overlap).
    #[test]
    fn layout_offsets_tile_the_flat_arrays() {
        let patterns = vec![
            canonical_binary_pattern(Topology::Chain, 0, 1),
            canonical_binary_pattern(Topology::Star, 1, 0),
        ];
        let layout = flatten_patterns(&patterns).unwrap();
        assert_eq!(layout.body_offset, vec![0, 2]);
        assert_eq!(layout.body_len, vec![2, 2]);
        assert_eq!(layout.atom_candidate_slot.len(), 4);
        assert_eq!(layout.atom_binding_offset, vec![0, 2, 4, 6]);
        assert_eq!(layout.binding_codes.len(), 8);
        assert_eq!(layout.join_count.len(), 2);
    }
}
