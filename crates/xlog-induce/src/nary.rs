//! Ordered n-ary rule-pattern layer for exact induction.
//!
//! Generalizes the four fixed 2-body binary topologies to rule patterns over
//! heads of arity `>= 1` with bodies of up to `max_body_atoms` atoms, where
//! every body-atom position binds either a head argument position or a
//! bounded, canonically-numbered existential join variable. The shipped
//! binary engine's templates are exactly four members of this space at
//! `head_arity = 2`, two body atoms and one join variable — see
//! [`canonical_binary_pattern`], which is the parity surface the general
//! enumeration is tested against.
//!
//! This module is deliberately pure host-side: pattern types, canonical-form
//! validation and deterministic enumeration. Scoring the patterns on device
//! is a separate stage that consumes [`NaryRulePattern`] values as its rule
//! templates.
//!
//! Two laws are enforced here rather than documented elsewhere:
//!
//! * **Canonical form.** Join variables are numbered densely in first
//!   appearance order across the body read left to right. Two patterns that
//!   differ only by join-variable renaming therefore cannot both exist, and
//!   the enumeration never produces alpha-duplicates by construction.
//! * **No silent truncation.** The enumeration refuses with a typed error
//!   when the pattern space exceeds `max_patterns`; it never quietly caps.

use xlog_core::{Result, XlogError};

/// A variable slot inside a body-atom binding pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PatternVar {
    /// Bound to head argument position `i` (0-based, `< head_arity`).
    Head(u8),
    /// Bound to existential join variable `j` (0-based). Join variables are
    /// canonical: `j` is dense in first-appearance order across the body.
    Join(u8),
}

/// One body atom: a candidate-relation slot plus its ordered bindings.
///
/// `candidate_slot` indexes the request's candidate list (the same slot the
/// binary engine calls `L`/`R` indices); the binding vector length is the
/// atom's arity and must match the slot's declared relation arity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BodyAtomPattern {
    pub candidate_slot: u32,
    pub bindings: Vec<PatternVar>,
}

/// A canonical n-ary rule pattern: `H(h0..h{a-1}) :- body...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NaryRulePattern {
    pub head_arity: u8,
    pub body: Vec<BodyAtomPattern>,
}

/// Enumeration bounds for one n-ary induction request.
#[derive(Debug, Clone, Copy)]
pub struct NaryEnumerationConfig {
    /// Maximum number of body atoms per pattern (the binary engine uses 2).
    pub max_body_atoms: u8,
    /// Maximum distinct existential join variables per pattern (binary: 1).
    pub max_join_vars: u8,
    /// Hard ceiling on the enumerated pattern count. Exceeding it is a typed
    /// refusal, never a silent cap.
    pub max_patterns: u32,
}

impl NaryRulePattern {
    /// Validate canonical form against a head arity, the candidate arity
    /// table and the enumeration bounds.
    ///
    /// The checks, in refusal order:
    /// 1. structural: nonzero head arity, nonempty body, body length bound;
    /// 2. per-atom: known candidate slot, binding length == declared arity;
    /// 3. variable legality: `Head(i) < head_arity`, join count bound;
    /// 4. canonical join numbering (dense, first-appearance order);
    /// 5. range restriction: every head position bound somewhere in the
    ///    body — a head position no body atom touches cannot be scored by
    ///    coverage and is refused as an unsafe variable rather than scored
    ///    vacuously.
    pub fn validate(
        &self,
        candidate_arities: &[u8],
        config: &NaryEnumerationConfig,
    ) -> Result<()> {
        if self.head_arity == 0 {
            return Err(XlogError::Type(
                "nary pattern: head arity must be at least 1".into(),
            ));
        }
        if self.body.is_empty() {
            return Err(XlogError::Type(
                "nary pattern: body must carry at least one atom".into(),
            ));
        }
        if self.body.len() > config.max_body_atoms as usize {
            return Err(XlogError::Type(format!(
                "nary pattern: body has {} atoms but max_body_atoms is {}",
                self.body.len(),
                config.max_body_atoms
            )));
        }
        let mut next_join: u8 = 0;
        let mut head_bound = vec![false; self.head_arity as usize];
        for (atom_index, atom) in self.body.iter().enumerate() {
            let declared = candidate_arities
                .get(atom.candidate_slot as usize)
                .copied()
                .ok_or_else(|| {
                    XlogError::Type(format!(
                        "nary pattern: body atom {atom_index} names candidate \
                         slot {} but only {} candidates exist",
                        atom.candidate_slot,
                        candidate_arities.len()
                    ))
                })?;
            if atom.bindings.len() != declared as usize {
                return Err(XlogError::Type(format!(
                    "nary pattern: body atom {atom_index} binds {} positions \
                     but candidate slot {} declares arity {declared}",
                    atom.bindings.len(),
                    atom.candidate_slot
                )));
            }
            for binding in &atom.bindings {
                match *binding {
                    PatternVar::Head(i) => {
                        if i >= self.head_arity {
                            return Err(XlogError::Type(format!(
                                "nary pattern: head position {i} out of range \
                                 for head arity {}",
                                self.head_arity
                            )));
                        }
                        head_bound[i as usize] = true;
                    }
                    PatternVar::Join(j) => {
                        if j > next_join {
                            return Err(XlogError::Type(format!(
                                "nary pattern: join variable z{j} appears \
                                 before z{} — join numbering must be dense in \
                                 first-appearance order",
                                j.saturating_sub(1)
                            )));
                        }
                        if j == next_join {
                            next_join += 1;
                            if next_join > config.max_join_vars {
                                return Err(XlogError::Type(format!(
                                    "nary pattern: {} join variables exceed \
                                     max_join_vars {}",
                                    next_join, config.max_join_vars
                                )));
                            }
                        }
                    }
                }
            }
        }
        if let Some(unbound) = head_bound.iter().position(|bound| !bound) {
            return Err(XlogError::UnsafeVariable(format!("h{unbound}")));
        }
        Ok(())
    }
}

/// The four shipped binary topologies expressed as n-ary patterns.
///
/// This is the parity surface: scoring these four patterns for a pair of
/// candidate slots must reproduce the binary engine's `(topology, L, R)`
/// results exactly.
pub fn canonical_binary_pattern(
    topology: crate::types::Topology,
    left_slot: u32,
    right_slot: u32,
) -> NaryRulePattern {
    use crate::types::Topology;
    use PatternVar::{Head, Join};
    let (left, right) = match topology {
        // H(X,Y) :- L(X,Z), R(Z,Y)
        Topology::Chain => (vec![Head(0), Join(0)], vec![Join(0), Head(1)]),
        // H(X,Y) :- L(X,Y), R(X,Y)
        Topology::Star => (vec![Head(0), Head(1)], vec![Head(0), Head(1)]),
        // H(X,Y) :- L(X,Z), R(X,Y)
        Topology::Fanout => (vec![Head(0), Join(0)], vec![Head(0), Head(1)]),
        // H(X,Y) :- L(X,Y), R(Z,Y)
        Topology::Fanin => (vec![Head(0), Head(1)], vec![Join(0), Head(1)]),
    };
    NaryRulePattern {
        head_arity: 2,
        body: vec![
            BodyAtomPattern {
                candidate_slot: left_slot,
                bindings: left,
            },
            BodyAtomPattern {
                candidate_slot: right_slot,
                bindings: right,
            },
        ],
    }
}

/// Deterministically enumerate every well-formed canonical pattern.
///
/// Order is lexicographic in `(body_len, candidate slots, bindings)` with
/// `Head(i)` ordered before `Join(j)` at each position, so two calls with the
/// same inputs return identical vectors. Canonical join numbering is
/// generated directly (a join slot may only introduce the next unused join
/// index), so no alpha-duplicate is ever produced or filtered.
///
/// Refuses with a typed error the moment the pattern count would exceed
/// `config.max_patterns`.
pub fn enumerate_patterns(
    head_arity: u8,
    candidate_arities: &[u8],
    config: &NaryEnumerationConfig,
) -> Result<Vec<NaryRulePattern>> {
    if head_arity == 0 {
        return Err(XlogError::Type(
            "nary enumeration: head arity must be at least 1".into(),
        ));
    }
    if candidate_arities.is_empty() {
        return Ok(Vec::new());
    }
    let mut patterns: Vec<NaryRulePattern> = Vec::new();
    let mut body: Vec<BodyAtomPattern> = Vec::new();
    for body_len in 1..=config.max_body_atoms {
        enumerate_bodies(
            head_arity,
            candidate_arities,
            config,
            body_len,
            &mut body,
            0,
            &mut patterns,
        )?;
    }
    Ok(patterns)
}

#[allow(clippy::too_many_arguments)]
fn enumerate_bodies(
    head_arity: u8,
    candidate_arities: &[u8],
    config: &NaryEnumerationConfig,
    body_len: u8,
    body: &mut Vec<BodyAtomPattern>,
    joins_used: u8,
    out: &mut Vec<NaryRulePattern>,
) -> Result<()> {
    if body.len() == body_len as usize {
        let candidate = NaryRulePattern {
            head_arity,
            body: body.clone(),
        };
        // Range restriction is the only law the canonical generator cannot
        // guarantee positionally; everything else holds by construction.
        let mut head_bound = vec![false; head_arity as usize];
        for atom in &candidate.body {
            for binding in &atom.bindings {
                if let PatternVar::Head(i) = *binding {
                    head_bound[i as usize] = true;
                }
            }
        }
        if head_bound.iter().all(|bound| *bound) {
            if out.len() as u32 >= config.max_patterns {
                return Err(XlogError::Execution(format!(
                    "nary enumeration: pattern space exceeds max_patterns {} \
                     (head_arity {head_arity}, {} candidates, max_body_atoms \
                     {}, max_join_vars {}); raise the bound explicitly or \
                     narrow the request — the engine never truncates silently",
                    config.max_patterns,
                    candidate_arities.len(),
                    config.max_body_atoms,
                    config.max_join_vars
                )));
            }
            out.push(candidate);
        }
        return Ok(());
    }
    for slot in 0..candidate_arities.len() as u32 {
        let arity = candidate_arities[slot as usize];
        let mut bindings: Vec<PatternVar> = Vec::with_capacity(arity as usize);
        enumerate_bindings(
            head_arity,
            candidate_arities,
            config,
            body_len,
            body,
            joins_used,
            slot,
            arity,
            &mut bindings,
            out,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn enumerate_bindings(
    head_arity: u8,
    candidate_arities: &[u8],
    config: &NaryEnumerationConfig,
    body_len: u8,
    body: &mut Vec<BodyAtomPattern>,
    joins_used: u8,
    slot: u32,
    arity: u8,
    bindings: &mut Vec<PatternVar>,
    out: &mut Vec<NaryRulePattern>,
) -> Result<()> {
    if bindings.len() == arity as usize {
        body.push(BodyAtomPattern {
            candidate_slot: slot,
            bindings: bindings.clone(),
        });
        let joins_now = joins_used.max(next_join_index(body));
        enumerate_bodies(
            head_arity,
            candidate_arities,
            config,
            body_len,
            body,
            joins_now,
            out,
        )?;
        body.pop();
        return Ok(());
    }
    for head in 0..head_arity {
        bindings.push(PatternVar::Head(head));
        enumerate_bindings(
            head_arity,
            candidate_arities,
            config,
            body_len,
            body,
            joins_used,
            slot,
            arity,
            bindings,
            out,
        )?;
        bindings.pop();
    }
    // Canonical growth: a join slot may reuse any join already introduced or
    // introduce exactly the next unused index, never a later one.
    let introduced_so_far = joins_used.max(bindings_join_watermark(bindings, joins_used));
    let reachable = introduced_so_far.saturating_add(1).min(config.max_join_vars);
    for join in 0..reachable {
        bindings.push(PatternVar::Join(join));
        enumerate_bindings(
            head_arity,
            candidate_arities,
            config,
            body_len,
            body,
            joins_used,
            slot,
            arity,
            bindings,
            out,
        )?;
        bindings.pop();
    }
    Ok(())
}

/// Highest join index introduced anywhere in `body`, as a next-free counter.
fn next_join_index(body: &[BodyAtomPattern]) -> u8 {
    let mut next = 0u8;
    for atom in body {
        for binding in &atom.bindings {
            if let PatternVar::Join(j) = *binding {
                next = next.max(j + 1);
            }
        }
    }
    next
}

/// Highest join index introduced in the partial `bindings`, as a next-free
/// counter floored at `joins_used` (joins introduced by earlier atoms).
fn bindings_join_watermark(bindings: &[PatternVar], joins_used: u8) -> u8 {
    let mut next = joins_used;
    for binding in bindings {
        if let PatternVar::Join(j) = *binding {
            next = next.max(j + 1);
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Topology;

    fn config(max_body_atoms: u8, max_join_vars: u8) -> NaryEnumerationConfig {
        NaryEnumerationConfig {
            max_body_atoms,
            max_join_vars,
            max_patterns: 1_000_000,
        }
    }

    #[test]
    fn canonical_binary_patterns_validate_and_match_templates() {
        let arities = [2u8, 2u8];
        let cfg = config(2, 1);
        for topology in Topology::ALL {
            let pattern = canonical_binary_pattern(topology, 0, 1);
            pattern
                .validate(&arities, &cfg)
                .unwrap_or_else(|e| panic!("{topology:?} failed validation: {e}"));
            assert_eq!(pattern.head_arity, 2);
            assert_eq!(pattern.body.len(), 2);
        }
        // Chain joins through z on the inner positions - the exact shipped
        // template H(X,Y) :- L(X,Z), R(Z,Y).
        let chain = canonical_binary_pattern(Topology::Chain, 0, 1);
        assert_eq!(
            chain.body[0].bindings,
            vec![PatternVar::Head(0), PatternVar::Join(0)]
        );
        assert_eq!(
            chain.body[1].bindings,
            vec![PatternVar::Join(0), PatternVar::Head(1)]
        );
    }

    #[test]
    fn enumeration_contains_all_four_binary_topologies() {
        let arities = [2u8, 2u8];
        let cfg = config(2, 1);
        let patterns = enumerate_patterns(2, &arities, &cfg).expect("enumerate");
        for topology in Topology::ALL {
            let expected = canonical_binary_pattern(topology, 0, 1);
            assert!(
                patterns.contains(&expected),
                "{topology:?} missing from the general enumeration"
            );
        }
    }

    #[test]
    fn enumeration_is_deterministic_and_canonical() {
        let arities = [2u8, 3u8];
        let cfg = config(2, 2);
        let first = enumerate_patterns(3, &arities, &cfg).expect("enumerate");
        let second = enumerate_patterns(3, &arities, &cfg).expect("enumerate");
        assert_eq!(first, second);
        for pattern in &first {
            pattern
                .validate(&arities, &cfg)
                .unwrap_or_else(|e| panic!("non-canonical pattern {pattern:?}: {e}"));
        }
    }

    #[test]
    fn enumeration_count_matches_hand_derivation() {
        // head_arity=1, one arity-1 candidate, max_body=2, max_join=1.
        // K=1: (H0) — a lone (J0) leaves the head unbound and is dropped.
        // K=2: (H0,H0), (H0,J0), (J0,H0); (J0,J0) drops (head unbound),
        // (J0,J1) is unreachable at max_join=1. Total = 4.
        let arities = [1u8];
        let cfg = config(2, 1);
        let patterns = enumerate_patterns(1, &arities, &cfg).expect("enumerate");
        assert_eq!(
            patterns.len(),
            4,
            "hand-derived space changed: {patterns:#?}"
        );
    }

    #[test]
    fn validation_refuses_out_of_range_head_position() {
        let pattern = NaryRulePattern {
            head_arity: 2,
            body: vec![BodyAtomPattern {
                candidate_slot: 0,
                bindings: vec![PatternVar::Head(2), PatternVar::Head(0)],
            }],
        };
        let err = pattern.validate(&[2], &config(2, 1)).unwrap_err();
        assert!(matches!(err, XlogError::Type(_)), "got {err:?}");
    }

    #[test]
    fn validation_refuses_unbound_head_position_as_unsafe_variable() {
        let pattern = NaryRulePattern {
            head_arity: 2,
            body: vec![BodyAtomPattern {
                candidate_slot: 0,
                bindings: vec![PatternVar::Head(0), PatternVar::Join(0)],
            }],
        };
        let err = pattern.validate(&[2], &config(2, 1)).unwrap_err();
        assert!(matches!(err, XlogError::UnsafeVariable(v) if v == "h1"));
    }

    #[test]
    fn validation_refuses_non_dense_join_numbering() {
        let pattern = NaryRulePattern {
            head_arity: 1,
            body: vec![BodyAtomPattern {
                candidate_slot: 0,
                bindings: vec![PatternVar::Head(0), PatternVar::Join(1)],
            }],
        };
        let err = pattern.validate(&[2], &config(2, 2)).unwrap_err();
        assert!(matches!(err, XlogError::Type(_)), "got {err:?}");
    }

    #[test]
    fn validation_refuses_arity_mismatch_against_declared_candidate() {
        let pattern = NaryRulePattern {
            head_arity: 1,
            body: vec![BodyAtomPattern {
                candidate_slot: 0,
                bindings: vec![PatternVar::Head(0)],
            }],
        };
        let err = pattern.validate(&[3], &config(2, 1)).unwrap_err();
        assert!(matches!(err, XlogError::Type(_)), "got {err:?}");
    }

    #[test]
    fn pattern_space_guard_refuses_instead_of_truncating() {
        let arities = [2u8, 2u8];
        let cfg = NaryEnumerationConfig {
            max_body_atoms: 2,
            max_join_vars: 1,
            max_patterns: 3,
        };
        let err = enumerate_patterns(2, &arities, &cfg).unwrap_err();
        assert!(
            matches!(err, XlogError::Execution(ref m) if m.contains("max_patterns")),
            "got {err:?}"
        );
    }
}
