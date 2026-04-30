//! Typed oracle gate.
//!
//! Wraps the structural [`super::evaluate_rule`],
//! [`super::evaluate_fixpoint`], and [`super::evaluate_scc_fixpoint`]
//! oracles with a relation-schema-driven type gate. For each rule:
//!
//! 1. Walk positive body atoms; for each `Term::Variable` whose
//!    predicate is present in `base_relations`, look up
//!    `relation.schema[position]` and unify into a per-rule
//!    `BTreeMap<String, ScalarType>`. Cross-atom disagreement is a
//!    [`RefEvalError::ConflictingVariableType`].
//! 2. Run [`super::analyze_typed`] with that map. Any
//!    [`Boundary::UnsupportedKeyType`](super::Boundary) for a
//!    join-key vertex becomes an [`RefEvalError::Ineligible`].
//! 3. Delegate to the structural evaluator.
//!
//! ## Locked policy: unknown ≠ unsupported
//!
//! A "missing vertex type" means *not derivable from base relation
//! schemas in this slice*, **not** "supported." The typed gate
//! rejects only known-unsupported join-key types. Vertices anchored
//! solely through predicates absent from `base_relations` (e.g.
//! SCC predicates derived during a fixpoint) carry no type at the
//! gate; per the policy locked by `evaluate_*_recursive_only_*`
//! tests, they pass through. Transitive SCC type propagation is a
//! follow-up slice.
//!
//! ## Why a separate module
//!
//! The structural evaluators (PR 2/3/4) take no type map. Wiring
//! type derivation into each evaluator's signature would either
//! (a) force callers to pre-compute the map even when they want
//! the structural-only behavior or (b) introduce a parallel
//! "with-types" code path inside each evaluator. Keeping the gate
//! as a thin orchestration layer above the existing oracles
//! preserves the structural API and makes the typed contract
//! testable in one place.

use super::{
    analyze_typed, evaluate_fixpoint, evaluate_rule, evaluate_scc_fixpoint, Eligibility,
    FixpointConfig, FixpointError, HypergraphRule, RefEvalError, RefRelation, RefRelationStore,
    RefValue, SccFixpointError, VariableOrder,
};
use crate::ast::{BodyLiteral, Rule, Term};
use std::collections::BTreeMap;
use xlog_core::ScalarType;

/// Evaluate a single rule with relation-schema-driven type gating.
///
/// Equivalent to [`evaluate_rule`] except that the rule is first
/// run through the typed gate: vertex types are derived from
/// `relations` (see module-level docs), then [`analyze_typed`] is
/// run; any [`super::Boundary::UnsupportedKeyType`] surfaces as
/// [`RefEvalError::Ineligible`]. Cross-atom type conflicts surface
/// as [`RefEvalError::ConflictingVariableType`].
///
/// Structural boundaries (negation, aggregation, ground fact, key
/// limit) flow through `analyze_typed` unchanged.
///
/// On a successful gate, delegates to [`evaluate_rule`] for the
/// actual computation. The structural analyzer there re-checks
/// boundaries (cheap, defensive), and returns the same row set
/// that [`evaluate_rule`] would have returned for an Eligible rule.
pub fn evaluate_rule_typed(
    rule: &Rule,
    relations: &RefRelationStore,
    order: &dyn VariableOrder,
) -> Result<Vec<Vec<RefValue>>, RefEvalError> {
    typed_gate(rule, relations)?;
    evaluate_rule(rule, relations, order)
}

/// Evaluate a recursive fixpoint with typed gating applied to every
/// rule against `base_relations` at function entry.
///
/// Failures from the gate surface as
/// [`FixpointError::RuleEval { rule_index, source }`] with `source`
/// being either [`RefEvalError::Ineligible`] or
/// [`RefEvalError::ConflictingVariableType`]. This matches the
/// per-rule error wrapping the structural [`evaluate_fixpoint`]
/// already uses, so callers do not need a separate error path for
/// the typed gate.
pub fn evaluate_fixpoint_typed(
    rules: &[Rule],
    base_relations: &RefRelationStore,
    target_predicate: &str,
    order: &dyn VariableOrder,
    config: &FixpointConfig,
) -> Result<RefRelation, FixpointError> {
    for (rule_index, rule) in rules.iter().enumerate() {
        if let Err(source) = typed_gate(rule, base_relations) {
            return Err(FixpointError::RuleEval { rule_index, source });
        }
    }
    evaluate_fixpoint(rules, base_relations, target_predicate, order, config)
}

/// Evaluate a multi-predicate SCC fixpoint with typed gating
/// applied to every rule against `base_relations` at function
/// entry.
///
/// Failures from the gate surface as
/// [`SccFixpointError::RuleEval { predicate, rule_index, source }`]
/// where `source` is either [`RefEvalError::Ineligible`] or
/// [`RefEvalError::ConflictingVariableType`]. Same wrapping
/// pattern as [`evaluate_fixpoint_typed`].
pub fn evaluate_scc_fixpoint_typed(
    rules: &BTreeMap<String, Vec<Rule>>,
    base_relations: &RefRelationStore,
    order: &dyn VariableOrder,
    config: &FixpointConfig,
) -> Result<RefRelationStore, SccFixpointError> {
    for (predicate, group) in rules.iter() {
        for (rule_index, rule) in group.iter().enumerate() {
            if let Err(source) = typed_gate(rule, base_relations) {
                return Err(SccFixpointError::RuleEval {
                    predicate: predicate.clone(),
                    rule_index,
                    source,
                });
            }
        }
    }
    evaluate_scc_fixpoint(rules, base_relations, order, config)
}

/// Shared typed-gate check. On the gate phase only — does NOT
/// evaluate the rule.
///
/// Returns `Ok(())` when the rule passes the typed gate (no
/// cross-atom type conflict, no unsupported join-key types known
/// from base relations). Returns `Err(RefEvalError::*)` otherwise.
fn typed_gate(rule: &Rule, relations: &RefRelationStore) -> Result<(), RefEvalError> {
    let vertex_types = derive_vertex_types(rule, relations)?;
    let hg = HypergraphRule::from_rule(rule);
    if let Eligibility::Ineligible(bs) = analyze_typed(&hg, &vertex_types) {
        return Err(RefEvalError::Ineligible(bs));
    }
    Ok(())
}

/// Derive variable types from the schemas of body-atom relations.
///
/// Only [`BodyLiteral::Positive`] atoms whose predicate is present
/// in `relations` contribute. For each such atom, every
/// [`Term::Variable`] argument is unified with the relation's
/// `schema[position]`. Variables that appear only through
/// predicates absent from `relations` (or only through non-positive
/// literals) do not appear in the returned map; per the locked
/// policy, [`analyze_typed`] does not flag them.
///
/// On the first atom that types a given variable, the type is
/// recorded together with the (predicate, position) pair. A later
/// atom that types the same variable to a *different* type
/// surfaces as [`RefEvalError::ConflictingVariableType`] with both
/// triples populated. Subsequent agreeing atoms are silent.
fn derive_vertex_types(
    rule: &Rule,
    relations: &RefRelationStore,
) -> Result<BTreeMap<String, ScalarType>, RefEvalError> {
    /// First-recorded site for a variable. Stored separately from
    /// the public type map so the conflict report can name the
    /// originating atom.
    struct FirstSite {
        predicate: String,
        position: usize,
        ty: ScalarType,
    }
    let mut sites: BTreeMap<String, FirstSite> = BTreeMap::new();
    for literal in &rule.body {
        let atom = match literal {
            BodyLiteral::Positive(a) => a,
            // Comparisons, negations, and is-expressions don't
            // constrain types via relation schema. Negation is also
            // a structural boundary that analyze_typed will reject;
            // is-expr likewise. We don't pre-empt those here.
            _ => continue,
        };
        let relation = match relations.get(&atom.predicate) {
            Some(r) => r,
            None => continue, // unknown predicate: no type info
        };
        // Defensive: an atom whose arity disagrees with its
        // relation's schema is rejected by `evaluate_rule`'s
        // `AtomSpec::build` with `RelationArityMismatch`. The gate
        // intentionally does NOT pre-empt that — leaving arity
        // checks to one place keeps the error surface minimal.
        // We just stop deriving from positions past the schema.
        let limit = atom.terms.len().min(relation.schema.len());
        for (position, term) in atom.terms[..limit].iter().enumerate() {
            let var_name = match term {
                Term::Variable(name) => name.clone(),
                _ => continue,
            };
            let ty = relation.schema[position];
            match sites.get(&var_name) {
                None => {
                    sites.insert(
                        var_name,
                        FirstSite {
                            predicate: atom.predicate.clone(),
                            position,
                            ty,
                        },
                    );
                }
                Some(prior) if prior.ty == ty => {
                    // Agreeing repeat — silent.
                }
                Some(prior) => {
                    return Err(RefEvalError::ConflictingVariableType {
                        var: var_name,
                        first_predicate: prior.predicate.clone(),
                        first_position: prior.position,
                        first_type: prior.ty,
                        second_predicate: atom.predicate.clone(),
                        second_position: position,
                        second_type: ty,
                    });
                }
            }
        }
    }
    Ok(sites
        .into_iter()
        .map(|(name, site)| (name, site.ty))
        .collect())
}
