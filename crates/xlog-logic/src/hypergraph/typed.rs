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
//! ## Locked policy: unknowable-after-inference ≠ unsupported
//!
//! A "missing vertex type" means *not derivable from base relation
//! schemas* AND *not derivable from PR 8 transitive SCC type
//! inference*, **not** "supported." The typed gate rejects only
//! known-unsupported join-key types.
//!
//! For the **single-rule** entry point ([`evaluate_rule_typed`]),
//! types come from `base_relations` only — there is no group
//! context for inference. A self-referencing or recursive rule
//! used through this path therefore retains the original
//! "unknown ≠ unsupported" behavior on its recursive body atoms;
//! callers needing inference-aware typing should drive
//! [`evaluate_fixpoint_typed`] or [`evaluate_scc_fixpoint_typed`].
//!
//! For the **group-aware** entry points ([`evaluate_fixpoint_typed`],
//! [`evaluate_scc_fixpoint_typed`]), PR 8 inference runs at entry
//! and feeds inferred SCC predicate schemas alongside
//! `base_relations` into per-rule type derivation. Cyclic-only
//! predicates (no base anchor anywhere in the rule graph) produce
//! all-`None` inferred schemas and pass the gate; this is the
//! narrowed policy now locked by
//! `cyclic_only_predicate_still_passes_typed_gate_locked_policy`.
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

use super::inference::{
    derive_vertex_types_with_inference, infer_scc_predicate_schemas, InferenceError,
    InferredSchemas,
};
use super::{
    analyze_typed, evaluate_fixpoint, evaluate_rule, evaluate_scc_fixpoint, Eligibility,
    ExecutorContext, FixpointConfig, FixpointError, HypergraphRule, RefEvalError, RefRelation,
    RefRelationStore, RefValue, SccFixpointError, VariableOrder,
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
/// ## Within-gate precedence: conflict before boundaries
///
/// Variable-type derivation runs before [`analyze_typed`]. A rule
/// that would fail *both* checks (e.g. a negated body **and** a
/// cross-atom type conflict on a positive atom) surfaces as
/// [`RefEvalError::ConflictingVariableType`], not as
/// [`RefEvalError::Ineligible`]. Type conflicts indicate the
/// fixture supplied a contradiction the analyzer cannot resolve;
/// reporting that first guides the caller to a fixable input.
/// Locked by `typed_gate_conflict_precedes_structural_boundary`.
///
/// On a successful gate, delegates to [`evaluate_rule`] for the
/// actual computation. The structural analyzer there re-checks
/// boundaries (cheap, defensive), and returns the same row set
/// that [`evaluate_rule`] would have returned for an Eligible rule.
///
/// ## SCC type inference is NOT engaged here
///
/// The PR 8 transitive type inference pass
/// ([`super::infer_scc_predicate_schemas`]) requires a rule
/// group as input. This single-rule entry point has no such
/// group, so a self-referencing rule's recursive body atom
/// contributes no type info. For that case, use
/// [`evaluate_fixpoint_typed`] (treating the rule as the target
/// of a single-rule fixpoint) or [`evaluate_scc_fixpoint_typed`].
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
///
/// ## Transitive type inference
///
/// Treats `target_predicate` as a single-element rule group and
/// runs [`infer_scc_predicate_schemas`] before per-rule typed
/// gating. The inferred schema for `target_predicate` is consulted
/// alongside `base_relations` when typing variables in body atoms
/// — so a variable anchored only via the recursive
/// `target_predicate` body atom now receives the inferred type
/// rather than staying untyped. Inference conflicts surface as
/// [`FixpointError::RuleEval { source: ConflictingVariableType }`]
/// (within-rule conflict) or as the new
/// [`FixpointError::RuleEval`]-wrapped synthetic conflict from
/// inference (back-prop conflict; same `RefEvalError` shape).
///
/// ## Structural-error precedence
///
/// The typed gate is skipped for any rule whose head predicate
/// does not equal `target_predicate`. Such a rule would be
/// rejected by the structural [`evaluate_fixpoint`] with
/// [`FixpointError::RuleNotForTarget`] — running the typed gate
/// first would mask that more precise diagnostic. Other structural
/// entry-validation errors (target-in-base, max-iterations,
/// schema-indeterminable, head-arity mismatch) are surfaced by
/// the delegation at the end of this function and so naturally
/// take precedence over typed-gate failures on rules that survive
/// the per-rule head-match filter.
#[allow(clippy::result_large_err)]
pub fn evaluate_fixpoint_typed(
    rules: &[Rule],
    base_relations: &RefRelationStore,
    target_predicate: &str,
    order: &dyn VariableOrder,
    config: &FixpointConfig,
) -> Result<RefRelation, FixpointError> {
    // ### Structural precedence pre-flight (PR 9 contract repair)
    //
    // Inference back-propagates from each rule's head into its
    // group key — if any rule in the input is misgrouped relative
    // to `target_predicate`, that back-propagation could surface
    // as `InferenceConflict` before structural validation has a
    // chance to emit `RuleNotForTarget`. Defer to the structural
    // evaluator BEFORE running inference, so the diagnostic order
    // matches PR 5/PR 6 expectations.
    if rules.iter().any(|r| r.head.predicate != target_predicate) {
        return evaluate_fixpoint(rules, base_relations, target_predicate, order, config);
    }
    // Inference treats target_predicate as a single-element SCC
    // group so the same machinery covers single-target fixpoint
    // and full SCC fixpoint.
    let mut group: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    group.insert(target_predicate.to_string(), rules.to_vec());
    let inferred = match infer_scc_predicate_schemas(&group, base_relations) {
        Ok(s) => s,
        Err(InferenceError::ConflictingPredicateColumnType {
            predicate,
            column,
            first_rule_index,
            first_type,
            second_rule_index,
            second_type,
        }) => {
            // Map the per-group rule index back to the input slice's
            // rule index so the caller's `rule_index` field stays
            // aligned with `rules`.
            let mapped_rule_index =
                nth_rule_index_with_head(rules, target_predicate, second_rule_index);
            return Err(FixpointError::RuleEval {
                rule_index: mapped_rule_index,
                source: RefEvalError::InferenceConflict {
                    predicate,
                    column,
                    first_rule_index,
                    first_type,
                    second_rule_index,
                    second_type,
                },
            });
        }
    };
    for (rule_index, rule) in rules.iter().enumerate() {
        // Pre-flight has guaranteed every rule heads target_predicate.
        if let Err(source) = typed_gate_with_inference(rule, base_relations, &inferred) {
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
///
/// ## Transitive type inference
///
/// Runs [`infer_scc_predicate_schemas`] over the full rule group
/// at entry. For each rule, the typed gate consults
/// `base_relations` AND the inferred schema of any SCC predicate
/// referenced in its body. Variables anchored only via SCC
/// predicates get their inferred types and flow through
/// [`analyze_typed`] like any other typed vertex. Cyclic-only
/// predicates (no base anchor anywhere) produce all-`None`
/// inferred schemas and pass the gate per the locked
/// "unknowable-after-inference ≠ unsupported" policy.
///
/// ## Structural-error precedence
///
/// The typed gate is skipped for any rule whose head predicate
/// does not equal its `BTreeMap` group key. Such a misgrouped
/// rule would be rejected by the structural
/// [`evaluate_scc_fixpoint`] with
/// [`SccFixpointError::RuleHeadPredicateMismatch`] — running the
/// typed gate first would mask that diagnostic. Other structural
/// entry-validation errors (predicate-in-base, max-iterations,
/// schema-indeterminable, head-arity mismatch) are surfaced by
/// the delegation at the end of this function.
#[allow(clippy::result_large_err)]
pub fn evaluate_scc_fixpoint_typed(
    rules: &BTreeMap<String, Vec<Rule>>,
    base_relations: &RefRelationStore,
    order: &dyn VariableOrder,
    config: &FixpointConfig,
) -> Result<RefRelationStore, SccFixpointError> {
    // ### Structural precedence pre-flight (PR 9 contract repair)
    //
    // Inference back-propagates from each rule's head into its
    // group key. If any rule is misgrouped (its head predicate
    // doesn't equal its `BTreeMap` group key), the back-prop
    // could surface as `InferenceConflict` before structural
    // validation has a chance to emit `RuleHeadPredicateMismatch`.
    // Defer to the structural evaluator BEFORE running inference,
    // so the diagnostic order matches PR 5/PR 6 expectations.
    if rules
        .iter()
        .any(|(predicate, group)| group.iter().any(|r| &r.head.predicate != predicate))
    {
        return evaluate_scc_fixpoint(rules, base_relations, order, config);
    }
    let inferred = match infer_scc_predicate_schemas(rules, base_relations) {
        Ok(s) => s,
        Err(InferenceError::ConflictingPredicateColumnType {
            predicate,
            column,
            first_rule_index,
            first_type,
            second_rule_index,
            second_type,
        }) => {
            return Err(SccFixpointError::RuleEval {
                predicate: predicate.clone(),
                rule_index: second_rule_index,
                source: RefEvalError::InferenceConflict {
                    predicate,
                    column,
                    first_rule_index,
                    first_type,
                    second_rule_index,
                    second_type,
                },
            });
        }
    };
    for (predicate, group) in rules.iter() {
        for (rule_index, rule) in group.iter().enumerate() {
            // Pre-flight has guaranteed every rule heads its group key.
            if let Err(source) = typed_gate_with_inference(rule, base_relations, &inferred) {
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

/// Shared typed-gate check that consults inferred SCC schemas
/// alongside `base_relations`. On the gate phase only — does NOT
/// evaluate the rule.
///
/// **Within-gate precedence.** Type conflicts (from
/// [`derive_vertex_types_with_inference`]) are reported before
/// [`super::Boundary`] verdicts (from [`analyze_typed`]); see the
/// `evaluate_rule_typed` doc comment for the rationale.
fn typed_gate_with_inference(
    rule: &Rule,
    relations: &RefRelationStore,
    inferred: &InferredSchemas,
) -> Result<(), RefEvalError> {
    let vertex_types = derive_vertex_types_with_inference(rule, relations, inferred)?;
    let hg = HypergraphRule::from_rule(rule);
    if let Eligibility::Ineligible(bs) =
        analyze_typed(&hg, &vertex_types, ExecutorContext::HashFallback)
    {
        return Err(RefEvalError::Ineligible(bs));
    }
    Ok(())
}

/// Map an SCC-group rule index back to the position in a flat
/// `&[Rule]` slice for [`evaluate_fixpoint_typed`]'s error
/// reporting. `n` is the index within the target predicate's
/// group; we walk the slice and return the position of the n-th
/// rule whose head matches `target`.
fn nth_rule_index_with_head(rules: &[Rule], target: &str, n: usize) -> usize {
    let mut counter = 0usize;
    for (idx, rule) in rules.iter().enumerate() {
        if rule.head.predicate == target {
            if counter == n {
                return idx;
            }
            counter += 1;
        }
    }
    // Fallback: if mapping fails (shouldn't, by construction),
    // return 0 so the error still surfaces. The structural
    // evaluator will catch any actual mismatch.
    0
}

/// Shared typed-gate check (base-only path). Used by
/// [`evaluate_rule_typed`].
///
/// Returns `Ok(())` when the rule passes the typed gate (no
/// cross-atom type conflict, no unsupported join-key types known
/// from base relations). Returns `Err(RefEvalError::*)` otherwise.
///
/// **Within-gate precedence.** Type conflicts (from
/// [`derive_vertex_types`]) are reported before
/// [`super::Boundary`] verdicts (from [`analyze_typed`]); see the
/// `evaluate_rule_typed` doc comment for the rationale.
fn typed_gate(rule: &Rule, relations: &RefRelationStore) -> Result<(), RefEvalError> {
    let vertex_types = derive_vertex_types(rule, relations)?;
    let hg = HypergraphRule::from_rule(rule);
    if let Eligibility::Ineligible(bs) =
        analyze_typed(&hg, &vertex_types, ExecutorContext::HashFallback)
    {
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
///
/// `pub(super)` so [`super::plan`] can reuse the same conflict
/// detection without duplicating the source-walk logic.
pub(super) fn derive_vertex_types(
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
