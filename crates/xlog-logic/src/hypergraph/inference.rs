//! Transitive type inference across SCC predicates.
//!
//! Closes the PR 5 policy gap: where a join-key vertex was anchored
//! only through SCC-recursive atoms, the typed gate previously left
//! it untyped under "unknown ≠ unsupported." This module propagates
//! types through the rule graph — body atoms type variables, head
//! atoms back-propagate to head-predicate columns, iterate to
//! fixpoint — so the typed gate has full type information when it
//! consults [`super::analyze_typed`].
//!
//! ## Where inference is engaged
//!
//! Only the **group-aware** typed entry points engage inference:
//!
//! * [`super::evaluate_scc_fixpoint_typed`] runs inference once at
//!   entry, then types each rule's body using the inferred schemas
//!   plus `base_relations`.
//! * [`super::evaluate_fixpoint_typed`] treats `target_predicate`
//!   as a single-element rule group and runs the same inference.
//!
//! The single-rule entry points retain the base-only typing policy
//! because they have no SCC structure to propagate over:
//!
//! * [`super::evaluate_rule_typed`] takes one rule.
//! * [`super::plan_rule`] / [`super::plan_rules`] plan per-rule.
//!
//! Callers that want SCC-aware planning should drive
//! [`super::evaluate_scc_fixpoint_typed`] directly or build their
//! own inference pass via [`infer_scc_predicate_schemas`].
//!
//! ## Conflict layering
//!
//! Inference detects only **back-propagation conflicts**: e.g.,
//! predicate `p`'s column 0 is `U32` from rule A's head and
//! `Symbol` from rule B's head → [`InferenceError::ConflictingPredicateColumnType`].
//! Within-rule body conflicts (variable `X` typed `U32` in one body
//! atom and `Symbol` in another) stay in the existing
//! [`super::typed`] flow and surface as
//! [`super::RefEvalError::ConflictingVariableType`]. Each conflict
//! type is detected at exactly one layer.
//!
//! ## Cyclic-only predicates
//!
//! When an SCC has no base anchor anywhere (e.g., `a(X) :- b(X),
//! b(X) :- a(X)` with no rule referencing `base_relations`), every
//! column converges to `None`. The typed gate must NOT reject such
//! rules: the policy narrows from "unknown ≠ unsupported" to
//! "unknowable-after-inference ≠ unsupported." Locked by
//! `cyclic_only_predicate_still_passes_typed_gate_locked_policy`.
//!
//! ## Strict-correctness behavior change
//!
//! Fixtures whose base-relation schemas disagreed but whose actual
//! rows happened to agree at runtime were previously silent (the
//! typed gate types each body atom independently). They now surface
//! as [`InferenceError::ConflictingPredicateColumnType`] when
//! back-propagating to a head predicate. That is a strict
//! correctness win, not a regression — fixtures with internally
//! contradictory schemas are now caught before evaluation rather
//! than silently corrupting downstream comparisons.

use super::reference::RefRelationStore;
use crate::ast::{BodyLiteral, Rule, Term};
use std::collections::BTreeMap;
use xlog_core::ScalarType;

/// Errors surfaced by [`infer_scc_predicate_schemas`].
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    /// Two rules contributing to the same head predicate disagree
    /// on the type of the same column. The first rule that types
    /// the column wins `first_*`; the rule that disagrees wins
    /// `second_*`.
    ConflictingPredicateColumnType {
        /// Head predicate name where the conflict was detected.
        predicate: String,
        /// 0-based column index where types disagree.
        column: usize,
        /// Rule index (within the predicate's rule group) that
        /// first typed the column.
        first_rule_index: usize,
        /// Type derived from the first rule's body for the head
        /// variable at this column.
        first_type: ScalarType,
        /// Rule index (within the predicate's rule group) whose
        /// derivation conflicts.
        second_rule_index: usize,
        /// Type derived from the conflicting rule's body for the
        /// head variable at this column.
        second_type: ScalarType,
    },
}

/// Per-predicate inferred schema. `Vec` length equals the head
/// arity; each element is `Some(t)` if inference established the
/// column's type, or `None` if the column remains unknowable
/// (e.g., cyclic-only predicate, or a head term whose body atoms
/// don't type the corresponding variable).
pub type InferredSchemas = BTreeMap<String, Vec<Option<ScalarType>>>;

/// Infer per-predicate schemas for a rule group via constraint
/// propagation through the rule graph.
///
/// Algorithm:
///
/// 1. Determine head arity per predicate from the first rule with
///    a non-empty head. (Predicates whose every rule has an empty
///    head are treated as 0-arity; in practice this is rare.)
/// 2. Initialize each predicate's schema as `vec![None; arity]`.
/// 3. Iterate: for each rule, compute a per-rule variable-to-type
///    map by walking body atoms (typing vars from
///    `base_relations` schemas first, then from currently-inferred
///    SCC predicate schemas where columns are `Some`). Then
///    back-propagate: for each `Term::Variable` in the head at
///    column `i`, if the variable has a derived type, propose it
///    as the type for `head_predicate.schema[i]`. Conflict if a
///    column has been previously typed differently.
/// 4. Stop when no schema column changes between iterations.
///
/// Within-rule body conflicts are NOT detected here; they are
/// caught by the existing [`super::typed`] gate during its own
/// per-rule type-derivation walk. See module docs for the
/// conflict-layering split.
pub fn infer_scc_predicate_schemas(
    rules: &BTreeMap<String, Vec<Rule>>,
    base_relations: &RefRelationStore,
) -> Result<InferredSchemas, InferenceError> {
    // Step 1+2: arity + initial schemas.
    let mut schemas: InferredSchemas = BTreeMap::new();
    for (predicate, group) in rules.iter() {
        let arity = group
            .iter()
            .find(|r| !r.head.terms.is_empty())
            .map(|r| r.head.terms.len())
            .unwrap_or(0);
        schemas.insert(predicate.clone(), vec![None; arity]);
    }
    // Track the rule index that first typed each column so the
    // conflict report can name both contributors.
    let mut origins: BTreeMap<(String, usize), usize> = BTreeMap::new();
    // Inference is monotonic: every iteration that changes
    // anything replaces a `None` with a `Some(_)`. The total
    // number of column slots across all SCC predicates is the
    // strict upper bound on iterations that produce change. We
    // add 1 to allow for the final no-change iteration that
    // detects convergence.
    let total_columns: usize = schemas.values().map(|s| s.len()).sum();
    let max_iterations = total_columns + 1;
    let mut converged = false;
    for _ in 0..max_iterations {
        let mut changed = false;
        for (predicate, group) in rules.iter() {
            for (rule_index, rule) in group.iter().enumerate() {
                let var_types = derive_rule_var_types(rule, base_relations, &schemas);
                // Back-propagate from head terms to head-predicate
                // columns.
                for (col, term) in rule.head.terms.iter().enumerate() {
                    let name = match term {
                        Term::Variable(n) => n,
                        // Head constants / aggregates / wildcards do
                        // not constrain a column type via inference.
                        // Their type would be locked by the value
                        // itself at evaluation time.
                        _ => continue,
                    };
                    let Some(&derived) = var_types.get(name) else {
                        continue;
                    };
                    let schema = schemas
                        .get_mut(predicate)
                        .expect("predicate in initialized schemas");
                    if col >= schema.len() {
                        // Head arity drift across rules — let the
                        // structural SCC fixpoint surface this as
                        // HeadArityMismatch. Inference doesn't
                        // pre-empt; just skip this column.
                        continue;
                    }
                    match schema[col] {
                        None => {
                            schema[col] = Some(derived);
                            origins.insert((predicate.clone(), col), rule_index);
                            changed = true;
                        }
                        Some(existing) if existing == derived => {
                            // Agreement — silent.
                        }
                        Some(existing) => {
                            let first_rule_index =
                                origins.get(&(predicate.clone(), col)).copied().unwrap_or(0);
                            return Err(InferenceError::ConflictingPredicateColumnType {
                                predicate: predicate.clone(),
                                column: col,
                                first_rule_index,
                                first_type: existing,
                                second_rule_index: rule_index,
                                second_type: derived,
                            });
                        }
                    }
                }
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    // Monotonic invariant: every iteration that changed something
    // replaced a None with a Some(_). The bound `total_columns + 1`
    // strictly exceeds the number of such iterations possible, so
    // failing to converge here indicates a future code change has
    // broken the monotonicity guarantee — a programmer error, not
    // a data error.
    debug_assert!(
        converged,
        "type inference failed to converge within {max_iterations} iterations \
         (monotonicity invariant violated)"
    );
    Ok(schemas)
}

/// Derive the per-variable type map for a single rule, consulting
/// both `base_relations` and currently-inferred SCC schemas.
///
/// Body conflicts (a variable typed two different ways across
/// body atoms within this rule) are NOT surfaced here — that is
/// the responsibility of [`super::typed::derive_vertex_types`],
/// which the typed gate calls before evaluation. This helper is
/// a *forward* propagation pass that prefers the first type seen
/// (in source order) and silently skips later disagreements; the
/// typed gate later catches the disagreement on the same rule
/// using its own walk.
fn derive_rule_var_types(
    rule: &Rule,
    base_relations: &RefRelationStore,
    inferred: &InferredSchemas,
) -> BTreeMap<String, ScalarType> {
    let mut var_types: BTreeMap<String, ScalarType> = BTreeMap::new();
    for literal in &rule.body {
        let body_atom = match literal {
            BodyLiteral::Positive(a) => a,
            _ => continue,
        };
        let schema_opt: Option<&[Option<ScalarType>]> =
            if let Some(rel) = base_relations.get(&body_atom.predicate) {
                // Build a transient "all-Some" view of the base schema.
                // We don't actually need to allocate — handle directly.
                let limit = body_atom.terms.len().min(rel.schema.len());
                for (pos, term) in body_atom.terms[..limit].iter().enumerate() {
                    if let Term::Variable(name) = term {
                        var_types.entry(name.clone()).or_insert(rel.schema[pos]);
                    }
                }
                None
            } else {
                inferred.get(&body_atom.predicate).map(|v| v.as_slice())
            };
        if let Some(schema) = schema_opt {
            let limit = body_atom.terms.len().min(schema.len());
            for (pos, term) in body_atom.terms[..limit].iter().enumerate() {
                if let Term::Variable(name) = term {
                    if let Some(ty) = schema[pos] {
                        var_types.entry(name.clone()).or_insert(ty);
                    }
                }
            }
        }
    }
    var_types
}

/// Build the typed-gate input map for a single rule using
/// inferred SCC schemas alongside base relations.
///
/// Mirrors [`super::typed::derive_vertex_types`]'s contract — same
/// conflict surface ([`super::RefEvalError::ConflictingVariableType`])
/// — but consults `inferred_schemas` whenever a body atom's
/// predicate is not in `base_relations`. Inferred columns marked
/// `None` are treated identically to "predicate absent": they
/// don't type the variable at that position.
///
/// Used by [`super::evaluate_scc_fixpoint_typed`] and
/// [`super::evaluate_fixpoint_typed`] inside their per-rule typed
/// gate to give [`super::analyze_typed`] full type information.
pub(super) fn derive_vertex_types_with_inference(
    rule: &Rule,
    base_relations: &RefRelationStore,
    inferred_schemas: &InferredSchemas,
) -> Result<BTreeMap<String, ScalarType>, super::RefEvalError> {
    /// First-recorded site for a variable; used to populate the
    /// `ConflictingVariableType` report when a second body atom
    /// types the variable differently.
    struct FirstSite {
        predicate: String,
        position: usize,
        ty: ScalarType,
    }
    let mut sites: BTreeMap<String, FirstSite> = BTreeMap::new();
    for literal in &rule.body {
        let body_atom = match literal {
            BodyLiteral::Positive(a) => a,
            _ => continue,
        };
        // Type each position. Base relation wins if both are
        // present (cannot happen — `base_relations` and
        // `inferred_schemas` keys are disjoint by construction in
        // the typed evaluators).
        let position_types: Vec<Option<ScalarType>> =
            if let Some(rel) = base_relations.get(&body_atom.predicate) {
                let limit = body_atom.terms.len().min(rel.schema.len());
                let mut v: Vec<Option<ScalarType>> = vec![None; body_atom.terms.len()];
                for (pos_idx, slot) in v.iter_mut().enumerate().take(limit) {
                    *slot = Some(rel.schema[pos_idx]);
                }
                v
            } else if let Some(schema) = inferred_schemas.get(&body_atom.predicate) {
                let limit = body_atom.terms.len().min(schema.len());
                let mut v: Vec<Option<ScalarType>> = vec![None; body_atom.terms.len()];
                for (pos_idx, slot) in v.iter_mut().enumerate().take(limit) {
                    *slot = schema[pos_idx];
                }
                v
            } else {
                continue; // predicate unknown, no type info
            };
        for (position, term) in body_atom.terms.iter().enumerate() {
            let var_name = match term {
                Term::Variable(name) => name.clone(),
                _ => continue,
            };
            let Some(ty) = position_types[position] else {
                continue;
            };
            match sites.get(&var_name) {
                None => {
                    sites.insert(
                        var_name,
                        FirstSite {
                            predicate: body_atom.predicate.clone(),
                            position,
                            ty,
                        },
                    );
                }
                Some(prior) if prior.ty == ty => {
                    // Agreeing repeat — silent.
                }
                Some(prior) => {
                    return Err(super::RefEvalError::ConflictingVariableType {
                        var: var_name,
                        first_predicate: prior.predicate.clone(),
                        first_position: prior.position,
                        first_type: prior.ty,
                        second_predicate: body_atom.predicate.clone(),
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
