//! Mixed plan contract: dispatch each rule into either the future
//! WCOJ multiway path or the existing binary-fallback lowering.
//!
//! Builds on PR 1's [`super::analyze_typed`] and the PR 5 typed
//! gate's vertex-type derivation to assemble per-rule
//! [`RulePlan`] values that downstream callers (planner,
//! mixed-execution evaluator, kernel test harness) consume to
//! decide which path each rule takes. This slice ships only the
//! contract — no executor integration, no RIR lowering, no CUDA,
//! no cost model beyond
//! [`super::AppearanceOrder`](super::var_order::AppearanceOrder).
//!
//! ## Verdicts
//!
//! Each rule produces exactly one of:
//!
//! * [`RulePlan::MultiwayCandidate`] — cleared the typed gate;
//!   ready for WCOJ. Carries the [`HypergraphRule`] and the
//!   [`super::AppearanceOrder`]-resolved variable order so the
//!   future kernel does not re-derive them.
//! * [`RulePlan::BinaryFallback`] — failed the typed gate on at
//!   least one structural / type-coverage [`Boundary`]. Carries
//!   **every** [`Boundary`] that fired so explain output and
//!   downstream callers see all reasons rather than only the
//!   first one.
//!
//! Type *conflicts* — a variable that gets contradictory types
//! from different body atoms — are NOT verdicts. They surface as
//! [`PlanError::ConflictingVariableType`]; the planner refuses to
//! plan a rule whose fixture is internally contradictory. Caller
//! must fix the fixture before re-planning.
//!
//! ## Determinism
//!
//! [`plan_rule`] / [`plan_rules`] are pure functions of their
//! inputs. [`explain_plans`] is **canonical**: plans are sorted
//! by `head_predicate` (lexicographic), with same-head ties
//! broken by the rendered line content itself — string-lex on
//! the verdict tag (so `binary-fallback` < `multiway`), then on
//! the boundary list or variable-order vector. Input position is
//! never the tie-breaker, so the output is identical for any
//! permutation of the input, including reversal of same-head
//! rules. Locked by
//! `explain_plans_is_canonical_under_same_head_reorder`.

use super::eligibility::{analyze_typed, Boundary, Eligibility};
use super::inference::{
    derive_vertex_types_with_inference, infer_scc_predicate_schemas, InferenceError,
};
use super::ir::{HypergraphRule, VertexId};
use super::reference::{RefEvalError, RefRelationStore};
use super::typed::derive_vertex_types;
use super::var_order::{AppearanceOrder, VariableOrder};
use crate::ast::Rule;
use std::collections::BTreeMap;
use std::fmt::Write;
use xlog_core::ScalarType;

/// Plan choice for a single rule.
///
/// Mirrors the dispatch contract: every rule goes either to the
/// future multiway WCOJ path or to the existing binary-join
/// fallback. The variant carries the metadata the downstream
/// path needs to act on the verdict — for multiway, the
/// hypergraph and a resolved variable order; for fallback, the
/// boundary list explaining why.
#[derive(Debug, Clone, PartialEq)]
pub enum RulePlan {
    /// Rule cleared the typed gate. Ready for the future WCOJ
    /// kernel path. The structural [`super::evaluate_rule`] /
    /// [`super::evaluate_rule_typed`] would also accept this rule
    /// today; PR 6 carries no execution itself, just the dispatch
    /// metadata.
    MultiwayCandidate {
        /// Predicate name of the rule head, copied from the
        /// source [`Rule`] for diagnostic / explain use.
        head_predicate: String,
        /// Hypergraph IR built from the rule body — carried so
        /// the future kernel does not rebuild it.
        hypergraph: HypergraphRule,
        /// Variable order produced by
        /// [`super::AppearanceOrder`](super::var_order::AppearanceOrder).
        /// Length equals `hypergraph.vertex_count()`.
        variable_order: Vec<VertexId>,
    },
    /// Rule cannot be planned as multiway. Caller must use the
    /// existing binary-join lowering path. The boundaries vector
    /// is non-empty and lists every reason the typed gate
    /// rejected the rule; preserving all of them keeps explain
    /// output and downstream telemetry honest about cumulative
    /// fallback drivers.
    BinaryFallback {
        /// Predicate name of the rule head.
        head_predicate: String,
        /// Every [`Boundary`] that fired for this rule.
        boundaries: Vec<Boundary>,
    },
}

/// Hard errors from [`plan_rule`] / [`plan_rules`] / [`plan_scc_rules`].
///
/// Distinct from [`RulePlan::BinaryFallback`]: a fallback verdict
/// means the rule is plannable, just on a different path. A plan
/// error means the rule cannot be planned at all under the
/// current fixture and must be fixed by the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    /// Same shape as [`RefEvalError::ConflictingVariableType`].
    /// Mirrored here (rather than re-exporting the eval-error
    /// variant) so the planner's error type is independent of the
    /// evaluator's.
    ConflictingVariableType {
        /// Variable name as it appears in the source rule.
        var: String,
        /// Predicate of the first atom that typed `var`.
        first_predicate: String,
        /// 0-based argument position within `first_predicate`.
        first_position: usize,
        /// Schema type at `(first_predicate, first_position)`.
        first_type: ScalarType,
        /// Predicate of the conflicting atom.
        second_predicate: String,
        /// 0-based argument position within `second_predicate`.
        second_position: usize,
        /// Schema type at `(second_predicate, second_position)`.
        second_type: ScalarType,
    },
    /// Cross-rule head-column conflict detected during PR 8 SCC
    /// type inference. Two rules contributing to the same head
    /// predicate disagree on the type of the same column.
    ///
    /// Mirrors [`RefEvalError::InferenceConflict`] so callers
    /// pattern-matching on plan errors can treat inference and
    /// eval conflicts symmetrically. Surfaced only by
    /// [`plan_scc_rules`]; per-rule [`plan_rule`] / [`plan_rules`]
    /// don't run inference (no group context).
    InferenceConflict {
        /// Head predicate name where the conflict was detected.
        predicate: String,
        /// 0-based column index where types disagree.
        column: usize,
        /// Rule index (within the predicate's group) that first
        /// typed the column.
        first_rule_index: usize,
        /// Type derived from the first rule's body.
        first_type: ScalarType,
        /// Rule index (within the predicate's group) that
        /// disagrees.
        second_rule_index: usize,
        /// Type derived from the conflicting rule's body.
        second_type: ScalarType,
    },
    /// A rule grouped under predicate `group_key` heads a
    /// different predicate. Surfaces from [`plan_scc_rules`] only;
    /// per-rule [`plan_rule`] / [`plan_rules`] don't have group
    /// context to validate against.
    ///
    /// Mirrors [`super::SccFixpointError::RuleHeadPredicateMismatch`]
    /// so the planner and the SCC fixpoint evaluator agree on the
    /// diagnostic for this fixture class. Without symmetry, a
    /// caller driving plan-then-evaluate would see the planner
    /// say "MultiwayCandidate" while the evaluator says
    /// "RuleHeadPredicateMismatch" — the same disagreement
    /// pattern PR 9 closed for unsupported-key cases.
    RuleHeadPredicateMismatch {
        /// `BTreeMap` key under which the rule was grouped.
        group_key: String,
        /// Index of the rule within that group.
        rule_index: usize,
        /// Head predicate observed on the rule.
        observed: String,
    },
}

/// Plan a single rule. See module-level docs for the contract.
pub fn plan_rule(rule: &Rule, base_relations: &RefRelationStore) -> Result<RulePlan, PlanError> {
    let vertex_types = match derive_vertex_types(rule, base_relations) {
        Ok(map) => map,
        Err(RefEvalError::ConflictingVariableType {
            var,
            first_predicate,
            first_position,
            first_type,
            second_predicate,
            second_position,
            second_type,
        }) => {
            return Err(PlanError::ConflictingVariableType {
                var,
                first_predicate,
                first_position,
                first_type,
                second_predicate,
                second_position,
                second_type,
            });
        }
        Err(other) => {
            // `derive_vertex_types` only ever returns
            // ConflictingVariableType in the Err arm — the
            // conflict variant is the only one constructed by the
            // function body. If a future change broadens the
            // signature, we'd want a fresh PlanError variant
            // rather than silently swallow.
            unreachable!(
                "derive_vertex_types contract returns only ConflictingVariableType: got {other:?}"
            );
        }
    };
    let hypergraph = HypergraphRule::from_rule(rule);
    match analyze_typed(&hypergraph, &vertex_types) {
        Eligibility::Eligible => {
            let variable_order = AppearanceOrder.order(&hypergraph);
            Ok(RulePlan::MultiwayCandidate {
                head_predicate: rule.head.predicate.clone(),
                hypergraph,
                variable_order,
            })
        }
        Eligibility::Ineligible(boundaries) => Ok(RulePlan::BinaryFallback {
            head_predicate: rule.head.predicate.clone(),
            boundaries,
        }),
    }
}

/// Plan a slice of rules. Order-preserving: `plans[i]` is the
/// plan for `rules[i]`.
///
/// Stops on the first [`PlanError`]. Callers that want
/// best-effort multi-rule planning should call [`plan_rule`]
/// per-rule and collect verdicts themselves.
///
/// Per-rule typing only — no SCC inference. For mutually
/// recursive predicate groups whose join keys are anchored only
/// through SCC body atoms, use [`plan_scc_rules`] so the same
/// transitive type inference that
/// [`super::evaluate_scc_fixpoint_typed`] runs is applied
/// before each rule's verdict.
pub fn plan_rules(
    rules: &[Rule],
    base_relations: &RefRelationStore,
) -> Result<Vec<RulePlan>, PlanError> {
    rules.iter().map(|r| plan_rule(r, base_relations)).collect()
}

/// Plan a mutually-recursive rule group with PR 8 transitive
/// type inference engaged.
///
/// Mirrors the input shape: returns a
/// `BTreeMap<predicate, Vec<RulePlan>>` where `result[p][i]`
/// corresponds to `rules[p][i]`. Each rule's verdict is computed
/// after running [`infer_scc_predicate_schemas`] over the full
/// group, so a recursive-only join key whose type is established
/// only via inference is now flagged with
/// [`super::Boundary::UnsupportedKeyType`] consistent with
/// [`super::evaluate_scc_fixpoint_typed`].
///
/// ## Why this exists separately from [`plan_rules`]
///
/// [`plan_rule`] / [`plan_rules`] type variables from
/// `base_relations` only — they have no group context and can't
/// run inference. Without [`plan_scc_rules`], a planner driving
/// the per-rule API would mark `even(X, Y) :- odd(X, Z), odd(Z, Y)`
/// as [`RulePlan::MultiwayCandidate`] even when `odd`'s schema
/// (inferred via PR 8) propagates an unsupported type to `Z` —
/// i.e., the planner and the SCC evaluator would disagree on
/// the same fixture. PR 9 closes that gap.
///
/// ## Errors
///
/// Returns [`PlanError::InferenceConflict`] for cross-rule
/// head-column conflicts detected during inference,
/// [`PlanError::ConflictingVariableType`] for within-rule body
/// conflicts (both layered the same way as
/// [`super::evaluate_scc_fixpoint_typed`]), and
/// [`PlanError::RuleHeadPredicateMismatch`] for misgrouped rules
/// (rule's head predicate ≠ its `BTreeMap` group key).
///
/// ## Structural-error precedence
///
/// Mirrors the [`super::evaluate_scc_fixpoint_typed`] pre-flight
/// (PR 9): if any rule is misgrouped, the function returns
/// [`PlanError::RuleHeadPredicateMismatch`] BEFORE running
/// inference, so a misgrouped rule whose body would also produce
/// inference conflicts surfaces the structural error first. This
/// keeps planner and evaluator verdicts symmetric for every
/// fixture class.
pub fn plan_scc_rules(
    rules: &BTreeMap<String, Vec<Rule>>,
    base_relations: &RefRelationStore,
) -> Result<BTreeMap<String, Vec<RulePlan>>, PlanError> {
    // Pre-flight: surface RuleHeadPredicateMismatch before
    // inference runs (see "Structural-error precedence" above).
    for (predicate, group) in rules.iter() {
        for (rule_index, rule) in group.iter().enumerate() {
            if &rule.head.predicate != predicate {
                return Err(PlanError::RuleHeadPredicateMismatch {
                    group_key: predicate.clone(),
                    rule_index,
                    observed: rule.head.predicate.clone(),
                });
            }
        }
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
            return Err(PlanError::InferenceConflict {
                predicate,
                column,
                first_rule_index,
                first_type,
                second_rule_index,
                second_type,
            });
        }
    };
    let mut out: BTreeMap<String, Vec<RulePlan>> = BTreeMap::new();
    for (predicate, group) in rules.iter() {
        let mut plans: Vec<RulePlan> = Vec::with_capacity(group.len());
        for rule in group {
            let vertex_types =
                match derive_vertex_types_with_inference(rule, base_relations, &inferred) {
                    Ok(map) => map,
                    Err(RefEvalError::ConflictingVariableType {
                        var,
                        first_predicate,
                        first_position,
                        first_type,
                        second_predicate,
                        second_position,
                        second_type,
                    }) => {
                        return Err(PlanError::ConflictingVariableType {
                            var,
                            first_predicate,
                            first_position,
                            first_type,
                            second_predicate,
                            second_position,
                            second_type,
                        });
                    }
                    Err(other) => unreachable!(
                        "derive_vertex_types_with_inference contract returns only \
                         ConflictingVariableType: got {other:?}"
                    ),
                };
            let hypergraph = HypergraphRule::from_rule(rule);
            let plan = match analyze_typed(&hypergraph, &vertex_types) {
                Eligibility::Eligible => {
                    let variable_order = AppearanceOrder.order(&hypergraph);
                    RulePlan::MultiwayCandidate {
                        head_predicate: rule.head.predicate.clone(),
                        hypergraph,
                        variable_order,
                    }
                }
                Eligibility::Ineligible(boundaries) => RulePlan::BinaryFallback {
                    head_predicate: rule.head.predicate.clone(),
                    boundaries,
                },
            };
            plans.push(plan);
        }
        out.insert(predicate.clone(), plans);
    }
    Ok(out)
}

/// Render a canonical textual explain of a plan slice.
///
/// Plans are sorted by `head_predicate` (lexicographic), with
/// same-head ties broken by the **rendered line body** itself
/// under string-lex ordering (so the verdict tag
/// `binary-fallback` sorts before `multiway`; the boundary list
/// or variable-order vector breaks remaining ties). Input
/// position is never consulted, so the output is identical for
/// any permutation of the input, including reversal of same-head
/// rules. Locked by
/// `explain_plans_is_canonical_under_same_head_reorder`.
///
/// The displayed per-line index is a **per-head rank** (0-based,
/// counting only same-head plans encountered earlier in sorted
/// order) so multiple rules under one head remain distinguishable
/// without leaking input position into the canonical form.
///
/// One line per rule, format:
///
/// ```text
/// {head_predicate}/{per_head_rank}: multiway vars=[X, Y, Z]
/// {head_predicate}/{per_head_rank}: binary-fallback boundaries=[BodyNegation, ...]
/// ```
pub fn explain_plans(plans: &[RulePlan]) -> String {
    // Pre-render each plan's *body* (everything after the
    // "head/rank: " prefix). The body is the canonical content
    // fingerprint we sort by — same head + same body → same
    // line, regardless of input position.
    let mut bodies: Vec<(&str, String)> =
        plans.iter().map(|p| (head_of(p), render_body(p))).collect();
    bodies.sort_by(|(ha, ba), (hb, bb)| ha.cmp(hb).then_with(|| ba.cmp(bb)));
    let mut out = String::new();
    let mut last_head: Option<&str> = None;
    let mut rank: usize = 0;
    for (head, body) in &bodies {
        match last_head {
            Some(prev) if prev == *head => rank += 1,
            _ => rank = 0,
        }
        last_head = Some(*head);
        let _ = writeln!(out, "{head}/{rank}: {body}");
    }
    out
}

/// Render the verdict-and-payload portion of a plan's explain
/// line. Used both for output assembly and (importantly) as the
/// same-head sort fingerprint in [`explain_plans`].
///
/// Under string-lex ordering on rendered bodies, the verdict tag
/// `binary-fallback` sorts before `multiway` — fallbacks
/// surface first within a predicate, so a reader scanning the
/// canonical explain can spot the dispatch obstacles before the
/// successful candidates.
fn render_body(plan: &RulePlan) -> String {
    match plan {
        RulePlan::MultiwayCandidate {
            hypergraph,
            variable_order,
            ..
        } => {
            let names: Vec<&str> = variable_order
                .iter()
                .map(|vid| hypergraph.vertex(*vid).name.as_str())
                .collect();
            format!("multiway vars=[{}]", names.join(", "))
        }
        RulePlan::BinaryFallback { boundaries, .. } => {
            format!("binary-fallback boundaries={boundaries:?}")
        }
    }
}

fn head_of(plan: &RulePlan) -> &str {
    match plan {
        RulePlan::MultiwayCandidate { head_predicate, .. } => head_predicate,
        RulePlan::BinaryFallback { head_predicate, .. } => head_predicate,
    }
}
