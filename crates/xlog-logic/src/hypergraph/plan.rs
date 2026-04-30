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
use super::ir::{HypergraphRule, VertexId};
use super::reference::{RefEvalError, RefRelationStore};
use super::typed::derive_vertex_types;
use super::var_order::{AppearanceOrder, VariableOrder};
use crate::ast::Rule;
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

/// Hard errors from [`plan_rule`] / [`plan_rules`].
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
pub fn plan_rules(
    rules: &[Rule],
    base_relations: &RefRelationStore,
) -> Result<Vec<RulePlan>, PlanError> {
    rules.iter().map(|r| plan_rule(r, base_relations)).collect()
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
