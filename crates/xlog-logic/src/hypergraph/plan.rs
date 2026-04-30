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
//! inputs. [`explain_plans`] sorts plans by `head_predicate`
//! (lexicographic), ties broken by input position; the output is
//! identical regardless of input order, locked by
//! `explain_plans_is_deterministic_and_sorts_by_head_then_position`.

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

/// Render a deterministic textual explain of a plan slice.
///
/// Plans are sorted by `head_predicate` (lexicographic), with
/// ties broken by their original input position. The displayed
/// per-line index is then a **per-head rank** (0-based, counting
/// only same-head plans encountered earlier in sorted order) —
/// using a per-head rank rather than the absolute input position
/// keeps the output identical regardless of caller-side
/// insertion order, while still distinguishing multiple rules
/// under the same head.
///
/// One line per rule, format:
///
/// ```text
/// {head_predicate}/{per_head_rank}: multiway vars=[X, Y, Z]
/// {head_predicate}/{per_head_rank}: binary-fallback boundaries=[BodyNegation, ...]
/// ```
pub fn explain_plans(plans: &[RulePlan]) -> String {
    let mut indexed: Vec<(usize, &RulePlan)> = plans.iter().enumerate().collect();
    indexed.sort_by(|(ia, a), (ib, b)| head_of(a).cmp(head_of(b)).then_with(|| ia.cmp(ib)));
    let mut out = String::new();
    let mut last_head: Option<&str> = None;
    let mut rank: usize = 0;
    for (_input_index, plan) in indexed {
        let head = head_of(plan);
        match last_head {
            Some(prev) if prev == head => rank += 1,
            _ => rank = 0,
        }
        last_head = Some(head);
        match plan {
            RulePlan::MultiwayCandidate {
                head_predicate,
                hypergraph,
                variable_order,
            } => {
                let names: Vec<&str> = variable_order
                    .iter()
                    .map(|vid| hypergraph.vertex(*vid).name.as_str())
                    .collect();
                let _ = writeln!(
                    out,
                    "{head_predicate}/{rank}: multiway vars=[{}]",
                    names.join(", ")
                );
            }
            RulePlan::BinaryFallback {
                head_predicate,
                boundaries,
            } => {
                let _ = writeln!(
                    out,
                    "{head_predicate}/{rank}: binary-fallback boundaries={boundaries:?}"
                );
            }
        }
    }
    out
}

fn head_of(plan: &RulePlan) -> &str {
    match plan {
        RulePlan::MultiwayCandidate { head_predicate, .. } => head_predicate,
        RulePlan::BinaryFallback { head_predicate, .. } => head_predicate,
    }
}
