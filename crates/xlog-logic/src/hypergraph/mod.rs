//! Hypergraph IR + WCOJ oracle stack (v0.6.2).
//!
//! A **parallel structure** to the existing AST-to-RIR lowering pipeline
//! (see [`crate::lower`]). The executor's consumed plan shape is
//! untouched — every consumer here is opt-in and pure-Rust.
//!
//! ## What this stack ships (PRs 1–6, all on local main)
//!
//! * **PR 1 — Foundation.**
//!   - [`ir::HypergraphRule`] — vertices = body variables, hyperedges =
//!     positive body atoms.
//!   - [`eligibility::analyze`] / [`eligibility::analyze_typed`] —
//!     decide Eligible vs Ineligible with a structured
//!     [`eligibility::Boundary`] list explaining why.
//!   - [`var_order::VariableOrder`] / [`var_order::AppearanceOrder`] —
//!     trait + trivial impl. Cost models slot in here later.
//!   - [`explain::explain`] — stable textual representation.
//! * **PR 2 — CPU reference evaluator.**
//!   [`reference::evaluate_rule`] over [`reference::RefRelationStore`];
//!   the WCOJ correctness oracle for all later kernels.
//! * **PR 3 — Single-target fixpoint.**
//!   [`fixpoint::evaluate_fixpoint`] for recursive single-predicate
//!   rules (transitive closure shape).
//! * **PR 4 — Multi-predicate SCC fixpoint.**
//!   [`scc::evaluate_scc_fixpoint`] for mutually-recursive predicate
//!   groups; correctness oracle for mixed-execution kernels.
//! * **PR 5 — Typed oracle gate.**
//!   [`typed::evaluate_rule_typed`] +
//!   [`typed::evaluate_fixpoint_typed`] +
//!   [`typed::evaluate_scc_fixpoint_typed`]: schema-driven type
//!   derivation from [`reference::RefRelationStore`] feeds
//!   [`eligibility::analyze_typed`] for join-key support gating.
//!   Locked policy: unknown-from-base ≠ unsupported.
//! * **PR 6 — Mixed plan contract.**
//!   [`plan::plan_rule`] / [`plan::plan_rules`] dispatch each rule
//!   into [`plan::RulePlan::MultiwayCandidate`] (ready for WCOJ) or
//!   [`plan::RulePlan::BinaryFallback`] (carries every Boundary that
//!   fired). [`plan::explain_plans`] renders a deterministic textual
//!   summary for mixed rule sets.
//!
//! ## What this stack still does NOT ship
//!
//! * No GPU / CUDA kernels — WCOJ kernel work is the next slice.
//! * No cost model beyond [`var_order::AppearanceOrder`].
//! * No integration into [`crate::lower`] or the executor — the
//!   hypergraph stack is constructed on demand from
//!   [`crate::ast::Rule`] values and consumed in tests, the reference
//!   oracles, and (later) the planner / mixed-execution evaluator.
//! * No transitive type inference across recursive SCC predicates —
//!   PR 5 explicitly defers that to a follow-up slice.

pub mod eligibility;
pub mod explain;
pub mod fixpoint;
pub mod inference;
pub mod ir;
pub mod plan;
pub mod reference;
pub mod scc;
pub mod typed;
pub mod var_order;

pub use eligibility::{analyze, analyze_typed, Boundary, Eligibility, WCOJ_SUPPORTED_KEY_TYPES};
pub use explain::explain;
pub use fixpoint::{evaluate_fixpoint, FixpointConfig, FixpointError};
pub use inference::{infer_scc_predicate_schemas, InferenceError, InferredSchemas};
pub use ir::{Hyperedge, HypergraphRule, Vertex, VertexId};
pub use plan::{explain_plans, plan_rule, plan_rules, PlanError, RulePlan};
pub use reference::{evaluate_rule, RefEvalError, RefRelation, RefRelationStore, RefValue};
pub use scc::{evaluate_scc_fixpoint, SccFixpointError};
pub use typed::{evaluate_fixpoint_typed, evaluate_rule_typed, evaluate_scc_fixpoint_typed};
pub use var_order::{AppearanceOrder, VariableOrder};
