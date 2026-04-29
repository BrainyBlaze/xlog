//! Hypergraph IR for rule-body planning (v0.6.2 Foundation).
//!
//! This module is a **parallel structure** to the existing AST-to-RIR
//! lowering pipeline (see [`crate::lower`]). It does not modify how
//! rules are executed today; instead, it provides a representation
//! suitable for *future* multiway-join planning, plus an eligibility
//! analyzer that decides whether a given rule could be planned as a
//! multiway join or must fall back to the existing binary-join lowering.
//!
//! ## Why a parallel structure
//!
//! PR 1's locked scope is "representation + boundaries + explain +
//! tests". The CPU reference evaluator (PR 2), GPU kernels (PR 3),
//! and integration into the executor (PR 4+) all build on top of
//! this IR — but they don't need it to live inside the existing
//! [`xlog_ir::rir`] tree to do their work. Keeping the hypergraph IR
//! separate keeps PR 1 reviewable in isolation and avoids touching
//! the executor's consumed plan shape until the planner is ready.
//!
//! ## What this PR ships
//!
//! * [`ir::HypergraphRule`] — vertices = body variables, hyperedges =
//!   positive body atoms. Vertices currently carry source name only;
//!   type / mode / selectivity metadata will attach to vertices in
//!   later PRs (PR 2 introduces the typed-analyze entry point that
//!   threads inferred [`xlog_core::ScalarType`]s through). Predicate
//!   names + arities live on hyperedges.
//! * [`eligibility::analyze`] — decides Eligible vs Ineligible with a
//!   structured [`eligibility::Boundary`] list explaining why.
//! * [`var_order::VariableOrder`] — trait with a single trivial
//!   [`var_order::AppearanceOrder`] implementation. Cost models slot
//!   in here later without breaking the trait shape.
//! * [`explain::explain`] — stable textual representation of the
//!   triple (hypergraph, eligibility verdict, variable order).
//!
//! ## What this PR explicitly does NOT ship
//!
//! * No CPU reference evaluator — that is PR 2.
//! * No GPU code or CUDA touches.
//! * No cost model beyond the trivial [`var_order::AppearanceOrder`].
//! * No integration into [`crate::lower`] or the executor — the
//!   hypergraph IR is constructed on demand from a [`crate::ast::Rule`]
//!   and consumed in tests + (later) the reference evaluator.

pub mod eligibility;
pub mod explain;
pub mod ir;
pub mod var_order;

pub use eligibility::{analyze, Boundary, Eligibility};
pub use explain::explain;
pub use ir::{Hyperedge, HypergraphRule, Vertex, VertexId};
pub use var_order::{AppearanceOrder, VariableOrder};
