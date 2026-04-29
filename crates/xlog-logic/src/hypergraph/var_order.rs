//! Variable-ordering interface for multiway-join planning.
//!
//! The variable order is the sequence in which a multiway evaluator
//! binds variables. Different orders produce identical *results* but
//! can vary widely in *cost* (intermediate sizes, work per step). PR
//! 1 defines the trait shape and ships one trivial implementation
//! ([`AppearanceOrder`]) so the rest of the planner has something
//! deterministic to call. Cost-aware implementations slot in here in
//! later PRs without breaking the trait.
//!
//! ## Trait signature rationale
//!
//! [`VariableOrder::order`] takes the full [`HypergraphRule`] (not
//! just a `&[Vertex]`) on purpose: future selectivity-aware
//! implementations need to inspect hyperedge structure to weigh
//! orderings. Taking the whole IR now means PR 1's trivial impl and
//! PR 3's selectivity-aware impl share one signature.

use super::ir::{HypergraphRule, VertexId};

/// Compute a variable order for a [`HypergraphRule`].
///
/// Returned vectors must:
///   * contain every [`VertexId`] in `hg.vertex_ids()` exactly once,
///   * be deterministic for a given input (same `hg` → same output),
///   * not depend on hidden mutable state (e.g. process-wide RNG).
///
/// Determinism is the contract that lets the explain output be
/// snapshot-tested. Implementations that want randomness should
/// expose a seeded constructor and document the seeding policy.
pub trait VariableOrder {
    /// Stable identifier for this order's strategy. Used by the
    /// explain output (e.g. `"appearance"`, `"selectivity-greedy"`).
    fn name(&self) -> &'static str;

    /// Compute the order. See trait-level contract for invariants.
    fn order(&self, hg: &HypergraphRule) -> Vec<VertexId>;
}

/// Trivial variable order: variables in their first-appearance
/// order across the body. Already the construction order produced
/// by [`HypergraphRule::from_rule`], so this is just an
/// `IntoIterator` over `hg.vertex_ids()`.
///
/// Useful as the default order for tests, and as a baseline that
/// future cost-aware implementations can be compared against.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppearanceOrder;

impl VariableOrder for AppearanceOrder {
    fn name(&self) -> &'static str {
        "appearance"
    }

    fn order(&self, hg: &HypergraphRule) -> Vec<VertexId> {
        hg.vertex_ids().collect()
    }
}
