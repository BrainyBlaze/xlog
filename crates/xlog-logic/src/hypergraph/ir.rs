//! Hypergraph IR data types: vertices (variables) and hyperedges (atoms).
//!
//! Construction is via [`HypergraphRule::from_rule`]. The resulting
//! structure preserves the rule's variable identity (no renaming) and
//! the body atoms' source order. Anonymous wildcards (`_`) are NOT
//! treated as vertices — each occurrence is a fresh unconstrained
//! position and contributes nothing to the join graph.

use crate::ast::{Atom, BodyLiteral, Rule, Term};
use std::collections::HashMap;

/// Stable index into a [`HypergraphRule::vertices`] vector.
///
/// Allocated in first-appearance order during construction. Two
/// [`Vertex`]es with the same logical variable name share one
/// `VertexId`. Used by [`Hyperedge::vertex_positions`] and
/// [`crate::hypergraph::var_order::VariableOrder`] implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VertexId(pub usize);

/// A variable in the rule body.
///
/// At PR 1 the only carried metadata is the source name. Type
/// inference results, mode information, and selectivity hints will
/// attach here in later PRs without changing the surrounding shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vertex {
    /// Logic variable name as it appears in the source rule
    /// (e.g. `"X"`, `"Path"`).
    pub name: String,
}

/// A positive body atom, as a hyperedge over [`VertexId`]s.
///
/// `vertex_positions[i] = Some(vid)` means argument position `i` of
/// the atom is the variable `vid`. `vertex_positions[i] = None`
/// means position `i` is either a constant or an anonymous wildcard
/// — both leave the position out of the join graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyperedge {
    /// Predicate name, copied from the source [`Atom`].
    pub predicate: String,
    /// Per-argument vertex assignment. `None` for constants and
    /// anonymous wildcards.
    pub vertex_positions: Vec<Option<VertexId>>,
}

impl Hyperedge {
    /// Arity of the underlying atom.
    pub fn arity(&self) -> usize {
        self.vertex_positions.len()
    }

    /// Distinct variable [`VertexId`]s referenced by this hyperedge,
    /// in source position order, deduplicated.
    pub fn vertices(&self) -> Vec<VertexId> {
        let mut seen = Vec::new();
        for v in self.vertex_positions.iter().flatten() {
            if !seen.contains(v) {
                seen.push(*v);
            }
        }
        seen
    }
}

/// A rule body represented as a hypergraph.
///
/// Vertices are body variables (named only — anonymous wildcards do
/// not participate in the graph). Hyperedges are positive body atoms.
/// `Negated`, `Comparison`, and `IsExpr` body literals are NOT
/// hyperedges; their presence is recorded in
/// [`HypergraphRule::has_negation`] / [`HypergraphRule::has_is_expr`]
/// so the eligibility analyzer can flag them as boundaries without
/// the IR pretending they're join structure.
///
/// Construction is total: every [`Rule`] produces a
/// [`HypergraphRule`]. Whether the rule is *eligible* for multiway
/// planning is a separate question handled by
/// [`crate::hypergraph::eligibility::analyze`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypergraphRule {
    /// Predicate name of the rule head.
    pub head_predicate: String,
    /// Variables in first-appearance order across the body. Empty
    /// for ground facts and bodies that contain only constants.
    pub vertices: Vec<Vertex>,
    /// Positive body atoms in source order.
    pub hyperedges: Vec<Hyperedge>,
    /// True if the head contains an [`crate::ast::AggExpr`].
    pub head_has_aggregation: bool,
    /// True if any body literal is [`BodyLiteral::Negated`].
    pub has_negation: bool,
    /// True if any body literal is [`BodyLiteral::IsExpr`].
    pub has_is_expr: bool,
    /// Number of body [`BodyLiteral::Comparison`] literals (filters).
    /// Comparisons do NOT block multiway eligibility — they are
    /// applied as filters on top of the join — but the count is
    /// recorded so the explain output can show them and so future
    /// PRs can reason about filter selectivity.
    pub comparison_count: usize,
    /// True if the rule is a ground fact (`body.is_empty()`).
    pub is_fact: bool,
}

impl HypergraphRule {
    /// Build a [`HypergraphRule`] from an AST [`Rule`]. Total: never
    /// fails. Anonymous wildcards (`Term::Anonymous`) and constants
    /// (`Term::Integer` / `Float` / `String` / `Symbol`) leave the
    /// hyperedge position as `None`. Aggregate terms in body atoms
    /// are not allowed by the parser (aggregates appear only in rule
    /// heads), but if one is encountered it is treated as a constant
    /// position — the head-aggregation flag captures the eligibility
    /// boundary regardless.
    pub fn from_rule(rule: &Rule) -> Self {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut name_to_id: HashMap<String, VertexId> = HashMap::new();

        let mut hyperedges = Vec::new();
        let mut has_negation = false;
        let mut has_is_expr = false;
        let mut comparison_count = 0;

        for literal in &rule.body {
            match literal {
                BodyLiteral::Positive(atom) => {
                    hyperedges.push(build_hyperedge(atom, &mut vertices, &mut name_to_id));
                }
                BodyLiteral::Negated(_) => {
                    has_negation = true;
                }
                BodyLiteral::Epistemic(_) => {
                    has_negation = true;
                }
                BodyLiteral::Comparison(_) => {
                    comparison_count += 1;
                }
                BodyLiteral::IsExpr(_) => {
                    has_is_expr = true;
                }
                BodyLiteral::Univ(_) => {
                    comparison_count += 1;
                }
            }
        }

        Self {
            head_predicate: rule.head.predicate.clone(),
            vertices,
            hyperedges,
            head_has_aggregation: rule.has_aggregation(),
            has_negation,
            has_is_expr,
            comparison_count,
            is_fact: rule.is_fact(),
        }
    }

    /// Number of distinct variables referenced by the body. Equals
    /// `self.vertices.len()`.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of positive body atoms.
    pub fn hyperedge_count(&self) -> usize {
        self.hyperedges.len()
    }

    /// Look up a vertex by id. Panics if `id` is out of range —
    /// `VertexId`s are only allocated by [`Self::from_rule`], so an
    /// out-of-range id is a programmer error, not a data error.
    pub fn vertex(&self, id: VertexId) -> &Vertex {
        &self.vertices[id.0]
    }

    /// Iterate over vertex ids in allocation (first-appearance) order.
    pub fn vertex_ids(&self) -> impl Iterator<Item = VertexId> + '_ {
        (0..self.vertices.len()).map(VertexId)
    }
}

fn build_hyperedge(
    atom: &Atom,
    vertices: &mut Vec<Vertex>,
    name_to_id: &mut HashMap<String, VertexId>,
) -> Hyperedge {
    let mut positions = Vec::with_capacity(atom.terms.len());
    for term in &atom.terms {
        let pos = match term {
            Term::Variable(name) => {
                if let Some(id) = name_to_id.get(name) {
                    Some(*id)
                } else {
                    let id = VertexId(vertices.len());
                    vertices.push(Vertex { name: name.clone() });
                    name_to_id.insert(name.clone(), id);
                    Some(id)
                }
            }
            // Anonymous wildcards do not participate in the join
            // graph: each occurrence is a fresh unconstrained
            // position and `_` shared across atoms is by convention
            // not the same variable.
            Term::Anonymous => None,
            // Constants and aggregate terms in body positions are
            // treated as fixed values — no vertex.
            Term::Integer(_)
            | Term::Float(_)
            | Term::String(_)
            | Term::Symbol(_)
            | Term::Aggregate(_)
            | Term::List(_)
            | Term::Cons { .. }
            | Term::Compound { .. }
            | Term::PredRef(_) => None,
        };
        positions.push(pos);
    }
    Hyperedge {
        predicate: atom.predicate.clone(),
        vertex_positions: positions,
    }
}
