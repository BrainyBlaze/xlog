//! Eligibility analysis: decide whether a [`HypergraphRule`] can be
//! planned as a multiway join, or must fall back to the existing
//! binary-join lowering.
//!
//! "Eligible" here means **could be planned as multiway** — it does
//! NOT mean "must use multiway". A rule with exactly two positive
//! atoms is eligible because the planner could legally choose either
//! a multiway plan or a binary plan; the choice is the planner's, not
//! the analyzer's. The CPU reference evaluator (PR 2) and later GPU
//! kernels (PR 3) consume both shapes.
//!
//! "Ineligible" means at least one [`Boundary`] makes multiway
//! planning either impossible (negation, aggregation in head) or
//! unsupported by the executor context under consideration. Each
//! boundary is reported separately so the explain output and tests
//! can lock in *why* a rule fell back.

use super::ir::{HypergraphRule, VertexId};
use std::collections::BTreeMap;
use xlog_core::ScalarType;

/// The maximum number of distinct join-key variables supported by
/// the existing binary-fallback executor. Borrowed verbatim from the
/// `pack_keys_gpu_on_stream` constraint in
/// `xlog-cuda/src/provider/relational.rs`.
///
/// This is a **binary-fallback** constraint, not a hypergraph
/// property. WCOJ eligibility uses [`ExecutorContext::WcojEligible`]
/// and a separate context limit; hash fallback continues to use
/// this value verbatim.
pub const BINARY_FALLBACK_KEY_LIMIT: usize = 4;

/// The widest K-clique shape the WCOJ planner architecture admits at
/// the eligibility layer. K=7 and K=8 are accepted here so the
/// Phase-2 templates can inherit the same planner contract.
pub const WCOJ_ELIGIBLE_KEY_LIMIT: usize = 8;

/// Executor capability context for join-key width checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorContext {
    /// Existing hash/binary fallback executor. The 4-key
    /// `pack_keys_gpu_on_stream` limit remains binding.
    HashFallback,
    /// WCOJ-capable planner path. K5 through K8 are admissible;
    /// K9+ remains outside the current executor contract.
    WcojEligible,
}

impl ExecutorContext {
    fn join_key_limit(self) -> usize {
        match self {
            Self::HashFallback => BINARY_FALLBACK_KEY_LIMIT,
            Self::WcojEligible => WCOJ_ELIGIBLE_KEY_LIMIT,
        }
    }
}

/// Verdict for a single rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Eligibility {
    /// The rule could be planned as a multiway join. Note: this is
    /// *eligibility*, not *requirement* — the eventual planner is
    /// free to lower an Eligible rule as a binary join if its cost
    /// model says so. Two-atom rules are Eligible: both multiway
    /// and binary are valid lowerings.
    Eligible,
    /// The rule must fall back to binary-join lowering. Each
    /// [`Boundary`] in the vector is one independent reason; the
    /// list is non-empty.
    Ineligible(Vec<Boundary>),
}

impl Eligibility {
    /// Iterate over boundaries (empty for [`Eligibility::Eligible`]).
    pub fn boundaries(&self) -> &[Boundary] {
        match self {
            Eligibility::Eligible => &[],
            Eligibility::Ineligible(bs) => bs,
        }
    }
}

/// One reason a rule is ineligible for multiway planning.
///
/// Boundaries are reported independently — a rule that is *both*
/// negated *and* over the key limit produces two boundaries, not one
/// "first-failed" boundary. This makes the eligibility report stable
/// against ordering changes in future analyzer passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Boundary {
    /// The rule is a ground fact (no body to plan).
    GroundFact,
    /// The rule head contains an aggregation expression. Multiway
    /// joins underneath an aggregation re-introduce the
    /// aggregation-boundary problem the binary lowering already
    /// handles via `GroupBy + Project`; we leave that machinery
    /// intact rather than re-derive it inside multiway.
    HeadAggregation,
    /// The body contains a [`crate::ast::BodyLiteral::Negated`]
    /// literal. Negated literals lower to set difference (`Diff`)
    /// in the binary path — multiway has no equivalent at this
    /// stage of the planner foundation.
    BodyNegation,
    /// The body contains a [`crate::ast::BodyLiteral::IsExpr`]
    /// computed binding. Binding-via-expression introduces a
    /// dependency between body literals that the join graph alone
    /// does not capture.
    BodyIsExpr,
    /// The body has fewer than two positive atoms. Multiway needs
    /// at least two atoms to be meaningful; one-atom or zero-atom
    /// bodies are not multiway candidates regardless of executor.
    InsufficientPositiveAtoms {
        /// Observed number of positive body atoms.
        positive_count: usize,
    },
    /// The total number of distinct join-key variables exceeds the
    /// binary-fallback executor's `pack_keys` limit. Carried as a
    /// runtime value so the explain output can name the count.
    /// See [`BINARY_FALLBACK_KEY_LIMIT`].
    JoinKeysExceedBinaryFallbackLimit {
        /// Executor context whose key-width cap was exceeded.
        context: ExecutorContext,
        /// Observed distinct join-key count.
        count: usize,
        /// Hard limit for the selected executor context.
        limit: usize,
    },
    /// A join-key variable has a [`ScalarType`] not supported by
    /// the executor under consideration.
    ///
    /// Emitted by [`analyze_typed`] when a join-key vertex has a
    /// known type (derived from a body atom's relation schema —
    /// see [`crate::hypergraph::typed::evaluate_rule_typed`] and
    /// the typed fixpoint variants) that is outside
    /// [`WCOJ_SUPPORTED_KEY_TYPES`]. Structural [`analyze`] never
    /// emits this variant.
    ///
    /// **Locked policy (PR 5):** unknown ≠ unsupported. A vertex
    /// whose type cannot be derived from the supplied
    /// [`crate::hypergraph::RefRelationStore`] is **not** flagged
    /// here. Transitive type propagation across recursive SCC
    /// predicates is a follow-up slice.
    UnsupportedKeyType {
        /// Source name of the variable whose type is unsupported.
        var: String,
        /// Inferred scalar type that the executor cannot handle.
        ty: ScalarType,
    },
}

/// Analyze a [`HypergraphRule`] and return the eligibility verdict.
///
/// All boundaries are checked; the returned vector is in a stable
/// order matching the [`Boundary`] enum's declaration order (modulo
/// the one-or-zero-positive-atoms check, which is reported with the
/// observed `positive_count`). Order stability matters for the
/// explain-output snapshot tests.
pub fn analyze(hg: &HypergraphRule, context: ExecutorContext) -> Eligibility {
    let mut boundaries = Vec::new();

    if hg.is_fact {
        boundaries.push(Boundary::GroundFact);
    }
    if hg.head_has_aggregation {
        boundaries.push(Boundary::HeadAggregation);
    }
    if hg.has_negation {
        boundaries.push(Boundary::BodyNegation);
    }
    if hg.has_is_expr {
        boundaries.push(Boundary::BodyIsExpr);
    }
    let positive_count = hg.hyperedge_count();
    if !hg.is_fact && positive_count < 2 {
        boundaries.push(Boundary::InsufficientPositiveAtoms { positive_count });
    }

    // Count distinct join-key variables. A "join-key variable" is
    // any vertex shared by two or more hyperedges. Variables that
    // appear in only one atom contribute to projection / output
    // schema but are not join keys.
    let join_key_count = count_join_keys(hg);
    let join_key_limit = context.join_key_limit();
    if join_key_count > join_key_limit {
        boundaries.push(Boundary::JoinKeysExceedBinaryFallbackLimit {
            context,
            count: join_key_count,
            limit: join_key_limit,
        });
    }

    if boundaries.is_empty() {
        Eligibility::Eligible
    } else {
        Eligibility::Ineligible(boundaries)
    }
}

/// Return true when [`analyze`] says the rule is eligible in the
/// selected executor context.
pub fn is_eligible(hg: &HypergraphRule, context: ExecutorContext) -> bool {
    matches!(analyze(hg, context), Eligibility::Eligible)
}

/// Count vertices that appear in two or more hyperedges. These are
/// the variables the planner must equi-join across atoms; vertices
/// in only one hyperedge do not constrain the cross-atom plan and
/// are not join keys.
fn count_join_keys(hg: &HypergraphRule) -> usize {
    let mut occurrences: Vec<usize> = vec![0; hg.vertex_count()];
    for edge in &hg.hyperedges {
        // Count each vertex AT MOST ONCE per hyperedge — a self-join
        // within a single atom (e.g. `p(X, X)`) is not a multi-atom
        // join key.
        for vid in edge.vertices() {
            let VertexId(idx) = vid;
            occurrences[idx] += 1;
        }
    }
    occurrences.iter().filter(|c| **c >= 2).count()
}

/// Scalar types that the WCOJ reference evaluator supports as
/// join-key types. Membership is checked by [`analyze_typed`] when
/// emitting [`Boundary::UnsupportedKeyType`].
///
/// The set is intentionally narrow and not configurable in PR 2:
/// only `U32`, `U64`, and `Symbol` are supported. Future PRs may
/// widen the set as the reference evaluator and (later) GPU
/// kernels grow type coverage; widening is a deliberate
/// configuration change, not a parameter to this function.
pub const WCOJ_SUPPORTED_KEY_TYPES: &[ScalarType] =
    &[ScalarType::U32, ScalarType::U64, ScalarType::Symbol];

/// Typed eligibility analysis.
///
/// Same as [`analyze`], but additionally consults `vertex_types` —
/// a map from variable name to inferred [`ScalarType`] — to emit
/// [`Boundary::UnsupportedKeyType`] for join-key vertices whose
/// type is outside [`WCOJ_SUPPORTED_KEY_TYPES`].
///
/// "Join-key vertex" matches the same definition used by [`analyze`]:
/// a vertex that appears in two or more hyperedges. Projection-only
/// vertices (those appearing in exactly one hyperedge) are NOT
/// checked — their types do not affect WCOJ planning.
///
/// **Locked policy (PR 5): unknown ≠ unsupported.** Vertices
/// missing from `vertex_types` are NOT flagged. The
/// [`crate::hypergraph::typed`] gate populates this map via
/// schema-driven derivation from a
/// [`crate::hypergraph::RefRelationStore`]; vertices anchored only
/// through predicates absent from that store (e.g. an SCC
/// predicate referenced recursively before its first iteration)
/// stay absent and pass through. Transitive type propagation
/// across recursive predicates is a follow-up slice.
pub fn analyze_typed(
    hg: &HypergraphRule,
    vertex_types: &BTreeMap<String, ScalarType>,
    context: ExecutorContext,
) -> Eligibility {
    // Start from the structural verdict so structural boundaries
    // (negation, aggregation, etc.) carry through.
    let base = analyze(hg, context);
    let mut boundaries: Vec<Boundary> = base.boundaries().to_vec();

    let join_key_ids = join_key_vertex_ids(hg);
    for vid in join_key_ids {
        let name = &hg.vertex(vid).name;
        if let Some(&ty) = vertex_types.get(name) {
            if !WCOJ_SUPPORTED_KEY_TYPES.contains(&ty) {
                boundaries.push(Boundary::UnsupportedKeyType {
                    var: name.clone(),
                    ty,
                });
            }
        }
    }

    if boundaries.is_empty() {
        Eligibility::Eligible
    } else {
        Eligibility::Ineligible(boundaries)
    }
}

/// Return the [`VertexId`]s of vertices that appear in two or more
/// hyperedges (the "join key" set), in vertex-id order.
fn join_key_vertex_ids(hg: &HypergraphRule) -> Vec<VertexId> {
    let mut occurrences: Vec<usize> = vec![0; hg.vertex_count()];
    for edge in &hg.hyperedges {
        for vid in edge.vertices() {
            let VertexId(idx) = vid;
            occurrences[idx] += 1;
        }
    }
    occurrences
        .iter()
        .enumerate()
        .filter(|(_, c)| **c >= 2)
        .map(|(i, _)| VertexId(i))
        .collect()
}
