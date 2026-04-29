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
//! unsupported by the binary-fallback executor we share constraints
//! with today (>4 keys per `pack_keys_gpu_on_stream`). Each boundary
//! is reported separately so the explain output and tests can lock
//! in *why* a rule fell back.

use super::ir::{HypergraphRule, VertexId};
use xlog_core::ScalarType;

/// The maximum number of distinct join-key variables supported by
/// the existing binary-fallback executor. Borrowed verbatim from the
/// `pack_keys_gpu_on_stream` constraint in
/// `xlog-cuda/src/provider/relational.rs`.
///
/// This is a **binary-fallback** constraint, not a hypergraph
/// property. Future WCOJ kernels may have a different (or no) limit
/// — when that happens, this constant is replaced by per-executor
/// caps and the [`Boundary::JoinKeysExceedBinaryFallbackLimit`]
/// variant grows an executor-context discriminator. PR 1 ships the
/// single-executor world.
pub const BINARY_FALLBACK_KEY_LIMIT: usize = 4;

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
    /// True for [`Eligibility::Eligible`].
    pub fn is_eligible(&self) -> bool {
        matches!(self, Eligibility::Eligible)
    }

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
        /// Observed distinct join-key count.
        count: usize,
        /// Hard limit borrowed from the binary-fallback executor
        /// (see [`BINARY_FALLBACK_KEY_LIMIT`]).
        limit: usize,
    },
    /// A join-key variable has a [`ScalarType`] not supported by
    /// the executor under consideration.
    ///
    /// **Not yet produced by [`analyze`] in PR 1.** Variable types
    /// are inferred by [`crate::typeinfer`] during lowering and are
    /// not available at hypergraph construction. PR 2 introduces a
    /// typed-analyze entry point that takes a per-variable type map
    /// and emits this boundary; the variant is defined now so the
    /// public surface is stable across PR boundaries.
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
pub fn analyze(hg: &HypergraphRule) -> Eligibility {
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

    // Count distinct join-key variables. A "join-key variable" at
    // PR 1 is any vertex shared by two or more hyperedges. (Variables
    // that appear in only one atom contribute to projection / output
    // schema but are not join keys.) The binary-fallback executor's
    // pack_keys constraint applies to join keys specifically, so this
    // is the right count to gate on.
    let join_key_count = count_join_keys(hg);
    if join_key_count > BINARY_FALLBACK_KEY_LIMIT {
        boundaries.push(Boundary::JoinKeysExceedBinaryFallbackLimit {
            count: join_key_count,
            limit: BINARY_FALLBACK_KEY_LIMIT,
        });
    }

    if boundaries.is_empty() {
        Eligibility::Eligible
    } else {
        Eligibility::Ineligible(boundaries)
    }
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
