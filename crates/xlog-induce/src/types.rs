//! Public types for the exact-induction engine.
//!
//! Mirrors the pyxlog `ExactInductionResult` / `ScoredCandidate` dataclasses
//! but speaks `RelId` instead of relation names — name resolution happens at
//! the pyxlog boundary in `crates/pyxlog/src/ilp_exact.rs`.

use xlog_core::RelId;

/// The four canonical 2-body topologies scored by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topology {
    /// `H(X,Y) :- L(X,Z), R(Z,Y)` — join on intermediate `Z`.
    Chain,
    /// `H(X,Y) :- L(X,Y), R(X,Y)` — shared `X,Y`.
    Star,
    /// `H(X,Y) :- L(X,Z), R(X,Y)` — left introduces `Z`.
    Fanout,
    /// `H(X,Y) :- L(X,Y), R(Z,Y)` — right introduces `Z`.
    Fanin,
}

impl Topology {
    /// All four topologies, in the scoring order used by the engine.
    pub const ALL: [Topology; 4] = [
        Topology::Chain,
        Topology::Star,
        Topology::Fanout,
        Topology::Fanin,
    ];

    /// Stable string form exposed to pyxlog / Python.
    pub fn as_str(&self) -> &'static str {
        match self {
            Topology::Chain => "chain",
            Topology::Star => "star",
            Topology::Fanout => "fanout",
            Topology::Fanin => "fanin",
        }
    }
}

/// Engine configuration for one exact-induction request.
#[derive(Debug, Clone, Copy)]
pub struct ExactInductionConfig {
    /// Number of top candidates to keep per topology (e.g. `2`).
    pub k_per_topology: u32,
    /// Reserved for future use; the native engine is inherently deterministic.
    pub deterministic: bool,
}

/// One scored `(left, right)` candidate for a single topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoredCandidate {
    pub topology: Topology,
    pub head_rel_idx: RelId,
    pub left_rel_idx: RelId,
    pub right_rel_idx: RelId,
    pub positives_covered: u32,
    pub negatives_covered: u32,
    /// 0-indexed within topology, after lexicographic ranking.
    pub local_rank: u32,
    /// Positive coverage of the next-ranked candidate in the same topology,
    /// or `0` if no next candidate exists.
    pub next_positives_covered: u32,
    /// Negative coverage of the next-ranked candidate.
    pub next_negatives_covered: u32,
    /// Count of candidates sharing the same `(positives_covered, negatives_covered)`.
    pub tie_class_size: u32,
}

/// Combined result from one `induce_exact()` call.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ExactInductionResult {
    /// Up to `k_per_topology × 4` candidates, grouped by topology then rank.
    pub candidates: Vec<ScoredCandidate>,
    /// Total number of `(topology, left, right)` triples scored.
    pub total_scored: u32,
    /// Number of body candidates considered (after validation).
    pub candidate_count: u32,
    /// Number of positive examples.
    pub positive_count: u32,
    /// Number of negative examples.
    pub negative_count: u32,
}
