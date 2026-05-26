//! Public types for the exact-induction engine.
//!
//! Mirrors the pyxlog `ExactInductionResult` / `ScoredCandidate` dataclasses
//! but speaks `RelId` instead of relation names — name resolution happens at
//! the pyxlog boundary in `crates/pyxlog/src/ilp_exact.rs`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

/// Origin class for a rule known to the induction/provenance surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleSourceKind {
    /// Rule was authored in source text.
    Source,
    /// Rule was produced by a native induction or compiler-generation step.
    Generated,
    /// Rule was mined by an induction workflow and registered with provenance.
    Mined,
    /// Rule came from an imported module.
    Imported,
    /// Rule was inserted at runtime through an API.
    RuntimeInjected,
}

/// One source row supporting a generated rule candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InductionSupportRow {
    /// Supporting relation name.
    pub relation: String,
    /// Row index within the supporting relation.
    pub row_index: u64,
    /// Stable row hash supplied by the caller or relation loader.
    pub row_hash: String,
}

impl InductionSupportRow {
    /// Create one support-row reference.
    pub fn new(relation: impl Into<String>, row_index: u64, row_hash: impl Into<String>) -> Self {
        Self {
            relation: relation.into(),
            row_index,
            row_hash: row_hash.into(),
        }
    }
}

/// Rejected candidate rule with support and falsification counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InductionAlternative {
    /// Candidate source fragment.
    pub rule_fragment: String,
    /// Number of positive examples covered by the rejected alternative.
    pub support_count: u64,
    /// Number of falsifying examples covered by the rejected alternative.
    pub falsification_count: u64,
}

impl InductionAlternative {
    /// Create one rejected-alternative record.
    pub fn new(
        rule_fragment: impl Into<String>,
        support_count: u64,
        falsification_count: u64,
    ) -> Self {
        Self {
            rule_fragment: rule_fragment.into(),
            support_count,
            falsification_count,
        }
    }
}

/// Provenance bundle for a generated or induced rule candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InducedRuleProvenance {
    /// Stable generated-rule id.
    pub rule_id: String,
    /// Rule source fragment.
    pub rule_fragment: String,
    /// Origin class for this rule.
    pub source_kind: RuleSourceKind,
    /// Deterministic hash over rule text and search metadata.
    pub generation_trace_hash: String,
    /// Size of the searched candidate space.
    pub search_space_size: u64,
    /// Predicate inventory available to the search.
    pub predicate_inventory: Vec<String>,
    /// Rows supporting the accepted candidate.
    pub support_rows: Vec<InductionSupportRow>,
    /// Rejected alternatives retained for diagnostics.
    pub rejected_alternatives: Vec<InductionAlternative>,
    /// Number of falsifying rows for the accepted candidate.
    pub falsification_count: u64,
}

impl InducedRuleProvenance {
    /// Create provenance for a generated rule candidate.
    pub fn new(
        rule_fragment: impl Into<String>,
        search_space_size: u64,
        predicate_inventory: Vec<String>,
    ) -> Self {
        let rule_fragment = rule_fragment.into();
        let generation_trace_hash =
            provenance_hash(&rule_fragment, search_space_size, &predicate_inventory);
        Self {
            rule_id: format!("generated:{}", generation_trace_hash),
            rule_fragment,
            source_kind: RuleSourceKind::Generated,
            generation_trace_hash,
            search_space_size,
            predicate_inventory,
            support_rows: Vec::new(),
            rejected_alternatives: Vec::new(),
            falsification_count: 0,
        }
    }

    /// Attach support rows.
    pub fn with_support_rows(mut self, support_rows: Vec<InductionSupportRow>) -> Self {
        self.support_rows = support_rows;
        self
    }

    /// Attach rejected alternatives.
    pub fn with_rejected_alternatives(
        mut self,
        rejected_alternatives: Vec<InductionAlternative>,
    ) -> Self {
        self.rejected_alternatives = rejected_alternatives;
        self
    }

    /// Attach the accepted candidate's falsification count.
    pub fn with_falsification_count(mut self, falsification_count: u64) -> Self {
        self.falsification_count = falsification_count;
        self
    }
}

/// In-memory registry for induced rule provenance records.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InducedRuleRegistry {
    rules: Vec<InducedRuleProvenance>,
}

impl InducedRuleRegistry {
    /// Create an empty induced-rule registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one induced or generated rule and return its stable rule id.
    pub fn register(&mut self, provenance: InducedRuleProvenance) -> String {
        let rule_id = provenance.rule_id.clone();
        self.rules.push(provenance);
        rule_id
    }

    /// Number of registered induced rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// True when no induced rules have been registered.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Registered induced rules in insertion order.
    pub fn rules(&self) -> &[InducedRuleProvenance] {
        &self.rules
    }
}

fn provenance_hash(
    rule_fragment: &str,
    search_space_size: u64,
    predicate_inventory: &[String],
) -> String {
    let mut hasher = DefaultHasher::new();
    rule_fragment.hash(&mut hasher);
    search_space_size.hash(&mut hasher);
    predicate_inventory.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
