use std::collections::HashMap;
use xlog_core::RelId;

/// Manages delta relation lifecycle during semi-naive fixpoint iteration.
///
/// Each recursive predicate gets a synthetic delta relation (with a unique RelId
/// and store name). The tracker maps predicate names to their delta identifiers,
/// tracks which predicates produced new tuples in the current iteration (convergence),
/// and provides cleanup of synthetic relations on completion.
///
/// The tracker does NOT own CudaBuffers — those live in the executor's RelationStore.
/// It owns the identity mapping and convergence state that was previously scattered
/// inline across execute_recursive_scc().
pub(crate) struct DeltaRelationTracker {
    /// Maps predicate name → (delta RelId, delta store name)
    entries: HashMap<String, (RelId, String)>,
    /// Whether any predicate produced new tuples in the current iteration
    any_changed: bool,
}

impl DeltaRelationTracker {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            any_changed: false,
        }
    }

    /// Register a delta relation for a recursive predicate.
    pub(crate) fn insert(&mut self, pred: String, rel_id: RelId, store_name: String) {
        self.entries.insert(pred, (rel_id, store_name));
    }

    /// Look up the delta (RelId, store_name) for a predicate.
    pub(crate) fn get(&self, pred: &str) -> Option<&(RelId, String)> {
        self.entries.get(pred)
    }

    /// Look up the delta store name for a predicate, returning an error if missing.
    pub(crate) fn delta_name(&self, pred: &str) -> xlog_core::Result<&str> {
        self.entries
            .get(pred)
            .map(|(_rel_id, name)| name.as_str())
            .ok_or_else(|| {
                xlog_core::XlogError::execution_ctx(
                    "delta_name",
                    "missing delta relation",
                    &pred,
                )
            })
    }

    /// Look up the delta RelId for a predicate, returning an error if missing.
    pub(crate) fn delta_rel_id(&self, pred: &str) -> xlog_core::Result<RelId> {
        self.entries
            .get(pred)
            .map(|(rel_id, _name)| *rel_id)
            .ok_or_else(|| {
                xlog_core::XlogError::execution_ctx(
                    "delta_rel_id",
                    "missing delta relation",
                    &pred,
                )
            })
    }

    /// Reset convergence state at the start of a new iteration.
    pub(crate) fn begin_iteration(&mut self) {
        self.any_changed = false;
    }

    /// Record that a predicate produced new tuples (non-empty delta).
    pub(crate) fn mark_changed(&mut self) {
        self.any_changed = true;
    }

    /// Returns true if no predicate produced new tuples in this iteration (fixpoint reached).
    pub(crate) fn is_converged(&self) -> bool {
        !self.any_changed
    }

    /// Consume the tracker, yielding owned entries for cleanup.
    pub(crate) fn into_inner(self) -> HashMap<String, (RelId, String)> {
        self.entries
    }
}
