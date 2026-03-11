use std::collections::HashMap;
use xlog_core::RelId;

/// Tracks delta relation name mappings during semi-naive fixpoint iteration.
///
/// Each recursive predicate gets a synthetic delta relation (with a unique RelId
/// and store name). This tracker maps predicate names to their delta identifiers.
pub(crate) struct DeltaRelationTracker {
    /// Maps predicate name → (delta RelId, delta store name)
    entries: HashMap<String, (RelId, String)>,
}

impl DeltaRelationTracker {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
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

    /// Iterate over all (pred_name, (rel_id, store_name)) entries.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, &(RelId, String))> {
        self.entries.iter()
    }

    /// Consume the tracker, yielding owned entries for cleanup.
    pub(crate) fn into_inner(self) -> HashMap<String, (RelId, String)> {
        self.entries
    }
}
