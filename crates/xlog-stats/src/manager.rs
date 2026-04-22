//! Statistics manager for GPU-resident relation metadata.
//!
//! This module provides the [`StatsManager`] type which maintains statistics for all
//! GPU-resident relations and their join selectivities. It is the central coordination
//! point for optimizer cost models and solver heuristics.

use std::collections::HashMap;
use xlog_core::RelId;

use crate::stats::{ColumnStats, JoinSelectivity, RelationStats};

/// Serializable snapshot of collected statistics.
///
/// This is intended for feeding runtime observations back into the compiler/optimizer.
#[derive(Debug, Clone, Default)]
pub struct StatsSnapshot {
    /// Per-relation statistics.
    pub relations: Vec<RelationStats>,
    /// Cached join selectivity models.
    pub join_selectivities: Vec<JoinSelectivity>,
    /// Optional mapping from runtime `RelId` to predicate name.
    ///
    /// When present, consumers should prefer this over raw `RelId` matching to avoid
    /// misapplying statistics across different programs where `RelId`s may be reused.
    pub rel_names: Vec<(RelId, String)>,
}

/// Manages GPU-resident statistics for all relations.
///
/// The `StatsManager` is the central repository for relation statistics and join
/// selectivity information. It provides methods for:
///
/// - Registering new relations and tracking their statistics
/// - Updating cardinality and access patterns
/// - Estimating join cardinalities using cached selectivity data
/// - Managing relation "heat" for LRU-style eviction
///
/// # Thread Safety
///
/// This type is not thread-safe. For concurrent access, wrap in appropriate
/// synchronization primitives (e.g., `RwLock`).
///
/// # Example
///
/// ```ignore
/// use xlog_stats::StatsManager;
/// use xlog_core::RelId;
///
/// let mut mgr = StatsManager::new();
///
/// // Register relations
/// mgr.register_relation(RelId(1));
/// mgr.register_relation(RelId(2));
///
/// // Update statistics
/// mgr.update_cardinality(RelId(1), 10_000);
/// mgr.update_cardinality(RelId(2), 5_000);
///
/// // Estimate join cardinality
/// let estimate = mgr.estimate_join_cardinality(RelId(1), RelId(2), &[0], &[0]);
/// ```
#[derive(Debug, Default)]
pub struct StatsManager {
    /// Per-relation statistics indexed by RelId
    relations: HashMap<RelId, RelationStats>,
    /// Join selectivity cache indexed by (smaller_rel_id, larger_rel_id) for canonical ordering
    join_selectivities: HashMap<(RelId, RelId), JoinSelectivity>,
}

impl StatsManager {
    /// Creates a new empty statistics manager.
    ///
    /// # Returns
    ///
    /// A new `StatsManager` with no registered relations.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new relation for statistics tracking.
    ///
    /// If the relation is already registered, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `rel_id` - The unique identifier for the relation
    pub fn register_relation(&mut self, rel_id: RelId) {
        self.relations
            .entry(rel_id)
            .or_insert_with(|| RelationStats::new(rel_id));
    }

    /// Create a snapshot of all currently tracked statistics.
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            relations: self.relations.values().cloned().collect(),
            join_selectivities: self.join_selectivities.values().cloned().collect(),
            rel_names: Vec::new(),
        }
    }

    /// Merge a previously captured snapshot into this manager.
    ///
    /// Existing entries are overwritten with the snapshot values.
    pub fn merge_snapshot(&mut self, snapshot: &StatsSnapshot) {
        for rel in &snapshot.relations {
            self.register_relation(rel.rel_id);
            if let Some(stats) = self.relations.get_mut(&rel.rel_id) {
                *stats = rel.clone();
            }
        }

        for js in &snapshot.join_selectivities {
            self.set_join_selectivity(
                js.left_rel,
                js.right_rel,
                js.left_keys.clone(),
                js.right_keys.clone(),
                js.selectivity,
            );
        }
    }

    /// Unregisters a relation, removing all associated statistics.
    ///
    /// Also removes any join selectivity entries involving this relation.
    ///
    /// # Arguments
    ///
    /// * `rel_id` - The relation to unregister
    ///
    /// # Returns
    ///
    /// The removed statistics if the relation was registered
    pub fn unregister_relation(&mut self, rel_id: RelId) -> Option<RelationStats> {
        // Remove join selectivities involving this relation
        self.join_selectivities
            .retain(|(left, right), _| *left != rel_id && *right != rel_id);

        self.relations.remove(&rel_id)
    }

    /// Gets immutable reference to relation statistics.
    ///
    /// # Arguments
    ///
    /// * `rel_id` - The relation to look up
    ///
    /// # Returns
    ///
    /// A reference to the statistics if the relation is registered
    pub fn get_relation_stats(&self, rel_id: RelId) -> Option<&RelationStats> {
        self.relations.get(&rel_id)
    }

    /// Gets mutable reference to relation statistics.
    ///
    /// # Arguments
    ///
    /// * `rel_id` - The relation to look up
    ///
    /// # Returns
    ///
    /// A mutable reference to the statistics if the relation is registered
    pub fn get_relation_stats_mut(&mut self, rel_id: RelId) -> Option<&mut RelationStats> {
        self.relations.get_mut(&rel_id)
    }

    /// Updates the cardinality (row count) for a relation.
    ///
    /// If the relation is not registered, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `rel_id` - The relation to update
    /// * `rows` - The new cardinality estimate
    pub fn update_cardinality(&mut self, rel_id: RelId, rows: u64) {
        if let Some(stats) = self.relations.get_mut(&rel_id) {
            stats.update_cardinality(rows);
        }
    }

    /// Updates the byte size estimate for a relation.
    ///
    /// If the relation is not registered, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `rel_id` - The relation to update
    /// * `bytes` - The estimated total size in bytes
    pub fn update_byte_size(&mut self, rel_id: RelId, bytes: u64) {
        if let Some(stats) = self.relations.get_mut(&rel_id) {
            stats.update_byte_size(bytes);
        }
    }

    /// Records an access to a relation, updating its heat and timestamp.
    ///
    /// If the relation is not registered, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `rel_id` - The relation that was accessed
    pub fn record_access(&mut self, rel_id: RelId) {
        if let Some(stats) = self.relations.get_mut(&rel_id) {
            stats.record_access();
        }
    }

    /// Adds column statistics to a relation.
    ///
    /// If the relation is not registered, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `rel_id` - The relation to update
    /// * `col_stats` - The column statistics to add
    pub fn add_column_stats(&mut self, rel_id: RelId, col_stats: ColumnStats) {
        if let Some(stats) = self.relations.get_mut(&rel_id) {
            stats.add_column(col_stats);
        }
    }

    /// Estimates the output cardinality for a join between two relations.
    ///
    /// Uses cached selectivity if available, otherwise uses a default heuristic.
    /// The estimation formula is: `left_card * right_card * selectivity`.
    ///
    /// # Arguments
    ///
    /// * `left_rel` - The left relation in the join
    /// * `right_rel` - The right relation in the join
    /// * `left_keys` - Column indices used as join keys on the left (currently for future use)
    /// * `right_keys` - Column indices used as join keys on the right (currently for future use)
    ///
    /// # Returns
    ///
    /// The estimated output cardinality (minimum of 1)
    pub fn estimate_join_cardinality(
        &self,
        left_rel: RelId,
        right_rel: RelId,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> u64 {
        // Get cardinalities with sensible defaults
        let left_card = self
            .relations
            .get(&left_rel)
            .map(|s| s.cardinality)
            .unwrap_or(1000);
        let right_card = self
            .relations
            .get(&right_rel)
            .map(|s| s.cardinality)
            .unwrap_or(1000);

        // Use canonical key ordering for selectivity lookup
        let key = Self::canonical_join_key(left_rel, right_rel);

        // Try to use cached selectivity
        if let Some(js) = self.join_selectivities.get(&key) {
            return js.estimate_output_rows(left_card, right_card);
        }

        // Try to estimate from column statistics
        if !left_keys.is_empty() && !right_keys.is_empty() {
            if let (Some(left_stats), Some(right_stats)) = (
                self.relations.get(&left_rel),
                self.relations.get(&right_rel),
            ) {
                // Use first key column for selectivity estimation
                let left_distinct = left_stats
                    .get_column(left_keys[0])
                    .map(|c| c.distinct_estimate)
                    .unwrap_or(0);
                let right_distinct = right_stats
                    .get_column(right_keys[0])
                    .map(|c| c.distinct_estimate)
                    .unwrap_or(0);

                if left_distinct > 0 && right_distinct > 0 {
                    let selectivity = JoinSelectivity::estimate_selectivity_from_stats(
                        left_distinct,
                        right_distinct,
                    );
                    return ((left_card as f64 * right_card as f64 * selectivity) as u64).max(1);
                }
            }
        }

        // Default: assume 10% selectivity as a reasonable heuristic
        // This is conservative and avoids underestimating join sizes
        let default_selectivity = 0.1;
        ((left_card as f64 * right_card as f64 * default_selectivity) as u64).max(1)
    }

    /// Records the result of a join execution to improve future estimates.
    ///
    /// Updates the selectivity model using exponential moving average:
    /// `new_selectivity = old_selectivity * 0.7 + observed_selectivity * 0.3`
    ///
    /// # Arguments
    ///
    /// * `left_rel` - The left relation in the join
    /// * `right_rel` - The right relation in the join
    /// * `left_keys` - Column indices used as join keys on the left
    /// * `right_keys` - Column indices used as join keys on the right
    /// * `input_rows` - Product of input relation cardinalities
    /// * `output_rows` - Actual output row count
    pub fn record_join_result(
        &mut self,
        left_rel: RelId,
        right_rel: RelId,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        input_rows: u64,
        output_rows: u64,
    ) {
        let key = Self::canonical_join_key(left_rel, right_rel);

        // Compute observed selectivity
        let observed_selectivity = if input_rows > 0 {
            (output_rows as f64 / input_rows as f64).clamp(0.0, 1.0)
        } else {
            0.1 // Default when no input
        };

        // Update or create the selectivity entry
        let entry = self.join_selectivities.entry(key).or_insert_with(|| {
            let (canonical_left, canonical_right) = key;
            JoinSelectivity::new(canonical_left, canonical_right)
        });

        // Update keys (store in canonical order)
        let (keys_left, keys_right) = if left_rel <= right_rel {
            (left_keys, right_keys)
        } else {
            (right_keys, left_keys)
        };
        entry.left_keys = keys_left;
        entry.right_keys = keys_right;

        // Exponential moving average for selectivity
        const EMA_OLD_WEIGHT: f64 = 0.7;
        const EMA_NEW_WEIGHT: f64 = 0.3;
        entry.selectivity =
            entry.selectivity * EMA_OLD_WEIGHT + observed_selectivity * EMA_NEW_WEIGHT;
    }

    /// Set (or overwrite) the join selectivity between two relations.
    ///
    /// This is useful for seeding the optimizer from external observations (e.g., runtime stats).
    pub fn set_join_selectivity(
        &mut self,
        left_rel: RelId,
        right_rel: RelId,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        selectivity: f64,
    ) {
        let key = Self::canonical_join_key(left_rel, right_rel);
        let entry = self.join_selectivities.entry(key).or_insert_with(|| {
            let (canonical_left, canonical_right) = key;
            JoinSelectivity::new(canonical_left, canonical_right)
        });

        // Store keys in canonical order.
        let (keys_left, keys_right) = if left_rel <= right_rel {
            (left_keys, right_keys)
        } else {
            (right_keys, left_keys)
        };
        entry.set_keys(keys_left, keys_right);
        entry.set_selectivity(selectivity);
    }

    /// Gets the cached join selectivity between two relations.
    ///
    /// # Arguments
    ///
    /// * `left_rel` - One relation in the join
    /// * `right_rel` - The other relation in the join
    ///
    /// # Returns
    ///
    /// A reference to the cached selectivity if present
    pub fn get_join_selectivity(
        &self,
        left_rel: RelId,
        right_rel: RelId,
    ) -> Option<&JoinSelectivity> {
        let key = Self::canonical_join_key(left_rel, right_rel);
        self.join_selectivities.get(&key)
    }

    /// Decays the heat of all relations by a multiplicative factor.
    ///
    /// This should be called periodically (e.g., during garbage collection
    /// or memory pressure events) to allow unused relations to cool down.
    ///
    /// # Arguments
    ///
    /// * `factor` - Multiplicative decay factor (typically 0.0 to 1.0)
    pub fn decay_all_heat(&mut self, factor: f32) {
        for stats in self.relations.values_mut() {
            stats.decay_heat(factor);
        }
    }

    /// Returns the IDs of all "hot" relations above a given heat threshold.
    ///
    /// This is useful for identifying frequently accessed relations that should
    /// be kept in GPU memory.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The minimum heat value to be considered "hot"
    ///
    /// # Returns
    ///
    /// A vector of RelIds for all relations with heat >= threshold
    pub fn hot_relations(&self, threshold: f32) -> Vec<RelId> {
        self.relations
            .iter()
            .filter(|(_, s)| s.heat >= threshold)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns the IDs of all "cold" relations below a given heat threshold.
    ///
    /// This is useful for identifying candidates for eviction from GPU memory.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The maximum heat value to be considered "cold"
    ///
    /// # Returns
    ///
    /// A vector of RelIds for all relations with heat < threshold
    pub fn cold_relations(&self, threshold: f32) -> Vec<RelId> {
        self.relations
            .iter()
            .filter(|(_, s)| s.heat < threshold)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns the total number of registered relations.
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    /// Returns an iterator over all registered relation IDs.
    pub fn relation_ids(&self) -> impl Iterator<Item = RelId> + '_ {
        self.relations.keys().copied()
    }

    /// Returns the total estimated bytes across all relations.
    pub fn total_byte_size(&self) -> u64 {
        self.relations.values().map(|s| s.byte_size).sum()
    }

    /// Returns the total cardinality across all relations.
    pub fn total_cardinality(&self) -> u64 {
        self.relations.values().map(|s| s.cardinality).sum()
    }

    /// Clears all statistics.
    ///
    /// Removes all relation statistics and join selectivities.
    pub fn clear(&mut self) {
        self.relations.clear();
        self.join_selectivities.clear();
    }

    /// Returns canonical key for join selectivity lookup.
    ///
    /// Ensures (smaller_id, larger_id) ordering for consistent lookups.
    #[inline]
    fn canonical_join_key(left: RelId, right: RelId) -> (RelId, RelId) {
        if left <= right {
            (left, right)
        } else {
            (right, left)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;

    #[test]
    fn test_stats_manager_new() {
        let mgr = StatsManager::new();
        assert!(mgr.get_relation_stats(RelId(1)).is_none());
        assert_eq!(mgr.relation_count(), 0);
    }

    #[test]
    fn test_stats_manager_default() {
        let mgr = StatsManager::default();
        assert_eq!(mgr.relation_count(), 0);
        assert!(mgr.get_relation_stats(RelId(42)).is_none());
    }

    #[test]
    fn test_stats_manager_register_relation() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        assert!(mgr.get_relation_stats(RelId(1)).is_some());
        assert_eq!(mgr.relation_count(), 1);
    }

    #[test]
    fn test_stats_manager_register_relation_idempotent() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.update_cardinality(RelId(1), 500);
        mgr.register_relation(RelId(1)); // Should not reset stats
        let stats = mgr.get_relation_stats(RelId(1)).unwrap();
        assert_eq!(stats.cardinality, 500);
    }

    #[test]
    fn test_stats_manager_register_multiple_relations() {
        let mut mgr = StatsManager::new();
        for i in 1..=10 {
            mgr.register_relation(RelId(i));
        }
        assert_eq!(mgr.relation_count(), 10);
        for i in 1..=10 {
            assert!(mgr.get_relation_stats(RelId(i)).is_some());
        }
    }

    #[test]
    fn test_stats_manager_unregister_relation() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.update_cardinality(RelId(1), 1000);

        let removed = mgr.unregister_relation(RelId(1));
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().cardinality, 1000);
        assert!(mgr.get_relation_stats(RelId(1)).is_none());
        assert_eq!(mgr.relation_count(), 0);
    }

    #[test]
    fn test_stats_manager_unregister_removes_join_selectivities() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.register_relation(RelId(3));
        mgr.update_cardinality(RelId(1), 1000);
        mgr.update_cardinality(RelId(2), 500);
        mgr.update_cardinality(RelId(3), 200);

        // Record join results to create selectivity entries
        mgr.record_join_result(RelId(1), RelId(2), vec![0], vec![0], 500_000, 1000);
        mgr.record_join_result(RelId(1), RelId(3), vec![0], vec![0], 200_000, 500);
        mgr.record_join_result(RelId(2), RelId(3), vec![0], vec![0], 100_000, 250);

        assert!(mgr.get_join_selectivity(RelId(1), RelId(2)).is_some());
        assert!(mgr.get_join_selectivity(RelId(1), RelId(3)).is_some());
        assert!(mgr.get_join_selectivity(RelId(2), RelId(3)).is_some());

        // Unregister relation 1 - should remove join selectivities with 1
        mgr.unregister_relation(RelId(1));

        assert!(mgr.get_join_selectivity(RelId(1), RelId(2)).is_none());
        assert!(mgr.get_join_selectivity(RelId(1), RelId(3)).is_none());
        // Join between 2 and 3 should still exist
        assert!(mgr.get_join_selectivity(RelId(2), RelId(3)).is_some());
    }

    #[test]
    fn test_stats_manager_update_cardinality() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.update_cardinality(RelId(1), 5000);
        let stats = mgr.get_relation_stats(RelId(1)).unwrap();
        assert_eq!(stats.cardinality, 5000);
    }

    #[test]
    fn test_stats_manager_update_cardinality_unregistered() {
        let mut mgr = StatsManager::new();
        // Should be a no-op for unregistered relation
        mgr.update_cardinality(RelId(1), 5000);
        assert!(mgr.get_relation_stats(RelId(1)).is_none());
    }

    #[test]
    fn test_stats_manager_update_byte_size() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.update_byte_size(RelId(1), 1024 * 1024);
        let stats = mgr.get_relation_stats(RelId(1)).unwrap();
        assert_eq!(stats.byte_size, 1024 * 1024);
    }

    #[test]
    fn test_stats_manager_record_access() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.record_access(RelId(1));
        let stats = mgr.get_relation_stats(RelId(1)).unwrap();
        assert!(stats.heat > 0.0);
        assert!(stats.last_access > 0);
    }

    #[test]
    fn test_stats_manager_record_access_multiple() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));

        for _ in 0..10 {
            mgr.record_access(RelId(1));
        }

        let stats = mgr.get_relation_stats(RelId(1)).unwrap();
        // After 10 accesses, heat should be quite high
        assert!(stats.heat > 0.5);
    }

    #[test]
    fn test_stats_manager_add_column_stats() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));

        let mut col_stats = ColumnStats::new(0, ScalarType::I64);
        col_stats.update_distinct(100);
        col_stats.update_range(0, 1000);
        mgr.add_column_stats(RelId(1), col_stats);

        let stats = mgr.get_relation_stats(RelId(1)).unwrap();
        assert_eq!(stats.column_stats.len(), 1);
        let col = stats.get_column(0).unwrap();
        assert_eq!(col.distinct_estimate, 100);
    }

    #[test]
    fn test_stats_manager_estimate_join() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.update_cardinality(RelId(1), 1000);
        mgr.update_cardinality(RelId(2), 500);

        let estimate = mgr.estimate_join_cardinality(RelId(1), RelId(2), &[0], &[0]);
        // Default selectivity assumes 10%: 1000 * 500 * 0.1 = 50000
        assert!(estimate > 0);
        assert!(estimate <= 1000 * 500);
    }

    #[test]
    fn test_stats_manager_estimate_join_with_column_stats() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.update_cardinality(RelId(1), 1000);
        mgr.update_cardinality(RelId(2), 500);

        // Add column stats with distinct values
        let mut col_stats1 = ColumnStats::new(0, ScalarType::I64);
        col_stats1.update_distinct(100);
        mgr.add_column_stats(RelId(1), col_stats1);

        let mut col_stats2 = ColumnStats::new(0, ScalarType::I64);
        col_stats2.update_distinct(50);
        mgr.add_column_stats(RelId(2), col_stats2);

        let estimate = mgr.estimate_join_cardinality(RelId(1), RelId(2), &[0], &[0]);
        // Selectivity = 1/max(100, 50) = 0.01
        // Expected: 1000 * 500 * 0.01 = 5000
        assert_eq!(estimate, 5000);
    }

    #[test]
    fn test_stats_manager_estimate_join_unregistered() {
        let mgr = StatsManager::new();
        // Should use default cardinality of 1000 for unregistered relations
        let estimate = mgr.estimate_join_cardinality(RelId(1), RelId(2), &[0], &[0]);
        // 1000 * 1000 * 0.1 = 100000
        assert_eq!(estimate, 100_000);
    }

    #[test]
    fn test_stats_manager_estimate_join_minimum_one() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.update_cardinality(RelId(1), 1);
        mgr.update_cardinality(RelId(2), 1);

        // Add column stats with high distinct count to make selectivity very low
        let mut col_stats1 = ColumnStats::new(0, ScalarType::I64);
        col_stats1.update_distinct(1_000_000);
        mgr.add_column_stats(RelId(1), col_stats1);

        let mut col_stats2 = ColumnStats::new(0, ScalarType::I64);
        col_stats2.update_distinct(1_000_000);
        mgr.add_column_stats(RelId(2), col_stats2);

        let estimate = mgr.estimate_join_cardinality(RelId(1), RelId(2), &[0], &[0]);
        // Should be at least 1
        assert!(estimate >= 1);
    }

    #[test]
    fn test_stats_manager_record_join_result() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.update_cardinality(RelId(1), 1000);
        mgr.update_cardinality(RelId(2), 500);

        // Record a join result
        mgr.record_join_result(RelId(1), RelId(2), vec![0], vec![0], 500_000, 1000);

        // Should have created a selectivity entry
        let js = mgr.get_join_selectivity(RelId(1), RelId(2)).unwrap();
        assert_eq!(js.left_keys, vec![0]);
        assert_eq!(js.right_keys, vec![0]);
        // Observed selectivity: 1000/500000 = 0.002
        // EMA: 1.0 * 0.7 + 0.002 * 0.3 ≈ 0.7006
        assert!(js.selectivity < 1.0);
    }

    #[test]
    fn test_stats_manager_record_join_result_canonical_order() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));

        // Record with reverse order - should use canonical key
        mgr.record_join_result(RelId(2), RelId(1), vec![0], vec![1], 1000, 100);

        // Should be able to look up with either order
        assert!(mgr.get_join_selectivity(RelId(1), RelId(2)).is_some());
        assert!(mgr.get_join_selectivity(RelId(2), RelId(1)).is_some());

        // Both lookups should return the same entry
        let js1 = mgr.get_join_selectivity(RelId(1), RelId(2)).unwrap();
        let js2 = mgr.get_join_selectivity(RelId(2), RelId(1)).unwrap();
        assert_eq!(js1.selectivity, js2.selectivity);
    }

    #[test]
    fn test_stats_manager_record_join_result_ema_update() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));

        // First observation
        mgr.record_join_result(RelId(1), RelId(2), vec![0], vec![0], 1000, 100);
        let sel1 = mgr
            .get_join_selectivity(RelId(1), RelId(2))
            .unwrap()
            .selectivity;

        // Second observation with different selectivity
        mgr.record_join_result(RelId(1), RelId(2), vec![0], vec![0], 1000, 500);
        let sel2 = mgr
            .get_join_selectivity(RelId(1), RelId(2))
            .unwrap()
            .selectivity;

        // Selectivity should have moved via EMA
        assert!(sel2 != sel1);
    }

    #[test]
    fn test_stats_manager_decay_all_heat() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));

        // Heat up relations
        for _ in 0..10 {
            mgr.record_access(RelId(1));
            mgr.record_access(RelId(2));
        }

        let heat1_before = mgr.get_relation_stats(RelId(1)).unwrap().heat;
        let heat2_before = mgr.get_relation_stats(RelId(2)).unwrap().heat;

        mgr.decay_all_heat(0.5);

        let heat1_after = mgr.get_relation_stats(RelId(1)).unwrap().heat;
        let heat2_after = mgr.get_relation_stats(RelId(2)).unwrap().heat;

        assert!((heat1_after - heat1_before * 0.5).abs() < 0.001);
        assert!((heat2_after - heat2_before * 0.5).abs() < 0.001);
    }

    #[test]
    fn test_stats_manager_hot_relations() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.register_relation(RelId(3));

        // Heat up only relation 1
        for _ in 0..20 {
            mgr.record_access(RelId(1));
        }

        let hot = mgr.hot_relations(0.5);
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0], RelId(1));
    }

    #[test]
    fn test_stats_manager_cold_relations() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.register_relation(RelId(3));

        // Heat up only relation 1
        for _ in 0..20 {
            mgr.record_access(RelId(1));
        }

        let cold = mgr.cold_relations(0.5);
        // Relations 2 and 3 should be cold
        assert_eq!(cold.len(), 2);
        assert!(cold.contains(&RelId(2)));
        assert!(cold.contains(&RelId(3)));
    }

    #[test]
    fn test_stats_manager_relation_ids() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(5));
        mgr.register_relation(RelId(10));
        mgr.register_relation(RelId(15));

        let ids: Vec<_> = mgr.relation_ids().collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&RelId(5)));
        assert!(ids.contains(&RelId(10)));
        assert!(ids.contains(&RelId(15)));
    }

    #[test]
    fn test_stats_manager_total_byte_size() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.update_byte_size(RelId(1), 1000);
        mgr.update_byte_size(RelId(2), 2000);

        assert_eq!(mgr.total_byte_size(), 3000);
    }

    #[test]
    fn test_stats_manager_total_cardinality() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.update_cardinality(RelId(1), 1000);
        mgr.update_cardinality(RelId(2), 2000);

        assert_eq!(mgr.total_cardinality(), 3000);
    }

    #[test]
    fn test_stats_manager_clear() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.record_join_result(RelId(1), RelId(2), vec![0], vec![0], 1000, 100);

        mgr.clear();

        assert_eq!(mgr.relation_count(), 0);
        assert!(mgr.get_relation_stats(RelId(1)).is_none());
        assert!(mgr.get_join_selectivity(RelId(1), RelId(2)).is_none());
    }

    #[test]
    fn test_stats_manager_get_relation_stats_mut() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));

        if let Some(stats) = mgr.get_relation_stats_mut(RelId(1)) {
            stats.update_cardinality(999);
            stats.has_index = true;
        }

        let stats = mgr.get_relation_stats(RelId(1)).unwrap();
        assert_eq!(stats.cardinality, 999);
        assert!(stats.has_index);
    }

    #[test]
    fn test_stats_manager_join_estimate_uses_cached_selectivity() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.update_cardinality(RelId(1), 1000);
        mgr.update_cardinality(RelId(2), 500);

        // Record a join with known selectivity
        // Observed: 2500 / 500000 = 0.005
        mgr.record_join_result(RelId(1), RelId(2), vec![0], vec![0], 500_000, 2500);

        // Subsequent estimates should use the cached selectivity
        let estimate = mgr.estimate_join_cardinality(RelId(1), RelId(2), &[0], &[0]);

        // The cached selectivity is an EMA, initial 1.0 * 0.7 + 0.005 * 0.3 = 0.7015
        // Estimate = 1000 * 500 * 0.7015 = 350750
        let js = mgr.get_join_selectivity(RelId(1), RelId(2)).unwrap();
        let expected = ((1000_f64 * 500_f64 * js.selectivity) as u64).max(1);
        assert_eq!(estimate, expected);
    }

    #[test]
    fn test_stats_manager_set_join_selectivity_canonicalizes_keys() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));

        // Set in reverse order; manager should store in canonical (1,2).
        mgr.set_join_selectivity(RelId(2), RelId(1), vec![3], vec![7], 0.05);

        let js = mgr.get_join_selectivity(RelId(1), RelId(2)).unwrap();
        assert_eq!(js.left_rel, RelId(1));
        assert_eq!(js.right_rel, RelId(2));
        assert_eq!(js.left_keys, vec![7]);
        assert_eq!(js.right_keys, vec![3]);
        assert!((js.selectivity - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_stats_manager_snapshot_and_merge() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.update_cardinality(RelId(1), 123);
        mgr.record_access(RelId(1));
        mgr.set_join_selectivity(RelId(1), RelId(2), vec![0], vec![0], 0.2);

        let snap = mgr.snapshot();

        let mut mgr2 = StatsManager::new();
        mgr2.merge_snapshot(&snap);

        let r1 = mgr2.get_relation_stats(RelId(1)).unwrap();
        assert_eq!(r1.cardinality, 123);

        let js = mgr2.get_join_selectivity(RelId(1), RelId(2)).unwrap();
        assert_eq!(js.left_keys, vec![0]);
        assert_eq!(js.right_keys, vec![0]);
        assert!((js.selectivity - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_canonical_join_key() {
        assert_eq!(
            StatsManager::canonical_join_key(RelId(1), RelId(2)),
            (RelId(1), RelId(2))
        );
        assert_eq!(
            StatsManager::canonical_join_key(RelId(2), RelId(1)),
            (RelId(1), RelId(2))
        );
        assert_eq!(
            StatsManager::canonical_join_key(RelId(5), RelId(5)),
            (RelId(5), RelId(5))
        );
    }
}
