//! Core statistics types for GPU-resident relation metadata.
//!
//! This module provides statistics tracking for relations and columns that are
//! used by the query optimizer and solver heuristics to make informed decisions
//! about query execution strategies.

use xlog_core::{RelId, ScalarType};

/// GPU-resident relation statistics.
///
/// Tracks cardinality, memory usage, access patterns, and column-level statistics
/// for relations stored on the GPU. These statistics drive optimizer cost models
/// and solver heuristics for efficient query execution.
#[derive(Debug, Clone)]
pub struct RelationStats {
    /// Unique identifier for the relation
    pub rel_id: RelId,
    /// Estimated number of rows in the relation
    pub cardinality: u64,
    /// Estimated total size in bytes on GPU
    pub byte_size: u64,
    /// Per-column statistics
    pub column_stats: Vec<ColumnStats>,
    /// Per-column prefix fan-out statistics for trie-style WCOJ planning.
    pub prefix_degrees: Vec<PrefixDegreeStats>,
    /// Per-column key heat/skew summaries for skew-aware WCOJ planning.
    pub key_heats: Vec<KeyHeatStats>,
    /// Access heat for LRU-style eviction (exponential moving average)
    pub heat: f32,
    /// Unix timestamp of last access
    pub last_access: u64,
    /// Whether an index exists for this relation
    pub has_index: bool,
}

impl RelationStats {
    /// Creates new statistics for a relation with default (empty) values.
    ///
    /// # Arguments
    /// * `rel_id` - The unique identifier for the relation
    ///
    /// # Returns
    /// A new `RelationStats` instance with zero cardinality, no columns, and cold heat.
    pub fn new(rel_id: RelId) -> Self {
        Self {
            rel_id,
            cardinality: 0,
            byte_size: 0,
            column_stats: Vec::new(),
            prefix_degrees: Vec::new(),
            key_heats: Vec::new(),
            heat: 0.0,
            last_access: 0,
            has_index: false,
        }
    }

    /// Updates the cardinality (row count) of the relation.
    ///
    /// This should be called after bulk loads, inserts, or when statistics
    /// are refreshed from the actual GPU-resident data.
    ///
    /// # Arguments
    /// * `rows` - The new cardinality estimate
    pub fn update_cardinality(&mut self, rows: u64) {
        self.cardinality = rows;
    }

    /// Updates the byte size estimate for the relation.
    ///
    /// # Arguments
    /// * `bytes` - The estimated total size in bytes
    pub fn update_byte_size(&mut self, bytes: u64) {
        self.byte_size = bytes;
    }

    /// Records an access to this relation, updating heat and timestamp.
    ///
    /// Uses an exponential moving average for heat calculation:
    /// `heat = heat * 0.9 + 0.1`
    ///
    /// This causes frequently accessed relations to maintain high heat
    /// while infrequently accessed ones cool down over time.
    pub fn record_access(&mut self) {
        // Exponential moving average for heat
        self.heat = self.heat * 0.9 + 0.1;
        self.last_access = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Decays the heat by a multiplicative factor.
    ///
    /// This should be called periodically (e.g., during garbage collection
    /// or memory pressure events) to allow unused relations to cool down.
    ///
    /// # Arguments
    /// * `factor` - Multiplicative decay factor (typically 0.0 to 1.0)
    pub fn decay_heat(&mut self, factor: f32) {
        self.heat *= factor;
    }

    /// Adds column statistics for a new column.
    ///
    /// # Arguments
    /// * `col_stats` - The column statistics to add
    pub fn add_column(&mut self, col_stats: ColumnStats) {
        self.column_stats.push(col_stats);
    }

    /// Gets column statistics by index.
    ///
    /// # Arguments
    /// * `col_idx` - The column index
    ///
    /// # Returns
    /// A reference to the column statistics if found
    pub fn get_column(&self, col_idx: usize) -> Option<&ColumnStats> {
        self.column_stats.iter().find(|c| c.col_idx == col_idx)
    }

    /// Gets mutable column statistics by index.
    ///
    /// # Arguments
    /// * `col_idx` - The column index
    ///
    /// # Returns
    /// A mutable reference to the column statistics if found
    pub fn get_column_mut(&mut self, col_idx: usize) -> Option<&mut ColumnStats> {
        self.column_stats.iter_mut().find(|c| c.col_idx == col_idx)
    }

    /// Adds prefix-degree statistics for a join-key column.
    ///
    /// Existing entries for the same column are retained; consumers use the first
    /// matching entry so snapshots can preserve historical observations.
    pub fn add_prefix_degree(&mut self, prefix_degree: PrefixDegreeStats) {
        self.prefix_degrees.push(prefix_degree);
    }

    /// Gets prefix-degree statistics by column index.
    pub fn get_prefix_degree(&self, col_idx: usize) -> Option<&PrefixDegreeStats> {
        self.prefix_degrees.iter().find(|p| p.col_idx == col_idx)
    }

    /// Adds key-heat statistics for a join-key column.
    ///
    /// This is distinct from relation-level [`RelationStats::heat`]: relation heat
    /// tracks access frequency, while key heat tracks per-key skew for a column.
    pub fn add_key_heat(&mut self, key_heat: KeyHeatStats) {
        self.key_heats.push(key_heat);
    }

    /// Gets key-heat statistics by column index.
    pub fn get_key_heat(&self, col_idx: usize) -> Option<&KeyHeatStats> {
        self.key_heats.iter().find(|h| h.col_idx == col_idx)
    }

    /// Estimates the selectivity for a given predicate cardinality.
    ///
    /// # Arguments
    /// * `estimated_matches` - The estimated number of matching rows
    ///
    /// # Returns
    /// The selectivity as a ratio (0.0 to 1.0)
    pub fn estimate_selectivity(&self, estimated_matches: u64) -> f64 {
        if self.cardinality == 0 {
            return 1.0;
        }
        (estimated_matches as f64 / self.cardinality as f64).clamp(0.0, 1.0)
    }
}

/// Prefix fan-out statistics for one relation column.
///
/// WCOJ planners use this as the trie prefix-degree signal: lower average and
/// bounded maximum fan-out usually mean less inner-loop work for a variable
/// order that binds the column early.
#[derive(Debug, Clone)]
pub struct PrefixDegreeStats {
    /// Column index within the relation.
    pub col_idx: usize,
    /// Average number of rows below one distinct prefix key.
    pub avg_degree: f64,
    /// High-water fan-out used as a skew guard.
    pub max_degree: f64,
}

impl PrefixDegreeStats {
    /// Creates prefix-degree statistics for a column.
    pub fn new(col_idx: usize, avg_degree: f64, max_degree: f64) -> Self {
        Self {
            col_idx,
            avg_degree,
            max_degree,
        }
    }
}

/// Per-key heat/skew statistics for one relation column.
///
/// The value is a compact summary of key-frequency imbalance. A value near zero
/// is cold/unskewed; larger values indicate pivot-heavy keys that should be
/// demoted by a skew-aware WCOJ planner.
#[derive(Debug, Clone)]
pub struct KeyHeatStats {
    /// Column index within the relation.
    pub col_idx: usize,
    /// Heat value for the heavy-key tail.
    pub heat: f64,
    /// Multiplicative skew factor for the heaviest observed keys.
    pub skew_factor: f64,
}

impl KeyHeatStats {
    /// Creates key-heat statistics for a column.
    pub fn new(col_idx: usize, heat: f64, skew_factor: f64) -> Self {
        Self {
            col_idx,
            heat,
            skew_factor,
        }
    }
}

/// Per-column statistics for optimizer cost estimation.
///
/// Tracks null counts, distinct value estimates, and value ranges for columns.
/// These statistics enable the optimizer to estimate filter selectivity and
/// join cardinalities.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Column index within the relation
    pub col_idx: usize,
    /// Data type of the column
    pub dtype: ScalarType,
    /// Count of null values (for nullable columns)
    pub null_count: u64,
    /// HyperLogLog-style distinct value estimate
    pub distinct_estimate: u64,
    /// Minimum value (for orderable types, stored as i64)
    pub min_value: Option<i64>,
    /// Maximum value (for orderable types, stored as i64)
    pub max_value: Option<i64>,
    /// Average value length for variable-length types (e.g., symbols)
    pub avg_width: Option<f32>,
}

impl ColumnStats {
    /// Creates new column statistics with default values.
    ///
    /// # Arguments
    /// * `col_idx` - The column index within the relation
    /// * `dtype` - The scalar type of the column
    ///
    /// # Returns
    /// A new `ColumnStats` instance with zero counts and no range information.
    pub fn new(col_idx: usize, dtype: ScalarType) -> Self {
        Self {
            col_idx,
            dtype,
            null_count: 0,
            distinct_estimate: 0,
            min_value: None,
            max_value: None,
            avg_width: None,
        }
    }

    /// Updates the distinct value estimate.
    ///
    /// This should be updated from HyperLogLog or similar cardinality estimation
    /// algorithms running on the GPU.
    ///
    /// # Arguments
    /// * `estimate` - The new distinct value estimate
    pub fn update_distinct(&mut self, estimate: u64) {
        self.distinct_estimate = estimate;
    }

    /// Updates the value range for this column.
    ///
    /// # Arguments
    /// * `min` - The minimum value (encoded as i64)
    /// * `max` - The maximum value (encoded as i64)
    pub fn update_range(&mut self, min: i64, max: i64) {
        self.min_value = Some(min);
        self.max_value = Some(max);
    }

    /// Updates the null count for this column.
    ///
    /// # Arguments
    /// * `count` - The number of null values
    pub fn update_null_count(&mut self, count: u64) {
        self.null_count = count;
    }

    /// Updates the average width for variable-length columns.
    ///
    /// # Arguments
    /// * `width` - The average value width in bytes
    pub fn update_avg_width(&mut self, width: f32) {
        self.avg_width = Some(width);
    }

    /// Estimates selectivity for an equality predicate.
    ///
    /// Uses the distinct value count to estimate selectivity. If no distinct
    /// count is available, returns a default estimate.
    ///
    /// # Arguments
    /// * `total_rows` - The total number of rows in the relation
    ///
    /// # Returns
    /// The estimated selectivity (0.0 to 1.0)
    pub fn equality_selectivity(&self, total_rows: u64) -> f64 {
        if self.distinct_estimate == 0 || total_rows == 0 {
            // Default selectivity when no statistics available
            return 0.1;
        }
        1.0 / self.distinct_estimate as f64
    }

    /// Estimates selectivity for a range predicate.
    ///
    /// Uses min/max values to estimate what fraction of the range is covered.
    /// Returns a default estimate if range statistics are unavailable.
    ///
    /// # Arguments
    /// * `low` - The lower bound of the range (inclusive)
    /// * `high` - The upper bound of the range (inclusive)
    ///
    /// # Returns
    /// The estimated selectivity (0.0 to 1.0)
    pub fn range_selectivity(&self, low: i64, high: i64) -> f64 {
        match (self.min_value, self.max_value) {
            (Some(col_min), Some(col_max)) if col_max > col_min => {
                let col_range = (col_max - col_min) as f64;
                let effective_low = low.max(col_min);
                let effective_high = high.min(col_max);
                if effective_high < effective_low {
                    return 0.0;
                }
                let query_range = (effective_high - effective_low) as f64;
                (query_range / col_range).clamp(0.0, 1.0)
            }
            _ => {
                // Default range selectivity when no statistics available
                0.25
            }
        }
    }

    /// Returns the storage size per value for this column type.
    pub fn value_size_bytes(&self) -> usize {
        self.dtype.size_bytes()
    }
}

/// Join selectivity model for estimating join output cardinality.
///
/// Tracks information about joins between two relations, including the join
/// keys and estimated selectivity. This is crucial for the optimizer to
/// choose between nested-loop, hash, and sort-merge join strategies.
#[derive(Debug, Clone)]
pub struct JoinSelectivity {
    /// Left relation in the join
    pub left_rel: RelId,
    /// Right relation in the join
    pub right_rel: RelId,
    /// Column indices used as join keys on the left relation
    pub left_keys: Vec<usize>,
    /// Column indices used as join keys on the right relation
    pub right_keys: Vec<usize>,
    /// Estimated selectivity factor (0.0 to 1.0)
    pub selectivity: f64,
    /// Whether this is a primary key to foreign key join
    pub is_pk_fk: bool,
    /// Cached join cardinality estimate (if computed)
    cached_output_estimate: Option<u64>,
}

impl JoinSelectivity {
    /// Creates a new join selectivity model between two relations.
    ///
    /// Initializes with default selectivity of 1.0 (cross product).
    ///
    /// # Arguments
    /// * `left_rel` - The left relation's ID
    /// * `right_rel` - The right relation's ID
    ///
    /// # Returns
    /// A new `JoinSelectivity` with default values.
    pub fn new(left_rel: RelId, right_rel: RelId) -> Self {
        Self {
            left_rel,
            right_rel,
            left_keys: Vec::new(),
            right_keys: Vec::new(),
            selectivity: 1.0,
            is_pk_fk: false,
            cached_output_estimate: None,
        }
    }

    /// Sets the join keys for both relations.
    ///
    /// # Arguments
    /// * `left_keys` - Column indices on the left relation
    /// * `right_keys` - Column indices on the right relation
    pub fn set_keys(&mut self, left_keys: Vec<usize>, right_keys: Vec<usize>) {
        debug_assert_eq!(
            left_keys.len(),
            right_keys.len(),
            "Join key counts must match"
        );
        self.left_keys = left_keys;
        self.right_keys = right_keys;
        self.cached_output_estimate = None;
    }

    /// Sets the selectivity factor.
    ///
    /// # Arguments
    /// * `selectivity` - The selectivity factor (0.0 to 1.0)
    pub fn set_selectivity(&mut self, selectivity: f64) {
        self.selectivity = selectivity.clamp(0.0, 1.0);
        self.cached_output_estimate = None;
    }

    /// Marks this as a primary key to foreign key join.
    ///
    /// PK-FK joins have special selectivity characteristics: the output
    /// cardinality equals the FK side's cardinality.
    pub fn mark_pk_fk(&mut self) {
        self.is_pk_fk = true;
    }

    /// Estimates the output row count for this join.
    ///
    /// For PK-FK joins, returns the cardinality of the FK side.
    /// For other joins, returns: left_rows * right_rows * selectivity
    ///
    /// # Arguments
    /// * `left_rows` - Cardinality of the left relation
    /// * `right_rows` - Cardinality of the right relation
    ///
    /// # Returns
    /// The estimated output cardinality (minimum of 1)
    pub fn estimate_output_rows(&self, left_rows: u64, right_rows: u64) -> u64 {
        if self.is_pk_fk {
            // FK side determines cardinality in PK-FK joins
            // Conventionally, right side is FK
            return right_rows;
        }
        ((left_rows as f64 * right_rows as f64 * self.selectivity) as u64).max(1)
    }

    /// Estimates selectivity from column statistics.
    ///
    /// Uses the "independence assumption" and distinct value counts:
    /// selectivity = 1 / max(distinct_left, distinct_right)
    ///
    /// # Arguments
    /// * `left_distinct` - Distinct value count for left join key
    /// * `right_distinct` - Distinct value count for right join key
    ///
    /// # Returns
    /// The estimated selectivity
    pub fn estimate_selectivity_from_stats(left_distinct: u64, right_distinct: u64) -> f64 {
        if left_distinct == 0 || right_distinct == 0 {
            return 1.0;
        }
        1.0 / left_distinct.max(right_distinct) as f64
    }

    /// Updates selectivity based on observed join statistics.
    ///
    /// This can be called after query execution to improve future estimates.
    ///
    /// # Arguments
    /// * `left_rows` - Actual left cardinality
    /// * `right_rows` - Actual right cardinality
    /// * `output_rows` - Actual output cardinality
    pub fn update_from_observation(&mut self, left_rows: u64, right_rows: u64, output_rows: u64) {
        let product = left_rows as f64 * right_rows as f64;
        if product > 0.0 {
            self.selectivity = (output_rows as f64 / product).clamp(0.0, 1.0);
            self.cached_output_estimate = Some(output_rows);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relation_stats_new() {
        let stats = RelationStats::new(RelId(1));
        assert_eq!(stats.rel_id, RelId(1));
        assert_eq!(stats.cardinality, 0);
        assert_eq!(stats.heat, 0.0);
        assert_eq!(stats.byte_size, 0);
        assert!(stats.column_stats.is_empty());
        assert!(!stats.has_index);
    }

    #[test]
    fn test_relation_stats_update_cardinality() {
        let mut stats = RelationStats::new(RelId(1));
        stats.update_cardinality(1000);
        assert_eq!(stats.cardinality, 1000);
    }

    #[test]
    fn test_relation_stats_update_byte_size() {
        let mut stats = RelationStats::new(RelId(1));
        stats.update_byte_size(4096);
        assert_eq!(stats.byte_size, 4096);
    }

    #[test]
    fn test_relation_stats_update_heat() {
        let mut stats = RelationStats::new(RelId(1));
        assert_eq!(stats.heat, 0.0);

        stats.record_access();
        assert!(stats.heat > 0.0);
        let heat_after_first = stats.heat;
        assert!((heat_after_first - 0.1).abs() < 0.001);

        stats.record_access();
        assert!(stats.heat > heat_after_first);
        // After second access: 0.1 * 0.9 + 0.1 = 0.19
        assert!((stats.heat - 0.19).abs() < 0.001);

        // Verify last_access was set
        assert!(stats.last_access > 0);
    }

    #[test]
    fn test_relation_stats_decay_heat() {
        let mut stats = RelationStats::new(RelId(1));
        stats.record_access();
        stats.record_access();
        let initial_heat = stats.heat;

        stats.decay_heat(0.5);
        assert!((stats.heat - initial_heat * 0.5).abs() < 0.001);
    }

    #[test]
    fn test_relation_stats_column_management() {
        let mut stats = RelationStats::new(RelId(1));
        let col0 = ColumnStats::new(0, ScalarType::U32);
        let col1 = ColumnStats::new(1, ScalarType::I64);

        stats.add_column(col0);
        stats.add_column(col1);

        assert_eq!(stats.column_stats.len(), 2);
        assert!(stats.get_column(0).is_some());
        assert!(stats.get_column(1).is_some());
        assert!(stats.get_column(2).is_none());

        // Test mutable access
        if let Some(col) = stats.get_column_mut(0) {
            col.update_distinct(100);
        }
        assert_eq!(stats.get_column(0).unwrap().distinct_estimate, 100);
    }

    #[test]
    fn test_relation_stats_estimate_selectivity() {
        let mut stats = RelationStats::new(RelId(1));
        stats.update_cardinality(1000);

        // 100 matches out of 1000 = 0.1 selectivity
        let sel = stats.estimate_selectivity(100);
        assert!((sel - 0.1).abs() < 0.001);

        // Edge case: zero cardinality
        let empty_stats = RelationStats::new(RelId(2));
        assert_eq!(empty_stats.estimate_selectivity(50), 1.0);
    }

    #[test]
    fn test_column_stats_new() {
        let col = ColumnStats::new(0, ScalarType::U32);
        assert_eq!(col.col_idx, 0);
        assert_eq!(col.dtype, ScalarType::U32);
        assert_eq!(col.distinct_estimate, 0);
        assert_eq!(col.null_count, 0);
        assert!(col.min_value.is_none());
        assert!(col.max_value.is_none());
        assert!(col.avg_width.is_none());
    }

    #[test]
    fn test_column_stats_update_distinct() {
        let mut col = ColumnStats::new(0, ScalarType::U32);
        col.update_distinct(500);
        assert_eq!(col.distinct_estimate, 500);
    }

    #[test]
    fn test_column_stats_update_range() {
        let mut col = ColumnStats::new(0, ScalarType::I32);
        col.update_range(-100, 100);
        assert_eq!(col.min_value, Some(-100));
        assert_eq!(col.max_value, Some(100));
    }

    #[test]
    fn test_column_stats_update_null_count() {
        let mut col = ColumnStats::new(0, ScalarType::U32);
        col.update_null_count(42);
        assert_eq!(col.null_count, 42);
    }

    #[test]
    fn test_column_stats_update_avg_width() {
        let mut col = ColumnStats::new(0, ScalarType::Symbol);
        col.update_avg_width(12.5);
        assert_eq!(col.avg_width, Some(12.5));
    }

    #[test]
    fn test_column_stats_equality_selectivity() {
        let mut col = ColumnStats::new(0, ScalarType::U32);
        col.update_distinct(100);

        let sel = col.equality_selectivity(1000);
        assert!((sel - 0.01).abs() < 0.0001); // 1/100 = 0.01

        // Edge case: no distinct estimate
        let empty_col = ColumnStats::new(1, ScalarType::U32);
        assert_eq!(empty_col.equality_selectivity(1000), 0.1); // default
    }

    #[test]
    fn test_column_stats_range_selectivity() {
        let mut col = ColumnStats::new(0, ScalarType::I64);
        col.update_range(0, 100);

        // Query for [25, 75] on column with range [0, 100]
        let sel = col.range_selectivity(25, 75);
        assert!((sel - 0.5).abs() < 0.001); // (75-25)/100 = 0.5

        // Query outside range
        let sel_outside = col.range_selectivity(200, 300);
        assert_eq!(sel_outside, 0.0);

        // Query partially overlapping
        let sel_partial = col.range_selectivity(50, 150);
        assert!((sel_partial - 0.5).abs() < 0.001); // (100-50)/100 = 0.5

        // No range stats available
        let empty_col = ColumnStats::new(1, ScalarType::I64);
        assert_eq!(empty_col.range_selectivity(0, 100), 0.25); // default
    }

    #[test]
    fn test_column_stats_value_size() {
        assert_eq!(ColumnStats::new(0, ScalarType::U32).value_size_bytes(), 4);
        assert_eq!(ColumnStats::new(0, ScalarType::U64).value_size_bytes(), 8);
        assert_eq!(ColumnStats::new(0, ScalarType::Bool).value_size_bytes(), 1);
    }

    #[test]
    fn test_join_selectivity_new() {
        let js = JoinSelectivity::new(RelId(1), RelId(2));
        assert_eq!(js.left_rel, RelId(1));
        assert_eq!(js.right_rel, RelId(2));
        assert!(js.left_keys.is_empty());
        assert!(js.right_keys.is_empty());
        assert_eq!(js.selectivity, 1.0);
        assert!(!js.is_pk_fk);
    }

    #[test]
    fn test_join_selectivity_set_keys() {
        let mut js = JoinSelectivity::new(RelId(1), RelId(2));
        js.set_keys(vec![0, 1], vec![0, 1]);
        assert_eq!(js.left_keys, vec![0, 1]);
        assert_eq!(js.right_keys, vec![0, 1]);
    }

    #[test]
    fn test_join_selectivity_set_selectivity() {
        let mut js = JoinSelectivity::new(RelId(1), RelId(2));
        js.set_selectivity(0.01);
        assert!((js.selectivity - 0.01).abs() < 0.0001);

        // Test clamping
        js.set_selectivity(2.0);
        assert_eq!(js.selectivity, 1.0);

        js.set_selectivity(-1.0);
        assert_eq!(js.selectivity, 0.0);
    }

    #[test]
    fn test_join_selectivity_estimate_output_rows() {
        let mut js = JoinSelectivity::new(RelId(1), RelId(2));
        js.set_selectivity(0.01);

        // 1000 * 500 * 0.01 = 5000
        let output = js.estimate_output_rows(1000, 500);
        assert_eq!(output, 5000);

        // Test minimum of 1
        js.set_selectivity(0.0);
        let output_min = js.estimate_output_rows(10, 10);
        assert_eq!(output_min, 1);
    }

    #[test]
    fn test_join_selectivity_pk_fk() {
        let mut js = JoinSelectivity::new(RelId(1), RelId(2));
        js.mark_pk_fk();
        assert!(js.is_pk_fk);

        // PK-FK join: output = FK side cardinality
        let output = js.estimate_output_rows(100, 500);
        assert_eq!(output, 500); // FK side (right) cardinality
    }

    #[test]
    fn test_join_selectivity_estimate_from_stats() {
        // Selectivity = 1 / max(100, 200) = 0.005
        let sel = JoinSelectivity::estimate_selectivity_from_stats(100, 200);
        assert!((sel - 0.005).abs() < 0.0001);

        // Edge case: zero distinct
        let sel_zero = JoinSelectivity::estimate_selectivity_from_stats(0, 100);
        assert_eq!(sel_zero, 1.0);
    }

    #[test]
    fn test_join_selectivity_update_from_observation() {
        let mut js = JoinSelectivity::new(RelId(1), RelId(2));
        js.update_from_observation(1000, 500, 2500);

        // Observed selectivity = 2500 / (1000 * 500) = 0.005
        assert!((js.selectivity - 0.005).abs() < 0.0001);
    }

    #[test]
    fn test_all_scalar_types_column_stats() {
        // Ensure we can create column stats for all scalar types
        let types = [
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F32,
            ScalarType::F64,
            ScalarType::Bool,
            ScalarType::Symbol,
        ];

        for (idx, dtype) in types.iter().enumerate() {
            let col = ColumnStats::new(idx, *dtype);
            assert_eq!(col.col_idx, idx);
            assert_eq!(col.dtype, *dtype);
            assert!(col.value_size_bytes() > 0);
        }
    }
}
