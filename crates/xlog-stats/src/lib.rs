//! GPU-resident statistics for query optimization and solver heuristics.
#![warn(missing_docs)]
//!
//! This crate provides statistics tracking for GPU-resident relations and columns
//! that are used by the query optimizer and solver heuristics to make informed
//! decisions about query execution strategies.
//!
//! # Core Types
//!
//! - [`StatsManager`]: Central coordinator for all relation statistics and join
//!   selectivity tracking.
//! - [`RelationStats`]: Tracks cardinality, memory usage, access patterns, and
//!   column-level statistics for GPU-resident relations.
//! - [`ColumnStats`]: Per-column statistics including distinct counts, null counts,
//!   and value ranges for selectivity estimation.
//! - [`JoinSelectivity`]: Models join behavior between relations for cardinality
//!   estimation.
//!
//! # Usage
//!
//! ```ignore
//! use xlog_stats::{StatsManager, RelationStats, ColumnStats};
//! use xlog_core::{RelId, ScalarType};
//!
//! // Create a stats manager and register relations
//! let mut mgr = StatsManager::new();
//! mgr.register_relation(RelId(1));
//! mgr.register_relation(RelId(2));
//!
//! // Update statistics
//! mgr.update_cardinality(RelId(1), 10_000);
//! mgr.update_cardinality(RelId(2), 5_000);
//!
//! // Add column statistics
//! let mut col_stats = ColumnStats::new(0, ScalarType::I64);
//! col_stats.update_distinct(500);
//! col_stats.update_range(0, 1000);
//! mgr.add_column_stats(RelId(1), col_stats);
//!
//! // Estimate join cardinality
//! let estimate = mgr.estimate_join_cardinality(RelId(1), RelId(2), &[0], &[0]);
//!
//! // Record actual join result to improve future estimates
//! mgr.record_join_result(RelId(1), RelId(2), vec![0], vec![0], 50_000_000, 25_000);
//!
//! // Track hot relations for LRU eviction
//! mgr.record_access(RelId(1));
//! let hot_rels = mgr.hot_relations(0.5);
//! ```

mod manager;
mod stats;

pub use manager::StatsManager;
pub use manager::StatsSnapshot;
pub use stats::{ColumnStats, JoinSelectivity, RelationStats};
