//! GPU-resident statistics for query optimization and solver heuristics.
//!
//! This crate provides statistics tracking for GPU-resident relations and columns
//! that are used by the query optimizer and solver heuristics to make informed
//! decisions about query execution strategies.
//!
//! # Core Types
//!
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
//! use xlog_stats::{RelationStats, ColumnStats};
//! use xlog_core::{RelId, ScalarType};
//!
//! // Create statistics for a new relation
//! let mut stats = RelationStats::new(RelId(1));
//! stats.update_cardinality(10_000);
//!
//! // Add column statistics
//! let mut col_stats = ColumnStats::new(0, ScalarType::I64);
//! col_stats.update_distinct(500);
//! col_stats.update_range(0, 1000);
//! stats.add_column(col_stats);
//!
//! // Record access for LRU tracking
//! stats.record_access();
//! ```

mod stats;

pub use stats::{ColumnStats, JoinSelectivity, RelationStats};
