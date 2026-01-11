//! GPU-resident statistics for query optimization and solver heuristics.

mod stats;
mod manager;

pub use stats::{RelationStats, ColumnStats, JoinSelectivity};
pub use manager::StatsManager;
