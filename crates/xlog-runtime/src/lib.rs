//! Execution engine for XLOG
//!
//! This crate provides the runtime execution engine for XLOG queries,
//! managing GPU relation storage and query execution.

pub mod executor;
pub mod ilp_registry;
pub mod profiler;
pub mod relation;
mod statistics;

pub use executor::Executor;
pub use ilp_registry::{IlpRegistry, IlpTagEntry, IlpTaggedResult, read_device_row_count};
pub use profiler::{ExecutionStats, MeasureGuard, OpStats, Profiler, StratumStats};
pub use relation::RelationStore;
pub use statistics::{JoinStats, JoinStrategy, QueryStatistics};
