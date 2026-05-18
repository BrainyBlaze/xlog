//! Execution engine for XLOG
#![warn(missing_docs)]
//!
//! This crate provides the runtime execution engine for XLOG queries,
//! managing GPU relation storage and query execution.

pub mod executor;
pub mod ilp_registry;
pub mod profiler;
pub mod relation;
mod statistics;

pub use executor::{
    EpistemicGpuCandidateGenerationTrace, EpistemicGpuExecutionResult,
    EpistemicGpuPreparedExecution, EpistemicGpuPropagationTrace, EpistemicGpuRuntimeCounters,
    EpistemicGpuRuntimePreflight, EpistemicGpuRuntimeTrace, EpistemicGpuRuntimeWcojCertification,
    EpistemicGpuWorkspace, EpistemicGpuWorkspaceCapacities, EpistemicGpuWorkspaceLayout,
    EpistemicGpuWorkspaceResetTrace, Executor,
};
pub use ilp_registry::{read_device_row_count, IlpRegistry, IlpTagEntry, IlpTaggedResult};
pub use profiler::{ExecutionStats, MeasureGuard, OpStats, Profiler, StratumStats};
pub use relation::RelationStore;
pub use statistics::{JoinStats, JoinStrategy, QueryStatistics};
