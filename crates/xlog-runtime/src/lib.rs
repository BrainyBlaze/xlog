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
    EpistemicGpuBatchExecutionResult, EpistemicGpuBatchExecutionTrace,
    EpistemicGpuCandidateGenerationTrace, EpistemicGpuCandidateValidationTrace,
    EpistemicGpuExecutionResult, EpistemicGpuFinalResultMaterializationTrace,
    EpistemicGpuFinalResultTransferTrace, EpistemicGpuFinalTupleMaterializationTrace,
    EpistemicGpuKernelTimingTrace, EpistemicGpuMaterializationTrace,
    EpistemicGpuModelMembershipSource, EpistemicGpuModelMembershipTrace,
    EpistemicGpuPreparedExecution, EpistemicGpuPropagationTrace, EpistemicGpuRejectionReason,
    EpistemicGpuRuntimeCounters, EpistemicGpuRuntimePreflight, EpistemicGpuRuntimeTrace,
    EpistemicGpuRuntimeWcojCertification, EpistemicGpuTransferBudgetTrace, EpistemicGpuWorkspace,
    EpistemicGpuWorkspaceCapacities, EpistemicGpuWorkspaceLayout, EpistemicGpuWorkspaceResetTrace,
    EpistemicGpuWorldViewValidationTrace, Executor,
};
pub use ilp_registry::{read_device_row_count, IlpRegistry, IlpTagEntry, IlpTaggedResult};
pub use profiler::{ExecutionStats, MeasureGuard, OpStats, Profiler, StratumStats};
pub use relation::RelationStore;
pub use statistics::{JoinStats, JoinStrategy, QueryStatistics};
