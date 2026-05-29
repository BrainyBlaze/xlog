//! Intermediate representations for XLOG
#![warn(missing_docs)]

pub mod eir;
pub mod epistemic_plan;
pub mod metadata;
pub mod plan;
pub mod rir;

pub use eir::{
    EirAtom, EirBodyLiteral, EirConstraint, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp,
    EirProgram, EirRule, EirTerm,
};
pub use epistemic_plan::{
    EpistemicConstraintPlan, EpistemicCpuFallbackCounters, EpistemicExecutablePlan,
    EpistemicGpuBufferKind, EpistemicGpuHotPathPhase, EpistemicGpuPlan, EpistemicReductionPlan,
    EpistemicSolverAssumptionBinding, EpistemicSolverCapability, EpistemicSolverServiceContract,
    EpistemicSolverStatusKind, EpistemicTupleMembershipBinding, EpistemicWcojReductionStatus,
};
pub use metadata::{LayoutHint, RirMeta, SkewSignature};
pub use plan::{CompiledRule, ExecutionPlan, PlanBuilder, Scc, Stratum};
pub use rir::{CompareOp, ConstValue, Expr, JoinType, ProjectExpr, RirNode};
