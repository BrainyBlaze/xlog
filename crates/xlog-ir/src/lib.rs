//! Intermediate representations for XLOG
#![warn(missing_docs)]

pub mod eir;
pub mod metadata;
pub mod plan;
pub mod rir;

pub use eir::{
    EirAtom, EirBodyLiteral, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EirProgram,
    EirRule,
};
pub use metadata::{LayoutHint, RirMeta, SkewSignature};
pub use plan::{CompiledRule, ExecutionPlan, PlanBuilder, Scc, Stratum};
pub use rir::{CompareOp, ConstValue, Expr, JoinType, ProjectExpr, RirNode};
