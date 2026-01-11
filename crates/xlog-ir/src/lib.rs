//! Intermediate representations for XLOG

pub mod rir;
pub mod metadata;
pub mod plan;

pub use rir::{RirNode, JoinType, Expr, CompareOp, ConstValue, ProjectExpr};
pub use metadata::{RirMeta, LayoutHint, SkewSignature};
pub use plan::{ExecutionPlan, Scc, Stratum, CompiledRule, PlanBuilder};
