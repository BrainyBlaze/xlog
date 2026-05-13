//! Core types and traits for XLOG
#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod symbol;
pub mod traits;
pub mod types;

pub use config::{
    CostModelKind, MemoryBudget, RuntimeConfig, ENV_WCOJ_W34_THRESHOLD, W34_FUSION_THRESHOLD,
};
pub use error::{Result, XlogError};
pub use traits::{GpuBuffer, KernelProvider, RelationStore};
pub use types::{AggOp, RelId, ScalarType, Schema};
