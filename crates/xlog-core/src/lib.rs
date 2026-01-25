//! Core types and traits for XLOG

pub mod config;
pub mod error;
pub mod symbol;
pub mod traits;
pub mod types;

pub use config::{MemoryBudget, RuntimeConfig};
pub use error::{Result, XlogError};
pub use traits::{GpuBuffer, KernelProvider, RelationStore};
pub use types::{AggOp, RelId, ScalarType, Schema};
