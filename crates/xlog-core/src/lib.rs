//! Core types and traits for XLOG

pub mod error;
pub mod config;
pub mod types;
pub mod traits;
pub mod symbol;

pub use error::{XlogError, Result};
pub use config::{MemoryBudget, RuntimeConfig};
pub use types::{hash_symbol_to_u32, ScalarType, Schema, RelId, AggOp};
pub use traits::{GpuBuffer, KernelProvider, RelationStore};
