//! Core types and traits for XLOG
#![warn(missing_docs)]

pub mod config;
pub mod config_value;
pub mod error;
pub mod symbol;
pub mod traits;
pub mod types;

pub use config::{CostModelKind, MemoryBudget, RuntimeConfig};
pub use config_value::{parse_bool_value, read_bool_env, resolve_bool};
pub use error::{Result, XlogError};
pub use traits::{GpuBuffer, KernelProvider, RelationStore};
pub use types::{AggOp, RelId, ScalarType, Schema};
