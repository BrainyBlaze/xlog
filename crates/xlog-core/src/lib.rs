//! Core types and traits for XLOG
#![warn(missing_docs)]

pub mod config;
pub mod config_value;
pub mod error;
pub mod float_order;
pub mod symbol;
pub mod types;

pub use config::{CostModelKind, MemoryBudget, RuntimeConfig};
pub use config_value::{parse_bool_value, read_bool_env, resolve_bool};
pub use error::{Result, XlogError};
pub use float_order::{
    f32_total_order_key, f32_total_order_key_from_bits, f64_total_order_key,
    f64_total_order_key_from_bits,
};
pub use types::{AggOp, RelId, ScalarType, Schema};
