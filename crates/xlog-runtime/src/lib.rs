//! Execution engine for XLOG
//!
//! This crate provides the runtime execution engine for XLOG queries,
//! managing GPU relation storage and query execution.

pub mod executor;
pub mod profiler;
pub mod relation;

pub use relation::RelationStore;
