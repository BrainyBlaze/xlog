//! Datalog frontend for XLOG
//!
//! This crate provides the parsing, analysis, and compilation pipeline
//! for XLOG Datalog programs.
//!
//! # Main Entry Point
//!
//! The primary way to use this crate is through the [`Compiler`] struct:
//!
//! ```ignore
//! use xlog_logic::Compiler;
//!
//! let mut compiler = Compiler::new();
//! let plan = compiler.compile(r#"
//!     edge(1, 2).
//!     edge(2, 3).
//!     reach(X, Y) :- edge(X, Y).
//!     reach(X, Z) :- reach(X, Y), edge(Y, Z).
//! "#)?;
//! ```
//!
//! # Modules
//!
//! - [`parser`] - Pest-based parser for XLOG syntax
//! - [`ast`] - Abstract Syntax Tree types
//! - [`stratify`] - Stratification analysis for negation/aggregation
//! - [`lower`] - Lowering from AST to Relational IR
//! - [`compile`] - Full compilation pipeline

pub mod parser;
pub mod ast;
pub mod stratify;
pub mod lower;
pub mod compile;
pub mod optimizer;

// Re-export main types
pub use parser::{parse_program, parse_statement};
pub use ast::{Atom, BodyLiteral, Constraint, Program, Query, Rule, Term};
pub use stratify::{stratify, Stratum, DependencyGraph, find_sccs_for_lowering};
pub use lower::Lowerer;
pub use compile::{Compiler, compile};
pub use optimizer::{Optimizer, OptimizerConfig, PlanCost};
