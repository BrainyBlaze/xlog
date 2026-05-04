//! Datalog frontend for XLOG
#![warn(missing_docs)]
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
//! - [`mod@stratify`] - Stratification analysis for negation/aggregation
//! - [`lower`] - Lowering from AST to Relational IR
//! - [`mod@compile`] - Full compilation pipeline

pub mod ast;
pub mod compile;
pub mod compiler_config;
pub mod expand;
pub mod function;
pub mod hypergraph;
pub mod lower;
pub mod module;
pub mod optimizer;
pub mod parser;
pub mod promote;
pub mod resolver;
pub mod stratify;
#[allow(dead_code)] // reserved API: type inference not yet wired to main pipeline
pub mod typeinfer;
pub mod wcoj_var_ordering;

// Re-export main types
pub use ast::{
    AnnotatedDisjunction, Atom, BodyLiteral, Constraint, Directives, Evidence, ProbCache,
    ProbEngine, ProbFact, ProbQuery, Program, Query, Rule, Term,
};
pub use compile::{compile, Compiler};
pub use expand::expand_program_functions;
pub use lower::Lowerer;
pub use optimizer::{Optimizer, OptimizerConfig, PlanCost};
pub use parser::{parse_program, parse_statement};
pub use stratify::{find_sccs_for_lowering, stratify, DependencyGraph, Stratum};
