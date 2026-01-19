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

pub mod ast;
pub mod compile;
pub mod expand;
pub mod function;
pub mod lower;
pub mod module;
pub mod optimizer;
pub mod parser;
pub mod resolver;
pub mod stratify;
pub mod typeinfer;

// Re-export main types
pub use ast::{
    AnnotatedDisjunction, Atom, BodyLiteral, Constraint, Directives, Evidence, ProbCache,
    ProbEngine, ProbFact, ProbQuery, Program, Query, Rule, Term,
};
pub use compile::{compile, Compiler};
pub use lower::Lowerer;
pub use optimizer::{Optimizer, OptimizerConfig, PlanCost};
pub use parser::{parse_program, parse_statement};
pub use expand::expand_program_functions;
pub use stratify::{find_sccs_for_lowering, stratify, DependencyGraph, Stratum};
