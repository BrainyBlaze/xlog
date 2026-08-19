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

pub mod arithmetic_eval;
pub mod ast;
pub mod compile;
pub mod compiler_config;
pub mod diagnostics;
pub mod eir;
pub mod epistemic;
pub mod expand;
mod fact_fast_path;
pub mod function;
pub mod ground_term_encoding;
pub mod hypergraph;
pub mod incremental_parse;
pub mod list_normalize;
pub mod lower;
pub mod magic_sets;
pub mod meta_normalize;
pub mod module;
pub mod module_diagnostics;
pub mod optimizer;
pub mod parser;
pub mod promote;
pub mod proof_trace;
pub mod resolver;
pub mod stratify;
pub mod wcoj_var_ordering;

// Re-export main types
pub use arithmetic_eval::{
    compare_arithmetic_values, evaluate_arithmetic_expression, ArithmeticValue,
};
pub use ast::{
    AnnotatedDisjunction, Atom, BodyLiteral, Constraint, Directives, EpistemicLiteral,
    EpistemicMode, EpistemicOp, Evidence, MagicSetsMode, ProbCache, ProbEngine, ProbFact,
    ProbMethod, ProbQuery, Program, Query, Rule, Term, Univ,
};
pub use compile::{compile, Compiler};
pub use diagnostics::{
    build_query_proof_traces, build_rule_provenance, format_atom, format_constraint_body,
    format_term, generated_function_variable_sources, query_proof_traces, rule_provenance,
    source_diagnostics, source_format_normalized_alternative, QueryProofTrace, RuleProvenance,
    RuleSourceKind,
};
pub use eir::build_eir;
pub use expand::{expand_program_functions, expand_program_functions_owned};
pub use incremental_parse::{
    IncrementalParseResult, ParseCacheStats, ParserSession, StatementSpan, StatementUnit,
};
pub use list_normalize::{normalize_list_builtins, normalize_list_builtins_owned};
pub use lower::Lowerer;
pub use magic_sets::{
    rewrite_magic_sets, rewrite_magic_sets_owned, MagicSetReport, MagicSetRewrite, MagicSetStatus,
};
pub use meta_normalize::{normalize_meta_builtins, normalize_meta_builtins_owned};
pub use module_diagnostics::{
    diagnose_module_boundaries, CandidateSourceKind, ModuleBoundaryInput, ModuleBoundaryReport,
    ModuleDeclaration, ModuleDeclarationKind, ModuleManifest, ModuleRole, ModuleViolation,
    ModuleViolationKind,
};
pub use optimizer::{Optimizer, OptimizerConfig, PlanCost};
pub use parser::{parse_program, parse_statement};
pub use proof_trace::{DifferentiableProofTraceMap, ProofTrace, ProofTraceSpec};
pub use stratify::{find_sccs_for_lowering, stratify, DependencyGraph, Stratum};
