//! Datalog frontend for XLOG

pub mod parser;
pub mod ast;
pub mod stratify;
pub mod lower;
pub mod compile;

pub use parser::{parse_program, parse_statement};
pub use ast::{Program, Rule, Atom, Term, BodyLiteral};
pub use stratify::{stratify, Stratum, DependencyGraph, find_sccs_for_lowering};
pub use lower::Lowerer;
