//! Frontend construction for the Epistemic Intermediate Representation.

use xlog_core::Result;
use xlog_ir::{
    EirAtom, EirBodyLiteral, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EirProgram,
    EirRule, EirTerm,
};

use crate::ast::{
    AggOp, Atom, BodyLiteral, EpistemicMode, EpistemicOp, Program, Rule as AstRule, Term,
};

/// Build EIR from parsed frontend AST without lowering to RIR.
pub fn build_eir(program: &Program) -> Result<EirProgram> {
    Ok(EirProgram {
        mode: convert_mode(program.directives.epistemic_mode_or_default()),
        rules: program.rules.iter().map(convert_rule).collect(),
    })
}

fn convert_rule(rule: &AstRule) -> EirRule {
    EirRule {
        head: convert_atom(&rule.head),
        body: rule.body.iter().map(convert_body_literal).collect(),
    }
}

fn convert_body_literal(lit: &BodyLiteral) -> EirBodyLiteral {
    match lit {
        BodyLiteral::Positive(atom) => EirBodyLiteral::Relational {
            negated: false,
            atom: convert_atom(atom),
        },
        BodyLiteral::Negated(atom) => EirBodyLiteral::Relational {
            negated: true,
            atom: convert_atom(atom),
        },
        BodyLiteral::Epistemic(lit) => EirBodyLiteral::Epistemic(EirEpistemicLiteral {
            op: convert_op(lit.op),
            negated: lit.negated,
            atom: convert_atom(&lit.atom),
        }),
        BodyLiteral::Comparison(_) => EirBodyLiteral::Constraint,
        BodyLiteral::IsExpr(_) => EirBodyLiteral::Binding,
        BodyLiteral::Univ(_) => EirBodyLiteral::Binding,
    }
}

fn convert_atom(atom: &Atom) -> EirAtom {
    EirAtom {
        predicate: atom.predicate.clone(),
        arity: atom.arity(),
        terms: atom.terms.iter().map(convert_term).collect(),
    }
}

fn convert_term(term: &Term) -> EirTerm {
    match term {
        Term::Variable(name) => EirTerm::Variable(name.clone()),
        Term::Anonymous => EirTerm::Anonymous,
        Term::Integer(value) => EirTerm::Integer(*value),
        Term::Float(value) => EirTerm::FloatBits(value.to_bits()),
        Term::String(value) => EirTerm::String(value.clone()),
        Term::Symbol(id) => EirTerm::Symbol(*id),
        Term::List(items) => EirTerm::List(items.iter().map(convert_term).collect()),
        Term::Cons { head, tail } => EirTerm::Cons {
            head: Box::new(convert_term(head)),
            tail: Box::new(convert_term(tail)),
        },
        Term::Compound { functor, args } => EirTerm::Compound {
            functor: functor.clone(),
            args: args.iter().map(convert_term).collect(),
        },
        Term::PredRef(name) => EirTerm::PredRef(name.clone()),
        Term::Aggregate(agg) => EirTerm::Aggregate {
            op: convert_agg_op(agg.op).to_string(),
            variable: agg.variable.clone(),
        },
    }
}

fn convert_agg_op(op: AggOp) -> &'static str {
    match op {
        AggOp::Count => "count",
        AggOp::Sum => "sum",
        AggOp::Min => "min",
        AggOp::Max => "max",
        AggOp::LogSumExp => "logsumexp",
    }
}

fn convert_mode(mode: EpistemicMode) -> EirEpistemicMode {
    match mode {
        EpistemicMode::G91 => EirEpistemicMode::G91,
        EpistemicMode::Faeel => EirEpistemicMode::Faeel,
    }
}

fn convert_op(op: EpistemicOp) -> EirEpistemicOp {
    match op {
        EpistemicOp::Know => EirEpistemicOp::Know,
        EpistemicOp::Possible => EirEpistemicOp::Possible,
    }
}
