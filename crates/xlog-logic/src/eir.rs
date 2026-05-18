//! Frontend construction for the Epistemic Intermediate Representation.

use xlog_core::Result;
use xlog_ir::{
    EirAtom, EirBodyLiteral, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EirProgram,
    EirRule,
};

use crate::ast::{Atom, BodyLiteral, EpistemicMode, EpistemicOp, Program, Rule as AstRule};

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
    }
}

fn convert_atom(atom: &Atom) -> EirAtom {
    EirAtom {
        predicate: atom.predicate.clone(),
        arity: atom.arity(),
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
