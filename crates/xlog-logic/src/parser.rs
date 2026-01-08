//! Parser for XLOG programs using Pest
//!
//! This module parses XLOG source text into AST structures.
//! It uses the Pest parser generator with a grammar defined in `grammar.pest`.

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use xlog_core::{ScalarType, XlogError, Result};

use crate::ast::{
    AggExpr, AggOp, Atom, BodyLiteral, CompOp, Comparison, Constraint,
    DomainDecl, PredDecl, Program, Query, Rule as AstRule, Term,
};

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct XlogParser;

/// Parse result containing the parsed pairs (for low-level access)
pub type ParseResult<'a> = pest::iterators::Pairs<'a, Rule>;

/// Parse an XLOG program string into an AST Program
pub fn parse_program(input: &str) -> Result<Program> {
    let pairs = XlogParser::parse(Rule::program, input)
        .map_err(|e| XlogError::Parse(e.to_string()))?;

    build_program(pairs)
}

/// Parse a single statement (low-level, returns pest pairs)
pub fn parse_statement(input: &str) -> Result<ParseResult<'_>> {
    XlogParser::parse(Rule::statement, input)
        .map_err(|e| XlogError::Parse(e.to_string()))
}

/// Parse a single atom (low-level, returns pest pairs)
pub fn parse_atom(input: &str) -> Result<ParseResult<'_>> {
    XlogParser::parse(Rule::atom, input)
        .map_err(|e| XlogError::Parse(e.to_string()))
}

// =============================================================================
// AST Building Functions
// =============================================================================

/// Build a Program AST from parsed pest pairs
fn build_program(pairs: ParseResult<'_>) -> Result<Program> {
    let mut program = Program::new();

    for pair in pairs {
        if pair.as_rule() == Rule::program {
            for inner in pair.into_inner() {
                match inner.as_rule() {
                    Rule::statement => {
                        build_statement(inner, &mut program)?;
                    }
                    Rule::EOI => {}
                    _ => {}
                }
            }
        }
    }

    Ok(program)
}

/// Build a statement and add it to the program
fn build_statement(pair: Pair<'_, Rule>, program: &mut Program) -> Result<()> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::domain_decl => {
                program.domains.push(build_domain_decl(inner)?);
            }
            Rule::pred_decl => {
                program.predicates.push(build_pred_decl(inner)?);
            }
            Rule::rule_def => {
                program.rules.push(build_rule(inner)?);
            }
            Rule::fact => {
                program.rules.push(build_fact(inner)?);
            }
            Rule::constraint => {
                program.constraints.push(build_constraint(inner)?);
            }
            Rule::query => {
                program.queries.push(build_query(inner)?);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Build a domain declaration
fn build_domain_decl(pair: Pair<'_, Rule>) -> Result<DomainDecl> {
    let mut inner = pair.into_inner();
    let name = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing domain name".to_string()))?
        .as_str()
        .to_string();
    let type_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing domain type".to_string()))?;
    let typ = build_type_spec(type_pair)?;

    Ok(DomainDecl { name, typ })
}

/// Build a predicate declaration
fn build_pred_decl(pair: Pair<'_, Rule>) -> Result<PredDecl> {
    let mut inner = pair.into_inner();
    let name = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing predicate name".to_string()))?
        .as_str()
        .to_string();

    let mut types = Vec::new();
    for type_pair in inner {
        if type_pair.as_rule() == Rule::type_list {
            for t in type_pair.into_inner() {
                types.push(build_type_spec(t)?);
            }
        }
    }

    Ok(PredDecl { name, types })
}

/// Build a type specification
fn build_type_spec(pair: Pair<'_, Rule>) -> Result<ScalarType> {
    let type_str = pair.as_str();
    match type_str {
        "u32" => Ok(ScalarType::U32),
        "u64" => Ok(ScalarType::U64),
        "i32" => Ok(ScalarType::I32),
        "i64" => Ok(ScalarType::I64),
        "f32" => Ok(ScalarType::F32),
        "f64" => Ok(ScalarType::F64),
        "bool" => Ok(ScalarType::Bool),
        "symbol" => Ok(ScalarType::Symbol),
        _ => Err(XlogError::Parse(format!("Unknown type: {}", type_str))),
    }
}

/// Build a rule (with body)
fn build_rule(pair: Pair<'_, Rule>) -> Result<AstRule> {
    let mut inner = pair.into_inner();
    let head_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing rule head".to_string()))?;
    let head = build_head(head_pair)?;

    let body_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing rule body".to_string()))?;
    let body = build_body(body_pair)?;

    Ok(AstRule { head, body })
}

/// Build a fact (rule with empty body)
fn build_fact(pair: Pair<'_, Rule>) -> Result<AstRule> {
    let mut inner = pair.into_inner();
    let atom_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing fact atom".to_string()))?;
    let head = build_atom(atom_pair)?;

    Ok(AstRule { head, body: vec![] })
}

/// Build a constraint
fn build_constraint(pair: Pair<'_, Rule>) -> Result<Constraint> {
    let mut inner = pair.into_inner();
    let body_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing constraint body".to_string()))?;
    let body = build_body(body_pair)?;

    Ok(Constraint { body })
}

/// Build a query
fn build_query(pair: Pair<'_, Rule>) -> Result<Query> {
    let mut inner = pair.into_inner();
    let atom_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing query atom".to_string()))?;
    let atom = build_atom(atom_pair)?;

    Ok(Query { atom })
}

/// Build a head (atom that may contain aggregates)
fn build_head(pair: Pair<'_, Rule>) -> Result<Atom> {
    let mut inner = pair.into_inner();
    let predicate = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing head predicate".to_string()))?
        .as_str()
        .to_string();

    let mut terms = Vec::new();
    for term_list in inner {
        if term_list.as_rule() == Rule::head_term_list {
            for head_term in term_list.into_inner() {
                terms.push(build_head_term(head_term)?);
            }
        }
    }

    Ok(Atom { predicate, terms })
}

/// Build a head term (can be aggregate or regular term)
fn build_head_term(pair: Pair<'_, Rule>) -> Result<Term> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| XlogError::Parse("Empty head term".to_string()))?;

    match inner.as_rule() {
        Rule::aggregate => build_aggregate(inner),
        Rule::agg_term => {
            // agg_term can contain aggregate or term
            let agg_inner = inner.into_inner().next()
                .ok_or_else(|| XlogError::Parse("Empty agg_term".to_string()))?;
            match agg_inner.as_rule() {
                Rule::aggregate => build_aggregate(agg_inner),
                Rule::term => build_term(agg_inner),
                _ => build_term(agg_inner),
            }
        }
        Rule::term => build_term(inner),
        _ => build_term(inner),
    }
}

/// Build an aggregate expression
fn build_aggregate(pair: Pair<'_, Rule>) -> Result<Term> {
    let mut inner = pair.into_inner();
    let op_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing aggregate operator".to_string()))?;
    let op = match op_pair.as_str() {
        "count" => AggOp::Count,
        "sum" => AggOp::Sum,
        "min" => AggOp::Min,
        "max" => AggOp::Max,
        _ => return Err(XlogError::Parse(format!("Unknown aggregate: {}", op_pair.as_str()))),
    };

    let var_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing aggregate variable".to_string()))?;
    let variable = var_pair.as_str().to_string();

    Ok(Term::Aggregate(AggExpr { op, variable }))
}

/// Build an atom
fn build_atom(pair: Pair<'_, Rule>) -> Result<Atom> {
    let mut inner = pair.into_inner();
    let predicate = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing atom predicate".to_string()))?
        .as_str()
        .to_string();

    let mut terms = Vec::new();
    for term_list in inner {
        if term_list.as_rule() == Rule::term_list {
            for term in term_list.into_inner() {
                terms.push(build_term(term)?);
            }
        }
    }

    Ok(Atom { predicate, terms })
}

/// Build a body (list of literals)
fn build_body(pair: Pair<'_, Rule>) -> Result<Vec<BodyLiteral>> {
    let mut literals = Vec::new();

    for lit in pair.into_inner() {
        literals.push(build_body_literal(lit)?);
    }

    Ok(literals)
}

/// Build a body literal
fn build_body_literal(pair: Pair<'_, Rule>) -> Result<BodyLiteral> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| XlogError::Parse("Empty body literal".to_string()))?;

    match inner.as_rule() {
        Rule::negated_atom => {
            let atom_pair = inner.into_inner().next()
                .ok_or_else(|| XlogError::Parse("Missing negated atom".to_string()))?;
            Ok(BodyLiteral::Negated(build_atom(atom_pair)?))
        }
        Rule::atom => {
            Ok(BodyLiteral::Positive(build_atom(inner)?))
        }
        Rule::comparison => {
            Ok(BodyLiteral::Comparison(build_comparison(inner)?))
        }
        _ => Err(XlogError::Parse(format!("Unknown body literal: {:?}", inner.as_rule()))),
    }
}

/// Build a comparison
fn build_comparison(pair: Pair<'_, Rule>) -> Result<Comparison> {
    let mut inner = pair.into_inner();

    let left_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing comparison left operand".to_string()))?;
    let left = build_term(left_pair)?;

    let op_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing comparison operator".to_string()))?;
    let op = match op_pair.as_str() {
        "==" | "=" => CompOp::Eq,
        "!=" => CompOp::Ne,
        "<" => CompOp::Lt,
        "<=" => CompOp::Le,
        ">" => CompOp::Gt,
        ">=" => CompOp::Ge,
        _ => return Err(XlogError::Parse(format!("Unknown comparison operator: {}", op_pair.as_str()))),
    };

    let right_pair = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing comparison right operand".to_string()))?;
    let right = build_term(right_pair)?;

    Ok(Comparison { left, op, right })
}

/// Build a term
fn build_term(pair: Pair<'_, Rule>) -> Result<Term> {
    // term can directly contain variable, integer, etc. or be wrapped
    let inner = if pair.as_rule() == Rule::term {
        pair.into_inner().next()
            .ok_or_else(|| XlogError::Parse("Empty term".to_string()))?
    } else {
        pair
    };

    match inner.as_rule() {
        Rule::variable => Ok(Term::Variable(inner.as_str().to_string())),
        Rule::integer => {
            let val: i64 = inner.as_str().parse()
                .map_err(|_| XlogError::Parse(format!("Invalid integer: {}", inner.as_str())))?;
            Ok(Term::Integer(val))
        }
        Rule::float_num => {
            let val: f64 = inner.as_str().parse()
                .map_err(|_| XlogError::Parse(format!("Invalid float: {}", inner.as_str())))?;
            Ok(Term::Float(val))
        }
        Rule::string_lit => {
            let s = inner.as_str();
            // Remove quotes
            let unquoted = &s[1..s.len()-1];
            Ok(Term::String(unquoted.to_string()))
        }
        Rule::ident => Ok(Term::Symbol(inner.as_str().to_string())),
        _ => Err(XlogError::Parse(format!("Unknown term type: {:?}", inner.as_rule()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fact() {
        let input = "edge(1, 2).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse fact: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.rules.len(), 1);
        assert!(program.rules[0].is_fact());
        assert_eq!(program.rules[0].head.predicate, "edge");
        assert_eq!(program.rules[0].head.terms.len(), 2);
        assert_eq!(program.rules[0].head.terms[0], Term::Integer(1));
        assert_eq!(program.rules[0].head.terms[1], Term::Integer(2));
    }

    #[test]
    fn test_parse_rule() {
        let input = "reach(X, Y) :- edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse rule: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.rules.len(), 1);
        assert!(!program.rules[0].is_fact());
        assert_eq!(program.rules[0].head.predicate, "reach");
        assert_eq!(program.rules[0].body.len(), 1);
    }

    #[test]
    fn test_parse_recursive_rule() {
        let input = "reach(X, Z) :- reach(X, Y), edge(Y, Z).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse recursive rule: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.rules.len(), 1);
        assert_eq!(program.rules[0].body.len(), 2);
    }

    #[test]
    fn test_parse_negation() {
        let input = "isolated(X) :- node(X), not edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse negation: {:?}", result.err());

        let program = result.unwrap();
        assert!(program.rules[0].has_negation());
        assert!(matches!(&program.rules[0].body[1], BodyLiteral::Negated(_)));
    }

    #[test]
    fn test_parse_aggregate() {
        let input = "out_degree(X, count(Y)) :- edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse aggregate: {:?}", result.err());

        let program = result.unwrap();
        assert!(program.rules[0].has_aggregation());
        if let Term::Aggregate(agg) = &program.rules[0].head.terms[1] {
            assert_eq!(agg.op, AggOp::Count);
            assert_eq!(agg.variable, "Y");
        } else {
            panic!("Expected aggregate term");
        }
    }

    #[test]
    fn test_parse_constraint() {
        let input = ":- reach(X, X).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse constraint: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.constraints.len(), 1);
        assert_eq!(program.constraints[0].body.len(), 1);
    }

    #[test]
    fn test_parse_query() {
        let input = "?- reach(1, N).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse query: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.queries.len(), 1);
        assert_eq!(program.queries[0].atom.predicate, "reach");
    }

    #[test]
    fn test_parse_full_program() {
        let input = r#"
            edge(1, 2).
            edge(2, 3).
            edge(3, 4).
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).
            ?- reach(1, N).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse full program: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.rules.len(), 5); // 3 facts + 2 rules
        assert_eq!(program.queries.len(), 1);
        assert_eq!(program.facts().count(), 3);
        assert_eq!(program.proper_rules().count(), 2);
    }

    #[test]
    fn test_parse_comparison() {
        let input = "small(X) :- value(X), X < 10.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse comparison: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.rules[0].body.len(), 2);
        if let BodyLiteral::Comparison(cmp) = &program.rules[0].body[1] {
            assert_eq!(cmp.op, CompOp::Lt);
            assert_eq!(cmp.left, Term::Variable("X".to_string()));
            assert_eq!(cmp.right, Term::Integer(10));
        } else {
            panic!("Expected comparison");
        }
    }

    #[test]
    fn test_parse_pred_decl() {
        let input = "pred edge(u32, u32).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse pred decl: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.predicates.len(), 1);
        assert_eq!(program.predicates[0].name, "edge");
        assert_eq!(program.predicates[0].types.len(), 2);
        assert_eq!(program.predicates[0].types[0], ScalarType::U32);
        assert_eq!(program.predicates[0].types[1], ScalarType::U32);
    }
}
