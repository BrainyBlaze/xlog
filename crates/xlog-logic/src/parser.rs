//! Parser for XLOG programs using Pest
//!
//! This module parses XLOG source text into AST structures.
//! It uses the Pest parser generator with a grammar defined in `grammar.pest`.

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use xlog_core::{ScalarType, XlogError, Result, symbol};

use crate::ast::{
    AggExpr, AggOp, AnnotatedDisjunction, ArithExpr, Atom, BodyLiteral, CompOp, Comparison,
    CondExpr, Constraint, DomainDecl, Evidence, FuncBody, FuncDef, FuncParam, IsExpr, PredDecl,
    ProbCache, ProbEngine, ProbFact, ProbQuery, Program, Query, Rule as AstRule, Term, UseDecl,
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
            Rule::use_stmt => {
                program.imports.push(build_use_stmt(inner));
            }
            Rule::domain_decl => {
                program.domains.push(build_domain_decl(inner)?);
            }
            Rule::pred_decl => {
                program.predicates.push(build_pred_decl(inner)?);
            }
            Rule::pragma => {
                apply_pragma(inner, program)?;
            }
            Rule::rule_def => {
                program.rules.push(build_rule(inner)?);
            }
            Rule::fact => {
                program.rules.push(build_fact(inner)?);
            }
            Rule::prob_fact => {
                program.prob_facts.push(build_prob_fact(inner)?);
            }
            Rule::annotated_disjunction => {
                program
                    .annotated_disjunctions
                    .push(build_annotated_disjunction(inner)?);
            }
            Rule::evidence_stmt => {
                program.evidence.push(build_evidence(inner)?);
            }
            Rule::prob_query => {
                program.prob_queries.push(build_prob_query(inner)?);
            }
            Rule::constraint => {
                program.constraints.push(build_constraint(inner)?);
            }
            Rule::query => {
                program.queries.push(build_query(inner)?);
            }
            Rule::func_def => {
                program.functions.push(parse_func_def(inner)?);
            }
            _ => {}
        }
    }
    Ok(())
}

fn apply_pragma(pair: Pair<'_, Rule>, program: &mut Program) -> Result<()> {
    let pragma = pair
        .into_inner()
        .next()
        .ok_or_else(|| XlogError::Parse("Empty pragma".to_string()))?;

    match pragma.as_rule() {
        Rule::pragma_prob_engine => {
            let value = pragma
                .into_inner()
                .next()
                .ok_or_else(|| XlogError::Parse("Missing prob_engine value".to_string()))?;
            let engine = match value.as_str() {
                "exact_ddnnf" => ProbEngine::ExactDdnnf,
                "mc" => ProbEngine::Mc,
                other => {
                    return Err(XlogError::Parse(format!(
                        "Unknown prob_engine value: {}",
                        other
                    )))
                }
            };
            program.directives.prob_engine = Some(engine);
        }
        Rule::pragma_prob_cache => {
            let value = pragma
                .into_inner()
                .next()
                .ok_or_else(|| XlogError::Parse("Missing prob_cache value".to_string()))?;
            let cache = match value.as_str() {
                "on" => ProbCache::On,
                "off" => ProbCache::Off,
                other => {
                    return Err(XlogError::Parse(format!(
                        "Unknown prob_cache value: {}",
                        other
                    )))
                }
            };
            program.directives.prob_cache = Some(cache);
        }
        Rule::pragma_max_recursion => {
            let value = pragma
                .into_inner()
                .next()
                .ok_or_else(|| XlogError::Parse("Missing max_recursion_depth value".to_string()))?;
            let depth: u32 = value.as_str().parse().map_err(|_| {
                XlogError::Parse(format!(
                    "Invalid max_recursion_depth value: {}",
                    value.as_str()
                ))
            })?;
            program.directives.max_recursion_depth = Some(depth);
        }
        _ => {}
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

/// Parse a module path (e.g., "utils/math" -> ["utils", "math"])
fn parse_module_path(pair: Pair<Rule>) -> Vec<String> {
    pair.as_str()
        .split('/')
        .map(|s| s.to_string())
        .collect()
}

/// Build a use statement
fn build_use_stmt(pair: Pair<Rule>) -> UseDecl {
    let mut inner = pair.into_inner();

    // Parse module path
    let path_pair = inner.next().unwrap();
    let module_path = parse_module_path(path_pair);

    // Parse optional import list
    let imports = inner.next().map(|import_list| {
        import_list.into_inner()
            .map(|p| p.as_str().to_string())
            .collect()
    });

    UseDecl { module_path, imports }
}

/// Build a predicate declaration
fn build_pred_decl(pair: Pair<'_, Rule>) -> Result<PredDecl> {
    let mut inner = pair.into_inner();
    let mut is_private = false;

    // Check for private modifier
    let first = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing predicate name".to_string()))?;

    let name_pair = if first.as_rule() == Rule::private_mod {
        is_private = true;
        inner.next()
            .ok_or_else(|| XlogError::Parse("Missing predicate name after private".to_string()))?
    } else {
        first
    };

    let name = name_pair.as_str().to_string();

    let mut types = Vec::new();
    for type_pair in inner {
        if type_pair.as_rule() == Rule::type_list {
            for t in type_pair.into_inner() {
                types.push(build_type_spec(t)?);
            }
        }
    }

    Ok(PredDecl { name, types, is_private })
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

/// Parse a function parameter: X or X: f64
fn parse_func_param(pair: Pair<'_, Rule>) -> Result<FuncParam> {
    let mut inner = pair.into_inner();
    let name = inner
        .next()
        .ok_or_else(|| XlogError::Parse("Missing parameter name".to_string()))?
        .as_str()
        .to_string();
    let typ = inner
        .next()
        .map(|ta| {
            // type_annotation contains type_spec
            build_type_spec(
                ta.into_inner()
                    .next()
                    .expect("type_annotation must contain type_spec"),
            )
        })
        .transpose()?;
    Ok(FuncParam { name, typ })
}

/// Parse a comparison operator
fn parse_cmp_op(pair: Pair<'_, Rule>) -> CompOp {
    match pair.as_str() {
        "==" | "=" => CompOp::Eq,
        "!=" => CompOp::Ne,
        "<" => CompOp::Lt,
        "<=" => CompOp::Le,
        ">" => CompOp::Gt,
        ">=" => CompOp::Ge,
        _ => unreachable!("unexpected comparison operator: {}", pair.as_str()),
    }
}

/// Parse a conditional expression: if X < 0 then 0 - X else X
fn parse_cond_expr(pair: Pair<'_, Rule>) -> Result<CondExpr> {
    let mut inner = pair.into_inner();

    // Parse condition test
    let cond_test = inner
        .next()
        .ok_or_else(|| XlogError::Parse("Missing condition test".to_string()))?;
    let mut test_inner = cond_test.into_inner();
    let cond_left = build_arith_expr(
        test_inner
            .next()
            .ok_or_else(|| XlogError::Parse("Missing left side of condition".to_string()))?,
    )?;
    let cond_op = parse_cmp_op(
        test_inner
            .next()
            .ok_or_else(|| XlogError::Parse("Missing condition operator".to_string()))?,
    );
    let cond_right = build_arith_expr(
        test_inner
            .next()
            .ok_or_else(|| XlogError::Parse("Missing right side of condition".to_string()))?,
    )?;

    // Parse then branch
    let then_branch = Box::new(parse_func_body(
        inner
            .next()
            .ok_or_else(|| XlogError::Parse("Missing then branch".to_string()))?,
    )?);

    // Parse else branch
    let else_branch = Box::new(parse_func_body(
        inner
            .next()
            .ok_or_else(|| XlogError::Parse("Missing else branch".to_string()))?,
    )?);

    Ok(CondExpr {
        cond_left,
        cond_op,
        cond_right,
        then_branch,
        else_branch,
    })
}

/// Parse a function body: arithmetic, conditional, or predicate-based
fn parse_func_body(pair: Pair<'_, Rule>) -> Result<FuncBody> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| XlogError::Parse("Empty function body".to_string()))?;
    match inner.as_rule() {
        Rule::func_body_pred => {
            let mut parts = inner.into_inner();
            let result = parts
                .next()
                .ok_or_else(|| XlogError::Parse("Missing result variable".to_string()))?
                .as_str()
                .to_string();
            let body = build_body(
                parts
                    .next()
                    .ok_or_else(|| XlogError::Parse("Missing predicate body".to_string()))?,
            )?;
            Ok(FuncBody::Predicate { result, body })
        }
        Rule::func_body_arith => {
            let arith_inner = inner
                .into_inner()
                .next()
                .ok_or_else(|| XlogError::Parse("Empty arithmetic body".to_string()))?;
            match arith_inner.as_rule() {
                Rule::cond_expr => Ok(FuncBody::Conditional(parse_cond_expr(arith_inner)?)),
                _ => Ok(FuncBody::Arithmetic(build_arith_expr(arith_inner)?)),
            }
        }
        _ => Err(XlogError::Parse(format!(
            "Unexpected rule in func_body: {:?}",
            inner.as_rule()
        ))),
    }
}

/// Parse a function definition
fn parse_func_def(pair: Pair<'_, Rule>) -> Result<FuncDef> {
    let mut inner = pair.into_inner();
    let mut is_private = false;

    // Check for private modifier
    let first = inner
        .next()
        .ok_or_else(|| XlogError::Parse("Empty function definition".to_string()))?;
    let name_pair = if first.as_rule() == Rule::private_mod {
        is_private = true;
        inner
            .next()
            .ok_or_else(|| XlogError::Parse("Missing function name after private".to_string()))?
    } else {
        first
    };

    let name = name_pair.as_str().to_string();

    let mut params = Vec::new();
    let mut return_type = None;
    let mut body = None;

    for p in inner {
        match p.as_rule() {
            Rule::func_params => {
                params = p
                    .into_inner()
                    .map(parse_func_param)
                    .collect::<Result<Vec<_>>>()?;
            }
            Rule::return_type => {
                return_type = Some(build_type_spec(
                    p.into_inner()
                        .next()
                        .ok_or_else(|| XlogError::Parse("Missing return type".to_string()))?,
                )?);
            }
            Rule::func_body => {
                body = Some(parse_func_body(p)?);
            }
            _ => {}
        }
    }

    Ok(FuncDef {
        name,
        params,
        return_type,
        body: body.ok_or_else(|| XlogError::Parse("Function must have a body".to_string()))?,
        is_private,
    })
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

fn build_prob_fact(pair: Pair<'_, Rule>) -> Result<ProbFact> {
    let choice = pair
        .into_inner()
        .next()
        .ok_or_else(|| XlogError::Parse("Missing probabilistic fact".to_string()))?;
    build_prob_choice(choice)
}

fn build_annotated_disjunction(pair: Pair<'_, Rule>) -> Result<AnnotatedDisjunction> {
    let mut choices = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::prob_choice {
            choices.push(build_prob_choice(inner)?);
        }
    }
    if choices.is_empty() {
        return Err(XlogError::Parse(
            "Annotated disjunction must have at least one choice".to_string(),
        ));
    }
    Ok(AnnotatedDisjunction { choices })
}

fn build_prob_choice(pair: Pair<'_, Rule>) -> Result<ProbFact> {
    let mut inner = pair.into_inner();
    let prob_pair = inner
        .next()
        .ok_or_else(|| XlogError::Parse("Missing probability".to_string()))?;
    let prob: f64 = prob_pair
        .as_str()
        .parse()
        .map_err(|_| XlogError::Parse(format!("Invalid probability: {}", prob_pair.as_str())))?;

    let atom_pair = inner
        .next()
        .ok_or_else(|| XlogError::Parse("Missing probabilistic atom".to_string()))?;
    let atom = build_atom(atom_pair)?;

    Ok(ProbFact { prob, atom })
}

fn build_evidence(pair: Pair<'_, Rule>) -> Result<Evidence> {
    let mut inner = pair.into_inner();
    let atom_pair = inner
        .next()
        .ok_or_else(|| XlogError::Parse("Missing evidence atom".to_string()))?;
    let atom = build_atom(atom_pair)?;

    let value_pair = inner
        .next()
        .ok_or_else(|| XlogError::Parse("Missing evidence value".to_string()))?;
    let value = match value_pair.as_str() {
        "true" => true,
        "false" => false,
        other => {
            return Err(XlogError::Parse(format!(
                "Invalid evidence value (expected true/false): {}",
                other
            )))
        }
    };

    Ok(Evidence { atom, value })
}

fn build_prob_query(pair: Pair<'_, Rule>) -> Result<ProbQuery> {
    let atom_pair = pair
        .into_inner()
        .next()
        .ok_or_else(|| XlogError::Parse("Missing query atom".to_string()))?;
    let atom = build_atom(atom_pair)?;
    Ok(ProbQuery { atom })
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
        "logsumexp" => AggOp::LogSumExp,
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
        Rule::is_expr => {
            Ok(BodyLiteral::IsExpr(build_is_expr(inner)?))
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
        Rule::var_or_anon => {
            // Unwrap var_or_anon to get either anonymous or variable
            let var_inner = inner.into_inner().next()
                .ok_or_else(|| XlogError::Parse("Empty var_or_anon".to_string()))?;
            match var_inner.as_rule() {
                Rule::anonymous => Ok(Term::Anonymous),
                Rule::variable => Ok(Term::Variable(var_inner.as_str().to_string())),
                _ => Err(XlogError::Parse(format!("Expected variable or anonymous, got: {:?}", var_inner.as_rule()))),
            }
        }
        Rule::variable => Ok(Term::Variable(inner.as_str().to_string())),
        Rule::anonymous => Ok(Term::Anonymous),
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
        Rule::ident => Ok(Term::Symbol(symbol::intern(inner.as_str()))),
        _ => Err(XlogError::Parse(format!("Unknown term type: {:?}", inner.as_rule()))),
    }
}

/// Build an arithmetic expression (handles additive operations + -)
/// Grammar: arith_expr = { arith_term ~ (arith_op_add ~ arith_term)* }
fn build_arith_expr(pair: Pair<'_, Rule>) -> Result<ArithExpr> {
    let mut inner = pair.into_inner();

    // First operand is always an arith_term
    let first = inner.next()
        .ok_or_else(|| XlogError::Parse("Empty arithmetic expression".to_string()))?;
    let mut result = build_arith_term(first)?;

    // Process remaining (operator, operand) pairs
    while let Some(op_pair) = inner.next() {
        let op_str = op_pair.as_str();
        let right_pair = inner.next()
            .ok_or_else(|| XlogError::Parse("Missing right operand in arith expression".to_string()))?;
        let right = build_arith_term(right_pair)?;

        result = match op_str {
            "+" => ArithExpr::Add(Box::new(result), Box::new(right)),
            "-" => ArithExpr::Sub(Box::new(result), Box::new(right)),
            _ => return Err(XlogError::Parse(format!("Unknown additive operator: {}", op_str))),
        };
    }

    Ok(result)
}

/// Build an arithmetic term (handles multiplicative operations * / %)
/// Grammar: arith_term = { arith_primary ~ (arith_op_mul ~ arith_primary)* }
fn build_arith_term(pair: Pair<'_, Rule>) -> Result<ArithExpr> {
    let mut inner = pair.into_inner();

    // First operand is always an arith_primary
    let first = inner.next()
        .ok_or_else(|| XlogError::Parse("Empty arithmetic term".to_string()))?;
    let mut result = build_arith_primary(first)?;

    // Process remaining (operator, operand) pairs
    while let Some(op_pair) = inner.next() {
        let op_str = op_pair.as_str();
        let right_pair = inner.next()
            .ok_or_else(|| XlogError::Parse("Missing right operand in arith term".to_string()))?;
        let right = build_arith_primary(right_pair)?;

        result = match op_str {
            "*" => ArithExpr::Mul(Box::new(result), Box::new(right)),
            "/" => ArithExpr::Div(Box::new(result), Box::new(right)),
            "%" => ArithExpr::Mod(Box::new(result), Box::new(right)),
            _ => return Err(XlogError::Parse(format!("Unknown multiplicative operator: {}", op_str))),
        };
    }

    Ok(result)
}

/// Build an arithmetic primary (leaf nodes, parentheses, and builtin functions)
/// Grammar: arith_primary = {
///     builtin_fn ~ "(" ~ arith_expr ~ ("," ~ (arith_expr | type_spec))* ~ ")" |
///     "(" ~ arith_expr ~ ")" |
///     variable |
///     integer |
///     float_num
/// }
fn build_arith_primary(pair: Pair<'_, Rule>) -> Result<ArithExpr> {
    let mut inner = pair.into_inner();
    let first = inner.next()
        .ok_or_else(|| XlogError::Parse("Empty arithmetic primary".to_string()))?;

    match first.as_rule() {
        Rule::builtin_fn => {
            let fn_name = first.as_str();
            // Collect all arguments
            let args: Vec<Pair<'_, Rule>> = inner.collect();

            match fn_name {
                "abs" => {
                    if args.len() != 1 {
                        return Err(XlogError::Parse("abs() takes exactly 1 argument".to_string()));
                    }
                    let arg = build_arith_expr(args.into_iter().next().unwrap())?;
                    Ok(ArithExpr::Abs(Box::new(arg)))
                }
                "min" => {
                    if args.len() != 2 {
                        return Err(XlogError::Parse("min() takes exactly 2 arguments".to_string()));
                    }
                    let mut args_iter = args.into_iter();
                    let arg1 = build_arith_expr(args_iter.next().unwrap())?;
                    let arg2 = build_arith_expr(args_iter.next().unwrap())?;
                    Ok(ArithExpr::Min(Box::new(arg1), Box::new(arg2)))
                }
                "max" => {
                    if args.len() != 2 {
                        return Err(XlogError::Parse("max() takes exactly 2 arguments".to_string()));
                    }
                    let mut args_iter = args.into_iter();
                    let arg1 = build_arith_expr(args_iter.next().unwrap())?;
                    let arg2 = build_arith_expr(args_iter.next().unwrap())?;
                    Ok(ArithExpr::Max(Box::new(arg1), Box::new(arg2)))
                }
                "pow" => {
                    if args.len() != 2 {
                        return Err(XlogError::Parse("pow() takes exactly 2 arguments".to_string()));
                    }
                    let mut args_iter = args.into_iter();
                    let arg1 = build_arith_expr(args_iter.next().unwrap())?;
                    let arg2 = build_arith_expr(args_iter.next().unwrap())?;
                    Ok(ArithExpr::Pow(Box::new(arg1), Box::new(arg2)))
                }
                "cast" => {
                    if args.len() != 2 {
                        return Err(XlogError::Parse("cast() takes exactly 2 arguments".to_string()));
                    }
                    let mut args_iter = args.into_iter();
                    let arg1 = build_arith_expr(args_iter.next().unwrap())?;
                    let type_pair = args_iter.next().unwrap();
                    let target_type = build_type_spec(type_pair)?;
                    Ok(ArithExpr::Cast(Box::new(arg1), target_type))
                }
                _ => Err(XlogError::Parse(format!("Unknown builtin function: {}", fn_name))),
            }
        }
        Rule::arith_expr => {
            // Parenthesized expression
            build_arith_expr(first)
        }
        Rule::variable => {
            Ok(ArithExpr::Variable(first.as_str().to_string()))
        }
        Rule::integer => {
            let val: i64 = first.as_str().parse()
                .map_err(|_| XlogError::Parse(format!("Invalid integer: {}", first.as_str())))?;
            Ok(ArithExpr::Integer(val))
        }
        Rule::float_num => {
            let val: f64 = first.as_str().parse()
                .map_err(|_| XlogError::Parse(format!("Invalid float: {}", first.as_str())))?;
            Ok(ArithExpr::Float(val))
        }
        Rule::func_call => {
            let mut call_inner = first.into_inner();
            let name = call_inner
                .next()
                .ok_or_else(|| XlogError::Parse("Missing function name".to_string()))?
                .as_str()
                .to_string();
            let args: Vec<ArithExpr> = call_inner
                .map(build_arith_expr)
                .collect::<Result<Vec<_>>>()?;
            Ok(ArithExpr::FuncCall { name, args })
        }
        _ => Err(XlogError::Parse(format!("Unexpected token in arith_primary: {:?}", first.as_rule()))),
    }
}

/// Build an is-expression: Z is X + Y
fn build_is_expr(pair: Pair<'_, Rule>) -> Result<IsExpr> {
    let mut inner = pair.into_inner();
    let target = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing target variable in is expression".to_string()))?
        .as_str()
        .to_string();
    let expr = build_arith_expr(inner.next()
        .ok_or_else(|| XlogError::Parse("Missing expression in is expression".to_string()))?)?;
    Ok(IsExpr { target, expr })
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
    fn test_parse_logsumexp_aggregate() {
        let input = "score(X, logsumexp(Y)) :- obs(X, Y).";
        let result = parse_program(input);
        assert!(
            result.is_ok(),
            "Failed to parse logsumexp aggregate: {:?}",
            result.err()
        );

        let program = result.unwrap();
        assert!(program.rules[0].has_aggregation());
        if let Term::Aggregate(agg) = &program.rules[0].head.terms[1] {
            assert_eq!(agg.op, AggOp::LogSumExp);
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

    #[test]
    fn test_parse_anonymous_wildcard() {
        // Test anonymous wildcard in body
        let input = "has_child(X) :- parent(X, _).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse anonymous wildcard: {:?}", result.err());

        let program = result.unwrap();
        assert_eq!(program.rules.len(), 1);
        let rule = &program.rules[0];
        assert_eq!(rule.head.predicate, "has_child");

        // Check body atom has anonymous term
        if let BodyLiteral::Positive(atom) = &rule.body[0] {
            assert_eq!(atom.predicate, "parent");
            assert_eq!(atom.terms.len(), 2);
            assert_eq!(atom.terms[0], Term::Variable("X".to_string()));
            assert_eq!(atom.terms[1], Term::Anonymous);
        } else {
            panic!("Expected positive atom");
        }
    }

    #[test]
    fn test_parse_multiple_wildcards() {
        // Multiple wildcards in same rule - each is independent
        let input = "exists(X) :- rel(X, _, _).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse multiple wildcards: {:?}", result.err());

        let program = result.unwrap();
        if let BodyLiteral::Positive(atom) = &program.rules[0].body[0] {
            assert_eq!(atom.terms.len(), 3);
            assert_eq!(atom.terms[0], Term::Variable("X".to_string()));
            assert_eq!(atom.terms[1], Term::Anonymous);
            assert_eq!(atom.terms[2], Term::Anonymous);
        }
    }

    #[test]
    fn test_parse_is_expr() {
        // Test that grammar accepts 'is' expressions (AST building is in Task 3)
        let input = "result(X, Z) :- input(X, Y), Z is Y + 1.";
        let result = XlogParser::parse(Rule::program, input);
        assert!(result.is_ok(), "Failed to parse is expression: {:?}", result.err());
    }

    #[test]
    fn test_parse_arithmetic_precedence() {
        // Multiplication before addition
        let input = "r(X, Z) :- p(X, A, B), Z is A + B * 2.";
        let result = parse_program(input).unwrap();
        let rule = &result.rules[0];
        assert_eq!(rule.body.len(), 2);
        assert!(matches!(&rule.body[1], BodyLiteral::IsExpr(_)));
    }

    #[test]
    fn test_parse_arithmetic_parentheses() {
        let input = "r(X, Z) :- p(X, A, B), Z is (A + B) * 2.";
        assert!(parse_program(input).is_ok());
    }

    #[test]
    fn test_parse_probabilistic_fact_syntax() {
        let input = "0.7::rain().";
        let result = parse_program(input);
        assert!(
            result.is_ok(),
            "Failed to parse probabilistic fact: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_annotated_disjunction_syntax() {
        let input = "0.6::coin(heads); 0.4::coin(tails).";
        let result = parse_program(input);
        assert!(
            result.is_ok(),
            "Failed to parse annotated disjunction: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_evidence_is_not_a_fact() {
        let input = "evidence(rain(), true).";
        let program = parse_program(input).unwrap();
        assert_eq!(
            program.rules.len(),
            0,
            "evidence/2 should not be parsed as a regular fact"
        );
    }

    #[test]
    fn test_parse_query_directive_is_not_a_fact() {
        let input = "query(reach(1,3)).";
        let program = parse_program(input).unwrap();
        assert_eq!(
            program.rules.len(),
            0,
            "query/1 should not be parsed as a regular fact"
        );
    }

    #[test]
    fn test_parse_prob_engine_pragma_syntax() {
        let input = "#pragma prob_engine = mc";
        let result = parse_program(input);
        assert!(
            result.is_ok(),
            "Failed to parse prob_engine pragma: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_probabilistic_fact_ast() {
        let program = parse_program("0.7::rain().").unwrap();
        assert_eq!(program.prob_facts.len(), 1);
        assert!((program.prob_facts[0].prob - 0.7).abs() < 1e-9);
        assert_eq!(program.prob_facts[0].atom.predicate, "rain");
        assert!(program.prob_facts[0].atom.terms.is_empty());
    }

    #[test]
    fn test_parse_annotated_disjunction_ast() {
        let program = parse_program("0.6::coin(heads); 0.4::coin(tails).").unwrap();
        assert_eq!(program.annotated_disjunctions.len(), 1);
        let ad = &program.annotated_disjunctions[0];
        assert_eq!(ad.choices.len(), 2);
        assert!((ad.choices[0].prob - 0.6).abs() < 1e-9);
        assert_eq!(ad.choices[0].atom.predicate, "coin");
        assert_eq!(ad.choices[0].atom.terms.len(), 1);
        assert_eq!(ad.choices[0].atom.terms[0], Term::Symbol(symbol::intern("heads")));
        assert!((ad.choices[1].prob - 0.4).abs() < 1e-9);
        assert_eq!(ad.choices[1].atom.terms[0], Term::Symbol(symbol::intern("tails")));
    }

    #[test]
    fn test_parse_evidence_ast() {
        let program = parse_program("evidence(rain(), true).").unwrap();
        assert_eq!(program.evidence.len(), 1);
        assert_eq!(program.evidence[0].atom.predicate, "rain");
        assert!(program.evidence[0].value);
    }

    #[test]
    fn test_parse_prob_query_ast() {
        let program = parse_program("query(reach(1,3)).").unwrap();
        assert_eq!(program.prob_queries.len(), 1);
        assert_eq!(program.prob_queries[0].atom.predicate, "reach");
        assert_eq!(program.prob_queries[0].atom.terms.len(), 2);
        assert_eq!(program.prob_queries[0].atom.terms[0], Term::Integer(1));
        assert_eq!(program.prob_queries[0].atom.terms[1], Term::Integer(3));
    }

    #[test]
    fn test_parse_prob_engine_pragma_ast() {
        let program = parse_program("#pragma prob_engine = mc").unwrap();
        assert_eq!(program.directives.prob_engine, Some(crate::ast::ProbEngine::Mc));
        assert_eq!(program.prob_engine(), crate::ast::ProbEngine::Mc);
    }

    #[test]
    fn test_parse_arithmetic_builtins() {
        let inputs = [
            "r(X, Z) :- p(X, Y), Z is abs(Y).",
            "r(X, Z) :- p(X, A, B), Z is min(A, B).",
            "r(X, Z) :- p(X, A, B), Z is max(A, B).",
            "r(X, Z) :- p(X, A, B), Z is pow(A, B).",
            "r(X, Z) :- p(X, Y), Z is cast(Y, f64).",
        ];
        for input in inputs {
            assert!(parse_program(input).is_ok(), "Failed to parse: {}", input);
        }
    }

    #[test]
    fn test_parse_arithmetic_nested() {
        let input = "r(X, Z) :- p(X, A, B, C), Z is abs(A - B) + min(B, C) * 2.";
        assert!(parse_program(input).is_ok());
    }
}
