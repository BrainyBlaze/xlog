//! Structured diagnostics for rule provenance and proof-oriented query traces.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use xlog_core::symbol;

use crate::ast::{AggOp, ArithExpr, Atom, BodyLiteral, CompOp, Program, Query, Rule, Term};

/// Introspection record for one compiled or generated rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleProvenance {
    /// Stable rule identifier derived from source kind and rule text.
    pub rule_id: String,
    /// `source`, `generated`, `mined`, `imported`, or `runtime_injected`.
    pub source_kind: String,
    /// Head predicate and arguments.
    pub head: String,
    /// Source location summary when the rule came from source text.
    pub source_span: Option<String>,
    /// Deterministic hash for generated-rule lineage.
    pub generation_trace_hash: Option<String>,
    /// Relations that support or activate this rule.
    pub support_relation_ids: Vec<String>,
    /// Relations that falsify or counterexample this rule, when known.
    pub counterexample_relation_ids: Vec<String>,
}

/// Proof-oriented explanation record for one source query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryProofTrace {
    /// Query atom in source-like syntax.
    pub query: String,
    /// Internal query-result relation name.
    pub answer_relation: String,
    /// Rule ids whose heads match the query predicate.
    pub rule_ids: Vec<String>,
    /// Source facts referenced by the matching rules.
    pub source_facts: Vec<String>,
    /// Candidate rules rejected during derivation, when available.
    pub rejected_alternatives: Vec<String>,
}

/// Build source and generated rule-provenance records for a program.
pub fn build_rule_provenance(
    program: &Program,
    generated_predicates: &[String],
) -> Vec<RuleProvenance> {
    let mut records = Vec::with_capacity(program.rules.len() + generated_predicates.len());
    for (idx, rule) in program.rules.iter().enumerate() {
        records.push(RuleProvenance {
            rule_id: source_rule_id(idx, rule),
            source_kind: "source".to_string(),
            head: format_atom(&rule.head),
            source_span: Some(format!("rule_index:{}", idx)),
            generation_trace_hash: None,
            support_relation_ids: rule
                .body_predicates()
                .into_iter()
                .map(str::to_string)
                .collect(),
            counterexample_relation_ids: Vec::new(),
        });
    }
    for (idx, predicate) in generated_predicates.iter().enumerate() {
        records.push(RuleProvenance {
            rule_id: generated_rule_id(idx, predicate),
            source_kind: "generated".to_string(),
            head: predicate.clone(),
            source_span: None,
            generation_trace_hash: Some(stable_hash_hex(&format!("generated:{}", predicate))),
            support_relation_ids: vec![predicate.clone()],
            counterexample_relation_ids: Vec::new(),
        });
    }
    records
}

/// Build coarse proof traces that name source rules and source facts per query.
pub fn build_query_proof_traces(program: &Program) -> Vec<QueryProofTrace> {
    program
        .queries
        .iter()
        .enumerate()
        .map(|(query_idx, query)| proof_trace_for_query(program, query_idx, query))
        .collect()
}

/// Format an atom in source-like syntax.
pub fn format_atom(atom: &Atom) -> String {
    let terms = atom
        .terms
        .iter()
        .map(format_term)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({})", atom.predicate, terms)
}

fn proof_trace_for_query(program: &Program, query_idx: usize, query: &Query) -> QueryProofTrace {
    let mut rule_ids = Vec::new();
    let mut body_predicates = HashSet::new();
    for (idx, rule) in program.rules.iter().enumerate() {
        if !rule.is_fact() && rule.head.predicate == query.atom.predicate {
            rule_ids.push(source_rule_id(idx, rule));
            for predicate in rule.body_predicates() {
                body_predicates.insert(predicate.to_string());
            }
        }
    }

    let source_facts = program
        .rules
        .iter()
        .filter(|rule| rule.is_fact() && body_predicates.contains(&rule.head.predicate))
        .map(format_rule)
        .collect();

    QueryProofTrace {
        query: format_atom(&query.atom),
        answer_relation: format!("__xlog_query_{}", query_idx),
        rule_ids,
        source_facts,
        rejected_alternatives: Vec::new(),
    }
}

fn source_rule_id(idx: usize, rule: &Rule) -> String {
    format!("source:{:04}:{}", idx, stable_hash_hex(&format_rule(rule)))
}

fn generated_rule_id(idx: usize, predicate: &str) -> String {
    format!("generated:{:04}:{}", idx, stable_hash_hex(predicate))
}

fn stable_hash_hex(value: &str) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn format_rule(rule: &Rule) -> String {
    if rule.is_fact() {
        return format!("{}.", format_atom(&rule.head));
    }
    let body = rule
        .body
        .iter()
        .map(format_body_literal)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} :- {}.", format_atom(&rule.head), body)
}

fn format_body_literal(literal: &BodyLiteral) -> String {
    match literal {
        BodyLiteral::Positive(atom) => format_atom(atom),
        BodyLiteral::Negated(atom) => format!("not {}", format_atom(atom)),
        BodyLiteral::Comparison(cmp) => {
            format!(
                "{} {} {}",
                format_term(&cmp.left),
                format_comp_op(cmp.op),
                format_term(&cmp.right)
            )
        }
        BodyLiteral::IsExpr(expr) => {
            format!("{} is {}", expr.target, format_arith_expr(&expr.expr))
        }
        BodyLiteral::Univ(univ) => {
            format!(
                "{} =.. {}",
                format_term(&univ.term),
                format_term(&univ.parts)
            )
        }
    }
}

fn format_term(term: &Term) -> String {
    match term {
        Term::Variable(name) => name.clone(),
        Term::Anonymous => "_".to_string(),
        Term::Integer(value) => value.to_string(),
        Term::Float(value) => value.to_string(),
        Term::String(value) => format!("\"{}\"", value),
        Term::Symbol(id) => symbol::resolve(*id),
        Term::List(items) => {
            let values = items.iter().map(format_term).collect::<Vec<_>>().join(", ");
            format!("[{}]", values)
        }
        Term::Cons { head, tail } => {
            format!("[{} | {}]", format_term(head), format_term(tail))
        }
        Term::Compound { functor, args } => {
            let values = args.iter().map(format_term).collect::<Vec<_>>().join(", ");
            format!("{}({})", functor, values)
        }
        Term::PredRef(name) => name.clone(),
        Term::Aggregate(agg) => format!("{}({})", format_agg_op(agg.op), agg.variable),
    }
}

fn format_arith_expr(expr: &ArithExpr) -> String {
    match expr {
        ArithExpr::Variable(name) => name.clone(),
        ArithExpr::Integer(value) => value.to_string(),
        ArithExpr::Float(value) => value.to_string(),
        ArithExpr::Add(l, r) => format!("({} + {})", format_arith_expr(l), format_arith_expr(r)),
        ArithExpr::Sub(l, r) => format!("({} - {})", format_arith_expr(l), format_arith_expr(r)),
        ArithExpr::Mul(l, r) => format!("({} * {})", format_arith_expr(l), format_arith_expr(r)),
        ArithExpr::Div(l, r) => format!("({} / {})", format_arith_expr(l), format_arith_expr(r)),
        ArithExpr::Mod(l, r) => format!("({} % {})", format_arith_expr(l), format_arith_expr(r)),
        ArithExpr::Abs(value) => format!("abs({})", format_arith_expr(value)),
        ArithExpr::Min(l, r) => format!("min({}, {})", format_arith_expr(l), format_arith_expr(r)),
        ArithExpr::Max(l, r) => format!("max({}, {})", format_arith_expr(l), format_arith_expr(r)),
        ArithExpr::Pow(l, r) => format!("pow({}, {})", format_arith_expr(l), format_arith_expr(r)),
        ArithExpr::Cast(value, ty) => format!("cast({}, {:?})", format_arith_expr(value), ty),
        ArithExpr::FuncCall { name, args } => {
            let values = args
                .iter()
                .map(format_arith_expr)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", name, values)
        }
        ArithExpr::Conditional {
            cond_left,
            cond_op,
            cond_right,
            then_expr,
            else_expr,
        } => format!(
            "if {} {} {} then {} else {}",
            format_arith_expr(cond_left),
            format_comp_op(*cond_op),
            format_arith_expr(cond_right),
            format_arith_expr(then_expr),
            format_arith_expr(else_expr)
        ),
    }
}

fn format_comp_op(op: CompOp) -> &'static str {
    match op {
        CompOp::Eq => "==",
        CompOp::Ne => "!=",
        CompOp::Lt => "<",
        CompOp::Le => "<=",
        CompOp::Gt => ">",
        CompOp::Ge => ">=",
    }
}

fn format_agg_op(op: AggOp) -> &'static str {
    match op {
        AggOp::Count => "count",
        AggOp::Sum => "sum",
        AggOp::Min => "min",
        AggOp::Max => "max",
        AggOp::LogSumExp => "logsumexp",
    }
}
