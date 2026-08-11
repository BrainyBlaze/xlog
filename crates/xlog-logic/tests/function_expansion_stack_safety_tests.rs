use std::fmt::Write as _;

use xlog_logic::ast::ArithExpr;
use xlog_logic::expand::ExpansionContext;
use xlog_logic::function::FunctionRegistry;
use xlog_logic::{expand_program_functions, parse_program, Compiler};

const LAST_FUNCTION: usize = 999;
const MAX_DEPTH: u32 = 1000;

fn arithmetic_chain(prefix: &str, step_body: impl Fn(usize) -> String) -> String {
    let mut source = format!("func {prefix}_0(X) = X.\n");
    for index in 1..=LAST_FUNCTION {
        writeln!(source, "func {prefix}_{index}(X) = {}.", step_body(index))
            .expect("write generated function");
    }
    source
}

fn expand_expression_only_chain(prefix: &str, step_body: impl Fn(usize) -> String) {
    let source = arithmetic_chain(prefix, step_body);
    let program = parse_program(&source).expect("parse generated function chain");
    let registry = FunctionRegistry::from_program(&program).expect("validate function chain");
    let mut context = ExpansionContext::new(&registry, MAX_DEPTH);

    let result = context
        .expand_call(
            &format!("{prefix}_{LAST_FUNCTION}"),
            &[ArithExpr::Integer(1)],
        )
        .expect("expand through the expression-only API at the configured limit");
    drop(result);
}

#[test]
fn compiler_accepts_nested_argument_chain_in_constraint_at_expansion_limit() {
    let mut source = arithmetic_chain("nested", |index| format!("nested_{}(X + 1)", index - 1));
    source.push_str("pred input(i64).\ninput(1).\n");
    writeln!(
        source,
        ":- input(X), Result is nested_{}(X).",
        LAST_FUNCTION
    )
    .expect("write constraint caller");
    let program = parse_program(&source).expect("parse generated constraint chain");
    let expanded = expand_program_functions(&program, MAX_DEPTH)
        .expect("expand constraint chain at configured limit");
    let plan = Compiler::new()
        .compile_program(&expanded)
        .expect("compile expanded constraint chain");

    drop(plan);
    drop(expanded);
    drop(program);
}

#[test]
fn expression_only_api_accepts_forwarding_chain_at_expansion_limit() {
    expand_expression_only_chain("forward", |index| format!("forward_{}(X)", index - 1));
}

#[test]
fn expression_only_api_accepts_non_tail_chain_at_expansion_limit() {
    expand_expression_only_chain("non_tail", |index| format!("non_tail_{}(X) + 1", index - 1));
}

#[test]
fn expression_only_api_accepts_nested_argument_chain_at_expansion_limit() {
    expand_expression_only_chain("nested", |index| format!("nested_{}(X + 1)", index - 1));
}

#[test]
fn expression_only_api_accepts_conditional_chain_at_expansion_limit() {
    expand_expression_only_chain("conditional", |index| {
        format!("if X = 0 then X else conditional_{}(X)", index - 1)
    });
}
