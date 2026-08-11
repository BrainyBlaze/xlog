use tempfile::TempDir;
use xlog_ir::eir::{EirBodyLiteral, EirTerm};
use xlog_logic::ast::{ArithExpr, BodyLiteral, FuncBody, Program, Rule, Term};
use xlog_logic::expand::ExpansionContext;
use xlog_logic::function::{FunctionError, FunctionRegistry};
use xlog_logic::resolver::ModuleResolver;
use xlog_logic::{build_eir, expand_program_functions, parse_program, Compiler};

fn expand(source: &str, max_depth: u32) -> Result<Program, FunctionError> {
    let program = parse_program(source).expect("parse predicate-function program");
    expand_program_functions(&program, max_depth)
}

fn compile_expanded(source: &str) -> Program {
    let expanded = expand(source, 100).expect("expand predicate functions");
    let mut compiler = Compiler::new();
    compiler
        .compile_program(&expanded)
        .expect("compile expanded program");
    expanded
}

fn proper_rule<'a>(program: &'a Program, predicate: &str) -> &'a Rule {
    program
        .proper_rules()
        .find(|rule| rule.head.predicate == predicate)
        .expect("find proper rule")
}

fn variable(term: &Term) -> &str {
    match term {
        Term::Variable(name) => name,
        other => panic!("expected variable, got {other:?}"),
    }
}

fn compile_acyclic_function_chain(step_body: impl Fn(usize) -> String) {
    use std::fmt::Write as _;

    const LAST_FUNCTION: usize = 999;
    let mut source = String::from(
        "pred input(i64).\n\
         pred answer(i64).\n\
         input(1).\n\
         func chain_0(X) = X.\n",
    );
    for index in 1..=LAST_FUNCTION {
        writeln!(source, "func chain_{index}(X) = {}.", step_body(index))
            .expect("write generated function");
    }
    writeln!(
        source,
        "answer(Y) :- input(X), Y is chain_{LAST_FUNCTION}(X).\n?- answer(Y)."
    )
    .expect("write chain caller");

    let program = parse_program(&source).expect("parse a 1000-call acyclic chain");
    let expanded =
        expand_program_functions(&program, 1000).expect("expand a 1000-call acyclic chain");
    let mut compiler = Compiler::new();
    compiler
        .compile_program(&expanded)
        .expect("compile a 1000-call acyclic chain");
    drop(compiler);
    drop(expanded);
    drop(program);
}

fn predicate_function_chain_source(step_body: impl Fn(usize) -> String) -> String {
    use std::fmt::Write as _;

    let mut source = String::from(
        "pred parent(u32, u32).\npred answer(u32).\nparent(1, 2).\n\
         func lookup0(X) = Parent :- parent(X, Parent).\n",
    );
    for index in 1..=999 {
        writeln!(
            source,
            "func lookup{index}(X) = Result :- Result is {}.",
            step_body(index)
        )
        .expect("write predicate function");
    }
    source.push_str("answer(Y) :- Y is lookup999(1).\n?- answer(Y).\n");
    source
}

#[test]
fn predicate_function_locals_do_not_capture_caller_variables() {
    let expanded = compile_expanded(
        r#"
        pred parent(u32, u32).
        pred marker(u32, u32).
        pred answer(u32).

        func get_grandparent(Child) = Grandparent :-
            parent(Child, Parent), parent(Parent, Grandparent).

        parent(1, 2).
        parent(2, 3).
        marker(8, 9).
        answer(Result) :-
            marker(Parent, Grandparent),
            Result is get_grandparent(1).
        "#,
    );
    let rule = proper_rule(&expanded, "answer");

    let BodyLiteral::Positive(first_inserted) = &rule.body[1] else {
        panic!("expected first inserted predicate literal")
    };
    let BodyLiteral::Positive(second_inserted) = &rule.body[2] else {
        panic!("expected second inserted predicate literal")
    };
    let local_parent = variable(&first_inserted.terms[1]);
    let local_result = variable(&second_inserted.terms[1]);

    assert_ne!(local_parent, "Parent");
    assert_ne!(local_result, "Grandparent");
    assert_eq!(variable(&second_inserted.terms[0]), local_parent);
    assert!(matches!(
        &rule.body[3],
        BodyLiteral::IsExpr(binding)
            if binding.target == "Result"
                && matches!(&binding.expr, ArithExpr::Variable(name) if name == local_result)
    ));
}

#[test]
fn multiple_predicate_calls_expand_in_source_order_with_independent_results() {
    let source = r#"
        pred parent(u32, u32).
        pred pair(u32, u32).
        func get_parent(Child) = Parent :- parent(Child, Parent).
        parent(1, 2).
        parent(2, 3).
        pair(First, Second) :-
            First is get_parent(1),
            Second is get_parent(2).
    "#;
    let first_expansion = compile_expanded(source);
    let second_expansion = compile_expanded(source);
    assert_eq!(
        proper_rule(&first_expansion, "pair"),
        proper_rule(&second_expansion, "pair")
    );

    let body = &proper_rule(&first_expansion, "pair").body;
    assert_eq!(body.len(), 4);
    let BodyLiteral::Positive(first_call) = &body[0] else {
        panic!("expected first call body")
    };
    let BodyLiteral::Positive(second_call) = &body[2] else {
        panic!("expected second call body")
    };
    assert!(matches!(first_call.terms[0], Term::Integer(1)));
    assert!(matches!(second_call.terms[0], Term::Integer(2)));
    assert_ne!(
        variable(&first_call.terms[1]),
        variable(&second_call.terms[1])
    );
    assert!(matches!(&body[1], BodyLiteral::IsExpr(binding) if binding.target == "First"));
    assert!(matches!(&body[3], BodyLiteral::IsExpr(binding) if binding.target == "Second"));
}

#[test]
fn predicate_calls_in_one_expression_expand_left_to_right() {
    let expanded = compile_expanded(
        r#"
        pred parent(u32, u32).
        pred total(u32).
        func get_parent(Child) = Parent :- parent(Child, Parent).
        parent(1, 2).
        parent(2, 3).
        total(Result) :- Result is get_parent(1) + get_parent(2).
        "#,
    );
    let body = &proper_rule(&expanded, "total").body;

    assert_eq!(body.len(), 3);
    assert!(
        matches!(&body[0], BodyLiteral::Positive(atom) if matches!(atom.terms[0], Term::Integer(1)))
    );
    assert!(
        matches!(&body[1], BodyLiteral::Positive(atom) if matches!(atom.terms[0], Term::Integer(2)))
    );
    assert!(matches!(&body[2], BodyLiteral::IsExpr(binding) if binding.target == "Result"));
}

#[test]
fn nested_predicate_calls_expand_before_their_containing_binding() {
    let expanded = compile_expanded(
        r#"
        pred parent(u32, u32).
        pred answer(u32).

        func get_parent(Child) = Parent :- parent(Child, Parent).
        func get_grandparent(Child) = Grandparent :-
            Grandparent is get_parent(get_parent(Child)).

        parent(1, 2).
        parent(2, 3).
        answer(Result) :- Result is get_grandparent(1).
        "#,
    );
    let body = &proper_rule(&expanded, "answer").body;

    assert_eq!(body.len(), 3);
    assert!(
        matches!(&body[0], BodyLiteral::Positive(atom) if matches!(atom.terms[0], Term::Integer(1)))
    );
    assert!(matches!(&body[1], BodyLiteral::Positive(atom) if atom.predicate == "parent"));
    assert!(matches!(&body[2], BodyLiteral::IsExpr(binding) if binding.target == "Result"));
}

#[test]
fn trailing_predicate_result_binding_is_inlined_into_the_caller() {
    let expanded = compile_expanded(
        r#"
        pred parent(u32, u32).
        pred answer(u32).

        func get_parent(Child) = Parent :- parent(Child, Parent).
        func lookup(Child) = Result :- Result is get_parent(Child).

        parent(1, 2).
        answer(Result) :- Result is lookup(1).
        "#,
    );
    let body = &proper_rule(&expanded, "answer").body;

    assert_eq!(body.len(), 2, "redundant predicate-result alias: {body:?}");
    assert!(matches!(
        &body[0],
        BodyLiteral::Positive(atom) if atom.predicate == "parent"
    ));
    assert!(
        matches!(
            &body[1],
            BodyLiteral::IsExpr(binding)
                if binding.target == "Result"
                    && matches!(&binding.expr, ArithExpr::Variable(name)
                        if name.starts_with("__XLOG_FUNCTION_GET_PARENT_Parent_"))
        ),
        "unexpected caller binding: {body:?}"
    );
}

#[test]
fn trailing_predicate_arithmetic_binding_keeps_its_evaluation_point() {
    let expanded = compile_expanded(
        r#"
        pred parent(u32, u32).
        pred answer(u32).
        func get_parent(Child) = Parent :- parent(Child, Parent).
        func divide_by_parent(Child) = Result :-
            Parent is get_parent(Child), Result is cast(1, u32) / Parent.
        parent(1, 2).
        answer(Result) :- Result is divide_by_parent(1).
        "#,
    );
    let body = &proper_rule(&expanded, "answer").body;

    assert_eq!(body.len(), 4, "arithmetic binding moved: {body:?}");
    assert!(matches!(
        &body[0],
        BodyLiteral::Positive(atom) if atom.predicate == "parent"
    ));
    assert!(matches!(
        &body[1],
        BodyLiteral::IsExpr(binding)
            if binding.target.starts_with("__XLOG_FUNCTION_DIVIDE_BY_PARENT_Parent_")
                && matches!(binding.expr, ArithExpr::Variable(_))
    ));
    assert!(matches!(
        &body[2],
        BodyLiteral::IsExpr(binding)
            if binding.target.starts_with("__XLOG_FUNCTION_DIVIDE_BY_PARENT_Result_")
                && matches!(binding.expr, ArithExpr::Div(_, _))
    ));
    assert!(matches!(
        &body[3],
        BodyLiteral::IsExpr(binding)
            if binding.target == "Result"
                && matches!(binding.expr, ArithExpr::Variable(_))
    ));
}

#[test]
fn trailing_alias_does_not_move_a_read_before_its_binding() {
    let source = r#"
        func read(Value) = Result :- Result is Value.
        func bind(Output) = Result :- Output is 1, Result is Output.
        answer(Result) :- Result is read(Value) + bind(Value).
    "#;
    let expanded = expand(source, 100).expect("expand ordered predicate bindings");

    let error = Compiler::new()
        .compile_program(&expanded)
        .expect_err("read before binding must remain invalid");

    assert!(
        error
            .to_string()
            .contains("Variable Value used in arithmetic but not bound"),
        "{error}"
    );
}

#[test]
fn arithmetic_function_can_wrap_a_predicate_function() {
    let expanded = compile_expanded(
        r#"
        pred parent(u32, u32).
        pred answer(u32).

        func get_parent(Child) = Parent :- parent(Child, Parent).
        func parent_plus_one(Child) = get_parent(Child) + cast(1, u32).

        parent(1, 2).
        answer(Result) :- Result is parent_plus_one(1).
        "#,
    );
    let body = &proper_rule(&expanded, "answer").body;

    assert!(matches!(
        &body[0],
        BodyLiteral::Positive(atom) if atom.predicate == "parent"
    ));
    assert!(matches!(
        &body[1],
        BodyLiteral::IsExpr(binding)
            if matches!(&binding.expr, ArithExpr::Add(left, right)
                if matches!(left.as_ref(), ArithExpr::Variable(_))
                    && matches!(right.as_ref(), ArithExpr::Cast(_, _)))
    ));
}

#[test]
fn predicate_function_substitution_reaches_nested_terms() {
    let expanded = expand(
        r#"
        func lookup(Value) = Result :- boxed(node(Value), Result).
        output(Result) :- Result is lookup(7).
        "#,
        100,
    )
    .expect("expand nested relational term");
    let body = &proper_rule(&expanded, "output").body;
    let BodyLiteral::Positive(atom) = &body[0] else {
        panic!("expected inserted boxed literal")
    };

    assert!(matches!(
        &atom.terms[0],
        Term::Compound { functor, args }
            if functor == "node" && matches!(args.as_slice(), [Term::Integer(7)])
    ));
}

#[test]
fn predicate_function_substitutes_deep_compound_terms_without_stack_growth() {
    const TERM_DEPTH: usize = 999;
    let mut program = parse_program(
        r#"
        func lookup(Value) = Result :- wrapped(Value, Result).
        output(Result) :- Result is lookup(7).
        "#,
    )
    .expect("parse predicate function");
    let FuncBody::Predicate { body, .. } = &mut program.functions[0].body else {
        panic!("expected predicate function body")
    };
    let BodyLiteral::Positive(atom) = &mut body[0] else {
        panic!("expected relational predicate body")
    };
    let mut nested = Term::Variable("Value".to_string());
    for _ in 0..TERM_DEPTH {
        nested = Term::Compound {
            functor: "node".to_string(),
            args: vec![nested],
        };
    }
    atom.terms[0] = nested;

    let expanded = expand_program_functions(&program, 100).expect("substitute deep compound term");
    let BodyLiteral::Positive(atom) = &proper_rule(&expanded, "output").body[0] else {
        panic!("expected inserted relational literal")
    };
    let mut term = &atom.terms[0];
    for _ in 0..TERM_DEPTH {
        let Term::Compound { functor, args } = term else {
            panic!("expected nested compound term")
        };
        assert_eq!(functor, "node");
        term = &args[0];
    }
    assert!(matches!(term, Term::Integer(7)));
}

#[test]
fn parameter_result_alias_preserves_the_call_argument() {
    let expanded = compile_expanded(
        r#"
        pred allowed(i64).
        pred answer(i64).
        func accepted(Value) = Value :- allowed(Value).
        allowed(7).
        answer(Result) :- Result is accepted(7).
        "#,
    );
    let body = &proper_rule(&expanded, "answer").body;

    assert!(matches!(
        &body[0],
        BodyLiteral::Positive(atom)
            if matches!(atom.terms.as_slice(), [Term::Integer(7)])
    ));
    assert!(matches!(
        &body[1],
        BodyLiteral::IsExpr(binding)
            if binding.target == "Result"
                && matches!(binding.expr, ArithExpr::Integer(7))
    ));
}

#[test]
fn complex_predicate_argument_in_a_relational_term_is_rejected() {
    let error = expand(
        r#"
        func get_parent(Child) = Parent :- parent(Child, Parent).
        answer(Parent) :- Parent is get_parent(1 + 1).
        "#,
        100,
    )
    .expect_err("complex relational argument must be rejected");

    assert!(
        error.to_string().contains(
            "cannot use an arithmetic expression argument for parameter `Child` in a term position"
        ),
        "{error}"
    );
    assert!(error.to_string().starts_with("error[E0510]:"), "{error}");
}

#[test]
fn predicate_call_in_a_conditional_branch_is_rejected() {
    let error = expand(
        r#"
        func get_parent(Child) = Parent :- parent(Child, Parent).
        func choose_parent(Child) =
            if Child > 0 then get_parent(Child) else 0.
        answer(Result) :- Result is choose_parent(1).
        "#,
        100,
    )
    .expect_err("conditional predicate call must be rejected");

    assert!(
        error
            .to_string()
            .contains("cannot be expanded inside a conditional branch"),
        "{error}"
    );
    assert!(error.to_string().starts_with("error[E0511]:"), "{error}");
}

#[test]
fn non_variable_predicate_binding_target_is_rejected() {
    let error = expand(
        r#"
        func bind_parameter(Value) = Result :- Value is 1, result(Result).
        answer(Result) :- Result is bind_parameter(5).
        "#,
        100,
    )
    .expect_err("constant predicate binding target must be rejected");

    assert!(
        error
            .to_string()
            .contains("cannot substitute a non-variable argument for binding target `Value`"),
        "{error}"
    );
    assert!(error.to_string().starts_with("error[E0512]:"), "{error}");
}

#[test]
fn predicate_function_recursion_obeys_the_expansion_limit() {
    let error = expand(
        r#"
        func repeat(Value) = Result :- Result is repeat(Value).
        answer(Result) :- Result is repeat(1).
        "#,
        3,
    )
    .expect_err("recursive predicate function must hit expansion limit");

    assert!(matches!(
        error,
        FunctionError::MaxRecursionDepth { name, depth }
            if name == "repeat" && depth == 3
    ));
}

#[test]
fn indirect_predicate_function_recursion_obeys_the_expansion_limit() {
    let error = expand(
        r#"
        func first(Value) = Result :- Result is second(Value).
        func second(Value) = Result :- Result is first(Value).
        answer(Result) :- Result is first(1).
        "#,
        3,
    )
    .expect_err("indirect predicate recursion must hit expansion limit");

    assert!(matches!(
        error,
        FunctionError::MaxRecursionDepth { name, depth }
            if name == "second" && depth == 3
    ));
}

#[test]
fn conditional_recursion_still_obeys_the_expansion_limit() {
    let error = expand(
        r#"
        func repeat(Value) = if Value = 0 then 0 else repeat(Value).
        answer(Result) :- Result is repeat(1).
        "#,
        3,
    )
    .expect_err("eager conditional recursion must hit expansion limit");

    assert!(matches!(
        &error,
        FunctionError::MaxRecursionDepth { name, depth }
            if name == "repeat" && *depth == 3
    ));
    assert!(error.to_string().starts_with("error[E0504]:"), "{error}");
}

#[test]
fn indirect_conditional_recursion_reports_the_call_at_the_configured_limit() {
    let error = expand(
        r#"
        func first(Value) = if Value = 0 then 0 else second(Value).
        func second(Value) = first(Value).
        answer(Result) :- Result is first(1).
        "#,
        3,
    )
    .expect_err("eager indirect recursion must hit the expansion limit");

    assert!(matches!(
        error,
        FunctionError::MaxRecursionDepth { name, depth }
            if name == "second" && depth == 3
    ));
}

#[test]
fn expansion_context_is_reusable_after_a_cycle_error() {
    let program = parse_program(
        r#"
        func repeat(Value) = if Value = 0 then 0 else repeat(Value).
        func identity(Value) = Value.
        "#,
    )
    .expect("parse functions");
    let registry = FunctionRegistry::from_program(&program).expect("validate functions");
    let mut context = ExpansionContext::new(&registry, 1000);

    let error = context
        .expand_call("repeat", &[ArithExpr::Integer(1)])
        .expect_err("cycle must fail");
    assert!(matches!(error, FunctionError::MaxRecursionDepth { .. }));
    assert_eq!(
        context
            .expand_call("identity", &[ArithExpr::Integer(7)])
            .expect("context state must be restored"),
        ArithExpr::Integer(7)
    );
}

#[test]
fn forwarding_chain_at_the_configured_limit_compiles_without_stack_growth() {
    compile_acyclic_function_chain(|index| format!("chain_{}(X)", index - 1));
}

#[test]
fn non_tail_chain_at_the_configured_limit_compiles_without_stack_growth() {
    compile_acyclic_function_chain(|index| format!("chain_{}(X) + 1", index - 1));
}

#[test]
fn nested_argument_chain_at_the_configured_limit_compiles_without_stack_growth() {
    compile_acyclic_function_chain(|index| format!("chain_{}(X + 1)", index - 1));
}

#[test]
fn conditional_chain_at_the_configured_limit_compiles_without_stack_growth() {
    compile_acyclic_function_chain(|index| format!("if X = 0 then X else chain_{}(X)", index - 1));
}

#[test]
fn predicate_function_chain_at_the_configured_limit_compiles_without_stack_growth() {
    const CHILD_ENV: &str = "XLOG_PREDICATE_FUNCTION_STACK_CHILD";
    if std::env::var_os(CHILD_ENV).is_some() {
        let source = predicate_function_chain_source(|index| format!("lookup{}(X)", index - 1));
        let program = parse_program(&source).expect("parse predicate function chain");
        let expanded =
            expand_program_functions(&program, 1000).expect("expand predicate function chain");
        let mut compiler = Compiler::new();
        let plan = compiler
            .compile_program(&expanded)
            .expect("compile predicate function chain");
        drop(plan);
        drop(compiler);
        drop(expanded);
        drop(program);
        return;
    }

    let output = std::process::Command::new(
        std::env::current_exe().expect("resolve predicate-function test binary"),
    )
    .args([
        "--exact",
        "predicate_function_chain_at_the_configured_limit_compiles_without_stack_growth",
        "--nocapture",
    ])
    .env(CHILD_ENV, "1")
    .output()
    .expect("run predicate-function stack child");
    assert!(
        output.status.success(),
        "predicate-function compile child failed with {status}:\n{stderr}",
        status = output.status,
        stderr = String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn predicate_function_arithmetic_chain_at_the_configured_limit_compiles_and_drops_without_stack_growth(
) {
    const CHILD_ENV: &str = "XLOG_PREDICATE_FUNCTION_ARITHMETIC_STACK_CHILD";
    if std::env::var_os(CHILD_ENV).is_some() {
        let source = predicate_function_chain_source(|index| {
            format!("lookup{}(X) + cast(0, u32)", index - 1)
        });
        let program = parse_program(&source).expect("parse predicate function arithmetic chain");
        let expanded = expand_program_functions(&program, 1000)
            .expect("expand predicate function arithmetic chain");
        let mut compiler = Compiler::new();
        let plan = compiler
            .compile_program(&expanded)
            .expect("compile predicate function arithmetic chain");
        drop(plan);
        drop(compiler);
        drop(expanded);
        drop(program);
        return;
    }

    let output = std::process::Command::new(
        std::env::current_exe().expect("resolve predicate-function test binary"),
    )
    .args([
        "--exact",
        "predicate_function_arithmetic_chain_at_the_configured_limit_compiles_and_drops_without_stack_growth",
        "--nocapture",
    ])
    .env(CHILD_ENV, "1")
    .output()
    .expect("run predicate-function arithmetic stack child");
    assert!(
        output.status.success(),
        "predicate-function arithmetic compile child failed with {status}:\n{stderr}",
        status = output.status,
        stderr = String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn predicate_function_arity_mismatch_is_reported() {
    let error = expand(
        r#"
        func get_parent(Child) = Parent :- parent(Child, Parent).
        answer(Result) :- Result is get_parent(1, 2).
        "#,
        100,
    )
    .expect_err("arity mismatch must be reported");

    assert!(
        error
            .to_string()
            .contains("function `get_parent` expects 1 argument but received 2"),
        "{error}"
    );
    assert!(error.to_string().starts_with("error[E0508]:"), "{error}");
}

#[test]
fn arity_is_checked_before_expanding_function_arguments() {
    let error = expand(
        r#"
        func unary(Value) = Value.
        func repeat(Value) = repeat(Value).
        answer(Result) :- Result is unary(repeat(1), 2).
        "#,
        3,
    )
    .expect_err("outer arity mismatch must precede recursive argument expansion");

    assert!(
        matches!(
            &error,
            FunctionError::ArityMismatch {
                name,
                expected: 1,
                received: 2,
            } if name == "unary"
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
fn expression_only_api_rejects_predicate_bodies_with_e0509() {
    let program = parse_program(
        r#"
        func get_parent(Child) = Parent :- parent(Child, Parent).
        "#,
    )
    .expect("parse predicate function");
    let mut registry = FunctionRegistry::new();
    registry
        .register(program.functions[0].clone())
        .expect("register predicate function");
    let mut context = ExpansionContext::new(&registry, 100);
    let error = context
        .expand_call("get_parent", &[ArithExpr::Integer(1)])
        .expect_err("expression-only expansion must reject relational literals");

    assert!(
        matches!(&error, FunctionError::PredicateBodyRequiresRuleContext { name } if name == "get_parent"),
        "unexpected error: {error:?}"
    );
    assert!(error.to_string().starts_with("error[E0509]:"), "{error}");
}

#[test]
fn unused_invalid_function_definitions_do_not_block_expansion() {
    let source = r#"
        pred conflict(u32).
        func first(Value) = missing_first(Value).
        func second(Value) = missing_second(Value).
        func repeat(Value) = repeat(Value).
        func conflict(Value) = Value.
        answer(1).
    "#;

    expand(source, 100).expect("unused invalid definitions are validated only by the registry API");
}

#[test]
fn used_undefined_function_reports_e0503() {
    let error = expand(
        r#"
        func lookup(Value) = missing(Value).
        answer(Result) :- Result is lookup(1).
        "#,
        100,
    )
    .expect_err("a called undefined function must fail expansion");

    assert!(
        matches!(&error, FunctionError::UndefinedFunction { name } if name == "missing"),
        "unexpected error: {error:?}"
    );
    assert!(error.to_string().starts_with("error[E0503]:"), "{error}");
}

#[test]
fn predicate_function_body_preserves_safe_negation_order() {
    compile_expanded(
        r#"
        pred candidate(u32, u32).
        pred blocked(u32).
        pred answer(u32).

        func allowed_candidate(Key) = Value :-
            candidate(Key, Value), not blocked(Value).

        candidate(1, 2).
        answer(Value) :- Value is allowed_candidate(1).
        "#,
    );
}

#[test]
fn predicate_functions_expand_inside_constraints() {
    let expanded = compile_expanded(
        r#"
        pred parent(u32, u32).
        func get_parent(Child) = Parent :- parent(Child, Parent).
        parent(1, 2).
        :- Parent is get_parent(1).
        "#,
    );

    assert_eq!(expanded.constraints[0].body.len(), 2);
    assert!(matches!(
        &expanded.constraints[0].body[0],
        BodyLiteral::Positive(atom) if atom.predicate == "parent"
    ));
}

#[test]
fn imported_predicate_functions_expand_after_module_merge() {
    let fixture = TempDir::new().expect("create module fixture");
    std::fs::write(
        fixture.path().join("family.xlog"),
        "pred parent(u32, u32).\n\
         parent(1, 2).\n\
         func get_parent(Child) = Parent :- parent(Child, Parent).\n",
    )
    .expect("write family module");

    let mut resolver = ModuleResolver::new(vec![]);
    resolver
        .load_module(fixture.path(), &["family".to_string()])
        .expect("load family module");
    let entry = parse_program(
        "use family.\n\
         pred answer(u32).\n\
         answer(Result) :- Result is get_parent(1).\n\
         ?- answer(Result).\n",
    )
    .expect("parse entry module");
    let merged = resolver.merge_imports(entry).expect("merge family module");
    let expanded = expand_program_functions(&merged, 100).expect("expand imported function");
    let mut compiler = Compiler::new();

    compiler
        .compile_program(&expanded)
        .expect("compile imported predicate function");
}

#[test]
fn predicate_function_substitution_preserves_modal_literals() {
    let expanded = expand(
        r#"
        func possible_parent(Child) = Parent :- possible parent(Child, Parent).
        answer(Parent) :- Parent is possible_parent(1).
        "#,
        100,
    )
    .expect("expand modal predicate body");
    let body = &proper_rule(&expanded, "answer").body;

    assert!(matches!(
        &body[0],
        BodyLiteral::Epistemic(modal)
            if modal.atom.predicate == "parent"
                && matches!(modal.atom.terms.as_slice(), [Term::Integer(1), Term::Variable(_)])
    ));
    assert!(matches!(&body[1], BodyLiteral::IsExpr(_)));

    let eir = build_eir(&expanded).expect("classify expanded modal body in EIR");
    let answer = eir
        .rules
        .iter()
        .find(|rule| rule.head.predicate == "answer")
        .expect("find answer EIR rule");
    assert!(matches!(
        &answer.body[0],
        EirBodyLiteral::Epistemic(modal)
            if modal.atom.predicate == "parent"
                && matches!(modal.atom.terms.as_slice(), [EirTerm::Integer(1), EirTerm::Variable(_)])
    ));
}

#[test]
fn predicate_function_expansion_preserves_an_aggregate_headed_caller() {
    let expanded = compile_expanded(
        r#"
        pred child(u32).
        pred parent(u32, u32).
        pred parent_count(u32, u64).
        func get_parent(Child) = Parent :- parent(Child, Parent).
        child(1).
        parent(1, 2).
        parent_count(Child, count(Parent)) :-
            child(Child), Parent is get_parent(Child).
        "#,
    );
    let rule = proper_rule(&expanded, "parent_count");

    assert!(matches!(&rule.head.terms[1], Term::Aggregate(_)));
    assert!(matches!(
        &rule.body[1],
        BodyLiteral::Positive(atom) if atom.predicate == "parent"
    ));
    assert!(matches!(
        &rule.body[2],
        BodyLiteral::IsExpr(binding) if binding.target == "Parent"
    ));
}
