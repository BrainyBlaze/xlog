use tempfile::TempDir;
use xlog_ir::eir::{EirBodyLiteral, EirTerm};
use xlog_logic::ast::{ArithExpr, BodyLiteral, Program, Rule, Term};
use xlog_logic::function::FunctionError;
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

    assert_eq!(body.len(), 4);
    assert!(
        matches!(&body[0], BodyLiteral::Positive(atom) if matches!(atom.terms[0], Term::Integer(1)))
    );
    assert!(matches!(&body[1], BodyLiteral::Positive(atom) if atom.predicate == "parent"));
    assert!(matches!(&body[2], BodyLiteral::IsExpr(_)));
    assert!(matches!(&body[3], BodyLiteral::IsExpr(binding) if binding.target == "Result"));
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
            "cannot substitute an arithmetic expression for relational parameter `Child`"
        ),
        "{error}"
    );
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
