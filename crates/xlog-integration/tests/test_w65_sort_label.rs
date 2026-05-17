use xlog_core::{ScalarType, Schema};
use xlog_gpu::logic::LogicProgram;

#[test]
fn w65_schema_new_assigns_non_default_sort_labels() {
    let schema = Schema::new(vec![
        ("pred".to_string(), ScalarType::I64),
        ("arg0".to_string(), ScalarType::I64),
        ("arg1".to_string(), ScalarType::I64),
    ]);

    assert_eq!(schema.sort_labels(), ["pred", "arg0", "arg1"]);
    assert!(schema.has_authoritative_sort_labels());
}

#[test]
fn w65_query_output_sort_labels_follow_query_variables() {
    let source = r#"
        pred wmir_committed(i64, i64, i64).
        pred wmir_rule(i64).
        pred support_1(i64, i64, i64, i64, i64, i64, i64).
        pred usable(i64, i64, i64).

        support_1(Head, A0, A1, RId, W0P, W0A0, W0A1) :-
            wmir_rule(RId),
            wmir_committed(Head, A0, A1),
            wmir_committed(W0P, W0A0, W0A1).
        usable(Head, A0, A1) :- support_1(Head, A0, A1, _, _, _, _).

        ?- support_1(Head, A0, A1, RId, W0P, W0A0, W0A1).
        ?- usable(P, A0, A1).
    "#;

    let program = LogicProgram::compile(source).expect("compile W65 support fixture");
    let support_schema = program
        .schema("__xlog_query_0")
        .expect("support query schema");
    let usable_schema = program
        .schema("__xlog_query_1")
        .expect("usable query schema");

    assert_eq!(
        support_schema.sort_labels(),
        ["Head", "A0", "A1", "RId", "W0P", "W0A0", "W0A1"]
    );
    assert_eq!(usable_schema.sort_labels(), ["P", "A0", "A1"]);
    assert!(support_schema.has_authoritative_sort_labels());
    assert!(usable_schema.has_authoritative_sort_labels());
}

#[test]
fn w65_pyxlog_logic_query_result_exposes_sort_labels() {
    let lib_src = include_str!("../../pyxlog/src/lib.rs");
    let logic_src = include_str!("../../pyxlog/src/logic.rs");

    assert!(
        lib_src.contains("pub sort_labels: Vec<String>"),
        "LogicQueryResult must expose per-column sort labels"
    );
    assert!(
        logic_src.contains("sort_labels: q.sort_labels"),
        "pyxlog result packing must preserve xlog-gpu query sort labels"
    );
}
