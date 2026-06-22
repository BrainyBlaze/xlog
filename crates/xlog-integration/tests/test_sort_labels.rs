#![allow(clippy::arc_with_non_send_sync)]

use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_gpu::logic::LogicProgram;

fn sort_label_test_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
}

#[test]
fn schema_new_assigns_non_default_sort_labels() {
    let schema = Schema::new(vec![
        ("pred".to_string(), ScalarType::I64),
        ("arg0".to_string(), ScalarType::I64),
        ("arg1".to_string(), ScalarType::I64),
    ]);

    assert_eq!(schema.sort_labels(), ["pred", "arg0", "arg1"]);
    assert!(schema.has_authoritative_sort_labels());
}

#[test]
fn query_output_sort_labels_follow_query_variables() {
    let source = r#"
        pred external_consumer_commit(i64, i64, i64).
        pred external_consumer_rule(i64).
        pred support_1(i64, i64, i64, i64, i64, i64, i64).
        pred usable(i64, i64, i64).

        support_1(Head, A0, A1, RId, Body0Pred, Body0Arg0, Body0Arg1) :-
            external_consumer_rule(RId),
            external_consumer_commit(Head, A0, A1),
            external_consumer_commit(Body0Pred, Body0Arg0, Body0Arg1).
        usable(Head, A0, A1) :- support_1(Head, A0, A1, _, _, _, _).

        ?- support_1(Head, A0, A1, RId, Body0Pred, Body0Arg0, Body0Arg1).
        ?- usable(P, A0, A1).
    "#;

    let program = LogicProgram::compile(source).expect("compile support sort-label fixture");
    let support_schema = program
        .schema("__xlog_query_0")
        .expect("support query schema");
    let usable_schema = program
        .schema("__xlog_query_1")
        .expect("usable query schema");

    assert_eq!(
        support_schema.sort_labels(),
        [
            "Head",
            "A0",
            "A1",
            "RId",
            "Body0Pred",
            "Body0Arg0",
            "Body0Arg1"
        ]
    );
    assert_eq!(usable_schema.sort_labels(), ["P", "A0", "A1"]);
    assert!(support_schema.has_authoritative_sort_labels());
    assert!(usable_schema.has_authoritative_sort_labels());
}

#[test]
fn runtime_query_result_sort_labels_follow_query_variables() {
    let Some(provider) = sort_label_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let source = r#"
        pred edge(u32, u32).
        pred reach(u32, u32).

        edge(1, 2).
        edge(2, 3).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).

        ?- reach(Source, Target).
    "#;

    let program = LogicProgram::compile(source).expect("compile recursive sort-label fixture");
    let result = program
        .evaluate(provider, std::collections::HashMap::new())
        .expect("evaluate recursive sort-label fixture");

    assert_eq!(result.queries.len(), 1);
    assert_eq!(result.queries[0].columns, ["Source", "Target"]);
    assert_eq!(result.queries[0].sort_labels, ["Source", "Target"]);
}

#[test]
fn external_consumer_style_support_source_emits_partial_unary_rows_by_semantics() {
    let Some(provider) = sort_label_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let source = r#"
        pred external_consumer_commit(i64, i64, i64).
        pred external_consumer_body_0(i64, i64, i64).
        pred external_consumer_body_1(i64, i64, i64).
        pred usable(i64, i64, i64).
        pred support_1(i64, i64, i64, i64, i64, i64, i64).
        pred support_2(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).

        external_consumer_body_0(4, 10006, 10012).
        external_consumer_body_1(4, 10006, 10012).
        external_consumer_commit(10012, 10022, 10022).

        usable(P, A0, A1) :- external_consumer_commit(P, A0, A1).

        support_1(Head, V0, V1, RId, Body0Pred, V0, V1) :-
            external_consumer_body_0(RId, Head, Body0Pred), usable(Body0Pred, V0, V1).

        support_2(Head, V0, V1, RId, Body0Pred, V0, V2, Body1Pred, V2, V1) :-
            external_consumer_body_0(RId, Head, Body0Pred),
            usable(Body0Pred, V0, V2),
            external_consumer_body_1(RId, Head, Body1Pred),
            usable(Body1Pred, V2, V1).

        ?- support_1(H, A0, A1, R, Body0Pred, Body0Arg0, Body0Arg1).
        ?- support_2(H, A0, A1, R, Body0Pred, Body0Arg0, Body0Arg1, Body1Pred, Body1Arg0, Body1Arg1).
    "#;

    let program =
        LogicProgram::compile(source).expect("compile external consumer-style sort-label fixture");
    let result = program
        .evaluate(provider, std::collections::HashMap::new())
        .expect("evaluate external consumer-style sort-label fixture");

    assert_eq!(result.queries.len(), 2);
    assert_eq!(
        result.queries[0].columns,
        ["H", "A0", "A1", "R", "Body0Pred", "Body0Arg0", "Body0Arg1"]
    );
    assert_eq!(
        result.queries[0].sort_labels,
        ["H", "A0", "A1", "R", "Body0Pred", "Body0Arg0", "Body0Arg1"]
    );
    assert_eq!(
        result.queries[0].buffer.num_rows(),
        1,
        "Datalog semantics require support_1 to emit the partial row because the source asks for external_consumer_body_0-only support"
    );
    assert_eq!(result.queries[1].buffer.num_rows(), 1);
}
