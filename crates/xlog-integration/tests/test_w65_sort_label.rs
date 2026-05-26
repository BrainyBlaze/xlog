#![allow(clippy::arc_with_non_send_sync)]

use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_gpu::logic::LogicProgram;

fn w65_test_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
}

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
fn w65_runtime_query_result_sort_labels_follow_query_variables() {
    let Some(provider) = w65_test_provider() else {
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

    let program = LogicProgram::compile(source).expect("compile recursive W65 fixture");
    let result = program
        .evaluate(provider, std::collections::HashMap::new())
        .expect("evaluate recursive W65 fixture");

    assert_eq!(result.queries.len(), 1);
    assert_eq!(result.queries[0].columns, ["Source", "Target"]);
    assert_eq!(result.queries[0].sort_labels, ["Source", "Target"]);
}

#[test]
fn w65_dts_style_support_source_emits_partial_unary_rows_by_semantics() {
    let Some(provider) = w65_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let source = r#"
        pred wmir_committed(i64, i64, i64).
        pred wmir_body_0(i64, i64, i64).
        pred wmir_body_1(i64, i64, i64).
        pred usable(i64, i64, i64).
        pred support_1(i64, i64, i64, i64, i64, i64, i64).
        pred support_2(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).

        wmir_body_0(4, 10006, 10012).
        wmir_body_1(4, 10006, 10012).
        wmir_committed(10012, 10022, 10022).

        usable(P, A0, A1) :- wmir_committed(P, A0, A1).

        support_1(Head, V0, V1, RId, W0P, V0, V1) :-
            wmir_body_0(RId, Head, W0P), usable(W0P, V0, V1).

        support_2(Head, V0, V1, RId, W0P, V0, V2, W1P, V2, V1) :-
            wmir_body_0(RId, Head, W0P),
            usable(W0P, V0, V2),
            wmir_body_1(RId, Head, W1P),
            usable(W1P, V2, V1).

        ?- support_1(H, A0, A1, R, W0P, W0A0, W0A1).
        ?- support_2(H, A0, A1, R, W0P, W0A0, W0A1, W1P, W1A0, W1A1).
    "#;

    let program = LogicProgram::compile(source).expect("compile DTS-style W65 fixture");
    let result = program
        .evaluate(provider, std::collections::HashMap::new())
        .expect("evaluate DTS-style W65 fixture");

    assert_eq!(result.queries.len(), 2);
    assert_eq!(
        result.queries[0].columns,
        ["H", "A0", "A1", "R", "W0P", "W0A0", "W0A1"]
    );
    assert_eq!(
        result.queries[0].sort_labels,
        ["H", "A0", "A1", "R", "W0P", "W0A0", "W0A1"]
    );
    assert_eq!(
        result.queries[0].buffer.num_rows(),
        1,
        "Datalog semantics require support_1 to emit the partial row because the source asks for wmir_body_0-only support"
    );
    assert_eq!(result.queries[1].buffer.num_rows(), 1);
}
