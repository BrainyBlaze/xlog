#[test]
fn runtime_clique_dispatch_consumes_kclique_plan_surface() {
    let source = include_str!("../src/executor/wcoj_dispatch.rs");
    let body = source
        .split("fn try_dispatch_wcoj_clique_k_on_body")
        .nth(1)
        .expect("try_dispatch_wcoj_clique_k_on_body present")
        .split("/// Number of times")
        .next()
        .expect("clique dispatch body before counters");

    assert!(body.contains("kclique"));
    assert!(body.contains("edge_permutation"));
    assert!(body.contains("column_swaps"));
    assert!(body.contains("sorted_layout_requirements"));
    assert!(
        body.matches("wcoj_layout_sort_").count() < 10,
        "K5 dispatch must not have the old unconditional all-10 layout-sort path"
    );
}

#[test]
fn provider_and_kernel_accept_plan_derived_launch_params() {
    let provider = include_str!("../../xlog-cuda/src/provider/wcoj.rs");
    let inner = provider
        .split("fn wcoj_clique_recorded_inner")
        .nth(1)
        .expect("wcoj_clique_recorded_inner present")
        .split("/// W3.2 — 5-clique")
        .next()
        .expect("inner body before public wrappers");

    assert!(inner.contains("leader_edge_idx"));
    assert!(inner.contains("edge_order"));
    assert!(inner.contains("iteration_order"));

    let kernel = include_str!("../../xlog-cuda/kernels/wcoj.cu");
    let body = kernel
        .split("// ===============================================================\n// W3.2")
        .nth(1)
        .expect("K-clique kernel section present");

    assert!(body.contains("leader_edge_idx"));
    assert!(body.contains("edge_order"));
    assert!(body.contains("iteration_order"));
    assert!(
        !body.contains("canonical"),
        "K-clique kernel body must not retain canonical-edge assumptions"
    );
    assert!(
        !body.contains("(0, 1)"),
        "K-clique kernel body must not retain fixed (0,1) leader text"
    );
}
