mod common;
use common::setup_provider;

#[test]
fn test_provider_loads_decision_dnnf_module_entrypoints() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // The provider must load the Decision-DNNF PTX under a stable module name so
    // downstream crates can resolve kernels by (module, function).
    let device = provider.device().inner();
    assert!(
        device.get_func("xlog_d4", "d4_validate_cnf").is_some(),
        "expected provider to load Decision-DNNF module and expose d4_validate_cnf"
    );
    assert!(
        device.get_func("xlog_d4", "d4_levelize_emit").is_some(),
        "expected provider to load Decision-DNNF module and expose d4_levelize_emit"
    );

    // BFS frontier expansion kernels must be present.
    assert!(
        device.get_func("xlog_d4", "d4_frontier_prepare").is_some(),
        "expected provider to expose d4_frontier_prepare"
    );
    assert!(
        device.get_func("xlog_d4", "d4_frontier_expand").is_some(),
        "expected provider to expose d4_frontier_expand"
    );
    assert!(
        device
            .get_func("xlog_d4", "d4_frontier_prepare_dense")
            .is_some(),
        "expected provider to expose d4_frontier_prepare_dense"
    );
    assert!(
        device
            .get_func("xlog_d4", "d4_frontier_expand_dense")
            .is_some(),
        "expected provider to expose d4_frontier_expand_dense"
    );

    // GPU-only assertion helpers for tests/invariants without host reads.
    assert!(
        device.get_func("xlog_d4", "d4_assert_dense_var").is_some(),
        "expected provider to expose d4_assert_dense_var"
    );
}
