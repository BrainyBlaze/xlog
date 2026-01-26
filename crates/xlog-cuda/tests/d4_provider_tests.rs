use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(CudaKernelProvider::new(device, memory).unwrap())
}

#[test]
fn test_provider_loads_d4_module_entrypoints() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // The provider must load the D4 PTX under a stable module name so
    // downstream crates can resolve kernels by (module, function).
    let device = provider.device().inner();
    assert!(
        device.get_func("xlog_d4", "d4_validate_cnf").is_some(),
        "expected provider to load D4 module and expose d4_validate_cnf"
    );
    assert!(
        device.get_func("xlog_d4", "d4_levelize_emit").is_some(),
        "expected provider to load D4 module and expose d4_levelize_emit"
    );

    // Task 4: BFS frontier expansion kernels must be present.
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
