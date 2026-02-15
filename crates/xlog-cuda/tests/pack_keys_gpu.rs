use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_pack_keys_gpu_generic_no_dtoh() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("c".to_string(), ScalarType::U32),
        ("d".to_string(), ScalarType::U32),
        ("e".to_string(), ScalarType::U32),
    ]);

    let a = vec![1u32, 2, 3];
    let b = vec![4u32, 5, 6];
    let c = vec![7u32, 8, 9];
    let d = vec![10u32, 11, 12];
    let e = vec![13u32, 14, 15];

    let buffer = provider
        .create_buffer_from_u32_columns(&[&a, &b, &c, &d, &e], schema)
        .unwrap();

    provider.reset_host_transfer_stats();

    let index = provider
        .build_join_index_v2(&buffer, &[0, 1, 2, 3, 4])
        .unwrap();
    assert!(index.estimated_bytes() > 0);

    let stats = provider.host_transfer_stats();
    assert_eq!(
        stats.dtoh_bytes, 0,
        "unexpected device-to-host transfers during GPU key packing: {} bytes",
        stats.dtoh_bytes
    );
}
