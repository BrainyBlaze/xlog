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
fn test_compact_device_mask_sets_device_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![1, 2, 3, 4, 5];
    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    // mask keeps odd indices
    let mask: Vec<u8> = vec![1, 0, 1, 0, 1];
    let mut d_mask = provider.memory().alloc::<u8>(mask.len()).unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&mask, &mut d_mask)
        .unwrap();

    let compacted = provider
        .compact_buffer_by_device_mask_counted(&buffer, &d_mask)
        .unwrap();

    assert_eq!(compacted.num_rows(), 5);

    let mut host_count = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(compacted.num_rows_device(), &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 3);
}
