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
fn test_device_row_count_tracks_host_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    let mut host_count = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 4);
}
