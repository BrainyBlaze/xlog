//! Tests for DLPack export (zero-copy interop)

use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{dlpack, CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_export_u32_column_to_dlpack() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![1, 2, 3, 4, 5];

    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    let table = provider.to_dlpack_table(buffer);
    let tensor = table.column(0).unwrap();

    let ptr = tensor.as_ptr();
    assert!(!ptr.is_null());

    // SAFETY: ptr is owned by DlpackManagedTensor for the duration of this test.
    let managed = unsafe { &*ptr };
    assert_eq!(managed.dl_tensor.device.device_type, dlpack::K_DLCUDA);
    assert_eq!(managed.dl_tensor.device.device_id, 0);
    assert_eq!(managed.dl_tensor.ndim, 1);
    assert!(!managed.dl_tensor.shape.is_null());

    // SAFETY: shape points to a 1-element array allocated in DlpackCtx.
    let shape0 = unsafe { *managed.dl_tensor.shape };
    assert_eq!(shape0, 5);

    assert_eq!(managed.dl_tensor.dtype.code, dlpack::K_DLUINT);
    assert_eq!(managed.dl_tensor.dtype.bits, 32);
    assert_eq!(managed.dl_tensor.dtype.lanes, 1);
    assert_eq!(managed.dl_tensor.byte_offset, 0);
    assert!(!managed.dl_tensor.data.is_null());
}

