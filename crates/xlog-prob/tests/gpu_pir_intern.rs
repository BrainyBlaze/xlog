use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{GpuPirInterner, PirBatch};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: failed to create provider: {}", e);
            None
        }
    }
}

#[test]
fn gpu_pir_intern_is_deterministic() {
    let Some(provider) = try_provider() else {
        return;
    };

    let mut interner = GpuPirInterner::new(&provider, 1024, 4096).unwrap();
    let batch = PirBatch::and_or_batch(vec![vec![0, 1], vec![1, 0]]);

    let ids_a = interner.intern_batch(&batch).unwrap();
    let ids_b = interner.intern_batch(&batch).unwrap();

    let mut a = vec![0u32; ids_a.len()];
    let mut b = vec![0u32; ids_b.len()];
    let device = provider.device().inner();
    device.dtoh_sync_copy_into(&ids_a, &mut a).unwrap();
    device.dtoh_sync_copy_into(&ids_b, &mut b).unwrap();

    assert_eq!(a, b);
    assert_eq!(a[0], a[1]);
}
