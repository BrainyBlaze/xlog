use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{apply_query_vars_device, restore_query_vars_device};

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
fn query_vars_apply_and_restore_on_device() {
    let Some(provider) = try_provider() else {
        return;
    };

    let host_log_false = vec![0.0f64, -1.0, -2.0, -3.0, -4.0];
    let var_cap = (host_log_false.len() - 1) as u32;

    let mut log_false = provider
        .memory()
        .alloc::<f64>(host_log_false.len())
        .unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&host_log_false, &mut log_false)
        .unwrap();

    let query_vars_host = vec![0u32, 2u32, 4u32];
    let mut query_vars = provider
        .memory()
        .alloc::<u32>(query_vars_host.len())
        .unwrap();
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&query_vars_host, &mut query_vars)
        .unwrap();

    let mut saved = provider
        .memory()
        .alloc::<f64>(query_vars_host.len())
        .unwrap();

    apply_query_vars_device(&provider, &query_vars, var_cap, &mut log_false, &mut saved)
        .expect("apply query vars");

    let mut log_false_host = vec![0.0f64; host_log_false.len()];
    let mut saved_host = vec![0.0f64; query_vars_host.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&log_false, &mut log_false_host)
        .unwrap();
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&saved, &mut saved_host)
        .unwrap();

    assert_eq!(saved_host[0], 0.0);
    assert_eq!(saved_host[1], host_log_false[2]);
    assert_eq!(saved_host[2], host_log_false[4]);
    assert_eq!(log_false_host[2], f64::NEG_INFINITY);
    assert_eq!(log_false_host[4], f64::NEG_INFINITY);

    restore_query_vars_device(&provider, &query_vars, var_cap, &mut log_false, &saved)
        .expect("restore query vars");

    let mut restored = vec![0.0f64; host_log_false.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&log_false, &mut restored)
        .unwrap();
    assert_eq!(restored, host_log_false);
}
