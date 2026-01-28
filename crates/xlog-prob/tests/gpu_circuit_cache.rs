use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::{
    GpuCacheLookup, GpuCircuitCache, GpuCircuitCacheConfig,
};

fn read_u32(provider: &Arc<CudaKernelProvider>, slice: &xlog_cuda::memory::TrackedCudaSlice<u32>) -> u32 {
    let mut host = vec![0u32; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(slice, &mut host)
        .expect("dtoh u32");
    host[0]
}

fn compile_needed_host(handle: &GpuCacheLookup) -> bool {
    let provider = handle.provider();
    read_u32(provider, handle.compile_needed_device()) != 0
}

#[test]
fn gpu_cache_hit_miss_and_eviction() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).expect("provider"));

    let config = GpuCircuitCacheConfig {
        num_slots: 2,
        table_size: 4,
        node_cap: 8,
        edge_cap: 16,
        level_cap: 8,
        var_cap: 8,
    };
    let mut cache = GpuCircuitCache::new(&provider, config).expect("cache");

    let k1 = 0x1111u64;
    let k2 = 0x2222u64;
    let k3 = 0x3333u64;

    let h1 = cache.lookup_or_insert(k1).expect("lookup k1");
    assert!(compile_needed_host(&h1));

    let h2 = cache.lookup_or_insert(k1).expect("lookup k1 again");
    assert!(!compile_needed_host(&h2));

    let h3 = cache.lookup_or_insert(k2).expect("lookup k2");
    assert!(compile_needed_host(&h3));

    let h4 = cache.lookup_or_insert(k3).expect("lookup k3");
    assert!(compile_needed_host(&h4));

    let h5 = cache.lookup_or_insert(k1).expect("lookup k1 post-evict");
    assert!(compile_needed_host(&h5));
}
