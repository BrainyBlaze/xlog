use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::CudaKernelProvider;
use xlog_prob::compilation::gpu_cache::{GpuCacheLookup, GpuCircuitCache, GpuCircuitCacheConfig};

fn read_u32(
    provider: &Arc<CudaKernelProvider>,
    slice: &xlog_cuda::memory::TrackedCudaSlice<u32>,
) -> u32 {
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
    let provider =
        match xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(1 << 30)).build() {
            Ok(provider) => Arc::new(provider),
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };

    let config = {
        let mut config = GpuCircuitCacheConfig::default();
        config.num_slots = 2;
        config.table_size = 4;
        config.node_cap = 8;
        config.edge_cap = 16;
        config.level_cap = 8;
        config.var_cap = 8;
        config
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
