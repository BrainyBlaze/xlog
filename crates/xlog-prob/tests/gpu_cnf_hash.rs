use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::hash_cnf_gpu;
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

fn cpu_hash_u64(vals: &[u64]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for &v in vals {
        h ^= v;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[test]
fn gpu_cnf_hash_matches_cpu_reference() {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1 << 30),
    ));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).expect("provider"));

    let clauses = vec![
        Clause::new(vec![Literal::new(1, true), Literal::new(2, false)]),
        Clause::new(vec![Literal::new(2, true)]),
    ];
    let instance = SolveInstance::new(2, clauses);
    let cnf = GpuCnf::from_host(&instance, &provider).expect("GpuCnf");

    let hashes = hash_cnf_gpu(&cnf, &provider).expect("hash");

    let mut host: Vec<u32> = Vec::new();
    let mut tmp = vec![0u32; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&cnf.num_vars, &mut tmp)
        .unwrap();
    host.push(tmp[0]);
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&cnf.num_clauses, &mut tmp)
        .unwrap();
    host.push(tmp[0]);
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&cnf.num_lits, &mut tmp)
        .unwrap();
    host.push(tmp[0]);

    let mut offsets = vec![0u32; cnf.clause_offsets.len()];
    let mut lits = vec![0i32; cnf.literals.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&cnf.clause_offsets, &mut offsets)
        .unwrap();
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&cnf.literals, &mut lits)
        .unwrap();

    for &v in &offsets {
        host.push(v);
    }
    for &v in &lits {
        host.push(v as u32);
    }

    let vals: Vec<u64> = host.iter().map(|&v| v as u64).collect();
    let cpu_hash = cpu_hash_u64(&vals);

    let mut gpu_hash = vec![0u64; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&hashes, &mut gpu_hash)
        .unwrap();

    assert_eq!(gpu_hash[0], cpu_hash);
}
