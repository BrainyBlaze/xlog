use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::cnf::encode_cnf;
use xlog_prob::compilation::{encode_cnf_gpu, GpuPirGraph, GpuPirRoots};
use xlog_prob::pir::{ChoiceVarId, LeafId, PirGraph};

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

fn canonicalize(clauses: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut out: Vec<Vec<i32>> = clauses
        .into_iter()
        .map(|mut c| {
            c.sort();
            c
        })
        .collect();
    out.sort();
    out
}

fn gpu_cnf_to_host(
    provider: &Arc<CudaKernelProvider>,
    cnf: &xlog_solve::GpuCnf,
) -> (u32, Vec<Vec<i32>>) {
    let device = provider.device().inner();
    let mut num_vars = [0u32; 1];
    let mut num_clauses = [0u32; 1];
    let mut num_lits = [0u32; 1];
    device.dtoh_sync_copy_into(&cnf.num_vars, &mut num_vars).unwrap();
    device
        .dtoh_sync_copy_into(&cnf.num_clauses, &mut num_clauses)
        .unwrap();
    device.dtoh_sync_copy_into(&cnf.num_lits, &mut num_lits).unwrap();

    let clauses_len = num_clauses[0] as usize;
    let lits_len = num_lits[0] as usize;
    let mut offsets = vec![0u32; clauses_len + 1];
    let mut lits = vec![0i32; lits_len];

    let offsets_view = cnf.clause_offsets.slice(0..(clauses_len + 1));
    let lits_view = cnf.literals.slice(0..lits_len);
    device.dtoh_sync_copy_into(&offsets_view, &mut offsets).unwrap();
    device.dtoh_sync_copy_into(&lits_view, &mut lits).unwrap();

    let mut clauses = Vec::with_capacity(clauses_len);
    for i in 0..clauses_len {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        clauses.push(lits[start..end].to_vec());
    }

    (num_vars[0], clauses)
}

#[test]
fn gpu_cnf_matches_cpu_encoding_simple() {
    let Some(provider) = try_provider() else {
        return;
    };

    let mut pir = PirGraph::new();
    let a = pir.lit(LeafId::new(0));
    let b = pir.neg_lit(LeafId::new(1));
    let and = pir.and(vec![a, b]);
    let t = pir.const_true();
    let f = pir.const_false();
    let dec = pir.decision(ChoiceVarId::new(0), f, t);
    let root = pir.or(vec![and, dec]);

    let cpu = encode_cnf(&pir, &[root]).unwrap();

    let gpu_pir = GpuPirGraph::from_host(&pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[root], &provider).unwrap();
    let gpu = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    let (gpu_vars, gpu_clauses) = gpu_cnf_to_host(&provider, &gpu.cnf);

    assert_eq!(gpu_vars, cpu.cnf.num_vars());
    assert_eq!(
        canonicalize(gpu_clauses),
        canonicalize(cpu.cnf.clauses().to_vec())
    );
}

#[test]
fn gpu_cnf_prunes_unreachable_nodes() {
    let Some(provider) = try_provider() else {
        return;
    };

    let mut pir = PirGraph::new();
    let a = pir.lit(LeafId::new(0));
    let b = pir.lit(LeafId::new(1));
    let r1 = pir.and(vec![a]);
    let _r2 = pir.or(vec![b]);

    let cpu = encode_cnf(&pir, &[r1]).unwrap();

    let gpu_pir = GpuPirGraph::from_host(&pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[r1], &provider).unwrap();
    let gpu = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    let (gpu_vars, gpu_clauses) = gpu_cnf_to_host(&provider, &gpu.cnf);
    assert_eq!(gpu_vars, cpu.cnf.num_vars());
    assert_eq!(
        canonicalize(gpu_clauses),
        canonicalize(cpu.cnf.clauses().to_vec())
    );
}
