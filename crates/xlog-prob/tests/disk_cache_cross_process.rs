//! GPU-gated cross-process determinism test for disk cache keys.
//!
//! Spawns two child processes that each build the same logical PIR graph
//! through HashMap-simulated non-deterministic interning, encode CNF,
//! upload to GPU, and compute hash_cnf_gpu (the cache key). The parent
//! asserts both processes produce the same hash.
//!
//! This verifies the invariant that makes cross-process disk cache hits
//! possible: identical XLOG programs produce identical cache keys regardless
//! of HashMap ordering differences between processes.
//!
//! When the disk cache infrastructure (Tasks 10-12) lands, this test should
//! be extended to assert that the second process reports a disk cache hit.

use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

use xlog_prob::cnf::encode_cnf;
use xlog_prob::pir::{LeafId, PirGraph, PirNodeId};

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::gpu_cache::hash_cnf_gpu;
use xlog_solve::{Clause, GpuCnf, Literal, SolveInstance};

/// Convert a CnfEncoding (DIMACS-style 1-indexed signed literals) to a
/// SolveInstance (0-indexed Literal structs) suitable for GpuCnf::from_host.
fn cnf_to_solve_instance(
    num_vars: u32,
    clauses: &[Vec<i32>],
) -> SolveInstance {
    let solve_clauses: Vec<Clause> = clauses
        .iter()
        .map(|clause| {
            let lits: Vec<Literal> = clause
                .iter()
                .map(|&lit| {
                    let var = (lit.unsigned_abs() - 1) as u32; // 1-indexed → 0-indexed
                    let negated = lit < 0;
                    Literal::new(var, negated)
                })
                .collect();
            Clause::new(lits)
        })
        .collect();
    SolveInstance::new(num_vars, solve_clauses)
}

/// Build a non-trivial PIR graph through HashMap-based interning and return
/// the GPU cache hash. Returns None if GPU is unavailable.
fn build_encode_and_hash_gpu() -> Option<u64> {
    // Initialize CUDA; skip if unavailable.
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(_) => return None,
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1 << 30),
    ));
    let provider = match CudaKernelProvider::new(device, memory) {
        Ok(p) => Arc::new(p),
        Err(_) => return None,
    };

    // Simulate provenance's HashMap-based interning.
    let mut intern: HashMap<String, ()> = HashMap::new();
    for i in 0..10u32 {
        intern.insert(format!("node_{}", i), ());
    }

    let mut pir = PirGraph::new();
    let mut leaf_by_id: HashMap<u32, PirNodeId> = HashMap::new();
    for (name, _) in &intern {
        let idx: u32 = name.strip_prefix("node_").unwrap().parse().unwrap();
        leaf_by_id.insert(idx, pir.lit(LeafId::new(idx)));
    }

    let left: Vec<PirNodeId> = (0..5).map(|i| leaf_by_id[&i]).collect();
    let right: Vec<PirNodeId> = (5..10).map(|i| leaf_by_id[&i]).collect();
    let and_left = pir.and(left);
    let and_right = pir.and(right);
    let root = pir.or(vec![and_left, and_right]);

    let enc = encode_cnf(&pir, &[root]).unwrap();
    let instance = cnf_to_solve_instance(enc.cnf.num_vars(), enc.cnf.clauses());
    let gpu_cnf = GpuCnf::from_host(&instance, &provider).unwrap();

    let hash_device = hash_cnf_gpu(&gpu_cnf, &provider).unwrap();
    let mut hash_host = vec![0u64; 1];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&hash_device, &mut hash_host)
        .unwrap();

    Some(hash_host[0])
}

/// Child-mode entry point. When XLOG_GPU_HASH_OUTPUT_PATH is set,
/// compute the GPU cache hash and write it to that file.
#[test]
fn disk_cache_cross_process_child() {
    let path = match std::env::var("XLOG_GPU_HASH_OUTPUT_PATH") {
        Ok(p) => p,
        Err(_) => return, // No-op when run directly
    };

    match build_encode_and_hash_gpu() {
        Some(hash) => std::fs::write(&path, format!("{hash}")).unwrap(),
        None => std::fs::write(&path, "NO_GPU").unwrap(),
    }
}

/// Spawn two separate processes that each build the same logical PIR graph
/// (through non-deterministic HashMap interning), encode CNF, and compute
/// the GPU cache hash. Assert both produce the same hash value.
#[test]
fn gpu_cnf_hash_is_identical_across_processes() {
    let exe = std::env::current_exe().unwrap();
    let tmp = std::env::temp_dir();
    let path_a = tmp.join("xlog_gpu_hash_cross_a.txt");
    let path_b = tmp.join("xlog_gpu_hash_cross_b.txt");

    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);

    let run = |label: &str, path: &std::path::Path| {
        let output = Command::new(&exe)
            .arg("--exact")
            .arg("disk_cache_cross_process_child")
            .env("XLOG_GPU_HASH_OUTPUT_PATH", path.to_str().unwrap())
            .output()
            .unwrap_or_else(|e| panic!("{label}: failed to spawn: {e}"));
        assert!(
            output.status.success(),
            "{label}: child process failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    };

    run("process_a", &path_a);
    run("process_b", &path_b);

    let a = std::fs::read_to_string(&path_a)
        .unwrap_or_else(|e| panic!("failed to read process_a output: {e}"));
    let b = std::fs::read_to_string(&path_b)
        .unwrap_or_else(|e| panic!("failed to read process_b output: {e}"));

    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);

    if a == "NO_GPU" || b == "NO_GPU" {
        eprintln!("Skipping test: CUDA runtime unavailable in child process");
        return;
    }

    assert!(!a.is_empty(), "process_a produced empty hash");
    assert_eq!(
        a, b,
        "GPU CNF hash (cache key) differs between processes: {a} vs {b}"
    );
}
