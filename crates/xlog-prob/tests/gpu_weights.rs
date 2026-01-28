use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_prob::compilation::gpu_cnf::GpuCnfVarTables;
use xlog_prob::compilation::gpu_weights::{build_evidence_by_var_gpu, build_weights_gpu};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::with_limit(1 << 30)));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: failed to create CUDA kernel provider: {}", e);
            None
        }
    }
}

fn upload_u32(provider: &Arc<CudaKernelProvider>, host: &[u32]) -> TrackedCudaSlice<u32> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<u32>(host.len()).expect("alloc u32");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(host, &mut buf)
        .expect("upload u32");
    buf
}

fn upload_f64(provider: &Arc<CudaKernelProvider>, host: &[f64]) -> TrackedCudaSlice<f64> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<f64>(host.len()).expect("alloc f64");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(host, &mut buf)
        .expect("upload f64");
    buf
}

fn upload_u8(provider: &Arc<CudaKernelProvider>, host: &[u8]) -> TrackedCudaSlice<u8> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<u8>(host.len()).expect("alloc u8");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(host, &mut buf)
        .expect("upload u8");
    buf
}

fn ln_prob(p: f64) -> f64 {
    if p == 0.0 {
        f64::NEG_INFINITY
    } else {
        p.ln()
    }
}

#[test]
fn gpu_weights_builds_log_tables_and_evidence() {
    let Some(provider) = try_provider() else {
        return;
    };

    // var_cap = 6 (DIMACS 1-based)
    let node_var = upload_u32(&provider, &[4, 0, 0]); // node 0 -> var 4
    let leaf_var = upload_u32(&provider, &[2, 5, 0]); // leaf 0 -> var 2, leaf 1 -> var 5
    let choice_var = upload_u32(&provider, &[3, 6]);  // choice 0 -> var 3, choice 1 -> var 6

    let vars = GpuCnfVarTables {
        node_var,
        leaf_var,
        choice_var,
        max_var: 6,
    };

    let leaf_probs = upload_f64(&provider, &[0.2, 0.7]);
    let choice_true = upload_f64(&provider, &[0.1, 0.6]);
    let choice_false = upload_f64(&provider, &[0.9, 0.4]);

    let evidence_nodes = upload_u32(&provider, &[0]);
    let evidence_vals = upload_u8(&provider, &[1]); // node 0 true -> var 4 true

    let evidence_by_var = build_evidence_by_var_gpu(
        &vars.node_var,
        &evidence_nodes,
        &evidence_vals,
        vars.max_var,
        &provider,
    )
    .expect("evidence map");

    let weights = build_weights_gpu(
        &vars,
        &leaf_probs,
        &choice_true,
        &choice_false,
        &evidence_by_var,
        &provider,
    )
    .expect("weights");

    let mut host_true = vec![0.0f64; (vars.max_var as usize) + 1];
    let mut host_false = vec![0.0f64; (vars.max_var as usize) + 1];

    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&weights.log_true, &mut host_true)
        .expect("read log_true");
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&weights.log_false, &mut host_false)
        .expect("read log_false");

    // leaf var 2
    assert!((host_true[2] - ln_prob(0.2)).abs() < 1e-9);
    assert!((host_false[2] - ln_prob(0.8)).abs() < 1e-9);
    // leaf var 5
    assert!((host_true[5] - ln_prob(0.7)).abs() < 1e-9);
    assert!((host_false[5] - ln_prob(0.3)).abs() < 1e-9);

    // choice var 3
    assert!((host_true[3] - ln_prob(0.1)).abs() < 1e-9);
    assert!((host_false[3] - ln_prob(0.9)).abs() < 1e-9);
    // choice var 6
    assert!((host_true[6] - ln_prob(0.6)).abs() < 1e-9);
    assert!((host_false[6] - ln_prob(0.4)).abs() < 1e-9);

    // evidence on var 4 (true) forces log_false to -inf
    assert!(host_false[4].is_infinite() && host_false[4].is_sign_negative());
}
