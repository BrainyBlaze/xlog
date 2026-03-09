#![cfg(feature = "host-io")]

use cudarc::driver::DeviceSlice;
use std::sync::Arc;
use xlog_core::{MemoryBudget, Result};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::mc::{McEvalConfig, McProgram};

fn setup_provider() -> Option<Arc<CudaKernelProvider>> {
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
    CudaKernelProvider::new(device, memory).ok().map(Arc::new)
}

#[test]
fn test_mc_device_counts_match_cpu() -> Result<()> {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let program = McProgram::compile_source(
        r#"
        0.5::a().
        query(a()).
    "#,
    )?;

    let cfg = McEvalConfig {
        samples: 16,
        seed: 123,
        confidence: 0.95,
        max_nonmonotone_iterations: 10,
        sampling_method: None,
    };

    let gpu_host = program.evaluate_gpu(cfg.clone())?;
    let gpu = program.evaluate_gpu_device(cfg)?;

    let mut host_counts = vec![0u32; gpu.query_counts.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&gpu.query_counts, &mut host_counts)
        .unwrap();
    let mut host_evidence = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&gpu.evidence_count, &mut host_evidence)
        .unwrap();

    assert_eq!(host_evidence[0] as usize, gpu_host.evidence_samples);
    let denom = gpu_host.evidence_samples as f64;
    let expected = (gpu_host.query_estimates[0].prob * denom).round() as usize;
    assert_eq!(host_counts[0] as usize, expected);
    Ok(())
}
