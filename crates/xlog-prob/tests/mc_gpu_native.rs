use std::sync::Arc;

use cudarc::driver::{DevicePtr, DeviceSlice, LaunchAsync};
use xlog_core::MemoryBudget;
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
fn mc_gpu_device_counts_match_expected_small() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let program = McProgram::compile_source(
        r#"
1.0::coin().
query(coin()).
"#,
    )
    .expect("compile program");

    let cfg = McEvalConfig {
        samples: 128,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 16,
        sampling_method: None,
        ..Default::default()
    };

    let device_result = program
        .evaluate_gpu_device_with_provider(cfg.clone(), provider.clone())
        .expect("evaluate_gpu_device_with_provider");

    assert_eq!(device_result.query_counts.len(), 1);

    let mut host_counts = vec![0u32; device_result.query_counts.len()];
    if !host_counts.is_empty() {
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&device_result.query_counts, &mut host_counts)
            .expect("dtoh query counts");
    }
    let mut host_evidence = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&device_result.evidence_count, &mut host_evidence)
        .expect("dtoh evidence count");

    assert_eq!(
        host_evidence[0] as usize,
        cfg.samples,
        "evidence_count={} query_count={}",
        host_evidence[0],
        host_counts.get(0).copied().unwrap_or(0)
    );
    assert_eq!(
        host_counts.get(0).copied().unwrap_or(0) as usize,
        cfg.samples
    );
}

#[test]
fn mc_host_read_apis_gated() {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("mc");
    path.push("mod.rs");

    let text = std::fs::read_to_string(&path).expect("read mc/mod.rs");
    assert!(
        text.contains("#[cfg(feature = \"host-io\")]\n    pub fn evaluate"),
        "evaluate() must be gated behind host-io"
    );
    assert!(
        text.contains("#[cfg(feature = \"host-io\")]\n    pub fn evaluate_cpu"),
        "evaluate_cpu() must be gated behind host-io"
    );
    assert!(
        text.contains("#[cfg(feature = \"host-io\")]\n    pub fn evaluate_gpu"),
        "evaluate_gpu() must be gated behind host-io"
    );
}

#[test]
fn mc_eval_kernels_set_evidence_ok_without_evidence() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mut d_query_count = provider
        .memory()
        .alloc::<u32>(1)
        .expect("alloc query count");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u32], &mut d_query_count)
        .expect("copy query count");
    let query_ptr = *d_query_count.device_ptr() as u64;

    let mut d_query_ptrs = provider.memory().alloc::<u64>(1).expect("alloc query ptrs");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[query_ptr], &mut d_query_ptrs)
        .expect("copy query ptrs");

    let mut d_evidence_ptrs = provider
        .memory()
        .alloc::<u64>(1)
        .expect("alloc evidence ptrs");
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_evidence_ptrs)
        .expect("zero evidence ptrs");
    let mut d_evidence_expected = provider
        .memory()
        .alloc::<u8>(1)
        .expect("alloc evidence expected");
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_evidence_expected)
        .expect("zero evidence expected");

    let mut d_query_flags = provider.memory().alloc::<u8>(1).expect("alloc query flags");
    let mut d_evidence_ok = provider.memory().alloc::<u8>(1).expect("alloc evidence ok");

    let truth_fn = provider
        .device()
        .inner()
        .get_func(
            xlog_cuda::provider::MC_EVAL_MODULE,
            xlog_cuda::provider::mc_eval_kernels::MC_EVAL_QUERY_EVIDENCE_TRUTH,
        )
        .expect("mc_eval_query_evidence_truth kernel");

    unsafe {
        truth_fn
            .clone()
            .launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &d_query_ptrs,
                    1u32,
                    &d_evidence_ptrs,
                    &d_evidence_expected,
                    0u32,
                    &mut d_query_flags,
                    &mut d_evidence_ok,
                ),
            )
            .expect("launch truth kernel");
    }

    provider
        .device()
        .synchronize()
        .expect("sync after truth kernel");

    let mut host_flags = [0u8];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&d_query_flags, &mut host_flags)
        .expect("copy query flags");
    let mut host_ok = [0u8];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&d_evidence_ok, &mut host_ok)
        .expect("copy evidence ok");

    assert_eq!(host_flags[0], 1u8);
    assert_eq!(host_ok[0], 1u8);
}

#[test]
fn mc_accumulate_counts_increments_on_ok() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mut d_query_flags = provider.memory().alloc::<u8>(1).expect("alloc query flags");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u8], &mut d_query_flags)
        .expect("copy query flags");
    let mut d_evidence_ok = provider.memory().alloc::<u8>(1).expect("alloc evidence ok");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u8], &mut d_evidence_ok)
        .expect("copy evidence ok");

    let mut d_query_counts = provider
        .memory()
        .alloc::<u32>(1)
        .expect("alloc query counts");
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_query_counts)
        .expect("zero query counts");
    let mut d_evidence_count = provider
        .memory()
        .alloc::<u32>(1)
        .expect("alloc evidence count");
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_evidence_count)
        .expect("zero evidence count");

    let accum_fn = provider
        .device()
        .inner()
        .get_func(
            xlog_cuda::provider::MC_EVAL_MODULE,
            xlog_cuda::provider::mc_eval_kernels::MC_EVAL_ACCUMULATE_COUNTS,
        )
        .expect("mc_accumulate_counts kernel");

    unsafe {
        accum_fn
            .clone()
            .launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &d_query_flags,
                    1u32,
                    &d_evidence_ok,
                    &mut d_query_counts,
                    &mut d_evidence_count,
                ),
            )
            .expect("launch accumulate kernel");
    }

    provider
        .device()
        .synchronize()
        .expect("sync after accumulate kernel");

    let mut host_query_counts = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&d_query_counts, &mut host_query_counts)
        .expect("copy query counts");
    let mut host_evidence_count = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&d_evidence_count, &mut host_evidence_count)
        .expect("copy evidence count");

    assert_eq!(host_query_counts[0], 1u32);
    assert_eq!(host_evidence_count[0], 1u32);
}

#[test]
fn mc_hot_path_no_device_row_count_helper() {
    let mut mc_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    mc_dir.push("src");
    mc_dir.push("mc");
    let mut text = String::new();
    for entry in std::fs::read_dir(&mc_dir).expect("read mc/ dir") {
        let entry = entry.expect("dir entry");
        if entry.path().extension().map_or(false, |e| e == "rs") {
            text.push_str(&std::fs::read_to_string(entry.path()).expect("read mc/*.rs"));
        }
    }
    assert!(!text.contains("device_row_count_u32(provider, &filtered)"));
}

#[test]
fn mc_behavior_tests_do_not_use_large_sample_budgets() {
    let text = std::fs::read_to_string("crates/xlog-prob/tests/mc.rs")
        .or_else(|_| std::fs::read_to_string("tests/mc.rs"))
        .or_else(|_| {
            let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            p.push("tests");
            p.push("mc.rs");
            std::fs::read_to_string(p)
        })
        .unwrap();
    assert!(!text.contains("samples: 80_000"), "mc.rs should not contain samples: 80_000");
}
