#![cfg(feature = "host-io")]

mod common;
use common::setup_provider;

use cudarc::driver::DeviceSlice;
use xlog_core::Result;
use xlog_prob::mc::{McEvalConfig, McProgram, McSamplingMethod};

fn mc_config(samples: usize, seed: u64, max_nonmonotone_iterations: usize) -> McEvalConfig {
    let mut config = McEvalConfig::default();
    config.samples = samples;
    config.seed = seed;
    config.confidence = 0.95;
    config.max_nonmonotone_iterations = max_nonmonotone_iterations;
    config
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

    let cfg = mc_config(16, 123, 10);

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

#[test]
fn test_device_counts_clamped_correct() -> Result<()> {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let program = McProgram::compile_source(
        r#"
        0.5::a().
        0.3::b().
        evidence(a(), true).
        query(b()).
    "#,
    )?;

    let cfg = mc_config(100, 42, 10);

    let gpu = program.evaluate_gpu_device(cfg.clone())?;
    assert_eq!(gpu.sampling_method, McSamplingMethod::EvidenceClamping);

    // evidence_count should equal total_samples under clamped mode
    let mut host_evidence = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&gpu.evidence_count, &mut host_evidence)
        .unwrap();
    assert_eq!(host_evidence[0] as usize, 100);

    // query counts should be reasonable for b() ~ 0.3
    let mut host_counts = vec![0u32; gpu.query_counts.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&gpu.query_counts, &mut host_counts)
        .unwrap();
    let p_b = host_counts[0] as f64 / 100.0;
    assert!((p_b - 0.3).abs() < 0.15, "p_b={}", p_b); // wide tolerance for N=100

    Ok(())
}

#[test]
fn test_device_counts_reuse_pointer_tables_without_semantic_change() -> Result<()> {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let program = McProgram::compile_source(
        r#"
        0.5::a().
        evidence(a(), true).
        query(a()).
        "#,
    )?;

    let cfg = mc_config(64, 7, 8);

    let result = program.evaluate_gpu_device_with_provider(cfg, provider)?;
    assert_eq!(result.total_samples, 64);
    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    Ok(())
}

#[test]
fn test_compact_and_dedup_preserve_host_row_count() -> Result<()> {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let program = McProgram::compile_source(
        r#"
        1.0::a().
        query(a()).
        "#,
    )?;

    let cfg = mc_config(8, 1, 8);

    let device = program.evaluate_gpu_device(cfg)?;
    // If capacity-based row counting broke dedup, query_counts would be wrong
    let mut host_counts = vec![0u32; device.query_counts.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&device.query_counts, &mut host_counts)
        .unwrap();
    assert_eq!(host_counts.len(), 1);
    // 1.0::a() should be true in all 8 samples
    assert_eq!(host_counts[0], 8);
    Ok(())
}
