#![cfg(feature = "host-io")]

use xlog_core::MemoryBudget;
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};

fn gpu_config(memory_bytes: u64) -> GpuConfig {
    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = memory_bytes;
    config
}

#[test]
fn exact_gpu_cache_hit_reuses_circuit() {
    let _provider =
        match xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(1 << 30)).build() {
            Ok(provider) => provider,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };

    let source = r#"
0.5::a().
query(a()).
"#;
    let config = gpu_config(1 << 30);

    let prog = ExactDdnnfProgram::compile_source_with_gpu(source, config).expect("compile");
    let r1 = prog.evaluate().expect("eval 1");
    let r2 = prog.evaluate().expect("eval 2");

    assert_eq!(r1.query_probs.len(), r2.query_probs.len());
    assert_eq!(
        r1.query_probs[0].prob.to_bits(),
        r2.query_probs[0].prob.to_bits()
    );
}
