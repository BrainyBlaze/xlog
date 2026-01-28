use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};

#[test]
fn exact_gpu_cache_hit_reuses_circuit() {
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
    let _provider = CudaKernelProvider::new(device, memory).expect("provider");

    let source = r#"
0.5::a().
query(a()).
"#;
    let config = GpuConfig {
        device_ordinal: 0,
        memory_bytes: 1 << 30,
    };

    let prog = ExactDdnnfProgram::compile_source_with_gpu(source, config).expect("compile");
    let r1 = prog.evaluate().expect("eval 1");
    let r2 = prog.evaluate().expect("eval 2");

    assert_eq!(r1.query_probs.len(), r2.query_probs.len());
    assert_eq!(r1.query_probs[0].prob.to_bits(), r2.query_probs[0].prob.to_bits());
}
