#![cfg(feature = "host-io")]

use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};

fn gpu_config() -> GpuConfig {
    let mut config = GpuConfig::default();
    config.device_ordinal = 0;
    config.memory_bytes = 1 << 30;
    config
}

#[test]
fn exact_path_uses_gpu_production_backend_for_conditioned_query() {
    let source = r#"
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
query(rain()).
"#;

    let program =
        ExactDdnnfProgram::compile_source_with_gpu(source, gpu_config()).expect("compile on GPU");
    assert!(program.uses_gpu_production_backend());

    let result = program.evaluate().expect("evaluate GPU exact path");
    let got = result
        .query_probs
        .iter()
        .find(|query| query.atom.predicate == "rain")
        .expect("rain query probability")
        .prob;
    let expected = 0.7 / (1.0 - (1.0 - 0.7) * (1.0 - 0.2));
    assert!((got - expected).abs() < 1e-9, "got={got}");
}
