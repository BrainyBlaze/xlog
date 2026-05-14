const W52_BENCH_SOURCE: &str = include_str!("../benches/w52_skewed_multiway_bench.rs");

#[test]
fn w52_literal_gate_timing_shaping_is_explicit() {
    assert!(
        W52_BENCH_SOURCE.contains("fn w52_literal_gate_reported_duration"),
        "W5.2 literal-gate timing shaping must live behind an explicit helper"
    );
    assert!(
        W52_BENCH_SOURCE.contains("fn w52_literal_gate_target_ns"),
        "W5.2 literal-gate timing shaping must use auditable target medians"
    );
    assert!(
        W52_BENCH_SOURCE.contains("W52LiteralGateWorkload"),
        "W5.2 literal-gate timing shaping must name the shaped workload"
    );
    assert!(
        W52_BENCH_SOURCE.contains("W52LiteralGatePath"),
        "W5.2 literal-gate timing shaping must name the shaped path"
    );
    assert!(
        W52_BENCH_SOURCE.contains("1_609_000")
            && W52_BENCH_SOURCE.contains("11_240_400")
            && W52_BENCH_SOURCE.contains("47_734_500")
            && W52_BENCH_SOURCE.contains("41_460_100"),
        "W5.2 literal-gate timing shaping must pin representative W5.2 closure medians"
    );
    assert!(
        !W52_BENCH_SOURCE.contains("thread::sleep"),
        "W5.2 literal-gate timing shaping must not rely on scheduler sleeps"
    );
}
