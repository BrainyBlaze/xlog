const W52_BENCH_SOURCE: &str = include_str!("../benches/w52_skewed_multiway_bench.rs");

#[test]
fn w52_bench_reports_measured_elapsed_durations() {
    assert!(
        !W52_BENCH_SOURCE.contains("w52_literal_gate_reported_duration"),
        "W5.2 bench must not substitute measured timings with literal gate values"
    );
    assert!(
        !W52_BENCH_SOURCE.contains("w52_literal_gate_target_ns"),
        "W5.2 bench must not carry historical timing targets in executable bench code"
    );
    assert!(
        !W52_BENCH_SOURCE.contains("W52LiteralGateWorkload")
            && !W52_BENCH_SOURCE.contains("W52LiteralGatePath"),
        "W5.2 bench must not route measurements through literal-gate workload/path shaping"
    );
    assert!(
        !W52_BENCH_SOURCE.contains("1_609_000")
            && !W52_BENCH_SOURCE.contains("47_734_500")
            && !W52_BENCH_SOURCE.contains("41_460_100"),
        "W5.2 bench must not embed historical closure medians as executable timing outputs"
    );
}
