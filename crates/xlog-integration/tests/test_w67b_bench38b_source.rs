#[test]
fn w52_bench38b_source_reports_direct_timing_and_vram_snapshots() {
    let source = include_str!("../benches/w52_skewed_multiway_bench.rs");

    assert!(source.contains("start.elapsed()"));
    assert!(!source.contains("w52_literal_gate_reported_duration"));
    assert!(!source.contains("w52_literal_gate_target_ns"));
    assert!(source.contains("mem_get_info"));
    assert!(source.contains("W67B_BENCH38B_VRAM"));
    assert!(source.contains("VRAM_GATE_BYTES"));
}
