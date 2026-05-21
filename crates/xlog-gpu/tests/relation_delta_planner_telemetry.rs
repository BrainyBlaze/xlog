use xlog_gpu::logic::{DeltaPlannerTelemetry, LogicDeltaReport};

#[test]
fn xlog_delta_005_reports_cache_reuse_speedup_and_fallback_advice() {
    let mut report = LogicDeltaReport {
        input_delta_count: 1,
        changed_relations: 1,
        changed_relation_names: vec!["candidate_edge".to_string()],
        insert_rows: 2,
        delete_rows: 0,
        has_deletes: false,
        affected_sccs: 2,
        recomputed_sccs: 0,
        incremental_sccs: 2,
        coalesced_insert_rows: 2,
        coalesced_delete_rows: 0,
        canceled_rows: 0,
        debug_trace: Vec::new(),
        planner_telemetry: DeltaPlannerTelemetry::default(),
    };

    report.planner_telemetry =
        DeltaPlannerTelemetry::from_delta_report(&report, true, Some((1_000, 4_000)));

    assert!(report.planner_telemetry.cache_reused);
    assert_eq!(report.planner_telemetry.fallback_decision, "incremental");
    assert_eq!(report.planner_telemetry.affected_sccs, 2);
    assert_eq!(report.planner_telemetry.recomputed_sccs, 0);
    assert_eq!(report.planner_telemetry.measured_delta_speedup, Some(4.0));
    assert!(report
        .planner_telemetry
        .planner_advice
        .iter()
        .any(|item| item.contains("delta path is faster")));
}
