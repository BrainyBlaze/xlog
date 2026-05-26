use xlog_cuda::provider::{HostLaunchMetadataTransferStats, HostTransferStats};
use xlog_ir::{EirEpistemicMode, EpistemicCpuFallbackCounters};
use xlog_runtime::{
    EpistemicGpuRuntimeCounters, EpistemicGpuRuntimePreflight, EpistemicGpuRuntimeTrace,
    EpistemicGpuRuntimeWcojCertification, EpistemicGpuTransferBudgetTrace,
    EpistemicGpuWorkspaceLayout,
};

#[test]
fn wcoj_reuse_gate_rejects_missing_runtime_dispatch_evidence() {
    let trace = EpistemicGpuRuntimeTrace::try_from_preflight_and_counters(
        preflight_requiring_kclique_wcoj(),
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters::default(),
    )
    .expect("monotonic runtime counters");

    match trace.wcoj_certification {
        EpistemicGpuRuntimeWcojCertification::MissingRequiredWcojDispatch {
            required_multiway_reductions,
            required_kclique_plans,
            observed_wcoj_dispatches,
            observed_kclique_dispatches,
        } => {
            assert_eq!(required_multiway_reductions, 1);
            assert_eq!(required_kclique_plans, 1);
            assert_eq!(observed_wcoj_dispatches, 0);
            assert_eq!(observed_kclique_dispatches, 0);
        }
        other => panic!("expected missing WCOJ dispatch evidence, got {other:?}"),
    }

    let err = trace
        .require_wcoj_certification()
        .expect_err("preflight-only WCOJ metadata must not certify production reuse");
    assert!(format!("{err}").contains("required_multiway_reductions=1"));
}

#[test]
fn wcoj_reuse_gate_rejects_missing_layout_and_metadata_evidence() {
    let layout_only_missing = EpistemicGpuRuntimeTrace::try_from_preflight_and_counters(
        preflight_requiring_kclique_wcoj(),
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters {
            wcoj_clique5_dispatch_count: 1,
            ..EpistemicGpuRuntimeCounters::default()
        },
    )
    .expect("monotonic runtime counters");
    assert!(matches!(
        layout_only_missing.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::MissingRequiredWcojLayout {
            required_sorted_layouts: 1,
            observed_layout_events: 0
        }
    ));

    let metadata_missing = EpistemicGpuRuntimeTrace::try_from_preflight_and_counters(
        preflight_requiring_kclique_wcoj(),
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters {
            wcoj_clique5_dispatch_count: 1,
            wcoj_layout_fast_path_hit_count: 1,
            ..EpistemicGpuRuntimeCounters::default()
        },
    )
    .expect("monotonic runtime counters");
    assert!(matches!(
        metadata_missing.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::MissingRequiredKcliqueMetadata {
            required_kclique_plans: 1,
            observed_metadata_builds: 0,
            observed_metadata_build_nanos: 0
        }
    ));
}

#[test]
fn wcoj_reuse_gate_certifies_existing_runtime_counters() {
    let trace = EpistemicGpuRuntimeTrace::try_from_preflight_and_counters(
        preflight_requiring_kclique_wcoj(),
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters {
            wcoj_clique5_dispatch_count: 1,
            wcoj_layout_fast_path_hit_count: 1,
            kclique_metadata_build_count: 1,
            kclique_metadata_build_nanos: 42,
            kclique_histogram_refresh_count: 2,
            kclique_histogram_refresh_nanos: 75,
            ..EpistemicGpuRuntimeCounters::default()
        },
    )
    .expect("monotonic runtime counters");

    match trace.wcoj_certification {
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches,
            certified_multiway_reductions,
            observed_kclique_dispatches,
            certified_edge_permutation_slots,
            certified_stream_groups,
            certified_skew_scheduled_plans,
            certified_sorted_layout_requirements,
            certified_helper_split_specs,
            certified_helper_relation_rules,
            certified_helper_relation_scans,
            observed_layout_fast_path_hits,
            observed_metadata_builds,
            observed_metadata_build_nanos,
            observed_histogram_refreshes,
            observed_histogram_refresh_nanos,
            ..
        } => {
            assert_eq!(observed_wcoj_dispatches, 1);
            assert_eq!(certified_multiway_reductions, 1);
            assert_eq!(observed_kclique_dispatches, 1);
            assert_eq!(certified_edge_permutation_slots, 10);
            assert_eq!(certified_stream_groups, 1);
            assert_eq!(certified_skew_scheduled_plans, 1);
            assert_eq!(certified_sorted_layout_requirements, 1);
            assert_eq!(certified_helper_split_specs, 1);
            assert_eq!(certified_helper_relation_rules, 1);
            assert_eq!(certified_helper_relation_scans, 1);
            assert_eq!(observed_layout_fast_path_hits, 1);
            assert_eq!(observed_metadata_builds, 1);
            assert_eq!(observed_metadata_build_nanos, 42);
            assert_eq!(observed_histogram_refreshes, 2);
            assert_eq!(observed_histogram_refresh_nanos, 75);
        }
        other => panic!("expected certified WCOJ reuse evidence, got {other:?}"),
    }
    trace
        .require_wcoj_certification()
        .expect("runtime WCOJ counters should certify production reuse");
}

#[test]
fn hot_path_transfer_budget_allows_launch_metadata_but_rejects_data_plane_transfers() {
    let trace = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats_with_launch_metadata(
        2,
        host_transfer_stats(0, 0, 0, 0),
        host_transfer_stats(0, 0, 0, 0),
        HostLaunchMetadataTransferStats::default(),
        HostLaunchMetadataTransferStats {
            htod_calls: 3,
            htod_bytes: 24,
        },
    )
    .expect("launch metadata is bounded orchestration, not data-plane fallback");
    assert_eq!(trace.tracked_data_plane_htod_calls, 0);
    assert_eq!(trace.tracked_launch_metadata_htod_calls, 3);
    assert_eq!(trace.tracked_aggregate_htod_calls, 3);
    assert_eq!(trace.per_candidate_host_round_trips, 0);

    let err = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats(
        2,
        host_transfer_stats(0, 0, 0, 0),
        host_transfer_stats(4, 0, 1, 0),
    )
    .expect_err("data-plane D2H inside the hot path must fail production reuse gates");
    assert!(format!("{err}").contains("tracked host transfer in GPU hot path"));
}

fn host_transfer_stats(
    dtoh_bytes: u64,
    htod_bytes: u64,
    dtoh_calls: u64,
    htod_calls: u64,
) -> HostTransferStats {
    HostTransferStats {
        dtoh_bytes,
        htod_bytes,
        dtoh_calls,
        htod_calls,
    }
}

fn preflight_requiring_kclique_wcoj() -> EpistemicGpuRuntimePreflight {
    EpistemicGpuRuntimePreflight {
        epistemic_mode: EirEpistemicMode::Faeel,
        workspace_layout: EpistemicGpuWorkspaceLayout {
            candidate_assumption_bytes: 2,
            world_view_bytes: 2,
            model_membership_bytes: 2,
            rejection_reason_slots: 2,
        },
        reduced_runtime_rule_count: 2,
        reduced_constraint_relation_count: 0,
        wcoj_required_reduction_count: 1,
        multiway_reduction_count: 1,
        kclique_wcoj_plan_count: 1,
        wcoj_triangle_route_count: 0,
        wcoj_4cycle_route_count: 0,
        kclique_wcoj_plan_count_by_arity: [1, 0, 0, 0],
        kclique_wcoj_max_arity: 5,
        kclique_wcoj_edge_permutation_count: 10,
        kclique_stream_group_count: 1,
        kclique_skew_scheduled_plan_count: 1,
        planned_hash_route_count: 0,
        planned_hash_planner_wins_count: 0,
        planned_hash_incomplete_stats_count: 0,
        planned_hash_cost_evidence_count: 0,
        sorted_layout_requirement_count: 1,
        helper_split_spec_count: 1,
        helper_relation_rule_count: 1,
        helper_relation_scan_count: 1,
        tuple_membership_binding_count: 1,
        solver_assumption_binding_count: 1,
        solver_required_capability_count: 5,
        solver_required_status_count: 4,
        know_operator_count: 1,
        possible_operator_count: 0,
        not_know_operator_count: 0,
        not_possible_operator_count: 0,
        cpu_fallbacks: EpistemicCpuFallbackCounters::default(),
    }
}
