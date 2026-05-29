#[cfg(feature = "epistemic-logic-tests")]
use std::collections::BTreeMap;
use std::sync::Arc;

use xlog_core::{MemoryBudget, RelId, RuntimeConfig, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::provider::{HostLaunchMetadataTransferStats, HostTransferStats};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::{
    rir::{
        CostPredictionRecord, HelperSplitSpec, KCliqueVariableOrder, LookupPerm, MultiwayPlan,
        PlannedHashReason, ProjectExpr, RirNode, SortedLayoutSpec, StreamGroupId, VariableOrder,
    },
    CompiledRule, EirAtom, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EirTerm,
    EpistemicCpuFallbackCounters, EpistemicExecutablePlan, EpistemicGpuPlan,
    EpistemicReductionPlan, EpistemicWcojReductionStatus, ExecutionPlan, RirMeta, Scc, Stratum,
};
use xlog_runtime::{
    EpistemicGpuCandidateGenerationTrace, EpistemicGpuCandidateValidationTrace,
    EpistemicGpuFinalResultMaterializationTrace, EpistemicGpuFinalTupleMaterializationTrace,
    EpistemicGpuKernelTimingTrace, EpistemicGpuMaterializationTrace,
    EpistemicGpuModelMembershipSource, EpistemicGpuModelMembershipTrace,
    EpistemicGpuPropagationTrace, EpistemicGpuRuntimeCounters, EpistemicGpuRuntimePreflight,
    EpistemicGpuRuntimeTrace, EpistemicGpuRuntimeWcojCertification,
    EpistemicGpuTransferBudgetTrace, EpistemicGpuWorkspaceCapacities, EpistemicGpuWorkspaceLayout,
    EpistemicGpuWorkspaceResetTrace, EpistemicGpuWorldViewValidationTrace, Executor,
};

#[cfg(feature = "epistemic-logic-tests")]
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, compile_epistemic_gpu_execution_with_stats_snapshot,
    compile_epistemic_gpu_split_execution,
};
#[cfg(feature = "epistemic-logic-tests")]
use xlog_logic::{parse_program, Compiler};
#[cfg(feature = "epistemic-logic-tests")]
use xlog_runtime::EpistemicGpuRejectionReason;
#[cfg(feature = "epistemic-logic-tests")]
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> std::result::Result<(), SinkError> {
        Ok(())
    }
}

struct RuntimeFixture {
    _device: Arc<CudaDevice>,
    _runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    _pool: Arc<StreamPool>,
}

fn runtime_fixture() -> Option<RuntimeFixture> {
    let device = match CudaDevice::new(0) {
        Ok(device) => Arc::new(device),
        Err(err) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {err}");
            return None;
        }
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 128 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(128 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider = match CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory))
    {
        Ok(provider) => Arc::new(provider),
        Err(err) => {
            eprintln!("Skipping test: CUDA kernel provider unavailable: {err}");
            return None;
        }
    };

    Some(RuntimeFixture {
        _device: device,
        _runtime: runtime,
        memory,
        provider,
        _pool: pool,
    })
}

fn assert_u32_resource_exhausted(
    err: xlog_core::XlogError,
    expected_context: &str,
    expected_estimated_bytes: u64,
) {
    match err {
        xlog_core::XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } => {
            assert_eq!(context, expected_context);
            assert_eq!(estimated_bytes, expected_estimated_bytes);
            assert_eq!(budget_bytes, u32::MAX as u64);
        }
        other => panic!("expected u32 resource exhaustion error, got {other:?}"),
    }
}

#[test]
fn workspace_layout_sizes_all_required_epistemic_gpu_buffers() {
    let plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![
            epistemic_literal("known_fact", EirEpistemicOp::Know),
            epistemic_literal("possible_fact", EirEpistemicOp::Possible),
        ],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "out".to_string(),
            relational_body_atoms: 3,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );

    let layout = EpistemicGpuWorkspaceLayout::for_plan(
        &plan,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    assert_eq!(layout.candidate_assumption_bytes, 16);
    assert_eq!(layout.world_view_bytes, 32);
    assert_eq!(layout.model_membership_bytes, 96);
    assert_eq!(layout.rejection_reason_slots, 8);
    assert_eq!(layout.total_bytes(), 176);
}

#[test]
fn workspace_layout_rejects_zero_candidate_capacity() {
    let plan = EpistemicGpuPlan::new(
        EirEpistemicMode::G91,
        vec![epistemic_literal("fact", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "out".to_string(),
            relational_body_atoms: 1,
            wcoj_status: EpistemicWcojReductionStatus::NotWcojCandidate,
        }],
    );

    let err = EpistemicGpuWorkspaceLayout::for_plan(
        &plan,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 0,
            max_worlds: 1,
            max_models_per_reduction: 1,
        },
    )
    .unwrap_err();

    match err {
        xlog_core::XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } => {
            assert_eq!(context, "epistemic GPU workspace candidates");
            assert_eq!(estimated_bytes, 0);
            assert_eq!(budget_bytes, 1);
        }
        other => panic!("expected workspace capacity error, got {other:?}"),
    }
}

#[test]
fn runtime_preflight_records_workspace_and_wcoj_route_surfaces() {
    let executable = executable_with_kclique_wcoj_plan();

    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    assert_eq!(preflight.workspace_layout.total_bytes(), 120);
    assert_eq!(preflight.reduced_runtime_rule_count, 2);
    assert_eq!(preflight.multiway_reduction_count, 1);
    assert_eq!(preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(preflight.kclique_wcoj_max_arity, 5);
    assert_eq!(preflight.kclique_wcoj_edge_permutation_count, 10);
    assert_eq!(preflight.kclique_stream_group_count, 1);
    assert_eq!(preflight.kclique_skew_scheduled_plan_count, 1);
    assert_eq!(preflight.planned_hash_route_count, 0);
    assert_eq!(preflight.sorted_layout_requirement_count, 2);
    assert_eq!(preflight.helper_split_spec_count, 1);
    assert_eq!(preflight.helper_relation_rule_count, 1);
    assert_eq!(preflight.helper_relation_scan_count, 1);
    assert_eq!(preflight.tuple_membership_binding_count, 1);
    assert!(preflight.cpu_fallbacks.is_zero());
}

#[test]
fn accepted_gpu_execution_dispatches_kclique_wcoj_reduction_through_runtime_path() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    let complete_k5_edges = complete_k5_edge_rows();

    for rel in 1..=10 {
        let name = format!("edge{rel}");
        executor.register_relation(RelId(rel), &name);
        executor.put_relation(
            &name,
            upload_binary_u32(&fixture.memory, &complete_k5_edges, "src", "dst"),
        );
    }
    executor.register_relation(RelId(99), "__w37_helper_99");

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable_with_live_kclique_wcoj_literal(),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("real epistemic runtime execution should dispatch the K-clique WCOJ reduction");

    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(result.prepared.preflight.helper_relation_rule_count, 1);
    assert_eq!(result.prepared.preflight.helper_relation_scan_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert!(result.prepared.preflight.cpu_fallbacks.is_zero());
    assert_eq!(result.trace.counter_delta.wcoj_clique5_dispatch_count, 1);
    assert_eq!(result.trace.counter_delta.kclique_metadata_build_count, 1);
    assert!(result.trace.counter_delta.kclique_metadata_build_nanos > 0);
    match result.trace.wcoj_certification {
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches,
            observed_kclique_dispatches,
            observed_layout_sorts,
            observed_layout_fast_path_hits,
            observed_metadata_builds,
            ..
        } => {
            assert_eq!(observed_wcoj_dispatches, 1);
            assert_eq!(observed_kclique_dispatches, 1);
            assert!(observed_layout_sorts + observed_layout_fast_path_hits >= 2);
            assert_eq!(observed_metadata_builds, 1);
        }
        other => panic!("expected certified WCOJ dispatch, got {other:?}"),
    }
    assert_eq!(result.output.arity(), 5);
    assert_eq!(result.final_output.arity(), 5);
    assert!(result.final_result_transfer.final_output_rows > 0);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        5
    );
    assert_eq!(result.transfer_budget.candidate_count, 2);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_bytes, 0);
    assert_eq!(result.transfer_budget.tracked_data_plane_htod_calls, 0);
    assert_eq!(result.transfer_budget.tracked_data_plane_htod_bytes, 0);
    assert!(result.transfer_budget.tracked_launch_metadata_htod_calls > 0);
    assert!(result.transfer_budget.tracked_launch_metadata_htod_bytes > 0);
    assert_eq!(
        result.transfer_budget.tracked_aggregate_htod_calls,
        result.transfer_budget.tracked_launch_metadata_htod_calls
    );
    assert_eq!(result.transfer_budget.per_candidate_host_round_trips, 0);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
        .require_runtime_dispatch_certification()
        .expect("runtime result should retain dispatch and semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_epistemic_kclique_reduction_dispatches_wcoj_runtime_path() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(EPISTEMIC_K5_MARKER_SRC).expect("parse epistemic K5 WCOJ program");
    let rel_ids = rel_ids_for_parsed_k5_reduced();
    let stats = k5_stats(&rel_ids, Some((3, 5.0)));
    let executable = compile_epistemic_gpu_execution_with_stats_snapshot(&program, Some(&stats))
        .expect("compile parsed epistemic K5 through production WCOJ planner");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }

    let complete_k5_edges = complete_k5_edge_rows();
    for (name, _, _) in K5_EDGES {
        executor.put_relation(
            name,
            upload_binary_u32(&fixture.memory, &complete_k5_edges, "src", "dst"),
        );
    }
    executor.put_relation(
        "marker",
        upload_single_u32_row(
            &fixture.memory,
            &[1, 2, 3, 4, 5],
            &["a", "b", "c", "d", "e"],
        ),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("parsed epistemic K5 should dispatch through runtime WCOJ path");

    assert_eq!(
        executable.gpu_plan.reductions[0].wcoj_status,
        EpistemicWcojReductionStatus::RequiresPlannerEligibility
    );
    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(result.prepared.preflight.kclique_wcoj_max_arity, 5);
    assert_eq!(
        result
            .prepared
            .preflight
            .kclique_wcoj_edge_permutation_count,
        10
    );
    assert_eq!(result.prepared.preflight.helper_split_spec_count, 1);
    assert_eq!(result.prepared.preflight.helper_relation_rule_count, 1);
    assert_eq!(result.prepared.preflight.helper_relation_scan_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert!(result.prepared.preflight.cpu_fallbacks.is_zero());
    assert_eq!(result.trace.counter_delta.wcoj_clique5_dispatch_count, 1);
    assert_eq!(result.trace.counter_delta.kclique_metadata_build_count, 1);
    assert!(result.trace.counter_delta.kclique_metadata_build_nanos > 0);
    match result.trace.wcoj_certification {
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches,
            observed_kclique_dispatches,
            observed_metadata_builds,
            certified_helper_split_specs,
            certified_helper_relation_rules,
            certified_helper_relation_scans,
            ..
        } => {
            assert_eq!(observed_wcoj_dispatches, 1);
            assert_eq!(observed_kclique_dispatches, 1);
            assert_eq!(observed_metadata_builds, 1);
            assert_eq!(certified_helper_split_specs, 1);
            assert_eq!(certified_helper_relation_rules, 1);
            assert_eq!(certified_helper_relation_scans, 1);
        }
        other => panic!("expected parsed K5 WCOJ certification, got {other:?}"),
    }
    assert_eq!(result.output.arity(), 5);
    assert_eq!(result.final_output.arity(), 5);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    let a_values = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download parsed K5 final A values");
    let b_values = fixture
        .provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download parsed K5 final B values");
    let c_values = fixture
        .provider
        .download_column::<u32>(&result.final_output, 2)
        .expect("download parsed K5 final C values");
    let d_values = fixture
        .provider
        .download_column::<u32>(&result.final_output, 3)
        .expect("download parsed K5 final D values");
    let e_values = fixture
        .provider
        .download_column::<u32>(&result.final_output, 4)
        .expect("download parsed K5 final E values");
    assert_eq!(
        (a_values, b_values, c_values, d_values, e_values),
        (vec![1], vec![2], vec![3], vec![4], vec![5])
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        5
    );
    assert_eq!(result.transfer_budget.candidate_count, 2);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_bytes, 0);
    assert_eq!(result.transfer_budget.tracked_data_plane_htod_calls, 0);
    assert_eq!(result.transfer_budget.tracked_data_plane_htod_bytes, 0);
    assert!(result.transfer_budget.tracked_launch_metadata_htod_calls > 0);
    assert!(result.transfer_budget.tracked_launch_metadata_htod_bytes > 0);
    assert_eq!(
        result.transfer_budget.tracked_aggregate_htod_calls,
        result.transfer_budget.tracked_launch_metadata_htod_calls
    );
    assert_eq!(result.transfer_budget.per_candidate_host_round_trips, 0);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidate_indices, vec![0]);
    assert_eq!(
        result
            .semantic_trace
            .typed_rejection_reasons()
            .expect("decode typed GPU rejection reason"),
        vec![EpistemicGpuRejectionReason::UnsatisfiedMembership]
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
        .require_runtime_dispatch_certification()
        .expect("parsed K5 runtime result should retain dispatch and semantic certification");
}

#[test]
fn accepted_gpu_execution_dispatches_triangle_wcoj_reduction_through_runtime_path() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut config = RuntimeConfig::default();
    config.wcoj_triangle_dispatch = Some(true);
    let mut executor = Executor::new_with_config(Arc::clone(&fixture.provider), config);
    executor.register_relation(RelId(1), "edge_xy");
    executor.register_relation(RelId(2), "edge_yz");
    executor.register_relation(RelId(3), "edge_xz");
    executor.put_relation(
        "edge_xy",
        upload_binary_u32(&fixture.memory, &[(1, 2)], "x", "y"),
    );
    executor.put_relation(
        "edge_yz",
        upload_binary_u32(&fixture.memory, &[(2, 3)], "y", "z"),
    );
    executor.put_relation(
        "edge_xz",
        upload_binary_u32(&fixture.memory, &[(1, 3)], "x", "z"),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable_with_live_triangle_wcoj_literal(),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("real epistemic runtime execution should dispatch the triangle WCOJ reduction");

    assert_eq!(result.prepared.preflight.multiway_reduction_count, 1);
    assert_eq!(result.prepared.preflight.wcoj_triangle_route_count, 1);
    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 0);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert!(result.prepared.preflight.cpu_fallbacks.is_zero());
    assert_eq!(result.trace.counter_delta.wcoj_triangle_dispatch_count, 1);
    let aggregate_kernel_timing = result.aggregate_kernel_timing();
    assert!(aggregate_kernel_timing.is_recorded());
    assert!(aggregate_kernel_timing.cuda_event_pairs > 0);
    assert!(aggregate_kernel_timing.timing_sync_ops > 0);
    match result.trace.wcoj_certification {
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches,
            observed_kclique_dispatches,
            certified_multiway_reductions,
            ..
        } => {
            assert_eq!(observed_wcoj_dispatches, 1);
            assert_eq!(observed_kclique_dispatches, 0);
            assert_eq!(certified_multiway_reductions, 1);
        }
        other => panic!("expected certified triangle WCOJ dispatch, got {other:?}"),
    }
    assert_eq!(result.output.arity(), 3);
    assert_eq!(result.final_output.arity(), 3);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        3
    );
    assert_eq!(result.transfer_budget.candidate_count, 2);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_bytes, 0);
    assert_eq!(result.transfer_budget.tracked_data_plane_htod_calls, 0);
    assert_eq!(result.transfer_budget.tracked_data_plane_htod_bytes, 0);
    assert_eq!(result.transfer_budget.per_candidate_host_round_trips, 0);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result
            .final_tuple_materialization
            .row_specific_membership_row_capacity,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let xs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final triangle x values");
    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download final triangle y values");
    let zs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 2)
        .expect("download final triangle z values");
    assert_eq!(xs, vec![1]);
    assert_eq!(ys, vec![2]);
    assert_eq!(zs, vec![3]);
    result
        .require_runtime_dispatch_certification()
        .expect("triangle runtime result should retain dispatch and semantic certification");
}

#[test]
fn accepted_gpu_execution_dispatches_4cycle_wcoj_reduction_through_runtime_path() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut config = RuntimeConfig::default();
    config.wcoj_4cycle_dispatch = Some(true);
    let mut executor = Executor::new_with_config(Arc::clone(&fixture.provider), config);
    executor.register_relation(RelId(1), "edge_wx");
    executor.register_relation(RelId(2), "edge_xy");
    executor.register_relation(RelId(3), "edge_yz");
    executor.register_relation(RelId(4), "edge_zw");
    executor.put_relation(
        "edge_wx",
        upload_binary_u32(&fixture.memory, &[(1, 2)], "w", "x"),
    );
    executor.put_relation(
        "edge_xy",
        upload_binary_u32(&fixture.memory, &[(2, 3)], "x", "y"),
    );
    executor.put_relation(
        "edge_yz",
        upload_binary_u32(&fixture.memory, &[(3, 4)], "y", "z"),
    );
    executor.put_relation(
        "edge_zw",
        upload_binary_u32(&fixture.memory, &[(4, 1)], "z", "w"),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable_with_live_v070_4cycle_wcoj_literal(),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("real epistemic runtime execution should dispatch the 4-cycle WCOJ reduction");

    assert_eq!(result.prepared.preflight.multiway_reduction_count, 1);
    assert_eq!(result.prepared.preflight.wcoj_4cycle_route_count, 1);
    assert_eq!(result.prepared.preflight.kclique_wcoj_plan_count, 0);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert!(result.prepared.preflight.cpu_fallbacks.is_zero());
    assert_eq!(result.trace.counter_delta.wcoj_4cycle_dispatch_count, 1);
    match result.trace.wcoj_certification {
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches,
            observed_kclique_dispatches,
            certified_multiway_reductions,
            ..
        } => {
            assert_eq!(observed_wcoj_dispatches, 1);
            assert_eq!(observed_kclique_dispatches, 0);
            assert_eq!(certified_multiway_reductions, 1);
        }
        other => panic!("expected certified 4-cycle WCOJ dispatch, got {other:?}"),
    }
    assert_eq!(result.output.arity(), 4);
    assert_eq!(result.final_output.arity(), 4);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.transfer_budget.candidate_count, 2);
    assert_eq!(result.transfer_budget.tracked_dtoh_calls, 0);
    assert_eq!(result.transfer_budget.tracked_dtoh_bytes, 0);
    assert_eq!(result.transfer_budget.tracked_data_plane_htod_calls, 0);
    assert_eq!(result.transfer_budget.tracked_data_plane_htod_bytes, 0);
    assert_eq!(result.transfer_budget.per_candidate_host_round_trips, 0);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result
            .final_tuple_materialization
            .row_specific_membership_row_capacity,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let ws = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final 4-cycle w values");
    let xs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download final 4-cycle x values");
    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 2)
        .expect("download final 4-cycle y values");
    let zs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 3)
        .expect("download final 4-cycle z values");
    assert_eq!(ws, vec![1]);
    assert_eq!(xs, vec![2]);
    assert_eq!(ys, vec![3]);
    assert_eq!(zs, vec![4]);
    result
        .require_runtime_dispatch_certification()
        .expect("4-cycle runtime result should retain dispatch and semantic certification");
}

#[test]
fn accepted_gpu_execution_dispatches_nondefault_4cycle_wcoj_leader_through_runtime_path() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut config = RuntimeConfig::default();
    config.wcoj_4cycle_dispatch = Some(true);
    let mut executor = Executor::new_with_config(Arc::clone(&fixture.provider), config);
    executor.register_relation(RelId(1), "edge_wx");
    executor.register_relation(RelId(2), "edge_xy");
    executor.register_relation(RelId(3), "edge_yz");
    executor.register_relation(RelId(4), "edge_zw");
    executor.put_relation(
        "edge_wx",
        upload_binary_u32(&fixture.memory, &[(1, 2)], "w", "x"),
    );
    executor.put_relation(
        "edge_xy",
        upload_binary_u32(&fixture.memory, &[(2, 3)], "x", "y"),
    );
    executor.put_relation(
        "edge_yz",
        upload_binary_u32(&fixture.memory, &[(3, 4)], "y", "z"),
    );
    executor.put_relation(
        "edge_zw",
        upload_binary_u32(&fixture.memory, &[(4, 1)], "z", "w"),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable_with_live_v070_4cycle_wcoj_literal_for_leader(1),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("nondefault 4-cycle WCOJ leader should dispatch through accepted runtime");

    assert_eq!(result.prepared.preflight.multiway_reduction_count, 1);
    assert_eq!(result.prepared.preflight.wcoj_4cycle_route_count, 1);
    assert_eq!(result.trace.counter_delta.wcoj_4cycle_dispatch_count, 1);
    assert_eq!(result.output.arity(), 4);
    assert_eq!(result.final_output.arity(), 4);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let ws = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download nondefault 4-cycle w values");
    let xs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download nondefault 4-cycle x values");
    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 2)
        .expect("download nondefault 4-cycle y values");
    let zs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 3)
        .expect("download nondefault 4-cycle z values");
    assert_eq!(ws, vec![1]);
    assert_eq!(xs, vec![2]);
    assert_eq!(ys, vec![3]);
    assert_eq!(zs, vec![4]);
    result
        .require_runtime_dispatch_certification()
        .expect("nondefault 4-cycle runtime result should retain dispatch certification");
}

#[test]
fn runtime_execution_rejects_nonzero_arity_row_count_only_membership_binding() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    executor.register_relation(RelId(1), "base");
    executor.put_relation("base", upload_unary_u32(&fixture.memory, &[7], "x"));

    let err = match executor.execute_epistemic_gpu_execution(
        &nonzero_arity_row_count_only_membership_executable(),
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 4,
            max_worlds: 2,
            max_models_per_reduction: 2,
        },
    ) {
        Ok(_) => panic!("nonzero-arity membership must not fall back to row-count-only evidence"),
        Err(err) => err,
    };

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU tuple membership binding");
            assert!(context.contains("literal_index 0"));
            assert!(context.contains("0 key columns for arity 1"));
        }
        other => panic!("expected tuple-membership binding error, got {other:?}"),
    }
}

#[test]
fn batch_execution_runs_split_components_through_gpu_runtime_subplans() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    executor.register_relation(RelId(11), "left_base");
    executor.register_relation(RelId(12), "right_base");
    executor.put_relation("left_base", upload_unary_u32(&fixture.memory, &[7], "x"));
    executor.put_relation("right_base", upload_unary_u32(&fixture.memory, &[9], "x"));

    let left = accepted_ground_literal_component_executable("left_base", "left_out", RelId(11), 7);
    let right =
        accepted_ground_literal_component_executable("right_base", "right_out", RelId(12), 9);
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &[&left, &right],
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("split components should execute through the GPU runtime batch adapter");

    assert_eq!(batch.results.len(), 2);
    batch
        .require_trace_matches_components("epistemic GPU split execution")
        .expect("batch trace must be derived from real component results");
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.cpu_solver_search_fallbacks, 0);
    assert_eq!(batch.trace.cpu_probability_recomputations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert_eq!(batch.trace.final_output_rows, 2);
    assert_eq!(batch.trace.accepted_world_views, 2);
    assert_eq!(batch.trace.know_operator_count, 2);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    for result in &batch.results {
        assert_eq!(result.final_result_transfer.final_output_rows, 1);
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            1
        );
        assert_eq!(result.semantic_trace.accepted_candidates, 1);
        assert_eq!(result.semantic_trace.accepted_world_views, 1);
        result
            .require_runtime_dispatch_certification()
            .expect("component runtime evidence must remain individually certified");
    }
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_split_components_execute_through_gpu_runtime_batch() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred left_seed(u32).
        pred left_gate(u32).
        pred left_out(u32).
        pred right_seed(u32).
        pred right_gate(u32).
        pred right_out(u32).

        left_out(X) :- left_seed(X), know left_gate(X).
        right_out(X) :- right_seed(X), know right_gate(X).
        "#,
    )
    .expect("parse independent epistemic split program");
    let split =
        compile_epistemic_gpu_split_execution(&program).expect("compile parsed split components");
    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);

    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for component in &split.components {
        for (name, rel) in &component.executable.relation_ids {
            executor.register_relation(*rel, name);
        }
    }
    for (name, rows) in [
        ("left_seed", &[7][..]),
        ("left_gate", &[7][..]),
        ("right_seed", &[9][..]),
        ("right_gate", &[9][..]),
    ] {
        executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
    }

    let executables: Vec<_> = split
        .components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("parsed split components should execute through GPU runtime batch adapter");

    assert_eq!(batch.results.len(), 2);
    batch
        .require_trace_matches_components("parsed epistemic GPU split execution")
        .expect("parsed split batch trace must be derived from real component results");
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.cpu_solver_search_fallbacks, 0);
    assert_eq!(batch.trace.cpu_probability_recomputations, 0);
    assert_eq!(batch.trace.final_output_rows, 2);
    assert_eq!(batch.trace.accepted_world_views, 2);
    assert_eq!(batch.trace.know_operator_count, 2);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    for result in &batch.results {
        assert_eq!(result.prepared.preflight.reduced_runtime_rule_count, 1);
        assert_eq!(result.final_result_transfer.final_output_rows, 1);
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            1
        );
        assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
        assert_eq!(result.semantic_trace.accepted_candidates, 1);
        assert_eq!(result.semantic_trace.rejected_candidates, 1);
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        result
            .require_runtime_dispatch_certification()
            .expect("parsed split component runtime evidence must remain certified");
    }
}

/// EGB-05 K1 equivalence: a split component's GPU output must be *row-identical*
/// to running that component's isolated subprogram through the UNSPLIT
/// single-execution path.
///
/// The unsplit engine rejects multi-output programs
/// (`require_single_epistemic_output_relation`), so whole-program "split vs
/// unsplit" cannot be expressed directly for an independent 2-component program.
/// Per-component identity here, combined with exactly-once recomposition
/// coverage (logic-side `recomposition_covers_each_relevant_rule_exactly_once`),
/// establishes whole-program equivalence. Both sides are real device runs;
/// neither output is hardcoded.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn split_component_outputs_match_unsplit_single_execution_outputs() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred left_edge(u32, u32).
        pred left_gate(u32).
        pred left_out(u32).
        pred right_edge(u32, u32).
        pred right_gate(u32).
        pred right_out(u32).

        left_out(Y) :- left_edge(X, Y), know left_gate(X).
        right_out(Y) :- right_edge(X, Y), know right_gate(X).
        "#,
    )
    .expect("parse independent two-component equivalence program");
    let split =
        compile_epistemic_gpu_split_execution(&program).expect("compile split components");
    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);

    let left_edge_rows = [(1u32, 10u32), (2, 20), (3, 30), (4, 10)];
    let left_gate_rows = [1u32, 3, 4];
    let right_edge_rows = [(5u32, 40u32), (6, 50), (7, 60), (8, 40)];
    let right_gate_rows = [5u32, 7];

    // --- SPLIT side: run the whole program through the batch adapter. ---
    let mut split_executor = Executor::new(Arc::clone(&fixture.provider));
    for component in split.recomposed_components() {
        for (name, rel) in &component.executable.relation_ids {
            split_executor.register_relation(*rel, name);
        }
    }
    split_executor.put_relation(
        "left_edge",
        upload_binary_u32(&fixture.memory, &left_edge_rows, "x", "y"),
    );
    split_executor.put_relation(
        "left_gate",
        upload_unary_u32(&fixture.memory, &left_gate_rows, "x"),
    );
    split_executor.put_relation(
        "right_edge",
        upload_binary_u32(&fixture.memory, &right_edge_rows, "x", "y"),
    );
    split_executor.put_relation(
        "right_gate",
        upload_unary_u32(&fixture.memory, &right_gate_rows, "x"),
    );
    let recomposed = split.recomposed_components();
    let executables: Vec<_> = recomposed
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = split_executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("split components should execute through GPU runtime batch adapter");
    assert_eq!(batch.results.len(), 2);
    // K4 fallback safety: split orchestration must keep CPU fallbacks at zero.
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.cpu_solver_search_fallbacks, 0);
    assert_eq!(batch.trace.cpu_probability_recomputations, 0);

    let mut split_left = fixture
        .provider
        .download_column::<u32>(&batch.results[0].final_output, 0)
        .expect("download split left output");
    let mut split_right = fixture
        .provider
        .download_column::<u32>(&batch.results[1].final_output, 0)
        .expect("download split right output");
    split_left.sort_unstable();
    split_right.sort_unstable();

    // --- UNSPLIT side: run each component's isolated subprogram through the
    // single-execution path and compare row-for-row. ---
    let unsplit_left = run_unsplit_single_component_output(
        &fixture,
        r#"
        pred left_edge(u32, u32).
        pred left_gate(u32).
        pred left_out(u32).
        left_out(Y) :- left_edge(X, Y), know left_gate(X).
        "#,
        &[("left_edge", &binary(&left_edge_rows)), ("left_gate", &unary(&left_gate_rows))],
    );
    let unsplit_right = run_unsplit_single_component_output(
        &fixture,
        r#"
        pred right_edge(u32, u32).
        pred right_gate(u32).
        pred right_out(u32).
        right_out(Y) :- right_edge(X, Y), know right_gate(X).
        "#,
        &[("right_edge", &binary(&right_edge_rows)), ("right_gate", &unary(&right_gate_rows))],
    );

    assert_eq!(
        split_left, unsplit_left,
        "split left component must equal unsplit single-execution output"
    );
    assert_eq!(
        split_right, unsplit_right,
        "split right component must equal unsplit single-execution output"
    );
    // Sanity: outputs are non-empty derived rows (not a vacuous match).
    assert!(!split_left.is_empty());
    assert!(!split_right.is_empty());

    for result in &batch.results {
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        result
            .require_runtime_dispatch_certification()
            .expect("split equivalence component evidence must remain certified");
    }
}

/// Compile a single-output epistemic subprogram through the UNSPLIT path, run it
/// on the device, and return its sorted output column. Used as the equivalence
/// reference for split-component outputs.
#[cfg(feature = "epistemic-logic-tests")]
fn run_unsplit_single_component_output(
    fixture: &RuntimeFixture,
    source: &str,
    relations: &[(&str, &[Vec<u32>])],
) -> Vec<u32> {
    let program = parse_program(source).expect("parse unsplit component subprogram");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("unsplit single-component compile must succeed");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    for (name, rows) in relations {
        let buffer = if rows.iter().all(|row| row.len() == 1) {
            let flat: Vec<u32> = rows.iter().map(|row| row[0]).collect();
            upload_unary_u32(&fixture.memory, &flat, "x")
        } else {
            let pairs: Vec<(u32, u32)> = rows.iter().map(|row| (row[0], row[1])).collect();
            upload_binary_u32(&fixture.memory, &pairs, "x", "y")
        };
        executor.put_relation(name, buffer);
    }
    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("unsplit single-component execution must succeed");
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
        .require_runtime_dispatch_certification()
        .expect("unsplit reference evidence must remain certified");
    let mut output = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download unsplit component output");
    output.sort_unstable();
    output
}

#[cfg(feature = "epistemic-logic-tests")]
fn unary(rows: &[u32]) -> Vec<Vec<u32>> {
    rows.iter().map(|value| vec![*value]).collect()
}

#[cfg(feature = "epistemic-logic-tests")]
fn binary(rows: &[(u32, u32)]) -> Vec<Vec<u32>> {
    rows.iter().map(|(a, b)| vec![*a, *b]).collect()
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_split_components_project_body_local_tuple_keys_in_gpu_batch() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred left_edge(u32, u32).
        pred left_gate(u32).
        pred left_out(u32).
        pred right_edge(u32, u32).
        pred right_blocked(u32).
        pred right_out(u32).

        left_out(Y) :- left_edge(X, Y), know left_gate(X).
        right_out(Y) :- right_edge(X, Y), not know right_blocked(X).
        "#,
    )
    .expect("parse split body-local tuple-key program");
    let split = compile_epistemic_gpu_split_execution(&program)
        .expect("compile split body-local tuple-key components");
    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);

    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for component in split.recomposed_components() {
        for (name, rel) in &component.executable.relation_ids {
            executor.register_relation(*rel, name);
        }
    }
    executor.put_relation(
        "left_edge",
        upload_binary_u32(
            &fixture.memory,
            &[(1, 10), (2, 20), (3, 30), (4, 10)],
            "x",
            "y",
        ),
    );
    executor.put_relation(
        "left_gate",
        upload_unary_u32(&fixture.memory, &[1, 3, 4], "x"),
    );
    executor.put_relation(
        "right_edge",
        upload_binary_u32(
            &fixture.memory,
            &[(5, 40), (6, 50), (7, 60), (8, 40)],
            "x",
            "y",
        ),
    );
    executor.put_relation(
        "right_blocked",
        upload_unary_u32(&fixture.memory, &[6], "x"),
    );

    let recomposed_components = split.recomposed_components();
    let executables: Vec<_> = recomposed_components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("split body-local tuple-key components should execute through GPU runtime batch");

    batch
        .require_trace_matches_components("body-local parsed epistemic GPU split execution")
        .expect("body-local split trace must be derived from real component results");
    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.cpu_solver_search_fallbacks, 0);
    assert_eq!(batch.trace.cpu_probability_recomputations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert_eq!(batch.trace.final_output_rows, 4);
    assert_eq!(batch.trace.final_output_payload_bytes, 16);
    assert_eq!(batch.trace.accepted_world_views, 2);
    assert_eq!(batch.trace.rejected_candidates, 2);
    assert_eq!(batch.trace.know_operator_count, 1);
    assert_eq!(batch.trace.not_know_operator_count, 1);
    assert_eq!(batch.trace.possible_operator_count, 0);
    assert_eq!(batch.trace.not_possible_operator_count, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let mut left_rows = fixture
        .provider
        .download_column::<u32>(&batch.results[0].final_output, 0)
        .expect("download split body-local left output values");
    let mut right_rows = fixture
        .provider
        .download_column::<u32>(&batch.results[1].final_output, 0)
        .expect("download split body-local right output values");
    left_rows.sort_unstable();
    right_rows.sort_unstable();
    assert_eq!(left_rows, vec![10, 30]);
    assert_eq!(right_rows, vec![40, 60]);

    assert_eq!(
        batch.results[0]
            .final_tuple_materialization
            .row_filter_count,
        1
    );
    assert_eq!(
        batch.results[0]
            .final_tuple_materialization
            .negated_row_filter_count,
        0
    );
    assert_eq!(
        batch.results[1]
            .final_tuple_materialization
            .row_filter_count,
        1
    );
    assert_eq!(
        batch.results[1]
            .final_tuple_materialization
            .negated_row_filter_count,
        1
    );
    for result in &batch.results {
        assert_eq!(result.output.arity(), 2);
        assert_eq!(result.final_output.arity(), 1);
        assert_eq!(result.final_result_transfer.final_output_rows, 2);
        assert_eq!(result.final_result_transfer.final_output_payload_bytes, 8);
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            1
        );
        assert_eq!(result.semantic_trace.accepted_candidates, 1);
        assert_eq!(result.semantic_trace.accepted_world_views, 1);
        assert_eq!(result.semantic_trace.rejected_candidates, 1);
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        result
            .require_runtime_dispatch_certification()
            .expect("body-local split component runtime evidence must remain certified");
    }
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_split_components_project_modal_body_local_tuple_keys_in_gpu_batch() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred left_edge(u32, u32).
        pred left_maybe(u32).
        pred left_out(u32).
        pred right_edge(u32, u32).
        pred right_maybe_blocked(u32).
        pred right_out(u32).

        left_out(Y) :- left_edge(X, Y), possible left_maybe(X).
        right_out(Y) :- right_edge(X, Y), not possible right_maybe_blocked(X).
        "#,
    )
    .expect("parse split modal body-local tuple-key program");
    let split = compile_epistemic_gpu_split_execution(&program)
        .expect("compile split modal body-local tuple-key components");
    assert_eq!(split.components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);

    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for component in split.recomposed_components() {
        for (name, rel) in &component.executable.relation_ids {
            executor.register_relation(*rel, name);
        }
    }
    executor.put_relation(
        "left_edge",
        upload_binary_u32(
            &fixture.memory,
            &[(1, 10), (2, 20), (3, 30), (4, 10)],
            "x",
            "y",
        ),
    );
    executor.put_relation(
        "left_maybe",
        upload_unary_u32(&fixture.memory, &[1, 3, 4], "x"),
    );
    executor.put_relation(
        "right_edge",
        upload_binary_u32(
            &fixture.memory,
            &[(5, 40), (6, 50), (7, 60), (8, 40)],
            "x",
            "y",
        ),
    );
    executor.put_relation(
        "right_maybe_blocked",
        upload_unary_u32(&fixture.memory, &[6], "x"),
    );

    let recomposed_components = split.recomposed_components();
    let executables: Vec<_> = recomposed_components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect(
            "split modal body-local tuple-key components should execute through GPU runtime batch",
        );

    batch
        .require_trace_matches_components("modal body-local parsed epistemic GPU split execution")
        .expect("modal body-local split trace must be derived from real component results");
    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.cpu_solver_search_fallbacks, 0);
    assert_eq!(batch.trace.cpu_probability_recomputations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert_eq!(batch.trace.final_output_rows, 4);
    assert_eq!(batch.trace.final_output_payload_bytes, 16);
    assert_eq!(batch.trace.accepted_world_views, 2);
    assert_eq!(batch.trace.rejected_candidates, 2);
    assert_eq!(batch.trace.know_operator_count, 0);
    assert_eq!(batch.trace.not_know_operator_count, 0);
    assert_eq!(batch.trace.possible_operator_count, 1);
    assert_eq!(batch.trace.not_possible_operator_count, 1);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let mut left_rows = fixture
        .provider
        .download_column::<u32>(&batch.results[0].final_output, 0)
        .expect("download split modal body-local left output values");
    let mut right_rows = fixture
        .provider
        .download_column::<u32>(&batch.results[1].final_output, 0)
        .expect("download split modal body-local right output values");
    left_rows.sort_unstable();
    right_rows.sort_unstable();
    assert_eq!(left_rows, vec![10, 30]);
    assert_eq!(right_rows, vec![40, 60]);

    assert_eq!(
        batch.results[0]
            .final_tuple_materialization
            .row_filter_count,
        1
    );
    assert_eq!(
        batch.results[0]
            .final_tuple_materialization
            .negated_row_filter_count,
        0
    );
    assert_eq!(
        batch.results[1]
            .final_tuple_materialization
            .row_filter_count,
        1
    );
    assert_eq!(
        batch.results[1]
            .final_tuple_materialization
            .negated_row_filter_count,
        1
    );
    for result in &batch.results {
        assert_eq!(result.output.arity(), 2);
        assert_eq!(result.final_output.arity(), 1);
        assert_eq!(result.final_result_transfer.final_output_rows, 2);
        assert_eq!(result.final_result_transfer.final_output_payload_bytes, 8);
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            1
        );
        assert_eq!(result.semantic_trace.accepted_candidates, 1);
        assert_eq!(result.semantic_trace.accepted_world_views, 1);
        assert_eq!(result.semantic_trace.rejected_candidates, 1);
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        result
            .require_runtime_dispatch_certification()
            .expect("modal body-local split component runtime evidence must remain certified");
    }
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_split_components_share_extensional_input_without_coalescing_runtime_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32).
        pred color(u32).
        pred a(u32).
        pred b(u32).

        a(X) :- node(X), know edge(X).
        b(X) :- node(X), know color(X).
        "#,
    )
    .expect("parse shared-extensional-input split program");
    let split =
        compile_epistemic_gpu_split_execution(&program).expect("compile shared-input split");
    let recomposed_components = split.recomposed_components();
    assert_eq!(recomposed_components.len(), 2);
    assert_eq!(split.recomposed_rule_indices(), vec![0, 1]);
    assert_eq!(recomposed_components[0].component.rule_indices, vec![0]);
    assert_eq!(recomposed_components[1].component.rule_indices, vec![1]);

    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for component in &recomposed_components {
        for (name, rel) in &component.executable.relation_ids {
            executor.register_relation(*rel, name);
        }
    }
    for (name, rows) in [
        ("node", &[1, 2, 3][..]),
        ("edge", &[1, 3][..]),
        ("color", &[2, 3][..]),
    ] {
        executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
    }

    let executables: Vec<_> = recomposed_components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 3,
            },
        )
        .expect("shared-input split components should execute through GPU runtime batch");

    batch
        .require_trace_matches_components("shared-input parsed epistemic GPU split execution")
        .expect("shared-input split trace must be derived from real component results");
    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.final_output_rows, 4);
    assert_eq!(batch.trace.accepted_world_views, 2);
    assert_eq!(batch.trace.know_operator_count, 2);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());

    let mut a_rows = fixture
        .provider
        .download_column::<u32>(&batch.results[0].final_output, 0)
        .expect("download split a output values");
    let mut b_rows = fixture
        .provider
        .download_column::<u32>(&batch.results[1].final_output, 0)
        .expect("download split b output values");
    a_rows.sort_unstable();
    b_rows.sort_unstable();
    assert_eq!(a_rows, vec![1, 3]);
    assert_eq!(b_rows, vec![2, 3]);

    let run_direct = |source: &str, inputs: &[(&str, &[u32])]| -> Vec<u32> {
        let program = parse_program(source).expect("parse direct component program");
        let executable =
            compile_epistemic_gpu_execution(&program).expect("compile direct component plan");
        let mut direct_executor = Executor::new(Arc::clone(&fixture.provider));
        for (name, rel) in &executable.relation_ids {
            direct_executor.register_relation(*rel, name);
        }
        for &(name, rows) in inputs {
            direct_executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
        }

        let result = direct_executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 2,
                    max_worlds: 2,
                    max_models_per_reduction: 3,
                },
            )
            .expect("direct component should execute through normal GPU runtime path");
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        result
            .require_runtime_dispatch_certification()
            .expect("direct component runtime evidence must remain certified");

        let mut rows = fixture
            .provider
            .download_column::<u32>(&result.final_output, 0)
            .expect("download direct component output values");
        rows.sort_unstable();
        rows
    };
    assert_eq!(
        a_rows,
        run_direct(
            r#"
            pred node(u32).
            pred edge(u32).
            pred a(u32).

            a(X) :- node(X), know edge(X).
            "#,
            &[("node", &[1, 2, 3][..]), ("edge", &[1, 3][..])]
        )
    );
    assert_eq!(
        b_rows,
        run_direct(
            r#"
            pred node(u32).
            pred color(u32).
            pred b(u32).

            b(X) :- node(X), know color(X).
            "#,
            &[("node", &[1, 2, 3][..]), ("color", &[2, 3][..])]
        )
    );

    for result in &batch.results {
        assert_eq!(result.prepared.preflight.reduced_runtime_rule_count, 1);
        assert_eq!(result.final_result_transfer.final_output_rows, 2);
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            1
        );
        assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
        assert_eq!(result.semantic_trace.accepted_candidates, 1);
        assert_eq!(result.semantic_trace.rejected_candidates, 1);
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        result
            .require_runtime_dispatch_certification()
            .expect("shared-input split component runtime evidence must remain certified");
    }
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_split_recomposes_gpu_component_values_in_source_rule_order() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred z_seed(u32).
        pred z_gate(u32).
        pred z_out(u32).
        pred a_seed(u32).
        pred a_gate(u32).
        pred a_out(u32).

        z_out(X) :- z_seed(X), know z_gate(X).
        a_out(X) :- a_seed(X), know a_gate(X).
        "#,
    )
    .expect("parse split program with source order distinct from predicate order");
    let split =
        compile_epistemic_gpu_split_execution(&program).expect("compile split GPU components");

    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for component in split.recomposed_components() {
        for (name, rel) in &component.executable.relation_ids {
            executor.register_relation(*rel, name);
        }
    }
    for (name, rows) in [
        ("z_seed", &[7][..]),
        ("z_gate", &[7][..]),
        ("a_seed", &[9][..]),
        ("a_gate", &[9][..]),
    ] {
        executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
    }

    let recomposed_components = split.recomposed_components();
    let executables: Vec<_> = recomposed_components
        .iter()
        .map(|component| &component.executable)
        .collect();
    let batch = executor
        .execute_epistemic_gpu_execution_batch_with_trace(
            &executables,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("source-ordered split components should execute through GPU runtime batch");

    batch
        .require_trace_matches_components("source-ordered parsed epistemic GPU split execution")
        .expect("source-ordered split trace must be derived from real component results");
    assert_eq!(batch.results.len(), 2);
    assert_eq!(batch.trace.component_count, 2);
    assert_eq!(batch.trace.gpu_runtime_component_executions, 2);
    assert_eq!(batch.trace.cpu_recomposition_steps, 0);
    assert_eq!(batch.trace.cpu_candidate_enumerations, 0);
    assert_eq!(batch.trace.cpu_world_view_validations, 0);
    assert_eq!(batch.trace.cpu_solver_search_fallbacks, 0);
    assert_eq!(batch.trace.cpu_probability_recomputations, 0);
    assert_eq!(batch.trace.tracked_dtoh_calls, 0);
    assert_eq!(batch.trace.per_candidate_host_round_trips, 0);
    assert_eq!(batch.trace.final_output_rows, 2);
    assert_eq!(batch.trace.final_output_payload_bytes, 8);
    assert_eq!(batch.trace.accepted_world_views, 2);
    assert_eq!(batch.trace.rejected_candidates, 2);
    assert_eq!(batch.trace.know_operator_count, 2);
    assert_eq!(batch.trace.possible_operator_count, 0);
    assert_eq!(batch.trace.not_know_operator_count, 0);
    assert_eq!(batch.trace.not_possible_operator_count, 0);
    assert!(batch.trace.aggregate_kernel_timing.is_recorded());
    assert_eq!(recomposed_components[0].component.rule_indices, vec![0]);
    assert_eq!(recomposed_components[1].component.rule_indices, vec![1]);

    let z_rows = fixture
        .provider
        .download_column::<u32>(&batch.results[0].final_output, 0)
        .expect("download recomposed z_out values");
    let a_rows = fixture
        .provider
        .download_column::<u32>(&batch.results[1].final_output, 0)
        .expect("download recomposed a_out values");
    assert_eq!(z_rows, vec![7]);
    assert_eq!(a_rows, vec![9]);

    for result in &batch.results {
        assert_eq!(result.prepared.preflight.reduced_runtime_rule_count, 1);
        assert_eq!(result.final_result_transfer.final_output_rows, 1);
        assert_eq!(result.final_result_transfer.final_output_payload_bytes, 4);
        assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
        assert_eq!(result.semantic_trace.accepted_candidates, 1);
        assert_eq!(result.semantic_trace.accepted_world_views, 1);
        assert_eq!(result.semantic_trace.rejected_candidate_indices, vec![0]);
        assert_eq!(result.semantic_trace.rejected_candidates, 1);
        assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
        assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
        assert!(result.aggregate_kernel_timing().is_recorded());
        result
            .require_runtime_dispatch_certification()
            .expect("source-ordered split component runtime evidence must remain certified");
    }
}

#[test]
fn runtime_preflight_rejects_helper_split_specs_without_production_helper_rewrite() {
    let executable = executable_with_kclique_helper_metadata_only_plan();

    let err = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .expect_err("helper-split metadata alone must not certify production helper reuse");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU helper-split certification");
            assert!(context.contains("helper_split_specs=1"));
            assert!(context.contains("helper_relation_rules=0"));
            assert!(context.contains("helper_relation_scans=0"));
        }
        other => panic!("expected helper-split certification error, got {other:?}"),
    }
}

#[test]
fn runtime_preflight_rejects_helper_relation_scan_outside_wcoj_plan() {
    let executable = executable_with_kclique_helper_scan_outside_wcoj_plan();

    let err = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .expect_err("helper relation scans must be consumed by the production WCOJ plan");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU helper-split certification");
            assert!(context.contains("helper_split_specs=1"));
            assert!(context.contains("helper_relation_rules=1"));
            assert!(context.contains("helper_relation_scans=0"));
        }
        other => panic!("expected helper-split certification error, got {other:?}"),
    }
}

#[test]
fn runtime_preflight_rejects_kclique_plan_without_edge_permutation() {
    let executable = executable_with_kclique_empty_edge_permutation_plan();

    let err = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .expect_err("K-clique WCOJ plans require live edge-permutation slots");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU K-clique WCOJ certification");
            assert!(context.contains("kclique_plans=1"));
            assert!(context.contains("edge_permutation_slots=0"));
        }
        other => panic!("expected K-clique WCOJ certification error, got {other:?}"),
    }
}

#[test]
fn runtime_preflight_records_epistemic_operator_metrics() {
    let executable = executable_with_operator_mix();

    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 2,
        },
    )
    .unwrap();

    assert_eq!(preflight.know_operator_count, 1);
    assert_eq!(preflight.possible_operator_count, 1);
    assert_eq!(preflight.not_know_operator_count, 1);
    assert_eq!(preflight.not_possible_operator_count, 1);
}

#[test]
fn runtime_execution_validates_epistemic_operator_mix_on_gpu_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    executor.register_relation(RelId(1), "seed");
    executor.put_relation("seed", upload_zero_arity(&fixture.memory, 1));
    for (rel, name, rows) in [
        (RelId(20), "known_gate", 1),
        (RelId(21), "possible_gate", 1),
        (RelId(22), "not_known_gate", 0),
        (RelId(23), "not_possible_gate", 0),
    ] {
        executor.register_relation(rel, name);
        executor.put_relation(name, upload_zero_arity(&fixture.memory, rows));
    }

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable_with_operator_mix(),
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 16,
                max_worlds: 4,
                max_models_per_reduction: 1,
            },
        )
        .expect("operator mix should execute through GPU world-view validation");

    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(
        result.model_membership.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(
        result.model_membership.tuple_source_row_count_device_reads,
        4
    );
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        0
    );
    assert_eq!(
        result.world_view_validation.model_membership_bytes_checked,
        64
    );
    assert_eq!(result.semantic_trace.generated_candidates, 16);
    assert_eq!(result.semantic_trace.guesses, 64);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![15]);
    assert_eq!(result.semantic_trace.rejected_candidates, 15);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    result
        .require_runtime_dispatch_certification()
        .expect("operator mix runtime path should retain semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn world_view_validation_kernel_distinguishes_know_all_from_possible_any_model() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred gate().
        pred out().
        gate().
        out() :- know gate(), possible gate().
        "#,
    )
    .expect("parse unbound modal operator program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile unbound modal GPU plan");
    let executor = Executor::new(Arc::clone(&fixture.provider));
    let candidate_count = 4;
    let models_per_reduction = 2;
    let literal_count = executable.gpu_plan.epistemic_literals.len();
    let reduction_count = executable.gpu_plan.reductions.len();
    assert_eq!(literal_count, 2);
    assert_eq!(reduction_count, 1);

    let mut workspace = executor
        .allocate_epistemic_gpu_workspace(
            &executable.gpu_plan,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: candidate_count,
                max_worlds: 2,
                max_models_per_reduction: models_per_reduction,
            },
        )
        .expect("allocate world-view validation workspace");
    executor
        .reset_epistemic_gpu_workspace(&mut workspace)
        .expect("reset world-view validation workspace");
    executor
        .generate_epistemic_gpu_candidates(&mut workspace, literal_count, candidate_count)
        .expect("generate modal candidates on GPU");
    executor
        .propagate_epistemic_gpu_candidates(&mut workspace, literal_count, candidate_count)
        .expect("propagate modal candidates on GPU");

    let mut membership = vec![0u8; workspace.layout.model_membership_bytes];
    for candidate in 0..candidate_count {
        for model in 0..models_per_reduction {
            for literal in 0..literal_count {
                let idx = (((candidate * reduction_count) * models_per_reduction + model)
                    * literal_count)
                    + literal;
                let active_model = 2u8;
                let literal_present = u8::from(model == 0);
                membership[idx] = active_model | literal_present;
            }
        }
    }
    fixture
        .provider
        .device()
        .inner()
        .htod_sync_copy_into(&membership, &mut workspace.model_membership)
        .expect("upload synthetic two-model membership state");

    let trace = executor
        .validate_epistemic_gpu_world_views(
            &mut workspace,
            &executable.gpu_plan,
            candidate_count,
            models_per_reduction,
        )
        .expect("validate multi-model modal assumptions on GPU");
    assert_eq!(trace.literal_count, literal_count);
    assert_eq!(trace.candidates_checked, candidate_count);
    assert_eq!(trace.reduction_count, reduction_count);
    assert_eq!(trace.models_per_reduction, models_per_reduction);
    assert_eq!(trace.model_membership_bytes_checked, 16);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(trace.kernel_timing.is_recorded());

    let rejection_reasons = fixture
        .provider
        .dtoh_small_metadata_untracked(&workspace.rejection_reasons, candidate_count)
        .expect("download bounded rejection metadata");
    assert_eq!(rejection_reasons, vec![5, 5, 0, 5]);
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn world_view_validation_kernel_accepts_not_known_and_not_possible_duals_across_models() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred partial().
        pred absent().
        pred out().
        partial().
        out() :- not know partial(), not possible absent().
        "#,
    )
    .expect("parse negated modal operator program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile negated modal GPU plan");
    let executor = Executor::new(Arc::clone(&fixture.provider));
    let candidate_count = 4;
    let models_per_reduction = 2;
    let literal_count = executable.gpu_plan.epistemic_literals.len();
    let reduction_count = executable.gpu_plan.reductions.len();
    assert_eq!(literal_count, 2);
    assert_eq!(reduction_count, 1);

    let mut workspace = executor
        .allocate_epistemic_gpu_workspace(
            &executable.gpu_plan,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: candidate_count,
                max_worlds: 2,
                max_models_per_reduction: models_per_reduction,
            },
        )
        .expect("allocate negated world-view validation workspace");
    executor
        .reset_epistemic_gpu_workspace(&mut workspace)
        .expect("reset negated world-view validation workspace");
    executor
        .generate_epistemic_gpu_candidates(&mut workspace, literal_count, candidate_count)
        .expect("generate negated modal candidates on GPU");
    executor
        .propagate_epistemic_gpu_candidates(&mut workspace, literal_count, candidate_count)
        .expect("propagate negated modal candidates on GPU");

    let mut membership = vec![0u8; workspace.layout.model_membership_bytes];
    for candidate in 0..candidate_count {
        for model in 0..models_per_reduction {
            let partial_idx =
                ((candidate * reduction_count) * models_per_reduction + model) * literal_count;
            let absent_idx = partial_idx + 1;
            let active_model = 2u8;
            membership[partial_idx] = active_model | u8::from(model == 0);
            membership[absent_idx] = active_model;
        }
    }
    fixture
        .provider
        .device()
        .inner()
        .htod_sync_copy_into(&membership, &mut workspace.model_membership)
        .expect("upload synthetic negated two-model membership state");

    let trace = executor
        .validate_epistemic_gpu_world_views(
            &mut workspace,
            &executable.gpu_plan,
            candidate_count,
            models_per_reduction,
        )
        .expect("validate negated multi-model modal assumptions on GPU");
    assert_eq!(trace.literal_count, literal_count);
    assert_eq!(trace.candidates_checked, candidate_count);
    assert_eq!(trace.reduction_count, reduction_count);
    assert_eq!(trace.models_per_reduction, models_per_reduction);
    assert_eq!(trace.model_membership_bytes_checked, 16);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(trace.kernel_timing.is_recorded());

    let rejection_reasons = fixture
        .provider
        .dtoh_small_metadata_untracked(&workspace.rejection_reasons, candidate_count)
        .expect("download bounded negated rejection metadata");
    assert_eq!(rejection_reasons, vec![5, 5, 5, 0]);
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn runtime_execution_rejects_candidate_capacity_below_epistemic_guess_space() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred a().
        pred b().
        pred c().
        pred out().
        out() :- know a(), possible b(), not possible c().
        "#,
    )
    .expect("parse candidate-capacity guard program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile candidate-capacity GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    let err = match executor.execute_epistemic_gpu_execution(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 4,
            max_worlds: 2,
            max_models_per_reduction: 1,
        },
    ) {
        Ok(_) => {
            panic!("runtime must reject insufficient candidate capacity before GPU generation")
        }
        Err(err) => err,
    };

    match err {
        xlog_core::XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } => {
            assert_eq!(context, "epistemic GPU execution candidate capacity");
            assert_eq!(estimated_bytes, 8);
            assert_eq!(budget_bytes, 4);
        }
        other => panic!("expected runtime candidate-capacity guard error, got {other:?}"),
    }
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_epistemic_program_executes_compiled_gpu_plan_on_runtime_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32).
        pred known_gate(u32).
        pred possible_gate(u32).
        pred not_known_gate(u32).
        pred not_possible_gate(u32).
        pred out(u32).

        out(X) :- seed(X), know known_gate(X), possible possible_gate(X),
                  not know not_known_gate(X), not possible not_possible_gate(X).
        "#,
    )
    .expect("parse all-operator epistemic program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile parsed epistemic GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rows) in [
        ("seed", &[7][..]),
        ("known_gate", &[7][..]),
        ("possible_gate", &[7][..]),
        ("not_known_gate", &[8][..]),
        ("not_possible_gate", &[9][..]),
    ] {
        let rel = *executable
            .relation_ids
            .get(name)
            .unwrap_or_else(|| panic!("compiled plan should expose relation id for {name}"));
        executor.register_relation(rel, name);
        executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
    }

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 16,
                max_worlds: 4,
                max_models_per_reduction: 1,
            },
        )
        .expect("compiled all-operator program should execute on GPU");

    assert_eq!(result.prepared.preflight.reduced_runtime_rule_count, 1);
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 4);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        2
    );
    assert_eq!(result.semantic_trace.generated_candidates, 16);
    assert_eq!(result.semantic_trace.guesses, 64);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![15]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 15);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    result
        .require_runtime_dispatch_certification()
        .expect("compiled runtime path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_negated_epistemic_operators_filter_distinct_gpu_tuple_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32).
        pred known_gate(u32).
        pred possible_gate(u32).
        pred not_known_gate(u32).
        pred not_possible_gate(u32).
        pred out(u32).

        out(X) :- seed(X), know known_gate(X), possible possible_gate(X),
                  not know not_known_gate(X), not possible not_possible_gate(X).
        "#,
    )
    .expect("parse negated-operator row-filter program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile negated-operator GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rows) in [
        ("seed", &[7, 8, 9][..]),
        ("known_gate", &[7, 8, 9][..]),
        ("possible_gate", &[7, 8, 9][..]),
        ("not_known_gate", &[7][..]),
        ("not_possible_gate", &[8][..]),
    ] {
        let rel = *executable
            .relation_ids
            .get(name)
            .unwrap_or_else(|| panic!("compiled plan should expose relation id for {name}"));
        executor.register_relation(rel, name);
        executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
    }

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 16,
                max_worlds: 4,
                max_models_per_reduction: 3,
            },
        )
        .expect("negated epistemic operators should filter GPU tuple values");

    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 4);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        2
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final negated-operator output values");
    assert_eq!(rows, vec![9]);
    result
        .require_runtime_dispatch_certification()
        .expect("negated-operator runtime path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_g91_self_supported_possible_executes_on_gpu_runtime_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred seed(u32).
        pred p(u32).

        p(X) :- seed(X), possible p(X).
        "#,
    )
    .expect("parse source-annotated G91 self-supported program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile G91 epistemic GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("seed", upload_unary_u32(&fixture.memory, &[7], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("G91 self-supported possible should execute through GPU runtime");

    assert!(result.prepared.preflight.is_g91_mode());
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidate_indices, vec![0]);
    assert_eq!(
        result
            .semantic_trace
            .typed_rejection_reasons()
            .expect("decode typed GPU G91 rejection reason"),
        vec![EpistemicGpuRejectionReason::UnsatisfiedMembership]
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final G91 output values");
    assert_eq!(rows, vec![7]);
    result
        .require_runtime_dispatch_certification()
        .expect("G91 runtime path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_faeel_tuple_founded_possible_executes_on_gpu_runtime_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32).
        pred p(u32).

        p(S) :- seed(S).
        p(X) :- seed(X), possible p(X).
        "#,
    )
    .expect("parse tuple-founded FAEEL possible program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile tuple-founded FAEEL GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("seed", upload_unary_u32(&fixture.memory, &[7], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("tuple-founded FAEEL possible should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final FAEEL tuple-founded output values");
    assert_eq!(rows, vec![7]);
    result
        .require_runtime_dispatch_certification()
        .expect("tuple-founded FAEEL runtime path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_faeel_ground_tuple_founded_possible_executes_on_gpu_runtime_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32).
        pred p(u32).

        p(S) :- seed(S).
        p(7) :- seed(7), possible p(7).
        "#,
    )
    .expect("parse ground tuple-founded FAEEL possible program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile ground tuple-founded FAEEL GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("seed", upload_unary_u32(&fixture.memory, &[7], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("ground tuple-founded FAEEL possible should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(
        executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
        vec![None]
    );
    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final ground FAEEL tuple-founded output values");
    assert_eq!(rows, vec![7]);
    result
        .require_runtime_dispatch_certification()
        .expect("ground tuple-founded FAEEL runtime path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_faeel_ground_possible_with_variable_headed_support_executes_on_gpu_runtime_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32).
        pred p(u32).

        p(S) :- seed(S).
        p(7) :- possible p(7).
        "#,
    )
    .expect("parse ground FAEEL possible program with variable-headed support");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile ground FAEEL possible program with variable-headed support");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("seed", upload_unary_u32(&fixture.memory, &[7], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("ground FAEEL possible with variable-headed support should execute on GPU");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(
        executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
        vec![None]
    );
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final ground FAEEL variable-head support output values");
    assert_eq!(rows, vec![7]);
    result
        .require_runtime_dispatch_certification()
        .expect("ground FAEEL variable-head support runtime path should retain GPU certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_ground_symbol_tuple_key_executes_on_gpu_runtime_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(symbol).
        pred gate(symbol).
        pred out(symbol).

        out(X) :- seed(X), know gate(label).
        "#,
    )
    .expect("parse ground symbol tuple-key program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile symbol tuple-key GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    let alpha = xlog_core::symbol::intern("alpha");
    let beta = xlog_core::symbol::intern("beta");
    let label = xlog_core::symbol::intern("label");
    executor.put_relation(
        "seed",
        upload_unary_typed_u32(&fixture.memory, &[alpha, beta], "x", ScalarType::Symbol),
    );
    executor.put_relation(
        "gate",
        upload_unary_typed_u32(&fixture.memory, &[label], "x", ScalarType::Symbol),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("ground symbol tuple key should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(
        executable.gpu_plan.tuple_membership_bindings[0].bound_output_columns,
        vec![None]
    );
    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(
        result.final_output.schema().column_type(0),
        Some(ScalarType::Symbol)
    );
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    let mut rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final symbol output values");
    rows.sort_unstable();
    assert_eq!(rows, vec![alpha, beta]);
    result
        .require_runtime_dispatch_certification()
        .expect("symbol tuple-key runtime path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_bound_not_possible_filters_final_gpu_tuple_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32).
        pred blocked(u32).
        pred out(u32).

        out(X) :- seed(X), not possible blocked(X).
        "#,
    )
    .expect("parse bound not-possible row-filter program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile not-possible GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("seed", upload_unary_u32(&fixture.memory, &[7, 8], "x"));
    executor.put_relation("blocked", upload_unary_u32(&fixture.memory, &[7], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("bound not-possible program should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(result.candidate_generation.generated_candidates, 2);
    assert_eq!(result.propagation.propagated_candidates, 2);
    assert_eq!(result.world_view_validation.candidates_checked, 2);
    assert_eq!(result.candidate_generation.kernel_launches, 1);
    assert_eq!(result.propagation.kernel_launches, 1);
    assert_eq!(result.candidate_validation.kernel_launches, 1);
    assert_eq!(result.model_membership.kernel_launches, 1);
    assert_eq!(result.world_view_validation.kernel_launches, 1);
    assert_eq!(result.materialization.kernel_launches, 1);
    assert_eq!(result.final_result_materialization.kernel_launches, 1);
    assert!(result.final_tuple_materialization.kernel_launches > 0);
    assert!(result.aggregate_kernel_timing().is_recorded());
    assert_eq!(result.semantic_trace.generated_candidates, 2);
    assert_eq!(result.semantic_trace.guesses, 2);
    assert_eq!(result.semantic_trace.propagated_candidates, 2);
    assert_eq!(result.semantic_trace.pruned_candidates, 0);
    assert_eq!(result.semantic_trace.tested_candidates, 2);
    assert_eq!(result.semantic_trace.reduced_model_slots_checked, 4);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.rejected_candidate_indices, vec![0]);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(
        result
            .semantic_trace
            .typed_rejection_reasons()
            .expect("decode typed GPU not-possible rejection reason"),
        vec![EpistemicGpuRejectionReason::UnsatisfiedMembership]
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final not-possible output values");
    assert_eq!(rows, vec![8]);
    result
        .require_runtime_dispatch_certification()
        .expect("bound not-possible row-filter path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_bound_not_know_filters_final_gpu_tuple_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32).
        pred known(u32).
        pred out(u32).

        out(X) :- seed(X), not know known(X).
        "#,
    )
    .expect("parse bound not-know row-filter program");
    let executable = compile_epistemic_gpu_execution(&program).expect("compile not-know GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("seed", upload_unary_u32(&fixture.memory, &[7, 8], "x"));
    executor.put_relation("known", upload_unary_u32(&fixture.memory, &[7], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("bound not-know program should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(result.candidate_generation.generated_candidates, 2);
    assert_eq!(result.propagation.propagated_candidates, 2);
    assert_eq!(result.world_view_validation.candidates_checked, 2);
    assert_eq!(result.candidate_generation.kernel_launches, 1);
    assert_eq!(result.propagation.kernel_launches, 1);
    assert_eq!(result.candidate_validation.kernel_launches, 1);
    assert_eq!(result.model_membership.kernel_launches, 1);
    assert_eq!(result.world_view_validation.kernel_launches, 1);
    assert_eq!(result.materialization.kernel_launches, 1);
    assert_eq!(result.final_result_materialization.kernel_launches, 1);
    assert!(result.final_tuple_materialization.kernel_launches > 0);
    assert!(result.aggregate_kernel_timing().is_recorded());
    assert_eq!(result.semantic_trace.generated_candidates, 2);
    assert_eq!(result.semantic_trace.guesses, 2);
    assert_eq!(result.semantic_trace.propagated_candidates, 2);
    assert_eq!(result.semantic_trace.pruned_candidates, 0);
    assert_eq!(result.semantic_trace.tested_candidates, 2);
    assert_eq!(result.semantic_trace.reduced_model_slots_checked, 4);
    assert_eq!(result.semantic_trace.accepted_candidate_indices, vec![1]);
    assert_eq!(result.semantic_trace.accepted_candidates, 1);
    assert_eq!(result.semantic_trace.accepted_world_views, 1);
    assert_eq!(result.semantic_trace.rejected_candidate_indices, vec![0]);
    assert_eq!(result.semantic_trace.rejected_candidates, 1);
    assert_eq!(
        result
            .semantic_trace
            .typed_rejection_reasons()
            .expect("decode typed GPU not-know rejection reason"),
        vec![EpistemicGpuRejectionReason::UnsatisfiedMembership]
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final not-know output values");
    assert_eq!(rows, vec![8]);
    result
        .require_runtime_dispatch_certification()
        .expect("bound not-know row-filter path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_bound_binary_know_filters_final_gpu_tuple_values_beyond_model_slot_window() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32, u32).
        pred gate(u32, u32).
        pred out(u32, u32).

        out(X, Y) :- seed(X, Y), know gate(X, Y).
        "#,
    )
    .expect("parse bound binary know row-filter program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile binary know GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "seed",
        upload_binary_u32(&fixture.memory, &[(1, 2), (1, 3), (2, 4)], "x", "y"),
    );
    executor.put_relation(
        "gate",
        upload_binary_u32(&fixture.memory, &[(1, 2), (2, 4)], "x", "y"),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 1,
            },
        )
        .expect("bound binary know program should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 2);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        2
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(
        result
            .final_tuple_materialization
            .row_specific_membership_row_capacity,
        1
    );
    assert_eq!(
        result
            .final_tuple_materialization
            .row_filter_row_capacity_outside_model_slot_window,
        2
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let xs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final binary know x values");
    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download final binary know y values");
    let mut rows: Vec<_> = xs.into_iter().zip(ys).collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 2), (2, 4)]);
    result
        .require_runtime_dispatch_certification()
        .expect("bound binary know row-filter path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_body_local_tuple_key_filters_before_public_projection_on_gpu() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred edge(u32, u32).
        pred source(u32).
        pred out(u32).

        out(Y) :- edge(X, Y), know source(X).
        "#,
    )
    .expect("parse body-local tuple-key program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile body-local tuple-key GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "edge",
        upload_binary_u32(
            &fixture.memory,
            &[(1, 10), (2, 20), (3, 30), (4, 10)],
            "x",
            "y",
        ),
    );
    executor.put_relation("source", upload_unary_u32(&fixture.memory, &[1, 3, 4], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("body-local tuple-key program should execute through GPU runtime");

    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final body-local output values");
    let mut rows = ys;
    rows.sort_unstable();
    assert_eq!(rows, vec![10, 30]);
    result
        .require_runtime_dispatch_certification()
        .expect("body-local tuple-key path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_body_local_possible_tuple_key_filters_before_public_projection_on_gpu() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred edge(u32, u32).
        pred source(u32).
        pred out(u32).

        out(Y) :- edge(X, Y), possible source(X).
        "#,
    )
    .expect("parse body-local possible tuple-key program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile body-local possible tuple-key GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "edge",
        upload_binary_u32(
            &fixture.memory,
            &[(1, 10), (2, 20), (3, 30), (4, 10)],
            "x",
            "y",
        ),
    );
    executor.put_relation("source", upload_unary_u32(&fixture.memory, &[1, 3, 4], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("body-local possible tuple-key program should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final body-local possible output values");
    let mut rows = ys;
    rows.sort_unstable();
    assert_eq!(rows, vec![10, 30]);
    result
        .require_runtime_dispatch_certification()
        .expect("body-local possible tuple-key path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_body_local_negated_tuple_key_filters_before_public_projection_on_gpu() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred edge(u32, u32).
        pred blocked(u32).
        pred out(u32).

        out(Y) :- edge(X, Y), not know blocked(X).
        "#,
    )
    .expect("parse body-local negated tuple-key program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile body-local negated tuple-key GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "edge",
        upload_binary_u32(
            &fixture.memory,
            &[(1, 10), (2, 20), (3, 30), (4, 10)],
            "x",
            "y",
        ),
    );
    executor.put_relation("blocked", upload_unary_u32(&fixture.memory, &[2], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("body-local negated tuple-key program should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.not_know_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final body-local negated output values");
    let mut rows = ys;
    rows.sort_unstable();
    assert_eq!(rows, vec![10, 30]);
    result
        .require_runtime_dispatch_certification()
        .expect("body-local negated tuple-key path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_body_local_not_possible_tuple_key_filters_before_public_projection_on_gpu() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred edge(u32, u32).
        pred blocked(u32).
        pred out(u32).

        out(Y) :- edge(X, Y), not possible blocked(X).
        "#,
    )
    .expect("parse body-local not-possible tuple-key program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile body-local not-possible tuple-key GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "edge",
        upload_binary_u32(
            &fixture.memory,
            &[(1, 10), (2, 20), (3, 30), (4, 10)],
            "x",
            "y",
        ),
    );
    executor.put_relation("blocked", upload_unary_u32(&fixture.memory, &[2], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 2,
            },
        )
        .expect("body-local not-possible tuple-key program should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.not_possible_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 2);
    assert_eq!(result.final_output.arity(), 1);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        1
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final body-local not-possible output values");
    let mut rows = ys;
    rows.sort_unstable();
    assert_eq!(rows, vec![10, 30]);
    result
        .require_runtime_dispatch_certification()
        .expect("body-local not-possible tuple-key path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_zero_arity_head_with_body_local_tuple_key_projects_empty_public_tuple_on_gpu() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred edge(u32).
        pred source(u32).
        pred out().

        out() :- edge(X), know source(X).
        "#,
    )
    .expect("parse zero-arity body-local tuple-key program");
    let executable = compile_epistemic_gpu_execution(&program)
        .expect("compile zero-arity body-local tuple-key GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("edge", upload_unary_u32(&fixture.memory, &[1, 2, 3], "x"));
    executor.put_relation("source", upload_unary_u32(&fixture.memory, &[1, 3], "x"));

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 3,
            },
        )
        .expect("zero-arity body-local tuple-key program should execute through GPU runtime");

    assert_eq!(result.output.arity(), 1);
    assert_eq!(result.final_output.arity(), 0);
    assert_eq!(result.final_result_transfer.final_output_rows, 1);
    assert_eq!(result.final_result_transfer.final_output_payload_bytes, 0);
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        1
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
        .require_runtime_dispatch_certification()
        .expect("zero-arity projection path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_bound_ternary_know_filters_final_gpu_tuple_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32, u32, u32).
        pred gate(u32, u32, u32).
        pred out(u32, u32, u32).

        out(X, Y, Z) :- seed(X, Y, Z), know gate(X, Y, Z).
        "#,
    )
    .expect("parse bound ternary know row-filter program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile ternary know GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "seed",
        upload_ternary_u32(
            &fixture.memory,
            &[(1, 2, 3), (1, 2, 4), (2, 3, 5), (9, 9, 9)],
            "x",
            "y",
            "z",
        ),
    );
    executor.put_relation(
        "gate",
        upload_ternary_u32(&fixture.memory, &[(1, 2, 3), (2, 3, 5)], "x", "y", "z"),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 4,
            },
        )
        .expect("bound ternary know program should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 3);
    assert_eq!(result.final_output.arity(), 3);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        3
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let xs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final ternary know x values");
    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download final ternary know y values");
    let zs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 2)
        .expect("download final ternary know z values");
    let mut rows: Vec<_> = xs
        .into_iter()
        .zip(ys)
        .zip(zs)
        .map(|((x, y), z)| (x, y, z))
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 2, 3), (2, 3, 5)]);
    result
        .require_runtime_dispatch_certification()
        .expect("bound ternary know row-filter path should retain GPU semantic certification");
}

#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn parsed_bound_quaternary_know_filters_final_gpu_tuple_values() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred seed(u32, u32, u32, u32).
        pred gate(u32, u32, u32, u32).
        pred out(u32, u32, u32, u32).

        out(W, X, Y, Z) :- seed(W, X, Y, Z), know gate(W, X, Y, Z).
        "#,
    )
    .expect("parse bound quaternary know row-filter program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile quaternary know GPU plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));

    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "seed",
        upload_quaternary_u32(
            &fixture.memory,
            &[(1, 2, 3, 4), (1, 2, 3, 5), (2, 3, 5, 8), (9, 9, 9, 9)],
            "w",
            "x",
            "y",
            "z",
        ),
    );
    executor.put_relation(
        "gate",
        upload_quaternary_u32(
            &fixture.memory,
            &[(1, 2, 3, 4), (2, 3, 5, 8)],
            "w",
            "x",
            "y",
            "z",
        ),
    );

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 2,
                max_worlds: 2,
                max_models_per_reduction: 4,
            },
        )
        .expect("bound quaternary know program should execute through GPU runtime");

    assert!(result.prepared.preflight.is_faeel_mode());
    assert_eq!(result.prepared.preflight.know_operator_count, 1);
    assert_eq!(result.prepared.preflight.tuple_membership_binding_count, 1);
    assert_eq!(result.output.arity(), 4);
    assert_eq!(result.final_output.arity(), 4);
    assert_eq!(result.final_result_transfer.final_output_rows, 2);
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        4
    );
    assert_eq!(result.final_tuple_materialization.row_filter_count, 1);
    assert_eq!(
        result.final_tuple_materialization.negated_row_filter_count,
        0
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);

    let ws = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download final quaternary know w values");
    let xs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download final quaternary know x values");
    let ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 2)
        .expect("download final quaternary know y values");
    let zs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 3)
        .expect("download final quaternary know z values");
    let mut rows: Vec<_> = ws
        .into_iter()
        .zip(xs)
        .zip(ys)
        .zip(zs)
        .map(|(((w, x), y), z)| (w, x, y, z))
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 2, 3, 4), (2, 3, 5, 8)]);
    result
        .require_runtime_dispatch_certification()
        .expect("bound quaternary know row-filter path should retain GPU semantic certification");
}

#[test]
fn runtime_preflight_rejects_nonzero_cpu_fallback_counters() {
    let cases: [(&str, fn(&mut EpistemicCpuFallbackCounters)); 4] = [
        ("candidate enumeration", |counters| {
            counters.candidate_enumeration = 1
        }),
        ("world-view validation", |counters| {
            counters.world_view_validation = 1
        }),
        ("solver search", |counters| counters.solver_search = 1),
        ("probabilistic recompute", |counters| {
            counters.probabilistic_recompute = 1
        }),
    ];

    for (label, set_counter) in cases {
        let mut executable = executable_with_kclique_wcoj_plan();
        set_counter(&mut executable.gpu_plan.cpu_fallbacks);

        let err = EpistemicGpuRuntimePreflight::for_executable_plan(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 8,
                max_worlds: 4,
                max_models_per_reduction: 6,
            },
        )
        .expect_err("preflight must reject every nonzero CPU fallback counter");

        match err {
            xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
                assert_eq!(construct, "epistemic GPU runtime preflight");
                assert!(
                    context.contains("nonzero CPU fallback counters"),
                    "missing fallback context for {label}: {context}"
                );
            }
            other => panic!("expected typed fallback counter error for {label}, got {other:?}"),
        }
    }
}

#[test]
fn runtime_preflight_rejects_missing_tuple_membership_bindings() {
    let mut executable = executable_with_kclique_wcoj_plan();
    executable.gpu_plan.tuple_membership_bindings.clear();

    let err = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .expect_err("preflight must fail closed without tuple-membership bindings");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU tuple membership binding");
            assert!(context.contains("expected 1 bindings"));
            assert!(context.contains("found 0"));
        }
        other => panic!("expected tuple-membership binding error, got {other:?}"),
    }
}

#[test]
fn runtime_wcoj_certification_rejects_preflight_only_metadata() {
    let executable = executable_with_kclique_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    let before = EpistemicGpuRuntimeCounters::default();
    let after = EpistemicGpuRuntimeCounters::default();
    let delta = after.saturating_delta_since(before);

    assert_eq!(
        EpistemicGpuRuntimeWcojCertification::for_preflight_and_delta(&preflight, &delta),
        EpistemicGpuRuntimeWcojCertification::MissingRequiredWcojDispatch {
            required_multiway_reductions: 1,
            required_kclique_plans: 1,
            observed_wcoj_dispatches: 0,
            observed_kclique_dispatches: 0,
        }
    );
}

#[test]
fn runtime_wcoj_certification_rejects_v070_multiway_without_dispatch() {
    let executable = executable_with_v070_4cycle_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    assert_eq!(preflight.multiway_reduction_count, 1);
    assert_eq!(preflight.kclique_wcoj_plan_count, 0);

    let before = EpistemicGpuRuntimeCounters::default();
    let after = EpistemicGpuRuntimeCounters::default();
    let delta = after.saturating_delta_since(before);

    assert_eq!(
        EpistemicGpuRuntimeWcojCertification::for_preflight_and_delta(&preflight, &delta),
        EpistemicGpuRuntimeWcojCertification::MissingRequiredWcojDispatch {
            required_multiway_reductions: 1,
            required_kclique_plans: 0,
            observed_wcoj_dispatches: 0,
            observed_kclique_dispatches: 0,
        }
    );
}

#[test]
fn runtime_wcoj_certification_accepts_actual_kclique_dispatch_delta() {
    let executable = executable_with_kclique_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    let before = EpistemicGpuRuntimeCounters::default();
    let after = EpistemicGpuRuntimeCounters {
        wcoj_clique5_dispatch_count: 1,
        kclique_metadata_build_count: 1,
        kclique_metadata_build_nanos: 42,
        wcoj_layout_sort_invocation_count: 2,
        ..EpistemicGpuRuntimeCounters::default()
    };
    let delta = after.saturating_delta_since(before);

    assert_eq!(
        EpistemicGpuRuntimeWcojCertification::for_preflight_and_delta(&preflight, &delta),
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1,
            certified_multiway_reductions: 1,
            observed_kclique_dispatches: 1,
            certified_edge_permutation_slots: 10,
            certified_stream_groups: 1,
            certified_skew_scheduled_plans: 1,
            certified_sorted_layout_requirements: 2,
            certified_helper_split_specs: 1,
            certified_helper_relation_rules: 1,
            certified_helper_relation_scans: 1,
            observed_layout_sorts: 2,
            observed_layout_fast_path_hits: 0,
            observed_metadata_builds: 1,
            observed_metadata_build_nanos: 42,
            observed_histogram_refreshes: 0,
            observed_histogram_refresh_nanos: 0,
        }
    );
}

#[test]
fn runtime_wcoj_certification_accepts_layout_fast_path_evidence() {
    let executable = executable_with_kclique_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    let before = EpistemicGpuRuntimeCounters::default();
    let after = EpistemicGpuRuntimeCounters {
        wcoj_clique5_dispatch_count: 1,
        wcoj_layout_fast_path_hit_count: 2,
        kclique_metadata_build_count: 1,
        kclique_metadata_build_nanos: 42,
        ..EpistemicGpuRuntimeCounters::default()
    };
    let delta = after.saturating_delta_since(before);

    assert_eq!(
        EpistemicGpuRuntimeWcojCertification::for_preflight_and_delta(&preflight, &delta),
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1,
            certified_multiway_reductions: 1,
            observed_kclique_dispatches: 1,
            certified_edge_permutation_slots: 10,
            certified_stream_groups: 1,
            certified_skew_scheduled_plans: 1,
            certified_sorted_layout_requirements: 2,
            certified_helper_split_specs: 1,
            certified_helper_relation_rules: 1,
            certified_helper_relation_scans: 1,
            observed_layout_sorts: 0,
            observed_layout_fast_path_hits: 2,
            observed_metadata_builds: 1,
            observed_metadata_build_nanos: 42,
            observed_histogram_refreshes: 0,
            observed_histogram_refresh_nanos: 0,
        }
    );
}

#[test]
fn runtime_wcoj_certification_accepts_cost_gated_planned_hash_route() {
    let executable = executable_with_planned_hash_route();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    assert_eq!(preflight.wcoj_required_reduction_count, 1);
    assert_eq!(preflight.multiway_reduction_count, 1);
    assert_eq!(preflight.planned_hash_route_count, 1);
    assert_eq!(preflight.planned_hash_planner_wins_count, 1);
    assert_eq!(preflight.planned_hash_incomplete_stats_count, 0);
    assert_eq!(preflight.planned_hash_cost_evidence_count, 1);

    let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(
        preflight,
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters::default(),
    );

    assert_eq!(
        trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::NotRequired {
            observed_wcoj_dispatches: 0,
            planned_hash_routes: 1,
            planned_hash_planner_wins: 1,
            planned_hash_incomplete_stats: 0,
            planned_hash_cost_evidence: 1,
        }
    );
    trace
        .require_wcoj_certification()
        .expect("cost-gated planned hash route should certify without WCOJ dispatch");
}

#[test]
fn runtime_trace_preserves_counter_snapshots_and_wcoj_certification() {
    let executable = executable_with_kclique_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    let before = EpistemicGpuRuntimeCounters {
        wcoj_clique5_dispatch_count: 2,
        kclique_metadata_build_count: 4,
        kclique_metadata_build_nanos: 10,
        kclique_histogram_refresh_count: 7,
        kclique_histogram_refresh_nanos: 100,
        wcoj_layout_sort_invocation_count: 8,
        ..EpistemicGpuRuntimeCounters::default()
    };
    let after = EpistemicGpuRuntimeCounters {
        wcoj_clique5_dispatch_count: 3,
        kclique_metadata_build_count: 5,
        kclique_metadata_build_nanos: 52,
        kclique_histogram_refresh_count: 9,
        kclique_histogram_refresh_nanos: 175,
        wcoj_layout_sort_invocation_count: 10,
        ..EpistemicGpuRuntimeCounters::default()
    };

    let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(preflight, before, after);

    assert_eq!(trace.preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(trace.counters_before.wcoj_clique5_dispatch_count, 2);
    assert_eq!(trace.counters_after.wcoj_clique5_dispatch_count, 3);
    assert_eq!(trace.counter_delta.wcoj_clique5_dispatch_count, 1);
    assert_eq!(trace.counter_delta.kclique_metadata_build_count, 1);
    assert_eq!(trace.counter_delta.kclique_metadata_build_nanos, 42);
    assert_eq!(trace.counter_delta.kclique_histogram_refresh_count, 2);
    assert_eq!(trace.counter_delta.kclique_histogram_refresh_nanos, 75);
    assert_eq!(trace.counter_delta.wcoj_layout_sort_invocation_count, 2);
    assert_eq!(
        trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1,
            certified_multiway_reductions: 1,
            observed_kclique_dispatches: 1,
            certified_edge_permutation_slots: 10,
            certified_stream_groups: 1,
            certified_skew_scheduled_plans: 1,
            certified_sorted_layout_requirements: 2,
            certified_helper_split_specs: 1,
            certified_helper_relation_rules: 1,
            certified_helper_relation_scans: 1,
            observed_layout_sorts: 2,
            observed_layout_fast_path_hits: 0,
            observed_metadata_builds: 1,
            observed_metadata_build_nanos: 42,
            observed_histogram_refreshes: 2,
            observed_histogram_refresh_nanos: 75,
        }
    );
    trace
        .require_wcoj_certification()
        .expect("certified trace should pass runtime WCOJ gate");
}

#[test]
fn runtime_trace_rejects_missing_required_wcoj_dispatch() {
    let executable = executable_with_kclique_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();
    let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(
        preflight,
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters::default(),
    );

    let err = trace
        .require_wcoj_certification()
        .expect_err("metadata-only WCOJ evidence must fail closed");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU WCOJ dispatch certification");
            assert!(context.contains("required_kclique_plans=1"));
            assert!(context.contains("observed_wcoj_dispatches=0"));
        }
        other => panic!("expected WCOJ certification error, got {other:?}"),
    }
}

#[test]
fn runtime_trace_rejects_kclique_dispatch_without_required_layout_evidence() {
    let executable = executable_with_kclique_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();
    assert_eq!(preflight.sorted_layout_requirement_count, 2);

    let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(
        preflight,
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters {
            wcoj_clique5_dispatch_count: 1,
            kclique_metadata_build_count: 1,
            kclique_metadata_build_nanos: 42,
            ..EpistemicGpuRuntimeCounters::default()
        },
    );

    let err = trace
        .require_wcoj_certification()
        .expect_err("layout-required WCOJ evidence must fail closed without layout counters");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU WCOJ layout certification");
            assert!(context.contains("required_sorted_layouts=2"));
            assert!(context.contains("observed_layout_events=0"));
        }
        other => panic!("expected WCOJ layout certification error, got {other:?}"),
    }
}

#[test]
fn runtime_trace_rejects_partial_kclique_layout_evidence() {
    let executable = executable_with_kclique_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();
    assert_eq!(preflight.sorted_layout_requirement_count, 2);

    let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(
        preflight,
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters {
            wcoj_clique5_dispatch_count: 1,
            wcoj_layout_sort_invocation_count: 1,
            kclique_metadata_build_count: 1,
            kclique_metadata_build_nanos: 42,
            ..EpistemicGpuRuntimeCounters::default()
        },
    );

    let err = trace
        .require_wcoj_certification()
        .expect_err("each sorted-layout requirement needs layout runtime evidence");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU WCOJ layout certification");
            assert!(context.contains("required_sorted_layouts=2"));
            assert!(context.contains("observed_layout_events=1"));
        }
        other => panic!("expected WCOJ layout certification error, got {other:?}"),
    }
}

#[test]
fn runtime_trace_rejects_kclique_metadata_build_without_timing() {
    let executable = executable_with_kclique_wcoj_plan();
    let preflight = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(
        preflight,
        EpistemicGpuRuntimeCounters::default(),
        EpistemicGpuRuntimeCounters {
            wcoj_clique5_dispatch_count: 1,
            wcoj_layout_sort_invocation_count: 2,
            kclique_metadata_build_count: 1,
            kclique_metadata_build_nanos: 0,
            ..EpistemicGpuRuntimeCounters::default()
        },
    );

    let err = trace
        .require_wcoj_certification()
        .expect_err("K-clique metadata certification requires measured build timing");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU K-clique metadata certification");
            assert!(context.contains("required_kclique_plans=1"));
            assert!(context.contains("observed_metadata_builds=1"));
            assert!(context.contains("observed_metadata_build_nanos=0"));
        }
        other => panic!("expected K-clique metadata certification error, got {other:?}"),
    }
}

#[test]
fn workspace_reset_trace_records_device_zeroing_for_all_buffers() {
    let plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![
            epistemic_literal("known_fact", EirEpistemicOp::Know),
            epistemic_literal("possible_fact", EirEpistemicOp::Possible),
        ],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "out".to_string(),
            relational_body_atoms: 3,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );
    let layout = EpistemicGpuWorkspaceLayout::for_plan(
        &plan,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap();

    let trace = EpistemicGpuWorkspaceResetTrace::for_layout(layout);

    assert_eq!(trace.candidate_assumption_bytes, 16);
    assert_eq!(trace.world_view_bytes, 32);
    assert_eq!(trace.model_membership_bytes, 96);
    assert_eq!(trace.rejection_reason_bytes, 32);
    assert_eq!(trace.total_zeroed_bytes(), 176);
    assert_eq!(trace.device_zero_ops, 4);
    assert_eq!(trace.host_write_ops, 0);
}

#[test]
fn candidate_generation_trace_records_device_kernel_without_host_writes() {
    let trace = EpistemicGpuCandidateGenerationTrace::for_counts(3, 8).unwrap();

    assert_eq!(trace.literal_count, 3);
    assert_eq!(trace.generated_candidates, 8);
    assert_eq!(trace.candidate_assumption_bytes, 24);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
}

#[test]
fn candidate_generation_trace_rejects_unlaunchable_candidate_mask_product() {
    let err = EpistemicGpuCandidateGenerationTrace::for_counts(31, 1usize << 31)
        .expect_err("candidate trace must fail before recording an unlaunchable GPU kernel");

    match err {
        xlog_core::XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } => {
            assert_eq!(context, "epistemic GPU candidate generation launch");
            assert_eq!(estimated_bytes, 31u64 << 31);
            assert_eq!(budget_bytes, u32::MAX as u64);
        }
        other => panic!("expected candidate launch bound error, got {other:?}"),
    }
}

#[test]
fn cuda_event_timing_trace_converts_milliseconds_to_nanoseconds() {
    let timing = EpistemicGpuKernelTimingTrace::from_cuda_elapsed_ms(0.125).unwrap();

    assert_eq!(timing.cuda_event_pairs, 1);
    assert_eq!(timing.timing_sync_ops, 1);
    assert_eq!(timing.kernel_elapsed_nanos, 125_000);
    assert!(timing.is_recorded());
}

#[test]
fn cuda_event_timing_trace_requires_timing_sync_evidence() {
    let timing = EpistemicGpuKernelTimingTrace {
        cuda_event_pairs: 1,
        timing_sync_ops: 0,
        kernel_elapsed_nanos: 125_000,
    };

    assert!(!timing.is_recorded());
}

#[test]
fn candidate_generation_trace_accepts_cuda_event_timing() {
    let timing = EpistemicGpuKernelTimingTrace::from_cuda_elapsed_ms(0.25).unwrap();
    let trace = EpistemicGpuCandidateGenerationTrace::for_counts(3, 8)
        .unwrap()
        .with_kernel_timing(timing);

    assert_eq!(trace.kernel_timing.cuda_event_pairs, 1);
    assert_eq!(trace.kernel_timing.timing_sync_ops, 1);
    assert_eq!(trace.kernel_timing.kernel_elapsed_nanos, 250_000);
}

#[test]
fn propagation_trace_records_device_kernel_without_host_writes() {
    let trace = EpistemicGpuPropagationTrace::for_counts(3, 8).unwrap();

    assert_eq!(trace.literal_count, 3);
    assert_eq!(trace.propagated_candidates, 8);
    assert_eq!(trace.world_view_bytes_written, 8);
    assert_eq!(trace.rejection_reason_slots_written, 8);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
}

#[test]
fn propagation_trace_rejects_unlaunchable_candidate_count() {
    let err = EpistemicGpuPropagationTrace::for_counts(1, u32::MAX as usize + 1)
        .expect_err("propagation trace must fail before recording an unlaunchable GPU kernel");

    assert_u32_resource_exhausted(err, "epistemic GPU propagation launch", u32::MAX as u64 + 1);
}

#[test]
fn candidate_validation_trace_records_device_kernel_without_host_writes() {
    let trace = EpistemicGpuCandidateValidationTrace::for_counts(3, 8).unwrap();

    assert_eq!(trace.literal_count, 3);
    assert_eq!(trace.validated_candidates, 8);
    assert_eq!(trace.candidate_assumption_bytes_checked, 24);
    assert_eq!(trace.world_view_bytes_checked, 8);
    assert_eq!(trace.rejection_reason_slots_written, 8);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
}

#[test]
fn candidate_validation_trace_rejects_unlaunchable_candidate_count() {
    let err = EpistemicGpuCandidateValidationTrace::for_counts(1, u32::MAX as usize + 1)
        .expect_err("validation trace must fail before recording an unlaunchable GPU kernel");

    assert_u32_resource_exhausted(err, "epistemic GPU validation launch", u32::MAX as u64 + 1);
}

#[test]
fn model_membership_trace_records_device_kernel_without_host_writes() {
    let trace = EpistemicGpuModelMembershipTrace::for_counts(3, 8, 2, 4).unwrap();

    assert_eq!(trace.literal_count, 3);
    assert_eq!(trace.candidates_checked, 8);
    assert_eq!(trace.reduction_count, 2);
    assert_eq!(trace.models_per_reduction, 4);
    assert_eq!(trace.model_membership_bytes_written, 192);
    assert_eq!(trace.output_row_count_device_reads, 1);
    assert_eq!(trace.rejection_reason_slots_checked, 8);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
}

#[test]
fn model_membership_trace_rejects_unlaunchable_membership_product() {
    let err = EpistemicGpuModelMembershipTrace::for_counts(2, (u32::MAX as usize / 2) + 1, 1, 1)
        .expect_err("model-membership trace must fail before recording an unlaunchable GPU kernel");

    assert_u32_resource_exhausted(
        err,
        "epistemic GPU model-membership launch",
        u32::MAX as u64 + 1,
    );
}

#[test]
fn model_membership_trace_fails_closed_until_stable_model_tuple_source_exists() {
    let trace = EpistemicGpuModelMembershipTrace::for_counts(3, 8, 2, 4).unwrap();

    assert_eq!(
        trace.membership_source,
        EpistemicGpuModelMembershipSource::ReducedOutputRowCountOnly
    );
    let err = trace
        .require_stable_model_tuple_source()
        .expect_err("row-count-gated staging is not semantic model membership");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(
                construct,
                "epistemic GPU stable-model membership certification"
            );
            assert!(context.contains("ReducedOutputRowCountOnly"));
            assert!(context.contains("actual reduced stable-model tuple membership"));
        }
        other => panic!("expected model-membership certification error, got {other:?}"),
    }
}

#[test]
fn model_membership_trace_accepts_stable_model_tuple_sources() {
    let trace =
        EpistemicGpuModelMembershipTrace::for_stable_model_tuple_sources(3, 8, 2, 4, 3).unwrap();

    assert_eq!(trace.literal_count, 3);
    assert_eq!(trace.candidates_checked, 8);
    assert_eq!(trace.reduction_count, 2);
    assert_eq!(trace.models_per_reduction, 4);
    assert_eq!(trace.model_membership_bytes_written, 192);
    assert_eq!(trace.output_row_count_device_reads, 0);
    assert_eq!(trace.tuple_source_row_count_device_reads, 3);
    assert_eq!(trace.tuple_source_key_column_device_reads, 0);
    assert_eq!(
        trace.membership_source,
        EpistemicGpuModelMembershipSource::StableModelTupleBuffer
    );
    assert_eq!(trace.kernel_launches, 3);
    assert_eq!(trace.host_write_ops, 0);
    trace
        .require_stable_model_tuple_source()
        .expect("stable tuple-source traces should certify model membership");
    trace
        .require_planned_tuple_key_column_reads(0)
        .expect("zero-arity tuple sources do not require tuple-key column reads");
}

#[test]
fn model_membership_trace_records_nonzero_arity_tuple_key_column_reads() {
    let trace = EpistemicGpuModelMembershipTrace::for_stable_model_tuple_sources_with_key_columns(
        3, 8, 2, 4, 3, 4,
    )
    .unwrap();

    assert_eq!(trace.output_row_count_device_reads, 0);
    assert_eq!(trace.tuple_source_row_count_device_reads, 3);
    assert_eq!(trace.tuple_source_key_column_device_reads, 4);
    assert_eq!(trace.kernel_launches, 3);
    assert_eq!(trace.host_write_ops, 0);
    trace
        .require_stable_model_tuple_source()
        .expect("tuple-key traces should certify stable tuple source membership");
    trace
        .require_planned_tuple_key_column_reads(4)
        .expect("tuple-key traces should match planned key column reads");
}

#[test]
fn tuple_source_model_membership_trace_rejects_unlaunchable_membership_product() {
    let err = EpistemicGpuModelMembershipTrace::for_stable_model_tuple_sources_with_key_columns(
        2,
        (u32::MAX as usize / 2) + 1,
        1,
        1,
        1,
        0,
    )
    .expect_err("tuple-source trace must fail before recording an unlaunchable GPU kernel");

    assert_u32_resource_exhausted(
        err,
        "epistemic GPU model-membership launch",
        u32::MAX as u64 + 1,
    );
}

#[test]
fn model_membership_trace_rejects_missing_nonzero_arity_tuple_key_column_reads() {
    let trace =
        EpistemicGpuModelMembershipTrace::for_stable_model_tuple_sources(3, 8, 2, 4, 3).unwrap();

    let err = trace
        .require_planned_tuple_key_column_reads(4)
        .expect_err("nonzero-arity tuple membership requires planned tuple-key device reads");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(
                construct,
                "epistemic GPU stable-model membership certification"
            );
            assert!(context.contains("tuple-key device column reads"));
            assert!(context.contains("expected=4"));
        }
        other => panic!("expected tuple-key certification error, got {other:?}"),
    }
}

#[test]
fn world_view_validation_trace_records_device_kernel_without_host_writes() {
    let trace = EpistemicGpuWorldViewValidationTrace::for_counts(3, 8, 2, 4).unwrap();

    assert_eq!(trace.literal_count, 3);
    assert_eq!(trace.candidates_checked, 8);
    assert_eq!(trace.reduction_count, 2);
    assert_eq!(trace.models_per_reduction, 4);
    assert_eq!(trace.model_membership_bytes_checked, 192);
    assert_eq!(trace.world_view_slots_checked, 8);
    assert_eq!(trace.rejection_reason_slots_written, 8);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
}

#[test]
fn world_view_validation_trace_rejects_unlaunchable_membership_product() {
    let err =
        EpistemicGpuWorldViewValidationTrace::for_counts(2, (u32::MAX as usize / 2) + 1, 1, 1)
            .expect_err("world-view trace must fail before recording an unlaunchable GPU kernel");

    assert_u32_resource_exhausted(
        err,
        "epistemic GPU world-view validation membership launch",
        u32::MAX as u64 + 1,
    );
}

#[test]
fn materialization_trace_records_device_kernel_without_host_writes() {
    let trace = EpistemicGpuMaterializationTrace::for_count(8).unwrap();

    assert_eq!(trace.materialized_candidates, 8);
    assert_eq!(trace.world_view_slots_written, 8);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
}

#[test]
fn materialization_trace_rejects_unlaunchable_candidate_count() {
    let err = EpistemicGpuMaterializationTrace::for_count(u32::MAX as usize + 1)
        .expect_err("materialization trace must fail before recording an unlaunchable GPU kernel");

    assert_u32_resource_exhausted(
        err,
        "epistemic GPU materialization launch",
        u32::MAX as u64 + 1,
    );
}

#[test]
fn final_result_materialization_trace_records_device_row_count_read_without_host_writes() {
    let trace = EpistemicGpuFinalResultMaterializationTrace::for_count(8).unwrap();

    assert_eq!(trace.materialized_candidates, 8);
    assert_eq!(trace.output_row_count_device_reads, 1);
    assert_eq!(trace.world_view_slots_written, 8);
    assert_eq!(trace.kernel_launches, 1);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
}

#[test]
fn final_result_materialization_trace_rejects_unlaunchable_candidate_count() {
    let err = EpistemicGpuFinalResultMaterializationTrace::for_count(u32::MAX as usize + 1)
        .expect_err("final-result trace must fail before recording an unlaunchable GPU kernel");

    assert_u32_resource_exhausted(
        err,
        "epistemic GPU final-result launch",
        u32::MAX as u64 + 1,
    );
}

#[test]
fn final_tuple_materialization_trace_records_device_tuple_buffer_without_host_writes() {
    let trace =
        EpistemicGpuFinalTupleMaterializationTrace::for_counts(2, 16, 128, 3, 8, 2, 4).unwrap();

    assert_eq!(trace.output_column_count, 2);
    assert_eq!(trace.output_row_capacity, 16);
    assert_eq!(trace.tuple_bytes_capacity, 128);
    assert_eq!(trace.output_row_count_device_reads, 3);
    assert_eq!(trace.model_membership_bytes_checked, 192);
    assert_eq!(trace.world_view_slots_checked, 8);
    assert_eq!(trace.row_filter_count, 0);
    assert_eq!(trace.negated_row_filter_count, 0);
    assert_eq!(trace.final_row_count_device_writes, 1);
    assert_eq!(trace.kernel_launches, 4);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
}

#[test]
fn final_tuple_materialization_trace_rejects_unlaunchable_output_rows() {
    let err = EpistemicGpuFinalTupleMaterializationTrace::for_counts(
        1,
        u32::MAX as usize + 1,
        128,
        3,
        8,
        2,
        4,
    )
    .expect_err("final tuple trace must reject unlaunchable row-map kernels");

    assert_u32_resource_exhausted(
        err,
        "epistemic GPU final-tuple output rows",
        u32::MAX as u64 + 1,
    );
}

#[test]
fn final_tuple_materialization_trace_rejects_unlaunchable_membership_product() {
    let err = EpistemicGpuFinalTupleMaterializationTrace::for_counts(
        1,
        16,
        128,
        2,
        (u32::MAX as usize / 2) + 1,
        1,
        1,
    )
    .expect_err("final tuple trace must reject unlaunchable membership kernels");

    assert_u32_resource_exhausted(
        err,
        "epistemic GPU final-tuple membership launch",
        u32::MAX as u64 + 1,
    );
}

#[test]
fn final_tuple_materialization_trace_records_row_filter_polarity_counts() {
    let trace = EpistemicGpuFinalTupleMaterializationTrace::for_counts(2, 16, 128, 3, 8, 2, 4)
        .unwrap()
        .with_row_filter_counts(2, 1)
        .unwrap();

    assert_eq!(trace.row_filter_count, 2);
    assert_eq!(trace.negated_row_filter_count, 1);
}

#[test]
fn final_tuple_materialization_evidence_rejects_impossible_row_filter_accounting() {
    let unfiltered =
        EpistemicGpuFinalTupleMaterializationTrace::for_counts(2, 16, 128, 3, 8, 2, 4).unwrap();
    unfiltered
        .require_row_filter_materialization_evidence("runtime row-filter evidence", 8)
        .unwrap();

    let mut covered_without_filters = unfiltered;
    covered_without_filters.row_specific_membership_row_capacity = 4;
    covered_without_filters.row_filter_row_capacity_outside_model_slot_window = 12;
    assert!(matches!(
        covered_without_filters.require_row_filter_materialization_evidence(
            "runtime row-filter evidence",
            8,
        ),
        Err(xlog_core::XlogError::UnsupportedEpistemicConstruct { context, .. })
            if context.contains("without row filters")
    ));

    let mut impossible_negated = unfiltered.with_row_filter_counts(1, 0).unwrap();
    impossible_negated.negated_row_filter_count = 2;
    assert!(matches!(
        impossible_negated.require_row_filter_materialization_evidence(
            "runtime row-filter evidence",
            8,
        ),
        Err(xlog_core::XlogError::UnsupportedEpistemicConstruct { context, .. })
            if context.contains("negated row filters")
    ));
}

#[test]
fn final_tuple_materialization_trace_rejects_launch_counter_overflow() {
    let err = EpistemicGpuFinalTupleMaterializationTrace::for_counts(
        u32::MAX as usize + 1,
        16,
        128,
        3,
        8,
        2,
        4,
    )
    .expect_err("final tuple trace must not truncate kernel launch counts");

    match err {
        xlog_core::XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } => {
            assert_eq!(context, "epistemic GPU final-tuple output columns");
            assert_eq!(estimated_bytes, u32::MAX as u64 + 1);
            assert_eq!(budget_bytes, u32::MAX as u64);
        }
        other => panic!("expected output-column overflow error, got {other:?}"),
    }
}

#[test]
fn transfer_budget_trace_rejects_tracked_host_transfers_in_gpu_hot_path() {
    let before = HostTransferStats {
        dtoh_bytes: 3,
        htod_bytes: 5,
        dtoh_calls: 1,
        htod_calls: 2,
    };
    let after = before;

    let trace = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats(8, before, after)
        .expect("unchanged transfer stats are accepted");
    assert_eq!(trace.candidate_count, 8);
    assert_eq!(trace.tracked_dtoh_bytes, 0);
    assert_eq!(trace.tracked_htod_bytes, 0);
    assert_eq!(trace.tracked_dtoh_calls, 0);
    assert_eq!(trace.tracked_htod_calls, 0);
    assert_eq!(trace.per_candidate_host_round_trips, 0);

    let after_with_dtoh = HostTransferStats {
        dtoh_bytes: 7,
        dtoh_calls: 2,
        ..after
    };
    let err = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats(8, before, after_with_dtoh)
        .expect_err("tracked D2H is not allowed in the epistemic GPU hot path");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU transfer budget");
            assert!(context.contains("tracked host transfer in GPU hot path"));
            assert!(context.contains("dtoh_calls=1"));
        }
        other => panic!("expected transfer-budget error, got {other:?}"),
    }

    let after_reset = HostTransferStats {
        dtoh_bytes: 2,
        ..after
    };
    let err = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats(8, before, after_reset)
        .expect_err("transfer counters must be monotonic during the hot path");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU transfer budget");
            assert!(context.contains("host transfer counter decreased"));
            assert!(context.contains("dtoh_bytes"));
        }
        other => panic!("expected transfer-budget monotonicity error, got {other:?}"),
    }
}

#[test]
fn transfer_budget_trace_rejects_launch_metadata_bytes_without_calls() {
    let data_plane = HostTransferStats {
        dtoh_bytes: 3,
        htod_bytes: 5,
        dtoh_calls: 1,
        htod_calls: 2,
    };
    let launch_before = HostLaunchMetadataTransferStats {
        htod_bytes: 8,
        htod_calls: 4,
    };
    let launch_after = HostLaunchMetadataTransferStats {
        htod_bytes: 12,
        htod_calls: 4,
    };

    let err = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats_with_launch_metadata(
        8,
        data_plane,
        data_plane,
        launch_before,
        launch_after,
    )
    .expect_err("launch metadata H2D bytes require matching H2D calls");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU transfer budget");
            assert!(context.contains("launch metadata H2D bytes require matching H2D calls"));
        }
        other => panic!("expected launch metadata transfer accounting error, got {other:?}"),
    }
}

#[test]
fn transfer_budget_trace_rejects_launch_metadata_calls_without_bytes() {
    let data_plane = HostTransferStats {
        dtoh_bytes: 3,
        htod_bytes: 5,
        dtoh_calls: 1,
        htod_calls: 2,
    };
    let launch_before = HostLaunchMetadataTransferStats {
        htod_bytes: 16,
        htod_calls: 4,
    };
    let launch_after = HostLaunchMetadataTransferStats {
        htod_bytes: 16,
        htod_calls: 5,
    };

    let err = EpistemicGpuTransferBudgetTrace::from_host_transfer_stats_with_launch_metadata(
        8,
        data_plane,
        data_plane,
        launch_before,
        launch_after,
    )
    .expect_err("launch metadata H2D calls require matching payload bytes");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU transfer budget");
            assert!(context.contains("launch metadata H2D calls require matching payload bytes"));
        }
        other => panic!("expected launch metadata transfer accounting error, got {other:?}"),
    }
}

fn epistemic_literal(predicate: &str, op: EirEpistemicOp) -> EirEpistemicLiteral {
    EirEpistemicLiteral {
        op,
        negated: false,
        atom: EirAtom {
            predicate: predicate.to_string(),
            arity: 0,
            terms: Vec::new(),
        },
    }
}

fn negated_epistemic_literal(predicate: &str, op: EirEpistemicOp) -> EirEpistemicLiteral {
    EirEpistemicLiteral {
        op,
        negated: true,
        atom: EirAtom {
            predicate: predicate.to_string(),
            arity: 0,
            terms: Vec::new(),
        },
    }
}

fn kclique_epistemic_literal() -> EirEpistemicLiteral {
    EirEpistemicLiteral {
        op: EirEpistemicOp::Know,
        negated: false,
        atom: EirAtom {
            predicate: "clique5".to_string(),
            arity: 5,
            terms: ["A", "B", "C", "D", "E"]
                .into_iter()
                .map(|name| EirTerm::Variable(name.to_string()))
                .collect(),
        },
    }
}

fn triangle_epistemic_literal() -> EirEpistemicLiteral {
    EirEpistemicLiteral {
        op: EirEpistemicOp::Know,
        negated: false,
        atom: EirAtom {
            predicate: "tri".to_string(),
            arity: 3,
            terms: ["X", "Y", "Z"]
                .into_iter()
                .map(|name| EirTerm::Variable(name.to_string()))
                .collect(),
        },
    }
}

fn cycle4_epistemic_literal() -> EirEpistemicLiteral {
    EirEpistemicLiteral {
        op: EirEpistemicOp::Know,
        negated: false,
        atom: EirAtom {
            predicate: "cycle4".to_string(),
            arity: 4,
            terms: ["W", "X", "Y", "Z"]
                .into_iter()
                .map(|name| EirTerm::Variable(name.to_string()))
                .collect(),
        },
    }
}

fn complete_k5_edge_rows() -> Vec<(u32, u32)> {
    let mut rows = Vec::new();
    for left in 1..=5 {
        for right in (left + 1)..=5 {
            rows.push((left, right));
        }
    }
    rows
}

fn upload_binary_u32(
    memory: &Arc<GpuMemoryManager>,
    rows: &[(u32, u32)],
    name_a: &str,
    name_b: &str,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = rows.len() * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !rows.is_empty() {
        let bytes0: Vec<u8> = rows
            .iter()
            .flat_map(|(left, _)| left.to_le_bytes())
            .collect();
        let bytes1: Vec<u8> = rows
            .iter()
            .flat_map(|(_, right)| right.to_le_bytes())
            .collect();
        device
            .htod_sync_copy_into(&bytes0, &mut col0)
            .expect("upload col0");
        device
            .htod_sync_copy_into(&bytes1, &mut col1)
            .expect("upload col1");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("upload row count");
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        u64::from(n),
        d_num_rows,
        Schema::new(vec![
            (name_a.to_string(), ScalarType::U32),
            (name_b.to_string(), ScalarType::U32),
        ]),
        n,
    )
}

#[cfg(feature = "epistemic-logic-tests")]
fn upload_ternary_u32(
    memory: &Arc<GpuMemoryManager>,
    rows: &[(u32, u32, u32)],
    name_a: &str,
    name_b: &str,
    name_c: &str,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = rows.len() * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut col2 = memory.alloc::<u8>(bytes_per_col).expect("alloc col2");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !rows.is_empty() {
        let bytes0: Vec<u8> = rows
            .iter()
            .flat_map(|(left, _, _)| left.to_le_bytes())
            .collect();
        let bytes1: Vec<u8> = rows
            .iter()
            .flat_map(|(_, middle, _)| middle.to_le_bytes())
            .collect();
        let bytes2: Vec<u8> = rows
            .iter()
            .flat_map(|(_, _, right)| right.to_le_bytes())
            .collect();
        device
            .htod_sync_copy_into(&bytes0, &mut col0)
            .expect("upload col0");
        device
            .htod_sync_copy_into(&bytes1, &mut col1)
            .expect("upload col1");
        device
            .htod_sync_copy_into(&bytes2, &mut col2)
            .expect("upload col2");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("upload row count");
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into(), col2.into()],
        u64::from(n),
        d_num_rows,
        Schema::new(vec![
            (name_a.to_string(), ScalarType::U32),
            (name_b.to_string(), ScalarType::U32),
            (name_c.to_string(), ScalarType::U32),
        ]),
        n,
    )
}

#[cfg(feature = "epistemic-logic-tests")]
fn upload_quaternary_u32(
    memory: &Arc<GpuMemoryManager>,
    rows: &[(u32, u32, u32, u32)],
    name_a: &str,
    name_b: &str,
    name_c: &str,
    name_d: &str,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = rows.len() * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut col2 = memory.alloc::<u8>(bytes_per_col).expect("alloc col2");
    let mut col3 = memory.alloc::<u8>(bytes_per_col).expect("alloc col3");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !rows.is_empty() {
        let bytes0: Vec<u8> = rows
            .iter()
            .flat_map(|(a, _, _, _)| a.to_le_bytes())
            .collect();
        let bytes1: Vec<u8> = rows
            .iter()
            .flat_map(|(_, b, _, _)| b.to_le_bytes())
            .collect();
        let bytes2: Vec<u8> = rows
            .iter()
            .flat_map(|(_, _, c, _)| c.to_le_bytes())
            .collect();
        let bytes3: Vec<u8> = rows
            .iter()
            .flat_map(|(_, _, _, d)| d.to_le_bytes())
            .collect();
        device
            .htod_sync_copy_into(&bytes0, &mut col0)
            .expect("upload col0");
        device
            .htod_sync_copy_into(&bytes1, &mut col1)
            .expect("upload col1");
        device
            .htod_sync_copy_into(&bytes2, &mut col2)
            .expect("upload col2");
        device
            .htod_sync_copy_into(&bytes3, &mut col3)
            .expect("upload col3");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("upload row count");
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into(), col2.into(), col3.into()],
        u64::from(n),
        d_num_rows,
        Schema::new(vec![
            (name_a.to_string(), ScalarType::U32),
            (name_b.to_string(), ScalarType::U32),
            (name_c.to_string(), ScalarType::U32),
            (name_d.to_string(), ScalarType::U32),
        ]),
        n,
    )
}

#[cfg(feature = "epistemic-logic-tests")]
fn upload_single_u32_row(
    memory: &Arc<GpuMemoryManager>,
    values: &[u32],
    names: &[&str],
) -> CudaBuffer {
    assert_eq!(values.len(), names.len());
    let mut columns = Vec::with_capacity(values.len());
    let device = memory.device().inner();
    for value in values {
        let mut col = memory
            .alloc::<u8>(std::mem::size_of::<u32>())
            .expect("alloc single-row col");
        let bytes = value.to_le_bytes();
        device
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("upload single-row col");
        columns.push(col.into());
    }
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc single-row count");
    device
        .htod_sync_copy_into(&[1u32], &mut d_num_rows)
        .expect("upload single-row count");

    CudaBuffer::from_columns_with_host_count(
        columns,
        1,
        d_num_rows,
        Schema::new(
            names
                .iter()
                .map(|name| ((*name).to_string(), ScalarType::U32))
                .collect(),
        ),
        1,
    )
}

fn upload_unary_u32(memory: &Arc<GpuMemoryManager>, rows: &[u32], name: &str) -> CudaBuffer {
    upload_unary_typed_u32(memory, rows, name, ScalarType::U32)
}

fn upload_unary_typed_u32(
    memory: &Arc<GpuMemoryManager>,
    rows: &[u32],
    name: &str,
    column_type: ScalarType,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = std::mem::size_of_val(rows);
    let mut col = memory.alloc::<u8>(bytes_per_col).expect("alloc unary col");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !rows.is_empty() {
        let bytes: Vec<u8> = rows.iter().flat_map(|value| value.to_le_bytes()).collect();
        device
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("upload unary col");
    }
    device
        .htod_sync_copy_into(&[n], &mut d_num_rows)
        .expect("upload unary row count");

    CudaBuffer::from_columns_with_host_count(
        vec![col.into()],
        u64::from(n),
        d_num_rows,
        Schema::new(vec![(name.to_string(), column_type)]),
        n,
    )
}

#[cfg(feature = "epistemic-logic-tests")]
const EPISTEMIC_K5_MARKER_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred marker(u32, u32, u32, u32, u32).
    pred clique5(u32, u32, u32, u32, u32).

    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4),
        know marker(V0, V1, V2, V3, V4).
"#;

#[cfg(feature = "epistemic-logic-tests")]
const REDUCED_K5_MARKER_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred marker(u32, u32, u32, u32, u32).
    pred clique5(u32, u32, u32, u32, u32).

    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4).
"#;

#[cfg(feature = "epistemic-logic-tests")]
const K5_EDGES: [(&str, usize, usize); 10] = [
    ("e01", 0, 1),
    ("e02", 0, 2),
    ("e03", 0, 3),
    ("e04", 0, 4),
    ("e12", 1, 2),
    ("e13", 1, 3),
    ("e14", 1, 4),
    ("e23", 2, 3),
    ("e24", 2, 4),
    ("e34", 3, 4),
];

#[cfg(feature = "epistemic-logic-tests")]
fn rel_ids_for_parsed_k5_reduced() -> BTreeMap<String, RelId> {
    let mut compiler = Compiler::new();
    let _ = compiler
        .compile(REDUCED_K5_MARKER_SRC)
        .expect("compile reduced K5 marker source");
    compiler
        .rel_ids()
        .iter()
        .map(|(name, rel)| (name.clone(), *rel))
        .collect()
}

#[cfg(feature = "epistemic-logic-tests")]
fn k5_stats(rel_ids: &BTreeMap<String, RelId>, hot: Option<(usize, f64)>) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    for (name, left, right) in K5_EDGES {
        let rel = *rel_ids.get(name).expect("edge rel id");
        snapshot.rel_names.push((rel, name.to_string()));
        let mut stats = RelationStats::new(rel);
        stats.update_cardinality(10_000);
        for (col_idx, variable) in [(0usize, left), (1usize, right)] {
            let mut col = ColumnStats::new(col_idx, ScalarType::U32);
            col.update_distinct(10_000);
            stats.add_column(col);
            stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 1.0, 1.25));
            let heat = match hot {
                Some((hot_var, hot_heat)) if hot_var == variable => hot_heat,
                _ => 0.25,
            };
            stats.add_key_heat(KeyHeatStats::new(col_idx, heat, heat));
        }
        snapshot.relations.push(stats);
    }

    for (left_idx, (left_name, left_i, left_j)) in K5_EDGES.iter().enumerate() {
        let left_rel = *rel_ids.get(*left_name).expect("left rel id");
        for (right_name, right_i, right_j) in K5_EDGES.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let right_rel = *rel_ids.get(*right_name).expect("right rel id");
                let mut sel = JoinSelectivity::new(left_rel, right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}

fn upload_zero_arity(memory: &Arc<GpuMemoryManager>, rows: u32) -> CudaBuffer {
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc zero-arity rows");
    memory
        .device()
        .inner()
        .htod_sync_copy_into(&[rows], &mut d_num_rows)
        .expect("upload zero-arity row count");

    CudaBuffer::from_columns_with_host_count(
        Vec::new(),
        u64::from(rows),
        d_num_rows,
        Schema::new(Vec::new()),
        rows,
    )
}

fn executable_with_operator_mix() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![
            epistemic_literal("known_gate", EirEpistemicOp::Know),
            epistemic_literal("possible_gate", EirEpistemicOp::Possible),
            negated_epistemic_literal("not_known_gate", EirEpistemicOp::Know),
            negated_epistemic_literal("not_possible_gate", EirEpistemicOp::Possible),
        ],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "out".to_string(),
            relational_body_atoms: 1,
            wcoj_status: EpistemicWcojReductionStatus::NotWcojCandidate,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::new(),
        reduced_runtime_plan: runtime_plan_with_scan_rule_execution(),
    }
}

fn nonzero_arity_row_count_only_membership_executable() -> EpistemicExecutablePlan {
    let mut gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![EirEpistemicLiteral {
            op: EirEpistemicOp::Know,
            negated: false,
            atom: EirAtom {
                predicate: "base".to_string(),
                arity: 1,
                terms: vec![EirTerm::Integer(7)],
            },
        }],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "out".to_string(),
            relational_body_atoms: 1,
            wcoj_status: EpistemicWcojReductionStatus::NotWcojCandidate,
        }],
    );
    gpu_plan.tuple_membership_bindings[0].key_columns.clear();

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::new(),
        reduced_runtime_plan: runtime_plan_with_scan_rule_execution(),
    }
}

fn accepted_ground_literal_component_executable(
    predicate: &str,
    output_predicate: &str,
    rel: RelId,
    value: i64,
) -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![EirEpistemicLiteral {
            op: EirEpistemicOp::Know,
            negated: false,
            atom: EirAtom {
                predicate: predicate.to_string(),
                arity: 1,
                terms: vec![EirTerm::Integer(value)],
            },
        }],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: output_predicate.to_string(),
            relational_body_atoms: 1,
            wcoj_status: EpistemicWcojReductionStatus::NotWcojCandidate,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::new(),
        reduced_runtime_plan: runtime_plan_scanning_relation_into_output(rel, output_predicate),
    }
}

fn executable_with_kclique_wcoj_plan() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![epistemic_literal("gate", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "clique5".to_string(),
            relational_body_atoms: 10,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::from([(
            "__w37_helper_99".to_string(),
            xlog_core::RelId(99),
        )]),
        reduced_runtime_plan: runtime_plan_with_kclique_wcoj(),
    }
}

fn executable_with_planned_hash_route() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![epistemic_literal("gate", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "clique5".to_string(),
            relational_body_atoms: 10,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::new(),
        reduced_runtime_plan: runtime_plan_with_planned_hash_route(),
    }
}

fn executable_with_live_kclique_wcoj_literal() -> EpistemicExecutablePlan {
    let mut gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![kclique_epistemic_literal()],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "clique5".to_string(),
            relational_body_atoms: 10,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );
    for (column, bound) in gpu_plan.tuple_membership_bindings[0]
        .bound_output_columns
        .iter_mut()
        .enumerate()
    {
        *bound = Some(column);
    }

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::from([(
            "__w37_helper_99".to_string(),
            xlog_core::RelId(99),
        )]),
        reduced_runtime_plan: runtime_plan_with_live_kclique_wcoj(),
    }
}

fn executable_with_kclique_helper_metadata_only_plan() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![epistemic_literal("gate", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "clique5".to_string(),
            relational_body_atoms: 10,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::new(),
        reduced_runtime_plan: runtime_plan_with_kclique_helper_metadata_only(),
    }
}

fn executable_with_kclique_helper_scan_outside_wcoj_plan() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![epistemic_literal("gate", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "clique5".to_string(),
            relational_body_atoms: 10,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::from([(
            "__w37_helper_99".to_string(),
            xlog_core::RelId(99),
        )]),
        reduced_runtime_plan: runtime_plan_with_helper_scan_outside_wcoj(),
    }
}

fn executable_with_kclique_empty_edge_permutation_plan() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![epistemic_literal("gate", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "clique5".to_string(),
            relational_body_atoms: 10,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::from([(
            "__w37_helper_99".to_string(),
            xlog_core::RelId(99),
        )]),
        reduced_runtime_plan: runtime_plan_with_empty_edge_permutation_kclique_wcoj(),
    }
}

fn executable_with_v070_4cycle_wcoj_plan() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![epistemic_literal("gate", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "cycle4".to_string(),
            relational_body_atoms: 4,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::new(),
        reduced_runtime_plan: runtime_plan_with_v070_4cycle_wcoj(),
    }
}

fn executable_with_live_v070_4cycle_wcoj_literal() -> EpistemicExecutablePlan {
    executable_with_live_v070_4cycle_wcoj_literal_with_plan(runtime_plan_with_v070_4cycle_wcoj())
}

fn executable_with_live_triangle_wcoj_literal() -> EpistemicExecutablePlan {
    let mut gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![triangle_epistemic_literal()],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "tri".to_string(),
            relational_body_atoms: 3,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );
    for (column, bound) in gpu_plan.tuple_membership_bindings[0]
        .bound_output_columns
        .iter_mut()
        .enumerate()
    {
        *bound = Some(column);
    }

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::new(),
        reduced_runtime_plan: runtime_plan_with_triangle_wcoj(),
    }
}

fn executable_with_live_v070_4cycle_wcoj_literal_for_leader(
    leader_idx: u8,
) -> EpistemicExecutablePlan {
    executable_with_live_v070_4cycle_wcoj_literal_with_plan(
        runtime_plan_with_v070_4cycle_wcoj_for_leader(leader_idx),
    )
}

fn executable_with_live_v070_4cycle_wcoj_literal_with_plan(
    reduced_runtime_plan: ExecutionPlan,
) -> EpistemicExecutablePlan {
    let mut gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![cycle4_epistemic_literal()],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            head_predicate: "cycle4".to_string(),
            relational_body_atoms: 4,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );
    for (column, bound) in gpu_plan.tuple_membership_bindings[0]
        .bound_output_columns
        .iter_mut()
        .enumerate()
    {
        *bound = Some(column);
    }

    EpistemicExecutablePlan {
        gpu_plan,
        relation_ids: std::collections::BTreeMap::new(),
        reduced_runtime_plan,
    }
}

fn runtime_plan_with_kclique_wcoj() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["__w37_helper_99".to_string(), "clique5".to_string()],
        is_recursive: false,
    }]);
    let mut inputs: Vec<_> = (1..=10)
        .map(|rel| RirNode::Scan {
            rel: xlog_core::RelId(rel),
        })
        .collect();
    inputs[5] = RirNode::Scan {
        rel: xlog_core::RelId(99),
    };
    plan.rules_by_scc = vec![vec![
        CompiledRule {
            head: "__w37_helper_99".to_string(),
            body: RirNode::Scan {
                rel: xlog_core::RelId(2),
            },
            meta: RirMeta::default(),
        },
        CompiledRule {
            head: "clique5".to_string(),
            body: RirNode::MultiWayJoin {
                inputs,
                slot_vars: vec![vec![Some(0), Some(1)]; 10],
                output_columns: vec![ProjectExpr::Column(0)],
                fallback: Box::new(RirNode::Unit),
                plan: Some(MultiwayPlan::WcojWithPlan(kclique_order())),
                var_order: Some(xlog_ir::rir::VariableOrder::kclique(kclique_order())),
            },
            meta: RirMeta::default(),
        },
    ]];
    plan
}

fn runtime_plan_with_live_kclique_wcoj() -> ExecutionPlan {
    let mut plan = runtime_plan_with_kclique_wcoj().with_strata(vec![Stratum {
        id: 0,
        sccs: vec![0],
    }]);
    if let RirNode::MultiWayJoin { output_columns, .. } = &mut plan.rules_by_scc[0][1].body {
        *output_columns = (0..5).map(ProjectExpr::Column).collect();
    }
    plan
}

fn runtime_plan_with_planned_hash_route() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["clique5".to_string()],
        is_recursive: false,
    }]);
    plan.rules_by_scc = vec![vec![CompiledRule {
        head: "clique5".to_string(),
        body: RirNode::MultiWayJoin {
            inputs: (1..=10)
                .map(|rel| RirNode::Scan {
                    rel: xlog_core::RelId(rel),
                })
                .collect(),
            slot_vars: vec![vec![Some(0), Some(1)]; 10],
            output_columns: vec![ProjectExpr::Column(0)],
            fallback: Box::new(RirNode::Unit),
            plan: Some(MultiwayPlan::PlannedHashRoute {
                reason: PlannedHashReason::PlannerPredictsHashWins,
                planner_evidence: CostPredictionRecord {
                    wcoj_cost: 100.0,
                    hash_cost: 10.0,
                },
            }),
            var_order: None,
        },
        meta: RirMeta::default(),
    }]];
    plan
}

fn runtime_plan_with_triangle_wcoj() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["tri".to_string()],
        is_recursive: false,
    }])
    .with_strata(vec![Stratum {
        id: 0,
        sccs: vec![0],
    }]);
    plan.rules_by_scc = vec![vec![CompiledRule {
        head: "tri".to_string(),
        body: RirNode::MultiWayJoin {
            inputs: (1..=3)
                .map(|rel| RirNode::Scan {
                    rel: xlog_core::RelId(rel),
                })
                .collect(),
            slot_vars: vec![
                vec![Some(0), Some(1)],
                vec![Some(1), Some(2)],
                vec![Some(0), Some(2)],
            ],
            output_columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
            fallback: Box::new(RirNode::Unit),
            plan: None,
            var_order: None,
        },
        meta: RirMeta::default(),
    }]];
    plan
}

fn runtime_plan_with_v070_4cycle_wcoj() -> ExecutionPlan {
    runtime_plan_with_v070_4cycle_wcoj_var_order(None)
}

fn runtime_plan_with_v070_4cycle_wcoj_for_leader(leader_idx: u8) -> ExecutionPlan {
    runtime_plan_with_v070_4cycle_wcoj_var_order(Some(cycle4_var_order(leader_idx)))
}

fn runtime_plan_with_v070_4cycle_wcoj_var_order(var_order: Option<VariableOrder>) -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["cycle4".to_string()],
        is_recursive: false,
    }])
    .with_strata(vec![Stratum {
        id: 0,
        sccs: vec![0],
    }]);
    plan.rules_by_scc = vec![vec![CompiledRule {
        head: "cycle4".to_string(),
        body: RirNode::MultiWayJoin {
            inputs: (1..=4)
                .map(|rel| RirNode::Scan {
                    rel: xlog_core::RelId(rel),
                })
                .collect(),
            slot_vars: vec![
                vec![Some(0), Some(1)],
                vec![Some(1), Some(2)],
                vec![Some(2), Some(3)],
                vec![Some(3), Some(0)],
            ],
            output_columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
                ProjectExpr::Column(5),
            ],
            fallback: Box::new(RirNode::Unit),
            plan: None,
            var_order,
        },
        meta: RirMeta::default(),
    }]];
    plan
}

fn cycle4_var_order(leader_idx: u8) -> VariableOrder {
    assert!(leader_idx < 4, "4-cycle leader index must be less than 4");
    let lookup_perms = (1..4)
        .map(|offset| LookupPerm {
            input_idx: ((leader_idx as usize + offset) % 4) as u8,
            swap_cols: false,
        })
        .collect();
    let kernel_output_cols = match leader_idx {
        0 => vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
        ],
        1 => vec![
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ],
        2 => vec![
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ],
        3 => vec![
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
        ],
        _ => unreachable!(),
    };
    VariableOrder::legacy(leader_idx, lookup_perms, kernel_output_cols)
}

fn runtime_plan_with_helper_scan_outside_wcoj() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec![
            "__w37_helper_99".to_string(),
            "helper_probe".to_string(),
            "clique5".to_string(),
        ],
        is_recursive: false,
    }]);
    plan.rules_by_scc = vec![vec![
        CompiledRule {
            head: "__w37_helper_99".to_string(),
            body: RirNode::Scan {
                rel: xlog_core::RelId(2),
            },
            meta: RirMeta::default(),
        },
        CompiledRule {
            head: "helper_probe".to_string(),
            body: RirNode::Scan {
                rel: xlog_core::RelId(99),
            },
            meta: RirMeta::default(),
        },
        CompiledRule {
            head: "clique5".to_string(),
            body: RirNode::MultiWayJoin {
                inputs: (1..=10)
                    .map(|rel| RirNode::Scan {
                        rel: xlog_core::RelId(rel),
                    })
                    .collect(),
                slot_vars: vec![vec![Some(0), Some(1)]; 10],
                output_columns: vec![ProjectExpr::Column(0)],
                fallback: Box::new(RirNode::Unit),
                plan: Some(MultiwayPlan::WcojWithPlan(kclique_order())),
                var_order: Some(xlog_ir::rir::VariableOrder::kclique(kclique_order())),
            },
            meta: RirMeta::default(),
        },
    ]];
    plan
}

fn runtime_plan_with_scan_rule() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["out".to_string()],
        is_recursive: false,
    }]);
    plan.rules_by_scc = vec![vec![CompiledRule {
        head: "out".to_string(),
        body: RirNode::Scan {
            rel: xlog_core::RelId(1),
        },
        meta: RirMeta::default(),
    }]];
    plan
}

fn runtime_plan_scanning_relation_into_output(rel: RelId, output_predicate: &str) -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec![output_predicate.to_string()],
        is_recursive: false,
    }])
    .with_strata(vec![Stratum {
        id: 0,
        sccs: vec![0],
    }]);
    plan.rules_by_scc = vec![vec![CompiledRule {
        head: output_predicate.to_string(),
        body: RirNode::Scan { rel },
        meta: RirMeta::with_schema(Schema::new(vec![("x".to_string(), ScalarType::U32)])),
    }]];
    plan
}

fn runtime_plan_with_scan_rule_execution() -> ExecutionPlan {
    runtime_plan_with_scan_rule().with_strata(vec![Stratum {
        id: 0,
        sccs: vec![0],
    }])
}

fn runtime_plan_with_empty_edge_permutation_kclique_wcoj() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["__w37_helper_99".to_string(), "clique5".to_string()],
        is_recursive: false,
    }]);
    let mut inputs: Vec<_> = (1..=10)
        .map(|rel| RirNode::Scan {
            rel: xlog_core::RelId(rel),
        })
        .collect();
    inputs[5] = RirNode::Scan {
        rel: xlog_core::RelId(99),
    };
    plan.rules_by_scc = vec![vec![
        CompiledRule {
            head: "__w37_helper_99".to_string(),
            body: RirNode::Scan {
                rel: xlog_core::RelId(2),
            },
            meta: RirMeta::default(),
        },
        CompiledRule {
            head: "clique5".to_string(),
            body: RirNode::MultiWayJoin {
                inputs,
                slot_vars: vec![vec![Some(0), Some(1)]; 10],
                output_columns: vec![ProjectExpr::Column(0)],
                fallback: Box::new(RirNode::Unit),
                plan: Some(MultiwayPlan::WcojWithPlan(
                    kclique_order_without_edge_permutation(),
                )),
                var_order: Some(xlog_ir::rir::VariableOrder::kclique(
                    kclique_order_without_edge_permutation(),
                )),
            },
            meta: RirMeta::default(),
        },
    ]];
    plan
}

fn runtime_plan_with_kclique_helper_metadata_only() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["clique5".to_string()],
        is_recursive: false,
    }]);
    plan.rules_by_scc = vec![vec![CompiledRule {
        head: "clique5".to_string(),
        body: RirNode::MultiWayJoin {
            inputs: (1..=10)
                .map(|rel| RirNode::Scan {
                    rel: xlog_core::RelId(rel),
                })
                .collect(),
            slot_vars: vec![vec![Some(0), Some(1)]; 10],
            output_columns: vec![ProjectExpr::Column(0)],
            fallback: Box::new(RirNode::Unit),
            plan: Some(MultiwayPlan::WcojWithPlan(kclique_order())),
            var_order: Some(xlog_ir::rir::VariableOrder::kclique(kclique_order())),
        },
        meta: RirMeta::default(),
    }]];
    plan
}

fn kclique_order() -> KCliqueVariableOrder {
    let mut variable_positions = [u8::MAX; xlog_ir::rir::K_CLIQUE_MAX_K];
    variable_positions[..5].copy_from_slice(&[0, 1, 2, 3, 4]);

    let mut edge_permutation = [u8::MAX; xlog_ir::rir::K_CLIQUE_MAX_EDGES];
    edge_permutation[..10].copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    KCliqueVariableOrder::new(
        5,
        variable_positions,
        edge_permutation,
        Vec::new(),
        SortedLayoutSpec {
            edge_slots: vec![0, 1],
            key_columns: vec![vec![0, 1], vec![1, 0]],
        },
        vec![HelperSplitSpec {
            helper_id: 0,
            variable: 3,
            edge_slots: vec![2, 5],
        }],
        StreamGroupId(0),
    )
}

fn kclique_order_without_edge_permutation() -> KCliqueVariableOrder {
    let mut variable_positions = [u8::MAX; xlog_ir::rir::K_CLIQUE_MAX_K];
    variable_positions[..5].copy_from_slice(&[0, 1, 2, 3, 4]);

    KCliqueVariableOrder::new(
        5,
        variable_positions,
        [u8::MAX; xlog_ir::rir::K_CLIQUE_MAX_EDGES],
        Vec::new(),
        SortedLayoutSpec {
            edge_slots: vec![0, 1],
            key_columns: vec![vec![0, 1], vec![1, 0]],
        },
        vec![HelperSplitSpec {
            helper_id: 0,
            variable: 3,
            edge_slots: vec![2, 5],
        }],
        StreamGroupId(0),
    )
}

// =====================================================================
// EGB-02: tuple-key / bound-value modal membership KPI pilots.
//
// Each pilot parses a real epistemic program, compiles it through the
// production lowering boundary, and executes it on the GPU device path.
// They assert (a) the exact founded result rows, (b) device-backed
// tuple-key column reads, and (c) zero forbidden CPU fallback counters.
// =====================================================================

#[cfg(feature = "epistemic-logic-tests")]
fn egb02_run_unary_result(
    fixture: &RuntimeFixture,
    source: &str,
    unary_inputs: &[(&str, &[u32])],
    binary_inputs: &[(&str, &[(u32, u32)])],
    ternary_inputs: &[(&str, &[(u32, u32, u32)])],
    expected_key_column_reads: u32,
) -> Vec<u32> {
    let program = parse_program(source).expect("parse EGB-02 pilot program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile EGB-02 pilot plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    for &(name, rows) in unary_inputs {
        executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
    }
    for &(name, rows) in binary_inputs {
        executor.put_relation(name, upload_binary_u32(&fixture.memory, rows, "x", "y"));
    }
    for &(name, rows) in ternary_inputs {
        executor.put_relation(
            name,
            upload_ternary_u32(&fixture.memory, rows, "x", "y", "z"),
        );
    }

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 8,
                max_worlds: 2,
                max_models_per_reduction: 4,
            },
        )
        .expect("EGB-02 pilot should execute through GPU runtime path");

    // K3/K4 lock evidence: device-backed tuple-key reads, zero CPU fallback.
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        expected_key_column_reads,
        "tuple-key column reads must be device-backed for {source:?}"
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
        .require_runtime_dispatch_certification()
        .expect("EGB-02 pilot runtime evidence must remain certified");

    let mut rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download EGB-02 pilot output column 0");
    rows.sort_unstable();
    rows
}

/// K1: ground arity-1 tuple key gates candidates on a present ground tuple.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_ground_arity_one_tuple_key_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // know flag(7) holds (flag has row 7), so every node row is founded.
    let present = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred flag(u32).
        pred gated(u32).

        gated(X) :- node(X), know flag(7).
        "#,
        &[("node", &[1, 2, 3][..]), ("flag", &[7][..])],
        &[],
        &[],
        1,
    );
    assert_eq!(present, vec![1, 2, 3]);

    // know flag(7) fails (flag lacks row 7), so no node row is founded.
    let absent = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred flag(u32).
        pred gated(u32).

        gated(X) :- node(X), know flag(7).
        "#,
        &[("node", &[1, 2, 3][..]), ("flag", &[9][..])],
        &[],
        &[],
        1,
    );
    assert!(absent.is_empty(), "absent ground key must found no rows");
}

/// K1: ground arity-2 tuple key matches a specific stable-model edge tuple.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_ground_arity_two_tuple_key_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let present = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred edge(u32, u32).
        pred gated(u32).

        gated(X) :- node(X), know edge(1, 10).
        "#,
        &[("node", &[1, 2, 3][..])],
        &[("edge", &[(1, 10), (2, 20)][..])],
        &[],
        2,
    );
    assert_eq!(present, vec![1, 2, 3]);

    let absent = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred edge(u32, u32).
        pred gated(u32).

        gated(X) :- node(X), know edge(1, 99).
        "#,
        &[("node", &[1, 2, 3][..])],
        &[("edge", &[(1, 10), (2, 20)][..])],
        &[],
        2,
    );
    assert!(absent.is_empty(), "absent ground edge must found no rows");
}

/// K1: ground arity-3 tuple key matches a specific stable-model triple.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_ground_arity_three_tuple_key_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let present = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred triple(u32, u32, u32).
        pred gated(u32).

        gated(X) :- node(X), know triple(1, 10, 100).
        "#,
        &[("node", &[5, 6][..])],
        &[],
        &[("triple", &[(1, 10, 100), (2, 20, 200)][..])],
        3,
    );
    assert_eq!(present, vec![5, 6]);

    let absent = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred triple(u32, u32, u32).
        pred gated(u32).

        gated(X) :- node(X), know triple(1, 10, 999).
        "#,
        &[("node", &[5, 6][..])],
        &[],
        &[("triple", &[(1, 10, 100), (2, 20, 200)][..])],
        3,
    );
    assert!(absent.is_empty(), "absent ground triple must found no rows");
}

/// K2: a single bound variable yields exactly the founded rows, deterministic.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_single_bound_variable_tuple_key_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let source = r#"
        pred node(u32).
        pred child(u32).
        pred known_child(u32).

        known_child(X) :- node(X), know child(X).
        "#;
    let inputs_unary = &[("node", &[1, 2, 3, 4][..]), ("child", &[2, 4][..])];
    let first = egb02_run_unary_result(&fixture, source, inputs_unary, &[], &[], 1);
    assert_eq!(first, vec![2, 4]);
    // Determinism across reruns (K2).
    let second = egb02_run_unary_result(&fixture, source, inputs_unary, &[], &[], 1);
    assert_eq!(first, second);
}

/// K2: multiple bound variables type-correct on both columns and match by value.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_multiple_bound_variables_tuple_key_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // out(X,Y) :- pair(X,Y), know edge(X,Y). Only pairs that are also edges
    // are founded.
    let program = parse_program(
        r#"
        pred pair(u32, u32).
        pred edge(u32, u32).
        pred out(u32, u32).

        out(X, Y) :- pair(X, Y), know edge(X, Y).
        "#,
    )
    .expect("parse multi-bound-variable pilot program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile multi-bound pilot plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "pair",
        upload_binary_u32(&fixture.memory, &[(1, 10), (2, 20), (3, 30)], "x", "y"),
    );
    executor.put_relation(
        "edge",
        upload_binary_u32(&fixture.memory, &[(1, 10), (3, 30), (4, 40)], "x", "y"),
    );
    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 8,
                max_worlds: 2,
                max_models_per_reduction: 4,
            },
        )
        .expect("multi-bound pilot should execute through GPU runtime path");
    assert_eq!(
        result.model_membership.tuple_source_key_column_device_reads,
        2,
        "two bound columns must read two device key columns"
    );
    assert_eq!(result.semantic_trace.cpu_candidate_enumerations, 0);
    assert_eq!(result.semantic_trace.cpu_world_view_validations, 0);
    result
        .require_runtime_dispatch_certification()
        .expect("multi-bound pilot runtime evidence must remain certified");
    let mut xs = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download multi-bound X column");
    let mut ys = fixture
        .provider
        .download_column::<u32>(&result.final_output, 1)
        .expect("download multi-bound Y column");
    xs.sort_unstable();
    ys.sort_unstable();
    assert_eq!(xs, vec![1, 3]);
    assert_eq!(ys, vec![10, 30]);
}

/// K2: a repeated variable enforces value equality across both key columns.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_repeated_variable_tuple_key_enforces_equality_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // X is bound by node(X). know loop(X, X) only matches stable-model tuples
    // where both columns equal X, so loop rows like (3, 7) must NOT found X=3.
    let founded = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred loop(u32, u32).
        pred reflexive(u32).

        reflexive(X) :- node(X), know loop(X, X).
        "#,
        &[("node", &[1, 2, 3, 4][..])],
        &[("loop", &[(1, 1), (2, 5), (4, 4), (3, 7)][..])],
        &[],
        2,
    );
    // Only loop(1,1) and loop(4,4) satisfy the repeated-variable equality.
    assert_eq!(founded, vec![1, 4]);
}

/// K1/K3: an anonymous position acts as a wildcard (no equality requirement).
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_anonymous_position_tuple_key_is_wildcard_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // out(X) :- node(X), know edge(X, _). X is founded iff edge has ANY row
    // whose first column equals X, regardless of the second column.
    let founded = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred edge(u32, u32).
        pred reachable(u32).

        reachable(X) :- node(X), know edge(X, _).
        "#,
        &[("node", &[1, 2, 3, 4][..])],
        &[("edge", &[(1, 10), (3, 30), (3, 99)][..])],
        &[],
        2,
    );
    // X=1 (edge 1,10) and X=3 (edge 3,30 / 3,99) match; X=2,4 have no edge.
    assert_eq!(founded, vec![1, 3]);
}

/// K1/K3: a pure-anonymous modal atom (no bound variable) is a global gate that
/// holds iff the tuple source is non-empty.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_pure_anonymous_global_gate_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // know flag(_): holds iff flag has any row.
    let nonempty = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred flag(u32).
        pred out(u32).

        out(X) :- node(X), know flag(_).
        "#,
        &[("node", &[1, 2, 3][..]), ("flag", &[9][..])],
        &[],
        &[],
        1,
    );
    assert_eq!(nonempty, vec![1, 2, 3], "anonymous gate holds when non-empty");

    let empty = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred flag(u32).
        pred out(u32).

        out(X) :- node(X), know flag(_).
        "#,
        &[("node", &[1, 2, 3][..]), ("flag", &[][..])],
        &[],
        &[],
        1,
    );
    assert!(empty.is_empty(), "anonymous gate fails when empty");

    // know edge(_, _): arity-2 pure-anonymous global gate.
    let edge_nonempty = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred edge(u32, u32).
        pred out(u32).

        out(X) :- node(X), know edge(_, _).
        "#,
        &[("node", &[5, 6][..])],
        &[("edge", &[(1, 10)][..])],
        &[],
        2,
    );
    assert_eq!(edge_nonempty, vec![5, 6]);
}

/// K1: an arity-0 (nullary) modal atom is a global gate on a fact relation.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_arity_zero_tuple_key_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // know ready(): holds iff ready has a (zero-arity) fact row.
    let run = |ready_rows: u32| -> Vec<u32> {
        let program = parse_program(
            r#"
            pred node(u32).
            pred ready().
            pred out(u32).

            out(X) :- node(X), know ready().
            "#,
        )
        .expect("parse arity-zero modal program");
        let executable =
            compile_epistemic_gpu_execution(&program).expect("compile arity-zero modal plan");
        let mut executor = Executor::new(Arc::clone(&fixture.provider));
        for (name, rel) in &executable.relation_ids {
            executor.register_relation(*rel, name);
        }
        executor.put_relation("node", upload_unary_u32(&fixture.memory, &[1, 2, 3], "x"));
        executor.put_relation("ready", upload_zero_arity(&fixture.memory, ready_rows));
        let result = executor
            .execute_epistemic_gpu_execution(
                &executable,
                EpistemicGpuWorkspaceCapacities {
                    max_candidates: 8,
                    max_worlds: 2,
                    max_models_per_reduction: 4,
                },
            )
            .expect("arity-zero modal pilot should execute through GPU runtime path");
        assert_eq!(
            result.model_membership.tuple_source_key_column_device_reads,
            0,
            "arity-zero tuple sources read no key columns"
        );
        result
            .require_runtime_dispatch_certification()
            .expect("arity-zero pilot runtime evidence must remain certified");
        let mut rows = fixture
            .provider
            .download_column::<u32>(&result.final_output, 0)
            .expect("download arity-zero output column 0");
        rows.sort_unstable();
        rows
    };
    assert_eq!(run(1), vec![1, 2, 3], "nullary gate holds when fact present");
    assert!(run(0).is_empty(), "nullary gate fails when fact absent");
}

/// K3: a rule mixing a per-row (bound-variable) modal literal with a global
/// gate (ground) modal literal fails closed with a typed diagnostic rather than
/// emitting rows that ignore the global gate.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_mixed_per_row_and_global_modal_is_fail_closed() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let program = parse_program(
        r#"
        pred node(u32).
        pred child(u32).
        pred flag(u32).
        pred out(u32).

        out(X) :- node(X), know child(X), know flag(7).
        "#,
    )
    .expect("parse mixed per-row + global modal program");
    let executable = match compile_epistemic_gpu_execution(&program) {
        Ok(executable) => executable,
        Err(xlog_core::XlogError::UnsupportedEpistemicConstruct { .. })
        | Err(xlog_core::XlogError::UnsafeVariable(_))
        | Err(xlog_core::XlogError::Type(_))
        | Err(xlog_core::XlogError::Compilation(_)) => return,
        Err(other) => panic!("unexpected compile error: {other:?}"),
    };
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation("node", upload_unary_u32(&fixture.memory, &[1, 2, 3], "x"));
    executor.put_relation("child", upload_unary_u32(&fixture.memory, &[1], "x"));
    executor.put_relation("flag", upload_unary_u32(&fixture.memory, &[9], "x"));
    let err = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 8,
                max_worlds: 2,
                max_models_per_reduction: 4,
            },
        )
        .err()
        .expect("mixed per-row + global modal membership must fail closed");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, .. } => {
            assert!(
                construct.contains("mixed per-row and global modal membership"),
                "unexpected construct: {construct}"
            );
        }
        other => panic!("expected mixed-membership diagnostic, got {other:?}"),
    }
}

/// K2/K3: an empty `possible` tuple source founds nothing; `not possible`
/// founds everything (the negated empty membership).
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_empty_possible_tuple_source_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let possible_rows = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred maybe(u32).
        pred out(u32).

        out(X) :- node(X), possible maybe(X).
        "#,
        &[("node", &[1, 2, 3][..]), ("maybe", &[][..])],
        &[],
        &[],
        1,
    );
    assert!(
        possible_rows.is_empty(),
        "empty possible source must found no rows"
    );

    let not_possible_rows = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred maybe(u32).
        pred out(u32).

        out(X) :- node(X), not possible maybe(X).
        "#,
        &[("node", &[1, 2, 3][..]), ("maybe", &[][..])],
        &[],
        &[],
        1,
    );
    assert_eq!(not_possible_rows, vec![1, 2, 3]);
}

/// K3: an unbound variable in a modal atom is rejected with a typed diagnostic
/// before any result materialization.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_unbound_modal_variable_is_fail_closed() {
    // Y appears only inside the modal atom; it is not bound by any relational
    // body atom or head term, so the program is out of fragment.
    let program = parse_program(
        r#"
        pred node(u32).
        pred edge(u32, u32).
        pred out(u32).

        out(X) :- node(X), know edge(X, Y).
        "#,
    )
    .expect("parse unbound-modal-variable program");
    let err = compile_epistemic_gpu_execution(&program)
        .err()
        .expect("unbound modal variable must be rejected before execution");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { .. }
        | xlog_core::XlogError::UnsafeVariable(_)
        | xlog_core::XlogError::Type(_)
        | xlog_core::XlogError::Compilation(_) => {}
        other => panic!("expected typed fail-closed diagnostic, got {other:?}"),
    }
}

/// K3: a type mismatch between a bound variable and the tuple-key column is
/// rejected with a typed diagnostic before result materialization.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_type_mismatch_bound_variable_is_fail_closed() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // node carries a Symbol column; child carries a U32 column. Binding X
    // (Symbol) against child's U32 key column is a type error.
    let program = parse_program(
        r#"
        pred node(symbol).
        pred child(u32).
        pred out(symbol).

        out(X) :- node(X), know child(X).
        "#,
    )
    .expect("parse type-mismatch program");
    let executable = match compile_epistemic_gpu_execution(&program) {
        Ok(executable) => executable,
        Err(xlog_core::XlogError::UnsupportedEpistemicConstruct { .. })
        | Err(xlog_core::XlogError::UnsafeVariable(_))
        | Err(xlog_core::XlogError::Type(_))
        | Err(xlog_core::XlogError::Compilation(_)) => return,
        Err(other) => panic!("unexpected compile error: {other:?}"),
    };
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    executor.put_relation(
        "node",
        upload_unary_typed_u32(&fixture.memory, &[1, 2], "x", ScalarType::Symbol),
    );
    executor.put_relation("child", upload_unary_u32(&fixture.memory, &[1], "x"));
    let err = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 8,
                max_worlds: 2,
                max_models_per_reduction: 4,
            },
        )
        .err()
        .expect("type mismatch must be rejected before result materialization");
    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { .. }
        | xlog_core::XlogError::UnsafeVariable(_)
        | xlog_core::XlogError::Type(_)
        | xlog_core::XlogError::Compilation(_) => {}
        other => panic!("expected typed type-mismatch diagnostic, got {other:?}"),
    }
}

/// K1/K3: pure-ground `not know` / `possible` / `not possible` global gates.
/// These ride the global membership gate (no bound output column), which must
/// honor body-literal semantics including negation.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_ground_global_gate_honors_negation_and_modality() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };

    // not know flag(7): holds iff flag(7) is absent.
    let not_know_absent = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred flag(u32).
        pred out(u32).

        out(X) :- node(X), not know flag(7).
        "#,
        &[("node", &[1, 2, 3][..]), ("flag", &[9][..])],
        &[],
        &[],
        1,
    );
    assert_eq!(not_know_absent, vec![1, 2, 3], "not know of absent holds");

    let not_know_present = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred flag(u32).
        pred out(u32).

        out(X) :- node(X), not know flag(7).
        "#,
        &[("node", &[1, 2, 3][..]), ("flag", &[7][..])],
        &[],
        &[],
        1,
    );
    assert!(
        not_know_present.is_empty(),
        "not know of present must found no rows"
    );

    // possible flag(7): holds iff flag(7) is present (single-world FAEEL).
    let possible_present = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred flag(u32).
        pred out(u32).

        out(X) :- node(X), possible flag(7).
        "#,
        &[("node", &[1, 2, 3][..]), ("flag", &[7][..])],
        &[],
        &[],
        1,
    );
    assert_eq!(possible_present, vec![1, 2, 3], "possible present holds");

    let possible_absent = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred flag(u32).
        pred out(u32).

        out(X) :- node(X), possible flag(7).
        "#,
        &[("node", &[1, 2, 3][..]), ("flag", &[9][..])],
        &[],
        &[],
        1,
    );
    assert!(
        possible_absent.is_empty(),
        "possible absent must found no rows"
    );
}

/// K1/K2: a modal atom mixing a bound variable and a ground value matches by
/// value on both positions.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb02_mixed_bound_and_ground_tuple_key_through_gpu_membership() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // out(X) :- node(X), know edge(X, 10). X founded iff edge(X, 10) holds.
    let founded = egb02_run_unary_result(
        &fixture,
        r#"
        pred node(u32).
        pred edge(u32, u32).
        pred out(u32).

        out(X) :- node(X), know edge(X, 10).
        "#,
        &[("node", &[1, 2, 3][..])],
        &[("edge", &[(1, 10), (2, 20)][..])],
        &[],
        2,
    );
    assert_eq!(founded, vec![1]);
}

// ---------------------------------------------------------------------------
// EGB-01: arbitrary-EIR candidate-world enumeration.
//
// These pilots prove the production device path
// (compile_epistemic_gpu_execution -> execute_epistemic_gpu_execution) derives
// the candidate assumption space FROM the program's EIR epistemic literals --
// no hand-supplied EpistemicInterpretation candidate lists -- generates the
// full 2^literal_count lattice on device, evaluates each candidate against the
// reduced stable-model semantics through the production runtime path, and emits
// the K2 trace counts. They route through the same single-plan GPU runtime path
// as every other accepted epistemic execution; nothing here touches the CPU
// fixture layer (run_generate_propagate_test).
// ---------------------------------------------------------------------------

/// EGB-01 K1/K2: a program with MULTIPLE epistemic literals derives its full
/// candidate space (2^literal_count) FROM the EIR program -- no fixture list --
/// and emits every required trace count (generated, propagated, tested,
/// accepted, rejected, rejection reasons) from device-derived semantics.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb01_multi_literal_program_enumerates_candidate_space_from_eir() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // Three epistemic literals across distinct predicates -> 2^3 = 8 candidates,
    // each a distinct assumption assignment derived purely from the EIR program.
    let program = parse_program(
        r#"
        pred seed(u32).
        pred a(u32).
        pred b(u32).
        pred c(u32).
        pred out(u32).

        out(X) :- seed(X), know a(X), possible b(X), not possible c(X).
        "#,
    )
    .expect("parse EGB-01 multi-literal program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile EGB-01 multi-literal plan");
    // The candidate space is derived from the program's epistemic literals,
    // not supplied by the caller.
    assert_eq!(
        executable.gpu_plan.epistemic_literals.len(),
        3,
        "candidate space must be derived from the three EIR epistemic literals"
    );

    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    // a(7) and b(7) present, c(7) absent => exactly the all-assumptions-satisfied
    // candidate (index 7) is accepted; every other candidate is rejected.
    for (name, rows) in [
        ("seed", &[7][..]),
        ("a", &[7][..]),
        ("b", &[7][..]),
        ("c", &[][..]),
    ] {
        executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
    }

    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 8,
                max_worlds: 4,
                max_models_per_reduction: 1,
            },
        )
        .expect("EGB-01 multi-literal program should enumerate and evaluate on device");

    // K2: every required trace count is emitted from device-derived semantics.
    let trace = &result.semantic_trace;
    assert_eq!(trace.generated_candidates, 8, "generated = 2^3");
    assert_eq!(trace.guesses, 24, "guesses = 8 candidates * 3 literals");
    assert_eq!(trace.propagated_candidates, 8, "propagated");
    assert_eq!(trace.tested_candidates, 8, "tested");
    assert_eq!(trace.accepted_candidates, 1, "accepted");
    assert_eq!(trace.rejected_candidates, 7, "rejected");
    assert_eq!(trace.accepted_candidate_indices, vec![7]);
    assert_eq!(
        trace.rejected_candidate_indices,
        vec![0, 1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        trace.rejection_reasons.len(),
        7,
        "a rejection reason is emitted for each rejected candidate"
    );
    assert!(
        trace
            .typed_rejection_reasons()
            .expect("decode typed rejection reasons")
            .iter()
            .all(|reason| *reason == EpistemicGpuRejectionReason::UnsatisfiedMembership),
        "rejected candidates fail membership against the EIR-derived assumptions"
    );
    // K4: no CPU fallback in candidate enumeration / world-view validation.
    assert_eq!(trace.cpu_candidate_enumerations, 0);
    assert_eq!(trace.cpu_world_view_validations, 0);
    assert!(result.prepared.preflight.cpu_fallbacks.is_zero());
    assert_eq!(result.transfer_budget.per_candidate_host_round_trips, 0);

    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download EGB-01 multi-literal output");
    assert_eq!(rows, vec![7]);
    result
        .require_runtime_dispatch_certification()
        .expect("EGB-01 multi-literal result must retain dispatch + semantic certification");
}

/// EGB-01 K3: an EIR-derived candidate space where EVERY candidate is rejected
/// returns Ok cleanly with an empty accepted world-view set -- distinguishable
/// from execution failure (which returns Err) -- and still emits a full
/// rejection trace.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb01_empty_accepted_world_view_is_distinct_from_failure() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    // know a(X) can never hold: a has no rows, so no candidate that assumes a
    // is founded, and the candidate that does not assume a fails the literal.
    // Every candidate in the 2^2 space is therefore rejected.
    let program = parse_program(
        r#"
        pred seed(u32).
        pred a(u32).
        pred b(u32).
        pred out(u32).

        out(X) :- seed(X), know a(X), possible b(X).
        "#,
    )
    .expect("parse EGB-01 empty-world-view program");
    let executable =
        compile_epistemic_gpu_execution(&program).expect("compile EGB-01 empty-world-view plan");
    let mut executor = Executor::new(Arc::clone(&fixture.provider));
    for (name, rel) in &executable.relation_ids {
        executor.register_relation(*rel, name);
    }
    for (name, rows) in [("seed", &[7][..]), ("a", &[][..]), ("b", &[7][..])] {
        executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
    }

    // Execution SUCCEEDS (Ok) even though zero candidates are accepted -- this is
    // the empty-accepted-world-view vs execution-failure distinction.
    let result = executor
        .execute_epistemic_gpu_execution(
            &executable,
            EpistemicGpuWorkspaceCapacities {
                max_candidates: 4,
                max_worlds: 4,
                max_models_per_reduction: 1,
            },
        )
        .expect("empty accepted world view must be Ok, not an execution failure");

    let trace = &result.semantic_trace;
    assert_eq!(trace.generated_candidates, 4, "generated = 2^2");
    assert_eq!(trace.tested_candidates, 4, "tested");
    assert_eq!(trace.accepted_candidates, 0, "no candidate is accepted");
    assert!(
        trace.accepted_candidate_indices.is_empty(),
        "accepted world-view set is empty"
    );
    assert_eq!(trace.accepted_world_views, 0);
    assert_eq!(trace.rejected_candidates, 4, "every candidate is rejected");
    assert_eq!(
        trace.rejection_reasons.len(),
        4,
        "a rejection reason is emitted for each rejected candidate"
    );
    assert_eq!(trace.cpu_candidate_enumerations, 0);
    assert_eq!(trace.cpu_world_view_validations, 0);

    // Empty accepted set => empty final output, not an error.
    assert_eq!(result.final_result_transfer.final_output_rows, 0);
    let rows = fixture
        .provider
        .download_column::<u32>(&result.final_output, 0)
        .expect("download empty EGB-01 output column");
    assert!(rows.is_empty(), "empty accepted world view yields no rows");
    result
        .require_runtime_dispatch_certification()
        .expect("empty-world-view result must still be a certified runtime dispatch");
}

/// EGB-01 K3: repeated deterministic runs of the same EIR-derived enumeration
/// produce identical candidate AND result sets.
#[cfg(feature = "epistemic-logic-tests")]
#[test]
fn egb01_repeated_runs_are_deterministic() {
    let Some(fixture) = runtime_fixture() else {
        return;
    };
    let source = r#"
        pred seed(u32).
        pred a(u32).
        pred b(u32).
        pred c(u32).
        pred out(u32).

        out(X) :- seed(X), know a(X), possible b(X), not possible c(X).
        "#;
    let capacities = EpistemicGpuWorkspaceCapacities {
        max_candidates: 8,
        max_worlds: 4,
        max_models_per_reduction: 1,
    };

    let run_once = || {
        let program = parse_program(source).expect("parse EGB-01 determinism program");
        let executable =
            compile_epistemic_gpu_execution(&program).expect("compile EGB-01 determinism plan");
        let mut executor = Executor::new(Arc::clone(&fixture.provider));
        for (name, rel) in &executable.relation_ids {
            executor.register_relation(*rel, name);
        }
        for (name, rows) in [
            ("seed", &[7, 8][..]),
            ("a", &[7, 8][..]),
            ("b", &[7, 8][..]),
            ("c", &[][..]),
        ] {
            executor.put_relation(name, upload_unary_u32(&fixture.memory, rows, "x"));
        }
        let result = executor
            .execute_epistemic_gpu_execution(&executable, capacities)
            .expect("EGB-01 determinism run should execute on device");
        let mut rows = fixture
            .provider
            .download_column::<u32>(&result.final_output, 0)
            .expect("download EGB-01 determinism output");
        rows.sort_unstable();
        (
            result.semantic_trace.accepted_candidate_indices.clone(),
            result.semantic_trace.rejected_candidate_indices.clone(),
            result.semantic_trace.rejection_reasons.clone(),
            rows,
        )
    };

    let first = run_once();
    let second = run_once();
    assert_eq!(
        first, second,
        "repeated deterministic runs must produce identical candidate and result sets"
    );
    // Candidate set was genuinely enumerated (not empty), so determinism is meaningful.
    assert_eq!(first.0, vec![7], "accepted candidate index is the all-true assignment");
    assert_eq!(first.3, vec![7, 8], "accepted world view materializes both seed rows");
}
