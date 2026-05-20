use xlog_cuda::provider::HostTransferStats;
use xlog_ir::{
    rir::{
        HelperSplitSpec, KCliqueVariableOrder, MultiwayPlan, ProjectExpr, RirNode,
        SortedLayoutSpec, StreamGroupId,
    },
    CompiledRule, EirAtom, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp,
    EpistemicExecutablePlan, EpistemicGpuPlan, EpistemicReductionPlan,
    EpistemicWcojReductionStatus, ExecutionPlan, RirMeta, Scc,
};
use xlog_runtime::{
    EpistemicGpuCandidateGenerationTrace, EpistemicGpuCandidateValidationTrace,
    EpistemicGpuFinalResultMaterializationTrace, EpistemicGpuFinalTupleMaterializationTrace,
    EpistemicGpuKernelTimingTrace, EpistemicGpuMaterializationTrace,
    EpistemicGpuModelMembershipSource, EpistemicGpuModelMembershipTrace,
    EpistemicGpuPropagationTrace, EpistemicGpuRuntimeCounters, EpistemicGpuRuntimePreflight,
    EpistemicGpuRuntimeTrace, EpistemicGpuRuntimeWcojCertification,
    EpistemicGpuTransferBudgetTrace, EpistemicGpuWorkspaceCapacities, EpistemicGpuWorkspaceLayout,
    EpistemicGpuWorkspaceResetTrace, EpistemicGpuWorldViewValidationTrace,
};

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
fn runtime_preflight_rejects_nonzero_cpu_fallback_counters() {
    let mut executable = executable_with_kclique_wcoj_plan();
    executable.gpu_plan.cpu_fallbacks.candidate_enumeration = 1;

    let err = EpistemicGpuRuntimePreflight::for_executable_plan(
        &executable,
        EpistemicGpuWorkspaceCapacities {
            max_candidates: 8,
            max_worlds: 4,
            max_models_per_reduction: 6,
        },
    )
    .unwrap_err();

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPU runtime preflight");
            assert!(context.contains("nonzero CPU fallback counters"));
        }
        other => panic!("expected typed fallback counter error, got {other:?}"),
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
        }
    );
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
        wcoj_layout_sort_invocation_count: 8,
        ..EpistemicGpuRuntimeCounters::default()
    };
    let after = EpistemicGpuRuntimeCounters {
        wcoj_clique5_dispatch_count: 3,
        kclique_metadata_build_count: 5,
        kclique_metadata_build_nanos: 52,
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
fn workspace_reset_runtime_path_uses_device_memsets_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("fn reset_epistemic_gpu_workspace"));
    assert!(source.contains("memset_zeros(&mut workspace.candidate_assumptions)"));
    assert!(source.contains("memset_zeros(&mut workspace.world_views)"));
    assert!(source.contains("memset_zeros(&mut workspace.model_membership)"));
    assert!(source.contains("memset_zeros(&mut workspace.rejection_reasons)"));
    assert!(!source.contains("upload_epistemic_gpu_workspace_reset"));
    assert!(!source.contains("copy_epistemic_gpu_workspace_reset_from_host"));
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
fn cuda_event_timing_trace_converts_milliseconds_to_nanoseconds() {
    let timing = EpistemicGpuKernelTimingTrace::from_cuda_elapsed_ms(0.125).unwrap();

    assert_eq!(timing.cuda_event_pairs, 1);
    assert_eq!(timing.timing_sync_ops, 1);
    assert_eq!(timing.kernel_elapsed_nanos, 125_000);
    assert!(timing.is_recorded());
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
fn candidate_generation_runtime_path_launches_epistemic_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("fn generate_epistemic_gpu_candidates"));
    assert!(source.contains("EPISTEMIC_GENERATE_CANDIDATE_ASSUMPTIONS_U8"));
    assert!(source.contains("func.clone().launch"));
    assert!(cuda.contains("epistemic_generate_candidate_assumptions_u8"));
    assert!(manifest.contains("\"epistemic_generate_candidate_assumptions_u8\""));
    assert!(!source.contains("upload_epistemic_candidate_assumptions"));
    assert!(!source.contains("copy_epistemic_candidates_from_host"));
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
fn propagation_runtime_path_launches_epistemic_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("fn propagate_epistemic_gpu_candidates"));
    assert!(source.contains("EPISTEMIC_PROPAGATE_CANDIDATES_U8"));
    assert!(source.contains("&workspace.candidate_assumptions"));
    assert!(source.contains("&mut workspace.world_views"));
    assert!(source.contains("&mut workspace.rejection_reasons"));
    assert!(cuda.contains("epistemic_propagate_candidates_u8"));
    assert!(manifest.contains("\"epistemic_propagate_candidates_u8\""));
    assert!(!source.contains("upload_epistemic_propagation"));
    assert!(!source.contains("copy_epistemic_propagation_from_host"));
}

#[test]
fn execution_result_records_candidate_and_propagation_kernel_traces() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("pub candidate_generation: EpistemicGpuCandidateGenerationTrace"));
    assert!(source.contains("pub propagation: EpistemicGpuPropagationTrace"));
    assert!(source.contains("let candidate_generation = self.generate_epistemic_gpu_candidates"));
    assert!(source.contains("let propagation = self.propagate_epistemic_gpu_candidates"));
    assert!(source.contains("candidate_generation,"));
    assert!(source.contains("propagation,"));

    let generation_pos = source
        .find("let candidate_generation = self.generate_epistemic_gpu_candidates")
        .expect("candidate-generation launch in execution path");
    let propagation_pos = source
        .find("let propagation = self.propagate_epistemic_gpu_candidates")
        .expect("propagation launch in execution path");
    let reduced_dispatch_pos = source
        .find("let _reduced_return = self.execute_plan(&executable.reduced_runtime_plan)?")
        .expect("reduced production runtime dispatch");

    assert!(generation_pos < propagation_pos);
    assert!(propagation_pos < reduced_dispatch_pos);
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
fn candidate_validation_runtime_path_launches_epistemic_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("fn validate_epistemic_gpu_candidates"));
    assert!(source.contains("EPISTEMIC_VALIDATE_CANDIDATE_BITS_U8"));
    assert!(source.contains("&workspace.candidate_assumptions"));
    assert!(source.contains("&workspace.world_views"));
    assert!(source.contains("&mut workspace.rejection_reasons"));
    assert!(cuda.contains("epistemic_validate_candidate_bits_u8"));
    assert!(manifest.contains("\"epistemic_validate_candidate_bits_u8\""));
    assert!(!source.contains("upload_epistemic_validation"));
    assert!(!source.contains("copy_epistemic_validation_from_host"));
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
}

#[test]
fn model_membership_runtime_path_launches_tuple_source_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("fn populate_epistemic_gpu_model_membership_from_tuple_sources"));
    assert!(source.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_U8"));
    assert!(source.contains("gpu_plan.tuple_membership_bindings"));
    assert!(source.contains("self.store()"));
    assert!(source.contains("get(binding.predicate.as_str())"));
    assert!(source.contains("source_relation.num_rows_device()"));
    assert!(source.contains("&workspace.candidate_assumptions"));
    assert!(source.contains("&workspace.world_views"));
    assert!(source.contains("&mut workspace.model_membership"));
    assert!(source.contains("&mut workspace.rejection_reasons"));
    assert!(source.contains("EpistemicGpuModelMembershipSource::StableModelTupleBuffer"));
    assert!(cuda.contains("epistemic_populate_model_membership_from_tuple_source_u8"));
    assert!(cuda.contains("tuple_source_row_count"));
    assert!(manifest.contains("\"epistemic_populate_model_membership_from_tuple_source_u8\""));
    assert!(!source.contains("upload_epistemic_model_membership"));
    assert!(!source.contains("copy_epistemic_model_membership_from_host"));
    assert!(!source.contains("dtoh_epistemic_model_membership_row_count"));
    assert!(!source.contains("cached_row_count().expect(\"epistemic model"));
}

#[test]
fn model_membership_runtime_path_launches_arity_one_tuple_key_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let provider = include_str!("../../xlog-cuda/src/provider/mod.rs");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("TupleSourceLaunch::ArityOne"));
    assert!(source.contains("binding.key_columns.as_slice()"));
    assert!(source.contains("source_relation.column(key_col)"));
    assert!(source.contains("column_type(key_col)"));
    assert!(source.contains("key_col0_type.size_bytes()"));
    assert!(source.contains("expected_key_col0_bits"));
    assert!(source.contains("expected_key_col0_type_code"));
    assert!(source.contains("tuple_source_key_column_device_reads"));
    assert!(source.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY1_U8"));
    assert!(provider.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY1_U8"));
    assert!(cuda.contains("epistemic_populate_model_membership_from_tuple_source_arity1_u8"));
    assert!(cuda.contains("tuple_key_col0"));
    assert!(cuda.contains("tuple_key_col0_width"));
    assert!(
        manifest.contains("\"epistemic_populate_model_membership_from_tuple_source_arity1_u8\"")
    );
    assert!(!source.contains("current GPU tuple-source kernel only certifies zero-arity tuples"));
    assert!(!source.contains("copy_epistemic_tuple_keys_from_host"));
    assert!(!source.contains("dtoh_epistemic_tuple_key"));
}

#[test]
fn model_membership_runtime_path_launches_arity_two_tuple_key_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let provider = include_str!("../../xlog-cuda/src/provider/mod.rs");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("TupleSourceLaunch::ArityTwo"));
    assert!(source.contains("&[key_col0, key_col1]"));
    assert!(source.contains("source_relation.column(key_col0)"));
    assert!(source.contains("source_relation.column(key_col1)"));
    assert!(source.contains("expected_key_col1_bits"));
    assert!(source.contains("expected_key_col1_type_code"));
    assert!(source.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY2_U8"));
    assert!(provider.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY2_U8"));
    assert!(cuda.contains("epistemic_populate_model_membership_from_tuple_source_arity2_u8"));
    assert!(cuda.contains("tuple_key_col1"));
    assert!(cuda.contains("tuple_key_col1_width"));
    assert!(
        manifest.contains("\"epistemic_populate_model_membership_from_tuple_source_arity2_u8\"")
    );
    assert!(!source.contains("copy_epistemic_tuple_keys_from_host"));
    assert!(!source.contains("dtoh_epistemic_tuple_key"));
}

#[test]
fn model_membership_runtime_path_launches_arity_three_tuple_key_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let provider = include_str!("../../xlog-cuda/src/provider/mod.rs");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("TupleSourceLaunch::ArityThree"));
    assert!(source.contains("&[key_col0, key_col1, key_col2]"));
    assert!(source.contains("source_relation.column(key_col2)"));
    assert!(source.contains("expected_key_col2_bits"));
    assert!(source.contains("expected_key_col2_type_code"));
    assert!(source.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY3_U8"));
    assert!(provider.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY3_U8"));
    assert!(cuda.contains("epistemic_populate_model_membership_from_tuple_source_arity3_u8"));
    assert!(cuda.contains("tuple_key_col2"));
    assert!(cuda.contains("tuple_key_col2_width"));
    assert!(
        manifest.contains("\"epistemic_populate_model_membership_from_tuple_source_arity3_u8\"")
    );
    assert!(
        !source.contains("current GPU tuple-source kernels certify arity-zero through arity-two")
    );
    assert!(!source.contains("copy_epistemic_tuple_keys_from_host"));
    assert!(!source.contains("dtoh_epistemic_tuple_key"));
}

#[test]
fn model_membership_runtime_path_launches_generic_arity_tuple_key_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let provider = include_str!("../../xlog-cuda/src/provider/mod.rs");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("TupleSourceLaunch::ArityN"));
    assert!(source.contains("key_col_ptrs"));
    assert!(source.contains("key_col_widths"));
    assert!(source.contains("expected_key_bits"));
    assert!(source.contains("expected_key_type_codes"));
    assert!(source.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY_N_U8"));
    assert!(provider.contains("EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY_N_U8"));
    assert!(cuda.contains("epistemic_populate_model_membership_from_tuple_source_arity_n_u8"));
    assert!(cuda.contains("epistemic_tuple_key_row_matches_arity_n"));
    assert!(cuda.contains("tuple_key_col_ptrs"));
    assert!(cuda.contains("tuple_key_col_widths"));
    assert!(
        manifest.contains("\"epistemic_populate_model_membership_from_tuple_source_arity_n_u8\"")
    );
    assert!(
        !source.contains("current GPU tuple-source kernels certify arity-zero through arity-three")
    );
    assert!(!source.contains("copy_epistemic_tuple_keys_from_host"));
    assert!(!source.contains("dtoh_epistemic_tuple_key"));
}

#[test]
fn model_membership_runtime_path_uses_bound_output_columns_for_variable_tuple_keys() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");

    assert!(source.contains("output: &CudaBuffer"));
    assert!(source
        .contains("let _reduced_return = self.execute_plan(&executable.reduced_runtime_plan)?"));
    assert!(source.contains("let output_relation = executable"));
    assert!(source.contains(".head_predicate"));
    assert!(source.contains("self.clone_buffer(reduced_output)?"));
    assert!(source.contains("&output,"));
    assert!(source.contains("&executable.gpu_plan,"));
    assert!(source.contains("EirTerm::Variable(variable_name)"));
    assert!(source.contains("binding.bound_output_columns[term_index]"));
    assert!(source.contains("output.column(bound_col_index)"));
    assert!(source.contains("tuple_key_match_modes"));
    assert!(source.contains("bound_value_col_ptrs"));
    assert!(source.contains("bound_value_col_widths"));
    assert!(source.contains("bound_value_row_count: output.num_rows_device()"));
    assert!(cuda.contains("tuple_key_match_modes"));
    assert!(cuda.contains("bound_value_col_ptrs"));
    assert!(cuda.contains("bound_value_col_widths"));
    assert!(cuda.contains("bound_value_row_count"));
    assert!(cuda.contains("epistemic_tuple_key_bound_cell_matches"));
    assert!(
        !source.contains("term {term:?} requires a bound value buffer for GPU tuple-key matching")
    );
    assert!(!source.contains("copy_epistemic_tuple_keys_from_host"));
    assert!(!source.contains("dtoh_epistemic_tuple_key"));
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
fn world_view_validation_runtime_path_launches_epistemic_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("fn validate_epistemic_gpu_world_views"));
    assert!(source.contains("EPISTEMIC_VALIDATE_WORLD_VIEWS_U8"));
    assert!(source.contains("&workspace.model_membership"));
    assert!(source.contains("&workspace.world_views"));
    assert!(source.contains("&mut workspace.rejection_reasons"));
    assert!(cuda.contains("epistemic_validate_world_views_u8"));
    assert!(manifest.contains("\"epistemic_validate_world_views_u8\""));
    assert!(!source.contains("upload_epistemic_world_view_validation"));
    assert!(!source.contains("copy_epistemic_world_view_validation_from_host"));
}

#[test]
fn execution_result_records_model_membership_and_world_view_validation_after_reduced_dispatch() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("pub model_membership: EpistemicGpuModelMembershipTrace"));
    assert!(source.contains("pub world_view_validation: EpistemicGpuWorldViewValidationTrace"));
    assert!(source.contains(
        "let model_membership = self.populate_epistemic_gpu_model_membership_from_tuple_sources"
    ));
    assert!(source.contains("let world_view_validation = self.validate_epistemic_gpu_world_views"));
    assert!(source.contains("model_membership,"));
    assert!(source.contains("world_view_validation,"));

    let reduced_dispatch_pos = source
        .find("let _reduced_return = self.execute_plan(&executable.reduced_runtime_plan)?")
        .expect("reduced production runtime dispatch");
    let reduced_output_clone_pos = source
        .find("self.clone_buffer(reduced_output)?")
        .expect("named reduced output clone");
    let wcoj_gate_pos = source
        .find("trace.require_wcoj_certification()?")
        .expect("runtime WCOJ certification gate");
    let membership_pos = source
        .find("let model_membership = self.populate_epistemic_gpu_model_membership_from_tuple_sources")
        .expect("model-membership launch in execution path");
    let world_validation_pos = source
        .find("let world_view_validation = self.validate_epistemic_gpu_world_views")
        .expect("world-view validation launch in execution path");
    let materialization_pos = source
        .find("self.materialize_epistemic_gpu_candidates")
        .expect("materialization launch in execution path");

    assert!(reduced_dispatch_pos < membership_pos);
    assert!(reduced_dispatch_pos < wcoj_gate_pos);
    assert!(wcoj_gate_pos < membership_pos);
    assert!(wcoj_gate_pos < reduced_output_clone_pos);
    assert!(reduced_output_clone_pos < membership_pos);
    assert!(membership_pos < world_validation_pos);
    assert!(world_validation_pos < materialization_pos);
}

#[test]
fn execution_result_records_validation_kernel_trace_before_reduced_dispatch() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("pub candidate_validation: EpistemicGpuCandidateValidationTrace"));
    assert!(source.contains("let candidate_validation = self.validate_epistemic_gpu_candidates"));
    assert!(source.contains("candidate_validation,"));

    let propagation_pos = source
        .find("let propagation = self.propagate_epistemic_gpu_candidates")
        .expect("propagation launch in execution path");
    let validation_pos = source
        .find("let candidate_validation = self.validate_epistemic_gpu_candidates")
        .expect("candidate-validation launch in execution path");
    let reduced_dispatch_pos = source
        .find("let _reduced_return = self.execute_plan(&executable.reduced_runtime_plan)?")
        .expect("reduced production runtime dispatch");

    assert!(propagation_pos < validation_pos);
    assert!(validation_pos < reduced_dispatch_pos);
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
fn final_tuple_materialization_trace_records_device_tuple_buffer_without_host_writes() {
    let trace =
        EpistemicGpuFinalTupleMaterializationTrace::for_counts(2, 16, 128, 3, 8, 2, 4).unwrap();

    assert_eq!(trace.output_column_count, 2);
    assert_eq!(trace.output_row_capacity, 16);
    assert_eq!(trace.tuple_bytes_capacity, 128);
    assert_eq!(trace.output_row_count_device_reads, 1);
    assert_eq!(trace.model_membership_bytes_checked, 192);
    assert_eq!(trace.world_view_slots_checked, 8);
    assert_eq!(trace.row_filter_count, 0);
    assert_eq!(trace.negated_row_filter_count, 0);
    assert_eq!(trace.final_row_count_device_writes, 1);
    assert_eq!(trace.kernel_launches, 2);
    assert_eq!(trace.host_write_ops, 0);
    assert!(!trace.kernel_timing.is_recorded());
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
fn staging_runtime_paths_record_cuda_event_timing_for_each_kernel() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("fn time_epistemic_gpu_kernel_launch"));
    assert!(source.contains("record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))"));
    assert!(source.contains("EpistemicGpuKernelTimingTrace::from_cuda_elapsed_ms"));
    assert!(source.contains("pub aggregate_kernel_timing: EpistemicGpuKernelTimingTrace"));
    assert!(source.contains("fn aggregate_kernel_timing(&self) -> EpistemicGpuKernelTimingTrace"));
    assert!(source.contains(".map(|result| result.aggregate_kernel_timing())"));

    for label in [
        "epistemic GPU candidate generation",
        "epistemic GPU candidate propagation",
        "epistemic GPU candidate validation",
        "epistemic GPU model membership",
        "epistemic GPU tuple-source model membership",
        "epistemic GPU world-view validation",
        "epistemic GPU candidate materialization",
        "epistemic GPU final result materialization",
        "epistemic GPU final tuple materialization",
    ] {
        assert!(
            source.match_indices(label).any(|(label_pos, _)| {
                source[..label_pos]
                    .rfind("time_epistemic_gpu_kernel_launch")
                    .is_some_and(|timing_call_prefix| label_pos - timing_call_prefix < 96)
            }),
            "{label} must be passed directly to the timing launch helper"
        );
    }
    assert_eq!(
        source.matches(".with_kernel_timing(kernel_timing)").count(),
        9
    );
}

#[test]
fn execution_result_records_hot_path_transfer_budget_without_resetting_stats() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("pub transfer_budget: EpistemicGpuTransferBudgetTrace"));
    assert!(source.contains("pub final_result_transfer: EpistemicGpuFinalResultTransferTrace"));
    assert!(source.contains("let transfer_budget_start = self.provider.host_transfer_stats()"));
    assert!(source.contains("let transfer_budget_end = self.provider.host_transfer_stats()"));
    assert!(source.contains("EpistemicGpuTransferBudgetTrace::from_host_transfer_stats"));
    assert!(source.contains("EpistemicGpuFinalResultTransferTrace::from_final_output"));
    assert!(source.contains("transfer_budget,"));
    assert!(source.contains("final_result_transfer,"));
    assert!(
        !source.contains("reset_host_transfer_stats"),
        "epistemic execution must snapshot the provider transfer counters, not reset shared stats"
    );

    let transfer_start_pos = source
        .find("let transfer_budget_start = self.provider.host_transfer_stats()")
        .expect("transfer-budget start snapshot");
    let candidate_generation_pos = source
        .find("let candidate_generation = self.generate_epistemic_gpu_candidates")
        .expect("candidate-generation launch");
    let final_materialization_pos = source
        .find("self.materialize_epistemic_gpu_final_results")
        .expect("final-result flag materialization launch");
    let final_tuple_pos = source
        .find("let (final_output, final_tuple_materialization)")
        .expect("final tuple materialization launch");
    let transfer_end_pos = source
        .find("let transfer_budget_end = self.provider.host_transfer_stats()")
        .expect("transfer-budget end snapshot");
    let final_transfer_pos = source
        .find("let final_result_transfer =")
        .expect("final-result transfer accounting");

    assert!(transfer_start_pos < candidate_generation_pos);
    assert!(final_materialization_pos < transfer_end_pos);
    assert!(final_materialization_pos < final_tuple_pos);
    assert!(final_tuple_pos < transfer_end_pos);
    assert!(transfer_end_pos < final_transfer_pos);
}

#[test]
fn materialization_runtime_path_launches_epistemic_kernel_not_host_writes() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("fn materialize_epistemic_gpu_candidates"));
    assert!(source.contains("EPISTEMIC_MATERIALIZE_ACCEPTED_CANDIDATES_U8"));
    assert!(source.contains("&workspace.rejection_reasons"));
    assert!(source.contains("&mut workspace.world_views"));
    assert!(cuda.contains("epistemic_materialize_accepted_candidates_u8"));
    assert!(manifest.contains("\"epistemic_materialize_accepted_candidates_u8\""));
    assert!(!source.contains("upload_epistemic_materialization"));
    assert!(!source.contains("copy_epistemic_materialization_from_host"));
}

#[test]
fn final_result_materialization_runtime_path_uses_output_device_row_count_not_host_count() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("fn materialize_epistemic_gpu_final_results"));
    assert!(source.contains("EPISTEMIC_MATERIALIZE_FINAL_RESULT_FLAGS_U8"));
    assert!(source.contains("output.num_rows_device()"));
    assert!(source.contains("&workspace.rejection_reasons"));
    assert!(source.contains("&mut workspace.world_views"));
    assert!(cuda.contains("epistemic_materialize_final_result_flags_u8"));
    assert!(manifest.contains("\"epistemic_materialize_final_result_flags_u8\""));
    assert!(!source.contains("upload_epistemic_final_results"));
    assert!(!source.contains("copy_epistemic_final_results_from_host"));
    assert!(!source.contains("cached_row_count().expect(\"epistemic"));
    assert!(!source.contains("dtoh_epistemic_final_result_row_count"));
}

#[test]
fn final_tuple_materialization_runtime_path_copies_output_columns_on_device() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");
    let manifest = include_str!("../../xlog-cuda/src/kernel_manifest_data.rs");

    assert!(source.contains("fn materialize_epistemic_gpu_final_tuples"));
    assert!(source.contains("EPISTEMIC_MATERIALIZE_FINAL_TUPLE_COLUMN_U8"));
    assert!(source.contains("EPISTEMIC_BUILD_FINAL_TUPLE_ROW_MAP_U8"));
    assert!(source.contains("let mut row_map ="));
    assert!(source.contains("epistemic final tuple row-map kernel not found"));
    assert!(source.contains("output.num_rows_device()"));
    assert!(source.contains("output.column(col_idx)"));
    assert!(source.contains("CudaBuffer::from_columns"));
    assert!(source
        .contains("pub final_tuple_materialization: EpistemicGpuFinalTupleMaterializationTrace"));
    assert!(source.contains("pub final_output: CudaBuffer"));
    assert!(cuda.contains("epistemic_build_final_tuple_row_map_u8"));
    assert!(cuda.contains("epistemic_materialize_final_tuple_column_u8"));
    assert!(manifest.contains("\"epistemic_build_final_tuple_row_map_u8\""));
    assert!(manifest.contains("\"epistemic_materialize_final_tuple_column_u8\""));
    assert!(!source.contains("upload_epistemic_final_tuple"));
    assert!(!source.contains("copy_epistemic_final_tuple_from_host"));
    assert!(!source.contains("dtoh_epistemic_final_tuple"));
}

#[test]
fn final_tuple_materialization_runtime_path_is_gated_by_gpu_model_membership() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");
    let cuda = include_str!("../../xlog-cuda/kernels/epistemic.cu");

    assert!(source.contains("literal_count: usize"));
    assert!(source.contains("reduction_count: usize"));
    assert!(source.contains("models_per_reduction: usize"));
    assert!(source.contains("&workspace.model_membership"));
    assert!(source.contains("&workspace.world_views"));
    assert!(source.contains("trace.model_membership_bytes_checked"));
    assert!(source.contains("trace.world_view_slots_checked"));
    assert!(source.contains("workspace.layout.model_membership_bytes"));
    assert!(source.contains("literal_count,"));
    assert!(source.contains("executable.gpu_plan.reductions.len(),"));
    assert!(source.contains("capacities.max_models_per_reduction,"));
    assert!(cuda.contains("epistemic_final_tuple_has_accepted_membership"));
    assert!(cuda.contains("const uint8_t* __restrict__ model_membership"));
    assert!(cuda.contains("const uint8_t* __restrict__ world_views"));
    assert!(cuda.contains("reduction_count * models_per_reduction * literal_count"));
    assert!(!cuda.contains("accepted_candidate |= (rejection_reasons[candidate] == 0u) ? 1u : 0u"));
}

#[test]
fn execution_result_records_materialization_kernels_after_world_view_validation() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("pub materialization: EpistemicGpuMaterializationTrace"));
    assert!(source
        .contains("pub final_result_materialization: EpistemicGpuFinalResultMaterializationTrace"));
    assert!(source
        .contains("pub final_tuple_materialization: EpistemicGpuFinalTupleMaterializationTrace"));
    assert!(source.contains("let materialization ="));
    assert!(source.contains("self.materialize_epistemic_gpu_candidates"));
    assert!(source.contains("let final_result_materialization ="));
    assert!(source.contains("self.materialize_epistemic_gpu_final_results"));
    assert!(source.contains("let (final_output, final_tuple_materialization) ="));
    assert!(source.contains("materialize_epistemic_gpu_final_tuples("));
    assert!(source.contains("materialization,"));
    assert!(source.contains("final_result_materialization,"));
    assert!(source.contains("final_tuple_materialization,"));
    assert!(source.contains("final_output,"));

    let world_validation_pos = source
        .find("let world_view_validation = self.validate_epistemic_gpu_world_views")
        .expect("world-view validation launch in execution path");
    let materialization_pos = source
        .find("self.materialize_epistemic_gpu_candidates")
        .expect("candidate materialization launch in execution path");
    let final_materialization_pos = source
        .find("self.materialize_epistemic_gpu_final_results")
        .expect("final-result materialization launch in execution path");
    let final_tuple_pos = source
        .find("let (final_output, final_tuple_materialization)")
        .expect("final tuple materialization launch in execution path");

    assert!(world_validation_pos < materialization_pos);
    assert!(materialization_pos < final_materialization_pos);
    assert!(final_materialization_pos < final_tuple_pos);
}

#[test]
fn execution_path_requires_actual_stable_model_membership_before_return() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("model_membership.require_stable_model_tuple_source()?"));

    let model_membership_pos = source
        .find("let model_membership = self.populate_epistemic_gpu_model_membership_from_tuple_sources")
        .expect("model-membership launch in execution path");
    let certification_pos = source
        .find("model_membership.require_stable_model_tuple_source()?")
        .expect("stable-model membership certification gate");
    let world_validation_pos = source
        .find("let world_view_validation = self.validate_epistemic_gpu_world_views")
        .expect("world-view validation launch in execution path");
    let final_tuple_pos = source
        .find("let (final_output, final_tuple_materialization)")
        .expect("final tuple materialization launch in execution path");
    let return_pos = source
        .find("Ok(EpistemicGpuExecutionResult")
        .expect("execution result return");

    assert!(model_membership_pos < certification_pos);
    assert!(certification_pos < world_validation_pos);
    assert!(certification_pos < final_tuple_pos);
    assert!(certification_pos < return_pos);
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
        reduced_runtime_plan: runtime_plan_with_scan_rule(),
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

fn runtime_plan_with_v070_4cycle_wcoj() -> ExecutionPlan {
    let mut plan = ExecutionPlan::new(vec![Scc {
        id: 0,
        predicates: vec!["cycle4".to_string()],
        is_recursive: false,
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
                ProjectExpr::Column(2),
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
