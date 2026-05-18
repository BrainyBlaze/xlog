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
    EpistemicGpuMaterializationTrace, EpistemicGpuPropagationTrace, EpistemicGpuRuntimeCounters,
    EpistemicGpuRuntimePreflight, EpistemicGpuRuntimeTrace, EpistemicGpuRuntimeWcojCertification,
    EpistemicGpuWorkspaceCapacities, EpistemicGpuWorkspaceLayout, EpistemicGpuWorkspaceResetTrace,
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
    assert_eq!(layout.model_membership_bytes, 12);
    assert_eq!(layout.rejection_reason_slots, 8);
    assert_eq!(layout.total_bytes(), 92);
}

#[test]
fn workspace_layout_rejects_zero_candidate_capacity() {
    let plan = EpistemicGpuPlan::new(
        EirEpistemicMode::G91,
        vec![epistemic_literal("fact", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
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

    assert_eq!(preflight.workspace_layout.total_bytes(), 78);
    assert_eq!(preflight.reduced_runtime_rule_count, 1);
    assert_eq!(preflight.multiway_reduction_count, 1);
    assert_eq!(preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(preflight.planned_hash_route_count, 0);
    assert_eq!(preflight.sorted_layout_requirement_count, 2);
    assert_eq!(preflight.helper_split_spec_count, 1);
    assert!(preflight.cpu_fallbacks.is_zero());
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
            required_kclique_plans: 1,
            observed_wcoj_dispatches: 0,
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
        wcoj_layout_sort_invocation_count: 2,
        ..EpistemicGpuRuntimeCounters::default()
    };
    let delta = after.saturating_delta_since(before);

    assert_eq!(
        EpistemicGpuRuntimeWcojCertification::for_preflight_and_delta(&preflight, &delta),
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1,
            observed_kclique_dispatches: 1,
            observed_layout_sorts: 2,
            observed_metadata_builds: 1,
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
        wcoj_layout_sort_invocation_count: 8,
        ..EpistemicGpuRuntimeCounters::default()
    };
    let after = EpistemicGpuRuntimeCounters {
        wcoj_clique5_dispatch_count: 3,
        kclique_metadata_build_count: 5,
        wcoj_layout_sort_invocation_count: 10,
        ..EpistemicGpuRuntimeCounters::default()
    };

    let trace = EpistemicGpuRuntimeTrace::from_preflight_and_counters(preflight, before, after);

    assert_eq!(trace.preflight.kclique_wcoj_plan_count, 1);
    assert_eq!(trace.counters_before.wcoj_clique5_dispatch_count, 2);
    assert_eq!(trace.counters_after.wcoj_clique5_dispatch_count, 3);
    assert_eq!(trace.counter_delta.wcoj_clique5_dispatch_count, 1);
    assert_eq!(trace.counter_delta.kclique_metadata_build_count, 1);
    assert_eq!(trace.counter_delta.wcoj_layout_sort_invocation_count, 2);
    assert_eq!(
        trace.wcoj_certification,
        EpistemicGpuRuntimeWcojCertification::Certified {
            observed_wcoj_dispatches: 1,
            observed_kclique_dispatches: 1,
            observed_layout_sorts: 2,
            observed_metadata_builds: 1,
        }
    );
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
    assert_eq!(trace.model_membership_bytes, 12);
    assert_eq!(trace.rejection_reason_bytes, 32);
    assert_eq!(trace.total_zeroed_bytes(), 92);
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
        .find("let output = self.execute_plan(&executable.reduced_runtime_plan)?")
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
        .find("let output = self.execute_plan(&executable.reduced_runtime_plan)?")
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
fn execution_result_records_materialization_kernel_trace_before_reduced_dispatch() {
    let source = include_str!("../src/executor/epistemic_workspace.rs");

    assert!(source.contains("pub materialization: EpistemicGpuMaterializationTrace"));
    assert!(source.contains("let materialization ="));
    assert!(source.contains("self.materialize_epistemic_gpu_candidates"));
    assert!(source.contains("materialization,"));

    let validation_pos = source
        .find("let candidate_validation = self.validate_epistemic_gpu_candidates")
        .expect("candidate-validation launch in execution path");
    let materialization_pos = source
        .find("self.materialize_epistemic_gpu_candidates")
        .expect("materialization launch in execution path");
    let reduced_dispatch_pos = source
        .find("let output = self.execute_plan(&executable.reduced_runtime_plan)?")
        .expect("reduced production runtime dispatch");

    assert!(validation_pos < materialization_pos);
    assert!(materialization_pos < reduced_dispatch_pos);
}

fn epistemic_literal(predicate: &str, op: EirEpistemicOp) -> EirEpistemicLiteral {
    EirEpistemicLiteral {
        op,
        negated: false,
        atom: EirAtom {
            predicate: predicate.to_string(),
            arity: 0,
        },
    }
}

fn executable_with_kclique_wcoj_plan() -> EpistemicExecutablePlan {
    let gpu_plan = EpistemicGpuPlan::new(
        EirEpistemicMode::Faeel,
        vec![epistemic_literal("gate", EirEpistemicOp::Know)],
        vec![EpistemicReductionPlan {
            rule_index: 0,
            relational_body_atoms: 10,
            wcoj_status: EpistemicWcojReductionStatus::RequiresPlannerEligibility,
        }],
    );

    EpistemicExecutablePlan {
        gpu_plan,
        reduced_runtime_plan: runtime_plan_with_kclique_wcoj(),
    }
}

fn runtime_plan_with_kclique_wcoj() -> ExecutionPlan {
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
