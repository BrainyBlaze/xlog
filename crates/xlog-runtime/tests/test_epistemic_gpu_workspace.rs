use xlog_ir::{
    EirAtom, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EpistemicGpuPlan,
    EpistemicReductionPlan, EpistemicWcojReductionStatus,
};
use xlog_runtime::{EpistemicGpuWorkspaceCapacities, EpistemicGpuWorkspaceLayout};

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
