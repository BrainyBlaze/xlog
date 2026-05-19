use xlog_core::XlogError;
use xlog_logic::epistemic::{
    run_generate_propagate_test, run_generate_propagate_test_with_mode, EpistemicInterpretation,
    FaeelNoModelReason, GeneratePropagateTestConfig,
};
use xlog_logic::{parse_program, EpistemicMode};

#[test]
fn gpt_reports_phase_counts_and_candidate_outcomes() {
    let program = parse_program("accepted() :- know fact().").unwrap();
    let candidates = vec![
        EpistemicInterpretation::new().with_known("fact", 0),
        EpistemicInterpretation::new(),
        EpistemicInterpretation::new()
            .with_known("fact", 0)
            .with_rejected("fact", 0),
    ];

    let outcome = run_generate_propagate_test(
        &program,
        candidates,
        GeneratePropagateTestConfig { max_candidates: 8 },
    )
    .unwrap();

    assert_eq!(outcome.trace.generated, 3);
    assert_eq!(outcome.trace.propagated, 2);
    assert_eq!(outcome.trace.pruned, 1);
    assert_eq!(outcome.trace.tested, 2);
    assert_eq!(outcome.trace.accepted, 1);
    assert_eq!(outcome.trace.rejected, 1);
    assert_eq!(outcome.trace.guesses, 3);
    assert_eq!(outcome.trace.reduced_program_models, 2);
    assert_eq!(outcome.trace.accepted_world_views, 1);
    assert_eq!(
        outcome.trace.rejection_reasons,
        vec![
            FaeelNoModelReason::Contradiction {
                predicate: "fact".to_string(),
                arity: 0,
            },
            FaeelNoModelReason::UnsatisfiedLiteral {
                predicate: "fact".to_string(),
                arity: 0,
            }
        ]
    );
    assert_eq!(outcome.accepted_candidate_indices, vec![0]);
    assert_eq!(outcome.rejected_candidate_indices, vec![2, 1]);
}

#[test]
fn gpt_candidate_guard_is_typed_and_bounded() {
    let program = parse_program("accepted() :- know fact().").unwrap();
    let candidates = vec![
        EpistemicInterpretation::new(),
        EpistemicInterpretation::new(),
        EpistemicInterpretation::new(),
    ];

    let err = run_generate_propagate_test(
        &program,
        candidates,
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .unwrap_err();

    match err {
        XlogError::ResourceExhausted {
            context,
            estimated_bytes,
            budget_bytes,
        } => {
            assert_eq!(context, "epistemic GPT candidate guard");
            assert_eq!(estimated_bytes, 3);
            assert_eq!(budget_bytes, 2);
        }
        other => panic!("expected typed GPT guard error, got {other:?}"),
    }
}

#[test]
fn gpt_can_run_g91_compatibility_mode() {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        p() :- possible p().
        "#,
    )
    .unwrap();

    let outcome = run_generate_propagate_test_with_mode(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_possible("p", 0),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
        EpistemicMode::G91,
    )
    .unwrap();

    assert_eq!(outcome.trace.generated, 2);
    assert_eq!(outcome.trace.propagated, 2);
    assert_eq!(outcome.trace.tested, 2);
    assert_eq!(outcome.trace.accepted, 1);
    assert_eq!(outcome.trace.accepted_world_views, 1);
    assert_eq!(outcome.trace.rejected, 1);
    assert_eq!(outcome.accepted_candidate_indices, vec![1]);
    assert_eq!(outcome.rejected_candidate_indices, vec![0]);
}
