use xlog_core::XlogError;
use xlog_logic::epistemic::{
    run_generate_propagate_test, EpistemicInterpretation, GeneratePropagateTestConfig,
};
use xlog_logic::parse_program;

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
    assert_eq!(outcome.accepted_candidate_indices, vec![0]);
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
