use xlog_logic::epistemic::{
    evaluate_faeel_candidate, EpistemicInterpretation, FaeelCandidateResult, FaeelNoModelReason,
};
use xlog_logic::parse_program;

#[test]
fn faeel_accepts_founded_known_candidate() {
    let program = parse_program("accepted() :- know fact().").unwrap();
    let interpretation = EpistemicInterpretation::new().with_known("fact", 0);

    assert_eq!(
        evaluate_faeel_candidate(&program, &interpretation).unwrap(),
        FaeelCandidateResult::Model
    );
}

#[test]
fn faeel_rejects_unfounded_possible_candidate_without_panic() {
    let program = parse_program("accepted() :- possible fact().").unwrap();
    let interpretation = EpistemicInterpretation::new().with_possible("fact", 0);

    assert_eq!(
        evaluate_faeel_candidate(&program, &interpretation).unwrap(),
        FaeelCandidateResult::NoModel(FaeelNoModelReason::UnfoundedPossible {
            predicate: "fact".to_string(),
            arity: 0,
        })
    );
}

#[test]
fn faeel_reports_contradiction_as_no_model() {
    let program = parse_program("accepted() :- know fact().").unwrap();
    let interpretation = EpistemicInterpretation::new()
        .with_known("fact", 0)
        .with_rejected("fact", 0);

    assert_eq!(
        evaluate_faeel_candidate(&program, &interpretation).unwrap(),
        FaeelCandidateResult::NoModel(FaeelNoModelReason::Contradiction {
            predicate: "fact".to_string(),
            arity: 0,
        })
    );
}

#[test]
fn faeel_rejects_global_candidate_contradiction_before_literal_walk() {
    let program = parse_program("accepted() :- know fact().").unwrap();
    let interpretation = EpistemicInterpretation::new()
        .with_known("fact", 0)
        .with_known("noise", 0)
        .with_rejected("noise", 0);

    assert_eq!(
        evaluate_faeel_candidate(&program, &interpretation).unwrap(),
        FaeelCandidateResult::NoModel(FaeelNoModelReason::Contradiction {
            predicate: "noise".to_string(),
            arity: 0,
        })
    );
}
