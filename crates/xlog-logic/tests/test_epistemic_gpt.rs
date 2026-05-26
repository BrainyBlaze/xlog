use xlog_core::XlogError;
use xlog_logic::ast::Term;
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
fn gpt_trace_counts_epistemic_guesses_inside_each_candidate() {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        accepted() :- know p(), possible q().
        "#,
    )
    .unwrap();
    let candidates = vec![EpistemicInterpretation::new()
        .with_known("p", 0)
        .with_possible("q", 0)];

    let outcome = run_generate_propagate_test(
        &program,
        candidates,
        GeneratePropagateTestConfig { max_candidates: 1 },
    )
    .unwrap();

    assert_eq!(outcome.trace.generated, 1);
    assert_eq!(outcome.trace.guesses, 2);
    assert_eq!(outcome.trace.propagated, 1);
    assert_eq!(outcome.trace.reduced_program_models, 1);
    assert_eq!(outcome.trace.accepted, 1);
    assert_eq!(outcome.trace.accepted_world_views, 1);
    assert_eq!(outcome.accepted_candidate_indices, vec![0]);
}

#[test]
fn gpt_accepts_not_known_only_when_candidate_does_not_know_atom() {
    let program = parse_program("accepted() :- not know blocked().").unwrap();
    let candidates = vec![
        EpistemicInterpretation::new(),
        EpistemicInterpretation::new().with_known("blocked", 0),
    ];

    let outcome = run_generate_propagate_test(
        &program,
        candidates,
        GeneratePropagateTestConfig { max_candidates: 2 },
    )
    .unwrap();

    assert_eq!(outcome.trace.generated, 2);
    assert_eq!(outcome.trace.guesses, 1);
    assert_eq!(outcome.trace.propagated, 2);
    assert_eq!(outcome.trace.tested, 2);
    assert_eq!(outcome.trace.accepted, 1);
    assert_eq!(outcome.trace.rejected, 1);
    assert_eq!(outcome.trace.accepted_world_views, 1);
    assert_eq!(outcome.accepted_candidate_indices, vec![0]);
    assert_eq!(outcome.rejected_candidate_indices, vec![1]);
    assert_eq!(
        outcome.trace.rejection_reasons,
        vec![FaeelNoModelReason::UnsatisfiedLiteral {
            predicate: "blocked".to_string(),
            arity: 0,
        }]
    );
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
fn gpt_rejects_candidate_when_ground_tuple_guess_mismatches_literal() {
    let program = parse_program("accepted() :- know edge(2).").unwrap();
    let candidates = vec![EpistemicInterpretation::new()
        .with_known_terms("edge", vec![Term::Integer(1)])
        .unwrap()];

    let outcome = run_generate_propagate_test(
        &program,
        candidates,
        GeneratePropagateTestConfig { max_candidates: 1 },
    )
    .unwrap();

    assert_eq!(outcome.trace.generated, 1);
    assert_eq!(outcome.trace.propagated, 1);
    assert_eq!(outcome.trace.tested, 1);
    assert_eq!(outcome.trace.accepted, 0);
    assert_eq!(outcome.trace.rejected, 1);
    assert_eq!(
        outcome.trace.rejection_reasons,
        vec![FaeelNoModelReason::UnsatisfiedLiteral {
            predicate: "edge".to_string(),
            arity: 1,
        }]
    );
    assert_eq!(outcome.accepted_candidate_indices, Vec::<usize>::new());
    assert_eq!(outcome.rejected_candidate_indices, vec![0]);
}

#[test]
fn gpt_traces_mixed_nonzero_tuple_guess_outcomes() {
    let program = parse_program(
        r#"
        accepted() :- know edge(1, 2), possible color(2), not possible blocked(9).
        "#,
    )
    .unwrap();
    let candidates = vec![
        EpistemicInterpretation::new()
            .with_known_terms("edge", vec![Term::Integer(1), Term::Integer(2)])
            .unwrap()
            .with_known_terms("color", vec![Term::Integer(2)])
            .unwrap(),
        EpistemicInterpretation::new()
            .with_known_terms("edge", vec![Term::Integer(1), Term::Integer(3)])
            .unwrap()
            .with_known_terms("color", vec![Term::Integer(2)])
            .unwrap(),
        EpistemicInterpretation::new()
            .with_known_terms("edge", vec![Term::Integer(1), Term::Integer(2)])
            .unwrap()
            .with_possible_terms("color", vec![Term::Integer(2)])
            .unwrap(),
        EpistemicInterpretation::new()
            .with_known_terms("edge", vec![Term::Integer(1), Term::Integer(2)])
            .unwrap()
            .with_known_terms("color", vec![Term::Integer(2)])
            .unwrap()
            .with_known_terms("blocked", vec![Term::Integer(9)])
            .unwrap(),
    ];

    let outcome = run_generate_propagate_test(
        &program,
        candidates,
        GeneratePropagateTestConfig { max_candidates: 4 },
    )
    .unwrap();

    assert_eq!(outcome.trace.generated, 4);
    assert_eq!(outcome.trace.guesses, 9);
    assert_eq!(outcome.trace.propagated, 4);
    assert_eq!(outcome.trace.pruned, 0);
    assert_eq!(outcome.trace.tested, 4);
    assert_eq!(outcome.trace.reduced_program_models, 4);
    assert_eq!(outcome.trace.accepted, 1);
    assert_eq!(outcome.trace.accepted_world_views, 1);
    assert_eq!(outcome.trace.rejected, 3);
    assert_eq!(outcome.accepted_candidate_indices, vec![0]);
    assert_eq!(outcome.rejected_candidate_indices, vec![1, 2, 3]);
    assert_eq!(
        outcome.trace.rejection_reasons,
        vec![
            FaeelNoModelReason::UnsatisfiedLiteral {
                predicate: "edge".to_string(),
                arity: 2,
            },
            FaeelNoModelReason::UnfoundedPossible {
                predicate: "color".to_string(),
                arity: 1,
            },
            FaeelNoModelReason::UnsatisfiedLiteral {
                predicate: "blocked".to_string(),
                arity: 1,
            },
        ]
    );
}

#[test]
fn gpt_rejects_epistemic_integrity_constraints_before_candidate_testing() {
    let program = parse_program(
        r#"
        accepted() :- know fact().
        :- possible blocked().
        "#,
    )
    .unwrap();

    let err = run_generate_propagate_test(
        &program,
        vec![EpistemicInterpretation::new()
            .with_known("fact", 0)
            .with_possible("blocked", 0)],
        GeneratePropagateTestConfig { max_candidates: 1 },
    )
    .unwrap_err();

    match err {
        XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "epistemic GPT constraint");
            assert!(context.contains("constraint[0]"));
            assert!(context.contains("possible blocked/0"));
        }
        other => panic!("expected typed GPT constraint boundary, got {other:?}"),
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

#[test]
fn gpt_default_entrypoint_honors_source_g91_mode_annotation() {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        p() :- possible p().
        "#,
    )
    .unwrap();

    let outcome = run_generate_propagate_test(
        &program,
        vec![
            EpistemicInterpretation::new(),
            EpistemicInterpretation::new().with_possible("p", 0),
        ],
        GeneratePropagateTestConfig { max_candidates: 2 },
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
