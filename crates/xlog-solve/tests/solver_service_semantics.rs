use xlog_solve::{
    Clause, Literal, SolveInstance, SolverPortfolioStatus, SolverService, SolverServiceBudget,
    SolverServiceStatus,
};

#[test]
fn incremental_assumptions_can_be_added_and_retracted() {
    let mut service = SolverService::new(SolveInstance::new(
        1,
        vec![Clause::new(vec![Literal::positive(0)])],
    ));

    let positive = service.assume(Literal::positive(0));
    assert_eq!(service.solve().status, SolverServiceStatus::Sat);

    let negative = service.assume(Literal::negative(0));
    assert_eq!(service.solve().status, SolverServiceStatus::Unsat);

    assert!(service.retract_assumption(negative));
    assert_eq!(service.solve().status, SolverServiceStatus::Sat);
    assert!(service.retract_assumption(positive));
}

#[test]
fn learned_clause_transfer_is_observable() {
    let mut source = SolverService::new(SolveInstance::new(
        1,
        vec![Clause::new(vec![Literal::positive(0)])],
    ));
    source.assume(Literal::negative(0));
    assert_eq!(source.solve().status, SolverServiceStatus::Unsat);

    let mut target = SolverService::new(SolveInstance::new(1, vec![]));
    let transferred = source.transfer_learned_clauses_to(&mut target);

    assert_eq!(transferred.clauses, 1);
    assert_eq!(target.trace().learned_clause_transfers, 1);
    assert_eq!(target.solve().status, SolverServiceStatus::Sat);
}

#[test]
fn maxsat_soft_constraints_return_expected_optimum() {
    let service = SolverService::new(SolveInstance::with_weights(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
        vec![5.0, 2.0],
    ));

    assert_eq!(service.solve().status, SolverServiceStatus::Optimal(5));
}

#[test]
fn cpu_fixture_reports_gpu_portfolio_as_unimplemented_blocker() {
    let service = SolverService::new(SolveInstance::new(0, vec![]));

    assert_eq!(
        service.gpu_portfolio_status(),
        SolverPortfolioStatus::Deferred {
            reason: "GPU portfolio solving is not implemented in the semantic-oracle facade and blocks solver-service integration closure",
        }
    );
}

#[test]
fn failure_modes_distinguish_unsat_unknown_and_timeout() {
    let unsat = SolverService::new(SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    ));
    assert_eq!(unsat.solve().status, SolverServiceStatus::Unsat);

    let unknown = SolverService::new(SolveInstance::new(1, vec![]));
    assert_eq!(
        unknown
            .solve_with_budget(SolverServiceBudget::NoSearch)
            .status,
        SolverServiceStatus::Unknown
    );

    let timeout = SolverService::new(SolveInstance::new(1, vec![]));
    assert_eq!(
        timeout
            .solve_with_budget(SolverServiceBudget::AssignmentLimit(0))
            .status,
        SolverServiceStatus::Timeout
    );
}
