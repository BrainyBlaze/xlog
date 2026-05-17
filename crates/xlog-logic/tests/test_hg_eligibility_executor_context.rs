use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    analyze, is_eligible, Boundary, ExecutorContext, HypergraphRule, BINARY_FALLBACK_KEY_LIMIT,
};

fn var(idx: usize) -> Term {
    Term::Variable(format!("V{idx}"))
}

fn atom(predicate: &str, terms: Vec<Term>) -> Atom {
    Atom {
        predicate: predicate.to_string(),
        terms,
    }
}

fn k_clique_rule(k: usize) -> Rule {
    let mut body = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            body.push(BodyLiteral::Positive(atom(
                &format!("e_{i}_{j}"),
                vec![var(i), var(j)],
            )));
        }
    }
    Rule {
        head: atom("out", (0..k).map(var).collect()),
        body,
    }
}

#[test]
fn executor_context_matrix_covers_hash_wcoj_and_over_wcoj_limits() {
    use ExecutorContext::{HashFallback, WcojEligible};

    let cases = [
        (5, HashFallback, false),
        (6, HashFallback, false),
        (7, HashFallback, false),
        (8, HashFallback, false),
        (5, WcojEligible, true),
        (6, WcojEligible, true),
        (7, WcojEligible, true),
        (8, WcojEligible, true),
        (9, HashFallback, false),
        (9, WcojEligible, false),
        (10, HashFallback, false),
        (10, WcojEligible, false),
    ];
    assert_eq!(cases.len(), 12);

    for (k, context, expected) in cases {
        let hg = HypergraphRule::from_rule(&k_clique_rule(k));
        assert_eq!(
            is_eligible(&hg, context),
            expected,
            "K{k} in {context:?} context"
        );
    }
}

#[test]
fn binary_fallback_key_limit_is_retained_and_context_gated() {
    use ExecutorContext::{HashFallback, WcojEligible};

    assert_eq!(BINARY_FALLBACK_KEY_LIMIT, 4);

    let k5 = HypergraphRule::from_rule(&k_clique_rule(5));
    let hash = analyze(&k5, HashFallback);
    assert!(hash
        .boundaries()
        .contains(&Boundary::JoinKeysExceedBinaryFallbackLimit {
            context: HashFallback,
            count: 5,
            limit: BINARY_FALLBACK_KEY_LIMIT,
        }));
    assert_eq!(
        analyze(&k5, WcojEligible),
        xlog_logic::hypergraph::Eligibility::Eligible
    );
}
