#![cfg(feature = "host-io")]

use xlog_prob::exact::{ExactDdnnfProgram, ExactResult};
use xlog_prob::provenance::Value;

static EXACT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn prob_of(result: &ExactResult, predicate: &str, args: &[Value]) -> f64 {
    result
        .query_probs
        .iter()
        .find(|q| q.atom.predicate == predicate && q.atom.args == args)
        .unwrap_or_else(|| {
            panic!(
                "missing query result for {} with args {:?}",
                predicate, args
            )
        })
        .prob
}

fn prob0(result: &ExactResult, predicate: &str) -> f64 {
    prob_of(result, predicate, &[])
}

#[test]
fn test_exact_ddnnf_wet_conditioning() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
query(rain()).
query(sprinkler()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    let p_wet = 1.0 - (1.0 - 0.7) * (1.0 - 0.2);
    let expected_rain = 0.7 / p_wet;
    let expected_sprinkler = 0.2 / p_wet;

    let got_rain = prob0(&result, "rain");
    let got_sprinkler = prob0(&result, "sprinkler");

    assert!(
        (got_rain - expected_rain).abs() < 1e-9,
        "got_rain={}",
        got_rain
    );
    assert!(
        (got_sprinkler - expected_sprinkler).abs() < 1e-9,
        "got_sprinkler={}",
        got_sprinkler
    );
}

#[test]
fn test_exact_ddnnf_supports_false_evidence_on_derived_atom() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), false).
query(rain()).
query(sprinkler()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    assert_eq!(prob0(&result, "rain"), 0.0);
    assert_eq!(prob0(&result, "sprinkler"), 0.0);
}

#[test]
fn test_exact_ddnnf_annotated_disjunction_probabilities() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.6::heads(); 0.4::tails().
query(heads()).
query(tails()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    assert!((prob0(&result, "heads") - 0.6).abs() < 1e-9);
    assert!((prob0(&result, "tails") - 0.4).abs() < 1e-9);
}

#[test]
fn test_exact_ddnnf_probabilistic_fact_marginal_probability() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.7::rain().
query(rain()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();
    assert!((prob0(&result, "rain") - 0.7).abs() < 1e-9);
}

#[test]
fn test_exact_ddnnf_rejects_zero_probability_evidence() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.0::rain().
evidence(rain(), true).
query(rain()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let err = compiled.evaluate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("evidence"), "msg={}", msg);
    assert!(
        msg.contains("P(E)=0") || msg.contains("zero"),
        "msg={}",
        msg
    );
}

#[test]
fn test_exact_ddnnf_recursive_reachability_probability() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.5::edge(1,2).
0.5::edge(2,3).
reach(X,Y) :- edge(X,Y).
reach(X,Z) :- reach(X,Y), edge(Y,Z).
query(reach(1,2)).
query(reach(1,3)).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    let p12 = prob_of(&result, "reach", &[Value::from(1_i64), Value::from(2_i64)]);
    let p13 = prob_of(&result, "reach", &[Value::from(1_i64), Value::from(3_i64)]);
    assert!((p12 - 0.5).abs() < 1e-9, "p12={}", p12);
    assert!((p13 - 0.25).abs() < 1e-9, "p13={}", p13);
}

#[test]
fn test_exact_ddnnf_non_monotone_wfs_simple_cycle() {
    // Test a simple non-monotone program: p :- not q. q :- not p.
    // Under WFS, both p and q are undefined (probability 0)
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
p() :- not q().
q() :- not p().
query(p()).
query(q()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    // Both p and q are in a cycle through negation, so both are undefined
    // Undefined atoms have probability 0
    let p_prob = prob0(&result, "p");
    let q_prob = prob0(&result, "q");
    assert!(
        p_prob < 1e-9,
        "P(p) should be 0 (undefined), got {}",
        p_prob
    );
    assert!(
        q_prob < 1e-9,
        "P(q) should be 0 (undefined), got {}",
        q_prob
    );
}

#[test]
fn test_exact_ddnnf_non_monotone_wfs_with_probabilistic_facts() {
    // Test a non-monotone program with probabilistic facts
    // base::0.5. p() :- base(), not q(). q() :- base(), not p().
    // Under WFS, when base() is true, both p and q are undefined
    // When base() is false, neither can be derived
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.5::base().
p() :- base(), not q().
q() :- base(), not p().
query(p()).
query(q()).
query(base()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    // base() has probability 0.5 as expected
    let base_prob = prob0(&result, "base");
    assert!(
        (base_prob - 0.5).abs() < 1e-9,
        "P(base) should be 0.5, got {}",
        base_prob
    );

    // Both p and q are in a cycle through negation, so both are undefined
    // Undefined atoms have probability 0
    let p_prob = prob0(&result, "p");
    let q_prob = prob0(&result, "q");
    assert!(
        p_prob < 1e-9,
        "P(p) should be 0 (undefined), got {}",
        p_prob
    );
    assert!(
        q_prob < 1e-9,
        "P(q) should be 0 (undefined), got {}",
        q_prob
    );
}

#[test]
fn test_exact_ddnnf_non_monotone_wfs_asymmetric() {
    // Test: p :- not q. q.
    // q is a fact, so it's true. Therefore p is false (not q is false).
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
p() :- not q().
q().
query(p()).
query(q()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    let p_prob = prob0(&result, "p");
    let q_prob = prob0(&result, "q");

    // q is a fact, so P(q) = 1
    assert!(
        (q_prob - 1.0).abs() < 1e-9,
        "P(q) should be 1.0, got {}",
        q_prob
    );
    // p depends on not q, and q is true, so p is false
    assert!(p_prob < 1e-9, "P(p) should be 0, got {}", p_prob);
}

#[test]
fn test_exact_ddnnf_non_monotone_wfs_chain() {
    // Test: a. b :- not a. c :- not b.
    // a is a fact (true)
    // b :- not a fails (a is true, so not a is false) -> b is false
    // c :- not b succeeds (b is false, so not b is true) -> c is true
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
a().
b() :- not a().
c() :- not b().
query(a()).
query(b()).
query(c()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    let a_prob = prob0(&result, "a");
    let b_prob = prob0(&result, "b");
    let c_prob = prob0(&result, "c");

    assert!(
        (a_prob - 1.0).abs() < 1e-9,
        "P(a) should be 1.0, got {}",
        a_prob
    );
    assert!(b_prob < 1e-9, "P(b) should be 0, got {}", b_prob);
    assert!(
        (c_prob - 1.0).abs() < 1e-9,
        "P(c) should be 1.0, got {}",
        c_prob
    );
}
