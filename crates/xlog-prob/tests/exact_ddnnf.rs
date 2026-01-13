use xlog_prob::exact::{ExactDdnnfProgram, ExactResult};
use xlog_prob::provenance::Value;

fn prob_of(result: &ExactResult, predicate: &str, args: &[Value]) -> f64 {
    result
        .query_probs
        .iter()
        .find(|q| q.atom.predicate == predicate && q.atom.args == args)
        .unwrap_or_else(|| panic!("missing query result for {} with args {:?}", predicate, args))
        .prob
}

fn prob0(result: &ExactResult, predicate: &str) -> f64 {
    prob_of(result, predicate, &[])
}

#[test]
fn test_exact_ddnnf_wet_conditioning() {
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

    assert!((got_rain - expected_rain).abs() < 1e-9, "got_rain={}", got_rain);
    assert!(
        (got_sprinkler - expected_sprinkler).abs() < 1e-9,
        "got_sprinkler={}",
        got_sprinkler
    );
}

#[test]
fn test_exact_ddnnf_supports_false_evidence_on_derived_atom() {
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
fn test_exact_ddnnf_rejects_zero_probability_evidence() {
    let source = r#"
0.0::rain().
evidence(rain(), true).
query(rain()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let err = compiled.evaluate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("evidence"), "msg={}", msg);
    assert!(msg.contains("P(E)=0") || msg.contains("zero"), "msg={}", msg);
}

#[test]
fn test_exact_ddnnf_recursive_reachability_probability() {
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
