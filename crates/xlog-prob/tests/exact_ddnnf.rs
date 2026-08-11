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
fn test_exact_ddnnf_canonicalizes_declared_ground_value_types() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
pred deterministic_float(f32).
pred deterministic_float_out(f32).
pred float_input(f32).
pred float_out(f32).
pred float_choice(f32).
pred float_choice_out(f32).
pred bool_input(bool).
pred bool_out(bool).
pred symbol_input(symbol).
pred symbol_out(symbol).
pred narrow_input(i32).
pred narrow_out(i32).

deterministic_float(0.1).
0.5::float_input(0.1).
0.5::float_choice(0.1); 0.5::float_choice(0.2).
0.5::bool_input(true).
0.5::symbol_input("alpha").
0.5::narrow_input(2147483647).

deterministic_float_out(Y) :- deterministic_float(X), Y is X + cast(0.0, f32).
float_out(Y) :- float_input(X), Y is X + cast(0.0, f32).
float_choice_out(Y) :- float_choice(X), Y is X + cast(0.0, f32).
bool_out(Y) :- bool_input(X), Y is cast(X, bool).
symbol_out(Y) :- symbol_input(X), Y is cast(X, symbol).
narrow_out(Y) :- narrow_input(X), Y is X + cast(0, i32).

evidence(float_out(0.1), true).
query(deterministic_float_out(0.1)).
query(float_input(0.1)).
query(float_choice_out(0.1)).
query(bool_out(true)).
query(symbol_out("alpha")).
query(narrow_out(2147483647)).
"#;

    let result = ExactDdnnfProgram::compile_source(source)
        .expect("compile schema-canonical exact program")
        .evaluate()
        .expect("evaluate schema-canonical exact program");
    let rounded_float = Value::F64(f64::from(0.1_f32).to_bits());

    assert_eq!(
        prob_of(
            &result,
            "deterministic_float_out",
            std::slice::from_ref(&rounded_float),
        ),
        1.0
    );
    assert_eq!(
        prob_of(&result, "float_input", std::slice::from_ref(&rounded_float),),
        1.0
    );
    assert!(
        (prob_of(
            &result,
            "float_choice_out",
            std::slice::from_ref(&rounded_float),
        ) - 0.5)
            .abs()
            < 1e-9
    );
    assert!((prob_of(&result, "bool_out", &[Value::I64(1)]) - 0.5).abs() < 1e-9);
    assert!(
        (prob_of(&result, "symbol_out", &[Value::String("alpha".to_string())],) - 0.5).abs() < 1e-9
    );
    assert!(
        (prob_of(&result, "narrow_out", &[Value::I64(i64::from(i32::MAX))],) - 0.5).abs() < 1e-9
    );
}

#[test]
fn test_exact_ddnnf_rejects_schema_incompatible_ground_values() {
    for (source, expected) in [
        (
            "pred small(i32).\n0.5::small(2147483648).\nquery(small(0)).\n",
            "out of range for I32",
        ),
        (
            "pred flag(bool).\n0.5::flag(2).\nquery(flag(0)).\n",
            "not valid for Bool",
        ),
    ] {
        let error = match ExactDdnnfProgram::compile_source(source) {
            Ok(_) => panic!("schema-incompatible probabilistic atom must be rejected"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?}, got {error}"
        );
    }
}

#[test]
fn test_exact_ddnnf_preserves_closed_world_query_and_evidence_only_predicates() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let source = "0.5::rain().\n\
                  evidence(frost(), false).\n\
                  evidence(snow(0.1), false).\n\
                  query(rain()).\n\
                  query(frost()).\n\
                  query(snow(0.1)).\n";

    let result = ExactDdnnfProgram::compile_source(source)
        .expect("compile closed-world exact program")
        .evaluate()
        .expect("evaluate closed-world exact program");

    assert!((prob0(&result, "rain") - 0.5).abs() < 1e-9);
    assert_eq!(prob0(&result, "frost"), 0.0);
    assert_eq!(
        prob_of(&result, "snow", &[Value::F64(0.1_f64.to_bits())]),
        0.0
    );
}

#[test]
fn test_exact_ddnnf_preserves_quoted_string_query_values() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let result =
        ExactDdnnfProgram::compile_source("0.5::gate(\"alpha\").\nquery(gate(\"alpha\")).\n")
            .expect("compile quoted-string exact program")
            .evaluate()
            .expect("evaluate quoted-string exact program");

    assert_eq!(
        prob_of(&result, "gate", &[Value::String("alpha".to_string())]),
        0.5
    );
}

#[test]
fn test_exact_ddnnf_materializes_runtime_f32_nan_and_infinity() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let source = "pred seed().\n\
                  pred nan_value(f32).\n\
                  pred infinite_value(f32).\n\
                  pred accepted().\n\
                  0.5::seed().\n\
                  nan_value(Y) :- seed(), Y is cast(0.0, f32) / cast(0.0, f32).\n\
                  infinite_value(Y) :- seed(), Y is cast(1.0, f32) / cast(0.0, f32).\n\
                  accepted() :- nan_value(_), infinite_value(_).\n\
                  query(accepted()).\n";

    let result = ExactDdnnfProgram::compile_source(source)
        .expect("compile non-finite f32 exact program")
        .evaluate()
        .expect("evaluate non-finite f32 exact program");
    assert!((prob0(&result, "accepted") - 0.5).abs() < 1e-9);
}

#[test]
fn test_exact_ddnnf_rejects_conflicting_quoted_and_bare_symbol_evidence() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let source = "0.5::gate(\"alpha\").\n\
                  evidence(gate(\"alpha\"), true).\n\
                  evidence(gate(alpha), false).\n\
                  query(gate(\"alpha\")).\n";

    let error = match ExactDdnnfProgram::compile_source(source) {
        Ok(program) => program
            .evaluate()
            .expect_err("contradictory symbol evidence must be rejected"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("conflicting evidence for gate"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_exact_ddnnf_deduplicates_equivalent_symbol_evidence() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let source = "0.5::gate(\"alpha\").\n\
                  evidence(gate(\"alpha\"), true).\n\
                  evidence(gate(alpha), true).\n\
                  query(gate(\"alpha\")).\n";

    let result = ExactDdnnfProgram::compile_source(source)
        .expect("compile equivalent symbol evidence")
        .evaluate()
        .expect("evaluate equivalent symbol evidence");
    assert_eq!(
        prob_of(&result, "gate", &[Value::String("alpha".to_string())]),
        1.0
    );
}

#[test]
fn test_exact_ddnnf_rejects_cross_width_rule_bindings() {
    for (source, expected) in [
        (
            "pred input(f32).\npred output(f64).\n0.5::input(0.1).\noutput(X) :- input(X).\nquery(output(0.1)).\n",
            "Type mismatch in rule for 'output'",
        ),
        (
            "pred input(u32).\npred output(u64).\n0.5::input(1).\noutput(X) :- input(X).\nquery(output(1)).\n",
            "Type mismatch in rule for 'output'",
        ),
        (
            "pred left(f32).\npred right(f64).\npred output(f32).\n0.5::left(0.1).\nright(0.1).\noutput(X) :- left(X), right(X).\nquery(output(0.1)).\n",
            "variable X is F32",
        ),
    ] {
        let error = match ExactDdnnfProgram::compile_source(source) {
            Ok(_) => panic!("cross-width rule binding must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn test_exact_ddnnf_infers_probabilistic_body_predicate_widths_from_ground_choices() {
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let source = r#"
0.5::probable(0).
0.5::exclusive(0); 0.5::exclusive(1).
observed(0).
from_fact(X) :- probable(X), observed(X).
from_choice(X) :- exclusive(X), observed(X).
query(from_fact(0)).
query(from_choice(0)).
"#;

    let result = ExactDdnnfProgram::compile_source(source)
        .expect("compile inferred-width probabilistic joins")
        .evaluate()
        .expect("evaluate inferred-width probabilistic joins");
    assert!((prob_of(&result, "from_fact", &[Value::I64(0)]) - 0.5).abs() < 1e-9);
    assert!((prob_of(&result, "from_choice", &[Value::I64(0)]) - 0.5).abs() < 1e-9);
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

#[test]
fn test_exact_ddnnf_two_sided_recursive_scc_converges() {
    // Regression: a mutually-recursive SCC with base probabilistic facts on BOTH
    // sides of the cycle previously never converged in the semi-naive provenance
    // fixpoint ("Provenance iteration limit (1024) exceeded for SCC"): the
    // convergence test compares hash-consed PIR node ids, and without OR/AND
    // flattening + absorption each round re-embedded the counterpart's formula
    // one level deeper (semantically fixed, syntactically new).
    // Fixpoint semantics: a holds iff va ∨ vb, so P(a) = P(b) = 1-(1-pa)(1-pb).
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.5406::a(1,2).
0.7143::b(1,2).
b(X,Y) :- a(X,Y).
a(X,Y) :- b(X,Y).
query(a(1,2)).
query(b(1,2)).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    let expected = 1.0 - (1.0 - 0.5406) * (1.0 - 0.7143);
    let pa = prob_of(&result, "a", &[Value::from(1_i64), Value::from(2_i64)]);
    let pb = prob_of(&result, "b", &[Value::from(1_i64), Value::from(2_i64)]);
    assert!(
        (pa - expected).abs() < 1e-9,
        "pa={} expected={}",
        pa,
        expected
    );
    assert!(
        (pb - expected).abs() < 1e-9,
        "pb={} expected={}",
        pb,
        expected
    );
}

#[test]
fn test_exact_ddnnf_two_sided_scc_duplicate_body_atom_converges() {
    // Same regression, in the live rule-graph shape that surfaced it: the SCC
    // rules repeat the body atom (X ∧ X ≡ X), which additionally exercises the
    // AND-of-OR absorption path (proof = And(delta-leaf, Or(full)) each round).
    let _lock = EXACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let source = r#"
0.5406::q10001(10034,10042).
0.7143::q10000(10034,10042).
q10000(A,B) :- q10001(A,B), q10001(A,B).
q10001(A,B) :- q10000(A,B), q10000(A,B).
query(q10000(10034,10042)).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    let expected = 1.0 - (1.0 - 0.5406) * (1.0 - 0.7143);
    let p = prob_of(
        &result,
        "q10000",
        &[Value::from(10034_i64), Value::from(10042_i64)],
    );
    assert!((p - expected).abs() < 1e-9, "p={} expected={}", p, expected);
}
