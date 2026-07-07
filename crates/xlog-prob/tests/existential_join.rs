#![cfg(feature = "host-io")]

//! Engine-level semantics for the Stage-B existential join.
//!
//! The pyxlog neural template path grounds a neural predicate over a real join
//! domain and lets the deterministic provenance machinery OR-aggregate the per-
//! domain contributions at each head binding. This test locks that foundation at
//! the `xlog-prob` exact layer — with the neural predicate stood in by per-event
//! probabilistic facts — so the engine semantics the Python layer relies on are
//! guarded against a future refactor, independent of the neural/template code.

use xlog_prob::exact::{ExactDdnnfProgram, ExactResult};
use xlog_prob::provenance::Value;

fn prob_of(result: &ExactResult, predicate: &str, args: &[Value]) -> f64 {
    result
        .query_probs
        .iter()
        .find(|q| q.atom.predicate == predicate && q.atom.args == args)
        .unwrap_or_else(|| panic!("missing query result for {} {:?}", predicate, args))
        .prob
}

/// A neural-predicate stand-in (`saliency_s`, one prob fact per event) joined to
/// an ordinary relation (`pre_before_post`) on the existential variable `Event`,
/// gated by a guard, must yield the guard-gated noisy-OR over the events joined
/// to each head binding `Edge`:
///   P(plastic(e)) = P(guard) * (1 - prod_{v : pre_before_post(v,e)} (1 - p_v)).
#[test]
fn existential_join_or_aggregation_semantics() {
    let source = r#"
0.7::saliency_s(0).
0.6::saliency_s(1).
0.8::saliency_s(2).
pre_before_post(0, 0).
pre_before_post(1, 0).
pre_before_post(2, 1).
0.8807970779778823::guard().
plastic(Edge) :- saliency_s(Event), pre_before_post(Event, Edge), guard().
query(plastic(0)).
query(plastic(1)).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate().unwrap();

    let g = 0.8807970779778823_f64; // sigmoid(2.0)
    let expected0 = g * (1.0 - (1.0 - 0.7) * (1.0 - 0.6)); // events {0,1} -> edge 0
    let expected1 = g * 0.8; // event {2} -> edge 1

    let got0 = prob_of(&result, "plastic", &[Value::I64(0)]);
    let got1 = prob_of(&result, "plastic", &[Value::I64(1)]);

    assert!(
        (got0 - expected0).abs() < 1e-9,
        "plastic(0)={} expected {}",
        got0,
        expected0
    );
    assert!(
        (got1 - expected1).abs() < 1e-9,
        "plastic(1)={} expected {}",
        got1,
        expected1
    );
}
