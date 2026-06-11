//! D4 decision-order hint: probability parity + frontier/compile-time measurement.
//!
//! The hint is host-side only: it renumbers leaf/choice variables from
//! provenance structure before CNF encoding, steering the deterministic
//! var-id tie-breaks of the (unchanged) D4 CUDA branching heuristic.

#![cfg(feature = "host-io")]

use std::collections::BTreeSet;

use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::pir::{PirNode, PirNodeId};
use xlog_prob::provenance::{extract_from_source, Provenance, Value};

fn has_cuda_device() -> bool {
    xlog_cuda::CudaDevice::new(0).is_ok()
}

/// Join-heavy probabilistic program: layered graph 0 -> {1,2,3} -> {4,5,6} -> 7
/// with recursive path rules. 15 probabilistic facts, shared derivations
/// (every s->t proof shares the middle-layer sub-derivations).
fn join_heavy_source() -> String {
    let mut source = String::new();
    let mut idx = 0usize;
    let mut edge = |s: &mut String, x: i64, y: i64| {
        let p = 0.3 + 0.05 * ((idx % 7) as f64);
        idx += 1;
        s.push_str(&format!("{p:.2}::edge({x}, {y}).\n"));
    };
    for a in 1..=3 {
        edge(&mut source, 0, a);
    }
    for a in 1..=3 {
        for b in 4..=6 {
            edge(&mut source, a, b);
        }
    }
    for b in 4..=6 {
        edge(&mut source, b, 7);
    }
    source.push_str(
        "path(X, Y) :- edge(X, Y).\n\
         path(X, Z) :- edge(X, Y), path(Y, Z).\n\
         query(path(0, 7)).\n",
    );
    source
}

fn finite_formula_prob(prov: &Provenance, root: PirNodeId) -> f64 {
    let leaves: Vec<_> = prov.leaf_probs.keys().copied().collect();
    let leaf_count = leaves.len();
    assert!(
        leaf_count <= 20,
        "test formula evaluator is intentionally finite"
    );

    let mut prob = 0.0;
    for mask in 0usize..(1usize << leaf_count) {
        let mut true_leaves = BTreeSet::new();
        let mut world_prob = 1.0;
        for (idx, leaf) in leaves.iter().enumerate() {
            let p = prov.leaf_probs[leaf];
            if (mask & (1usize << idx)) != 0 {
                true_leaves.insert(*leaf);
                world_prob *= p;
            } else {
                world_prob *= 1.0 - p;
            }
        }
        if eval_formula(prov, root, &true_leaves) {
            prob += world_prob;
        }
    }
    prob
}

fn eval_formula(
    prov: &Provenance,
    node: PirNodeId,
    true_leaves: &BTreeSet<xlog_prob::LeafId>,
) -> bool {
    match prov.pir.node(node).expect("valid PIR node") {
        PirNode::Const(v) => *v,
        PirNode::Lit { leaf } => true_leaves.contains(leaf),
        PirNode::NegLit { leaf } => !true_leaves.contains(leaf),
        PirNode::And { children } => children
            .iter()
            .all(|child| eval_formula(prov, *child, true_leaves)),
        PirNode::Or { children } => children
            .iter()
            .any(|child| eval_formula(prov, *child, true_leaves)),
        PirNode::Decision { .. } => panic!("test fixture uses probabilistic facts only"),
    }
}

struct Measurement {
    prob: f64,
    frontier_items: u32,
    d4_compile_sec: f64,
    verify_sec: f64,
    compile_wall_sec: f64,
}

fn compile_and_measure(source: &str, hint: bool, cache_dir: &std::path::Path) -> Measurement {
    // Fresh disk-cache dir per compile so every round is a real D4 compile.
    std::env::set_var("XLOG_CIRCUIT_CACHE_DIR", cache_dir);
    let mut config = GpuConfig::default();
    config.memory_bytes = 1 << 30;
    config.decision_order_hint = hint;
    let t0 = std::time::Instant::now();
    let compiled = ExactDdnnfProgram::compile_source_with_gpu(source, config)
        .expect("compile join-heavy exact program");
    let compile_wall_sec = t0.elapsed().as_secs_f64();
    let profile = compiled
        .last_compile_profile()
        .expect("profile must be populated under XLOG_WARMUP_PROFILE=1")
        .clone();
    assert!(
        !profile.gpu_cache_hit && !profile.disk_cache_hit,
        "measurement requires a real D4 compile (hint={hint}): {profile:?}"
    );
    let result = compiled.evaluate().expect("evaluate join-heavy program");
    let prob = result
        .query_probs
        .iter()
        .find(|q| q.atom.predicate == "path")
        .expect("path query result")
        .prob;
    Measurement {
        prob,
        frontier_items: profile.frontier_items,
        d4_compile_sec: profile.d4_compile_sec,
        verify_sec: profile.verify_sec,
        compile_wall_sec,
    }
}

// Note: a larger 19-fact 4-layer variant of this fixture was probed for
// evidence and hard-fails D4 Phase-1 compilation in BOTH hint modes
// (device-side trap, CUDA_ERROR_LAUNCH_FAILED, at 1 GB and 8 GB budgets) —
// recorded in docs/evidence/2026-06-11-d4-structure-hints/README.md. The
// committed measurement therefore uses the largest fixture the current
// compiler handles in this harness (15 facts).

#[test]
fn decision_order_hint_preserves_probabilities_and_reports_frontier() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // Isolate the disk cache and enable per-stage profiling. This test is the
    // only test in this binary, so process-global env mutation is safe.
    let base_dir = std::env::temp_dir().join(format!(
        "xlog-d4-hint-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&base_dir).expect("create isolated cache dir");
    std::env::set_var("XLOG_WARMUP_PROFILE", "1");

    let source = join_heavy_source();

    // Host finite oracle over the (hint-independent) provenance semantics.
    let prov = extract_from_source(&source).expect("extract join-heavy provenance");
    let root = prov
        .query_formula("path", &[Value::I64(0), Value::I64(7)])
        .expect("path(0,7) formula");
    let oracle = finite_formula_prob(&prov, root);

    // Alternating repeated rounds to separate signal from launch/JIT noise.
    const ROUNDS: usize = 3;
    let mut off_runs: Vec<Measurement> = Vec::new();
    let mut on_runs: Vec<Measurement> = Vec::new();
    for round in 0..ROUNDS {
        for hint in [false, true] {
            let cache_dir = base_dir.join(format!("round-{round}-hint-{hint}"));
            std::fs::create_dir_all(&cache_dir).expect("create round cache dir");
            let m = compile_and_measure(&source, hint, &cache_dir);
            assert!(
                (m.prob - oracle).abs() <= 1e-9,
                "round {round} hint={hint}: probability {} != oracle {oracle}",
                m.prob
            );
            eprintln!(
                "d4 decision-order hint measurement: round={round} hint={hint} \
                 frontier_items={} d4_compile_sec={:.4} verify_sec={:.4} \
                 compile_wall_sec={:.2} prob={:.12}",
                m.frontier_items, m.d4_compile_sec, m.verify_sec, m.compile_wall_sec, m.prob
            );
            if hint {
                on_runs.push(m);
            } else {
                off_runs.push(m);
            }
        }
    }

    let median = |runs: &mut Vec<f64>| -> f64 {
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        runs[runs.len() / 2]
    };
    let med_d4_off = median(&mut off_runs.iter().map(|m| m.d4_compile_sec).collect());
    let med_d4_on = median(&mut on_runs.iter().map(|m| m.d4_compile_sec).collect());
    let med_verify_off = median(&mut off_runs.iter().map(|m| m.verify_sec).collect());
    let med_verify_on = median(&mut on_runs.iter().map(|m| m.verify_sec).collect());
    eprintln!(
        "d4 decision-order hint medians: frontier_items off={} on={}, \
         d4_compile_sec off={med_d4_off:.4} on={med_d4_on:.4}, \
         verify_sec off={med_verify_off:.4} on={med_verify_on:.4}, oracle={oracle:.12}",
        off_runs[0].frontier_items, on_runs[0].frontier_items,
    );

    for (off, on) in off_runs.iter().zip(on_runs.iter()) {
        assert!(
            (on.prob - off.prob).abs() <= 1e-9,
            "hint must not change probabilities: off={} on={}",
            off.prob,
            on.prob
        );
    }

    let _ = std::fs::remove_dir_all(&base_dir);
}
