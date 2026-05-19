// crates/xlog-logic/tests/test_hypergraph_certification.rs
//! v0.6.2 hypergraph certification workloads (PR 7).
//!
//! End-to-end coverage on representative WCOJ workloads, exercising
//! the full oracle stack assembled in PRs 1–6:
//!
//!   plan_rules → typed oracle eval → canonical explain
//!
//! Each test below stands up a self-contained fixture (rules +
//! base relations), drives it through every layer, and asserts:
//!   1. plan_rules dispatch verdicts (multiway / fallback)
//!   2. typed oracle row equivalence (vs hand-computed expected)
//!   3. canonical explain contains the expected structural shape
//!
//! Workloads:
//!
//!   - **Triangle**: 3-atom equi-join (`tri(X, Y, Z) :- e(X,Y), e(Y,Z), e(X,Z)`).
//!     Pure multiway; no recursion. Locks the simplest WCOJ shape.
//!   - **Same Generation**: classic recursive sg over a parent
//!     relation. Two rules, both multiway. Drives `evaluate_fixpoint_typed`.
//!     Expected set computed by an independent nested-loop
//!     reference implementation (`sg_reference`) so any oracle
//!     regression triggers a two-implementation disagreement.
//!   - **Skewed multiway**: 3-atom join where one relation has many
//!     rows and the others have few. Tests that the oracle is correct
//!     under selectivity asymmetry — independent of (future) cost
//!     model choices.
//!   - **Deep recursive frontier**: long-chain transitive closure
//!     using a 2-hop base case + recursive step. Tests fixpoint
//!     convergence for multi-iteration recursion via
//!     `evaluate_fixpoint_typed`.
//!   - **Mutually-recursive parity SCC**: even/odd path parity
//!     over a chain — true SCC (`even_path` ↔ `odd_path`), each
//!     side has a multiway base + multiway step. Drives
//!     `evaluate_scc_fixpoint_typed`, the only entry point that
//!     needs cross-predicate schema freezing and predicate-order
//!     determinism.
//!
//! Hard boundary: no RIR, no executor, no CUDA, no cost model.

use std::collections::{BTreeMap, BTreeSet};
use xlog_core::ScalarType;
use xlog_logic::ast::{Atom, BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{
    evaluate_fixpoint_typed, evaluate_rule_typed, evaluate_scc_fixpoint_typed, explain_plans,
    plan_rules, AppearanceOrder, FixpointConfig, RefRelation, RefRelationStore, RefValue, RulePlan,
};

// ---------------------------------------------------------------
// Test helpers (shared across all workloads)
// ---------------------------------------------------------------

fn var(name: &str) -> Term {
    Term::Variable(name.to_string())
}

fn atom(predicate: &str, terms: Vec<Term>) -> Atom {
    Atom {
        predicate: predicate.to_string(),
        terms,
    }
}

fn pos(predicate: &str, terms: Vec<Term>) -> BodyLiteral {
    BodyLiteral::Positive(atom(predicate, terms))
}

fn rule_with(head: Atom, body: Vec<BodyLiteral>) -> Rule {
    Rule { head, body }
}

fn u32_relation(rows: &[&[u32]]) -> RefRelation {
    let arity = rows.first().map(|r| r.len()).unwrap_or(0);
    RefRelation {
        schema: vec![ScalarType::U32; arity],
        rows: rows
            .iter()
            .map(|r| r.iter().map(|v| RefValue::U32(*v)).collect())
            .collect(),
    }
}

fn store_with_one(name: &str, rel: RefRelation) -> RefRelationStore {
    let mut s: RefRelationStore = BTreeMap::new();
    s.insert(name.to_string(), rel);
    s
}

/// Project a result row Vec<Vec<RefValue>> down to (u32, u32, u32)
/// triples for arity-3 relations.
fn triples(rows: &[Vec<RefValue>]) -> Vec<(u32, u32, u32)> {
    let mut out: Vec<(u32, u32, u32)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1], &r[2]) {
            (RefValue::U32(a), RefValue::U32(b), RefValue::U32(c)) => (*a, *b, *c),
            other => panic!("unexpected row shape: {other:?}"),
        })
        .collect();
    out.sort();
    out
}

/// Project a result row Vec<Vec<RefValue>> down to (u32, u32) pairs
/// for arity-2 relations.
fn pairs(rows: &[Vec<RefValue>]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (RefValue::U32(a), RefValue::U32(b)) => (*a, *b),
            other => panic!("unexpected row shape: {other:?}"),
        })
        .collect();
    out.sort();
    out
}

fn pairs_from_rel(rel: &RefRelation) -> Vec<(u32, u32)> {
    pairs(&rel.rows)
}

// ---------------------------------------------------------------
// Workload 1: Triangle
// ---------------------------------------------------------------

#[test]
fn triangle_certification() {
    // tri(X, Y, Z) :- e(X, Y), e(Y, Z), e(X, Z)
    //
    // Directed graph chosen so the K_4-minus-nothing subgraph
    // {1, 2, 3, 4} produces a known set of directed triangles,
    // plus a disjoint smaller triangle on {5, 6, 7}:
    //
    //   1 → 2, 1 → 3, 1 → 4
    //   2 → 3, 2 → 4
    //   3 → 4
    //   5 → 6, 5 → 7, 6 → 7
    //
    // Directed triangles tri(X,Y,Z) where all three edges exist:
    //   (1,2,3): e(1,2), e(2,3), e(1,3)
    //   (1,2,4): e(1,2), e(2,4), e(1,4)
    //   (1,3,4): e(1,3), e(3,4), e(1,4)
    //   (2,3,4): e(2,3), e(3,4), e(2,4)
    //   (5,6,7): e(5,6), e(6,7), e(5,7)
    let r = rule_with(
        atom("tri", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
            pos("e", vec![var("X"), var("Z")]),
        ],
    );
    let edges = u32_relation(&[
        &[1, 2],
        &[1, 3],
        &[1, 4],
        &[2, 3],
        &[2, 4],
        &[3, 4],
        &[5, 6],
        &[5, 7],
        &[6, 7],
    ]);
    let store = store_with_one("e", edges);

    // Layer 1: plan_rules dispatch
    let plans = plan_rules(&[r.clone()], &store).expect("triangle plans");
    assert_eq!(plans.len(), 1);
    match &plans[0] {
        RulePlan::MultiwayCandidate {
            head_predicate,
            hypergraph,
            variable_order,
        } => {
            assert_eq!(head_predicate, "tri");
            assert_eq!(hypergraph.hyperedge_count(), 3);
            assert_eq!(variable_order.len(), 3);
        }
        other => panic!("expected MultiwayCandidate, got {other:?}"),
    }

    // Layer 2: typed oracle eval
    let rows = evaluate_rule_typed(&r, &store, &AppearanceOrder).expect("triangle eval");
    assert_eq!(
        triples(&rows),
        vec![(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (5, 6, 7),]
    );

    // Layer 3: canonical explain
    let explained = explain_plans(&plans);
    assert!(
        explained.contains("tri/0: multiway"),
        "expected multiway tri in explain:\n{explained}"
    );
}

// ---------------------------------------------------------------
// Workload 2: Same Generation
// ---------------------------------------------------------------

/// Independent reference implementation of Same Generation. Used
/// to verify the oracle's output via two-implementation agreement
/// rather than a hand-traced expected set. Plain nested-loop
/// fixpoint over the parent edges; no rule machinery.
fn sg_reference(parent_edges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut sg: BTreeSet<(u32, u32)> = BTreeSet::new();
    // Base case: every pair (X, Y) sharing some parent P.
    for (x, p_x) in parent_edges {
        for (y, p_y) in parent_edges {
            if p_x == p_y {
                sg.insert((*x, *y));
            }
        }
    }
    // Step case: for each (A, B) currently in sg, every (X, Y)
    // such that parent(X) = A and parent(Y) = B joins in.
    // Iterate to fixpoint.
    loop {
        let snapshot: Vec<(u32, u32)> = sg.iter().copied().collect();
        let before_len = sg.len();
        for (a, b) in &snapshot {
            for (x, p_x) in parent_edges {
                if p_x != a {
                    continue;
                }
                for (y, p_y) in parent_edges {
                    if p_y != b {
                        continue;
                    }
                    sg.insert((*x, *y));
                }
            }
        }
        if sg.len() == before_len {
            break;
        }
    }
    sg.into_iter().collect()
}

#[test]
fn same_generation_certification() {
    // sg(X, Y) :- parent(X, P), parent(Y, P).            -- siblings + self
    // sg(X, Y) :- parent(X, A), sg(A, B), parent(Y, B).  -- one-gen step
    //
    // Family tree (child → parent):
    //   1 → 10, 2 → 10           (siblings sharing parent 10)
    //   11 → 1, 13 → 1           (siblings sharing parent 1)
    //   12 → 2                    (only child of 2)
    //   14 → 12                   (one gen below 11/12/13)
    //
    // Expected set is computed by an independent nested-loop SG
    // fixpoint (`sg_reference`) to give two-implementation
    // agreement — the oracle's output and the reference must
    // match. Locks regression detection beyond a hand-trace.
    let r_base = rule_with(
        atom("sg", vec![var("X"), var("Y")]),
        vec![
            pos("parent", vec![var("X"), var("P")]),
            pos("parent", vec![var("Y"), var("P")]),
        ],
    );
    let r_step = rule_with(
        atom("sg", vec![var("X"), var("Y")]),
        vec![
            pos("parent", vec![var("X"), var("A")]),
            pos("sg", vec![var("A"), var("B")]),
            pos("parent", vec![var("Y"), var("B")]),
        ],
    );
    let parent_pairs: Vec<(u32, u32)> = vec![(1, 10), (2, 10), (11, 1), (12, 2), (13, 1), (14, 12)];
    let parent_rows: Vec<Vec<u32>> = parent_pairs.iter().map(|(c, p)| vec![*c, *p]).collect();
    let parent_refs: Vec<&[u32]> = parent_rows.iter().map(|v| v.as_slice()).collect();
    let store = store_with_one("parent", u32_relation(&parent_refs));
    let rules = vec![r_base, r_step];

    // Layer 1: plan_rules → both multiway
    let plans = plan_rules(&rules, &store).expect("sg plans");
    assert_eq!(plans.len(), 2);
    for (i, p) in plans.iter().enumerate() {
        match p {
            RulePlan::MultiwayCandidate { head_predicate, .. } => {
                assert_eq!(head_predicate, "sg", "rule index {i}");
            }
            other => panic!("expected MultiwayCandidate for rule {i}, got {other:?}"),
        }
    }

    // Layer 2: typed fixpoint eval, compared against independent
    // reference implementation.
    let result = evaluate_fixpoint_typed(
        &rules,
        &store,
        "sg",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("sg fixpoint converges");
    let expected = sg_reference(&parent_pairs);
    assert_eq!(
        pairs_from_rel(&result),
        expected,
        "oracle output disagrees with reference SG impl"
    );

    // Layer 3: canonical explain — both rules under same head.
    let explained = explain_plans(&plans);
    assert!(
        explained.contains("sg/0:") && explained.contains("sg/1:"),
        "expected both sg rules in explain:\n{explained}"
    );
    let multiway_count = explained.matches("multiway vars=").count();
    assert_eq!(multiway_count, 2, "expected 2 multiway lines:\n{explained}");
}

// ---------------------------------------------------------------
// Workload 5: Mutually-recursive SCC (parity over a chain)
// ---------------------------------------------------------------

#[test]
fn mutually_recursive_parity_scc_certification() {
    // Drives evaluate_scc_fixpoint_typed end-to-end on a true
    // SCC (a ↔ b mutual recursion). Each predicate has a 2-hop
    // base case + a recursive step; both are multiway.
    //
    //   even_path(X, Z) :- e(X, M), e(M, Z).             -- base, 2-hop
    //   even_path(X, Z) :- e(X, Y), odd_path(Y, Z).      -- step
    //   odd_path(X, Z)  :- e(X, M), e(M, Y), e(Y, Z).    -- base, 3-hop
    //   odd_path(X, Z)  :- e(X, Y), even_path(Y, Z).     -- step
    //
    // Chain: 1 → 2 → 3 → 4 → 5 → 6 (length 5).
    // Reachable distances from i to j (j > i): j - i ∈ {1, 2, 3, 4, 5}.
    // even_path covers j - i ∈ {2, 4}, odd_path covers j - i ∈ {3, 5}.
    let even_seed = rule_with(
        atom("even_path", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("M")]),
            pos("e", vec![var("M"), var("Z")]),
        ],
    );
    let even_step = rule_with(
        atom("even_path", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("odd_path", vec![var("Y"), var("Z")]),
        ],
    );
    let odd_seed = rule_with(
        atom("odd_path", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("M")]),
            pos("e", vec![var("M"), var("Y")]),
            pos("e", vec![var("Y"), var("Z")]),
        ],
    );
    let odd_step = rule_with(
        atom("odd_path", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("even_path", vec![var("Y"), var("Z")]),
        ],
    );
    let chain_len: u32 = 6;
    let chain_rows: Vec<Vec<u32>> = (1..chain_len).map(|i| vec![i, i + 1]).collect();
    let chain_refs: Vec<&[u32]> = chain_rows.iter().map(|v| v.as_slice()).collect();
    let store = store_with_one("e", u32_relation(&chain_refs));
    let mut rules: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
    rules.insert("even_path".into(), vec![even_seed, even_step]);
    rules.insert("odd_path".into(), vec![odd_seed, odd_step]);

    // Layer 1: plan_rules over a flattened slice — every rule
    // must be MultiwayCandidate.
    let flat: Vec<Rule> = rules.values().flatten().cloned().collect();
    let plans = plan_rules(&flat, &store).expect("scc plans");
    assert_eq!(plans.len(), 4);
    for (i, p) in plans.iter().enumerate() {
        match p {
            RulePlan::MultiwayCandidate { head_predicate, .. } => {
                assert!(
                    head_predicate == "even_path" || head_predicate == "odd_path",
                    "unexpected head at index {i}: {head_predicate}"
                );
            }
            other => panic!("expected MultiwayCandidate for rule {i}, got {other:?}"),
        }
    }

    // Layer 2: SCC typed fixpoint eval
    let result =
        evaluate_scc_fixpoint_typed(&rules, &store, &AppearanceOrder, &FixpointConfig::default())
            .expect("SCC fixpoint converges");

    // Expected: even_path covers (i, i + d) for d ∈ {2, 4} when
    // i + d ≤ chain_len; odd_path covers d ∈ {3, 5}.
    let mut even_expected: Vec<(u32, u32)> = Vec::new();
    for i in 1..=chain_len {
        for d in [2u32, 4u32] {
            if i + d <= chain_len {
                even_expected.push((i, i + d));
            }
        }
    }
    even_expected.sort();
    let mut odd_expected: Vec<(u32, u32)> = Vec::new();
    for i in 1..=chain_len {
        for d in [3u32, 5u32] {
            if i + d <= chain_len {
                odd_expected.push((i, i + d));
            }
        }
    }
    odd_expected.sort();

    assert_eq!(
        pairs_from_rel(result.get("even_path").expect("even_path present")),
        even_expected
    );
    assert_eq!(
        pairs_from_rel(result.get("odd_path").expect("odd_path present")),
        odd_expected
    );

    // Layer 3: canonical explain — all 4 rules surface as multiway.
    let explained = explain_plans(&plans);
    let multiway_count = explained.matches("multiway vars=").count();
    assert_eq!(multiway_count, 4, "expected 4 multiway lines:\n{explained}");
    // Sanity: even_path appears before odd_path lex.
    let pos_even = explained.find("even_path/").expect("even_path present");
    let pos_odd = explained.find("odd_path/").expect("odd_path present");
    assert!(
        pos_even < pos_odd,
        "expected even_path before odd_path in canonical explain:\n{explained}"
    );
}

// ---------------------------------------------------------------
// Workload 3: Skewed multiway
// ---------------------------------------------------------------

#[test]
fn skewed_multiway_certification() {
    // result(X, Y, Z) :- big(X, Y), small_a(Y, Z), small_b(X, Z)
    //
    // big has many rows (a few thousand could be used in a perf
    // bench — here we use 32 rows to keep the test fast); small_a
    // and small_b each have 4 rows. The oracle correctness should
    // not depend on join order, even though a future cost model
    // would reorder. This test locks the row set, not any particular
    // execution order.
    let r = rule_with(
        atom("result", vec![var("X"), var("Y"), var("Z")]),
        vec![
            pos("big", vec![var("X"), var("Y")]),
            pos("small_a", vec![var("Y"), var("Z")]),
            pos("small_b", vec![var("X"), var("Z")]),
        ],
    );

    // big covers all (X, Y) where X, Y ∈ 1..=8 except diagonal.
    let big_rows: Vec<Vec<u32>> = (1u32..=8)
        .flat_map(|x| (1u32..=8).filter(move |y| *y != x).map(move |y| vec![x, y]))
        .collect();
    let big_refs: Vec<&[u32]> = big_rows.iter().map(|v| v.as_slice()).collect();
    let big = u32_relation(&big_refs);

    // small_a: only (Y=2, Z=10), (Y=3, Z=20), (Y=4, Z=30), (Y=5, Z=40)
    let small_a = u32_relation(&[&[2, 10], &[3, 20], &[4, 30], &[5, 40]]);
    // small_b: only (X=1, Z=10), (X=2, Z=20), (X=3, Z=30), (X=4, Z=40)
    let small_b = u32_relation(&[&[1, 10], &[2, 20], &[3, 30], &[4, 40]]);

    let mut store: RefRelationStore = BTreeMap::new();
    store.insert("big".into(), big);
    store.insert("small_a".into(), small_a);
    store.insert("small_b".into(), small_b);

    // Layer 1: plan dispatch — must be multiway.
    let plans = plan_rules(&[r.clone()], &store).expect("skewed plans");
    match &plans[0] {
        RulePlan::MultiwayCandidate { hypergraph, .. } => {
            assert_eq!(hypergraph.hyperedge_count(), 3);
        }
        other => panic!("expected MultiwayCandidate, got {other:?}"),
    }

    // Layer 2: typed eval
    let rows = evaluate_rule_typed(&r, &store, &AppearanceOrder).expect("skewed eval");
    // Hand-computed: for each (X, Z) pair in small_b (4 pairs),
    // find Y such that small_a(Y, Z) AND big(X, Y) AND X ≠ Y.
    //   X=1, Z=10: small_a(Y,10) → Y=2; big(1,2) ✓ → (1,2,10)
    //   X=2, Z=20: small_a(Y,20) → Y=3; big(2,3) ✓ → (2,3,20)
    //   X=3, Z=30: small_a(Y,30) → Y=4; big(3,4) ✓ → (3,4,30)
    //   X=4, Z=40: small_a(Y,40) → Y=5; big(4,5) ✓ → (4,5,40)
    let expected = vec![(1, 2, 10), (2, 3, 20), (3, 4, 30), (4, 5, 40)];
    assert_eq!(triples(&rows), expected);

    // Layer 3: explain
    let explained = explain_plans(&plans);
    assert!(
        explained.contains("result/0: multiway"),
        "expected multiway result in explain:\n{explained}"
    );
}

// ---------------------------------------------------------------
// Workload 4: Deep recursive frontier
// ---------------------------------------------------------------

#[test]
fn deep_recursive_frontier_certification() {
    // 2-hop base + recursive step over a long linear chain.
    //
    //   reach(X, Y) :- e(X, M), e(M, Y).
    //   reach(X, Z) :- e(X, Y), reach(Y, Z).
    //
    // Chain: 1 → 2 → 3 → … → 12 (length 11). Reach should
    // enumerate every (i, j) where j > i + 1 (since base case
    // skips 1-hop).
    let r_base = rule_with(
        atom("reach", vec![var("X"), var("Y")]),
        vec![
            pos("e", vec![var("X"), var("M")]),
            pos("e", vec![var("M"), var("Y")]),
        ],
    );
    let r_step = rule_with(
        atom("reach", vec![var("X"), var("Z")]),
        vec![
            pos("e", vec![var("X"), var("Y")]),
            pos("reach", vec![var("Y"), var("Z")]),
        ],
    );
    let chain_len: u32 = 12;
    let chain_rows: Vec<Vec<u32>> = (1..chain_len).map(|i| vec![i, i + 1]).collect();
    let chain_refs: Vec<&[u32]> = chain_rows.iter().map(|v| v.as_slice()).collect();
    let edges = u32_relation(&chain_refs);
    let store = store_with_one("e", edges);
    let rules = vec![r_base, r_step];

    // Layer 1: plan dispatch — both multiway
    let plans = plan_rules(&rules, &store).expect("frontier plans");
    assert_eq!(plans.len(), 2);
    for (i, p) in plans.iter().enumerate() {
        match p {
            RulePlan::MultiwayCandidate { head_predicate, .. } => {
                assert_eq!(head_predicate, "reach", "rule index {i}");
            }
            other => panic!("expected MultiwayCandidate for rule {i}, got {other:?}"),
        }
    }

    // Layer 2: typed fixpoint eval
    let result = evaluate_fixpoint_typed(
        &rules,
        &store,
        "reach",
        &AppearanceOrder,
        &FixpointConfig::default(),
    )
    .expect("frontier fixpoint converges");

    // Expected: every (i, j) with 1 ≤ i < j ≤ 12 AND j - i ≥ 2.
    // (Base case skips 1-hop; recursive step extends 2-hop+.)
    let mut expected: Vec<(u32, u32)> = Vec::new();
    for i in 1..chain_len {
        for j in (i + 2)..=chain_len {
            expected.push((i, j));
        }
    }
    expected.sort();
    assert_eq!(pairs_from_rel(&result), expected);

    // Layer 3: canonical explain — both reach rules multiway.
    let explained = explain_plans(&plans);
    assert!(
        explained.contains("reach/0:") && explained.contains("reach/1:"),
        "expected both reach rules in explain:\n{explained}"
    );
    let multiway_count = explained.matches("multiway vars=").count();
    assert_eq!(multiway_count, 2, "expected 2 multiway lines:\n{explained}");
}
