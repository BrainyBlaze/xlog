//! W2.6 step 7 Part B — compile-time leader divergence via
//! hand-built `StatsSnapshot`. 4 tests: 2 shapes (triangle +
//! 4-cycle) × 2 signal types (heat-bias + selectivity-bias).
//! Hand-built snapshots pin EXACT heat / selectivity / cardinality
//! values reaching the compile-time cost model — sidesteps EMA
//! smoothing so the locked formula's leader decision is
//! deterministic. Part C exercises the runtime → snapshot capture
//! path end-to-end; Part B exercises the snapshot → cost-model
//! half independently.

use xlog_core::RelId;
use xlog_ir::rir::VariableOrder;
use xlog_ir::RirNode;
use xlog_logic::compile::Compiler;
use xlog_logic::compiler_config::{CompilerConfig, WcojVarOrderingKind};
use xlog_stats::{JoinSelectivity, RelationStats, StatsSnapshot};

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

type SelectivitySeed<'a> = (&'a str, &'a str, f64, Vec<usize>, Vec<usize>);

/// Build a `StatsSnapshot` literal with heat + cardinality per rel.
/// `seeded` = (predicate_name, cardinality, heat) per rel; RelIds
/// assigned 0..n in input order. The compiler resolves them via
/// predicate name during the optimizer's stats merge.
fn make_snapshot_with_heat(
    seeded: &[(&str, u64, f32)],
    selectivities: &[SelectivitySeed<'_>],
) -> StatsSnapshot {
    let relations: Vec<RelationStats> = seeded
        .iter()
        .enumerate()
        .map(|(i, (_, card, heat))| {
            let mut s = RelationStats::new(RelId(i as u32));
            s.cardinality = *card;
            s.heat = *heat;
            s
        })
        .collect();
    let rel_names: Vec<(RelId, String)> = seeded
        .iter()
        .enumerate()
        .map(|(i, (name, _, _))| (RelId(i as u32), (*name).to_string()))
        .collect();
    // Resolve predicate names → snapshot RelIds for selectivities.
    let name_to_id: std::collections::HashMap<&str, RelId> = seeded
        .iter()
        .enumerate()
        .map(|(i, (name, _, _))| (*name, RelId(i as u32)))
        .collect();
    let join_selectivities: Vec<JoinSelectivity> = selectivities
        .iter()
        .map(|(left_name, right_name, sel, lk, rk)| {
            let left_rel = name_to_id[left_name];
            let right_rel = name_to_id[right_name];
            // Canonicalize per StatsManager's canonical_join_key
            // (smaller RelId on the left).
            let (l, r, lk, rk) = if left_rel.0 <= right_rel.0 {
                (left_rel, right_rel, lk.clone(), rk.clone())
            } else {
                (right_rel, left_rel, rk.clone(), lk.clone())
            };
            let mut js = JoinSelectivity::new(l, r);
            js.set_keys(lk, rk);
            js.selectivity = *sel;
            js
        })
        .collect();
    StatsSnapshot {
        relations,
        join_selectivities,
        rel_names,
    }
}

fn first_var_order(plan: &xlog_ir::ExecutionPlan) -> Option<VariableOrder> {
    fn find(node: &RirNode) -> Option<VariableOrder> {
        match node {
            RirNode::MultiWayJoin { var_order, .. } => var_order.clone(),
            RirNode::Filter { input, .. }
            | RirNode::Project { input, .. }
            | RirNode::GroupBy { input, .. }
            | RirNode::Distinct { input, .. } => find(input),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                find(left).or_else(|| find(right))
            }
            RirNode::Union { inputs } => inputs.iter().find_map(find),
            RirNode::Fixpoint {
                base, recursive, ..
            } => find(base).or_else(|| find(recursive)),
            _ => None,
        }
    }
    plan.rules_by_scc
        .iter()
        .flatten()
        .find_map(|r| find(&r.body))
}

const TRIANGLE_SRC: &str = "\
e1(1, 2). e1(3, 4).
e2(2, 5). e2(4, 6).
e3(1, 5). e3(3, 6).
result(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).
";

const CYCLE4_SRC: &str = "\
e1(1, 2). e1(3, 4).
e2(2, 5). e2(4, 6).
e3(5, 7). e3(6, 8).
e4(7, 1). e4(8, 3).
result(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
";

fn compile_with(
    source: &str,
    snapshot: &StatsSnapshot,
    kind: WcojVarOrderingKind,
) -> xlog_ir::ExecutionPlan {
    let mut compiler = Compiler::new();
    let config = CompilerConfig {
        wcoj_variable_ordering: kind,
        ..CompilerConfig::default()
    };
    compiler
        .compile_with_config_and_stats_snapshot(source, &config, Some(snapshot))
        .expect("compile_with_config_and_stats_snapshot")
}

// ===============================================================
// Triangle — heat-bias snapshot drives leader_idx ≠ 0 under HeatAware
// ===============================================================

#[test]
fn triangle_heat_bias_heat_aware_picks_non_default_leader_card_eq_returns_none() {
    // All rels at card = 100; e1 (canonical idx 0) at heat = 0.5;
    // others at heat = 0.0. No selectivity records.
    // Locked formula: score idx0 = 100*3*2 = 600; idx1/2 = 100*1*2 = 200.
    // argmin = idx 1 (first hit). Ratio 200/600 = 0.333 ≤ 0.5
    // → HeatAware returns Some(1).
    let snap =
        make_snapshot_with_heat(&[("e1", 100, 0.5), ("e2", 100, 0.0), ("e3", 100, 0.0)], &[]);
    let plan_heat = compile_with(TRIANGLE_SRC, &snap, WcojVarOrderingKind::HeatAware);
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(vo_heat.leader_idx, 1);

    // Same snapshot under LeaderCardinality: cards equal → W2.1
    // short-circuits with argmin == 0 → returns None.
    let plan_card = compile_with(TRIANGLE_SRC, &snap, WcojVarOrderingKind::LeaderCardinality);
    let vo_card = first_var_order(&plan_card);
    assert!(
        vo_card.is_none(),
        "LeaderCardinality on same snapshot must return None (cards equal); got {:?}",
        vo_card
    );
}

#[test]
fn triangle_selectivity_bias_heat_aware_picks_not_in_tight_edge() {
    // Cards equal at 100; heat = 0 across all rels. ONE
    // selectivity record on (e1, e2) with sel = 0.01.
    // Penalties: rel_e1 = 1/0.01 + 1 = 101; rel_e2 = 101;
    // rel_e3 = 1 + 1 = 2. Heat factor 1.
    // Scores: idx0 = idx1 = 100*1*101 = 10100; idx2 = 100*1*2 = 200.
    // argmin = idx 2. Ratio 200/10100 ≈ 0.020 ≤ 0.5
    // → HeatAware returns Some(2).
    let snap = make_snapshot_with_heat(
        &[("e1", 100, 0.0), ("e2", 100, 0.0), ("e3", 100, 0.0)],
        &[("e1", "e2", 0.01, vec![1], vec![0])],
    );
    let plan_heat = compile_with(TRIANGLE_SRC, &snap, WcojVarOrderingKind::HeatAware);
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(vo_heat.leader_idx, 2);

    // Same snapshot under LeaderCardinality: cards equal → None.
    let plan_card = compile_with(TRIANGLE_SRC, &snap, WcojVarOrderingKind::LeaderCardinality);
    assert!(first_var_order(&plan_card).is_none());
}

// ===============================================================
// 4-cycle — same shape on the 4-rel topology
// ===============================================================

#[test]
fn cycle4_heat_bias_heat_aware_picks_non_default_leader_card_eq_returns_none() {
    // 4 rels at card = 100; e1 (idx 0) heat = 0.5; others 0.
    // Score idx0 = 100*3*2 = 600; idx1/2/3 = 100*1*2 = 200.
    // argmin = idx 1 (first hit). Ratio 0.333 ≤ 0.5 → Some(1).
    let snap = make_snapshot_with_heat(
        &[
            ("e1", 100, 0.5),
            ("e2", 100, 0.0),
            ("e3", 100, 0.0),
            ("e4", 100, 0.0),
        ],
        &[],
    );
    let plan_heat = compile_with(CYCLE4_SRC, &snap, WcojVarOrderingKind::HeatAware);
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(vo_heat.leader_idx, 1);

    let plan_card = compile_with(CYCLE4_SRC, &snap, WcojVarOrderingKind::LeaderCardinality);
    assert!(first_var_order(&plan_card).is_none());
}

#[test]
fn cycle4_selectivity_bias_heat_aware_picks_not_in_tight_edge() {
    // 4 rels at card = 100; heat 0; ONE tight edge (e1, e2)
    // sel = 0.01. Penalties: e1 ∈ {(0,1) tight, (3,0) default}
    //   → 101 + 1 = 102. Wait, check edge keys. 4-cycle (3,0)
    //   reversed: rel0.col0 ↔ rel3.col1, so the edge between e1
    //   and e4 (idx 3). The tight edge is only (e1, e2) here.
    //
    //   e1 ∈ {(0,1) tight=0.01, (3,0)default}: 1/0.01 + 1 = 101.
    //   e2 ∈ {(0,1) tight, (1,2) default}: 101.
    //   e3 ∈ {(1,2), (2,3)}: 1 + 1 = 2.
    //   e4 ∈ {(2,3), (3,0)}: 2.
    // Scores: idx0=idx1=100*1*101=10100; idx2=idx3=100*1*2=200.
    // argmin = idx 2 (first hit). Ratio 200/10100 ≈ 0.020 ≤ 0.5
    // → HeatAware returns Some(2).
    let snap = make_snapshot_with_heat(
        &[
            ("e1", 100, 0.0),
            ("e2", 100, 0.0),
            ("e3", 100, 0.0),
            ("e4", 100, 0.0),
        ],
        &[("e1", "e2", 0.01, vec![1], vec![0])],
    );
    let plan_heat = compile_with(CYCLE4_SRC, &snap, WcojVarOrderingKind::HeatAware);
    let vo_heat = first_var_order(&plan_heat).expect("HeatAware must set var_order");
    assert_eq!(vo_heat.leader_idx, 2);

    let plan_card = compile_with(CYCLE4_SRC, &snap, WcojVarOrderingKind::LeaderCardinality);
    assert!(first_var_order(&plan_card).is_none());
}
