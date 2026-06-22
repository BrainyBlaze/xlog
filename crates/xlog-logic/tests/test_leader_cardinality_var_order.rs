//! Leader-cardinality variable-ordering acceptance gate (10 tests).
//!
//! Compile-time leader-decision tests. Each test:
//! 1. Compiles a triangle/4-cycle source with a crafted
//!    `StatsSnapshot` via
//!    `Compiler::compile_with_config_and_stats_snapshot`.
//! 2. Asserts the resulting `RirNode::MultiWayJoin.var_order`
//!    matches the leader-cardinality permutation table.
//! 3. Asserts `MultiWayJoin.output_columns` is unchanged from the
//!    binary-fallback projection so existing binary-fallback
//!    consumers continue to read it directly.
//!
//! Coverage:
//!   * Triangle leader picks (3): e_xy / e_yz / e_xz.
//!   * 4-cycle leader picks (4): e_wx / e_xy / e_yz / e_zw.
//!   * Default-leader-already-min short-circuit (1): uniform
//!     stats → cost model returns None for both shapes.
//!     (Missing-stats `card_of` short-circuit is unit-tested at
//!     `xlog_logic::wcoj_var_ordering::tests::
//!     missing_stats_returns_none_safety_floor`.)
//!   * Activation contract (2): same fixture, default config →
//!     None; LeaderCardinality config → Some(...).

use xlog_core::RelId;
use xlog_ir::rir::{ProjectExpr, VariableOrder};
use xlog_ir::RirNode;
use xlog_logic::compile::Compiler;
use xlog_logic::compiler_config::{CompilerConfig, WcojVarOrderingKind};
use xlog_stats::{RelationStats, StatsSnapshot};

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

/// Build a `StatsSnapshot` keyed by predicate name + cardinality.
/// RelIds are assigned 0..n in input order; the compiler resolves
/// them via predicate name during the optimizer's stats merge.
fn make_snapshot(seeded: &[(&str, u64)]) -> StatsSnapshot {
    let relations: Vec<RelationStats> = seeded
        .iter()
        .enumerate()
        .map(|(i, (_, card))| {
            let mut s = RelationStats::new(RelId(i as u32));
            s.cardinality = *card;
            s
        })
        .collect();
    let rel_names: Vec<(RelId, String)> = seeded
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (RelId(i as u32), (*name).to_string()))
        .collect();
    StatsSnapshot {
        relations,
        join_selectivities: vec![],
        rel_names,
    }
}

/// Walk the post-compile plan and return the first
/// `RirNode::MultiWayJoin` (these fixtures emit exactly one per
/// rule). Returns the var_order + output_columns directly so each
/// test can assert on both.
fn first_multiway(plan: &xlog_ir::ExecutionPlan) -> (Option<VariableOrder>, Vec<ProjectExpr>) {
    fn find(node: &RirNode) -> Option<(Option<VariableOrder>, Vec<ProjectExpr>)> {
        match node {
            RirNode::MultiWayJoin {
                var_order,
                output_columns,
                ..
            } => Some((var_order.clone(), output_columns.clone())),
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
    for rules in &plan.rules_by_scc {
        for rule in rules {
            if let Some(found) = find(&rule.body) {
                return found;
            }
        }
    }
    panic!("plan has no MultiWayJoin");
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

fn compile_with_var_ordering_config(
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
// Triangle leader picks (3 tests)
// ===============================================================

#[test]
fn triangle_picks_e_xy_default_when_e_xy_smallest() {
    // e1 (e_xy) is smallest. Cost model returns None because the
    // default leader is already optimal — Var Order stays None,
    // bit-identical to the original binary-fallback path.
    let snap = make_snapshot(&[("e1", 100), ("e2", 1000), ("e3", 1000)]);
    let plan = compile_with_var_ordering_config(
        TRIANGLE_SRC,
        &snap,
        WcojVarOrderingKind::LeaderCardinality,
    );
    let (var_order, output_columns) = first_multiway(&plan);
    assert!(
        var_order.is_none(),
        "default leader case should leave var_order = None, got {:?}",
        var_order
    );
    // Original binary-fallback projection is unchanged.
    assert_eq!(
        output_columns,
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ],
        "MultiWayJoin.output_columns must remain the original binary-fallback projection"
    );
}

#[test]
fn triangle_picks_e_yz_when_e_yz_smallest() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000)]);
    let plan = compile_with_var_ordering_config(
        TRIANGLE_SRC,
        &snap,
        WcojVarOrderingKind::LeaderCardinality,
    );
    let (var_order, output_columns) = first_multiway(&plan);
    let vo = var_order.expect("triangle e_yz fixture must set var_order");
    assert_eq!(vo.leader_idx, 1, "e_yz leader is canonical idx 1");
    // Locked head_proj for e_yz leader = [Column(2), Column(0), Column(1)]
    assert_eq!(
        vo.kernel_output_cols,
        vec![
            ProjectExpr::Column(2),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ]
    );
    // Both lookup_perms swap_cols == true for e_yz leader.
    assert_eq!(vo.lookup_perms.len(), 2);
    assert!(vo.lookup_perms.iter().all(|p| p.swap_cols));
    // MultiWayJoin.output_columns stays as the original
    // binary-fallback projection; existing matchers continue reading it directly.
    assert_eq!(
        output_columns,
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ]
    );
}

#[test]
fn triangle_picks_e_xz_when_e_xz_smallest() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 50)]);
    let plan = compile_with_var_ordering_config(
        TRIANGLE_SRC,
        &snap,
        WcojVarOrderingKind::LeaderCardinality,
    );
    let (var_order, output_columns) = first_multiway(&plan);
    let vo = var_order.expect("triangle e_xz fixture must set var_order");
    assert_eq!(vo.leader_idx, 2, "e_xz leader is canonical idx 2");
    // Locked head_proj for e_xz leader = [Column(0), Column(2), Column(1)]
    assert_eq!(
        vo.kernel_output_cols,
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(2),
            ProjectExpr::Column(1),
        ]
    );
    // e_xz leader: slot 1 = e_yz swapped, slot 2 = e_xy as-is.
    assert_eq!(vo.lookup_perms.len(), 2);
    assert!(vo.lookup_perms[0].swap_cols);
    assert!(!vo.lookup_perms[1].swap_cols);
    assert_eq!(
        output_columns,
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ]
    );
}

// ===============================================================
// 4-cycle leader picks (4 tests, all rotation-only)
// ===============================================================

#[test]
fn cycle4_picks_e_wx_default_when_e_wx_smallest() {
    let snap = make_snapshot(&[("e1", 100), ("e2", 1000), ("e3", 1000), ("e4", 1000)]);
    let plan =
        compile_with_var_ordering_config(CYCLE4_SRC, &snap, WcojVarOrderingKind::LeaderCardinality);
    let (var_order, _) = first_multiway(&plan);
    assert!(
        var_order.is_none(),
        "default leader case should leave var_order = None"
    );
}

#[test]
fn cycle4_picks_e_xy_when_e_xy_smallest() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000), ("e4", 1000)]);
    let plan =
        compile_with_var_ordering_config(CYCLE4_SRC, &snap, WcojVarOrderingKind::LeaderCardinality);
    let (var_order, _) = first_multiway(&plan);
    let vo = var_order.expect("4-cycle e_xy fixture must set var_order");
    assert_eq!(vo.leader_idx, 1);
    // 4-cycle is rotation-only: no col-swaps.
    assert!(vo.lookup_perms.iter().all(|p| !p.swap_cols));
    // Locked head_proj for e_xy leader = [3, 0, 1, 2]
    assert_eq!(
        vo.kernel_output_cols,
        vec![
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ]
    );
}

#[test]
fn cycle4_picks_e_yz_when_e_yz_smallest() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 50), ("e4", 1000)]);
    let plan =
        compile_with_var_ordering_config(CYCLE4_SRC, &snap, WcojVarOrderingKind::LeaderCardinality);
    let (var_order, _) = first_multiway(&plan);
    let vo = var_order.expect("4-cycle e_yz fixture must set var_order");
    assert_eq!(vo.leader_idx, 2);
    assert!(vo.lookup_perms.iter().all(|p| !p.swap_cols));
    assert_eq!(
        vo.kernel_output_cols,
        vec![
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ]
    );
}

#[test]
fn cycle4_picks_e_zw_when_e_zw_smallest() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 1000), ("e4", 50)]);
    let plan =
        compile_with_var_ordering_config(CYCLE4_SRC, &snap, WcojVarOrderingKind::LeaderCardinality);
    let (var_order, _) = first_multiway(&plan);
    let vo = var_order.expect("4-cycle e_zw fixture must set var_order");
    assert_eq!(vo.leader_idx, 3);
    assert!(vo.lookup_perms.iter().all(|p| !p.swap_cols));
    assert_eq!(
        vo.kernel_output_cols,
        vec![
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
        ]
    );
}

// ===============================================================
// Default-leader-already-min short-circuit (1 test, single
// fixture covers both triangle + 4-cycle).
//
// NOTE: the original variable-ordering plan framed this test as a
// missing-stats safety floor. The actual missing-stats
// safety-floor semantics (`card_of` returning None on zero card)
// is exercised at the unit-test level by
// `wcoj_var_ordering::tests::missing_stats_returns_none_safety_floor`.
// At the compile-time / promoter level, the more reachable
// short-circuit is "default leader is already the min" — that's
// what this test pins.
// ===============================================================

#[test]
fn default_leader_already_min_returns_none_for_both_shapes() {
    // Every input has the SAME card. Then argmin is canonical
    // idx 0 (default leader; ties resolve to the first index),
    // which the cost model short-circuits to None ("no reorder
    // needed"). Bit-identical to the original binary-fallback path.
    let snap = make_snapshot(&[("e1", 100), ("e2", 100), ("e3", 100)]);
    let triangle_plan = compile_with_var_ordering_config(
        TRIANGLE_SRC,
        &snap,
        WcojVarOrderingKind::LeaderCardinality,
    );
    let (triangle_vo, _) = first_multiway(&triangle_plan);
    assert!(
        triangle_vo.is_none(),
        "uniform-stats triangle must produce var_order = None (default leader optimal)"
    );
    let snap4 = make_snapshot(&[("e1", 100), ("e2", 100), ("e3", 100), ("e4", 100)]);
    let cycle4_plan = compile_with_var_ordering_config(
        CYCLE4_SRC,
        &snap4,
        WcojVarOrderingKind::LeaderCardinality,
    );
    let (cycle4_vo, _) = first_multiway(&cycle4_plan);
    assert!(
        cycle4_vo.is_none(),
        "uniform-stats 4-cycle must produce var_order = None"
    );
}

// ===============================================================
// Activation contract (2 tests): same stats, default vs
// LeaderCardinality. The only difference between the two test cases is
// `WcojVarOrderingKind`.
// ===============================================================

#[test]
fn default_config_leaves_var_order_none_even_with_triggering_stats() {
    // Stats favor e_yz at ratio 50/1000 = 0.05 (well under 0.5
    // threshold). With Disabled config, the cost model
    // short-circuits and var_order stays None.
    let snap = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000)]);
    let plan = compile_with_var_ordering_config(TRIANGLE_SRC, &snap, WcojVarOrderingKind::Disabled);
    let (var_order, _) = first_multiway(&plan);
    assert!(
        var_order.is_none(),
        "Disabled config must leave var_order = None even with triggering stats"
    );
}

#[test]
fn leader_cardinality_config_sets_var_order_some_with_same_stats() {
    // Same stats as the previous test, only WcojVarOrderingKind
    // differs.
    let snap = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000)]);
    let plan = compile_with_var_ordering_config(
        TRIANGLE_SRC,
        &snap,
        WcojVarOrderingKind::LeaderCardinality,
    );
    let (var_order, _) = first_multiway(&plan);
    let vo = var_order.expect("LeaderCardinality must set var_order on triggering stats");
    assert_eq!(vo.leader_idx, 1, "stats favor e_yz (canonical idx 1)");
}
