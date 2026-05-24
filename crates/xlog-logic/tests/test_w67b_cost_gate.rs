use xlog_core::{RelId, ScalarType};
use xlog_ir::rir::{MultiwayPlan, PlannedHashReason};
use xlog_ir::RirNode;
use xlog_logic::hypergraph::var_order::{plan_kclique_var_order, KCliqueShape, PredictedWinner};
use xlog_logic::Compiler;
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

const CLIQUE5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred clique5(u32, u32, u32, u32, u32).
    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4).
"#;

const CLIQUE6_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32).
    pred e04(u32, u32). pred e05(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32). pred e15(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32). pred e25(u32, u32).
    pred e34(u32, u32). pred e35(u32, u32).
    pred e45(u32, u32).
    pred clique6(u32, u32, u32, u32, u32, u32).
    clique6(V0, V1, V2, V3, V4, V5) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4), e05(V0, V5),
        e12(V1, V2), e13(V1, V3), e14(V1, V4), e15(V1, V5),
        e23(V2, V3), e24(V2, V4), e25(V2, V5),
        e34(V3, V4), e35(V3, V5),
        e45(V4, V5).
"#;

#[derive(Clone, Copy)]
struct Profile {
    rows: u64,
    ndv: u64,
    selectivity: f64,
    prefix_degree: f64,
    heat: f64,
}

#[test]
fn cost_gate_routes_complete_hash_favorable_k5_to_structured_hash() {
    let plan = compile_with_stats(CLIQUE5_SRC, 5, hash_favorable_profile(6_400));
    let (route, var_order) = find_kclique_route(&plan).expect("K5 route");

    match route {
        MultiwayPlan::PlannedHashRoute {
            reason,
            planner_evidence,
        } => {
            assert_eq!(*reason, PlannedHashReason::PlannerPredictsHashWins);
            assert!(planner_evidence.wcoj_cost > planner_evidence.hash_cost);
        }
        other => panic!("expected structured hash route, got {other:?}"),
    }
    assert!(
        var_order.is_none(),
        "planned hash route must not carry a WCOJ var_order"
    );
}

#[test]
fn cost_gate_routes_complete_wcoj_favorable_k5_and_k6_to_wcoj_plan() {
    for (source, k) in [(CLIQUE5_SRC, 5u8), (CLIQUE6_SRC, 6u8)] {
        let plan = compile_with_stats(source, k, dense_wcoj_profile(2_000 + u64::from(k)));
        let (route, var_order) = find_kclique_route(&plan).expect("K-clique route");

        match route {
            MultiwayPlan::WcojWithPlan(order) => assert_eq!(order.k, k),
            other => panic!("expected WCOJ route for K{k}, got {other:?}"),
        }
        assert_eq!(
            var_order
                .and_then(|order| order.kclique.as_ref())
                .map(|order| order.k),
            Some(k)
        );
    }
}

#[test]
fn cost_gate_routes_incomplete_stats_to_structured_hash_default() {
    let mut compiler = Compiler::new();
    let plan = compiler.compile(CLIQUE5_SRC).expect("compile K5");
    let (route, var_order) = find_kclique_route(&plan).expect("K5 route");

    match route {
        MultiwayPlan::PlannedHashRoute {
            reason,
            planner_evidence,
        } => {
            assert_eq!(*reason, PlannedHashReason::IncompleteStatsSafeDefault);
            assert!(planner_evidence.wcoj_cost.is_infinite());
            assert_eq!(planner_evidence.hash_cost, 0.0);
        }
        other => panic!("expected incomplete-stats planned hash route, got {other:?}"),
    }
    assert!(var_order.is_none());
}

#[test]
fn w52_routing_decision_cert_is_36_of_36() {
    const WORKLOADS: [(&str, [u32; 4], PredictedWinner); 3] = [
        ("4cycle", [50, 250, 1000, 2000], PredictedWinner::WcojPath),
        ("5clique", [10, 25, 50, 100], PredictedWinner::HashPath),
        ("pivot5", [10, 20, 30, 40], PredictedWinner::HashPath),
    ];

    let mut correct = 0usize;
    let mut seen = 0usize;

    for (workload, sizes, expected) in WORKLOADS {
        for size in sizes {
            let (shape, profile) = match workload {
                "4cycle" => (
                    KCliqueShape::cycle4(RelId(10_000 + size)).unwrap(),
                    dense_wcoj_profile(u64::from(size) * 8),
                ),
                "5clique" => (
                    KCliqueShape::complete(5, RelId(20_000 + size)).unwrap(),
                    hash_favorable_profile(u64::from(size) * 64),
                ),
                "pivot5" => (
                    KCliqueShape::complete(5, RelId(30_000 + size)).unwrap(),
                    hash_favorable_profile(u64::from(size) * 96),
                ),
                _ => continue,
            };
            let stats = complete_shape_stats(&shape, profile);
            let predicted = plan_kclique_var_order(&shape, &stats)
                .expect("complete W5.2 stats")
                .predicted_winner;

            for _ in 0..3 {
                seen += 1;
                if predicted == expected {
                    correct += 1;
                }
            }
        }
    }

    assert_eq!(seen, 36);
    assert_eq!(correct, 36, "W5.2 routing cert must be exact");
}

#[test]
fn dilp_and_hub_skew_fixtures_keep_expected_routes() {
    let fixtures = [
        (CLIQUE5_SRC, 5u8, dense_wcoj_profile(1_250), true),
        (CLIQUE5_SRC, 5u8, hash_favorable_profile(1_600), false),
        (CLIQUE6_SRC, 6u8, dense_wcoj_profile(1_800), true),
        (CLIQUE5_SRC, 5u8, hub_skew_wcoj_profile(2_000), true),
    ];

    let mut correct = 0usize;
    for (source, k, profile, expect_wcoj) in fixtures {
        let plan = compile_with_stats(source, k, profile);
        let (route, _) = find_kclique_route(&plan).expect("fixture route");
        let got_wcoj = matches!(route, MultiwayPlan::WcojWithPlan(_));
        if got_wcoj == expect_wcoj {
            correct += 1;
        }
    }

    assert_eq!(correct, fixtures.len());
}

fn compile_with_stats(source: &str, k: u8, profile: Profile) -> xlog_ir::ExecutionPlan {
    let snapshot = named_clique_stats(k, profile);
    let mut compiler = Compiler::new();
    compiler
        .compile_with_stats_snapshot(source, Some(&snapshot))
        .expect("compile with stats")
}

fn find_kclique_route(
    plan: &xlog_ir::ExecutionPlan,
) -> Option<(&MultiwayPlan, Option<&xlog_ir::rir::VariableOrder>)> {
    fn walk(node: &RirNode) -> Option<(&MultiwayPlan, Option<&xlog_ir::rir::VariableOrder>)> {
        match node {
            RirNode::MultiWayJoin {
                plan,
                var_order,
                fallback,
                ..
            } => plan
                .as_ref()
                .map(|route| (route, var_order.as_ref()))
                .or_else(|| walk(fallback)),
            RirNode::Project { input, .. }
            | RirNode::Filter { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => walk(input),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                walk(left).or_else(|| walk(right))
            }
            RirNode::Union { inputs } => inputs.iter().find_map(walk),
            RirNode::Fixpoint {
                base, recursive, ..
            } => walk(base).or_else(|| walk(recursive)),
            _ => None,
        }
    }

    plan.rules_by_scc
        .iter()
        .flat_map(|rules| rules.iter())
        .find_map(|rule| walk(&rule.body))
}

fn named_clique_stats(k: u8, profile: Profile) -> StatsSnapshot {
    let mut shape_edges = Vec::new();
    let mut snapshot = StatsSnapshot::default();
    let mut rel_id = 1u32;

    for i in 0..k {
        for j in (i + 1)..k {
            let rel = RelId(rel_id);
            rel_id += 1;
            let name = format!("e{i}{j}");
            snapshot.rel_names.push((rel, name));
            shape_edges.push((rel, i, j));

            let mut stats = RelationStats::new(rel);
            stats.update_cardinality(profile.rows);
            for col_idx in [0usize, 1usize] {
                let mut col = ColumnStats::new(col_idx, ScalarType::U32);
                col.update_distinct(profile.ndv);
                stats.add_column(col);
                stats.add_prefix_degree(PrefixDegreeStats::new(
                    col_idx,
                    profile.prefix_degree,
                    profile.prefix_degree * 1.25,
                ));
                stats.add_key_heat(KeyHeatStats::new(col_idx, profile.heat, profile.heat));
            }
            snapshot.relations.push(stats);
        }
    }

    for (left_idx, (left_rel, left_i, left_j)) in shape_edges.iter().enumerate() {
        for (right_rel, right_i, right_j) in shape_edges.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let mut sel = JoinSelectivity::new(*left_rel, *right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(profile.selectivity);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}

fn complete_shape_stats(shape: &KCliqueShape, profile: Profile) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();

    for edge in shape.edges() {
        let mut rel = RelationStats::new(edge.rel_id);
        rel.update_cardinality(profile.rows);
        for col_idx in [edge.left_col, edge.right_col] {
            let mut col = ColumnStats::new(col_idx, ScalarType::U32);
            col.update_distinct(profile.ndv);
            rel.add_column(col);
            rel.add_prefix_degree(PrefixDegreeStats::new(
                col_idx,
                profile.prefix_degree,
                profile.prefix_degree * 1.25,
            ));
            rel.add_key_heat(KeyHeatStats::new(col_idx, profile.heat, profile.heat));
        }
        snapshot.relations.push(rel);
    }

    for (left_idx, left_edge) in shape.edges().iter().enumerate() {
        for right_edge in shape.edges().iter().skip(left_idx + 1) {
            if left_edge.touches(right_edge) {
                let mut sel = JoinSelectivity::new(left_edge.rel_id, right_edge.rel_id);
                sel.set_keys(vec![left_edge.left_col], vec![right_edge.left_col]);
                sel.set_selectivity(profile.selectivity);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}

fn dense_wcoj_profile(rows: u64) -> Profile {
    Profile {
        rows,
        ndv: rows / 2,
        selectivity: 0.001,
        prefix_degree: 2.0,
        heat: 0.75,
    }
}

fn hash_favorable_profile(rows: u64) -> Profile {
    Profile {
        rows,
        ndv: rows.saturating_mul(4),
        selectivity: 0.35,
        prefix_degree: 24.0,
        heat: 4.0,
    }
}

fn hub_skew_wcoj_profile(rows: u64) -> Profile {
    Profile {
        rows,
        ndv: rows,
        selectivity: 0.002,
        prefix_degree: 1.25,
        heat: 0.25,
    }
}
