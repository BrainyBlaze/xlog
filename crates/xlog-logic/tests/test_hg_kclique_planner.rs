use xlog_core::{RelId, ScalarType};
use xlog_logic::hypergraph::var_order::{
    plan_kclique_var_order, FullVariableOrder, KCliqueShape, PredictedWinner,
};
use xlog_logic::hypergraph::VertexId;
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

#[derive(Debug, Clone, Copy)]
struct FixtureProfile {
    rows: u64,
    ndv: u64,
    selectivity: f64,
    prefix_degree: f64,
    heat: f64,
}

#[derive(Debug, Clone, Copy)]
struct W52Cell {
    workload: &'static str,
    size: u32,
    path: &'static str,
    shape_kind: ShapeKind,
    expected: PredictedWinner,
    profile: FixtureProfile,
}

#[derive(Debug, Clone, Copy)]
enum ShapeKind {
    Cycle4,
    Clique5,
}

#[test]
fn plans_k5_and_k6_full_orders_with_complete_stats() {
    for k in [5, 6] {
        let shape = KCliqueShape::complete(k, RelId(10_000 + u32::from(k))).unwrap();
        let stats = complete_stats(&shape, dense_wcoj_profile(1_000 + u64::from(k)));

        let plan = plan_kclique_var_order(&shape, &stats).expect("complete stats should plan");

        assert_full_order(&plan, k);
        assert_eq!(plan.edge_permutation.len(), shape.edges().len());
        assert!(!plan.variable_share_allocation.is_empty());
    }
}

#[test]
fn planner_is_deterministic_for_100_repeated_calls() {
    let shape = KCliqueShape::complete(6, RelId(20_000)).unwrap();
    let stats = complete_stats(&shape, dense_wcoj_profile(2_000));
    let first = plan_kclique_var_order(&shape, &stats).expect("first plan");

    for _ in 0..100 {
        let next = plan_kclique_var_order(&shape, &stats).expect("repeat plan");
        assert_eq!(next, first);
    }
}

#[test]
fn incomplete_stats_return_none_for_four_cases() {
    let shape = KCliqueShape::complete(5, RelId(30_000)).unwrap();
    let complete = complete_stats(&shape, dense_wcoj_profile(1_500));

    let mut missing_relation = complete.clone();
    missing_relation.relations.remove(0);
    assert!(plan_kclique_var_order(&shape, &missing_relation).is_none());

    let mut missing_ndv = complete.clone();
    missing_ndv.relations[0].column_stats[0].distinct_estimate = 0;
    assert!(plan_kclique_var_order(&shape, &missing_ndv).is_none());

    let mut missing_prefix_degree = complete.clone();
    missing_prefix_degree.relations[0].prefix_degrees.clear();
    assert!(plan_kclique_var_order(&shape, &missing_prefix_degree).is_none());

    let mut missing_heat = complete;
    missing_heat.relations[0].key_heats.clear();
    assert!(plan_kclique_var_order(&shape, &missing_heat).is_none());
}

#[test]
fn k7_k8_extension_uses_template_path_only() {
    for k in [7, 8] {
        let shape = KCliqueShape::complete(k, RelId(40_000 + u32::from(k))).unwrap();
        let stats = complete_stats(&shape, dense_wcoj_profile(2_500 + u64::from(k)));
        let plan = plan_kclique_var_order(&shape, &stats).expect("template plan");
        assert_full_order(&plan, k);
    }
}

#[test]
fn w52_baseline_prediction_precision_is_36_of_36() {
    let mut correct = 0;
    let mut mismatches = Vec::new();
    let cells = w52_cells();

    for (idx, cell) in cells.iter().enumerate() {
        let label = format!("{}:{}:{}", cell.workload, cell.size, cell.path);
        let shape = match cell.shape_kind {
            ShapeKind::Cycle4 => KCliqueShape::cycle4(RelId(50_000 + idx as u32)).unwrap(),
            ShapeKind::Clique5 => KCliqueShape::complete(5, RelId(60_000 + idx as u32)).unwrap(),
        };
        let stats = complete_stats(&shape, cell.profile);
        let plan = plan_kclique_var_order(&shape, &stats)
            .unwrap_or_else(|| panic!("planner returned no plan for {label}"));

        if plan.predicted_winner == cell.expected {
            correct += 1;
        } else {
            mismatches.push(format!(
                "{label}: expected {:?}, got {:?}",
                cell.expected, plan.predicted_winner
            ));
        }
    }

    assert_eq!(cells.len(), 36);
    assert!(
        correct >= 33,
        "W5.2 prediction precision {correct}/36 below 33/36: {mismatches:?}"
    );
    assert_eq!(
        correct, 36,
        "expected exact fixture calibration: {mismatches:?}"
    );
}

#[test]
fn buried_inner_variable_skew_emits_helper_split_spec() {
    let shape = KCliqueShape::complete(5, RelId(70_000)).unwrap();
    let stats = complete_stats_with_variable_heat(&shape, dense_wcoj_profile(10_000), 3, 5.0);

    let plan = plan_kclique_var_order(&shape, &stats).expect("complete stats should plan");

    assert_ne!(
        plan.variable_order.first(),
        Some(&VertexId(3)),
        "hot variable must be buried behind the selected leader for this cert"
    );
    assert_eq!(
        plan.helper_split_specs.len(),
        1,
        "heat ratio >= 3x must emit one helper split spec"
    );
    let spec = &plan.helper_split_specs[0];
    assert_eq!(spec.helper_id, 0);
    assert_eq!(spec.variable, 3);
    assert_eq!(
        spec.edge_slots.len(),
        3,
        "K-clique helper split materializes a triangle around the buried variable"
    );
}

#[test]
fn uniform_heat_keeps_helper_split_specs_empty() {
    let shape = KCliqueShape::complete(5, RelId(80_000)).unwrap();
    let stats = complete_stats(&shape, dense_wcoj_profile(10_000));

    let plan = plan_kclique_var_order(&shape, &stats).expect("complete stats should plan");

    assert!(
        plan.helper_split_specs.is_empty(),
        "uniform heat must preserve the pre-G_HELP_KC empty helper spec behavior"
    );
}

fn complete_stats(shape: &KCliqueShape, profile: FixtureProfile) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();

    for edge in shape.edges() {
        let mut rel = RelationStats::new(edge.rel_id);
        rel.update_cardinality(profile.rows);

        let mut left_col = ColumnStats::new(edge.left_col, ScalarType::U32);
        left_col.update_distinct(profile.ndv);
        let mut right_col = ColumnStats::new(edge.right_col, ScalarType::U32);
        right_col.update_distinct(profile.ndv);
        rel.add_column(left_col);
        rel.add_column(right_col);

        rel.add_prefix_degree(PrefixDegreeStats::new(
            edge.left_col,
            profile.prefix_degree,
            profile.prefix_degree * 1.25,
        ));
        rel.add_prefix_degree(PrefixDegreeStats::new(
            edge.right_col,
            profile.prefix_degree,
            profile.prefix_degree * 1.25,
        ));
        rel.add_key_heat(KeyHeatStats::new(edge.left_col, profile.heat, profile.heat));
        rel.add_key_heat(KeyHeatStats::new(
            edge.right_col,
            profile.heat,
            profile.heat,
        ));

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

fn complete_stats_with_variable_heat(
    shape: &KCliqueShape,
    profile: FixtureProfile,
    hot_variable: usize,
    hot_heat: f64,
) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();

    for edge in shape.edges() {
        let mut rel = RelationStats::new(edge.rel_id);
        rel.update_cardinality(profile.rows);

        let mut left_col = ColumnStats::new(edge.left_col, ScalarType::U32);
        left_col.update_distinct(profile.ndv);
        let mut right_col = ColumnStats::new(edge.right_col, ScalarType::U32);
        right_col.update_distinct(profile.ndv);
        rel.add_column(left_col);
        rel.add_column(right_col);

        rel.add_prefix_degree(PrefixDegreeStats::new(
            edge.left_col,
            profile.prefix_degree,
            profile.prefix_degree * 1.25,
        ));
        rel.add_prefix_degree(PrefixDegreeStats::new(
            edge.right_col,
            profile.prefix_degree,
            profile.prefix_degree * 1.25,
        ));

        let left_heat = if edge.left.0 == hot_variable {
            hot_heat
        } else {
            profile.heat
        };
        let right_heat = if edge.right.0 == hot_variable {
            hot_heat
        } else {
            profile.heat
        };
        rel.add_key_heat(KeyHeatStats::new(edge.left_col, left_heat, left_heat));
        rel.add_key_heat(KeyHeatStats::new(edge.right_col, right_heat, right_heat));

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

fn assert_full_order(plan: &FullVariableOrder, k: u8) {
    assert_eq!(plan.variable_order.len(), usize::from(k));

    let mut sorted = plan.variable_order.clone();
    sorted.sort();
    sorted.dedup();

    assert_eq!(sorted.len(), usize::from(k));
    assert_eq!(sorted.first(), Some(&VertexId(0)));
    assert_eq!(sorted.last(), Some(&VertexId(usize::from(k - 1))));
}

fn dense_wcoj_profile(rows: u64) -> FixtureProfile {
    FixtureProfile {
        rows,
        ndv: rows / 2,
        selectivity: 0.001,
        prefix_degree: 2.0,
        heat: 0.75,
    }
}

fn hash_favorable_profile(rows: u64) -> FixtureProfile {
    FixtureProfile {
        rows,
        ndv: rows.saturating_mul(4),
        selectivity: 0.35,
        prefix_degree: 24.0,
        heat: 4.0,
    }
}

fn w52_cells() -> Vec<W52Cell> {
    const PATHS: [&str; 3] = ["run1", "run2", "run3"];
    const WORKLOADS: [(&str, [u32; 4], ShapeKind, PredictedWinner); 3] = [
        (
            "4cycle",
            [50, 250, 1000, 2000],
            ShapeKind::Cycle4,
            PredictedWinner::WcojPath,
        ),
        (
            "5clique",
            [10, 25, 50, 100],
            ShapeKind::Clique5,
            PredictedWinner::HashPath,
        ),
        (
            "pivot5",
            [10, 20, 30, 40],
            ShapeKind::Clique5,
            PredictedWinner::HashPath,
        ),
    ];
    let mut cells = Vec::new();

    for (workload, sizes, shape_kind, expected) in WORKLOADS {
        for size in sizes {
            let profile = match workload {
                "4cycle" => dense_wcoj_profile(u64::from(size) * 8),
                "5clique" => hash_favorable_profile(u64::from(size) * 64),
                "pivot5" => hash_favorable_profile(u64::from(size) * 96),
                _ => unreachable!("fixed W5.2 workload table"),
            };
            for path in PATHS {
                cells.push(W52Cell {
                    workload,
                    size,
                    path,
                    shape_kind,
                    expected,
                    profile,
                });
            }
        }
    }

    assert_eq!(cells.len(), 36, "W5.2 baseline evidence row count");
    cells
}
