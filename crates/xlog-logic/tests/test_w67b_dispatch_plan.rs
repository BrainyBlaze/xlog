use xlog_core::{RelId, ScalarType};
use xlog_ir::RirNode;
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

#[test]
fn promoter_attaches_kclique_var_order_for_k5_and_k6() {
    for (source, k) in [(CLIQUE5_SRC, 5u8), (CLIQUE6_SRC, 6u8)] {
        let mut compiler = Compiler::new();
        let snapshot = named_clique_stats(k);
        let plan = compiler
            .compile_with_stats_snapshot(source, Some(&snapshot))
            .expect("compile clique");
        let order = find_kclique_order(&plan).unwrap_or_else(|| {
            panic!("K{k} promotion must attach KCliqueVariableOrder, not var_order None")
        });

        assert_eq!(order.k, k);
        assert_eq!(live_prefix(&order.edge_permutation).len(), edge_count(k));
        assert!(
            !order.sorted_layout_requirements.edge_slots.is_empty(),
            "K{k} plan must carry sorted-layout requirements"
        );
    }
}

#[test]
fn promoter_source_calls_planner_and_has_no_kclique_var_order_none_path() {
    let source = include_str!("../src/promote.rs");
    let body = source
        .split("fn try_promote_clique_k")
        .nth(1)
        .expect("try_promote_clique_k present")
        .split("#[cfg(test)]")
        .next()
        .expect("function body before tests");

    assert!(body.contains("plan_kclique_var_order"));
    assert!(body.contains("VariableOrder::kclique"));
    assert!(
        !body.contains("var_order: None"),
        "K5/K6 promotion must not preserve the legacy no-plan path"
    );
}

fn find_kclique_order(
    plan: &xlog_ir::ExecutionPlan,
) -> Option<&xlog_ir::rir::KCliqueVariableOrder> {
    fn walk(node: &RirNode) -> Option<&xlog_ir::rir::KCliqueVariableOrder> {
        match node {
            RirNode::MultiWayJoin {
                var_order,
                fallback,
                ..
            } => var_order
                .as_ref()
                .and_then(|order| order.kclique.as_ref())
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

fn live_prefix(values: &[u8]) -> Vec<u8> {
    values
        .iter()
        .copied()
        .take_while(|value| *value != u8::MAX)
        .collect()
}

fn edge_count(k: u8) -> usize {
    usize::from(k) * usize::from(k - 1) / 2
}

fn named_clique_stats(k: u8) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    let mut edges = Vec::new();
    let mut rel_id = 1u32;

    for i in 0..k {
        for j in (i + 1)..k {
            let rel = RelId(rel_id);
            rel_id += 1;
            snapshot.rel_names.push((rel, format!("e{i}{j}")));
            edges.push((rel, i, j));

            let mut stats = RelationStats::new(rel);
            stats.update_cardinality(2_000 + u64::from(k));
            for col_idx in [0usize, 1usize] {
                let mut col = ColumnStats::new(col_idx, ScalarType::U32);
                col.update_distinct(1_000 + u64::from(k));
                stats.add_column(col);
                stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 2.0, 2.5));
                stats.add_key_heat(KeyHeatStats::new(col_idx, 0.75, 0.75));
            }
            snapshot.relations.push(stats);
        }
    }

    for (left_idx, (left_rel, left_i, left_j)) in edges.iter().enumerate() {
        for (right_rel, right_i, right_j) in edges.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let mut sel = JoinSelectivity::new(*left_rel, *right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}
