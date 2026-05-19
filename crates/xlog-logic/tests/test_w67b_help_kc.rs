use std::collections::BTreeMap;

use xlog_core::{RelId, ScalarType};
use xlog_ir::rir::{MultiwayPlan, RirNode};
use xlog_logic::Compiler;
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

const K5_SRC: &str = r#"
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

const K5_EDGES: [(&str, usize, usize); 10] = [
    ("e01", 0, 1),
    ("e02", 0, 2),
    ("e03", 0, 3),
    ("e04", 0, 4),
    ("e12", 1, 2),
    ("e13", 1, 3),
    ("e14", 1, 4),
    ("e23", 2, 3),
    ("e24", 2, 4),
    ("e34", 3, 4),
];

#[test]
fn buried_skew_k5_compile_creates_helper_relation_and_outer_kclique_uses_it() {
    let rel_ids = rel_ids_for_k5();
    let snapshot = k5_stats(&rel_ids, Some((3, 5.0)));
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_with_stats_snapshot(K5_SRC, Some(&snapshot))
        .expect("compile buried-skew K5");
    let helper = compiler
        .rel_ids()
        .iter()
        .find_map(|(name, rel)| name.starts_with("__w37_helper_").then_some((name, *rel)))
        .expect("G_HELP_KC must allocate one helper relation");

    assert_eq!(
        plan.rules_by_scc
            .iter()
            .flatten()
            .filter(|rule| rule.head == *helper.0)
            .count(),
        1,
        "helper rule must be emitted before the outer K-clique rule"
    );

    let outer = plan
        .rules_by_scc
        .iter()
        .flatten()
        .find(|rule| rule.head == "clique5")
        .expect("outer clique5 rule");
    let kclique = find_kclique_multiway(&outer.body).expect("outer clique5 must stay K-clique");
    assert!(
        kclique
            .inputs
            .iter()
            .any(|input| matches!(input, RirNode::Scan { rel } if *rel == helper.1)),
        "outer K-clique inputs must use the emitted helper relation"
    );
    let specs = kclique
        .var_order
        .as_ref()
        .and_then(|order| order.kclique.as_ref())
        .map(|order| &order.helper_split_specs)
        .expect("outer K-clique must carry KCliqueVariableOrder");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].variable, 3);
    assert!(matches!(kclique.plan, Some(MultiwayPlan::WcojWithPlan(_))));
}

#[test]
fn uniform_k5_compile_keeps_helper_split_empty_and_allocates_no_helper() {
    let rel_ids = rel_ids_for_k5();
    let snapshot = k5_stats(&rel_ids, None);
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile_with_stats_snapshot(K5_SRC, Some(&snapshot))
        .expect("compile uniform K5");

    assert!(
        !compiler
            .rel_ids()
            .keys()
            .any(|name| name.starts_with("__w37_helper_")),
        "uniform K5 must not allocate helper relations"
    );
    let outer = plan
        .rules_by_scc
        .iter()
        .flatten()
        .find(|rule| rule.head == "clique5")
        .expect("outer clique5 rule");
    let kclique = find_kclique_multiway(&outer.body).expect("uniform K5 must still promote");
    let specs = kclique
        .var_order
        .as_ref()
        .and_then(|order| order.kclique.as_ref())
        .map(|order| &order.helper_split_specs)
        .expect("outer K-clique must carry KCliqueVariableOrder");
    assert!(specs.is_empty());
}

#[test]
fn help_kc_source_contract_invokes_helper_split_pass() {
    let promote = include_str!("../src/promote.rs");
    let compile = include_str!("../src/compile.rs");
    let optimizer = include_str!("../src/optimizer.rs");
    let required = "Paper §5 Figure 3: Helper-relation splitting elevates buried inner-variable skew per Authorization 5 (2026-05-17)";

    assert!(promote.contains(required));
    assert!(
        !promote.contains("Vec::<HelperSplitSpec>::new()"),
        "K-clique promoter must not always emit an empty helper split vec"
    );
    assert!(
        compile.contains("helper_split_pass::run_kclique_specs"),
        "compile pipeline must invoke the Phase-1 G4 helper_split_pass for K-clique specs"
    );
    assert!(
        optimizer.contains("pub fn run_kclique_specs"),
        "Phase-1 G4 helper_split_pass must expose a K-clique spec entry"
    );
}

struct KcliqueNode<'a> {
    inputs: &'a [RirNode],
    plan: &'a Option<MultiwayPlan>,
    var_order: &'a Option<xlog_ir::rir::VariableOrder>,
}

fn find_kclique_multiway(node: &RirNode) -> Option<KcliqueNode<'_>> {
    match node {
        RirNode::MultiWayJoin {
            inputs,
            plan,
            var_order,
            ..
        } if inputs.len() == 10 => Some(KcliqueNode {
            inputs,
            plan,
            var_order,
        }),
        RirNode::Project { input, .. }
        | RirNode::Filter { input, .. }
        | RirNode::Distinct { input, .. }
        | RirNode::GroupBy { input, .. } => find_kclique_multiway(input),
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            find_kclique_multiway(left).or_else(|| find_kclique_multiway(right))
        }
        RirNode::Union { inputs } => inputs.iter().find_map(find_kclique_multiway),
        _ => None,
    }
}

fn rel_ids_for_k5() -> BTreeMap<String, RelId> {
    let mut compiler = Compiler::new();
    let _ = compiler.compile(K5_SRC).expect("compile ids");
    compiler
        .rel_ids()
        .iter()
        .map(|(name, rel)| (name.clone(), *rel))
        .collect()
}

fn k5_stats(rel_ids: &BTreeMap<String, RelId>, hot: Option<(usize, f64)>) -> StatsSnapshot {
    let mut snapshot = StatsSnapshot::default();
    for (name, left, right) in K5_EDGES {
        let rel = *rel_ids.get(name).expect("edge rel id");
        snapshot.rel_names.push((rel, name.to_string()));
        let mut stats = RelationStats::new(rel);
        stats.update_cardinality(10_000);
        for (col_idx, variable) in [(0usize, left), (1usize, right)] {
            let mut col = ColumnStats::new(col_idx, ScalarType::U32);
            col.update_distinct(10_000);
            stats.add_column(col);
            stats.add_prefix_degree(PrefixDegreeStats::new(col_idx, 1.0, 1.25));
            let heat = match hot {
                Some((hot_var, hot_heat)) if hot_var == variable => hot_heat,
                _ => 0.25,
            };
            stats.add_key_heat(KeyHeatStats::new(col_idx, heat, heat));
        }
        snapshot.relations.push(stats);
    }

    for (left_idx, (left_name, left_i, left_j)) in K5_EDGES.iter().enumerate() {
        let left_rel = *rel_ids.get(*left_name).expect("left rel id");
        for (right_name, right_i, right_j) in K5_EDGES.iter().skip(left_idx + 1) {
            if left_i == right_i || left_i == right_j || left_j == right_i || left_j == right_j {
                let right_rel = *rel_ids.get(*right_name).expect("right rel id");
                let mut sel = JoinSelectivity::new(left_rel, right_rel);
                sel.set_keys(vec![0], vec![0]);
                sel.set_selectivity(0.001);
                snapshot.join_selectivities.push(sel);
            }
        }
    }

    snapshot
}
