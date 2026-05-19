use std::collections::BTreeMap;

use xlog_core::{RelId, ScalarType};
use xlog_ir::rir::MultiwayPlan;
use xlog_ir::{EirEpistemicMode, ExecutionPlan, RirNode};
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, compile_epistemic_gpu_execution_with_stats_snapshot,
};
use xlog_logic::{parse_program, Compiler};
use xlog_stats::{
    ColumnStats, JoinSelectivity, KeyHeatStats, PrefixDegreeStats, RelationStats, StatsSnapshot,
};

#[test]
fn epistemic_executable_plan_lowers_reduced_program_through_runtime_plan() {
    let program = parse_program(
        r#"
        node(1).
        edge(1).
        accepted(X) :- node(X), know edge(X).
        "#,
    )
    .unwrap();

    let executable = compile_epistemic_gpu_execution(&program).unwrap();

    assert!(executable.gpu_plan.cpu_fallbacks.is_zero());
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 1);
    assert_eq!(compiled_rule_count(&executable.reduced_runtime_plan), 3);
}

#[test]
fn wcoj_eligible_epistemic_reduction_reaches_multiway_runtime_plan() {
    let program = parse_program(
        r#"
        edge(1, 2).
        edge(2, 3).
        edge(1, 3).
        choice().

        accepted(X, Y, Z) :-
            edge(X, Y),
            edge(Y, Z),
            edge(X, Z),
            possible choice().
        "#,
    )
    .unwrap();

    let executable = compile_epistemic_gpu_execution(&program).unwrap();

    assert!(executable
        .gpu_plan
        .reductions
        .iter()
        .any(|reduction| reduction.relational_body_atoms == 3));
    assert!(plan_contains_multiway_join(
        &executable.reduced_runtime_plan
    ));
}

#[test]
fn epistemic_kclique_reduction_reuses_38b_planner_layout_and_helper_split_surface() {
    let program = parse_program(EPISTEMIC_K5_SRC).unwrap();
    let rel_ids = rel_ids_for_reduced_k5();
    let snapshot = k5_stats(&rel_ids, Some((3, 5.0)));

    let executable =
        compile_epistemic_gpu_execution_with_stats_snapshot(&program, Some(&snapshot)).unwrap();
    let kclique = find_kclique_multiway(&executable.reduced_runtime_plan)
        .expect("epistemic K5 reduction must reach production K-clique MultiWayJoin");
    let order = kclique
        .var_order
        .as_ref()
        .and_then(|order| order.kclique.as_ref())
        .expect("epistemic K5 reduction must reuse KCliqueVariableOrder");

    assert!(matches!(kclique.plan, Some(MultiwayPlan::WcojWithPlan(_))));
    assert_eq!(order.k, 5);
    assert!(
        !order.sorted_layout_requirements.edge_slots.is_empty(),
        "K-clique epistemic reduction must carry sorted-layout requirements"
    );
    assert_eq!(
        order.helper_split_specs.len(),
        1,
        "buried-skew epistemic reduction must reuse helper-splitting specs"
    );
}

#[test]
fn faeel_gpu_execution_rejects_self_supported_possible_before_runtime_dispatch() {
    let program = parse_program(
        r#"
        pred p().
        p() :- possible p().
        "#,
    )
    .unwrap();

    let err = compile_epistemic_gpu_execution(&program)
        .expect_err("default FAEEL lowering must reject self-supported possible");

    match err {
        xlog_core::XlogError::UnsupportedEpistemicConstruct { construct, context } => {
            assert_eq!(construct, "FAEEL foundedness guard");
            assert!(context.contains("rule[0]"));
            assert!(context.contains("possible p/0"));
            assert!(context.contains("self-supported"));
        }
        other => panic!("expected FAEEL foundedness rejection, got {other:?}"),
    }
}

#[test]
fn faeel_gpu_execution_allows_self_possible_with_independent_founded_support() {
    let program = parse_program(
        r#"
        pred seed().
        pred p().
        seed().
        p() :- seed().
        p() :- possible p().
        "#,
    )
    .unwrap();

    let executable = compile_epistemic_gpu_execution(&program)
        .expect("independently founded FAEEL support should permit self possible");

    assert_eq!(executable.gpu_plan.mode, EirEpistemicMode::Faeel);
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 1);
}

#[test]
fn g91_gpu_execution_allows_self_supported_possible_compatibility_fixture() {
    let program = parse_program(
        r#"
        #pragma epistemic_mode = g91
        pred p().
        p() :- possible p().
        "#,
    )
    .unwrap();

    let executable = compile_epistemic_gpu_execution(&program)
        .expect("G91 compatibility mode permits self-supported possible fixtures");

    assert_eq!(executable.gpu_plan.mode, EirEpistemicMode::G91);
    assert_eq!(executable.gpu_plan.epistemic_literals.len(), 1);
    assert_eq!(compiled_rule_count(&executable.reduced_runtime_plan), 1);
}

const EPISTEMIC_K5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred gate().
    pred clique5(u32, u32, u32, u32, u32).
    gate().
    clique5(V0, V1, V2, V3, V4) :-
        e01(V0, V1), e02(V0, V2), e03(V0, V3), e04(V0, V4),
        e12(V1, V2), e13(V1, V3), e14(V1, V4),
        e23(V2, V3), e24(V2, V4),
        e34(V3, V4),
        know gate().
"#;

const REDUCED_K5_SRC: &str = r#"
    pred e01(u32, u32). pred e02(u32, u32). pred e03(u32, u32). pred e04(u32, u32).
    pred e12(u32, u32). pred e13(u32, u32). pred e14(u32, u32).
    pred e23(u32, u32). pred e24(u32, u32).
    pred e34(u32, u32).
    pred gate().
    pred clique5(u32, u32, u32, u32, u32).
    gate().
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

fn compiled_rule_count(plan: &ExecutionPlan) -> usize {
    plan.rules_by_scc.iter().map(Vec::len).sum()
}

fn plan_contains_multiway_join(plan: &ExecutionPlan) -> bool {
    plan.rules_by_scc
        .iter()
        .flatten()
        .any(|rule| node_contains_multiway_join(&rule.body))
}

fn node_contains_multiway_join(node: &RirNode) -> bool {
    match node {
        RirNode::MultiWayJoin { .. } => true,
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::Distinct { input, .. }
        | RirNode::GroupBy { input, .. } => node_contains_multiway_join(input),
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            node_contains_multiway_join(left) || node_contains_multiway_join(right)
        }
        RirNode::Union { inputs } => inputs.iter().any(node_contains_multiway_join),
        RirNode::Fixpoint {
            base, recursive, ..
        } => node_contains_multiway_join(base) || node_contains_multiway_join(recursive),
        RirNode::ChainJoin { fallback, .. } => node_contains_multiway_join(fallback),
        RirNode::TensorMaskedJoin { .. } | RirNode::Scan { .. } | RirNode::Unit => false,
    }
}

struct KcliqueNode<'a> {
    plan: &'a Option<MultiwayPlan>,
    var_order: &'a Option<xlog_ir::rir::VariableOrder>,
}

fn find_kclique_multiway(plan: &ExecutionPlan) -> Option<KcliqueNode<'_>> {
    fn walk(node: &RirNode) -> Option<KcliqueNode<'_>> {
        match node {
            RirNode::MultiWayJoin {
                inputs,
                plan,
                var_order,
                ..
            } if inputs.len() == 10 => Some(KcliqueNode { plan, var_order }),
            RirNode::Filter { input, .. }
            | RirNode::Project { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => walk(input),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                walk(left).or_else(|| walk(right))
            }
            RirNode::Union { inputs } => inputs.iter().find_map(walk),
            RirNode::Fixpoint {
                base, recursive, ..
            } => walk(base).or_else(|| walk(recursive)),
            RirNode::ChainJoin { fallback, .. } => walk(fallback),
            RirNode::TensorMaskedJoin { .. } | RirNode::Scan { .. } | RirNode::Unit => None,
            RirNode::MultiWayJoin { fallback, .. } => walk(fallback),
        }
    }

    plan.rules_by_scc
        .iter()
        .flatten()
        .find_map(|rule| walk(&rule.body))
}

fn rel_ids_for_reduced_k5() -> BTreeMap<String, RelId> {
    let mut compiler = Compiler::new();
    let _ = compiler
        .compile(REDUCED_K5_SRC)
        .expect("compile reduced K5");
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
