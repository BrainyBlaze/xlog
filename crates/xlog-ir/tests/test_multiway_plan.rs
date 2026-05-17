use xlog_ir::rir::{
    ColumnSwap, CostPredictionRecord, HelperSplitSpec, KCliqueVariableOrder, MultiwayPlan,
    PlannedHashReason, SortedLayoutSpec, StreamGroupId, K_CLIQUE_MAX_EDGES, K_CLIQUE_MAX_K,
};

#[test]
fn multiway_plan_route_surface_covers_wcoj_and_planned_hash() {
    let kclique = KCliqueVariableOrder::new(
        5,
        [0, 1, 2, 3, 4, u8::MAX, u8::MAX, u8::MAX],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
        ],
        Vec::<ColumnSwap>::new(),
        SortedLayoutSpec {
            edge_slots: vec![0],
            key_columns: vec![vec![0, 1]],
        },
        Vec::<HelperSplitSpec>::new(),
        StreamGroupId(0),
    );
    assert_eq!(K_CLIQUE_MAX_K, 8);
    assert_eq!(K_CLIQUE_MAX_EDGES, 28);

    let wcoj = MultiwayPlan::WcojWithPlan(kclique.clone());
    let hash = MultiwayPlan::PlannedHashRoute {
        reason: PlannedHashReason::PlannerPredictsHashWins,
        planner_evidence: CostPredictionRecord {
            wcoj_cost: 12.0,
            hash_cost: 4.0,
        },
    };

    let covered = match (&wcoj, &hash) {
        (
            MultiwayPlan::WcojWithPlan(plan),
            MultiwayPlan::PlannedHashRoute {
                reason,
                planner_evidence,
            },
        ) => {
            plan == &kclique
                && *reason == PlannedHashReason::PlannerPredictsHashWins
                && planner_evidence.wcoj_cost > planner_evidence.hash_cost
        }
        _ => false,
    };

    assert!(covered, "route match must cover both MultiwayPlan variants");
}

#[test]
fn incomplete_stats_evidence_has_stable_safe_default_costs() {
    let evidence = CostPredictionRecord::empty();

    assert!(evidence.wcoj_cost.is_infinite());
    assert_eq!(evidence.hash_cost, 0.0);
}
