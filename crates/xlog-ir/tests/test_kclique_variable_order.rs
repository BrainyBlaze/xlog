use xlog_core::RelId;
use xlog_ir::rir::{
    ColumnSwap, HelperSplitSpec, KCliqueVariableOrder, LookupPerm, ProjectExpr, RirNode,
    SortedLayoutSpec, StreamGroupId, VariableOrder, K_CLIQUE_MAX_EDGES, K_CLIQUE_MAX_K,
};

#[test]
fn kclique_variable_order_surface_has_required_fields() {
    let order = kclique_order(5);

    assert_eq!(order.k, 5);
    assert_eq!(order.variable_positions.len(), K_CLIQUE_MAX_K);
    assert_eq!(order.edge_permutation.len(), K_CLIQUE_MAX_EDGES);
    assert_eq!(
        order.column_swaps,
        vec![ColumnSwap {
            edge_slot: 3,
            swap_cols: true
        }]
    );
    assert_eq!(order.sorted_layout_requirements.edge_slots, vec![0, 1, 2]);
    assert_eq!(order.helper_split_specs[0].variable, 2);
    assert_eq!(order.stream_group, StreamGroupId(7));
}

#[test]
fn legacy_triangle_and_cycle_variable_order_shape_is_preserved() {
    let legacy = VariableOrder::legacy(
        2,
        vec![
            LookupPerm {
                input_idx: 1,
                swap_cols: true,
            },
            LookupPerm {
                input_idx: 0,
                swap_cols: false,
            },
        ],
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(2),
            ProjectExpr::Column(1),
        ],
    );

    assert_eq!(legacy.leader_idx, 2);
    assert_eq!(legacy.lookup_perms.len(), 2);
    assert_eq!(legacy.kernel_output_cols.len(), 3);
    assert!(legacy.kclique.is_none());
}

#[test]
fn k5_k6_orders_round_trip_through_multiway_var_order() {
    for k in [5, 6] {
        let plan = KCliqueVariableOrder::new(
            k,
            variable_positions(k),
            edge_permutation(k),
            vec![],
            SortedLayoutSpec {
                edge_slots: (0..edge_count(k)).collect(),
                key_columns: vec![vec![0, 1]; edge_count(k) as usize],
            },
            vec![],
            StreamGroupId(k),
        );
        let node = RirNode::MultiWayJoin {
            inputs: (0..edge_count(k))
                .map(|idx| RirNode::Scan {
                    rel: RelId(u32::from(idx) + 1),
                })
                .collect(),
            slot_vars: vec![vec![Some(0), Some(1)]; edge_count(k) as usize],
            output_columns: (0..k)
                .map(|idx| ProjectExpr::Column(idx as usize))
                .collect(),
            fallback: Box::new(RirNode::Unit),
            var_order: Some(VariableOrder::kclique(plan.clone())),
        };

        let RirNode::MultiWayJoin {
            var_order: Some(round_trip),
            ..
        } = node
        else {
            panic!("expected MultiWayJoin with var_order");
        };

        assert_eq!(round_trip.kclique, Some(plan));
    }
}

#[test]
fn kclique_variable_order_equality_semantics_are_structural() {
    let a = kclique_order(5);
    let same = kclique_order(5);
    let mut different_var = kclique_order(5);
    different_var.variable_positions[0] = 4;
    let mut different_edge = kclique_order(5);
    different_edge.edge_permutation[0] = 9;

    assert_eq!(a, same);
    assert_ne!(a, different_var);
    assert_ne!(a, different_edge);
    assert_ne!(
        VariableOrder::kclique(a),
        VariableOrder::legacy(0, vec![], vec![])
    );
}

fn kclique_order(k: u8) -> KCliqueVariableOrder {
    KCliqueVariableOrder::new(
        k,
        variable_positions(k),
        edge_permutation(k),
        vec![ColumnSwap {
            edge_slot: 3,
            swap_cols: true,
        }],
        SortedLayoutSpec {
            edge_slots: vec![0, 1, 2],
            key_columns: vec![vec![0, 1], vec![1, 0], vec![0, 1]],
        },
        vec![HelperSplitSpec {
            helper_id: 11,
            variable: 2,
            edge_slots: vec![1, 2],
        }],
        StreamGroupId(7),
    )
}

fn variable_positions(k: u8) -> [u8; K_CLIQUE_MAX_K] {
    let mut positions = [u8::MAX; K_CLIQUE_MAX_K];
    for idx in 0..k {
        positions[idx as usize] = idx;
    }
    positions
}

fn edge_permutation(k: u8) -> [u8; K_CLIQUE_MAX_EDGES] {
    let mut permutation = [u8::MAX; K_CLIQUE_MAX_EDGES];
    for idx in 0..edge_count(k) {
        permutation[idx as usize] = idx;
    }
    permutation
}

fn edge_count(k: u8) -> u8 {
    k * (k - 1) / 2
}
