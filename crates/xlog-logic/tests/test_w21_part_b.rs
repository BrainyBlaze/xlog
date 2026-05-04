//! W2.1 step 7 — Part B acceptance gate (7 tests).
//!
//! Runtime-routing contract tests: each test compiles a fixture
//! whose stats favor a specific leader, then asserts the resulting
//! `MultiWayJoin.var_order` matches the **locked permutation table**
//! the dispatcher relies on:
//!   * `var_order.leader_idx` equals the requested leader.
//!   * `var_order.lookup_perms[i].input_idx` is the canonical
//!     promoter index for slot `i+1` per the plan's table.
//!   * `var_order.lookup_perms[i].swap_cols` matches the table
//!     (triangle e_yz / e_xz need swaps; 4-cycle is rotation-only).
//!   * `var_order.kernel_output_cols == head_proj` from the table.
//!   * `MultiWayJoin.output_columns` is **unchanged** from the
//!     binary-fallback projection — slice 1/2/W2.2 binary-fallback
//!     consumers continue reading it directly.
//!
//! Per-slot **schema** and **content** assertions specified in the
//! W2.1 plan §"Part B" are deferred to Part C (end-to-end row-set
//! parity), which exercises the full GPU pipeline. The IR-level
//! contract here is what the dispatcher's
//! `prepare_kernel_inputs` rotation logic depends on, and Part B
//! pins it deterministically.

use xlog_core::RelId;
use xlog_ir::rir::{ProjectExpr, VariableOrder};
use xlog_ir::RirNode;
use xlog_logic::compile::Compiler;
use xlog_logic::compiler_config::{CompilerConfig, WcojVarOrderingKind};
use xlog_stats::{RelationStats, StatsSnapshot};

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

fn first_multiway(
    plan: &xlog_ir::ExecutionPlan,
) -> (VariableOrder, Vec<ProjectExpr>) {
    fn find(node: &RirNode) -> Option<(VariableOrder, Vec<ProjectExpr>)> {
        match node {
            RirNode::MultiWayJoin {
                var_order,
                output_columns,
                ..
            } => var_order.as_ref().map(|vo| (vo.clone(), output_columns.clone())),
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
    panic!("plan has no MultiWayJoin with var_order = Some");
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

fn compile_w21(
    source: &str,
    snapshot: &StatsSnapshot,
) -> xlog_ir::ExecutionPlan {
    let mut compiler = Compiler::new();
    let config = CompilerConfig {
        wcoj_variable_ordering: WcojVarOrderingKind::LeaderCardinality,
        ..CompilerConfig::default()
    };
    compiler
        .compile_with_config_and_stats_snapshot(source, &config, Some(snapshot))
        .expect("compile_with_config_and_stats_snapshot")
}

const CANONICAL_TRIANGLE_FALLBACK: [ProjectExpr; 3] = [
    ProjectExpr::Column(0),
    ProjectExpr::Column(1),
    ProjectExpr::Column(3),
];

const CANONICAL_4CYCLE_FALLBACK: [ProjectExpr; 4] = [
    ProjectExpr::Column(0),
    ProjectExpr::Column(1),
    ProjectExpr::Column(3),
    ProjectExpr::Column(5),
];

// ===============================================================
// Triangle dispatch routing per leader (3 tests)
// ===============================================================

#[test]
fn dispatch_routing_triangle_e_yz_leader() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000)]);
    let plan = compile_w21(TRIANGLE_SRC, &snap);
    let (vo, output_columns) = first_multiway(&plan);
    // Locked table for triangle e_yz leader (canonical idx 1):
    //   * lookup_perms[0] = (input_idx 2 = e_xz, swap = true)
    //   * lookup_perms[1] = (input_idx 0 = e_xy, swap = true)
    //   * kernel_output_cols = [Column(2), Column(0), Column(1)]
    assert_eq!(vo.leader_idx, 1);
    assert_eq!(vo.lookup_perms.len(), 2);
    assert_eq!(vo.lookup_perms[0].input_idx, 2);
    assert!(vo.lookup_perms[0].swap_cols);
    assert_eq!(vo.lookup_perms[1].input_idx, 0);
    assert!(vo.lookup_perms[1].swap_cols);
    assert_eq!(
        vo.kernel_output_cols,
        vec![
            ProjectExpr::Column(2),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ]
    );
    assert_eq!(output_columns, CANONICAL_TRIANGLE_FALLBACK.to_vec());
}

#[test]
fn dispatch_routing_triangle_e_xz_leader() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 50)]);
    let plan = compile_w21(TRIANGLE_SRC, &snap);
    let (vo, output_columns) = first_multiway(&plan);
    // Locked table for triangle e_xz leader (canonical idx 2):
    //   * lookup_perms[0] = (input_idx 1 = e_yz, swap = true)
    //   * lookup_perms[1] = (input_idx 0 = e_xy, swap = false)
    //   * kernel_output_cols = [Column(0), Column(2), Column(1)]
    assert_eq!(vo.leader_idx, 2);
    assert_eq!(vo.lookup_perms.len(), 2);
    assert_eq!(vo.lookup_perms[0].input_idx, 1);
    assert!(vo.lookup_perms[0].swap_cols);
    assert_eq!(vo.lookup_perms[1].input_idx, 0);
    assert!(!vo.lookup_perms[1].swap_cols);
    assert_eq!(
        vo.kernel_output_cols,
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(2),
            ProjectExpr::Column(1),
        ]
    );
    assert_eq!(output_columns, CANONICAL_TRIANGLE_FALLBACK.to_vec());
}

#[test]
fn dispatch_routing_triangle_lookup_perms_omit_leader() {
    // Across the 2 non-default triangle leaders, lookup_perms.len()
    // must equal 2 (slots 1 and 2 only — leader's slot 0 lives in
    // var_order.leader_idx and is never duplicated). Pin this via
    // the e_xy default-leader case (no var_order; that's a Part A
    // assertion) and the e_yz / e_xz cases.
    let fixtures: [&[(&str, u64)]; 2] = [
        // e_yz fixture
        &[("e1", 1000u64), ("e2", 50), ("e3", 1000)],
        // e_xz fixture
        &[("e1", 1000u64), ("e2", 1000), ("e3", 50)],
    ];
    for snap_seed in fixtures.iter() {
        let snap = make_snapshot(snap_seed);
        let plan = compile_w21(TRIANGLE_SRC, &snap);
        let (vo, _) = first_multiway(&plan);
        assert_eq!(
            vo.lookup_perms.len(),
            2,
            "triangle leader_idx={} must produce 2 lookup_perms, got {}",
            vo.leader_idx,
            vo.lookup_perms.len()
        );
        // No lookup_perm references the leader's own input_idx
        // (leader sits in var_order.leader_idx, never repeated).
        for lp in &vo.lookup_perms {
            assert_ne!(
                lp.input_idx, vo.leader_idx,
                "lookup_perms must not repeat the leader's input_idx"
            );
        }
    }
}

// ===============================================================
// 4-cycle dispatch routing per leader (4 tests, all rotation-only)
// ===============================================================

#[test]
fn dispatch_routing_cycle4_e_xy_leader() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 50), ("e3", 1000), ("e4", 1000)]);
    let plan = compile_w21(CYCLE4_SRC, &snap);
    let (vo, output_columns) = first_multiway(&plan);
    // Locked table for 4-cycle e_xy leader (canonical idx 1):
    // rotation-only: slots 1, 2, 3 = inputs 2, 3, 0.
    assert_eq!(vo.leader_idx, 1);
    assert_eq!(
        vo.lookup_perms.iter().map(|p| p.input_idx).collect::<Vec<_>>(),
        vec![2, 3, 0]
    );
    assert!(vo.lookup_perms.iter().all(|p| !p.swap_cols));
    assert_eq!(
        vo.kernel_output_cols,
        vec![
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ]
    );
    assert_eq!(output_columns, CANONICAL_4CYCLE_FALLBACK.to_vec());
}

#[test]
fn dispatch_routing_cycle4_e_yz_leader() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 50), ("e4", 1000)]);
    let plan = compile_w21(CYCLE4_SRC, &snap);
    let (vo, output_columns) = first_multiway(&plan);
    // Locked table for 4-cycle e_yz leader (canonical idx 2):
    // rotation-only: slots 1, 2, 3 = inputs 3, 0, 1.
    assert_eq!(vo.leader_idx, 2);
    assert_eq!(
        vo.lookup_perms.iter().map(|p| p.input_idx).collect::<Vec<_>>(),
        vec![3, 0, 1]
    );
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
    assert_eq!(output_columns, CANONICAL_4CYCLE_FALLBACK.to_vec());
}

#[test]
fn dispatch_routing_cycle4_e_zw_leader() {
    let snap = make_snapshot(&[("e1", 1000), ("e2", 1000), ("e3", 1000), ("e4", 50)]);
    let plan = compile_w21(CYCLE4_SRC, &snap);
    let (vo, output_columns) = first_multiway(&plan);
    // Locked table for 4-cycle e_zw leader (canonical idx 3):
    // rotation-only: slots 1, 2, 3 = inputs 0, 1, 2.
    assert_eq!(vo.leader_idx, 3);
    assert_eq!(
        vo.lookup_perms.iter().map(|p| p.input_idx).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
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
    assert_eq!(output_columns, CANONICAL_4CYCLE_FALLBACK.to_vec());
}

#[test]
fn dispatch_routing_cycle4_lookup_perms_omit_leader_and_no_swap() {
    // Cross-leader contract: 4-cycle is rotation-only (no swap)
    // AND lookup_perms never repeats the leader's own input_idx.
    // 3 non-default 4-cycle leaders.
    let fixtures: [(u8, &[(&str, u64)]); 3] = [
        (1u8, &[("e1", 1000u64), ("e2", 50), ("e3", 1000), ("e4", 1000)]),
        (2, &[("e1", 1000u64), ("e2", 1000), ("e3", 50), ("e4", 1000)]),
        (3, &[("e1", 1000u64), ("e2", 1000), ("e3", 1000), ("e4", 50)]),
    ];
    for (leader_idx, seed) in fixtures.iter() {
        let snap = make_snapshot(seed);
        let plan = compile_w21(CYCLE4_SRC, &snap);
        let (vo, _) = first_multiway(&plan);
        assert_eq!(vo.leader_idx, *leader_idx);
        assert_eq!(
            vo.lookup_perms.len(),
            3,
            "4-cycle non-default leader must produce 3 lookup_perms"
        );
        // None of the lookup_perms references the leader's own
        // canonical idx, AND none has swap_cols == true.
        for lp in &vo.lookup_perms {
            assert_ne!(lp.input_idx, *leader_idx);
            assert!(
                !lp.swap_cols,
                "4-cycle leader_idx={} must not request col-swaps (rotation-only)",
                leader_idx
            );
        }
    }
}
