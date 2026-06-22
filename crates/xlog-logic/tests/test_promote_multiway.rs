//! End-to-end compiler-pipeline coverage for `promote_multiway`.
//!
//! The unit tests in `xlog-logic::promote::tests` cover the pass on
//! synthesized RIR. This file exercises the full
//! parse → lower → optimize → promote chain through `Compiler`,
//! pinning the contract that a triangle Datalog program lands as a
//! `MultiWayJoin` after compilation.

use xlog_ir::{ProjectExpr, RirNode};
use xlog_logic::Compiler;

// Source-only triangle (no facts), mirroring
// `tests/test_wcoj_rir_shape_cert.rs`. Facts in the source can
// trigger additional rule-graph nodes that perturb the optimizer's
// estimates, leading the bushy planner to choose a different
// join shape than the dedicated multiway dispatch matcher recognizes.
const TRIANGLE_PROGRAM: &str = "triangle(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).";

#[test]
fn compile_triangle_program_produces_multiway_body() {
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile(TRIANGLE_PROGRAM)
        .expect("compile must succeed");

    let triangle_rule = plan
        .rules_by_scc
        .iter()
        .flatten()
        .find(|r| r.head == "triangle")
        .expect("triangle rule must be present");

    match &triangle_rule.body {
        RirNode::MultiWayJoin {
            inputs,
            slot_vars,
            output_columns,
            fallback,
            var_order: _,
            ..
        } => {
            assert_eq!(inputs.len(), 3);
            for (slot_idx, scan) in inputs.iter().enumerate() {
                match scan {
                    RirNode::Scan { .. } => {}
                    other => panic!("input slot {} must be a Scan, got {:?}", slot_idx, other),
                }
            }
            assert_eq!(
                slot_vars,
                &vec![
                    vec![Some(0u32), Some(1)],
                    vec![Some(1u32), Some(2)],
                    vec![Some(0u32), Some(2)],
                ]
            );
            assert_eq!(
                output_columns,
                &vec![
                    ProjectExpr::Column(0),
                    ProjectExpr::Column(1),
                    ProjectExpr::Column(3),
                ]
            );
            // Fallback is the post-optimizer Project { Join { Join, Scan } }.
            assert!(matches!(fallback.as_ref(), RirNode::Project { .. }));
        }
        other => panic!("expected MultiWayJoin, got {:?}", other),
    }
}

#[test]
fn compile_non_triangle_program_does_not_promote() {
    // A binary-join rule is not eligible.
    let src = r#"
edge(1, 2).
reach(X, Y) :- edge(X, Y).
"#;
    let mut compiler = Compiler::new();
    let plan = compiler.compile(src).expect("compile must succeed");
    let reach = plan
        .rules_by_scc
        .iter()
        .flatten()
        .find(|r| r.head == "reach")
        .expect("reach rule must be present");
    // Body must not be a MultiWayJoin.
    assert!(
        !matches!(&reach.body, RirNode::MultiWayJoin { .. }),
        "binary-join rule must not be promoted"
    );
}

// 4-cycle compiler-pipeline coverage.

const FOUR_CYCLE_PROGRAM: &str = "cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).";

#[test]
fn compile_4cycle_program_produces_multiway_body() {
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile(FOUR_CYCLE_PROGRAM)
        .expect("compile must succeed");

    let cycle_rule = plan
        .rules_by_scc
        .iter()
        .flatten()
        .find(|r| r.head == "cycle4")
        .expect("cycle4 rule must be present");

    match &cycle_rule.body {
        RirNode::MultiWayJoin {
            inputs,
            slot_vars,
            output_columns,
            fallback,
            var_order: _,
            ..
        } => {
            assert_eq!(inputs.len(), 4);
            for (slot_idx, scan) in inputs.iter().enumerate() {
                match scan {
                    RirNode::Scan { .. } => {}
                    other => panic!("input slot {} must be a Scan, got {:?}", slot_idx, other),
                }
            }
            assert_eq!(
                slot_vars,
                &vec![
                    vec![Some(0u32), Some(1)],
                    vec![Some(1u32), Some(2)],
                    vec![Some(2u32), Some(3)],
                    vec![Some(3u32), Some(0)],
                ]
            );
            assert_eq!(
                output_columns,
                &vec![
                    ProjectExpr::Column(0),
                    ProjectExpr::Column(1),
                    ProjectExpr::Column(3),
                    ProjectExpr::Column(5),
                ]
            );
            // Fallback is the post-optimizer Project { Join { Join, Join } }
            // (bushy 4-cycle shape, distinct from triangle's left-deep).
            assert!(matches!(fallback.as_ref(), RirNode::Project { .. }));
        }
        other => panic!("expected MultiWayJoin, got {:?}", other),
    }
}

// General Free Join multiway compiler-pipeline coverage. A 4-atom
// chain has no dedicated kernel shape; the general promoter
// must land it as a generic MultiWayJoin (plan: None) carrying dense
// first-occurrence variable classes in slot_vars.
const CHAIN4_PROGRAM: &str = "q(A, B) :- r(A, X), s(X, Y), t(Y, Z), u(Z, B).";

#[test]
fn compile_general_chain_program_produces_generic_multiway_body() {
    let mut compiler = Compiler::new();
    let plan = compiler
        .compile(CHAIN4_PROGRAM)
        .expect("compile must succeed");

    let rule = plan
        .rules_by_scc
        .iter()
        .flatten()
        .find(|r| r.head == "q")
        .expect("q rule must be present");

    match &rule.body {
        RirNode::MultiWayJoin {
            inputs,
            slot_vars,
            output_columns,
            fallback,
            plan,
            var_order,
        } => {
            assert_eq!(inputs.len(), 4);
            for (slot_idx, scan) in inputs.iter().enumerate() {
                match scan {
                    RirNode::Scan { .. } => {}
                    other => panic!("input slot {} must be a Scan, got {:?}", slot_idx, other),
                }
            }
            // Vars in first-occurrence order: A=0, X=1, Y=2, Z=3, B=4.
            assert_eq!(
                slot_vars,
                &vec![
                    vec![Some(0u32), Some(1)],
                    vec![Some(1u32), Some(2)],
                    vec![Some(2u32), Some(3)],
                    vec![Some(3u32), Some(4)],
                ]
            );
            assert_eq!(
                output_columns,
                &vec![ProjectExpr::Column(0), ProjectExpr::Column(7)]
            );
            assert_eq!(
                plan,
                &Some(xlog_ir::rir::MultiwayPlan::FreeJoin),
                "general promotion must carry the FreeJoin provenance marker"
            );
            assert!(
                var_order.is_none(),
                "general promotion carries no var order"
            );
            assert!(matches!(fallback.as_ref(), RirNode::Project { .. }));
        }
        other => panic!("expected generic MultiWayJoin, got {:?}", other),
    }
}
