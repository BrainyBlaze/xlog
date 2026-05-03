//! v0.6.5 slice 1 step 4 — end-to-end Compiler-pipeline coverage
//! for `promote_multiway`.
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
// join shape than the v0.6.2 dispatch matcher recognizes.
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
        } => {
            assert_eq!(inputs.len(), 3);
            for (slot_idx, scan) in inputs.iter().enumerate() {
                match scan {
                    RirNode::Scan { .. } => {}
                    other => panic!(
                        "input slot {} must be a Scan, got {:?}",
                        slot_idx, other
                    ),
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
