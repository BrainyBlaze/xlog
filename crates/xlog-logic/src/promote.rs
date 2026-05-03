//! v0.6.5 slice 1 — `MultiWayJoin` promotion pass.
//!
//! Walks an [`ExecutionPlan`] (post-lowering, post-optimizer) and
//! rewrites recognized triangle subtrees in `rule.body` to
//! [`RirNode::MultiWayJoin`]. Idempotent.
//!
//! ## Eligibility (slice 1)
//!
//! Exact-match against the canonical lowered-and-optimized triangle
//! shape — the same tree the v0.6.2 executor's `match_triangle_rir`
//! recognized:
//!
//! ```text
//! Project {
//!     input: Join {
//!         left: Join {
//!             left: Scan(rel_xy),
//!             right: Scan(rel_yz),
//!             left_keys: [1],
//!             right_keys: [0],
//!             join_type: Inner,
//!         },
//!         right: Scan(rel_xz),
//!         left_keys: [0, 3],
//!         right_keys: [0, 1],
//!         join_type: Inner,
//!     },
//!     columns: [Column(0), Column(1), Column(3)],
//! }
//! ```
//!
//! Anything else (different shape, predicate-pushdown-altered Join,
//! recursive SCC bodies, computed-projection variants) is left
//! untouched. The promoter does not introduce new eligibility; that
//! is later-slice work in v0.6.5.
//!
//! ## Fallback identity invariant
//!
//! The promoter captures the exact post-optimizer subtree as
//! `MultiWayJoin.fallback`. Executing `fallback` produces the same
//! row set as the pre-promotion tree — guaranteed by being the
//! identical [`RirNode`].
//!
//! ## Out of scope
//!
//! * Cost model, selectivity reordering, variable-ordering choices.
//! * Recursive SCC bodies. The promoter only walks the top-level
//!   `rule.body` of each `CompiledRule`; bodies inside recursive
//!   SCCs are still wrapped with `Fixpoint` and are not eligible.
//! * 4-way / general-arity admission.

use xlog_ir::rir::ProjectExpr;
use xlog_ir::{ExecutionPlan, JoinType, RirNode};

/// Walk an `ExecutionPlan` and rewrite eligible triangle subtrees
/// in each rule body to `RirNode::MultiWayJoin`. Idempotent.
///
/// `CompiledRule.meta` is preserved unchanged — the metadata is
/// rule-level (head schema, row estimates, layout hints), not
/// node-level, and the promoter does not alter rule semantics.
pub fn promote_multiway(plan: &mut ExecutionPlan) {
    for rules in &mut plan.rules_by_scc {
        for rule in rules.iter_mut() {
            if let Some(promoted) = try_promote_triangle(&rule.body) {
                rule.body = promoted;
            }
        }
    }
}

/// Attempt to recognize the canonical triangle and produce the
/// equivalent `MultiWayJoin`. Returns `None` for any deviation.
fn try_promote_triangle(node: &RirNode) -> Option<RirNode> {
    let RirNode::Project {
        input: outer_input,
        columns,
    } = node
    else {
        return None;
    };
    if columns.as_slice()
        != [
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ]
    {
        return None;
    }
    let RirNode::Join {
        left: l1,
        right: r1,
        left_keys: lk1,
        right_keys: rk1,
        join_type: jt1,
    } = outer_input.as_ref()
    else {
        return None;
    };
    if !matches!(jt1, JoinType::Inner) {
        return None;
    }
    if lk1.as_slice() != [0usize, 3] || rk1.as_slice() != [0usize, 1] {
        return None;
    }
    let RirNode::Scan { rel: rel_xz } = r1.as_ref() else {
        return None;
    };
    let RirNode::Join {
        left: l2,
        right: r2,
        left_keys: lk2,
        right_keys: rk2,
        join_type: jt2,
    } = l1.as_ref()
    else {
        return None;
    };
    if !matches!(jt2, JoinType::Inner) {
        return None;
    }
    if lk2.as_slice() != [1usize] || rk2.as_slice() != [0usize] {
        return None;
    }
    let RirNode::Scan { rel: rel_xy } = l2.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: rel_yz } = r2.as_ref() else {
        return None;
    };

    let inputs = vec![
        RirNode::Scan { rel: *rel_xy },
        RirNode::Scan { rel: *rel_yz },
        RirNode::Scan { rel: *rel_xz },
    ];
    // Canonical triangle slot_vars: [[V_X, V_Y], [V_Y, V_Z],
    // [V_X, V_Z]] with V_X=0, V_Y=1, V_Z=2.
    let slot_vars = vec![
        vec![Some(0u32), Some(1)],
        vec![Some(1u32), Some(2)],
        vec![Some(0u32), Some(2)],
    ];
    let output_columns = columns.clone();
    let fallback = Box::new(node.clone());
    Some(RirNode::MultiWayJoin {
        inputs,
        slot_vars,
        output_columns,
        fallback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::RelId;
    use xlog_ir::{CompiledRule, ExecutionPlan, PlanBuilder, Scc};

    fn canonical_triangle_tree() -> RirNode {
        let inner = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(RirNode::Scan { rel: RelId(3) }),
            left_keys: vec![0, 3],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
        }
    }

    fn plan_with_body(body: RirNode) -> ExecutionPlan {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["t".to_string()],
            is_recursive: false,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "t".to_string(),
                body,
                meta: Default::default(),
            },
        );
        builder.build()
    }

    #[test]
    fn promotes_canonical_triangle() {
        let mut plan = plan_with_body(canonical_triangle_tree());
        promote_multiway(&mut plan);
        let body = &plan.rules_by_scc[0][0].body;
        match body {
            RirNode::MultiWayJoin {
                inputs,
                slot_vars,
                output_columns,
                fallback,
            } => {
                assert_eq!(inputs.len(), 3);
                assert!(matches!(inputs[0], RirNode::Scan { rel: RelId(1) }));
                assert!(matches!(inputs[1], RirNode::Scan { rel: RelId(2) }));
                assert!(matches!(inputs[2], RirNode::Scan { rel: RelId(3) }));
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
                // Fallback is the exact pre-promotion tree.
                assert!(matches!(fallback.as_ref(), RirNode::Project { .. }));
            }
            other => panic!("expected MultiWayJoin, got {:?}", other),
        }
    }

    #[test]
    fn fallback_is_structurally_equal_to_input() {
        let pre = canonical_triangle_tree();
        let mut plan = plan_with_body(pre.clone());
        promote_multiway(&mut plan);
        let body = &plan.rules_by_scc[0][0].body;
        let RirNode::MultiWayJoin { fallback, .. } = body else {
            panic!("expected MultiWayJoin");
        };
        // Use Debug equality as a structural-equality proxy. RirNode
        // doesn't impl PartialEq directly because Expr doesn't; the
        // Debug output is deterministic and structurally faithful.
        assert_eq!(format!("{:?}", fallback.as_ref()), format!("{:?}", pre));
    }

    #[test]
    fn idempotent_under_repeat_calls() {
        let mut plan = plan_with_body(canonical_triangle_tree());
        promote_multiway(&mut plan);
        let first = format!("{:?}", &plan.rules_by_scc[0][0].body);
        promote_multiway(&mut plan);
        let second = format!("{:?}", &plan.rules_by_scc[0][0].body);
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_non_triangle_projection_columns() {
        // Output columns rotated: [Column(1), Column(0), Column(3)].
        let inner = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(RirNode::Scan { rel: RelId(3) }),
            left_keys: vec![0, 3],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let body = RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(1),
                ProjectExpr::Column(0),
                ProjectExpr::Column(3),
            ],
        };
        let mut plan = plan_with_body(body);
        promote_multiway(&mut plan);
        // Body untouched.
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::Project { .. }
        ));
    }

    #[test]
    fn rejects_non_inner_join() {
        let inner = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::LeftOuter,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(RirNode::Scan { rel: RelId(3) }),
            left_keys: vec![0, 3],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let body = RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
        };
        let mut plan = plan_with_body(body);
        promote_multiway(&mut plan);
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::Project { .. }
        ));
    }

    #[test]
    fn rejects_filter_above_outer_join() {
        // An optimizer may insert a Filter between the outer Project
        // and the outer Join. v1 promoter does not recognize this.
        let inner = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(RirNode::Scan { rel: RelId(3) }),
            left_keys: vec![0, 3],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let filtered = RirNode::Filter {
            input: Box::new(outer),
            predicate: xlog_ir::Expr::Column(0),
        };
        let body = RirNode::Project {
            input: Box::new(filtered),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
        };
        let mut plan = plan_with_body(body);
        promote_multiway(&mut plan);
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::Project { .. }
        ));
    }

    #[test]
    fn meta_preserved_byte_for_byte() {
        use xlog_core::Schema;
        use xlog_ir::metadata::RirMeta;

        let schema = Schema::new(vec![
            ("x".to_string(), xlog_core::ScalarType::U32),
            ("y".to_string(), xlog_core::ScalarType::U32),
            ("z".to_string(), xlog_core::ScalarType::U32),
        ]);
        let meta_pre = RirMeta::with_schema(schema).with_rows(100, 250);

        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["t".to_string()],
            is_recursive: false,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "t".to_string(),
                body: canonical_triangle_tree(),
                meta: meta_pre.clone(),
            },
        );
        let mut plan = builder.build();
        promote_multiway(&mut plan);
        assert_eq!(
            format!("{:?}", &plan.rules_by_scc[0][0].meta),
            format!("{:?}", meta_pre),
        );
    }
}
