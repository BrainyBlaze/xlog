//! v0.6.5 slice 1 — `MultiWayJoin` promotion pass.
//! v0.6.5 slice 4 — recursive-SCC promotion gated on linear recursion.
//!
//! Walks an [`ExecutionPlan`] (post-lowering, post-optimizer) and
//! rewrites recognized triangle / 4-cycle subtrees in `rule.body` to
//! [`RirNode::MultiWayJoin`]. Idempotent.
//!
//! ## Eligibility
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
//! 4-cycle has the analogous canonical lowered shape (slice 2). Any
//! deviation in shape, predicate-pushdown-altered Join, or
//! computed-projection variants is left untouched.
//!
//! ## Recursive SCC handling (slice 4)
//!
//! The promoter no longer blanket-skips recursive SCCs. Instead it
//! gates per-rule on the number of body Scans whose RelId resolves
//! to a predicate inside the rule's head SCC:
//!
//! | Recursive Scans in body | Slice 4 behavior          |
//! |-------------------------|---------------------------|
//! | 0 (stable rule)         | Promote                   |
//! | 1 (linear recursion)    | Promote                   |
//! | ≥ 2 (multi-recursion)   | Skip — defer to slice 4.2 |
//!
//! The recursive engine (`Executor::execute_recursive_scc`) consumes
//! the resulting `MultiWayJoin` via
//! `execute_wcoj_or_fallback_node`, dispatching WCOJ kernels on the
//! seeding pass and on each variant evaluation. Multi-recursive
//! bodies stay binary-join because the per-variant union+dedup
//! interaction with WCOJ is out of slice 4 scope.
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
//! * Multi-recursive (≥ 2) WCOJ — slice 4.2 / v0.6.6.
//! * 4-way / general-arity admission — slice 5.

use std::collections::{HashMap, HashSet};
use xlog_core::RelId;
use xlog_ir::plan::Scc;
use xlog_ir::rir::ProjectExpr;
use xlog_ir::{ExecutionPlan, JoinType, RirNode};

/// Walk an `ExecutionPlan` and rewrite eligible triangle / 4-cycle
/// subtrees in each rule body to `RirNode::MultiWayJoin`. Idempotent.
///
/// `CompiledRule.meta` is preserved unchanged — the metadata is
/// rule-level (head schema, row estimates, layout hints), not
/// node-level, and the promoter does not alter rule semantics.
///
/// `rel_ids` is the canonical predicate-name → RelId map used to
/// resolve body Scans against the head SCC's predicate set. Pass
/// `Compiler::rel_ids()` (or `Lowerer::rel_ids()`) at the call site.
///
/// **Recursive SCC bodies (slice 4).** A rule whose body contains
/// at most one Scan in its head SCC's predicate set is promoted —
/// stable rules (count 0) and linear-recursive rules (count 1).
/// Bodies with ≥ 2 recursive Scans are left as binary-join trees;
/// see slice 4.2 for multi-recursive WCOJ.
pub fn promote_multiway(plan: &mut ExecutionPlan, rel_ids: &HashMap<String, RelId>) {
    // Snapshot SCCs by index so we can pass &Scc into helpers
    // while holding `&mut plan.rules_by_scc`.
    let sccs_snapshot: Vec<Scc> = plan.sccs.clone();
    for (scc_id, rules) in plan.rules_by_scc.iter_mut().enumerate() {
        let head_scc = match sccs_snapshot.get(scc_id) {
            Some(scc) => scc,
            None => continue,
        };
        let head_rel_set = build_head_rel_set(head_scc, rel_ids);
        for rule in rules.iter_mut() {
            // Slice 4 gate: skip multi-recursive bodies. Stable
            // (count 0) and linear-recursive (count 1) bodies fall
            // through to the existing shape-match dispatch.
            if recursive_scan_count(&rule.body, &head_rel_set) > 1 {
                continue;
            }
            // Try triangle first, then 4-cycle. A body cannot match
            // both (different atom counts), so order is a doc anchor,
            // not a correctness gate. Future shapes append to this
            // chain in their own slice.
            if let Some(promoted) = try_promote_triangle(&rule.body) {
                rule.body = promoted;
                continue;
            }
            if let Some(promoted) = try_promote_4cycle(&rule.body) {
                rule.body = promoted;
            }
        }
    }
}

/// Resolve the head SCC's predicates to the set of RelIds that
/// would count as "recursive" Scans inside its rule bodies. Returns
/// an empty set when the SCC's predicates aren't in `rel_ids` (e.g.
/// in synthetic test plans without a real lowerer); empty set means
/// every Scan resolves to non-recursive, which is the correct
/// default for slice 1–3 byte-preservation.
fn build_head_rel_set(scc: &Scc, rel_ids: &HashMap<String, RelId>) -> HashSet<RelId> {
    scc.predicates
        .iter()
        .filter_map(|p| rel_ids.get(p).copied())
        .collect()
}

/// Count Scans in `body` whose RelId is in `head_rel_set`. Walks
/// every RIR variant including `MultiWayJoin` inputs and fallback;
/// idempotent on already-promoted bodies.
fn recursive_scan_count(body: &RirNode, head_rel_set: &HashSet<RelId>) -> usize {
    match body {
        RirNode::Unit => 0,
        RirNode::Scan { rel } => usize::from(head_rel_set.contains(rel)),
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::GroupBy { input, .. }
        | RirNode::Distinct { input, .. } => recursive_scan_count(input, head_rel_set),
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            recursive_scan_count(left, head_rel_set) + recursive_scan_count(right, head_rel_set)
        }
        RirNode::Union { inputs } => inputs
            .iter()
            .map(|n| recursive_scan_count(n, head_rel_set))
            .sum(),
        RirNode::Fixpoint {
            base, recursive, ..
        } => {
            recursive_scan_count(base, head_rel_set) + recursive_scan_count(recursive, head_rel_set)
        }
        RirNode::TensorMaskedJoin { rel_index, .. } => rel_index
            .iter()
            .filter(|(rid, _)| head_rel_set.contains(rid))
            .count(),
        // For an already-promoted body, count from `inputs` —
        // matches the `collect_scan_rels` invariant. The fallback
        // subtree references the same RelId set by promoter
        // construction, so counting both would double-count.
        RirNode::MultiWayJoin { inputs, .. } => inputs
            .iter()
            .map(|n| recursive_scan_count(n, head_rel_set))
            .sum(),
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

/// v0.6.5 slice 2 — recognize the canonical 4-cycle subtree and
/// produce the equivalent `MultiWayJoin`.
///
/// Target rule:
///
/// ```text
/// cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).
/// ```
///
/// Lowered + optimized shape (verified against
/// `Compiler::compile()`): bushy `Project { Join { Join, Join } }`
/// with `output_columns = [Column(0), Column(1), Column(3), Column(5)]`,
/// outer Join keys `[0, 3] / [3, 0]`, inner Join keys `[1] / [0]`.
/// This differs from triangle's left-deep shape; both promoters
/// coexist by matching their respective canonical trees.
///
/// Returns `None` for any deviation. Strict by design — slice 2's
/// walker contract states only matchers/promoters with explicit
/// shape qualifiers may shape-lock; this matcher locks 4-cycle
/// specifically.
fn try_promote_4cycle(node: &RirNode) -> Option<RirNode> {
    // Outer Project with the canonical (W, X, Y, Z) head columns.
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
            ProjectExpr::Column(5),
        ]
    {
        return None;
    }

    // Outer Join: bushy (left=Join, right=Join), keys [0,3]/[3,0].
    let RirNode::Join {
        left: outer_l,
        right: outer_r,
        left_keys: olk,
        right_keys: ork,
        join_type: ojt,
    } = outer_input.as_ref()
    else {
        return None;
    };
    if !matches!(ojt, JoinType::Inner) {
        return None;
    }
    if olk.as_slice() != [0usize, 3] || ork.as_slice() != [3usize, 0] {
        return None;
    }

    // Outer-left Join: e1(W,X) ⋈ e2(X,Y) on X. keys [1]/[0].
    let RirNode::Join {
        left: ll,
        right: lr,
        left_keys: ilk_l,
        right_keys: irk_l,
        join_type: ijt_l,
    } = outer_l.as_ref()
    else {
        return None;
    };
    if !matches!(ijt_l, JoinType::Inner) {
        return None;
    }
    if ilk_l.as_slice() != [1usize] || irk_l.as_slice() != [0usize] {
        return None;
    }
    let RirNode::Scan { rel: rel_e1 } = ll.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: rel_e2 } = lr.as_ref() else {
        return None;
    };

    // Outer-right Join: e3(Y,Z) ⋈ e4(Z,W) on Z. keys [1]/[0].
    let RirNode::Join {
        left: rl,
        right: rr,
        left_keys: ilk_r,
        right_keys: irk_r,
        join_type: ijt_r,
    } = outer_r.as_ref()
    else {
        return None;
    };
    if !matches!(ijt_r, JoinType::Inner) {
        return None;
    }
    if ilk_r.as_slice() != [1usize] || irk_r.as_slice() != [0usize] {
        return None;
    }
    let RirNode::Scan { rel: rel_e3 } = rl.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: rel_e4 } = rr.as_ref() else {
        return None;
    };

    let inputs = vec![
        RirNode::Scan { rel: *rel_e1 },
        RirNode::Scan { rel: *rel_e2 },
        RirNode::Scan { rel: *rel_e3 },
        RirNode::Scan { rel: *rel_e4 },
    ];
    // Canonical 4-cycle slot_vars: [[V_W, V_X], [V_X, V_Y],
    // [V_Y, V_Z], [V_Z, V_W]] with V_W=0, V_X=1, V_Y=2, V_Z=3.
    let slot_vars = vec![
        vec![Some(0u32), Some(1)],
        vec![Some(1u32), Some(2)],
        vec![Some(2u32), Some(3)],
        vec![Some(3u32), Some(0)],
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
        promote_multiway(&mut plan, &HashMap::new());
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
        promote_multiway(&mut plan, &HashMap::new());
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
        promote_multiway(&mut plan, &HashMap::new());
        let first = format!("{:?}", &plan.rules_by_scc[0][0].body);
        promote_multiway(&mut plan, &HashMap::new());
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
        promote_multiway(&mut plan, &HashMap::new());
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
        promote_multiway(&mut plan, &HashMap::new());
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
        promote_multiway(&mut plan, &HashMap::new());
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
        promote_multiway(&mut plan, &HashMap::new());
        assert_eq!(
            format!("{:?}", &plan.rules_by_scc[0][0].meta),
            format!("{:?}", meta_pre),
        );
    }

    // -----------------------------------------------------------
    // v0.6.5 slice 4 — recursive-SCC promotion gates
    // -----------------------------------------------------------

    /// Slice 4 contract: a stable triangle (zero recursive Scans) in
    /// a recursive SCC IS promoted — the recursive engine's seeding
    /// pass dispatches WCOJ via `execute_wcoj_or_fallback_node`.
    /// Body Scans are RelId(1)/(2)/(3) and the SCC's predicate "tri"
    /// is not in `rel_ids`, so the count is 0.
    #[test]
    fn promotes_stable_triangle_in_recursive_scc() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["tri".to_string()],
            is_recursive: true,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "tri".to_string(),
                body: canonical_triangle_tree(),
                meta: Default::default(),
            },
        );
        let mut plan = builder.build();
        // No "tri" entry in rel_ids → all body Scans are extensional
        // from this SCC's POV → count == 0 → promote.
        promote_multiway(&mut plan, &HashMap::new());
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Slice 4 contract: a linear-recursive triangle (exactly one
    /// in-SCC Scan) IS promoted. Build a triangle whose RelId(2)
    /// corresponds to the head SCC's predicate "tri" and assert
    /// promotion despite `is_recursive: true`.
    #[test]
    fn promotes_linear_recursive_triangle() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["tri".to_string()],
            is_recursive: true,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "tri".to_string(),
                body: canonical_triangle_tree(),
                meta: Default::default(),
            },
        );
        let mut plan = builder.build();
        let mut rel_ids = HashMap::new();
        rel_ids.insert("tri".to_string(), RelId(2)); // 1 of 3 Scans
        promote_multiway(&mut plan, &rel_ids);
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Slice 4 contract: ≥ 2 recursive Scans → NOT promoted. Body
    /// stays as the original `Project { Join { ... } }` binary-join
    /// tree. Mark "tri" → RelId(1) and "tri" predicate set covers
    /// RelId(1)+(2) by aliasing — but easier: add a second SCC
    /// member predicate that maps to RelId(2) so two of the three
    /// body Scans count as in-SCC.
    #[test]
    fn skips_multirec_triangle_in_recursive_scc() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["tri_a".to_string(), "tri_b".to_string()],
            is_recursive: true,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "tri_a".to_string(),
                body: canonical_triangle_tree(),
                meta: Default::default(),
            },
        );
        let mut plan = builder.build();
        let mut rel_ids = HashMap::new();
        rel_ids.insert("tri_a".to_string(), RelId(1));
        rel_ids.insert("tri_b".to_string(), RelId(2));
        // Count == 2 ≥ 2 → skip.
        promote_multiway(&mut plan, &rel_ids);
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::Project { .. }
        ));
    }

    /// Mixed plan: a recursive SCC with a linear-recursive triangle
    /// (count 1) AND a non-recursive SCC with a stable triangle
    /// (count 0). BOTH get promoted under the slice 4 contract.
    #[test]
    fn promotes_linear_rec_and_non_rec_sccs_in_mixed_plan() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["rec".to_string()],
            is_recursive: true,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "rec".to_string(),
                body: canonical_triangle_tree(),
                meta: Default::default(),
            },
        );
        builder.add_scc(Scc {
            id: 1,
            predicates: vec!["nonrec".to_string()],
            is_recursive: false,
        });
        builder.add_rule(
            1,
            CompiledRule {
                head: "nonrec".to_string(),
                body: canonical_triangle_tree(),
                meta: Default::default(),
            },
        );
        let mut plan = builder.build();
        let mut rel_ids = HashMap::new();
        rel_ids.insert("rec".to_string(), RelId(1)); // count 1 in SCC 0
                                                     // SCC 1 has no rec scans (count 0 in SCC 1 because "nonrec"
                                                     // not in body).
        promote_multiway(&mut plan, &rel_ids);
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
        assert!(matches!(
            &plan.rules_by_scc[1][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    // -----------------------------------------------------------
    // v0.6.5 slice 2 — 4-cycle promotion
    // -----------------------------------------------------------

    /// Build the canonical lowered+optimized 4-cycle subtree —
    /// `Project { Join { Join, Join } }` bushy shape with outer
    /// keys [0,3]/[3,0] and inner keys [1]/[0]. Verified against
    /// `Compiler::compile()` output for
    /// `cycle4(W,X,Y,Z) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W).`.
    fn canonical_4cycle_tree() -> RirNode {
        let inner_l = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let inner_r = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(3) }),
            right: Box::new(RirNode::Scan { rel: RelId(4) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner_l),
            right: Box::new(inner_r),
            left_keys: vec![0, 3],
            right_keys: vec![3, 0],
            join_type: JoinType::Inner,
        };
        RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
                ProjectExpr::Column(5),
            ],
        }
    }

    fn plan_with_4cycle_body(body: RirNode) -> ExecutionPlan {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["cycle4".to_string()],
            is_recursive: false,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "cycle4".to_string(),
                body,
                meta: Default::default(),
            },
        );
        builder.build()
    }

    #[test]
    fn promotes_canonical_4cycle() {
        let mut plan = plan_with_4cycle_body(canonical_4cycle_tree());
        promote_multiway(&mut plan, &HashMap::new());
        let body = &plan.rules_by_scc[0][0].body;
        match body {
            RirNode::MultiWayJoin {
                inputs,
                slot_vars,
                output_columns,
                fallback,
            } => {
                assert_eq!(inputs.len(), 4);
                assert!(matches!(inputs[0], RirNode::Scan { rel: RelId(1) }));
                assert!(matches!(inputs[1], RirNode::Scan { rel: RelId(2) }));
                assert!(matches!(inputs[2], RirNode::Scan { rel: RelId(3) }));
                assert!(matches!(inputs[3], RirNode::Scan { rel: RelId(4) }));
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
                assert!(matches!(fallback.as_ref(), RirNode::Project { .. }));
            }
            other => panic!("expected MultiWayJoin, got {:?}", other),
        }
    }

    #[test]
    fn fallback_4cycle_is_structurally_equal_to_input() {
        let pre = canonical_4cycle_tree();
        let mut plan = plan_with_4cycle_body(pre.clone());
        promote_multiway(&mut plan, &HashMap::new());
        let body = &plan.rules_by_scc[0][0].body;
        let RirNode::MultiWayJoin { fallback, .. } = body else {
            panic!("expected MultiWayJoin");
        };
        assert_eq!(format!("{:?}", fallback.as_ref()), format!("{:?}", pre));
    }

    #[test]
    fn idempotent_4cycle_under_repeat_calls() {
        let mut plan = plan_with_4cycle_body(canonical_4cycle_tree());
        promote_multiway(&mut plan, &HashMap::new());
        let first = format!("{:?}", &plan.rules_by_scc[0][0].body);
        promote_multiway(&mut plan, &HashMap::new());
        let second = format!("{:?}", &plan.rules_by_scc[0][0].body);
        assert_eq!(first, second);
    }

    #[test]
    fn rejects_4cycle_with_left_deep_shape() {
        // Left-deep with 3 atoms = triangle, not 4-cycle. Even
        // though it might have 4 head columns through some other
        // construction, the strict 4-cycle matcher requires the
        // bushy outer-Join shape with outer-right being a Join.
        // This input is the triangle's left-deep tree — it should
        // NOT match try_promote_4cycle.
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
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
                ProjectExpr::Column(5),
            ],
        };
        // This shape will be matched by try_promote_triangle if
        // its head columns are [0,1,3] — they aren't here (4 cols),
        // so triangle declines. 4-cycle declines because outer-right
        // is a Scan not a Join.
        assert!(try_promote_4cycle(&body).is_none());
    }

    #[test]
    fn rejects_4cycle_with_rotated_columns() {
        let mut body = canonical_4cycle_tree();
        if let RirNode::Project { columns, .. } = &mut body {
            // Rotate: swap col 0 and col 1.
            columns.swap(0, 1);
        }
        assert!(try_promote_4cycle(&body).is_none());
    }

    #[test]
    fn rejects_4cycle_with_non_inner_outer_join() {
        let mut body = canonical_4cycle_tree();
        if let RirNode::Project { input, .. } = &mut body {
            if let RirNode::Join { join_type, .. } = input.as_mut() {
                *join_type = JoinType::LeftOuter;
            }
        }
        assert!(try_promote_4cycle(&body).is_none());
    }

    #[test]
    fn rejects_4cycle_with_wrong_outer_keys() {
        let mut body = canonical_4cycle_tree();
        if let RirNode::Project { input, .. } = &mut body {
            if let RirNode::Join { left_keys, .. } = input.as_mut() {
                *left_keys = vec![0, 4]; // not [0, 3]
            }
        }
        assert!(try_promote_4cycle(&body).is_none());
    }

    /// Slice 4 contract: stable 4-cycle (zero recursive Scans) IS
    /// promoted in a recursive SCC.
    #[test]
    fn promotes_stable_4cycle_in_recursive_scc() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["rec_cycle".to_string()],
            is_recursive: true,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "rec_cycle".to_string(),
                body: canonical_4cycle_tree(),
                meta: Default::default(),
            },
        );
        let mut plan = builder.build();
        // No "rec_cycle" entry in rel_ids → count 0 → promote.
        promote_multiway(&mut plan, &HashMap::new());
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Slice 4 contract: linear-recursive 4-cycle (count 1) IS
    /// promoted.
    #[test]
    fn promotes_linear_recursive_4cycle() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["rec_cycle".to_string()],
            is_recursive: true,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "rec_cycle".to_string(),
                body: canonical_4cycle_tree(),
                meta: Default::default(),
            },
        );
        let mut plan = builder.build();
        let mut rel_ids = HashMap::new();
        // canonical_4cycle_tree uses 4 Scans; map exactly one to
        // rec_cycle (RelId selection from 4cycle_tree below).
        rel_ids.insert("rec_cycle".to_string(), RelId(2));
        promote_multiway(&mut plan, &rel_ids);
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Slice 4 contract: ≥ 2 recursive Scans in a 4-cycle body →
    /// NOT promoted.
    #[test]
    fn skips_multirec_4cycle_in_recursive_scc() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["rc_a".to_string(), "rc_b".to_string()],
            is_recursive: true,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "rc_a".to_string(),
                body: canonical_4cycle_tree(),
                meta: Default::default(),
            },
        );
        let mut plan = builder.build();
        let mut rel_ids = HashMap::new();
        rel_ids.insert("rc_a".to_string(), RelId(1));
        rel_ids.insert("rc_b".to_string(), RelId(2));
        promote_multiway(&mut plan, &rel_ids);
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::Project { .. }
        ));
    }

    #[test]
    fn triangle_does_not_match_4cycle_promoter() {
        // Pin the cross-shape contract: a body matched by
        // try_promote_triangle must NOT also match try_promote_4cycle.
        // Both promoters should be exclusive.
        let triangle = canonical_triangle_tree();
        assert!(try_promote_4cycle(&triangle).is_none());
        let four_cycle = canonical_4cycle_tree();
        assert!(try_promote_triangle(&four_cycle).is_none());
    }
}
