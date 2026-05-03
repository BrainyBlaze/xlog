//! v0.6.5 slice 1 — `RirNode::MultiWayJoin` IR variant smoke tests.
//!
//! Asserts the new variant integrates with the IR's recursion
//! helpers: `is_leaf`, `referenced_relations` (via the private
//! `collect_relations` walker). Behavioral semantics (promotion,
//! fallback descent, GPU dispatch) are tested in their owning
//! crates; this file pins only the IR-level invariants.
//!
//! Per the v0.6.5 slice 1 plan:
//! `referenced_relations` recurses into `inputs` (the canonical
//! WCOJ slot order). The `fallback` subtree references the same
//! set by promoter invariant; we keep the canonical answer
//! minimal.

use xlog_core::RelId;
use xlog_ir::rir::ProjectExpr;
use xlog_ir::{JoinType, RirNode};

/// Build a canonical triangle `MultiWayJoin` with relation IDs
/// (1, 2, 3) for the three slots and a representative fallback
/// tree. Used across multiple tests.
fn canonical_triangle_multiway() -> RirNode {
    let scan_xy = RirNode::Scan { rel: RelId(1) };
    let scan_yz = RirNode::Scan { rel: RelId(2) };
    let scan_xz = RirNode::Scan { rel: RelId(3) };

    // Fallback tree mirrors the lowerer's nested Join+Project
    // shape for `t(X,Y,Z) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)`.
    let inner_join = RirNode::Join {
        left: Box::new(scan_xy.clone()),
        right: Box::new(scan_yz.clone()),
        left_keys: vec![1],
        right_keys: vec![0],
        join_type: JoinType::Inner,
    };
    let outer_join = RirNode::Join {
        left: Box::new(inner_join),
        right: Box::new(scan_xz.clone()),
        left_keys: vec![0, 3],
        right_keys: vec![0, 1],
        join_type: JoinType::Inner,
    };
    let fallback = RirNode::Project {
        input: Box::new(outer_join),
        columns: vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ],
    };

    RirNode::MultiWayJoin {
        inputs: vec![scan_xy, scan_yz, scan_xz],
        slot_vars: vec![
            vec![Some(0), Some(1)],
            vec![Some(1), Some(2)],
            vec![Some(0), Some(2)],
        ],
        output_columns: vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ],
        fallback: Box::new(fallback),
    }
}

#[test]
fn multiway_join_is_not_a_leaf() {
    let node = canonical_triangle_multiway();
    assert!(!node.is_leaf());
}

#[test]
fn referenced_relations_recurses_into_inputs() {
    let node = canonical_triangle_multiway();
    let rels = node.referenced_relations();
    // Order is implementation-defined; rely on set membership.
    assert!(rels.contains(&RelId(1)));
    assert!(rels.contains(&RelId(2)));
    assert!(rels.contains(&RelId(3)));
}

#[test]
fn referenced_relations_does_not_double_count_via_fallback() {
    // The plan invariant: collect_relations recurses into `inputs`
    // only, not `fallback`. If both were walked, RelId(1), (2), (3)
    // would each appear twice (once from `inputs`, once from the
    // fallback's Scan leaves). One pass → 3 entries (one per
    // input scan).
    let node = canonical_triangle_multiway();
    let rels = node.referenced_relations();
    assert_eq!(
        rels.len(),
        3,
        "expected 3 entries (one per WCOJ slot), got {}: {:?}",
        rels.len(),
        rels,
    );
}

#[test]
fn multiway_join_with_empty_inputs_collects_nothing() {
    // Defensive: the IR variant accepts arbitrary `inputs`. A
    // zero-input MultiWayJoin (not currently emitted by any
    // promoter) reports zero relations from the inputs walk and
    // does NOT walk fallback.
    let stub_fallback = RirNode::Unit;
    let node = RirNode::MultiWayJoin {
        inputs: vec![],
        slot_vars: vec![],
        output_columns: vec![],
        fallback: Box::new(stub_fallback),
    };
    assert!(node.referenced_relations().is_empty());
    assert!(!node.is_leaf());
}

/// v0.6.5 slice 2 (D4) — shape-agnosticism guard.
///
/// Slice 1 added `MultiWayJoin` with a triangle-only promoter, but
/// the IR variant itself is shape-agnostic. Slice 2a (4-way) will
/// add a 4-input promoter; this test pins the contract that
/// `referenced_relations` on a synthesized 4-input MultiWayJoin
/// reports four distinct relations from `inputs` alone, regardless
/// of arity.
#[test]
fn referenced_relations_handles_4_inputs() {
    let scans = [10u32, 20, 30, 40].map(|id| RirNode::Scan { rel: RelId(id) });
    let node = RirNode::MultiWayJoin {
        inputs: scans.to_vec(),
        // Synthetic 4-cycle slot_vars: [[A,B],[B,C],[C,D],[A,D]].
        slot_vars: vec![
            vec![Some(0), Some(1)],
            vec![Some(1), Some(2)],
            vec![Some(2), Some(3)],
            vec![Some(0), Some(3)],
        ],
        output_columns: vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
        ],
        // Stub fallback — the test does not execute this; it only
        // exercises the IR walker.
        fallback: Box::new(RirNode::Unit),
    };
    let rels = node.referenced_relations();
    assert_eq!(
        rels.len(),
        4,
        "expected 4 entries (one per input slot), got {}: {:?}",
        rels.len(),
        rels,
    );
    for id in [10, 20, 30, 40] {
        assert!(
            rels.contains(&RelId(id)),
            "RelId({}) missing from {:?}",
            id,
            rels,
        );
    }
}
