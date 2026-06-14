//! `MultiWayJoin` promotion pass.
//! `ChainJoin` promotion for 2-atom chains.
//! Recursive-SCC promotion for occurrence-level recursive bodies.
//!
//! Walks an [`ExecutionPlan`] (post-lowering, post-optimizer) and
//! rewrites recognized triangle / 4-cycle / K-clique subtrees in
//! `rule.body` to [`RirNode::MultiWayJoin`], plus recognized 2-atom
//! chains to [`RirNode::ChainJoin`]. Idempotent.
//!
//! ## Eligibility
//!
//! Exact-match against the canonical lowered-and-optimized triangle
//! shape — the same tree shape recognized by the legacy triangle
//! executor matcher:
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
//! 4-cycle has the analogous canonical lowered shape. Any
//! deviation in shape, predicate-pushdown-altered Join, or
//! computed-projection variants is left untouched.
//!
//! ## Recursive SCC handling
//!
//! The promoter does not blanket-skip recursive SCCs. It gates
//! per-rule on the number of body Scans whose RelId resolves to a
//! predicate inside the rule's head SCC:
//!
//! | Recursive Scans in body | Behavior                                 |
//! |-------------------------|------------------------------------------|
//! | 0 (stable rule)         | Promote                                  |
//! | 1 (linear recursion)    | Promote                                  |
//! | ≥ 2 (multi-recursion)   | Promote                                  |
//!
//! Per arXiv:2604.20073, semi-naïve evaluation reasons over
//! body-clause OCCURRENCES, not predicate names. Multi-recursive
//! bodies including same-predicate self-recursive
//! occurrences (e.g. `tri(X,Y,Z) :- p(X,Y), p(Y,Z), q(X,Z)` with
//! `p` recursive) are admitted. The recursive engine
//! (`Executor::execute_recursive_scc`) consumes the resulting
//! `MultiWayJoin` via `execute_wcoj_or_fallback_node`, dispatching
//! WCOJ kernels on the seeding pass and on each iteration's variant.
//! The runtime occurrence-identity rewrite at
//! `crates/xlog-runtime/src/executor/rewrite.rs:303-311 + :477-504`
//! ensures per-variant rewrites preserve the N-th occurrence
//! independently in `MultiWayJoin.inputs` and `MultiWayJoin.fallback`.
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
//! * Stream-aligned multiplexing and adaptive histogram resolution
//!   for recursive cliques.
//! * 4-way / general-arity admission beyond triangle / 4-cycle /
//!   supported K-clique and generic Free Join paths.

use std::collections::HashMap;
use xlog_core::RelId;
use xlog_ir::rir::{
    ColumnSwap, CostPredictionRecord as RirCostPredictionRecord, KCliqueVariableOrder,
    MultiwayPlan, PlannedHashReason, ProjectExpr, SortedLayoutSpec, StreamGroupId, VariableOrder,
    K_CLIQUE_MAX_EDGES, K_CLIQUE_MAX_K,
};
use xlog_ir::{ExecutionPlan, JoinType, RirNode};
use xlog_stats::{StatsManager, StatsSnapshot};

use crate::compiler_config::CompilerConfig;
use crate::hypergraph::var_order::{
    plan_kclique_var_order, FullVariableOrder, KCliqueEdge, KCliqueShape,
};
use crate::hypergraph::VertexId;
use crate::wcoj_var_ordering::{wcoj_cost_gate_predicts_wcoj, WcojVariableOrderingModel};

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
/// **Recursive SCC bodies.** Multi-recursive
/// bodies are admitted, including same-predicate self-recursive
/// occurrences (per arXiv:2604.20073 — semi-naïve evaluation reasons
/// over body-clause OCCURRENCES, not predicate names). The triangle /
/// 4-cycle shape gates already cap atom count at 3 / 4, so the
/// recursive-Scan count is implicitly bounded; the runtime's
/// per-variant rewrite + dispatch loop in `execute_recursive_scc`
/// handles N variants correctly with the occurrence-identity rewrite.
pub fn promote_multiway(
    plan: &mut ExecutionPlan,
    _rel_ids: &HashMap<String, RelId>,
    stats: &StatsManager,
    config: &CompilerConfig,
) {
    // Generic Free Join: the general multiway promoter sizes Scan leaves
    // from relation arities (the clique walker's hardcoded arity-2
    // assumption does not generalize). Cloned up front because the
    // per-rule loop holds `rules_by_scc` mutably.
    let rel_arities = plan.rel_arities.clone();
    for (scc_id, rules) in plan.rules_by_scc.iter_mut().enumerate() {
        if plan.sccs.get(scc_id).is_none() {
            continue;
        }
        for rule in rules.iter_mut() {
            // Occurrence-level recursive gate: the previous
            // `recursive_scan_count > 1` cutoff is absent. The
            // triangle / 4-cycle shape gates (try_promote_*) cap atom
            // count at 3 / 4, implicitly bounding the recursive-Scan
            // count at the rule's atom count. Multi-recursive bodies
            // (count >= 2), including same-predicate self-recursive
            // (e.g. `tri(X,Y,Z) :- p(X,Y), p(Y,Z), q(X,Z)`), are
            // admitted; the runtime's per-variant rewrite + dispatch
            // loop in `execute_recursive_scc` handles N variants
            // correctly after the occurrence-identity rewrite at
            // `crates/xlog-runtime/src/executor/rewrite.rs:303-311 +
            // :477-504`.
            // ChainJoin promotion: a 2-atom chain is not
            // paper-prescribed WCOJ; it is xlog-original routing
            // motivated by profiler traces. Production emits a
            // first-class `ChainJoin` so walkers and dispatchers can
            // distinguish the chain route from paper-derived
            // `MultiWayJoin` shapes.
            // Aggregate group-by fusion: aggregate rules wrap the join tree as
            // Project{final} -> GroupBy -> Project{group} -> <join tree>,
            // so the shape gates below (which see only the outer Project)
            // never promote triangle bodies under aggregate heads. Descend
            // through the wrapper and promote the inner triangle in place;
            // the executor's fused group-by-root count hook dispatches it,
            // and a declined dispatch still executes the embedded binary
            // fallback unchanged.
            if try_promote_triangle_inside_aggregate(&mut rule.body, stats, config) {
                continue;
            }
            // 4-cycle aggregate descent: sibling of the descent above (atom-count gates
            // make the two matchers disjoint). The executor's fused hook
            // dispatches Count only; declined dispatches still execute the
            // embedded binary fallback unchanged.
            if try_promote_4cycle_inside_aggregate(&mut rule.body, stats, config) {
                continue;
            }
            // K-clique aggregate descent: k = 5, 6 sibling of the descents above
            // (scan-count gates keep all three matchers disjoint). The
            // executor's fused hook dispatches Count grouped by the
            // plan's root variable only; declined dispatches still
            // execute the embedded binary fallback unchanged.
            if try_promote_clique_inside_aggregate(&mut rule.body, stats) {
                continue;
            }
            // Generic multiway aggregate descent: sibling of the descents above
            // — any >=3-atom inner-join tree under an aggregate
            // wrapper that no dedicated descent recognized becomes a
            // FreeJoin-marked MultiWayJoin. The executor's fused hook
            // dispatches factorized count-by-root only; declined
            // dispatches still execute the embedded binary fallback
            // unchanged.
            if try_promote_general_multiway_inside_aggregate(&mut rule.body, &rel_arities) {
                continue;
            }
            if let Some(promoted) = try_promote_chain(&rule.body) {
                rule.body = promoted;
                continue;
            }
            // Canonical-shape robustness: the lowerer's bushy DP planner may
            // emit a right-deep `Project(Join(Scan, Join(Scan,
            // Scan)))` triangle for small-card inputs (snapshot-driven
            // recompile flow) — semantically a triangle, but the original
            // canonical-shape matcher rejects it. `normalize_*_to_left_deep`
            // detects those alternative-but-equivalent shapes and
            // commutativity-rewrites them to the canonical left-deep
            // form before matching. Idempotent on already-canonical
            // bodies — left-deep input passes through unchanged.
            let normalized_tri = normalize_triangle_to_left_deep(&rule.body);
            let body_for_tri = normalized_tri.as_ref().unwrap_or(&rule.body);
            if let Some(promoted) = try_promote_triangle(body_for_tri, stats, config) {
                rule.body = promoted;
                continue;
            }
            let normalized_4c = normalize_4cycle_to_bushy(&rule.body);
            let body_for_4c = normalized_4c.as_ref().unwrap_or(&rule.body);
            if let Some(promoted) = try_promote_4cycle(body_for_4c, stats, config) {
                rule.body = promoted;
                continue;
            }
            // K-clique promotion for k=5..k=8. Tree-flatten +
            // complete-K_k validation. Robust to left-deep /
            // right-deep / bushy. Order is doc anchor only;
            // a body matching one K cannot also match another
            // (different scan count). Recursive clique bodies are
            // admitted so the runtime can rebuild leader-edge
            // metadata from the current semi-naive store state.
            if let Some(promoted) = try_promote_clique_k(&rule.body, 5, stats)
                .or_else(|| try_promote_clique_k(&rule.body, 6, stats))
                .or_else(|| try_promote_clique_k(&rule.body, 7, stats))
                .or_else(|| try_promote_clique_k(&rule.body, 8, stats))
            {
                rule.body = promoted;
                continue;
            }
            // Generic Free Join — general >=3-atom inner-join bodies that
            // no dedicated shape promoter recognized become a generic
            // `MultiWayJoin` (plan: None). The executor derives a Free
            // Join plan from `slot_vars` at dispatch time
            // (`try_dispatch_free_join`); structural declines
            // (non-prefix bound columns, non-u32/Symbol inputs,
            // repeated cover vars, dedicated-shape carve-outs) execute
            // the embedded binary fallback unchanged.
            if let Some(promoted) = try_promote_general_multiway(&rule.body, &rel_arities) {
                rule.body = promoted;
                continue;
            }
        }
    }
}

/// Triangle semantic-slot encoding: encode each atom-column slot as a u8 in `0..6` —
/// `(atom_idx * 2) + col_idx` where `atom_idx` is `0` =
/// inner-left, `1` = inner-right, `2` = outer-third.
fn ac_idx(atom_idx: u8, col_idx: u8) -> u8 {
    debug_assert!(atom_idx < 3);
    debug_assert!(col_idx < 2);
    atom_idx * 2 + col_idx
}

/// Map an inner-output column index `0..4` to the underlying
/// atom-column. `0..2` come from the inner-left scan;
/// `2..4` from the inner-right scan.
fn inner_output_ac(k: usize) -> Option<(u8, u8)> {
    match k {
        0 => Some((0, 0)),
        1 => Some((0, 1)),
        2 => Some((1, 0)),
        3 => Some((1, 1)),
        _ => None,
    }
}

/// Map an outer-output column index `0..6` to the underlying
/// atom-column. `0..4` come from the inner subtree;
/// `4..6` from the outer-third scan.
fn outer_output_ac(k: usize) -> Option<(u8, u8)> {
    match k {
        0..=3 => inner_output_ac(k),
        4 => Some((2, 0)),
        5 => Some((2, 1)),
        _ => None,
    }
}

/// Tiny union-find over `0..6`. Returns the root of `x` after
/// path compression.
fn uf_find(parent: &mut [u8; 6], x: u8) -> u8 {
    let mut root = x;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    let mut cur = x;
    while parent[cur as usize] != root {
        let next = parent[cur as usize];
        parent[cur as usize] = root;
        cur = next;
    }
    root
}

fn uf_union(parent: &mut [u8; 6], a: u8, b: u8) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[rb as usize] = ra;
    }
}

/// Semantic-slot inference for triangle bodies. Given
/// the inner-pair scans + outer scan + key shapes + project
/// columns, deduce which atom is the XY-edge / YZ-edge /
/// XZ-edge from the variable-equivalence graph. Returns
/// `(rel_xy, rel_yz, rel_xz)` regardless of the body's
/// positional layout.
///
/// Returns `None` when:
///   * Key arities don't match the canonical lowered triangle
///     (1-element inner keys, 2-element outer keys).
///   * Project doesn't have exactly 3 `Column(_)` entries.
///   * The variable-equivalence graph doesn't yield exactly 3
///     classes of size 2 (i.e., not a triangle's 3-vars-each-
///     bound-twice signature).
///   * The 3 head columns don't pick 3 distinct equivalence
///     classes.
///   * Any atom doesn't bind exactly 2 of the 3 head variables.
#[allow(clippy::too_many_arguments)]
fn infer_triangle_semantics(
    inner_left_rel: RelId,
    inner_right_rel: RelId,
    outer_third_rel: RelId,
    lk2: &[usize],
    rk2: &[usize],
    lk1: &[usize],
    rk1: &[usize],
    project_cols: &[ProjectExpr],
) -> Option<(RelId, RelId, RelId)> {
    if lk2.len() != 1 || rk2.len() != 1 {
        return None;
    }
    if lk1.len() != 2 || rk1.len() != 2 {
        return None;
    }
    if project_cols.len() != 3 {
        return None;
    }
    if lk2[0] >= 2 || rk2[0] >= 2 {
        return None;
    }
    if lk1.iter().any(|k| *k >= 4) || rk1.iter().any(|k| *k >= 2) {
        return None;
    }

    // Build equivalence classes via union-find over the 6
    // atom-column slots.
    let mut parent = [0u8, 1, 2, 3, 4, 5];

    // Inner join: inner_left.col[lk2[0]] ≡ inner_right.col[rk2[0]]
    uf_union(
        &mut parent,
        ac_idx(0, lk2[0] as u8),
        ac_idx(1, rk2[0] as u8),
    );

    // Outer join: 2 equivalences from (lk1, rk1).
    for i in 0..2 {
        let (inner_atom, inner_col) = inner_output_ac(lk1[i])?;
        uf_union(
            &mut parent,
            ac_idx(inner_atom, inner_col),
            ac_idx(2, rk1[i] as u8),
        );
    }

    // Roots define classes. We need exactly 3 distinct roots
    // covering all 6 slots, with each class of size 2.
    let roots: [u8; 6] = std::array::from_fn(|i| uf_find(&mut parent, i as u8));
    let mut counts: HashMap<u8, u8> = HashMap::new();
    for r in &roots {
        *counts.entry(*r).or_insert(0) += 1;
    }
    if counts.len() != 3 || counts.values().any(|c| *c != 2) {
        return None;
    }

    // Map head columns (project_cols) to their equivalence
    // classes.
    let mut head_classes: [u8; 3] = [0; 3];
    for (i, pc) in project_cols.iter().enumerate() {
        let ProjectExpr::Column(k) = pc else {
            return None;
        };
        let (atom, col) = outer_output_ac(*k)?;
        head_classes[i] = uf_find(&mut parent, ac_idx(atom, col));
    }
    // Head must pick 3 distinct classes (X, Y, Z).
    if head_classes[0] == head_classes[1]
        || head_classes[0] == head_classes[2]
        || head_classes[1] == head_classes[2]
    {
        return None;
    }
    let x_class = head_classes[0];
    let y_class = head_classes[1];
    let z_class = head_classes[2];

    // For each atom (0=inner_left, 1=inner_right, 2=third),
    // determine which two head-vars it binds.
    let atom_classes = |atom_idx: u8| -> (u8, u8) {
        (
            roots[ac_idx(atom_idx, 0) as usize],
            roots[ac_idx(atom_idx, 1) as usize],
        )
    };

    let atom_rels = [inner_left_rel, inner_right_rel, outer_third_rel];
    let mut rel_xy: Option<RelId> = None;
    let mut rel_yz: Option<RelId> = None;
    let mut rel_xz: Option<RelId> = None;

    for atom_idx in 0..3u8 {
        let (c0, c1) = atom_classes(atom_idx);
        let binds_x = c0 == x_class || c1 == x_class;
        let binds_y = c0 == y_class || c1 == y_class;
        let binds_z = c0 == z_class || c1 == z_class;
        match (binds_x, binds_y, binds_z) {
            (true, true, false) => rel_xy = Some(atom_rels[atom_idx as usize]),
            (false, true, true) => rel_yz = Some(atom_rels[atom_idx as usize]),
            (true, false, true) => rel_xz = Some(atom_rels[atom_idx as usize]),
            _ => return None,
        }
    }

    Some((rel_xy?, rel_yz?, rel_xz?))
}

/// Right-deep triangle normalization: detect a right-deep triangle body
/// `Project(Join(Scan(third), Join(Scan(inner_l), Scan(inner_r))))`
/// and commutativity-rewrite it to the canonical left-deep
/// form `Project(Join(Join(Scan(inner_l), Scan(inner_r)),
/// Scan(third)))`. Returns `None` for any non-matching shape
/// (left-deep / fully unknown / nested deeper) — caller falls
/// back to the original body.
///
/// The rewrite preserves semantics under inner-join
/// commutativity. Project columns must be remapped because
/// the output column layout swaps:
///   * Right-deep: `[third.0, third.1, inner_l.0, inner_l.1,
///     inner_r.0, inner_r.1]`
///   * Left-deep:  `[inner_l.0, inner_l.1, inner_r.0, inner_r.1,
///     third.0, third.1]`
///   * Remap formula: `new_k = (old_k + 4) % 6`.
///     The outer Join's `(left_keys, right_keys)` swap correspondingly
///     (the side they reference switches). Inner Join's keys are
///     unchanged.
fn normalize_triangle_to_left_deep(node: &RirNode) -> Option<RirNode> {
    let RirNode::Project {
        input: outer_input,
        columns,
    } = node
    else {
        return None;
    };
    let RirNode::Join {
        left: outer_l,
        right: outer_r,
        left_keys: outer_lk,
        right_keys: outer_rk,
        join_type: outer_jt,
    } = outer_input.as_ref()
    else {
        return None;
    };
    if !matches!(outer_jt, JoinType::Inner) {
        return None;
    }
    // Right-deep triangle requires outer.left = Scan, outer.right = Join.
    let RirNode::Scan { rel: _ } = outer_l.as_ref() else {
        return None;
    };
    let RirNode::Join { .. } = outer_r.as_ref() else {
        return None;
    };
    // Verify outer.right's structure (inner Join of two Scans) so
    // we don't mistakenly rewrite non-triangle right-deep trees.
    let RirNode::Join {
        left: inner_l,
        right: inner_r,
        ..
    } = outer_r.as_ref()
    else {
        return None;
    };
    if !matches!(inner_l.as_ref(), RirNode::Scan { .. })
        || !matches!(inner_r.as_ref(), RirNode::Scan { .. })
    {
        return None;
    }
    // Swap: new outer.left = old outer.right (inner Join);
    //       new outer.right = old outer.left (third Scan);
    //       new outer.left_keys = old outer.right_keys;
    //       new outer.right_keys = old outer.left_keys.
    let new_outer = RirNode::Join {
        left: outer_r.clone(),
        right: outer_l.clone(),
        left_keys: outer_rk.clone(),
        right_keys: outer_lk.clone(),
        join_type: JoinType::Inner,
    };
    // Remap project columns: (k + 4) % 6.
    let new_columns: Vec<ProjectExpr> = columns
        .iter()
        .map(|expr| match expr {
            ProjectExpr::Column(k) => ProjectExpr::Column((*k + 4) % 6),
            other => other.clone(),
        })
        .collect();
    Some(RirNode::Project {
        input: Box::new(new_outer),
        columns: new_columns,
    })
}

/// Right-deep 4-cycle normalization: detect a fully right-deep 4-cycle body
/// `Project(Join(Scan(R0), Join(Scan(R1), Join(Scan(R2), Scan(R3)))))`
/// (the lowerer's bushy DP can pick this shape at small
/// cardinalities) and rewrite to the canonical bushy form
/// `Project(Join(Join(Scan(R0), Scan(R1)), Join(Scan(R2), Scan(R3))))`
/// that the canonical 4-cycle promoter matches.
///
/// The output column layout is preserved (both forms produce
/// `[R0.0, R0.1, R1.0, R1.1, R2.0, R2.1, R3.0, R3.1]`), so
/// project columns pass through unchanged.
///
/// Validation: only rewrites when the right-deep keys exactly
/// match the canonical 4-cycle topology
/// (rotation-only `[1]/[0]` on each inner Join + outer keys
/// `[0,1]/[5,0]`). Other shapes return `None` — caller falls
/// back to the original body (and previous canonical-shape behavior).
fn normalize_4cycle_to_bushy(node: &RirNode) -> Option<RirNode> {
    let RirNode::Project {
        input: outer_input,
        columns,
    } = node
    else {
        return None;
    };
    let RirNode::Join {
        left: outer_l,
        right: outer_r,
        left_keys: outer_lk,
        right_keys: outer_rk,
        join_type: outer_jt,
    } = outer_input.as_ref()
    else {
        return None;
    };
    if !matches!(outer_jt, JoinType::Inner) {
        return None;
    }
    // Right-deep canonical-cycle pattern: outer.left = Scan(R0);
    // outer.right = Join(Scan(R1), Join(Scan(R2), Scan(R3))).
    let RirNode::Scan { rel: r0 } = outer_l.as_ref() else {
        return None;
    };
    let RirNode::Join {
        left: middle_l,
        right: middle_r,
        left_keys: middle_lk,
        right_keys: middle_rk,
        join_type: middle_jt,
    } = outer_r.as_ref()
    else {
        return None;
    };
    if !matches!(middle_jt, JoinType::Inner) {
        return None;
    }
    let RirNode::Scan { rel: r1 } = middle_l.as_ref() else {
        return None;
    };
    let RirNode::Join {
        left: deep_l,
        right: deep_r,
        left_keys: deep_lk,
        right_keys: deep_rk,
        join_type: deep_jt,
    } = middle_r.as_ref()
    else {
        return None;
    };
    if !matches!(deep_jt, JoinType::Inner) {
        return None;
    }
    let RirNode::Scan { rel: r2 } = deep_l.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: r3 } = deep_r.as_ref() else {
        return None;
    };
    // Validate canonical-cycle keys.
    if outer_lk.as_slice() != [0, 1] || outer_rk.as_slice() != [5, 0] {
        return None;
    }
    if middle_lk.as_slice() != [1] || middle_rk.as_slice() != [0] {
        return None;
    }
    if deep_lk.as_slice() != [1] || deep_rk.as_slice() != [0] {
        return None;
    }
    // Reconstruct bushy form. Inner pairs use the canonical
    // (e1,e2) and (e3,e4) edge keys [1]/[0]. Outer connects the
    // two subtrees on (e2,e3) [Y] and (e4,e1) [W] cycle edges:
    //   left_subtree output: [R0.0, R0.1, R1.0, R1.1]
    //   right_subtree output: [R2.0, R2.1, R3.0, R3.1]
    //   left_keys [3, 0]: R1.1 (Y), R0.0 (W)
    //   right_keys [0, 3]: R2.0 (Y), R3.1 (W)
    let inner_left = RirNode::Join {
        left: Box::new(RirNode::Scan { rel: *r0 }),
        right: Box::new(RirNode::Scan { rel: *r1 }),
        left_keys: vec![1],
        right_keys: vec![0],
        join_type: JoinType::Inner,
    };
    let inner_right = RirNode::Join {
        left: Box::new(RirNode::Scan { rel: *r2 }),
        right: Box::new(RirNode::Scan { rel: *r3 }),
        left_keys: vec![1],
        right_keys: vec![0],
        join_type: JoinType::Inner,
    };
    let new_outer = RirNode::Join {
        left: Box::new(inner_left),
        right: Box::new(inner_right),
        left_keys: vec![3, 0],
        right_keys: vec![0, 3],
        join_type: JoinType::Inner,
    };
    Some(RirNode::Project {
        input: Box::new(new_outer),
        columns: columns.clone(),
    })
}

/// Semantic triangle recognition: recognize the canonical triangle in any valid
/// inner-key combination (`[1]/[0]`, `[1]/[1]`, `[0]/[0]`)
/// and produce the equivalent `MultiWayJoin` with
/// `inputs` arranged in canonical semantic order
/// Aggregate group-by fusion: promote a triangle join tree sitting inside an
/// aggregate rule's `Project{ GroupBy { Project { <join tree> } } }`
/// wrapper. The inner tree is matched exactly like a top-level triangle
/// body (via a synthesized canonical (X, Y, Z) projection), and on success
/// the group projection's column indices are remapped from join-output
/// space into the MultiWayJoin's (X, Y, Z) output space. Returns false —
/// leaving the rule untouched — on any structural mismatch.
fn try_promote_triangle_inside_aggregate(
    body: &mut RirNode,
    stats: &StatsManager,
    config: &CompilerConfig,
) -> bool {
    let RirNode::Project { input: gb, .. } = body else {
        return false;
    };
    let RirNode::GroupBy {
        input: group_input, ..
    } = gb.as_mut()
    else {
        return false;
    };
    let RirNode::Project {
        input: inner,
        columns: group_cols,
    } = group_input.as_mut()
    else {
        return false;
    };
    // The matcher expects the canonical left-deep triangle projection
    // (X, Y, Z) = join-output columns [0, 1, 3].
    let canonical = RirNode::Project {
        input: inner.clone(),
        columns: vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
        ],
    };
    let normalized = normalize_triangle_to_left_deep(&canonical);
    let candidate = normalized.as_ref().unwrap_or(&canonical);
    let Some(promoted) = try_promote_triangle(candidate, stats, config) else {
        return false;
    };
    let RirNode::MultiWayJoin { output_columns, .. } = &promoted else {
        return false;
    };
    let mut remapped: Vec<ProjectExpr> = Vec::with_capacity(group_cols.len());
    for col in group_cols.iter() {
        let ProjectExpr::Column(c) = col else {
            return false;
        };
        let Some(pos) = output_columns
            .iter()
            .position(|oc| matches!(oc, ProjectExpr::Column(x) if x == c))
        else {
            return false;
        };
        remapped.push(ProjectExpr::Column(pos));
    }
    *group_cols = remapped;
    **inner = promoted;
    true
}

/// 4-cycle aggregate group-by fusion: sibling of
/// [`try_promote_triangle_inside_aggregate`]. Promotes a 4-cycle join tree
/// sitting inside an aggregate rule's
/// `Project{ GroupBy { Project { <join tree> } } }` wrapper. The inner tree
/// is matched exactly like a top-level 4-cycle body via a synthesized
/// canonical (W, X, Y, Z) projection (join-output columns [0, 1, 3, 5] of
/// the canonical-cycle tree), and on success the group projection's column
/// indices are remapped from join-output space into the MultiWayJoin's
/// (W, X, Y, Z) output space — output position 0 is the variable-order
/// root W by construction, which is what the executor's fused dispatch
/// requires for the group key. Returns false — leaving the rule
/// untouched — on any structural mismatch.
fn try_promote_4cycle_inside_aggregate(
    body: &mut RirNode,
    stats: &StatsManager,
    config: &CompilerConfig,
) -> bool {
    let RirNode::Project { input: gb, .. } = body else {
        return false;
    };
    let RirNode::GroupBy {
        input: group_input, ..
    } = gb.as_mut()
    else {
        return false;
    };
    let RirNode::Project {
        input: inner,
        columns: group_cols,
    } = group_input.as_mut()
    else {
        return false;
    };
    // The matcher expects the canonical 4-cycle projection
    // (W, X, Y, Z) = join-output columns [0, 1, 3, 5].
    let canonical = RirNode::Project {
        input: inner.clone(),
        columns: vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(3),
            ProjectExpr::Column(5),
        ],
    };
    let normalized = normalize_4cycle_to_bushy(&canonical);
    let candidate = normalized.as_ref().unwrap_or(&canonical);
    let Some(promoted) = try_promote_4cycle(candidate, stats, config) else {
        return false;
    };
    let RirNode::MultiWayJoin { output_columns, .. } = &promoted else {
        return false;
    };
    let mut remapped: Vec<ProjectExpr> = Vec::with_capacity(group_cols.len());
    for col in group_cols.iter() {
        let ProjectExpr::Column(c) = col else {
            return false;
        };
        let Some(pos) = output_columns
            .iter()
            .position(|oc| matches!(oc, ProjectExpr::Column(x) if x == c))
        else {
            return false;
        };
        remapped.push(ProjectExpr::Column(pos));
    }
    *group_cols = remapped;
    **inner = promoted;
    true
}

/// K-clique aggregate group-by fusion: k = 5, 6 sibling of
/// [`try_promote_triangle_inside_aggregate`]. Promotes a complete-K_k
/// join tree sitting inside an aggregate rule's
/// `Project{ GroupBy { Project { <join tree> } } }` wrapper.
///
/// Unlike triangle/4-cycle, the clique matcher derives head variables
/// from a head projection over the join tree. The aggregate wrapper has
/// no such projection, so this descent synthesizes one: union-find over
/// the flattened global slot space yields the k variable classes, and
/// one representative slot per class (in first-appearance order) forms
/// the canonical k-variable projection handed to
/// [`try_promote_clique_k`]. On success the group projection's column
/// indices are remapped from join-output (global slot) space into the
/// clique head-variable space via the same classes. K = 7/8 bodies are
/// left untouched (no fused count kernels exist for them — the
/// non-aggregate clique route would buy nothing under an aggregate
/// wrapper that the executor cannot fuse). Returns false — leaving the
/// rule untouched — on any structural mismatch.
fn try_promote_clique_inside_aggregate(body: &mut RirNode, stats: &StatsManager) -> bool {
    let RirNode::Project { input: gb, .. } = body else {
        return false;
    };
    let RirNode::GroupBy {
        input: group_input, ..
    } = gb.as_mut()
    else {
        return false;
    };
    let RirNode::Project {
        input: inner,
        columns: group_cols,
    } = group_input.as_mut()
    else {
        return false;
    };
    // Flatten the inner join tree directly (no head projection exists
    // under the aggregate wrapper).
    let mut scans: Vec<RelId> = Vec::new();
    let mut key_pairs: Vec<(usize, usize)> = Vec::new();
    if walk_clique_node(inner, &mut scans, &mut key_pairs).is_none() {
        return false;
    }
    let k = match scans.len() {
        10 => 5,
        15 => 6,
        _ => return false,
    };
    // Union-find on the global slot space; representative slot per
    // class in first-appearance order defines the head variables.
    let n_slots = 2 * scans.len();
    let mut parent: Vec<usize> = (0..n_slots).collect();
    for (a, b) in &key_pairs {
        if *a >= n_slots || *b >= n_slots {
            return false;
        }
        uf_union_clique(&mut parent, *a, *b);
    }
    let mut class_roots: Vec<usize> = Vec::new();
    let mut first_slot_of_class: HashMap<usize, usize> = HashMap::new();
    for slot in 0..n_slots {
        let root = uf_find_clique(&mut parent, slot);
        if !first_slot_of_class.contains_key(&root) {
            first_slot_of_class.insert(root, slot);
            class_roots.push(root);
        }
    }
    if class_roots.len() != k {
        return false;
    }
    // The clique matcher's orientation invariant requires every atom's
    // col0 class to precede its col1 class in head order. The class
    // digraph of a canonical complete-K_k body (one atom per unordered
    // pair, consistent orientation) is a transitive tournament, whose
    // unique topological order sorts classes by in-degree: head index h
    // = the class with in-degree h. Duplicate in-degrees mean the
    // orientation is not transitive (the top-level matcher would reject
    // it too) — leave the rule untouched. NOTE: first-appearance slot
    // order is NOT usable here because the lowerer's bushy DP plan
    // reorders the scan leaves.
    let class_idx: HashMap<usize, usize> = class_roots
        .iter()
        .enumerate()
        .map(|(idx, root)| (*root, idx))
        .collect();
    let mut in_degree = vec![0usize; k];
    for atom in 0..scans.len() {
        let cls_a = uf_find_clique(&mut parent, 2 * atom);
        let cls_b = uf_find_clique(&mut parent, 2 * atom + 1);
        if cls_a == cls_b {
            return false;
        }
        in_degree[class_idx[&cls_b]] += 1;
    }
    let mut class_pos_by_head = vec![usize::MAX; k];
    for (pos, deg) in in_degree.iter().enumerate() {
        if *deg >= k || class_pos_by_head[*deg] != usize::MAX {
            return false;
        }
        class_pos_by_head[*deg] = pos;
    }
    let representative_slots: Vec<usize> = class_pos_by_head
        .iter()
        .map(|pos| first_slot_of_class[&class_roots[*pos]])
        .collect();
    let class_to_head: HashMap<usize, usize> = class_pos_by_head
        .iter()
        .enumerate()
        .map(|(head_idx, pos)| (class_roots[*pos], head_idx))
        .collect();
    let canonical = RirNode::Project {
        input: inner.clone(),
        columns: representative_slots
            .iter()
            .map(|slot| ProjectExpr::Column(*slot))
            .collect(),
    };
    let Some(promoted) = try_promote_clique_k(&canonical, k, stats) else {
        return false;
    };
    // Remap the group projection from join-output (global slot) space
    // into the clique head-variable space via the slot classes.
    let mut remapped: Vec<ProjectExpr> = Vec::with_capacity(group_cols.len());
    for col in group_cols.iter() {
        let ProjectExpr::Column(c) = col else {
            return false;
        };
        if *c >= n_slots {
            return false;
        }
        let root = uf_find_clique(&mut parent, *c);
        let Some(head_idx) = class_to_head.get(&root) else {
            return false;
        };
        remapped.push(ProjectExpr::Column(*head_idx));
    }
    *group_cols = remapped;
    **inner = promoted;
    true
}

/// `[XY, YZ, XZ]` regardless of positional layout. Returns
/// `None` for shape deviations.
fn try_promote_triangle(
    node: &RirNode,
    stats: &StatsManager,
    config: &CompilerConfig,
) -> Option<RirNode> {
    let RirNode::Project {
        input: outer_input,
        columns,
    } = node
    else {
        return None;
    };
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
    let RirNode::Scan { rel: rel_third } = r1.as_ref() else {
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
    let RirNode::Scan { rel: rel_inner_l } = l2.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: rel_inner_r } = r2.as_ref() else {
        return None;
    };

    // Variable-graph deduction of which atom is XY / YZ / XZ.
    let (rel_xy, rel_yz, rel_xz) = infer_triangle_semantics(
        *rel_inner_l,
        *rel_inner_r,
        *rel_third,
        lk2,
        rk2,
        lk1,
        rk1,
        columns,
    )?;

    let inputs = vec![
        RirNode::Scan { rel: rel_xy },
        RirNode::Scan { rel: rel_yz },
        RirNode::Scan { rel: rel_xz },
    ];
    // Canonical triangle slot_vars: [[V_X, V_Y], [V_Y, V_Z],
    // [V_X, V_Z]] with V_X=0, V_Y=1, V_Z=2. Shape-fixed
    // regardless of the body's positional layout.
    let slot_vars = vec![
        vec![Some(0u32), Some(1)],
        vec![Some(1u32), Some(2)],
        vec![Some(0u32), Some(2)],
    ];
    let output_columns = columns.clone();
    let fallback = Box::new(node.clone());
    // Optional variable ordering: dispatch to the cost model selected by
    // `config.wcoj_variable_ordering`. With
    // `CompilerConfig::default()` (Disabled), no cost model
    // runs and default promotion behavior is bit-identical.
    use crate::compiler_config::WcojVarOrderingKind;
    use crate::wcoj_var_ordering::{
        build_triangle_var_order, HeatAwareLeaderModel, LeaderCardinalityModel,
    };
    let leader_idx = match config.wcoj_variable_ordering {
        WcojVarOrderingKind::Disabled => None,
        WcojVarOrderingKind::LeaderCardinality => {
            LeaderCardinalityModel.pick_triangle_leader([rel_xy, rel_yz, rel_xz], stats, config)
        }
        WcojVarOrderingKind::HeatAware => {
            HeatAwareLeaderModel.pick_triangle_leader([rel_xy, rel_yz, rel_xz], stats, config)
        }
    };
    let var_order = leader_idx.map(build_triangle_var_order);
    Some(RirNode::MultiWayJoin {
        inputs,
        slot_vars,
        output_columns,
        fallback,
        plan: None,
        var_order,
    })
}

/// Recognize a 2-atom inner chain
/// `Project(Join(Scan, Scan))` with exactly one shared key column
/// and wrap it as a production `ChainJoin`.
fn try_promote_chain(node: &RirNode) -> Option<RirNode> {
    let RirNode::Project { input, columns } = node else {
        return None;
    };
    let RirNode::Join {
        left,
        right,
        left_keys,
        right_keys,
        join_type,
    } = input.as_ref()
    else {
        return None;
    };
    if !matches!(join_type, JoinType::Inner) {
        return None;
    }
    if left_keys.len() != 1 || right_keys.len() != 1 {
        return None;
    }
    let left_key = left_keys[0];
    let right_key = right_keys[0];
    if left_key >= 2 || right_key >= 2 {
        return None;
    }
    let RirNode::Scan { rel: rel_left } = left.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: rel_right } = right.as_ref() else {
        return None;
    };

    Some(RirNode::ChainJoin {
        left: Box::new(RirNode::Scan { rel: *rel_left }),
        right: Box::new(RirNode::Scan { rel: *rel_right }),
        left_key,
        right_key,
        output_columns: columns.clone(),
        fallback: Box::new(node.clone()),
    })
}

/// 4-cycle semantic-slot encoding: 4-cycle has 4 atoms × 2 cols = 8 slots. Encode as
/// `(atom_idx * 2) + col_idx` where `atom_idx` is `0` =
/// outer-left's left, `1` = outer-left's right, `2` =
/// outer-right's left, `3` = outer-right's right.
fn ac_idx_4(atom_idx: u8, col_idx: u8) -> u8 {
    debug_assert!(atom_idx < 4);
    debug_assert!(col_idx < 2);
    atom_idx * 2 + col_idx
}

fn outer_left_inner_output_ac(k: usize) -> Option<(u8, u8)> {
    match k {
        0 => Some((0, 0)),
        1 => Some((0, 1)),
        2 => Some((1, 0)),
        3 => Some((1, 1)),
        _ => None,
    }
}

fn outer_right_inner_output_ac(k: usize) -> Option<(u8, u8)> {
    match k {
        0 => Some((2, 0)),
        1 => Some((2, 1)),
        2 => Some((3, 0)),
        3 => Some((3, 1)),
        _ => None,
    }
}

fn outer_4cycle_output_ac(k: usize) -> Option<(u8, u8)> {
    match k {
        0..=3 => outer_left_inner_output_ac(k),
        4..=7 => outer_right_inner_output_ac(k - 4),
        _ => None,
    }
}

fn uf_find_8(parent: &mut [u8; 8], x: u8) -> u8 {
    let mut root = x;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    let mut cur = x;
    while parent[cur as usize] != root {
        let next = parent[cur as usize];
        parent[cur as usize] = root;
        cur = next;
    }
    root
}

fn uf_union_8(parent: &mut [u8; 8], a: u8, b: u8) {
    let ra = uf_find_8(parent, a);
    let rb = uf_find_8(parent, b);
    if ra != rb {
        parent[rb as usize] = ra;
    }
}

/// Semantic-slot inference for 4-cycle bodies. Given
/// the four scans + key shapes + project columns, deduce
/// which atom is the WX-edge / XY-edge / YZ-edge / ZW-edge
/// from the variable-equivalence graph. Returns
/// `(rel_wx, rel_xy, rel_yz, rel_zw)` regardless of the
/// body's positional layout.
#[allow(clippy::too_many_arguments)]
fn infer_4cycle_semantics(
    rel_ll: RelId,
    rel_lr: RelId,
    rel_rl: RelId,
    rel_rr: RelId,
    ilk_l: &[usize],
    irk_l: &[usize],
    ilk_r: &[usize],
    irk_r: &[usize],
    olk: &[usize],
    ork: &[usize],
    project_cols: &[ProjectExpr],
) -> Option<(RelId, RelId, RelId, RelId)> {
    if ilk_l.len() != 1 || irk_l.len() != 1 {
        return None;
    }
    if ilk_r.len() != 1 || irk_r.len() != 1 {
        return None;
    }
    if olk.len() != 2 || ork.len() != 2 {
        return None;
    }
    if project_cols.len() != 4 {
        return None;
    }
    if ilk_l[0] >= 2 || irk_l[0] >= 2 || ilk_r[0] >= 2 || irk_r[0] >= 2 {
        return None;
    }
    if olk.iter().any(|k| *k >= 4) || ork.iter().any(|k| *k >= 4) {
        return None;
    }

    let mut parent = [0u8, 1, 2, 3, 4, 5, 6, 7];

    // Outer-left inner: ll.col[ilk_l[0]] ≡ lr.col[irk_l[0]].
    uf_union_8(
        &mut parent,
        ac_idx_4(0, ilk_l[0] as u8),
        ac_idx_4(1, irk_l[0] as u8),
    );

    // Outer-right inner: rl.col[ilk_r[0]] ≡ rr.col[irk_r[0]].
    uf_union_8(
        &mut parent,
        ac_idx_4(2, ilk_r[0] as u8),
        ac_idx_4(3, irk_r[0] as u8),
    );

    // Outer join: 2 equivalences from (olk, ork). olk indexes
    // outer-left's inner output (cols 0..4); ork indexes
    // outer-right's inner output (cols 0..4).
    for i in 0..2 {
        let (la, lc) = outer_left_inner_output_ac(olk[i])?;
        let (ra, rc) = outer_right_inner_output_ac(ork[i])?;
        uf_union_8(&mut parent, ac_idx_4(la, lc), ac_idx_4(ra, rc));
    }

    let roots: [u8; 8] = std::array::from_fn(|i| uf_find_8(&mut parent, i as u8));
    let mut counts: HashMap<u8, u8> = HashMap::new();
    for r in &roots {
        *counts.entry(*r).or_insert(0) += 1;
    }
    if counts.len() != 4 || counts.values().any(|c| *c != 2) {
        return None;
    }

    // Map head columns to equivalence classes.
    let mut head_classes: [u8; 4] = [0; 4];
    for (i, pc) in project_cols.iter().enumerate() {
        let ProjectExpr::Column(k) = pc else {
            return None;
        };
        let (atom, col) = outer_4cycle_output_ac(*k)?;
        head_classes[i] = uf_find_8(&mut parent, ac_idx_4(atom, col));
    }
    // Head must pick 4 distinct classes.
    for i in 0..4 {
        for j in (i + 1)..4 {
            if head_classes[i] == head_classes[j] {
                return None;
            }
        }
    }
    let w_class = head_classes[0];
    let x_class = head_classes[1];
    let y_class = head_classes[2];
    let z_class = head_classes[3];

    let atom_classes = |atom_idx: u8| -> (u8, u8) {
        (
            roots[ac_idx_4(atom_idx, 0) as usize],
            roots[ac_idx_4(atom_idx, 1) as usize],
        )
    };

    let atom_rels = [rel_ll, rel_lr, rel_rl, rel_rr];
    let mut rel_wx: Option<RelId> = None;
    let mut rel_xy: Option<RelId> = None;
    let mut rel_yz: Option<RelId> = None;
    let mut rel_zw: Option<RelId> = None;

    for atom_idx in 0..4u8 {
        let (c0, c1) = atom_classes(atom_idx);
        let binds_w = c0 == w_class || c1 == w_class;
        let binds_x = c0 == x_class || c1 == x_class;
        let binds_y = c0 == y_class || c1 == y_class;
        let binds_z = c0 == z_class || c1 == z_class;
        match (binds_w, binds_x, binds_y, binds_z) {
            (true, true, false, false) => rel_wx = Some(atom_rels[atom_idx as usize]),
            (false, true, true, false) => rel_xy = Some(atom_rels[atom_idx as usize]),
            (false, false, true, true) => rel_yz = Some(atom_rels[atom_idx as usize]),
            (true, false, false, true) => rel_zw = Some(atom_rels[atom_idx as usize]),
            _ => return None,
        }
    }

    Some((rel_wx?, rel_xy?, rel_yz?, rel_zw?))
}

/// Recognize the canonical 4-cycle subtree and
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
/// Returns `None` for any deviation. Strict by design: only
/// matchers/promoters with explicit shape qualifiers may require an
/// exact shape, and this matcher requires 4-cycle specifically.
fn try_promote_4cycle(
    node: &RirNode,
    stats: &StatsManager,
    config: &CompilerConfig,
) -> Option<RirNode> {
    let RirNode::Project {
        input: outer_input,
        columns,
    } = node
    else {
        return None;
    };
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
    let RirNode::Scan { rel: rel_ll } = ll.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: rel_lr } = lr.as_ref() else {
        return None;
    };
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
    let RirNode::Scan { rel: rel_rl } = rl.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: rel_rr } = rr.as_ref() else {
        return None;
    };

    // Variable-graph deduction of which atom is
    // WX / XY / YZ / ZW.
    let (rel_wx, rel_xy, rel_yz, rel_zw) = infer_4cycle_semantics(
        *rel_ll, *rel_lr, *rel_rl, *rel_rr, ilk_l, irk_l, ilk_r, irk_r, olk, ork, columns,
    )?;

    let inputs = vec![
        RirNode::Scan { rel: rel_wx },
        RirNode::Scan { rel: rel_xy },
        RirNode::Scan { rel: rel_yz },
        RirNode::Scan { rel: rel_zw },
    ];
    // Shape-fixed canonical 4-cycle slot_vars:
    // [[V_W, V_X], [V_X, V_Y], [V_Y, V_Z], [V_Z, V_W]]
    // with V_W=0, V_X=1, V_Y=2, V_Z=3. Encoded in head-label
    // space (head[0]=W=0, head[1]=X=1, head[2]=Y=2, head[3]=Z=3).
    let slot_vars = vec![
        vec![Some(0u32), Some(1)],
        vec![Some(1u32), Some(2)],
        vec![Some(2u32), Some(3)],
        vec![Some(3u32), Some(0)],
    ];
    let output_columns = columns.clone();
    let fallback = Box::new(node.clone());
    // Ask the cost model whether to set a non-default
    // leader. With `CompilerConfig::default()` (Disabled), this
    // always returns None.
    use crate::compiler_config::WcojVarOrderingKind;
    use crate::wcoj_var_ordering::{
        build_cycle4_var_order, HeatAwareLeaderModel, LeaderCardinalityModel,
    };
    let leader_idx_4 = match config.wcoj_variable_ordering {
        WcojVarOrderingKind::Disabled => None,
        WcojVarOrderingKind::LeaderCardinality => LeaderCardinalityModel.pick_4cycle_leader(
            [rel_wx, rel_xy, rel_yz, rel_zw],
            stats,
            config,
        ),
        WcojVarOrderingKind::HeatAware => {
            HeatAwareLeaderModel.pick_4cycle_leader([rel_wx, rel_xy, rel_yz, rel_zw], stats, config)
        }
    };
    let var_order = leader_idx_4.map(build_cycle4_var_order);
    Some(RirNode::MultiWayJoin {
        inputs,
        slot_vars,
        output_columns,
        fallback,
        plan: None,
        var_order,
    })
}

// ===============================================================
// K-clique promoter (k = 5..8).
//
// Tree-flatten + complete-K_k validation. Robust to left-deep /
// right-deep / bushy lowered trees. Rejects:
//   * Filter / comparison wrappers (semantic preservation gate).
//   * Non-canonical nodes (anything other than Project/Join/Scan).
//   * Self-edge atoms (e(X, X) — same var in both columns).
//   * Reversed atoms (e(v_j, v_i) for canonical (v_i, v_j) with
//     i < j) — this promoter does not implement column-swap layout
//     for clique edges.
//   * Constants in atom positions.
//   * Recursive scan bodies are admitted for K-clique
//     metadata refresh during semi-naive fixpoint.
//   * Atom multisets that don't form the complete K_k edge set.
// ===============================================================

/// Canonical edge index for (i, j) with 0 <= i < j < k.
fn clique_edge_idx(i: usize, j: usize, k: usize) -> usize {
    debug_assert!(i < j && j < k);
    i * (2 * k - i - 1) / 2 + (j - i - 1)
}

/// Tiny union-find on atom-column slots (position space).
fn uf_find_clique(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn uf_union_clique(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find_clique(parent, a);
    let rb = uf_find_clique(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

/// Flatten the join tree. Walks Project/Join/Scan only;
/// rejects on Filter or any other RIR variant. Returns
/// `(scans, key_equiv_pairs, project_columns)` on success.
///
/// `scans[i]` = RelId of the i-th leaf Scan in left-to-right
/// traversal order. The "global slot space" assigns slot indices
/// `2*i` and `2*i + 1` to atom i's col0 and col1 respectively.
///
/// `key_equiv_pairs` is a list of `(global_slot_a, global_slot_b)`
/// pairs derived from each Join's `(left_keys, right_keys)` —
/// every Join key references LOCAL join output positions, which
/// the flatten translates to global slot positions.
///
/// `project_columns` is the outermost Project's column list,
/// each entry's index translated to a global slot position.
#[allow(clippy::type_complexity)]
fn flatten_clique_body(body: &RirNode) -> Option<(Vec<RelId>, Vec<(usize, usize)>, Vec<usize>)> {
    let RirNode::Project { input, columns } = body else {
        return None;
    };
    let mut scans: Vec<RelId> = Vec::new();
    let mut key_pairs: Vec<(usize, usize)> = Vec::new();
    let _width = walk_clique_node(input, &mut scans, &mut key_pairs)?;
    let mut project_globals: Vec<usize> = Vec::with_capacity(columns.len());
    for c in columns {
        let xlog_ir::rir::ProjectExpr::Column(k) = c else {
            return None;
        };
        project_globals.push(*k);
    }
    Some((scans, key_pairs, project_globals))
}

/// Recursive walker. Returns the width (in global slots) of the
/// subtree, after accumulating its scans + key-equiv pairs.
fn walk_clique_node(
    node: &RirNode,
    scans: &mut Vec<RelId>,
    key_pairs: &mut Vec<(usize, usize)>,
) -> Option<usize> {
    match node {
        RirNode::Scan { rel } => {
            scans.push(*rel);
            Some(2)
        }
        RirNode::Join {
            left,
            right,
            left_keys,
            right_keys,
            join_type,
        } => {
            if !matches!(join_type, JoinType::Inner) {
                return None;
            }
            let left_offset = scans.len() * 2;
            let left_width = walk_clique_node(left, scans, key_pairs)?;
            let right_offset = left_offset + left_width;
            let right_width = walk_clique_node(right, scans, key_pairs)?;
            if left_keys.len() != right_keys.len() {
                return None;
            }
            for (lk, rk) in left_keys.iter().zip(right_keys.iter()) {
                if *lk >= left_width || *rk >= right_width {
                    return None;
                }
                key_pairs.push((left_offset + *lk, right_offset + *rk));
            }
            Some(left_width + right_width)
        }
        // Reject Project (only outermost is allowed; we already
        // peeled it off in flatten_clique_body), Filter, and
        // any other RIR variant.
        _ => None,
    }
}

/// Generic Free Join — general multiway flatten walker. Arity-aware
/// sibling of [`walk_clique_node`]: sizes each Scan leaf from
/// `arities` instead of the clique walker's hardcoded arity-2
/// assumption, so global slot offsets are running width sums.
/// Walks Join/Scan only; rejects non-Inner joins, keyless
/// (Cartesian) joins — those stay on the bench-grounded nested-loop
/// routing — and any other RIR variant. Returns the
/// subtree width in global slots.
fn walk_general_node(
    node: &RirNode,
    arities: &HashMap<RelId, usize>,
    scans: &mut Vec<RelId>,
    widths: &mut Vec<usize>,
    key_pairs: &mut Vec<(usize, usize)>,
) -> Option<usize> {
    match node {
        RirNode::Scan { rel } => {
            let width = *arities.get(rel)?;
            if width == 0 {
                return None;
            }
            scans.push(*rel);
            widths.push(width);
            Some(width)
        }
        RirNode::Join {
            left,
            right,
            left_keys,
            right_keys,
            join_type,
        } => {
            if !matches!(join_type, JoinType::Inner) {
                return None;
            }
            // Offset of the left subtree = total width accumulated so
            // far (widths fills in left-to-right traversal order).
            let left_offset: usize = widths.iter().sum();
            let left_width = walk_general_node(left, arities, scans, widths, key_pairs)?;
            let right_offset = left_offset + left_width;
            let right_width = walk_general_node(right, arities, scans, widths, key_pairs)?;
            if left_keys.len() != right_keys.len() || left_keys.is_empty() {
                return None;
            }
            for (lk, rk) in left_keys.iter().zip(right_keys.iter()) {
                if *lk >= left_width || *rk >= right_width {
                    return None;
                }
                key_pairs.push((left_offset + *lk, right_offset + *rk));
            }
            Some(left_width + right_width)
        }
        _ => None,
    }
}

/// Generic Free Join — promote a general ≥3-atom inner-join body (any
/// arity mix, any join-tree shape) to a generic `MultiWayJoin`
/// carrying dense variable classes in `slot_vars` and `plan: None`.
///
/// Runs after every dedicated shape promoter declined in the
/// per-rule loop (triangle / 4-cycle / K-clique successes `continue`
/// past it), so dedicated shapes never reach this path. The executor
/// (`try_dispatch_free_join`) derives a Free Join plan from
/// `slot_vars` at dispatch time and silently declines to the
/// embedded binary fallback for shapes the GPU engine cannot run.
///
/// Idempotent: only `Project(inner-join tree)` bodies match, so an
/// already-promoted `MultiWayJoin` passes through unchanged.
fn try_promote_general_multiway(
    body: &RirNode,
    arities: &HashMap<RelId, usize>,
) -> Option<RirNode> {
    let RirNode::Project { input, columns } = body else {
        return None;
    };
    let mut scans: Vec<RelId> = Vec::new();
    let mut widths: Vec<usize> = Vec::new();
    let mut key_pairs: Vec<(usize, usize)> = Vec::new();
    let total = walk_general_node(input, arities, &mut scans, &mut widths, &mut key_pairs)?;
    if scans.len() < 3 {
        return None;
    }

    // Union-find over the global slot space, then dense class ids in
    // first-occurrence slot order. The executor treats `slot_vars`
    // entries as opaque variable classes (it densely remaps again on
    // its side), so first-occurrence numbering is canonical enough.
    let mut parent: Vec<usize> = (0..total).collect();
    for (a, b) in &key_pairs {
        if *a >= total || *b >= total {
            return None;
        }
        uf_union_clique(&mut parent, *a, *b);
    }
    let mut class_of_root: HashMap<usize, u32> = HashMap::new();
    let mut slot_class: Vec<u32> = Vec::with_capacity(total);
    for slot in 0..total {
        let root = uf_find_clique(&mut parent, slot);
        let next = class_of_root.len() as u32;
        let cls = *class_of_root.entry(root).or_insert(next);
        slot_class.push(cls);
    }

    // Every projection entry must be a plain column inside the slot
    // space; computed projections stay on the binary path.
    for c in columns {
        let ProjectExpr::Column(k) = c else {
            return None;
        };
        if *k >= total {
            return None;
        }
    }

    let inputs: Vec<RirNode> = scans
        .iter()
        .map(|rel| RirNode::Scan { rel: *rel })
        .collect();
    let mut slot_vars: Vec<Vec<Option<u32>>> = Vec::with_capacity(scans.len());
    let mut offset = 0usize;
    for &w in &widths {
        slot_vars.push((offset..offset + w).map(|s| Some(slot_class[s])).collect());
        offset += w;
    }

    Some(RirNode::MultiWayJoin {
        inputs,
        slot_vars,
        output_columns: columns.clone(),
        fallback: Box::new(body.clone()),
        // Provenance marker: `inputs` are the fallback's Scan leaves
        // in traversal order, so `output_columns` coincides with the
        // concatenated-inputs column space — the contract the Free
        // Join dispatcher requires (dedicated promoters reorder
        // `inputs` canonically and must never carry this variant).
        plan: Some(MultiwayPlan::FreeJoin),
        var_order: None,
    })
}

/// Generic multiway aggregate group-by fusion: general multiway sibling of
/// [`try_promote_triangle_inside_aggregate`]. Promotes any >=3-atom
/// inner-join tree sitting inside an aggregate rule's
/// `Project{ GroupBy { Project { <join tree> } } }` wrapper through
/// [`try_promote_general_multiway`].
///
/// Unlike the dedicated descents, no column remapping gymnastics are
/// needed: FreeJoin-marked nodes keep `output_columns` in the
/// join-output column space (inputs are the tree's Scan leaves in
/// traversal order), so the group projection's columns are handed to
/// the promoter as the head projection verbatim and the group
/// projection becomes the identity over the MultiWayJoin's output.
/// Returns false — leaving the rule untouched — on any structural
/// mismatch.
fn try_promote_general_multiway_inside_aggregate(
    body: &mut RirNode,
    arities: &HashMap<RelId, usize>,
) -> bool {
    let RirNode::Project { input: gb, .. } = body else {
        return false;
    };
    let RirNode::GroupBy {
        input: group_input, ..
    } = gb.as_mut()
    else {
        return false;
    };
    let RirNode::Project {
        input: inner,
        columns: group_cols,
    } = group_input.as_mut()
    else {
        return false;
    };
    let candidate = RirNode::Project {
        input: inner.clone(),
        columns: group_cols.clone(),
    };
    let Some(promoted) = try_promote_general_multiway(&candidate, arities) else {
        return false;
    };
    // The MultiWayJoin's output IS the group projection, so the group
    // projection becomes positional identity over it.
    *group_cols = (0..group_cols.len()).map(ProjectExpr::Column).collect();
    **inner = promoted;
    true
}

/// K-clique promoter for k ∈ {5, 6, 7, 8}.
///
/// Uses tree-flatten + complete-K_k validation. Robust to left-deep
/// / right-deep / bushy.
/// Rejects filter wrappers, reversed atoms, self-edges,
/// constants, and any non-canonical shape; admits recursive
/// bodies so runtime metadata refresh can observe each merge.
fn try_promote_clique_k(body: &RirNode, k: usize, stats: &StatsManager) -> Option<RirNode> {
    if !(5..=8).contains(&k) {
        return None;
    }
    let expected_edges = k * (k - 1) / 2;

    // 1. Flatten body. Rejects Filter, non-canonical nodes.
    let (scans, key_pairs, project_globals) = flatten_clique_body(body)?;

    // 2. Scan count must equal C(k, 2).
    if scans.len() != expected_edges {
        return None;
    }

    // 3. Head must have exactly k variables.
    if project_globals.len() != k {
        return None;
    }

    // 4. Union-find on global slot space (size 2 * expected_edges)
    // to derive variable equivalence classes.
    let n_slots = 2 * expected_edges;
    let mut parent: Vec<usize> = (0..n_slots).collect();
    for (a, b) in &key_pairs {
        if *a >= n_slots || *b >= n_slots {
            return None;
        }
        uf_union_clique(&mut parent, *a, *b);
    }

    // 5. Find unique class representative per Project column.
    // Each Project column references a global slot; resolve to
    // its class root.
    let mut head_class: Vec<usize> = Vec::with_capacity(k);
    for col in &project_globals {
        if *col >= n_slots {
            return None;
        }
        head_class.push(uf_find_clique(&mut parent, *col));
    }
    // 6. The k head classes must be distinct.
    let mut sorted_head_classes = head_class.clone();
    sorted_head_classes.sort();
    sorted_head_classes.dedup();
    if sorted_head_classes.len() != k {
        return None;
    }

    // 7. Total distinct classes across all slots must equal k.
    // (No "extra" non-head variables in atom slots — every slot's
    // class must be one of the k head classes.)
    let mut all_class_count: HashMap<usize, usize> = HashMap::new();
    for slot in 0..n_slots {
        let root = uf_find_clique(&mut parent, slot);
        *all_class_count.entry(root).or_insert(0) += 1;
    }
    if all_class_count.len() != k {
        return None;
    }
    // 8. Every class must have exactly k-1 slots (each variable
    // appears in exactly k-1 atoms = clique edges incident to it).
    for &count in all_class_count.values() {
        if count != k - 1 {
            return None;
        }
    }

    // 9. Map class → head-var index. The Project column ordering
    // defines the head-var indices (Project col i = head var i).
    let mut class_to_head_idx: HashMap<usize, usize> = HashMap::new();
    for (head_idx, cls) in head_class.iter().enumerate() {
        class_to_head_idx.insert(*cls, head_idx);
    }

    // 10. For each scan, derive its (var_a, var_b) pair from the
    // class memberships of its two slots. Reject self-edges
    // (both slots in same class) and atoms touching a non-head
    // class (already filtered above by class-count check, but
    // defensive).
    let mut atom_pairs: Vec<(usize, usize)> = Vec::with_capacity(expected_edges);
    let mut canonical_to_scan_idx: HashMap<(usize, usize), usize> = HashMap::new();
    for (atom_i, _rel) in scans.iter().enumerate() {
        let slot_a = 2 * atom_i;
        let slot_b = 2 * atom_i + 1;
        let cls_a = uf_find_clique(&mut parent, slot_a);
        let cls_b = uf_find_clique(&mut parent, slot_b);
        if cls_a == cls_b {
            // Self-edge e(X, X) — reject.
            return None;
        }
        let head_a = class_to_head_idx.get(&cls_a)?;
        let head_b = class_to_head_idx.get(&cls_b)?;
        // 11. Reversed-atom rejection: canonical form requires
        // col0 maps to lower head idx, col1 to higher. If col0
        // is at higher idx, the atom is reversed (this promoter does
        // not implement column-swap layout for clique edges).
        if *head_a > *head_b {
            return None;
        }
        let (lo, hi) = (*head_a, *head_b);
        atom_pairs.push((lo, hi));
        // 12. Reject duplicate edges (same (i, j) appearing twice).
        if canonical_to_scan_idx.insert((lo, hi), atom_i).is_some() {
            return None;
        }
    }

    // 13. Verify the atom_pairs set equals the complete K_k edge
    // set {(i, j) | 0 <= i < j < k}.
    if canonical_to_scan_idx.len() != expected_edges {
        return None;
    }
    for i in 0..k {
        for j in (i + 1)..k {
            if !canonical_to_scan_idx.contains_key(&(i, j)) {
                return None;
            }
        }
    }

    // 14. Reorder scans into canonical lex (i, j) order.
    let mut reordered_scans: Vec<RelId> = Vec::with_capacity(expected_edges);
    for i in 0..k {
        for j in (i + 1)..k {
            let scan_idx = canonical_to_scan_idx[&(i, j)];
            reordered_scans.push(scans[scan_idx]);
        }
    }

    // 15. Build canonical MultiWayJoin.inputs (one Scan per
    // canonical edge in lex order) + slot_vars (each atom's
    // (col0, col1) bind (head_i, head_j) for canonical edge
    // (i, j)).
    let inputs: Vec<RirNode> = reordered_scans
        .iter()
        .map(|rel| RirNode::Scan { rel: *rel })
        .collect();
    let mut slot_vars: Vec<Vec<Option<u32>>> = Vec::with_capacity(expected_edges);
    for i in 0..k {
        for j in (i + 1)..k {
            let _ = clique_edge_idx(i, j, k); // sanity; locked invariant
            slot_vars.push(vec![Some(i as u32), Some(j as u32)]);
        }
    }
    // Project's existing column list defines the head-output
    // mapping; we preserve it on the MultiWayJoin's
    // output_columns.
    let RirNode::Project { columns, .. } = body else {
        return None;
    };
    let output_columns = columns.clone();
    let fallback = Box::new(body.clone());
    let shape = build_kclique_shape(k, &reordered_scans)?;
    let planner_stats = kclique_planner_stats(stats);

    // Cost-planned K-clique routing follows the arXiv paper's
    // conditional-win-on-skew caveat: recognized paper-aligned shapes
    // emit a positive route. Hash is represented by
    // `PlannedHashRoute`, never by a post-recognition raw decline.
    let (plan, var_order) = match plan_kclique_var_order(&shape, &planner_stats) {
        Some(full_order) => {
            let evidence = rir_cost_prediction(&full_order);
            if wcoj_cost_gate_predicts_wcoj(evidence.wcoj_cost, evidence.hash_cost) {
                let kclique_order = kclique_variable_order_from_plan(&shape, &full_order)?;
                (
                    MultiwayPlan::WcojWithPlan(kclique_order.clone()),
                    Some(VariableOrder::kclique(kclique_order)),
                )
            } else {
                (
                    MultiwayPlan::PlannedHashRoute {
                        reason: PlannedHashReason::PlannerPredictsHashWins,
                        planner_evidence: evidence,
                    },
                    None,
                )
            }
        }
        None => (
            MultiwayPlan::PlannedHashRoute {
                reason: PlannedHashReason::IncompleteStatsSafeDefault,
                planner_evidence: RirCostPredictionRecord::empty(),
            },
            None,
        ),
    };

    Some(RirNode::MultiWayJoin {
        inputs,
        slot_vars,
        output_columns,
        fallback,
        plan: Some(plan),
        var_order,
    })
}

fn build_kclique_shape(k: usize, rels: &[RelId]) -> Option<KCliqueShape> {
    let mut edges = Vec::with_capacity(rels.len());
    let mut idx = 0usize;
    for i in 0..k {
        for j in (i + 1)..k {
            let rel_id = *rels.get(idx)?;
            edges.push(KCliqueEdge {
                rel_id,
                left: VertexId(i),
                right: VertexId(j),
                left_col: 0,
                right_col: 1,
            });
            idx += 1;
        }
    }
    KCliqueShape::from_edges(k as u8, edges)
}

fn kclique_planner_stats(stats: &StatsManager) -> StatsSnapshot {
    stats.snapshot()
}

fn rir_cost_prediction(plan: &FullVariableOrder) -> RirCostPredictionRecord {
    RirCostPredictionRecord {
        wcoj_cost: plan.cost_prediction.wcoj_cost,
        hash_cost: plan.cost_prediction.hash_cost,
    }
}

fn kclique_variable_order_from_plan(
    shape: &KCliqueShape,
    plan: &FullVariableOrder,
) -> Option<KCliqueVariableOrder> {
    let k = shape.variable_count();
    let expected_edges = usize::from(k) * usize::from(k - 1) / 2;
    if plan.variable_order.len() != usize::from(k) || plan.edge_permutation.len() != expected_edges
    {
        return None;
    }

    let mut variable_positions = [u8::MAX; K_CLIQUE_MAX_K];
    for (position, variable) in plan.variable_order.iter().enumerate() {
        if variable.0 >= usize::from(k) {
            return None;
        }
        variable_positions[variable.0] = position as u8;
    }

    let mut edge_permutation = [u8::MAX; K_CLIQUE_MAX_EDGES];
    let mut column_swaps = Vec::new();
    let mut leader_slot = None;
    for (slot, edge_idx) in plan.edge_permutation.iter().copied().enumerate() {
        let edge = shape.edges().get(edge_idx)?;
        let left_pos = variable_positions[edge.left.0];
        let right_pos = variable_positions[edge.right.0];
        if left_pos == u8::MAX || right_pos == u8::MAX {
            return None;
        }
        edge_permutation[slot] = edge_idx as u8;
        if left_pos > right_pos {
            column_swaps.push(ColumnSwap {
                edge_slot: slot as u8,
                swap_cols: true,
            });
        }
        if [left_pos, right_pos].into_iter().min() == Some(0)
            && [left_pos, right_pos].into_iter().max() == Some(1)
        {
            leader_slot = Some(slot as u8);
        }
    }

    let sorted_edge_slots = vec![leader_slot.unwrap_or(0)];
    let sorted_layout_requirements = SortedLayoutSpec {
        edge_slots: sorted_edge_slots,
        key_columns: vec![vec![0, 1]],
    };

    // Helper-relation splitting elevates buried inner-variable skew.
    let helper_split_specs = plan.helper_split_specs.clone();
    Some(KCliqueVariableOrder::new(
        k,
        variable_positions,
        edge_permutation,
        column_swaps,
        sorted_layout_requirements,
        helper_split_specs,
        StreamGroupId(0),
    ))
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

    fn canonical_chain_tree() -> RirNode {
        let join = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        RirNode::Project {
            input: Box::new(join),
            columns: vec![ProjectExpr::Column(0), ProjectExpr::Column(3)],
        }
    }

    #[test]
    fn promotes_canonical_chain() {
        let mut plan = plan_with_body(canonical_chain_tree());
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let body = &plan.rules_by_scc[0][0].body;
        match body {
            RirNode::ChainJoin {
                left,
                right,
                left_key,
                right_key,
                output_columns,
                fallback,
            } => {
                assert!(matches!(left.as_ref(), RirNode::Scan { rel: RelId(1) }));
                assert!(matches!(right.as_ref(), RirNode::Scan { rel: RelId(2) }));
                assert_eq!(*left_key, 1);
                assert_eq!(*right_key, 0);
                assert_eq!(
                    output_columns,
                    &vec![ProjectExpr::Column(0), ProjectExpr::Column(3)]
                );
                assert!(matches!(fallback.as_ref(), RirNode::Project { .. }));
            }
            other => panic!("expected ChainJoin, got {:?}", other),
        }
    }

    #[test]
    fn chain_promotion_rejects_non_inner_join() {
        let mut body = canonical_chain_tree();
        if let RirNode::Project { input, .. } = &mut body {
            if let RirNode::Join { join_type, .. } = input.as_mut() {
                *join_type = JoinType::LeftOuter;
            }
        }
        assert!(try_promote_chain(&body).is_none());
    }

    #[test]
    fn chain_promotion_rejects_multi_key_join() {
        let mut body = canonical_chain_tree();
        if let RirNode::Project { input, .. } = &mut body {
            if let RirNode::Join {
                left_keys,
                right_keys,
                ..
            } = input.as_mut()
            {
                *left_keys = vec![0, 1];
                *right_keys = vec![0, 1];
            }
        }
        assert!(try_promote_chain(&body).is_none());
    }

    #[test]
    fn promotes_canonical_triangle() {
        let mut plan = plan_with_body(canonical_triangle_tree());
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let body = &plan.rules_by_scc[0][0].body;
        match body {
            RirNode::MultiWayJoin {
                inputs,
                slot_vars,
                output_columns,
                fallback,
                var_order: _,
                ..
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let first = format!("{:?}", &plan.rules_by_scc[0][0].body);
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let second = format!("{:?}", &plan.rules_by_scc[0][0].body);
        assert_eq!(first, second);
    }

    /// Triangle with X-shared inner pair — inner keys
    /// `[0]/[0]`, outer keys `[1, 3]/[0, 1]`, project `[0, 1, 3]`.
    /// Body atoms at positions: l2 = e_xy, r2 = e_xz, r1 = e_yz.
    /// Promoter must reorder inputs to canonical semantic order
    /// `[XY, YZ, XZ]` = `[RelId(1), RelId(3), RelId(2)]`.
    #[test]
    fn promotes_triangle_with_x_shared_inner_pair() {
        let inner = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }), // e_xy
            right: Box::new(RirNode::Scan { rel: RelId(2) }), // e_xz
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(RirNode::Scan { rel: RelId(3) }), // e_yz
            left_keys: vec![1, 3],
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let RirNode::MultiWayJoin {
            inputs, slot_vars, ..
        } = &plan.rules_by_scc[0][0].body
        else {
            panic!("expected MultiWayJoin after promotion");
        };
        // Semantic-order inputs: [XY, YZ, XZ] = [RelId(1), RelId(3), RelId(2)].
        let scan_rels: Vec<RelId> = inputs
            .iter()
            .map(|n| match n {
                RirNode::Scan { rel } => *rel,
                _ => panic!("expected Scan in MultiWayJoin inputs"),
            })
            .collect();
        assert_eq!(scan_rels, vec![RelId(1), RelId(3), RelId(2)]);
        // Shape-fixed canonical slot_vars.
        assert_eq!(
            slot_vars,
            &vec![
                vec![Some(0u32), Some(1)],
                vec![Some(1u32), Some(2)],
                vec![Some(0u32), Some(2)],
            ]
        );
    }

    /// Triangle with Z-shared inner pair — inner keys
    /// `[1]/[1]`, outer keys `[0, 2]/[0, 1]`, project `[0, 2, 3]`.
    /// Body atoms at positions: l2 = e_xz, r2 = e_yz, r1 = e_xy.
    /// Promoter must reorder inputs to canonical semantic order
    /// `[XY, YZ, XZ]` = `[RelId(3), RelId(2), RelId(1)]`.
    #[test]
    fn promotes_triangle_with_z_shared_inner_pair() {
        let inner = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }), // e_xz
            right: Box::new(RirNode::Scan { rel: RelId(2) }), // e_yz
            left_keys: vec![1],
            right_keys: vec![1],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(RirNode::Scan { rel: RelId(3) }), // e_xy
            left_keys: vec![0, 2],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let body = RirNode::Project {
            input: Box::new(outer),
            // Inner cols: [X, Z, Y, Z]. Outer cols: [X, Z, Y, Z, X, Y].
            // Head (X, Y, Z) = [0, 2, 3].
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(2),
                ProjectExpr::Column(3),
            ],
        };
        let mut plan = plan_with_body(body);
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let RirNode::MultiWayJoin {
            inputs, slot_vars, ..
        } = &plan.rules_by_scc[0][0].body
        else {
            panic!("expected MultiWayJoin after promotion");
        };
        let scan_rels: Vec<RelId> = inputs
            .iter()
            .map(|n| match n {
                RirNode::Scan { rel } => *rel,
                _ => panic!("expected Scan in MultiWayJoin inputs"),
            })
            .collect();
        assert_eq!(scan_rels, vec![RelId(3), RelId(2), RelId(1)]);
        assert_eq!(
            slot_vars,
            &vec![
                vec![Some(0u32), Some(1)],
                vec![Some(1u32), Some(2)],
                vec![Some(0u32), Some(2)],
            ]
        );
    }

    #[test]
    fn promotes_triangle_with_rotated_projection_columns() {
        // The original strict rejection of non-canonical projection
        // columns is intentionally relaxed. The variable-graph
        // promoter recognizes the triangle
        // topology regardless of which head position picks
        // which variable, as long as the 3 head columns pick
        // 3 distinct equivalence classes. The emitted
        // `MultiWayJoin` carries shape-fixed `slot_vars` —
        // [[0,1], [1,2], [0,2]] — which encode the triangle's
        // topology in the head-label space (head[0] = label 0,
        // head[1] = label 1, head[2] = label 2). `output_columns`
        // are preserved verbatim from the body's projection.
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
            // Rotated: head extracts (Y, X, Z) instead of (X, Y, Z).
            // Topology is still a triangle.
            columns: vec![
                ProjectExpr::Column(1),
                ProjectExpr::Column(0),
                ProjectExpr::Column(3),
            ],
        };
        let mut plan = plan_with_body(body);
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        // Body now promoted to MultiWayJoin by the semantic-slot contract.
        let RirNode::MultiWayJoin {
            slot_vars,
            output_columns,
            ..
        } = &plan.rules_by_scc[0][0].body
        else {
            panic!("expected MultiWayJoin after promotion");
        };
        // Shape-fixed slot_vars in head-label space.
        assert_eq!(
            slot_vars,
            &vec![
                vec![Some(0u32), Some(1)],
                vec![Some(1u32), Some(2)],
                vec![Some(0u32), Some(2)],
            ]
        );
        // output_columns preserved verbatim.
        assert_eq!(
            output_columns,
            &vec![
                ProjectExpr::Column(1),
                ProjectExpr::Column(0),
                ProjectExpr::Column(3),
            ]
        );
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::Project { .. }
        ));
    }

    #[test]
    fn rejects_filter_above_outer_join() {
        // An optimizer may insert a Filter between the outer Project
        // and the outer Join. This promoter does not recognize this.
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        assert_eq!(
            format!("{:?}", &plan.rules_by_scc[0][0].meta),
            format!("{:?}", meta_pre),
        );
    }

    // -----------------------------------------------------------
    // Recursive-SCC promotion gates
    // -----------------------------------------------------------

    /// Recursive promotion contract: a stable triangle (zero recursive Scans) in
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Recursive promotion contract: a linear-recursive triangle (exactly one
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
        promote_multiway(
            &mut plan,
            &rel_ids,
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Occurrence-level recursive promotion contract: a recursive SCC
    /// body with ≥ 2 recursive Scans (here: 2 distinct in-SCC
    /// predicates) IS promoted to `MultiWayJoin`. Mark "tri_a" →
    /// RelId(1) and "tri_b" → RelId(2) so two of the three body
    /// Scans count as in-SCC.
    #[test]
    fn promotes_multirec_triangle_in_recursive_scc() {
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
        // Count == 2 ≥ 2 → occurrence-level recursive promotion admits it.
        promote_multiway(
            &mut plan,
            &rel_ids,
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Mixed plan: a recursive SCC with a linear-recursive triangle
    /// (count 1) AND a non-recursive SCC with a stable triangle
    /// (count 0). BOTH get promoted under the recursive promotion contract.
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
        promote_multiway(
            &mut plan,
            &rel_ids,
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
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
    // 4-cycle promotion
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let body = &plan.rules_by_scc[0][0].body;
        match body {
            RirNode::MultiWayJoin {
                inputs,
                slot_vars,
                output_columns,
                fallback,
                var_order: _,
                ..
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let body = &plan.rules_by_scc[0][0].body;
        let RirNode::MultiWayJoin { fallback, .. } = body else {
            panic!("expected MultiWayJoin");
        };
        assert_eq!(format!("{:?}", fallback.as_ref()), format!("{:?}", pre));
    }

    #[test]
    fn idempotent_4cycle_under_repeat_calls() {
        let mut plan = plan_with_4cycle_body(canonical_4cycle_tree());
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let first = format!("{:?}", &plan.rules_by_scc[0][0].body);
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
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
        assert!(
            try_promote_4cycle(&body, &StatsManager::new(), &CompilerConfig::default()).is_none()
        );
    }

    /// 4-cycle with the alternative bushy grouping
    /// `(e2⋈e3) + (e4⋈e1)` — left-inner shares Y, right-inner
    /// shares W. Atoms at positions: ll=e2(X,Y), lr=e3(Y,Z),
    /// rl=e4(Z,W), rr=e1(W,X). Promoter must reorder inputs
    /// to canonical semantic order `[WX, XY, YZ, ZW]` =
    /// `[RelId(1), RelId(2), RelId(3), RelId(4)]` regardless
    /// of positional layout.
    #[test]
    fn promotes_4cycle_with_alternative_inner_grouping() {
        // RelId(1)=e1 (W,X), RelId(2)=e2 (X,Y), RelId(3)=e3 (Y,Z), RelId(4)=e4 (Z,W).
        // Alternative grouping: (e2⋈e3 on Y) + (e4⋈e1 on W).
        let inner_left = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(2) }),
            right: Box::new(RirNode::Scan { rel: RelId(3) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let inner_right = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(4) }),
            right: Box::new(RirNode::Scan { rel: RelId(1) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner_left),
            right: Box::new(inner_right),
            // Join on X (ll.col[0] == rr.col[1]) and Z (lr.col[1] == rl.col[0]).
            // outer-left output cols [X, Y, Y, Z]; outer-right output cols [Z, W, W, X].
            left_keys: vec![0, 3],
            right_keys: vec![3, 0],
            join_type: JoinType::Inner,
        };
        // Outer cols: [X(0), Y(1), Y(2), Z(3), Z(4), W(5), W(6), X(7)].
        // Project to (W, X, Y, Z): [5, 0, 1, 3].
        let body = RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(5),
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
        };
        let mut plan = plan_with_body(body);
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        let RirNode::MultiWayJoin {
            inputs, slot_vars, ..
        } = &plan.rules_by_scc[0][0].body
        else {
            panic!("expected MultiWayJoin after promotion");
        };
        let scan_rels: Vec<RelId> = inputs
            .iter()
            .map(|n| match n {
                RirNode::Scan { rel } => *rel,
                _ => panic!("expected Scan in MultiWayJoin inputs"),
            })
            .collect();
        // Semantic order [WX, XY, YZ, ZW] = [e1, e2, e3, e4]
        // = [RelId(1), RelId(2), RelId(3), RelId(4)].
        assert_eq!(
            scan_rels,
            vec![RelId(1), RelId(2), RelId(3), RelId(4)],
            "inputs must be in semantic order regardless of positional layout"
        );
        // Shape-fixed canonical 4-cycle slot_vars.
        assert_eq!(
            slot_vars,
            &vec![
                vec![Some(0u32), Some(1)],
                vec![Some(1u32), Some(2)],
                vec![Some(2u32), Some(3)],
                vec![Some(3u32), Some(0)],
            ]
        );
    }

    #[test]
    fn rejects_4cycle_with_rotated_columns() {
        let mut body = canonical_4cycle_tree();
        if let RirNode::Project { columns, .. } = &mut body {
            // Rotate: swap col 0 and col 1.
            columns.swap(0, 1);
        }
        assert!(
            try_promote_4cycle(&body, &StatsManager::new(), &CompilerConfig::default()).is_none()
        );
    }

    #[test]
    fn rejects_4cycle_with_non_inner_outer_join() {
        let mut body = canonical_4cycle_tree();
        if let RirNode::Project { input, .. } = &mut body {
            if let RirNode::Join { join_type, .. } = input.as_mut() {
                *join_type = JoinType::LeftOuter;
            }
        }
        assert!(
            try_promote_4cycle(&body, &StatsManager::new(), &CompilerConfig::default()).is_none()
        );
    }

    #[test]
    fn rejects_4cycle_with_wrong_outer_keys() {
        let mut body = canonical_4cycle_tree();
        if let RirNode::Project { input, .. } = &mut body {
            if let RirNode::Join { left_keys, .. } = input.as_mut() {
                *left_keys = vec![0, 4]; // not [0, 3]
            }
        }
        assert!(
            try_promote_4cycle(&body, &StatsManager::new(), &CompilerConfig::default()).is_none()
        );
    }

    /// Recursive promotion contract: stable 4-cycle (zero recursive Scans) IS
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
        promote_multiway(
            &mut plan,
            &HashMap::new(),
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Recursive promotion contract: linear-recursive 4-cycle (count 1) IS
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
        promote_multiway(
            &mut plan,
            &rel_ids,
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    /// Occurrence-level recursive promotion contract: ≥ 2 recursive
    /// Scans in a 4-cycle body IS promoted to `MultiWayJoin`.
    #[test]
    fn promotes_multirec_4cycle_in_recursive_scc() {
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
        promote_multiway(
            &mut plan,
            &rel_ids,
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
        assert!(matches!(
            &plan.rules_by_scc[0][0].body,
            RirNode::MultiWayJoin { .. }
        ));
    }

    #[test]
    fn triangle_does_not_match_4cycle_promoter() {
        // Pin the cross-shape contract: a body matched by
        // try_promote_triangle must NOT also match try_promote_4cycle.
        // Both promoters should be exclusive.
        let triangle = canonical_triangle_tree();
        assert!(
            try_promote_4cycle(&triangle, &StatsManager::new(), &CompilerConfig::default())
                .is_none()
        );
        let four_cycle = canonical_4cycle_tree();
        assert!(try_promote_triangle(
            &four_cycle,
            &StatsManager::new(),
            &CompilerConfig::default()
        )
        .is_none());
    }
}
