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
use xlog_stats::StatsManager;

use crate::compiler_config::CompilerConfig;
use crate::wcoj_var_ordering::WcojVariableOrderingModel;

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
pub fn promote_multiway(
    plan: &mut ExecutionPlan,
    rel_ids: &HashMap<String, RelId>,
    stats: &StatsManager,
    config: &CompilerConfig,
) {
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
            if let Some(promoted) = try_promote_triangle(&rule.body, stats, config) {
                rule.body = promoted;
                continue;
            }
            if let Some(promoted) = try_promote_4cycle(&rule.body, stats, config) {
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

/// W2.2: encode each atom-column slot as a u8 in `0..6` —
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

/// W2.2 — semantic-slot inference for triangle bodies. Given
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

/// W2.2: recognize the canonical triangle in any valid
/// inner-key combination (`[1]/[0]`, `[1]/[1]`, `[0]/[0]`)
/// and produce the equivalent `MultiWayJoin` with
/// `inputs` arranged in canonical semantic order
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
    // W2.1 + W2.6: dispatch to the cost model selected by
    // `config.wcoj_variable_ordering`. With
    // `CompilerConfig::default()` (Disabled), no cost model
    // runs and slice 1/2/4/W2.2 behavior is bit-identical.
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
        var_order,
    })
}

/// W2.2: 4-cycle has 4 atoms × 2 cols = 8 slots. Encode as
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

/// W2.2 — semantic-slot inference for 4-cycle bodies. Given
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

    // W2.2 — variable-graph deduction of which atom is
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
    // W2.1: ask the cost model whether to set a non-default
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
        WcojVarOrderingKind::HeatAware => HeatAwareLeaderModel.pick_4cycle_leader(
            [rel_wx, rel_xy, rel_yz, rel_zw],
            stats,
            config,
        ),
    };
    let var_order = leader_idx_4.map(build_cycle4_var_order);
    Some(RirNode::MultiWayJoin {
        inputs,
        slot_vars,
        output_columns,
        fallback,
        var_order,
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

    /// W2.2: triangle with X-shared inner pair — inner keys
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

    /// W2.2: triangle with Z-shared inner pair — inner keys
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
        // v0.6.5 W2.2: slice 1's strict rejection of non-canonical
        // projection columns is intentionally relaxed. The
        // variable-graph promoter recognizes the triangle
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
        // Body now promoted to MultiWayJoin (W2.2 contract).
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
        promote_multiway(
            &mut plan,
            &rel_ids,
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
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

    /// W2.2: 4-cycle with the alternative bushy grouping
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
        promote_multiway(
            &mut plan,
            &rel_ids,
            &StatsManager::new(),
            &CompilerConfig::default(),
        );
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
