//! Tree rewriting and incremental delta recomputation.

use std::collections::{HashMap, HashSet};

use xlog_core::{RelId, Result, XlogError};
use xlog_ir::{ExecutionPlan, JoinType, RirNode};

use super::Executor;
use super::RelationDelta;

impl Executor {
    /// Apply base-relation deltas and recompute affected SCCs (no recompilation).
    ///
    /// This provides correctness for both insertions and deletions by recomputing any SCCs that
    /// depend (directly or transitively) on the changed relations.
    pub fn apply_deltas_and_recompute(
        &mut self,
        plan: &ExecutionPlan,
        deltas: &HashMap<String, RelationDelta>,
    ) -> Result<()> {
        if deltas.is_empty() {
            return Ok(());
        }

        let has_deletes = deltas
            .values()
            .any(|d| d.delete.as_ref().map(|b| !b.is_empty()).unwrap_or(false));

        // 1) Apply EDB updates.
        for (name, delta) in deltas {
            let existing = self.store.get(name);

            let base_schema = existing
                .map(|b| b.schema().clone())
                .or_else(|| delta.insert.as_ref().map(|b| b.schema().clone()))
                .or_else(|| delta.delete.as_ref().map(|b| b.schema().clone()))
                .ok_or_else(|| {
                    XlogError::Execution(format!(
                        "Delta update for {} has no existing relation and no schema",
                        name
                    ))
                })?;

            let mut updated = if let Some(buf) = existing {
                self.clone_buffer(buf)?
            } else {
                self.create_empty_buffer(base_schema)?
            };

            if let Some(delete_buf) = &delta.delete {
                updated = self.provider.diff_gpu(&updated, delete_buf)?;
            }
            if let Some(insert_buf) = &delta.insert {
                updated = self.provider.union_gpu(&updated, insert_buf)?;
            }

            self.store_put(name, updated);
        }

        // 2) Compute affected SCC closure.
        let changed_preds: HashSet<&str> = deltas.keys().map(|s| s.as_str()).collect();

        let mut pred_to_scc: HashMap<&str, u32> = HashMap::new();
        for scc in &plan.sccs {
            for pred in &scc.predicates {
                pred_to_scc.insert(pred.as_str(), scc.id);
            }
        }

        let mut dependents: HashMap<u32, Vec<u32>> = HashMap::new();
        for (scc_id, rules) in plan.rules_by_scc.iter().enumerate() {
            let scc_id = scc_id as u32;
            for rule in rules {
                let mut rels = Vec::new();
                Self::collect_scan_rels(&rule.body, &mut rels);
                for rel in rels {
                    let Some(name) = self.get_rel_name(rel) else {
                        continue;
                    };
                    let Some(&dep_scc) = pred_to_scc.get(name) else {
                        continue;
                    };
                    if dep_scc == scc_id {
                        continue;
                    }
                    dependents.entry(dep_scc).or_default().push(scc_id);
                }
            }
        }

        let mut affected: HashSet<u32> = HashSet::new();
        let mut queue: Vec<u32> = Vec::new();
        for pred in &changed_preds {
            if let Some(&scc) = pred_to_scc.get(*pred) {
                affected.insert(scc);
                queue.push(scc);
            }
        }

        while let Some(scc) = queue.pop() {
            if let Some(deps) = dependents.get(&scc) {
                for &next in deps {
                    if affected.insert(next) {
                        queue.push(next);
                    }
                }
            }
        }

        if affected.is_empty() {
            return Ok(());
        }

        fn contains_non_monotonic_ops(node: &RirNode) -> bool {
            match node {
                RirNode::Unit | RirNode::Scan { .. } => false,
                RirNode::Filter { input, .. }
                | RirNode::Project { input, .. }
                | RirNode::Distinct { input, .. } => contains_non_monotonic_ops(input),
                RirNode::Union { inputs } => inputs.iter().any(contains_non_monotonic_ops),
                RirNode::GroupBy { .. } | RirNode::Diff { .. } => true,
                RirNode::Join {
                    left,
                    right,
                    join_type,
                    ..
                } => {
                    matches!(join_type, JoinType::Anti | JoinType::LeftOuter)
                        || contains_non_monotonic_ops(left)
                        || contains_non_monotonic_ops(right)
                }
                RirNode::Fixpoint {
                    base, recursive, ..
                } => contains_non_monotonic_ops(base) || contains_non_monotonic_ops(recursive),
                RirNode::TensorMaskedJoin { .. } => false,
                // v0.6.5: walk the fallback. The promoter only wraps
                // already-monotonic triangle subtrees in v1, but the
                // fallback is the load-bearing source of truth.
                RirNode::MultiWayJoin { fallback, .. } => contains_non_monotonic_ops(fallback),
            }
        }

        // 3) Decide which SCCs must be recomputed (cleared first).
        //
        // If there are deletes, we always recompute for correctness.
        // If there are only inserts, we can incrementally update SCCs that are monotone w.r.t.
        // insertion (no anti-joins, diffs, or aggregates) and do a targeted recompute for the rest.
        let mut recompute_sccs: HashSet<u32> = HashSet::new();
        if has_deletes {
            recompute_sccs = affected.clone();
        } else {
            for &scc_id in &affected {
                if let Some(rules) = plan.rules_by_scc.get(scc_id as usize) {
                    if rules.iter().any(|r| contains_non_monotonic_ops(&r.body)) {
                        recompute_sccs.insert(scc_id);
                    }
                }
            }

            // If any SCC is recomputed due to non-monotonic ops, all dependents must also be
            // recomputed because their prior outputs may now be invalid.
            let mut queue: Vec<u32> = recompute_sccs.iter().copied().collect();
            while let Some(scc) = queue.pop() {
                if let Some(deps) = dependents.get(&scc) {
                    for &next in deps {
                        if !affected.contains(&next) {
                            continue;
                        }
                        if recompute_sccs.insert(next) {
                            queue.push(next);
                        }
                    }
                }
            }
        }

        // 4) Clear IDB relations for SCCs we are recomputing (but never clear directly-updated bases).
        for scc_id in &recompute_sccs {
            let Some(scc) = plan.sccs.iter().find(|s| s.id == *scc_id) else {
                continue;
            };

            for pred in &scc.predicates {
                if changed_preds.contains(pred.as_str()) {
                    continue;
                }
                let schema = self
                    .store
                    .get(pred)
                    .map(|b| b.schema().clone())
                    .or_else(|| {
                        plan.rules_by_scc
                            .get(*scc_id as usize)
                            .and_then(|rules| rules.iter().find(|r| r.head == pred.as_str()))
                            .and_then(|r| {
                                let schema = r.meta.schema.clone();
                                if schema.arity() > 0 {
                                    Some(schema)
                                } else {
                                    None
                                }
                            })
                    })
                    .ok_or_else(|| {
                        XlogError::Execution(format!(
                            "Missing schema for predicate {} during recompute",
                            pred
                        ))
                    })?;

                let empty = self.create_empty_buffer(schema)?;
                self.store_put(pred, empty);
            }
        }

        // 5) Re-execute affected SCCs in plan order (incremental for insert-only monotone SCCs).
        for stratum in &plan.strata {
            for &scc_id in &stratum.sccs {
                if !affected.contains(&scc_id) {
                    continue;
                }
                let rules = plan.rules_by_scc.get(scc_id as usize).ok_or_else(|| {
                    XlogError::Execution(format!("Missing rules for SCC {}", scc_id))
                })?;
                let is_recursive = plan
                    .sccs
                    .iter()
                    .find(|s| s.id == scc_id)
                    .map(|s| s.is_recursive)
                    .unwrap_or(false);

                if is_recursive {
                    self.execute_recursive_scc(rules)?;
                } else {
                    self.execute_non_recursive_scc(rules)?;
                }
            }
        }

        Ok(())
    }

    pub(crate) fn collect_scan_rels(node: &RirNode, out: &mut Vec<RelId>) {
        match node {
            RirNode::Unit => {}
            RirNode::Scan { rel } => out.push(*rel),
            RirNode::Filter { input, .. } | RirNode::Project { input, .. } => {
                Self::collect_scan_rels(input, out);
            }
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                Self::collect_scan_rels(left, out);
                Self::collect_scan_rels(right, out);
            }
            RirNode::GroupBy { input, .. } | RirNode::Distinct { input, .. } => {
                Self::collect_scan_rels(input, out);
            }
            RirNode::Union { inputs } => {
                for input in inputs {
                    Self::collect_scan_rels(input, out);
                }
            }
            RirNode::Fixpoint {
                base, recursive, ..
            } => {
                Self::collect_scan_rels(base, out);
                Self::collect_scan_rels(recursive, out);
            }
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                for (rel_id, _) in rel_index {
                    out.push(*rel_id);
                }
            }
            // v0.6.5: collect from `inputs` only — the fallback subtree
            // references the same set by promoter invariant.
            RirNode::MultiWayJoin { inputs, .. } => {
                for input in inputs {
                    Self::collect_scan_rels(input, out);
                }
            }
        }
    }

    pub(crate) fn rewrite_scan_nth(
        node: &RirNode,
        target: RelId,
        nth: usize,
        replacement: RelId,
    ) -> Option<RirNode> {
        let mut remaining = nth;
        let (rewritten, replaced) =
            Self::rewrite_scan_nth_impl(node, target, &mut remaining, replacement);
        replaced.then_some(rewritten)
    }

    fn rewrite_scan_nth_impl(
        node: &RirNode,
        target: RelId,
        remaining: &mut usize,
        replacement: RelId,
    ) -> (RirNode, bool) {
        match node {
            RirNode::Unit => (RirNode::Unit, false),
            RirNode::Scan { rel } => {
                if *rel == target {
                    if *remaining == 0 {
                        // W4.1 (paper P1): replace exactly one occurrence
                        // per `rewrite_scan_nth` call, then mark this walk
                        // "done" via the `usize::MAX` sentinel so subsequent
                        // matches in the same walk do NOT replace again.
                        // Without this, a body with 2+ same-predicate
                        // recursive Scans would have ALL occurrences after
                        // `nth` overwritten when the caller intended only
                        // the `nth`-th to be substituted.
                        *remaining = usize::MAX;
                        return (RirNode::Scan { rel: replacement }, true);
                    }
                    if *remaining != usize::MAX {
                        *remaining -= 1;
                    }
                }
                (node.clone(), false)
            }

            RirNode::Filter { input, predicate } => {
                let (new_input, replaced) =
                    Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                (
                    RirNode::Filter {
                        input: Box::new(new_input),
                        predicate: predicate.clone(),
                    },
                    replaced,
                )
            }

            RirNode::Project { input, columns } => {
                let (new_input, replaced) =
                    Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                (
                    RirNode::Project {
                        input: Box::new(new_input),
                        columns: columns.clone(),
                    },
                    replaced,
                )
            }

            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => {
                let (new_left, replaced_left) =
                    Self::rewrite_scan_nth_impl(left, target, remaining, replacement);
                if replaced_left {
                    return (
                        RirNode::Join {
                            left: Box::new(new_left),
                            right: right.clone(),
                            left_keys: left_keys.clone(),
                            right_keys: right_keys.clone(),
                            join_type: *join_type,
                        },
                        true,
                    );
                }
                let (new_right, replaced_right) =
                    Self::rewrite_scan_nth_impl(right, target, remaining, replacement);
                (
                    RirNode::Join {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                        left_keys: left_keys.clone(),
                        right_keys: right_keys.clone(),
                        join_type: *join_type,
                    },
                    replaced_right,
                )
            }

            RirNode::GroupBy {
                input,
                key_cols,
                aggs,
            } => {
                let (new_input, replaced) =
                    Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                (
                    RirNode::GroupBy {
                        input: Box::new(new_input),
                        key_cols: key_cols.clone(),
                        aggs: aggs.clone(),
                    },
                    replaced,
                )
            }

            RirNode::Union { inputs } => {
                let mut replaced_any = false;
                let mut new_inputs = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let (new_input, replaced) =
                        Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                    replaced_any |= replaced;
                    new_inputs.push(new_input);
                }
                (RirNode::Union { inputs: new_inputs }, replaced_any)
            }

            RirNode::Distinct { input, key_cols } => {
                let (new_input, replaced) =
                    Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                (
                    RirNode::Distinct {
                        input: Box::new(new_input),
                        key_cols: key_cols.clone(),
                    },
                    replaced,
                )
            }

            RirNode::Diff { left, right } => {
                let (new_left, replaced_left) =
                    Self::rewrite_scan_nth_impl(left, target, remaining, replacement);
                if replaced_left {
                    return (
                        RirNode::Diff {
                            left: Box::new(new_left),
                            right: right.clone(),
                        },
                        true,
                    );
                }
                let (new_right, replaced_right) =
                    Self::rewrite_scan_nth_impl(right, target, remaining, replacement);
                (
                    RirNode::Diff {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                    },
                    replaced_right,
                )
            }

            RirNode::Fixpoint {
                scc_id,
                base,
                recursive,
                delta_rel,
                full_rel,
            } => {
                let (new_base, replaced_base) =
                    Self::rewrite_scan_nth_impl(base, target, remaining, replacement);
                if replaced_base {
                    return (
                        RirNode::Fixpoint {
                            scc_id: *scc_id,
                            base: Box::new(new_base),
                            recursive: recursive.clone(),
                            delta_rel: *delta_rel,
                            full_rel: *full_rel,
                        },
                        true,
                    );
                }
                let (new_recursive, replaced_recursive) =
                    Self::rewrite_scan_nth_impl(recursive, target, remaining, replacement);
                (
                    RirNode::Fixpoint {
                        scc_id: *scc_id,
                        base: Box::new(new_base),
                        recursive: Box::new(new_recursive),
                        delta_rel: *delta_rel,
                        full_rel: *full_rel,
                    },
                    replaced_recursive,
                )
            }
            RirNode::TensorMaskedJoin { .. } => {
                // TensorMaskedJoin is a leaf node — no child scans to rewrite.
                (node.clone(), false)
            }
            // W4.1 (paper P1): rewrite `inputs` and `fallback` with
            // SEPARATE `remaining` counter copies — both views are the
            // same logical body, so each must independently target the
            // N-th occurrence. Sharing one counter across the two walks
            // contaminated the fallback's count by the inputs' consumed
            // matches, which produced wrong-occurrence substitutions on
            // self-recursive bodies. The outer caller's `remaining` is
            // updated to whatever the inputs walk consumed, so siblings
            // of this MultiWayJoin (rare; typically wrapped in Project)
            // see consistent counting.
            RirNode::MultiWayJoin {
                inputs,
                slot_vars,
                output_columns,
                fallback,
                var_order,
            } => {
                let starting_remaining = *remaining;
                let mut inputs_remaining = starting_remaining;
                let mut new_inputs = Vec::with_capacity(inputs.len());
                let mut any_replaced = false;
                for inp in inputs {
                    let (new_inp, replaced) = Self::rewrite_scan_nth_impl(
                        inp,
                        target,
                        &mut inputs_remaining,
                        replacement,
                    );
                    any_replaced |= replaced;
                    new_inputs.push(new_inp);
                }
                let mut fallback_remaining = starting_remaining;
                let (new_fallback, fallback_replaced) = Self::rewrite_scan_nth_impl(
                    fallback,
                    target,
                    &mut fallback_remaining,
                    replacement,
                );
                *remaining = inputs_remaining;
                (
                    RirNode::MultiWayJoin {
                        inputs: new_inputs,
                        slot_vars: slot_vars.clone(),
                        output_columns: output_columns.clone(),
                        fallback: Box::new(new_fallback),
                        var_order: var_order.clone(),
                    },
                    any_replaced || fallback_replaced,
                )
            }
        }
    }
}

#[cfg(test)]
mod multiway_walker_tests {
    //! v0.6.5 slice 1: walker arm coverage for `MultiWayJoin` in the
    //! rewrite module. `contains_non_monotonic_ops` is a nested `fn`
    //! inside an `Executor` method and is not directly callable; its
    //! arm is exercised through integration tests in step 5. The two
    //! `pub(crate)` walkers below are testable in isolation.

    use super::*;
    use xlog_ir::rir::ProjectExpr;

    fn triangle_multiway(a: RelId, b: RelId, c: RelId) -> RirNode {
        let scan_a = RirNode::Scan { rel: a };
        let scan_b = RirNode::Scan { rel: b };
        let scan_c = RirNode::Scan { rel: c };
        let inner = RirNode::Join {
            left: Box::new(scan_a.clone()),
            right: Box::new(scan_b.clone()),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(scan_c.clone()),
            left_keys: vec![0, 3],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let fallback = RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
        };
        RirNode::MultiWayJoin {
            inputs: vec![scan_a, scan_b, scan_c],
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
            var_order: None,
        }
    }

    #[test]
    fn collect_scan_rels_walks_multiway_inputs_only() {
        let node = triangle_multiway(RelId(10), RelId(20), RelId(30));
        let mut out = Vec::new();
        Executor::collect_scan_rels(&node, &mut out);
        // One entry per input slot; fallback is NOT walked (would
        // double-count to 6 entries if it were).
        assert_eq!(out.len(), 3, "expected 3 scan rels, got: {:?}", out);
        assert!(out.contains(&RelId(10)));
        assert!(out.contains(&RelId(20)));
        assert!(out.contains(&RelId(30)));
    }

    #[test]
    fn rewrite_scan_nth_rewrites_inputs_and_fallback() {
        let node = triangle_multiway(RelId(10), RelId(20), RelId(30));
        // Replace the second occurrence of RelId(10). Across the
        // MultiWayJoin, RelId(10) appears once in `inputs[0]` and
        // once inside `fallback` (the outer join's leftmost leaf).
        // `rewrite_scan_nth` walks the inputs first (count 1), then
        // the fallback (count 2). With nth=1 we rewrite the second
        // hit — i.e., the fallback occurrence.
        let rewritten = Executor::rewrite_scan_nth(&node, RelId(10), 1, RelId(99))
            .expect("rewrite must succeed");
        match rewritten {
            RirNode::MultiWayJoin {
                inputs, fallback, ..
            } => {
                // Input[0] is unchanged.
                assert!(matches!(inputs[0], RirNode::Scan { rel: RelId(10) }));
                // Fallback has the RelId(10) leaf swapped to RelId(99).
                fn find_99(n: &RirNode) -> bool {
                    match n {
                        RirNode::Scan { rel: RelId(99) } => true,
                        RirNode::Project { input, .. } => find_99(input),
                        RirNode::Join { left, right, .. } => find_99(left) || find_99(right),
                        _ => false,
                    }
                }
                assert!(find_99(&fallback), "fallback must contain RelId(99)");
            }
            _ => panic!("expected MultiWayJoin after rewrite"),
        }
    }

    /// v0.6.5 slice 2 (D4) — shape-agnosticism guard.
    ///
    /// Slice 1's promoter is triangle-only; future slices will add
    /// 4-input shapes. The walker arms in `collect_scan_rels` and
    /// `rewrite_scan_nth_impl` must NOT hard-code `inputs.len() ==
    /// 3`. Synthesize a 4-input `MultiWayJoin` directly and exercise
    /// the walker. This test does NOT execute the IR through the
    /// runtime — it only pins the walker's contract.
    fn fourway_multiway(a: RelId, b: RelId, c: RelId, d: RelId) -> RirNode {
        // Synthetic 4-cycle slot_vars [[A,B],[B,C],[C,D],[A,D]] with
        // a stub fallback whose Scan leaves repeat each rel once.
        let inner1 = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: a }),
            right: Box::new(RirNode::Scan { rel: b }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let inner2 = RirNode::Join {
            left: Box::new(inner1),
            right: Box::new(RirNode::Scan { rel: c }),
            left_keys: vec![3],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner2),
            right: Box::new(RirNode::Scan { rel: d }),
            left_keys: vec![0, 5],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let fallback = RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                xlog_ir::rir::ProjectExpr::Column(0),
                xlog_ir::rir::ProjectExpr::Column(1),
                xlog_ir::rir::ProjectExpr::Column(3),
                xlog_ir::rir::ProjectExpr::Column(5),
            ],
        };
        RirNode::MultiWayJoin {
            inputs: vec![
                RirNode::Scan { rel: a },
                RirNode::Scan { rel: b },
                RirNode::Scan { rel: c },
                RirNode::Scan { rel: d },
            ],
            slot_vars: vec![
                vec![Some(0), Some(1)],
                vec![Some(1), Some(2)],
                vec![Some(2), Some(3)],
                vec![Some(0), Some(3)],
            ],
            output_columns: vec![
                xlog_ir::rir::ProjectExpr::Column(0),
                xlog_ir::rir::ProjectExpr::Column(1),
                xlog_ir::rir::ProjectExpr::Column(2),
                xlog_ir::rir::ProjectExpr::Column(3),
            ],
            fallback: Box::new(fallback),
            var_order: None,
        }
    }

    #[test]
    fn collect_scan_rels_handles_4_inputs() {
        let node = fourway_multiway(RelId(10), RelId(20), RelId(30), RelId(40));
        let mut out = Vec::new();
        Executor::collect_scan_rels(&node, &mut out);
        assert_eq!(
            out.len(),
            4,
            "expected 4 scan rels, got {} entries: {:?}",
            out.len(),
            out
        );
        for id in [10, 20, 30, 40] {
            assert!(out.contains(&RelId(id)), "RelId({}) missing", id);
        }
    }

    #[test]
    fn rewrite_scan_nth_handles_4_inputs_and_fallback() {
        let node = fourway_multiway(RelId(10), RelId(20), RelId(30), RelId(40));
        // RelId(40) appears once in `inputs[3]` and once inside
        // `fallback` (the outer join's right scan). nth=0 rewrites
        // the first hit (input[3]); nth=1 rewrites the fallback hit.
        let rewritten_first = Executor::rewrite_scan_nth(&node, RelId(40), 0, RelId(99))
            .expect("first rewrite must succeed");
        let RirNode::MultiWayJoin { inputs, .. } = rewritten_first else {
            panic!("expected MultiWayJoin");
        };
        assert!(matches!(inputs[3], RirNode::Scan { rel: RelId(99) }));

        let rewritten_second = Executor::rewrite_scan_nth(&node, RelId(40), 1, RelId(99))
            .expect("second rewrite must succeed");
        let RirNode::MultiWayJoin {
            inputs, fallback, ..
        } = rewritten_second
        else {
            panic!("expected MultiWayJoin");
        };
        assert!(matches!(inputs[3], RirNode::Scan { rel: RelId(40) }));
        fn find_99(n: &RirNode) -> bool {
            match n {
                RirNode::Scan { rel: RelId(99) } => true,
                RirNode::Project { input, .. } => find_99(input),
                RirNode::Join { left, right, .. } => find_99(left) || find_99(right),
                _ => false,
            }
        }
        assert!(find_99(&fallback), "fallback must contain RelId(99)");
    }
}

#[cfg(test)]
mod w41_rewrite_scan_nth_occurrence_identity_tests {
    //! W4.1 (paper P1) — `rewrite_scan_nth` occurrence-identity
    //! preservation. The Step-6 fix at `rewrite.rs:Scan case` (sentinel
    //! post-replacement) and `:MultiWayJoin arm` (separate `remaining`
    //! counters for inputs vs fallback) ensures:
    //!
    //! 1. For a body with N same-predicate occurrences, calling
    //!    `rewrite_scan_nth(body, target, occ=k, replacement)` substitutes
    //!    EXACTLY ONE occurrence (the k-th) — not 0, not >1.
    //!
    //! 2. For a `MultiWayJoin` whose `inputs` and `fallback` both contain
    //!    the target, occ=k substitutes the k-th occurrence INDEPENDENTLY
    //!    in inputs AND in fallback (both views share the same logical
    //!    body; both must reflect the same logical rewrite).
    //!
    //! Pre-W4.1 behavior bugs (now fixed):
    //! - Scan case early-returned on match without decrementing
    //!   `remaining`, so subsequent matches in the same walk would also
    //!   replace at remaining==0.
    //! - MultiWayJoin arm shared `&mut remaining` across the inputs walk
    //!   and the subsequent fallback walk; the fallback walk's counter
    //!   was contaminated by inputs' consumption.
    //!
    //! Both bugs latent on distinct-recursive-predicate fixtures (slice 4
    //! single-rec, MULTIREC_TRIANGLE with r1+r2 distinct); manifest on
    //! same-predicate self-recursive bodies admitted by W4.1 Step 5.

    use super::*;
    use xlog_ir::rir::ProjectExpr;
    use xlog_ir::JoinType;

    /// Build a synthetic `MultiWayJoin` whose `inputs` are 3 same-
    /// predicate Scans (`Scan { rel: target_rel }` × 3) plus a fallback
    /// that mirrors the same 3 Scans inside a left-deep Join chain.
    /// This is the structural shape of a self-recursive triangle body
    /// like `tri(X,Y,Z) :- p(X,Y), p(Y,Z), p(X,Z)` after promotion: 3
    /// inputs slots all targeting `p`, fallback containing 3 `p` Scans
    /// in the binary-join expansion. The fallback's left-deep Join
    /// produces canonical depth-first walk order
    /// `[innermost-left, inner-right, outer-right]` — matching the
    /// inputs' left-to-right order, so the k-th input occurrence and
    /// the k-th fallback occurrence correspond to the same logical
    /// body slot.
    fn three_same_predicate_multiway(target_rel: RelId) -> RirNode {
        let inputs = vec![
            RirNode::Scan { rel: target_rel },
            RirNode::Scan { rel: target_rel },
            RirNode::Scan { rel: target_rel },
        ];
        let inner = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: target_rel }),
            right: Box::new(RirNode::Scan { rel: target_rel }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(RirNode::Scan { rel: target_rel }),
            left_keys: vec![0, 3],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let fallback = RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
        };
        RirNode::MultiWayJoin {
            inputs,
            slot_vars: vec![
                vec![Some(0), Some(1)],
                vec![Some(1), Some(2)],
                vec![Some(0), Some(2)],
            ],
            output_columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(2),
            ],
            fallback: Box::new(fallback),
            var_order: None,
        }
    }

    /// Walk a body depth-first / left-first and collect every Scan
    /// RelId in encounter order. For a `MultiWayJoin`, walks `inputs`
    /// in order then `fallback`. Used by the regression tests to
    /// assert EXACT post-rewrite positional identity.
    fn collect_scans_in_order(node: &RirNode, out: &mut Vec<RelId>) {
        match node {
            RirNode::Unit => {}
            RirNode::Scan { rel } => out.push(*rel),
            RirNode::Filter { input, .. }
            | RirNode::Project { input, .. }
            | RirNode::GroupBy { input, .. }
            | RirNode::Distinct { input, .. } => collect_scans_in_order(input, out),
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                collect_scans_in_order(left, out);
                collect_scans_in_order(right, out);
            }
            RirNode::Union { inputs } => {
                for n in inputs {
                    collect_scans_in_order(n, out);
                }
            }
            RirNode::Fixpoint {
                base, recursive, ..
            } => {
                collect_scans_in_order(base, out);
                collect_scans_in_order(recursive, out);
            }
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                for (rid, _) in rel_index {
                    out.push(*rid);
                }
            }
            RirNode::MultiWayJoin {
                inputs, fallback, ..
            } => {
                for inp in inputs {
                    collect_scans_in_order(inp, out);
                }
                collect_scans_in_order(fallback, out);
            }
        }
    }

    /// Pre-W4.1 bug pin: with 3 occurrences of `target` in `inputs`
    /// and 3 in `fallback`, the sentinel/separate-counter fix BOTH
    /// applied makes occ=k substitute the k-th occurrence in
    /// **inputs walk order** AND the k-th occurrence in **fallback
    /// walk order** — and ONLY those two positions; all other
    /// occurrences remain unchanged.
    ///
    /// This test asserts EXACT positional identity (not just total
    /// replacement count). A broken implementation that always
    /// rewrites occurrence 0 for every occ would pass a count-only
    /// check but fails this positional assertion.
    #[test]
    fn rewrite_scan_nth_replaces_exact_kth_occurrence_in_inputs_and_fallback() {
        let target = RelId(7);
        let body = three_same_predicate_multiway(target);

        // Pre-rewrite: 6 Scans of target in canonical walk order.
        // [input[0], input[1], input[2], fallback's innermost-left,
        //  fallback's inner-right, fallback's outer-right].
        let mut pre = Vec::new();
        collect_scans_in_order(&body, &mut pre);
        assert_eq!(
            pre,
            vec![target, target, target, target, target, target],
            "pre-rewrite: 6 target Scans in canonical walk order"
        );

        // For each occ in {0, 1, 2}, the k-th occurrence in the
        // INPUTS walk AND the k-th occurrence in the FALLBACK walk
        // are replaced — and nothing else.
        for occ in 0..3 {
            // Use a distinct RelId per occ so a buggy implementation
            // that always rewrites occurrence 0 for every occ would
            // produce a different post-rewrite Scan order than
            // expected.
            let replacement = RelId(100 + occ as u32);
            let rewritten = Executor::rewrite_scan_nth(&body, target, occ, replacement)
                .unwrap_or_else(|| panic!("occ={} must succeed", occ));

            let mut post = Vec::new();
            collect_scans_in_order(&rewritten, &mut post);

            // Build expected sequence: positions 0..3 = inputs walk,
            // positions 3..6 = fallback walk. Position `occ` in each
            // half becomes `replacement`; all others remain `target`.
            let mut expected = vec![target; 6];
            expected[occ] = replacement; // k-th input occurrence
            expected[3 + occ] = replacement; // k-th fallback occurrence

            assert_eq!(
                post, expected,
                "occ={}: post-rewrite Scan order must replace EXACTLY the k-th occurrence in inputs AND fallback; got {:?}, expected {:?}",
                occ, post, expected
            );
        }
    }

    /// Pre-W4.1 bug pin: occ=0 of a target appearing in input[0] AND
    /// in fallback's leftmost leaf substitutes BOTH copies (input/
    /// fallback symmetry). Locks paper-P1's "logical body shared
    /// between inputs and fallback" semantic.
    ///
    /// This test asserts the EXACT post-rewrite shape (input[0]
    /// becomes replacement; the rest of the inputs+fallback structure
    /// is identical to pre-rewrite except fallback's leftmost-Scan
    /// becomes replacement). Complementary to
    /// `rewrite_scan_nth_replaces_exact_kth_occurrence_in_inputs_and_fallback`
    /// above which exercises occ ∈ {0, 1, 2}; this test is the
    /// focused occ=0 cert.
    #[test]
    fn rewrite_scan_nth_input_fallback_symmetry_at_occ_0() {
        let target = RelId(7);
        let replacement = RelId(99);
        let body = three_same_predicate_multiway(target);

        let rewritten = Executor::rewrite_scan_nth(&body, target, 0, replacement)
            .expect("occ=0 must succeed");

        match rewritten {
            RirNode::MultiWayJoin {
                inputs, fallback, ..
            } => {
                // input[0] must be the replacement (the 0th occurrence
                // in inputs).
                assert!(
                    matches!(inputs[0], RirNode::Scan { rel } if rel == replacement),
                    "input[0] must be replacement; got {:?}",
                    inputs[0]
                );
                // input[1] and input[2] must remain the original target.
                assert!(matches!(inputs[1], RirNode::Scan { rel } if rel == target));
                assert!(matches!(inputs[2], RirNode::Scan { rel } if rel == target));
                // Fallback walk order: [innermost-left, inner-right,
                // outer-right]. occ=0 must replace the innermost-left
                // (position 0 in fallback walk) ONLY.
                let mut fallback_scans = Vec::new();
                collect_scans_in_order(&fallback, &mut fallback_scans);
                assert_eq!(
                    fallback_scans,
                    vec![replacement, target, target],
                    "fallback walk order: occ=0 must replace position 0 only"
                );
            }
            _ => panic!("expected MultiWayJoin after rewrite"),
        }
    }
}
