//! Host-side Decision-DNNF compiler decision-order hint derived from provenance structure.
//!
//! The GPU-native Decision-DNNF branching heuristic (smallest open clause, tie-broken by clause
//! id, then by minimum variable id within the clause) is deterministic and
//! lives in the CUDA kernels; it takes no priority input. This pass steers
//! the variable-id tie-break from the host instead: leaf and choice variables
//! are renumbered by descending structural fanout in the PIR DAG, so that
//! variables shared across many derivations ("rule-guard"-like variables)
//! receive the smallest CNF variable ids and are case-split first.
//!
//! Exact weighted model counting is invariant under variable renumbering, so
//! query probabilities are unchanged; only the compile-time search shape
//! (frontier/DFS splits) can differ.

use std::cmp::Reverse;
use std::collections::BTreeMap;

use crate::pir::{ChoiceVarId, LeafId, PirNode};
use crate::provenance::Provenance;

/// Renumber leaf/choice variables of `provenance` by descending PIR fanout.
///
/// Ties are broken by the original id, so the pass is deterministic and is a
/// no-op permutation when all fanouts are equal and already ordered.
pub(crate) fn apply_decision_order_hint(mut provenance: Provenance) -> Provenance {
    let nodes = provenance.pir.nodes();

    // Parent-reference count per PIR node id.
    let mut node_refs: Vec<u64> = vec![0; nodes.len()];
    for node in nodes {
        match node {
            PirNode::And { children } | PirNode::Or { children } => {
                for child in children {
                    node_refs[child.as_u32() as usize] += 1;
                }
            }
            PirNode::Decision {
                child_false,
                child_true,
                ..
            } => {
                node_refs[child_false.as_u32() as usize] += 1;
                node_refs[child_true.as_u32() as usize] += 1;
            }
            PirNode::Const(_) | PirNode::Lit { .. } | PirNode::NegLit { .. } => {}
        }
    }

    // Structural score per leaf / choice variable: total references to the
    // PIR nodes that mention it.
    let mut leaf_score: BTreeMap<LeafId, u64> = BTreeMap::new();
    let mut choice_score: BTreeMap<ChoiceVarId, u64> = BTreeMap::new();
    for leaf in provenance.leaf_probs.keys() {
        leaf_score.insert(*leaf, 0);
    }
    for var in provenance.choice_probs.keys() {
        choice_score.insert(*var, 0);
    }
    for (idx, node) in nodes.iter().enumerate() {
        match node {
            PirNode::Lit { leaf } | PirNode::NegLit { leaf } => {
                *leaf_score.entry(*leaf).or_insert(0) += node_refs[idx];
            }
            PirNode::Decision { var, .. } => {
                *choice_score.entry(*var).or_insert(0) += node_refs[idx];
            }
            PirNode::Const(_) | PirNode::And { .. } | PirNode::Or { .. } => {}
        }
    }

    // Dense rank permutations: descending score, ties by ascending original id.
    let mut leaves: Vec<LeafId> = leaf_score.keys().copied().collect();
    leaves.sort_by_key(|leaf| (Reverse(leaf_score[leaf]), leaf.as_u32()));
    let leaf_map: BTreeMap<LeafId, LeafId> = leaves
        .iter()
        .enumerate()
        .map(|(rank, old)| (*old, LeafId::new(rank as u32)))
        .collect();

    let mut choices: Vec<ChoiceVarId> = choice_score.keys().copied().collect();
    choices.sort_by_key(|var| (Reverse(choice_score[var]), var.as_u32()));
    let choice_map: BTreeMap<ChoiceVarId, ChoiceVarId> = choices
        .iter()
        .enumerate()
        .map(|(rank, old)| (*old, ChoiceVarId::new(rank as u32)))
        .collect();

    // Rewrite PIR nodes in place (node ids are untouched, so PirNodeId-based
    // references such as tuple formulas and evidence roots stay valid).
    for node in provenance.pir.nodes_mut() {
        match node {
            PirNode::Lit { leaf } | PirNode::NegLit { leaf } => {
                if let Some(new) = leaf_map.get(leaf) {
                    *leaf = *new;
                }
            }
            PirNode::Decision { var, .. } => {
                if let Some(new) = choice_map.get(var) {
                    *var = *new;
                }
            }
            PirNode::Const(_) | PirNode::And { .. } | PirNode::Or { .. } => {}
        }
    }

    // Remap the id-keyed side tables consistently.
    provenance.leaf_probs = std::mem::take(&mut provenance.leaf_probs)
        .into_iter()
        .map(|(leaf, p)| (leaf_map[&leaf], p))
        .collect();
    provenance.leaf_atoms = std::mem::take(&mut provenance.leaf_atoms)
        .into_iter()
        .map(|(leaf, atom)| (leaf_map[&leaf], atom))
        .collect();
    provenance.choice_probs = std::mem::take(&mut provenance.choice_probs)
        .into_iter()
        .map(|(var, probs)| (choice_map[&var], probs))
        .collect();
    provenance.choice_sources = std::mem::take(&mut provenance.choice_sources)
        .into_iter()
        .map(|(var, source)| (choice_map[&var], source))
        .collect();

    provenance
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::apply_decision_order_hint;
    use crate::provenance::{extract_from_source, GroundAtom};

    /// Join-heavy fixture: first/last-layer edges participate in many shared
    /// path derivations, middle-layer edges in fewer, so fanouts differ and
    /// the hint must produce a non-trivial permutation — while preserving the
    /// (atom -> probability) association exactly.
    #[test]
    fn hint_permutes_leaves_by_fanout_and_preserves_atom_probs() {
        let mut source = String::new();
        for a in 1..=3 {
            source.push_str(&format!("0.3::edge(0, {a}).\n"));
        }
        for a in 1..=3 {
            for b in 4..=6 {
                source.push_str(&format!("0.4::edge({a}, {b}).\n"));
            }
        }
        for b in 4..=6 {
            source.push_str(&format!("0.5::edge({b}, 7).\n"));
        }
        source.push_str(
            "path(X, Y) :- edge(X, Y).\n\
             path(X, Z) :- edge(X, Y), path(Y, Z).\n\
             query(path(0, 7)).\n",
        );

        let before = extract_from_source(&source).expect("extract fixture");
        let atom_probs_before: BTreeMap<GroundAtom, f64> = before
            .leaf_atoms
            .iter()
            .map(|(leaf, atom)| (atom.clone(), before.leaf_probs[leaf]))
            .collect();
        let order_before: Vec<GroundAtom> = before.leaf_atoms.values().cloned().collect();

        let after = apply_decision_order_hint(extract_from_source(&source).expect("re-extract"));
        let atom_probs_after: BTreeMap<GroundAtom, f64> = after
            .leaf_atoms
            .iter()
            .map(|(leaf, atom)| (atom.clone(), after.leaf_probs[leaf]))
            .collect();
        let order_after: Vec<GroundAtom> = after.leaf_atoms.values().cloned().collect();

        assert_eq!(
            atom_probs_before, atom_probs_after,
            "hint must preserve atom->probability association"
        );
        assert_ne!(
            order_before, order_after,
            "hint must produce a non-trivial leaf order on a skewed-fanout fixture"
        );
        assert_eq!(after.pir.len(), before.pir.len());
    }
}
