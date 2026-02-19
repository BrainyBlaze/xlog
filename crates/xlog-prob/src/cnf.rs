//! CNF emission for PIR via Tseitin encoding (DIMACS).

use std::collections::{BTreeMap, HashMap, HashSet};

use xlog_core::{Result, XlogError};

use crate::pir::{ChoiceVarId, LeafId, PirGraph, PirNode, PirNodeId};

/// A CNF formula in DIMACS-style representation.
#[derive(Debug, Clone, Default)]
pub struct CnfFormula {
    num_vars: u32,
    clauses: Vec<Vec<i32>>,
}

impl CnfFormula {
    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    pub fn clauses(&self) -> &[Vec<i32>] {
        &self.clauses
    }

    pub fn to_dimacs(&self) -> String {
        let mut out = String::new();
        out.push_str("c xlog-prob cnf\n");
        out.push_str(&format!("p cnf {} {}\n", self.num_vars, self.clauses.len()));
        for clause in &self.clauses {
            for lit in clause {
                out.push_str(&format!("{} ", lit));
            }
            out.push_str("0\n");
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct CnfEncoding {
    pub cnf: CnfFormula,
    pub node_var: BTreeMap<PirNodeId, u32>,
    pub leaf_var: BTreeMap<LeafId, u32>,
    pub choice_var: BTreeMap<ChoiceVarId, u32>,
}

/// FNV-1a 64-bit — deterministic, process-independent.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Compute a canonical hash for each reachable PIR node.
/// The hash depends only on node structure (variant + leaf/choice IDs + children's
/// canonical hashes), not on PirNodeId numeric values.
fn canonical_hashes(
    pir: &PirGraph,
    levels: &[Vec<PirNodeId>],
) -> HashMap<PirNodeId, u64> {
    let mut canon: HashMap<PirNodeId, u64> = HashMap::new();
    for level in levels {
        for &id in level {
            let node = pir.node(id).unwrap();
            let h = match node {
                PirNode::Const(b) => fnv1a(&[0, *b as u8]),
                PirNode::Lit { leaf } => {
                    let mut buf = [0u8; 5];
                    buf[0] = 1;
                    buf[1..5].copy_from_slice(&leaf.as_u32().to_le_bytes());
                    fnv1a(&buf)
                }
                PirNode::NegLit { leaf } => {
                    let mut buf = [0u8; 5];
                    buf[0] = 2;
                    buf[1..5].copy_from_slice(&leaf.as_u32().to_le_bytes());
                    fnv1a(&buf)
                }
                PirNode::And { children } => {
                    let mut child_h: Vec<u64> =
                        children.iter().map(|c| canon[c]).collect();
                    child_h.sort();
                    let mut buf = vec![3u8];
                    for h in child_h {
                        buf.extend_from_slice(&h.to_le_bytes());
                    }
                    fnv1a(&buf)
                }
                PirNode::Or { children } => {
                    let mut child_h: Vec<u64> =
                        children.iter().map(|c| canon[c]).collect();
                    child_h.sort();
                    let mut buf = vec![4u8];
                    for h in child_h {
                        buf.extend_from_slice(&h.to_le_bytes());
                    }
                    fnv1a(&buf)
                }
                PirNode::Decision {
                    var,
                    child_false,
                    child_true,
                } => {
                    let mut buf = vec![5u8];
                    buf.extend_from_slice(&var.as_u32().to_le_bytes());
                    buf.extend_from_slice(&canon[child_false].to_le_bytes());
                    buf.extend_from_slice(&canon[child_true].to_le_bytes());
                    fnv1a(&buf)
                }
            };
            canon.insert(id, h);
        }
    }
    canon
}

pub fn encode_cnf(pir: &PirGraph, roots: &[PirNodeId]) -> Result<CnfEncoding> {
    if roots.is_empty() {
        return Err(XlogError::Compilation(
            "Cannot encode CNF for empty PIR root set".to_string(),
        ));
    }

    let mut visited: HashSet<PirNodeId> = HashSet::new();
    let mut leaf_ids: HashSet<LeafId> = HashSet::new();
    let mut choice_ids: HashSet<ChoiceVarId> = HashSet::new();

    let mut stack: Vec<PirNodeId> = roots.to_vec();
    while let Some(node_id) = stack.pop() {
        if !visited.insert(node_id) {
            continue;
        }

        let node = pir.node(node_id).ok_or_else(|| {
            XlogError::Compilation(format!(
                "Invalid PIR node id while encoding CNF: {:?}",
                node_id
            ))
        })?;

        match node {
            PirNode::Const(_) => {}
            PirNode::Lit { leaf } | PirNode::NegLit { leaf } => {
                leaf_ids.insert(*leaf);
            }
            PirNode::And { children } | PirNode::Or { children } => {
                stack.extend(children.iter().copied());
            }
            PirNode::Decision {
                var,
                child_false,
                child_true,
            } => {
                choice_ids.insert(*var);
                stack.push(*child_false);
                stack.push(*child_true);
            }
        }
    }

    let mut leaf_list: Vec<LeafId> = leaf_ids.into_iter().collect();
    leaf_list.sort();
    let mut choice_list: Vec<ChoiceVarId> = choice_ids.into_iter().collect();
    choice_list.sort();

    let mut next_var: u32 = 1;
    let mut leaf_var: BTreeMap<LeafId, u32> = BTreeMap::new();
    for leaf in leaf_list {
        leaf_var.insert(leaf, next_var);
        next_var += 1;
    }

    let mut choice_var: BTreeMap<ChoiceVarId, u32> = BTreeMap::new();
    for choice in choice_list {
        choice_var.insert(choice, next_var);
        next_var += 1;
    }

    // Compute canonical hashes for structural ordering independent of PirNodeId values.
    let levels = pir.levelize(roots)?;
    let canon = canonical_hashes(pir, &levels);

    let mut node_ids: Vec<PirNodeId> = visited.into_iter().collect();
    node_ids.sort_by_key(|id| canon.get(id).copied().unwrap_or(0));

    let mut node_var: BTreeMap<PirNodeId, u32> = BTreeMap::new();
    for node_id in node_ids {
        let node = pir.node(node_id).ok_or_else(|| {
            XlogError::Compilation(format!(
                "Invalid PIR node id while encoding CNF: {:?}",
                node_id
            ))
        })?;

        let var_id = match node {
            PirNode::Lit { leaf } => *leaf_var.get(leaf).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Missing CNF var for PIR leaf {:?} referenced by node {:?}",
                    leaf, node_id
                ))
            })?,
            PirNode::NegLit { .. } => {
                // NegLit gets its own variable, which will be constrained to !leaf_var
                let v = next_var;
                next_var += 1;
                v
            }
            _ => {
                let v = next_var;
                next_var += 1;
                v
            }
        };

        node_var.insert(node_id, var_id);
    }

    let num_vars = next_var - 1;
    let mut clauses: Vec<Vec<i32>> = Vec::new();

    // Re-sort each level by canonical hash for deterministic clause emission.
    for level in &levels {
        let mut sorted_level = level.clone();
        sorted_level.sort_by_key(|id| canon.get(id).copied().unwrap_or(0));
        for node_id in sorted_level {
            let node = pir.node(node_id).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Invalid PIR node id while emitting CNF clauses: {:?}",
                    node_id
                ))
            })?;
            let v = *node_var.get(&node_id).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Missing CNF var for PIR node {:?} while emitting clauses",
                    node_id
                ))
            })? as i32;

            match node {
                PirNode::Const(true) => clauses.push(vec![v]),
                PirNode::Const(false) => clauses.push(vec![-v]),
                PirNode::Lit { .. } => {}
                PirNode::NegLit { leaf } => {
                    // NegLit uses opposite polarity: node_var <-> !leaf_var
                    let leaf_v = *leaf_var.get(leaf).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Missing CNF var for NegLit leaf {:?} at node {:?}",
                            leaf, node_id
                        ))
                    })? as i32;
                    // v <-> !leaf_v  means:  (v | leaf_v) & (!v | !leaf_v)
                    clauses.push(vec![v, leaf_v]); // v=false -> leaf_v=true (contrapositive: !leaf_v -> v)
                    clauses.push(vec![-v, -leaf_v]); // v=true -> leaf_v=false
                }
                PirNode::And { children } => {
                    if children.is_empty() {
                        clauses.push(vec![v]);
                        continue;
                    }
                    // Sort children by canonical hash for deterministic clause order.
                    let mut sorted_children = children.clone();
                    sorted_children
                        .sort_by_key(|id| canon.get(id).copied().unwrap_or(0));
                    for &child in &sorted_children {
                        let c = *node_var.get(&child).ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "Missing CNF var for AND child {:?} of {:?}",
                                child, node_id
                            ))
                        })? as i32;
                        clauses.push(vec![-v, c]);
                    }
                    let mut clause = Vec::with_capacity(children.len() + 1);
                    for &child in &sorted_children {
                        let c = *node_var.get(&child).ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "Missing CNF var for AND child {:?} of {:?}",
                                child, node_id
                            ))
                        })? as i32;
                        clause.push(-c);
                    }
                    clause.push(v);
                    clauses.push(clause);
                }
                PirNode::Or { children } => {
                    if children.is_empty() {
                        clauses.push(vec![-v]);
                        continue;
                    }
                    // Sort children by canonical hash for deterministic clause order.
                    let mut sorted_children = children.clone();
                    sorted_children
                        .sort_by_key(|id| canon.get(id).copied().unwrap_or(0));
                    for &child in &sorted_children {
                        let c = *node_var.get(&child).ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "Missing CNF var for OR child {:?} of {:?}",
                                child, node_id
                            ))
                        })? as i32;
                        clauses.push(vec![-c, v]);
                    }
                    let mut clause = Vec::with_capacity(children.len() + 1);
                    clause.push(-v);
                    for &child in &sorted_children {
                        let c = *node_var.get(&child).ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "Missing CNF var for OR child {:?} of {:?}",
                                child, node_id
                            ))
                        })? as i32;
                        clause.push(c);
                    }
                    clauses.push(clause);
                }
                PirNode::Decision {
                    var,
                    child_false,
                    child_true,
                } => {
                    let x = *choice_var.get(var).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Missing CNF var for decision variable {:?} at node {:?}",
                            var, node_id
                        ))
                    })? as i32;

                    let f = *node_var.get(child_false).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Missing CNF var for decision false child {:?} at node {:?}",
                            child_false, node_id
                        ))
                    })? as i32;
                    let t = *node_var.get(child_true).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Missing CNF var for decision true child {:?} at node {:?}",
                            child_true, node_id
                        ))
                    })? as i32;

                    // v <-> (x ? t : f)
                    clauses.push(vec![-x, -t, v]); // (x & t) -> v
                    clauses.push(vec![x, -f, v]); // (!x & f) -> v
                    clauses.push(vec![-x, t, -v]); // (v & x) -> t
                    clauses.push(vec![x, f, -v]); // (v & !x) -> f
                }
            }
        }
    }

    Ok(CnfEncoding {
        cnf: CnfFormula { num_vars, clauses },
        node_var,
        leaf_var,
        choice_var,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::{ChoiceVarId, LeafId, PirGraph};

    fn is_sat_with_unit_clauses(cnf: &CnfFormula, units: &[i32]) -> bool {
        let num_vars = cnf.num_vars() as usize;
        assert!(
            num_vars <= 20,
            "test sat checker only supports small CNFs (vars={})",
            num_vars
        );

        let mut clauses: Vec<&[i32]> = cnf.clauses().iter().map(|c| c.as_slice()).collect();
        let unit_clauses: Vec<Vec<i32>> = units.iter().map(|&u| vec![u]).collect();
        for uc in &unit_clauses {
            clauses.push(uc.as_slice());
        }

        'assign: for mask in 0..(1u64 << num_vars) {
            for clause in &clauses {
                let mut clause_sat = false;
                for &lit in *clause {
                    let var = lit.unsigned_abs() as usize;
                    assert!(var >= 1 && var <= num_vars);
                    let bit = (mask >> (var - 1)) & 1;
                    let val = bit == 1;
                    let lit_sat = if lit > 0 { val } else { !val };
                    if lit_sat {
                        clause_sat = true;
                        break;
                    }
                }
                if !clause_sat {
                    continue 'assign;
                }
            }
            return true;
        }
        false
    }

    #[test]
    fn test_encode_cnf_does_not_force_root_assignment() {
        let mut pir = PirGraph::new();
        let a = pir.lit(LeafId::new(0));

        let encoding = encode_cnf(&pir, &[a]).unwrap();
        let var_a = *encoding.leaf_var.get(&LeafId::new(0)).unwrap() as i32;

        assert!(is_sat_with_unit_clauses(&encoding.cnf, &[-var_a]));
    }

    #[test]
    fn test_tseitin_and_requires_both_children() {
        let mut pir = PirGraph::new();
        let a = pir.lit(LeafId::new(0));
        let b = pir.lit(LeafId::new(1));
        let root = pir.and(vec![a, b]);

        let encoding = encode_cnf(&pir, &[root]).unwrap();
        let var_root = *encoding.node_var.get(&root).unwrap() as i32;
        let var_a = *encoding.leaf_var.get(&LeafId::new(0)).unwrap() as i32;
        let var_b = *encoding.leaf_var.get(&LeafId::new(1)).unwrap() as i32;

        assert!(is_sat_with_unit_clauses(
            &encoding.cnf,
            &[var_root, var_a, var_b]
        ));
        assert!(!is_sat_with_unit_clauses(
            &encoding.cnf,
            &[var_root, var_a, -var_b]
        ));
        assert!(!is_sat_with_unit_clauses(
            &encoding.cnf,
            &[var_root, -var_a, var_b]
        ));
        assert!(!is_sat_with_unit_clauses(
            &encoding.cnf,
            &[var_root, -var_a, -var_b]
        ));
    }

    #[test]
    fn test_tseitin_or_requires_one_child() {
        let mut pir = PirGraph::new();
        let a = pir.lit(LeafId::new(0));
        let b = pir.lit(LeafId::new(1));
        let root = pir.or(vec![a, b]);

        let encoding = encode_cnf(&pir, &[root]).unwrap();
        let var_root = *encoding.node_var.get(&root).unwrap() as i32;
        let var_a = *encoding.leaf_var.get(&LeafId::new(0)).unwrap() as i32;
        let var_b = *encoding.leaf_var.get(&LeafId::new(1)).unwrap() as i32;

        assert!(is_sat_with_unit_clauses(
            &encoding.cnf,
            &[var_root, var_a, var_b]
        ));
        assert!(is_sat_with_unit_clauses(
            &encoding.cnf,
            &[var_root, var_a, -var_b]
        ));
        assert!(is_sat_with_unit_clauses(
            &encoding.cnf,
            &[var_root, -var_a, var_b]
        ));
        assert!(!is_sat_with_unit_clauses(
            &encoding.cnf,
            &[var_root, -var_a, -var_b]
        ));
    }

    #[test]
    fn test_tseitin_decision_mux_matches_choice_var() {
        let mut pir = PirGraph::new();
        let t = pir.const_true();
        let f = pir.const_false();
        let root = pir.decision(ChoiceVarId::new(0), f, t);

        let encoding = encode_cnf(&pir, &[root]).unwrap();
        let var_root = *encoding.node_var.get(&root).unwrap() as i32;
        let x = *encoding.choice_var.get(&ChoiceVarId::new(0)).unwrap() as i32;

        assert!(is_sat_with_unit_clauses(&encoding.cnf, &[var_root, x]));
        assert!(!is_sat_with_unit_clauses(&encoding.cnf, &[var_root, -x]));
    }

    #[test]
    fn test_dimacs_is_well_formed() {
        let mut pir = PirGraph::new();
        let a = pir.lit(LeafId::new(0));
        let root = pir.or(vec![a]);

        let encoding = encode_cnf(&pir, &[root]).unwrap();
        let dimacs = encoding.cnf.to_dimacs();

        assert!(dimacs.contains("\np cnf "));
        assert!(dimacs.lines().any(|l| l.ends_with('0')));
    }

    #[test]
    fn test_tseitin_neg_lit_uses_negated_polarity() {
        let mut pir = PirGraph::new();
        let a = pir.neg_lit(LeafId::new(0)); // NegLit instead of Lit
        let root = pir.or(vec![a]);

        let encoding = encode_cnf(&pir, &[root]).unwrap();
        let var_root = *encoding.node_var.get(&root).unwrap() as i32;
        let var_a = *encoding.leaf_var.get(&LeafId::new(0)).unwrap() as i32;

        // When NegLit node is true, the underlying leaf var should be FALSE
        // (because NegLit represents "not leaf")
        // So if root is true and NegLit(a) makes it true, leaf a must be false
        assert!(
            is_sat_with_unit_clauses(&encoding.cnf, &[var_root, -var_a]),
            "root=true with leaf=false should be SAT"
        );

        // If leaf is true, then NegLit(leaf) is false, so root (which is Or of NegLit) should be false
        // Actually wait - the Or node can be true or false. Let's think more carefully.
        //
        // The CNF encodes: root <-> Or(neg_lit_node)
        //                  neg_lit_node <-> !leaf
        // So: root=true requires neg_lit_node=true requires leaf=false
        assert!(
            !is_sat_with_unit_clauses(&encoding.cnf, &[var_root, var_a]),
            "root=true with leaf=true should be UNSAT (NegLit of true leaf is false)"
        );
    }
}
