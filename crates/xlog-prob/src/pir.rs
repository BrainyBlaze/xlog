//! Provenance IR (PIR) for probabilistic compilation.

use xlog_core::{Result, XlogError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PirNodeId(u32);

impl PirNodeId {
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LeafId(u32);

impl LeafId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChoiceVarId(u32);

impl ChoiceVarId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PirNode {
    Const(bool),
    Lit { leaf: LeafId },
    And { children: Vec<PirNodeId> },
    Or { children: Vec<PirNodeId> },
    Decision {
        var: ChoiceVarId,
        child_false: PirNodeId,
        child_true: PirNodeId,
    },
}

#[derive(Debug, Default, Clone)]
pub struct PirGraph {
    nodes: Vec<PirNode>,
}

impl PirGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn node(&self, id: PirNodeId) -> Option<&PirNode> {
        self.nodes.get(id.0 as usize)
    }

    fn push_node(&mut self, node: PirNode) -> PirNodeId {
        let id = PirNodeId(u32::try_from(self.nodes.len()).expect("PIR node id overflow"));
        self.nodes.push(node);
        id
    }

    pub fn const_true(&mut self) -> PirNodeId {
        self.push_node(PirNode::Const(true))
    }

    pub fn const_false(&mut self) -> PirNodeId {
        self.push_node(PirNode::Const(false))
    }

    pub fn lit(&mut self, leaf: LeafId) -> PirNodeId {
        self.push_node(PirNode::Lit { leaf })
    }

    pub fn and(&mut self, children: Vec<PirNodeId>) -> PirNodeId {
        self.push_node(PirNode::And { children })
    }

    pub fn or(&mut self, children: Vec<PirNodeId>) -> PirNodeId {
        self.push_node(PirNode::Or { children })
    }

    pub fn decision(&mut self, var: ChoiceVarId, child_false: PirNodeId, child_true: PirNodeId) -> PirNodeId {
        self.push_node(PirNode::Decision {
            var,
            child_false,
            child_true,
        })
    }

    pub fn levelize(&self, roots: &[PirNodeId]) -> Result<Vec<Vec<PirNodeId>>> {
        use std::collections::{HashMap, HashSet};

        let mut visiting: HashSet<PirNodeId> = HashSet::new();
        let mut levels: HashMap<PirNodeId, u32> = HashMap::new();

        fn compute_level(
            graph: &PirGraph,
            id: PirNodeId,
            visiting: &mut HashSet<PirNodeId>,
            levels: &mut HashMap<PirNodeId, u32>,
        ) -> Result<u32> {
            if let Some(&lvl) = levels.get(&id) {
                return Ok(lvl);
            }
            if !visiting.insert(id) {
                return Err(XlogError::Compilation(format!(
                    "Cycle detected while levelizing PIR at node {:?}",
                    id
                )));
            }

            let node = graph.node(id).ok_or_else(|| {
                XlogError::Compilation(format!("Invalid PIR node id while levelizing: {:?}", id))
            })?;

            let lvl = match node {
                PirNode::Const(_) | PirNode::Lit { .. } => 0,
                PirNode::And { children } | PirNode::Or { children } => {
                    let mut max_child = 0u32;
                    for &child in children {
                        let child_lvl = compute_level(graph, child, visiting, levels)?;
                        max_child = max_child.max(child_lvl);
                    }
                    max_child + 1
                }
                PirNode::Decision {
                    child_false,
                    child_true,
                    ..
                } => {
                    let lf = compute_level(graph, *child_false, visiting, levels)?;
                    let lt = compute_level(graph, *child_true, visiting, levels)?;
                    lf.max(lt) + 1
                }
            };

            visiting.remove(&id);
            levels.insert(id, lvl);
            Ok(lvl)
        }

        for &root in roots {
            compute_level(self, root, &mut visiting, &mut levels)?;
        }

        let max_level = levels.values().copied().max().unwrap_or(0);
        let mut buckets: Vec<Vec<PirNodeId>> = vec![Vec::new(); (max_level as usize) + 1];

        let mut ids: Vec<PirNodeId> = levels.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        for id in ids {
            let lvl = levels[&id] as usize;
            buckets[lvl].push(id);
        }

        Ok(buckets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levelize_simple_dag() {
        let mut pir = PirGraph::new();
        let a = pir.lit(LeafId(0));
        let b = pir.lit(LeafId(1));
        let and_ab = pir.and(vec![a, b]);
        let root = pir.or(vec![and_ab, a]);

        let levels = pir.levelize(&[root]).unwrap();

        assert!(!levels.is_empty());
        assert!(levels.iter().any(|lvl| lvl.contains(&root)));
        assert!(levels.iter().any(|lvl| lvl.contains(&a)));
        assert!(levels.iter().any(|lvl| lvl.contains(&b)));
    }

    #[test]
    fn test_levelize_decision_node() {
        let mut pir = PirGraph::new();
        let t = pir.const_true();
        let f = pir.const_false();
        let d = pir.decision(ChoiceVarId(0), f, t);

        let levels = pir.levelize(&[d]).unwrap();
        assert!(levels.iter().any(|lvl| lvl.contains(&d)));
        assert!(levels.iter().any(|lvl| lvl.contains(&t)));
        assert!(levels.iter().any(|lvl| lvl.contains(&f)));
    }
}
