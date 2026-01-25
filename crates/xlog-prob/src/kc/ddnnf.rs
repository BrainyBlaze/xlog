//! Decision-DNNF parser and CPU reference evaluator.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use xlog_core::{Result, XlogError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdnnfNodeKind {
    Or,
    And,
    True,
    False,
}

#[derive(Debug, Clone)]
pub struct DdnnfNode {
    pub kind: DdnnfNodeKind,
}

#[derive(Debug, Clone)]
pub struct DdnnfEdge {
    pub from: u32,
    pub to: u32,
    pub lits: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct DecisionDnnf {
    root: u32,
    nodes: BTreeMap<u32, DdnnfNode>,
    edges: Vec<DdnnfEdge>,
    outgoing: BTreeMap<u32, Vec<usize>>,
    max_var: u32,
}

impl DecisionDnnf {
    pub fn root(&self) -> u32 {
        self.root
    }

    pub fn max_var(&self) -> u32 {
        self.max_var
    }

    pub fn node_kind(&self, node_id: u32) -> Option<DdnnfNodeKind> {
        self.nodes.get(&node_id).map(|n| n.kind)
    }

    pub fn outgoing_edge_indices(&self, node_id: u32) -> Option<&[usize]> {
        self.outgoing.get(&node_id).map(|v| v.as_slice())
    }

    pub fn edge(&self, edge_idx: usize) -> Option<&DdnnfEdge> {
        self.edges.get(edge_idx)
    }

    pub fn parse_str(input: &str) -> Result<Self> {
        let mut nodes: BTreeMap<u32, DdnnfNode> = BTreeMap::new();
        let mut edges: Vec<DdnnfEdge> = Vec::new();
        let mut targets: HashSet<u32> = HashSet::new();
        let mut max_var: u32 = 0;

        for (line_no, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            let mut tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            if tokens.last() != Some(&"0") {
                return Err(XlogError::Compilation(format!(
                    "Decision-DNNF parse error at line {}: missing 0 terminator",
                    line_no + 1
                )));
            }
            tokens.pop();
            if tokens.is_empty() {
                return Err(XlogError::Compilation(format!(
                    "Decision-DNNF parse error at line {}: empty record before terminator",
                    line_no + 1
                )));
            }

            match tokens[0] {
                "o" | "a" | "t" | "f" => {
                    if tokens.len() < 2 {
                        return Err(XlogError::Compilation(format!(
                            "Decision-DNNF parse error at line {}: node record missing id",
                            line_no + 1
                        )));
                    }
                    let id: u32 = tokens[1].parse().map_err(|_| {
                        XlogError::Compilation(format!(
                            "Decision-DNNF parse error at line {}: invalid node id '{}'",
                            line_no + 1,
                            tokens[1]
                        ))
                    })?;

                    let kind = match tokens[0] {
                        "o" => DdnnfNodeKind::Or,
                        "a" => DdnnfNodeKind::And,
                        "t" => DdnnfNodeKind::True,
                        "f" => DdnnfNodeKind::False,
                        _ => unreachable!(),
                    };

                    if nodes.insert(id, DdnnfNode { kind }).is_some() {
                        return Err(XlogError::Compilation(format!(
                            "Decision-DNNF parse error at line {}: duplicate node id {}",
                            line_no + 1,
                            id
                        )));
                    }
                }
                _ => {
                    if tokens.len() < 2 {
                        return Err(XlogError::Compilation(format!(
                            "Decision-DNNF parse error at line {}: edge record missing dst",
                            line_no + 1
                        )));
                    }
                    let from: u32 = tokens[0].parse().map_err(|_| {
                        XlogError::Compilation(format!(
                            "Decision-DNNF parse error at line {}: invalid edge src '{}'",
                            line_no + 1,
                            tokens[0]
                        ))
                    })?;
                    let to: u32 = tokens[1].parse().map_err(|_| {
                        XlogError::Compilation(format!(
                            "Decision-DNNF parse error at line {}: invalid edge dst '{}'",
                            line_no + 1,
                            tokens[1]
                        ))
                    })?;

                    let mut lits: Vec<i32> = Vec::new();
                    for &tok in &tokens[2..] {
                        let lit: i32 = tok.parse().map_err(|_| {
                            XlogError::Compilation(format!(
                                "Decision-DNNF parse error at line {}: invalid literal '{}'",
                                line_no + 1,
                                tok
                            ))
                        })?;
                        if lit == 0 {
                            return Err(XlogError::Compilation(format!(
                                "Decision-DNNF parse error at line {}: literal cannot be 0",
                                line_no + 1
                            )));
                        }
                        max_var = max_var.max(lit.unsigned_abs());
                        lits.push(lit);
                    }

                    let edge_id = edges.len();
                    edges.push(DdnnfEdge { from, to, lits });
                    targets.insert(to);

                    // outgoing filled later after validation.
                    let _ = edge_id;
                }
            }
        }

        if nodes.is_empty() {
            return Err(XlogError::Compilation(
                "Decision-DNNF parse error: no nodes found".to_string(),
            ));
        }

        for edge in &edges {
            let from_kind = nodes.get(&edge.from).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Decision-DNNF parse error: edge references unknown src node {}",
                    edge.from
                ))
            })?;
            let _to_kind = nodes.get(&edge.to).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Decision-DNNF parse error: edge references unknown dst node {}",
                    edge.to
                ))
            })?;

            match from_kind.kind {
                DdnnfNodeKind::Or | DdnnfNodeKind::And => {}
                DdnnfNodeKind::True | DdnnfNodeKind::False => {
                    return Err(XlogError::Compilation(format!(
                        "Decision-DNNF parse error: leaf node {} cannot have outgoing edges",
                        edge.from
                    )));
                }
            }
        }

        let declared: BTreeSet<u32> = nodes.keys().copied().collect();
        let target_set: BTreeSet<u32> = targets.into_iter().collect();
        let roots: Vec<u32> = declared.difference(&target_set).copied().collect();
        let root = match roots.as_slice() {
            [only] => *only,
            [] => {
                return Err(XlogError::Compilation(
                    "Decision-DNNF parse error: could not infer root (no root candidates)"
                        .to_string(),
                ))
            }
            many => {
                return Err(XlogError::Compilation(format!(
                    "Decision-DNNF parse error: could not infer unique root (candidates: {:?})",
                    many
                )))
            }
        };

        let mut outgoing: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (idx, edge) in edges.iter().enumerate() {
            outgoing.entry(edge.from).or_default().push(idx);
        }

        // Optional cycle check (defensive).
        Self::check_acyclic(root, &nodes, &edges, &outgoing)?;

        Ok(Self {
            root,
            nodes,
            edges,
            outgoing,
            max_var,
        })
    }

    fn check_acyclic(
        root: u32,
        nodes: &BTreeMap<u32, DdnnfNode>,
        edges: &[DdnnfEdge],
        outgoing: &BTreeMap<u32, Vec<usize>>,
    ) -> Result<()> {
        let mut visiting: HashSet<u32> = HashSet::new();
        let mut visited: HashSet<u32> = HashSet::new();

        fn dfs(
            node_id: u32,
            nodes: &BTreeMap<u32, DdnnfNode>,
            edges: &[DdnnfEdge],
            outgoing: &BTreeMap<u32, Vec<usize>>,
            visiting: &mut HashSet<u32>,
            visited: &mut HashSet<u32>,
        ) -> Result<()> {
            if visited.contains(&node_id) {
                return Ok(());
            }
            if !visiting.insert(node_id) {
                return Err(XlogError::Compilation(format!(
                    "Decision-DNNF parse error: cycle detected at node {}",
                    node_id
                )));
            }

            let node = nodes.get(&node_id).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Decision-DNNF parse error: unknown node {} during cycle check",
                    node_id
                ))
            })?;

            match node.kind {
                DdnnfNodeKind::True | DdnnfNodeKind::False => {}
                DdnnfNodeKind::Or | DdnnfNodeKind::And => {
                    if let Some(out) = outgoing.get(&node_id) {
                        for &edge_idx in out {
                            let edge = &edges[edge_idx];
                            dfs(edge.to, nodes, edges, outgoing, visiting, visited)?;
                        }
                    }
                }
            }

            visiting.remove(&node_id);
            visited.insert(node_id);
            Ok(())
        }

        dfs(root, nodes, edges, outgoing, &mut visiting, &mut visited)
    }

    pub fn eval_log_wmc<F>(&self, var_log_weights: F) -> Result<f64>
    where
        F: Fn(u32) -> (f64, f64),
    {
        let mut memo: HashMap<u32, f64> = HashMap::new();

        fn logsumexp(values: &[f64]) -> f64 {
            let mut max = f64::NEG_INFINITY;
            for &v in values {
                if v > max {
                    max = v;
                }
            }
            if max.is_infinite() {
                return max;
            }
            let mut sum = 0.0;
            for &v in values {
                sum += (v - max).exp();
            }
            max + sum.ln()
        }

        fn eval_node<F>(
            node_id: u32,
            ddnnf: &DecisionDnnf,
            memo: &mut HashMap<u32, f64>,
            var_log_weights: &F,
        ) -> Result<f64>
        where
            F: Fn(u32) -> (f64, f64),
        {
            if let Some(&v) = memo.get(&node_id) {
                return Ok(v);
            }

            let node = ddnnf.nodes.get(&node_id).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Decision-DNNF eval error: unknown node {}",
                    node_id
                ))
            })?;

            let value = match node.kind {
                DdnnfNodeKind::True => 0.0,
                DdnnfNodeKind::False => f64::NEG_INFINITY,
                DdnnfNodeKind::And => {
                    let out = ddnnf.outgoing.get(&node_id).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Decision-DNNF eval error: AND node {} has no children",
                            node_id
                        ))
                    })?;

                    let mut acc = 0.0;
                    for &edge_idx in out {
                        let edge = &ddnnf.edges[edge_idx];
                        let child = eval_node(edge.to, ddnnf, memo, var_log_weights)?;
                        let mut lit_sum = 0.0;
                        for &lit in &edge.lits {
                            let var = lit.unsigned_abs();
                            let (t, f) = var_log_weights(var);
                            lit_sum += if lit > 0 { t } else { f };
                        }
                        acc += lit_sum + child;
                    }
                    acc
                }
                DdnnfNodeKind::Or => {
                    let out = ddnnf.outgoing.get(&node_id).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Decision-DNNF eval error: OR node {} has no children",
                            node_id
                        ))
                    })?;

                    let mut branch_vals: Vec<f64> = Vec::with_capacity(out.len());
                    for &edge_idx in out {
                        let edge = &ddnnf.edges[edge_idx];
                        let child = eval_node(edge.to, ddnnf, memo, var_log_weights)?;
                        let mut lit_sum = 0.0;
                        for &lit in &edge.lits {
                            let var = lit.unsigned_abs();
                            let (t, f) = var_log_weights(var);
                            lit_sum += if lit > 0 { t } else { f };
                        }
                        branch_vals.push(lit_sum + child);
                    }
                    logsumexp(&branch_vals)
                }
            };

            memo.insert(node_id, value);
            Ok(value)
        }

        eval_node(self.root, self, &mut memo, &var_log_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_eval_identity_variable() {
        // Represents the formula: x1
        let nnf = r#"
o 1 0
t 2 0
f 3 0
1 2 1 0
1 3 -1 0
"#;

        let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
        assert_eq!(ddnnf.root(), 1);
        assert_eq!(ddnnf.max_var(), 1);

        let p = 0.3_f64;
        let log_wmc = ddnnf
            .eval_log_wmc(|var| match var {
                1 => (p.ln(), (1.0 - p).ln()),
                _ => panic!("unexpected var {}", var),
            })
            .unwrap();

        assert!((log_wmc - p.ln()).abs() < 1e-9, "log_wmc={}", log_wmc);
    }

    #[test]
    fn test_parse_detects_missing_terminator() {
        let nnf = "t 1";
        let err = DecisionDnnf::parse_str(nnf).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("terminator"), "msg={}", msg);
    }
}
