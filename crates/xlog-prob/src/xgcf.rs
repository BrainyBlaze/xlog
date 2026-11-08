//! Conversion from compiler-owned decision-DNNF into the shared XGCF representation.

use std::collections::HashMap;

use xlog_core::{Result, XlogError};
use xlog_ir::semantic_measure::{levelize_xgcf, Xgcf, XgcfNodeType};

use crate::kc::ddnnf::{DdnnfEdge, DdnnfNodeKind, DecisionDnnf};

/// Converts a decision-DNNF circuit into XGCF and validates its topology.
pub fn from_ddnnf(ddnnf: &DecisionDnnf) -> Result<Xgcf> {
    XgcfBuilder::new(ddnnf).build()
}

struct XgcfBuilder<'a> {
    ddnnf: &'a DecisionDnnf,
    node_type: Vec<XgcfNodeType>,
    child_offsets: Vec<u32>,
    child_indices: Vec<u32>,
    lit: Vec<i32>,
    decision_var: Vec<u32>,
    decision_child_false: Vec<u32>,
    decision_child_true: Vec<u32>,
    lit_cache: HashMap<i32, u32>,
    ddnnf_cache: HashMap<u32, u32>,
    const0: u32,
    const1: u32,
}

impl<'a> XgcfBuilder<'a> {
    fn new(ddnnf: &'a DecisionDnnf) -> Self {
        let mut builder = Self {
            ddnnf,
            node_type: Vec::new(),
            child_offsets: Vec::new(),
            child_indices: Vec::new(),
            lit: Vec::new(),
            decision_var: Vec::new(),
            decision_child_false: Vec::new(),
            decision_child_true: Vec::new(),
            lit_cache: HashMap::new(),
            ddnnf_cache: HashMap::new(),
            const0: 0,
            const1: 0,
        };

        builder.const0 = builder.push_const(false);
        builder.const1 = builder.push_const(true);
        builder
    }

    fn push_base_node(&mut self, ty: XgcfNodeType) -> u32 {
        let idx = u32::try_from(self.node_type.len()).expect("XGCF node index overflow");
        self.node_type.push(ty);
        self.child_offsets.push(self.child_indices.len() as u32);
        self.lit.push(0);
        self.decision_var.push(0);
        self.decision_child_false.push(0);
        self.decision_child_true.push(0);
        idx
    }

    fn push_const(&mut self, value: bool) -> u32 {
        self.push_base_node(if value {
            XgcfNodeType::Const1
        } else {
            XgcfNodeType::Const0
        })
    }

    fn get_lit_node(&mut self, lit: i32) -> Result<u32> {
        if lit == 0 {
            return Err(XlogError::Compilation(
                "Cannot create XGCF LIT for 0 literal".to_string(),
            ));
        }
        if let Some(&idx) = self.lit_cache.get(&lit) {
            return Ok(idx);
        }
        let idx = self.push_base_node(XgcfNodeType::Lit);
        self.lit[idx as usize] = lit;
        self.lit_cache.insert(lit, idx);
        Ok(idx)
    }

    fn push_and(&mut self, mut children: Vec<u32>) -> Result<u32> {
        if children.contains(&self.const0) {
            return Ok(self.const0);
        }
        children.retain(|&c| c != self.const1);
        children.sort();
        children.dedup();
        match children.as_slice() {
            [] => Ok(self.const1),
            [only] => Ok(*only),
            _ => {
                let idx = self.push_base_node(XgcfNodeType::And);
                self.child_indices.extend_from_slice(&children);
                Ok(idx)
            }
        }
    }

    fn push_or(&mut self, mut children: Vec<u32>) -> Result<u32> {
        children.retain(|&c| c != self.const0);
        children.sort();
        children.dedup();
        match children.as_slice() {
            [] => Ok(self.const0),
            [only] => Ok(*only),
            _ => {
                let idx = self.push_base_node(XgcfNodeType::Or);
                self.child_indices.extend_from_slice(&children);
                Ok(idx)
            }
        }
    }

    fn push_decision(&mut self, var: u32, child_false: u32, child_true: u32) -> Result<u32> {
        if var == 0 {
            return Err(XlogError::Compilation(
                "Cannot create XGCF DECISION with var=0".to_string(),
            ));
        }
        let idx = self.push_base_node(XgcfNodeType::Decision);
        self.decision_var[idx as usize] = var;
        self.decision_child_false[idx as usize] = child_false;
        self.decision_child_true[idx as usize] = child_true;
        Ok(idx)
    }

    fn build(mut self) -> Result<Xgcf> {
        let root = self.convert_ddnnf_node(self.ddnnf.root())?;

        // Finalize offsets (sentinel).
        self.child_offsets.push(self.child_indices.len() as u32);

        let roots = vec![root];
        let (level_offsets, level_nodes) = levelize_xgcf(
            &self.node_type,
            &self.child_offsets,
            &self.child_indices,
            &self.decision_child_false,
            &self.decision_child_true,
            &roots,
        )?;

        Ok(Xgcf {
            node_type: self.node_type,
            child_offsets: self.child_offsets,
            child_indices: self.child_indices,
            lit: self.lit,
            decision_var: self.decision_var,
            decision_child_false: self.decision_child_false,
            decision_child_true: self.decision_child_true,
            roots,
            level_offsets,
            level_nodes,
        })
    }

    fn convert_ddnnf_node(&mut self, node_id: u32) -> Result<u32> {
        if let Some(&idx) = self.ddnnf_cache.get(&node_id) {
            return Ok(idx);
        }
        let kind = self.ddnnf.node_kind(node_id).ok_or_else(|| {
            XlogError::Compilation(format!("XGCF build error: unknown DDNNF node {}", node_id))
        })?;

        let idx = match kind {
            DdnnfNodeKind::True => self.const1,
            DdnnfNodeKind::False => self.const0,
            DdnnfNodeKind::And => {
                let out = self.ddnnf.outgoing_edge_indices(node_id).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "XGCF build error: AND node {} has no outgoing edges",
                        node_id
                    ))
                })?;
                let mut child_nodes: Vec<u32> = Vec::with_capacity(out.len());
                for &edge_idx in out {
                    child_nodes.push(self.convert_ddnnf_edge_branch(edge_idx, None)?);
                }
                self.push_and(child_nodes)?
            }
            DdnnfNodeKind::Or => {
                let out = self.ddnnf.outgoing_edge_indices(node_id).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "XGCF build error: OR node {} has no outgoing edges",
                        node_id
                    ))
                })?;
                if out.len() == 2 {
                    let e0 = self.ddnnf.edge(out[0]).ok_or_else(|| {
                        XlogError::Compilation(format!("XGCF build error: missing edge {}", out[0]))
                    })?;
                    let e1 = self.ddnnf.edge(out[1]).ok_or_else(|| {
                        XlogError::Compilation(format!("XGCF build error: missing edge {}", out[1]))
                    })?;

                    if let Some((var, edge_true, edge_false)) = infer_decision_var(e0, e1)? {
                        let edge_true = out[edge_true];
                        let edge_false = out[edge_false];
                        let child_true =
                            self.convert_ddnnf_edge_branch(edge_true, Some(var as i32))?;
                        let child_false =
                            self.convert_ddnnf_edge_branch(edge_false, Some(-(var as i32)))?;
                        self.push_decision(var, child_false, child_true)?
                    } else {
                        let mut child_nodes: Vec<u32> = Vec::with_capacity(out.len());
                        for &edge_idx in out {
                            child_nodes.push(self.convert_ddnnf_edge_branch(edge_idx, None)?);
                        }
                        self.push_or(child_nodes)?
                    }
                } else {
                    let mut child_nodes: Vec<u32> = Vec::with_capacity(out.len());
                    for &edge_idx in out {
                        child_nodes.push(self.convert_ddnnf_edge_branch(edge_idx, None)?);
                    }
                    self.push_or(child_nodes)?
                }
            }
        };

        self.ddnnf_cache.insert(node_id, idx);
        Ok(idx)
    }

    fn convert_ddnnf_edge_branch(&mut self, edge_idx: usize, drop_lit: Option<i32>) -> Result<u32> {
        let edge = self.ddnnf.edge(edge_idx).ok_or_else(|| {
            XlogError::Compilation(format!("XGCF build error: missing edge {}", edge_idx))
        })?;

        let child = self.convert_ddnnf_node(edge.to)?;

        let mut children: Vec<u32> = Vec::new();
        children.push(child);

        let mut dropped = false;
        for &lit in &edge.lits {
            if let Some(dl) = drop_lit {
                if !dropped && lit == dl {
                    dropped = true;
                    continue;
                }
            }
            children.push(self.get_lit_node(lit)?);
        }

        if let Some(dl) = drop_lit {
            if !dropped {
                return Err(XlogError::Compilation(format!(
                    "XGCF build error: expected to drop literal {} on edge {}->{} but not present",
                    dl, edge.from, edge.to
                )));
            }
        }

        self.push_and(children)
    }
}

fn infer_decision_var(e0: &DdnnfEdge, e1: &DdnnfEdge) -> Result<Option<(u32, usize, usize)>> {
    fn sign_map(lits: &[i32]) -> Result<HashMap<u32, bool>> {
        let mut map: HashMap<u32, bool> = HashMap::new();
        for &lit in lits {
            let var = lit.unsigned_abs();
            let sign = lit > 0;
            if let Some(prev) = map.insert(var, sign) {
                if prev != sign {
                    return Err(XlogError::Compilation(format!(
                        "XGCF build error: conflicting literals {} and {} in same branch",
                        var, lit
                    )));
                }
            }
        }
        Ok(map)
    }

    let m0 = sign_map(&e0.lits)?;
    let m1 = sign_map(&e1.lits)?;

    let mut candidates: Vec<u32> = Vec::new();
    for (var, &s0) in &m0 {
        if let Some(&s1) = m1.get(var) {
            if s0 != s1 {
                candidates.push(*var);
            }
        }
    }

    if candidates.len() != 1 {
        return Ok(None);
    }
    let var = candidates[0];

    let edge0_is_true = m0.get(&var).copied().unwrap_or(false);
    let (edge_true, edge_false) = if edge0_is_true {
        (0usize, 1usize)
    } else {
        (1usize, 0usize)
    };

    Ok(Some((var, edge_true, edge_false)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kc::ddnnf::DecisionDnnf;

    #[test]
    fn test_xgcf_matches_ddnnf_on_single_decision() {
        let nnf = r#"
o 1 0
t 2 0
f 3 0
1 2 1 0
1 3 -1 0
"#;
        let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
        let xgcf = from_ddnnf(&ddnnf).unwrap();

        let p = 0.37_f64;
        let w = |var: u32| match var {
            1 => (p.ln(), (1.0 - p).ln()),
            _ => panic!("unexpected var {}", var),
        };

        let a = ddnnf.eval_log_wmc(w).unwrap();
        let b = xgcf.eval_log_wmc(w).unwrap();
        assert!((a - b).abs() < 1e-9, "ddnnf={} xgcf={}", a, b);
    }

    #[test]
    fn test_xgcf_matches_ddnnf_on_two_stage_decision() {
        // Formula: x1 OR x2, represented as a decision on x1, then x2.
        let nnf = r#"
o 1 0
o 2 0
t 3 0
f 4 0
1 3 1 0
1 2 -1 0
2 3 2 0
2 4 -2 0
"#;
        let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
        let xgcf = from_ddnnf(&ddnnf).unwrap();

        let p1 = 0.2_f64;
        let p2 = 0.6_f64;
        let w = |var: u32| match var {
            1 => (p1.ln(), (1.0 - p1).ln()),
            2 => (p2.ln(), (1.0 - p2).ln()),
            _ => panic!("unexpected var {}", var),
        };

        let a = ddnnf.eval_log_wmc(w).unwrap();
        let b = xgcf.eval_log_wmc(w).unwrap();
        assert!((a - b).abs() < 1e-9, "ddnnf={} xgcf={}", a, b);
    }

    #[test]
    fn circuit_evaluators_share_logsumexp_contract() {
        let nnf = r#"
o 1 0
t 2 0
f 3 0
1 2 1 0
1 3 -1 0
"#;
        let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
        let xgcf = from_ddnnf(&ddnnf).unwrap();

        let ddnnf_error = ddnnf
            .eval_log_wmc(|_| (f64::NAN, f64::NEG_INFINITY))
            .expect_err("Decision-DNNF must reject NaN weights");
        let xgcf_error = xgcf
            .eval_log_wmc(|_| (f64::NAN, f64::NEG_INFINITY))
            .expect_err("XGCF must reject NaN weights");
        let weights = vec![(0.0, 0.0), (f64::NAN, f64::NEG_INFINITY)];
        let gradient_error = xgcf
            .eval_log_wmc_and_grads(&weights)
            .expect_err("XGCF gradient evaluation must reject NaN weights");

        for (name, error) in [
            ("Decision-DNNF", ddnnf_error),
            ("XGCF", xgcf_error),
            ("XGCF gradients", gradient_error),
        ] {
            assert!(
                matches!(&error, XlogError::Compilation(message) if message.contains("NaN")),
                "{name}: unexpected error: {error}"
            );
        }

        let ddnnf_log_z = ddnnf.eval_log_wmc(|_| (f64::INFINITY, -1.0)).unwrap();
        let xgcf_log_z = xgcf.eval_log_wmc(|_| (f64::INFINITY, -1.0)).unwrap();
        let weights = vec![(0.0, 0.0), (f64::INFINITY, -1.0)];
        let gradient_error = xgcf
            .eval_log_wmc_and_grads(&weights)
            .expect_err("Decision normalization must reject non-finite gradients");

        for (name, value) in [("Decision-DNNF", ddnnf_log_z), ("XGCF", xgcf_log_z)] {
            assert!(
                value.is_infinite() && value.is_sign_positive(),
                "{name}: expected positive infinity, got {value}"
            );
        }
        assert!(
            matches!(&gradient_error, XlogError::Compilation(message) if message.contains("gradient") && message.contains("non-finite")),
            "XGCF gradients: unexpected error: {gradient_error}"
        );
    }

    #[test]
    fn deterministic_circuit_evaluators_reject_nan_weights() {
        let nnf = r#"
a 1 0
t 2 0
1 2 1 0
"#;
        let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
        let xgcf = from_ddnnf(&ddnnf).unwrap();

        let ddnnf_error = ddnnf
            .eval_log_wmc(|_| (f64::NAN, 0.0))
            .expect_err("deterministic Decision-DNNF must reject NaN weights");
        let xgcf_error = xgcf
            .eval_log_wmc(|_| (f64::NAN, 0.0))
            .expect_err("deterministic XGCF must reject NaN weights");
        let gradient_error = xgcf
            .eval_log_wmc_and_grads(&[(0.0, 0.0), (f64::NAN, 0.0)])
            .expect_err("deterministic XGCF gradients must reject NaN weights");

        for (name, error) in [
            ("Decision-DNNF", ddnnf_error),
            ("XGCF", xgcf_error),
            ("XGCF gradients", gradient_error),
        ] {
            assert!(
                matches!(&error, XlogError::Compilation(message) if message.contains("NaN")),
                "{name}: unexpected error: {error}"
            );
        }
    }

    #[test]
    fn deterministic_circuit_evaluators_reject_computed_nan() {
        let nnf = r#"
a 1 0
t 2 0
1 2 1 -2 0
"#;
        let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
        let xgcf = from_ddnnf(&ddnnf).unwrap();
        let weights = [(0.0, 0.0), (f64::INFINITY, 0.0), (0.0, f64::NEG_INFINITY)];
        let weight = |var: u32| weights[var as usize];

        let ddnnf_error = ddnnf
            .eval_log_wmc(weight)
            .expect_err("Decision-DNNF must reject +inf + -inf");
        let xgcf_error = xgcf
            .eval_log_wmc(weight)
            .expect_err("XGCF must reject +inf + -inf");
        let gradient_error = xgcf
            .eval_log_wmc_and_grads(&weights)
            .expect_err("XGCF gradients must reject computed NaN");

        for (name, error) in [("Decision-DNNF", ddnnf_error), ("XGCF", xgcf_error)] {
            assert!(
                matches!(&error, XlogError::Compilation(message) if message.contains("NaN")),
                "{name}: unexpected error: {error}"
            );
        }
        assert!(
            matches!(&gradient_error, XlogError::Compilation(message) if message.contains("NaN")),
            "XGCF gradients: unexpected error: {gradient_error}"
        );
    }

    #[test]
    fn gradient_evaluation_rejects_nan_when_another_weight_is_positive_infinity() {
        let nnf = r#"
a 1 0
t 2 0
1 2 1 2 0
"#;
        let ddnnf = DecisionDnnf::parse_str(nnf).unwrap();
        let xgcf = from_ddnnf(&ddnnf).unwrap();
        let weights = [(0.0, 0.0), (f64::INFINITY, 0.0), (f64::NAN, 0.0)];

        let error = xgcf
            .eval_log_wmc_and_grads(&weights)
            .expect_err("NaN must take precedence over positive infinity");
        assert!(
            matches!(&error, XlogError::Compilation(message) if message.contains("NaN")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn reserved_weight_slot_zero_does_not_affect_value_or_gradients() {
        let nnf = r#"
a 1 0
t 2 0
1 2 1 0
"#;
        let xgcf = from_ddnnf(&DecisionDnnf::parse_str(nnf).unwrap()).unwrap();
        let baseline_weights = [(0.0, 0.0), (-0.25, -0.75)];
        let reserved_non_finite = [(f64::NAN, f64::INFINITY), (-0.25, -0.75)];

        let baseline_value = xgcf
            .eval_log_wmc(|var| baseline_weights[var as usize])
            .unwrap();
        let reserved_value = xgcf
            .eval_log_wmc(|var| reserved_non_finite[var as usize])
            .unwrap();
        assert_eq!(reserved_value, baseline_value);

        let baseline_grads = xgcf.eval_log_wmc_and_grads(&baseline_weights).unwrap();
        let reserved_grads = xgcf
            .eval_log_wmc_and_grads(&reserved_non_finite)
            .expect("reserved slot 0 must not be validated or consumed");
        assert_eq!(reserved_grads, baseline_grads);
    }

    #[test]
    fn deterministic_gradient_allows_selected_and_unused_positive_infinity() {
        let nnf = r#"
a 1 0
t 2 0
1 2 3 0
"#;
        let xgcf = from_ddnnf(&DecisionDnnf::parse_str(nnf).unwrap()).unwrap();
        let weights = [
            (0.0, 0.0),
            (f64::INFINITY, f64::INFINITY),
            (f64::INFINITY, f64::INFINITY),
            (f64::INFINITY, f64::INFINITY),
        ];

        let (log_z, grad_true, grad_false) = xgcf
            .eval_log_wmc_and_grads(&weights)
            .expect("+inf is valid when deterministic backward outputs stay finite");
        assert!(log_z.is_infinite() && log_z.is_sign_positive());
        assert_eq!(grad_true, vec![0.0, 0.0, 0.0, 1.0]);
        assert_eq!(grad_false, vec![0.0; 4]);
    }
}
