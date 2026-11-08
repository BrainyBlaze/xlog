//! XLOG GPU Circuit Format (XGCF) - host-side representation and CPU evaluator.

use std::collections::HashMap;

use xlog_core::{Result, XlogError};

use crate::circuit_numeric::{
    circuit_logsumexp, validate_circuit_gradient_values, validate_circuit_log_weight_pair,
    validate_circuit_value,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Node opcode in the structure-of-arrays XLOG GPU Circuit Format.
pub enum XgcfNodeType {
    /// Boolean false constant.
    Const0 = 0,
    /// Boolean true constant.
    Const1 = 1,
    /// Signed literal whose variable is stored in [`Xgcf::lit`].
    Lit = 2,
    /// Conjunction over the node's child range.
    And = 3,
    /// Deterministic disjunction over the node's child range.
    Or = 4,
    /// Binary decision with explicit false and true children.
    Decision = 5,
}

#[derive(Debug, Clone)]
/// Host-side structure-of-arrays representation of a compiled probabilistic circuit.
pub struct Xgcf {
    /// Opcode for every node.
    pub node_type: Vec<XgcfNodeType>,
    /// CSR offsets into [`Self::child_indices`] for each node.
    pub child_offsets: Vec<u32>,
    /// Flattened children of conjunction and disjunction nodes.
    pub child_indices: Vec<u32>,
    /// Signed DIMACS literal for each literal node, or zero otherwise.
    pub lit: Vec<i32>,
    /// Decision variable for each decision node, or zero otherwise.
    pub decision_var: Vec<u32>,
    /// False-branch child for each decision node.
    pub decision_child_false: Vec<u32>,
    /// True-branch child for each decision node.
    pub decision_child_true: Vec<u32>,
    /// Circuit root node indices.
    pub roots: Vec<u32>,
    /// CSR offsets into [`Self::level_nodes`] for bottom-up evaluation levels.
    pub level_offsets: Vec<u32>,
    /// Node indices ordered by bottom-up evaluation level.
    pub level_nodes: Vec<u32>,
}

impl Xgcf {
    /// Return a semantically equivalent circuit that is **smooth** with respect to
    /// the subset of variables marked as random in `is_random_var`.
    ///
    /// Smoothness guarantees that, for any OR/DECISION node, all branches mention
    /// the same set of random variables. This makes WMC evaluation and gradients
    /// correct even when evidence/queries force random variables in non-smooth
    /// Decision-DNNF outputs.
    pub fn smooth_random_vars(&self, is_random_var: &[bool]) -> Result<Self> {
        XgcfSmoother::new(self, is_random_var)?.smooth()
    }

    /// Evaluates the circuit's weighted model count in log space on the CPU.
    pub fn eval_log_wmc<F>(&self, var_log_weights: F) -> Result<f64>
    where
        F: Fn(u32) -> (f64, f64),
    {
        if self.roots.len() != 1 {
            return Err(XlogError::Compilation(format!(
                "XGCF eval expects exactly 1 root, got {}",
                self.roots.len()
            )));
        }

        let n = self.node_type.len();
        if self.child_offsets.len() != n + 1 {
            return Err(XlogError::Compilation(format!(
                "XGCF invariant violation: child_offsets len {} != num_nodes+1 ({})",
                self.child_offsets.len(),
                n + 1
            )));
        }
        if self.lit.len() != n
            || self.decision_var.len() != n
            || self.decision_child_false.len() != n
            || self.decision_child_true.len() != n
        {
            return Err(XlogError::Compilation(
                "XGCF invariant violation: per-node arrays length mismatch".to_string(),
            ));
        }
        if self.level_offsets.is_empty() || *self.level_offsets.first().unwrap() != 0 {
            return Err(XlogError::Compilation(
                "XGCF invariant violation: level_offsets must start at 0".to_string(),
            ));
        }
        if *self.level_offsets.last().unwrap() != self.level_nodes.len() as u32 {
            return Err(XlogError::Compilation(
                "XGCF invariant violation: level_offsets last != level_nodes.len".to_string(),
            ));
        }

        let mut values: Vec<f64> = vec![0.0; n];

        let num_levels = self.level_offsets.len() - 1;
        for level in 0..num_levels {
            let start = self.level_offsets[level] as usize;
            let end = self.level_offsets[level + 1] as usize;
            for &node_u32 in &self.level_nodes[start..end] {
                let idx = node_u32 as usize;
                let v = match self.node_type[idx] {
                    XgcfNodeType::Const0 => f64::NEG_INFINITY,
                    XgcfNodeType::Const1 => 0.0,
                    XgcfNodeType::Lit => {
                        let lit = self.lit[idx];
                        if lit == 0 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF invariant violation: LIT node {} has lit=0",
                                idx
                            )));
                        }
                        let var = lit.unsigned_abs();
                        let (t, f) = validate_circuit_log_weight_pair(var_log_weights(var))?;
                        if lit > 0 {
                            t
                        } else {
                            f
                        }
                    }
                    XgcfNodeType::And => {
                        let c0 = self.child_offsets[idx] as usize;
                        let c1 = self.child_offsets[idx + 1] as usize;
                        if c0 == c1 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF eval error: AND node {} has no children",
                                idx
                            )));
                        }
                        let mut acc = 0.0;
                        for &child in &self.child_indices[c0..c1] {
                            acc += values[child as usize];
                        }
                        acc
                    }
                    XgcfNodeType::Or => {
                        let c0 = self.child_offsets[idx] as usize;
                        let c1 = self.child_offsets[idx + 1] as usize;
                        if c0 == c1 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF eval error: OR node {} has no children",
                                idx
                            )));
                        }
                        let mut branch_vals: Vec<f64> = Vec::with_capacity(c1 - c0);
                        for &child in &self.child_indices[c0..c1] {
                            branch_vals.push(values[child as usize]);
                        }
                        circuit_logsumexp(&branch_vals)?
                    }
                    XgcfNodeType::Decision => {
                        let var = self.decision_var[idx];
                        if var == 0 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF invariant violation: DECISION node {} has var=0",
                                idx
                            )));
                        }
                        let child_false = self.decision_child_false[idx] as usize;
                        let child_true = self.decision_child_true[idx] as usize;
                        let (t, f) = validate_circuit_log_weight_pair(var_log_weights(var))?;
                        circuit_logsumexp(&[f + values[child_false], t + values[child_true]])?
                    }
                };
                values[idx] = validate_circuit_value(v)?;
            }
        }

        Ok(values[self.roots[0] as usize])
    }

    /// Evaluates log weighted model count and literal log-weight gradients on the CPU.
    pub fn eval_log_wmc_and_grads(
        &self,
        var_log_weights: &[(f64, f64)],
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        if self.roots.len() != 1 {
            return Err(XlogError::Compilation(format!(
                "XGCF eval expects exactly 1 root, got {}",
                self.roots.len()
            )));
        }

        let n = self.node_type.len();
        if self.child_offsets.len() != n + 1 {
            return Err(XlogError::Compilation(format!(
                "XGCF invariant violation: child_offsets len {} != num_nodes+1 ({})",
                self.child_offsets.len(),
                n + 1
            )));
        }
        if self.lit.len() != n
            || self.decision_var.len() != n
            || self.decision_child_false.len() != n
            || self.decision_child_true.len() != n
        {
            return Err(XlogError::Compilation(
                "XGCF invariant violation: per-node arrays length mismatch".to_string(),
            ));
        }
        if self.level_offsets.is_empty() || *self.level_offsets.first().unwrap() != 0 {
            return Err(XlogError::Compilation(
                "XGCF invariant violation: level_offsets must start at 0".to_string(),
            ));
        }
        if *self.level_offsets.last().unwrap() != self.level_nodes.len() as u32 {
            return Err(XlogError::Compilation(
                "XGCF invariant violation: level_offsets last != level_nodes.len".to_string(),
            ));
        }

        let mut max_var: u32 = 0;
        for (&ty, &lit) in self.node_type.iter().zip(self.lit.iter()) {
            if ty == XgcfNodeType::Lit && lit != 0 {
                max_var = max_var.max(lit.unsigned_abs());
            }
        }
        for &var in &self.decision_var {
            max_var = max_var.max(var);
        }

        let weights_len = (max_var as usize) + 1;
        if var_log_weights.len() < weights_len {
            return Err(XlogError::Compilation(format!(
                "XGCF eval expects weight table len >= {}, got {}",
                weights_len,
                var_log_weights.len()
            )));
        }
        let mut values: Vec<f64> = vec![0.0; n];

        let num_levels = self.level_offsets.len() - 1;
        for level in 0..num_levels {
            let start = self.level_offsets[level] as usize;
            let end = self.level_offsets[level + 1] as usize;
            for &node_u32 in &self.level_nodes[start..end] {
                let idx = node_u32 as usize;
                let v = match self.node_type[idx] {
                    XgcfNodeType::Const0 => f64::NEG_INFINITY,
                    XgcfNodeType::Const1 => 0.0,
                    XgcfNodeType::Lit => {
                        let lit = self.lit[idx];
                        if lit == 0 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF invariant violation: LIT node {} has lit=0",
                                idx
                            )));
                        }
                        let var = lit.unsigned_abs();
                        let (t, f) =
                            validate_circuit_log_weight_pair(var_log_weights[var as usize])?;
                        if lit > 0 {
                            t
                        } else {
                            f
                        }
                    }
                    XgcfNodeType::And => {
                        let c0 = self.child_offsets[idx] as usize;
                        let c1 = self.child_offsets[idx + 1] as usize;
                        if c0 == c1 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF eval error: AND node {} has no children",
                                idx
                            )));
                        }
                        let mut acc = 0.0;
                        for &child in &self.child_indices[c0..c1] {
                            acc += values[child as usize];
                        }
                        acc
                    }
                    XgcfNodeType::Or => {
                        let c0 = self.child_offsets[idx] as usize;
                        let c1 = self.child_offsets[idx + 1] as usize;
                        if c0 == c1 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF eval error: OR node {} has no children",
                                idx
                            )));
                        }
                        let mut branch_vals: Vec<f64> = Vec::with_capacity(c1 - c0);
                        for &child in &self.child_indices[c0..c1] {
                            branch_vals.push(values[child as usize]);
                        }
                        circuit_logsumexp(&branch_vals)?
                    }
                    XgcfNodeType::Decision => {
                        let var = self.decision_var[idx];
                        if var == 0 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF invariant violation: DECISION node {} has var=0",
                                idx
                            )));
                        }
                        let child_false = self.decision_child_false[idx] as usize;
                        let child_true = self.decision_child_true[idx] as usize;
                        let (t, f) =
                            validate_circuit_log_weight_pair(var_log_weights[var as usize])?;
                        circuit_logsumexp(&[f + values[child_false], t + values[child_true]])?
                    }
                };
                values[idx] = validate_circuit_value(v)?;
            }
        }

        let root_idx = self.roots[0] as usize;
        let log_z = values[root_idx];

        let mut adj: Vec<f64> = vec![0.0; n];
        adj[root_idx] = 1.0;

        let mut grad_true: Vec<f64> = vec![0.0; weights_len];
        let mut grad_false: Vec<f64> = vec![0.0; weights_len];

        for level in (0..num_levels).rev() {
            let start = self.level_offsets[level] as usize;
            let end = self.level_offsets[level + 1] as usize;
            for &node_u32 in &self.level_nodes[start..end] {
                let idx = node_u32 as usize;
                let a = adj[idx];
                if a == 0.0 {
                    continue;
                }

                match self.node_type[idx] {
                    XgcfNodeType::Const0 | XgcfNodeType::Const1 => {}
                    XgcfNodeType::Lit => {
                        let lit = self.lit[idx];
                        if lit == 0 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF invariant violation: LIT node {} has lit=0",
                                idx
                            )));
                        }
                        let var = lit.unsigned_abs() as usize;
                        if lit > 0 {
                            grad_true[var] += a;
                        } else {
                            grad_false[var] += a;
                        }
                    }
                    XgcfNodeType::And => {
                        let c0 = self.child_offsets[idx] as usize;
                        let c1 = self.child_offsets[idx + 1] as usize;
                        for &child in &self.child_indices[c0..c1] {
                            adj[child as usize] += a;
                        }
                    }
                    XgcfNodeType::Or => {
                        let parent_v = values[idx];
                        if parent_v.is_infinite() && parent_v.is_sign_negative() {
                            continue;
                        }
                        let c0 = self.child_offsets[idx] as usize;
                        let c1 = self.child_offsets[idx + 1] as usize;
                        for &child in &self.child_indices[c0..c1] {
                            let child_idx = child as usize;
                            let child_v = values[child_idx];
                            if child_v.is_infinite() && child_v.is_sign_negative() {
                                continue;
                            }
                            let ratio = (child_v - parent_v).exp();
                            if ratio != 0.0 {
                                adj[child_idx] += a * ratio;
                            }
                        }
                    }
                    XgcfNodeType::Decision => {
                        let var = self.decision_var[idx] as usize;
                        if var == 0 {
                            return Err(XlogError::Compilation(format!(
                                "XGCF invariant violation: DECISION node {} has var=0",
                                idx
                            )));
                        }

                        let parent_v = values[idx];
                        if parent_v.is_infinite() && parent_v.is_sign_negative() {
                            continue;
                        }

                        let child_false = self.decision_child_false[idx] as usize;
                        let child_true = self.decision_child_true[idx] as usize;
                        let (t, f) = var_log_weights[var];

                        let vf = values[child_false];
                        let vt = values[child_true];

                        let mut p_false = 0.0;
                        let mut p_true = 0.0;
                        if !(vf.is_infinite() && vf.is_sign_negative()) {
                            p_false = (f + vf - parent_v).exp();
                        }
                        if !(vt.is_infinite() && vt.is_sign_negative()) {
                            p_true = (t + vt - parent_v).exp();
                        }

                        if p_false != 0.0 {
                            adj[child_false] += a * p_false;
                            grad_false[var] += a * p_false;
                        }
                        if p_true != 0.0 {
                            adj[child_true] += a * p_true;
                            grad_true[var] += a * p_true;
                        }
                    }
                }
            }
        }

        validate_circuit_gradient_values(&grad_true, &grad_false)?;
        Ok((log_z, grad_true, grad_false))
    }
}

fn max_var_in_circuit(circuit: &Xgcf) -> Result<u32> {
    let mut max_var: u32 = 0;
    for (&ty, &lit) in circuit.node_type.iter().zip(circuit.lit.iter()) {
        if ty == XgcfNodeType::Lit {
            if lit == 0 {
                return Err(XlogError::Compilation(
                    "XGCF invariant violation: LIT node has lit=0".to_string(),
                ));
            }
            max_var = max_var.max(lit.unsigned_abs());
        }
    }
    for (&ty, &var) in circuit.node_type.iter().zip(circuit.decision_var.iter()) {
        if ty == XgcfNodeType::Decision {
            if var == 0 {
                return Err(XlogError::Compilation(
                    "XGCF invariant violation: DECISION node has var=0".to_string(),
                ));
            }
            max_var = max_var.max(var);
        }
    }
    Ok(max_var)
}

fn merge_union_sorted(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        let va = a[i];
        let vb = b[j];
        if va == vb {
            out.push(va);
            i += 1;
            j += 1;
        } else if va < vb {
            out.push(va);
            i += 1;
        } else {
            out.push(vb);
            j += 1;
        }
    }
    if i < a.len() {
        out.extend_from_slice(&a[i..]);
    }
    if j < b.len() {
        out.extend_from_slice(&b[j..]);
    }
}

fn sorted_difference(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() {
        let va = a[i];
        while j < b.len() && b[j] < va {
            j += 1;
        }
        if j < b.len() && b[j] == va {
            i += 1;
            j += 1;
            continue;
        }
        out.push(va);
        i += 1;
    }
}

fn insert_sorted_unique(sorted: &mut Vec<u32>, var: u32) {
    match sorted.binary_search(&var) {
        Ok(_) => {}
        Err(pos) => sorted.insert(pos, var),
    }
}

fn compute_random_support(circuit: &Xgcf, is_random_var: &[bool]) -> Result<Vec<Vec<u32>>> {
    let n = circuit.node_type.len();
    let mut support: Vec<Vec<u32>> = vec![Vec::new(); n];

    let num_levels = circuit.level_offsets.len().saturating_sub(1);
    for level in 0..num_levels {
        let start = circuit.level_offsets[level] as usize;
        let end = circuit.level_offsets[level + 1] as usize;
        for &node_u32 in &circuit.level_nodes[start..end] {
            let idx = node_u32 as usize;
            match circuit.node_type[idx] {
                XgcfNodeType::Const0 | XgcfNodeType::Const1 => {}
                XgcfNodeType::Lit => {
                    let lit = circuit.lit[idx];
                    if lit == 0 {
                        return Err(XlogError::Compilation(format!(
                            "XGCF invariant violation: LIT node {} has lit=0",
                            idx
                        )));
                    }
                    let var = lit.unsigned_abs() as usize;
                    if var < is_random_var.len() && is_random_var[var] {
                        support[idx].push(var as u32);
                    }
                }
                XgcfNodeType::And | XgcfNodeType::Or => {
                    let c0 = circuit.child_offsets[idx] as usize;
                    let c1 = circuit.child_offsets[idx + 1] as usize;
                    let mut acc: Vec<u32> = Vec::new();
                    let mut tmp: Vec<u32> = Vec::new();
                    for &child in &circuit.child_indices[c0..c1] {
                        let child_idx = child as usize;
                        merge_union_sorted(&acc, &support[child_idx], &mut tmp);
                        std::mem::swap(&mut acc, &mut tmp);
                    }
                    support[idx] = acc;
                }
                XgcfNodeType::Decision => {
                    let var = circuit.decision_var[idx];
                    if var == 0 {
                        return Err(XlogError::Compilation(format!(
                            "XGCF invariant violation: DECISION node {} has var=0",
                            idx
                        )));
                    }
                    let child_false = circuit.decision_child_false[idx] as usize;
                    let child_true = circuit.decision_child_true[idx] as usize;

                    let mut acc: Vec<u32> = Vec::new();
                    merge_union_sorted(&support[child_false], &support[child_true], &mut acc);

                    let var_usize = var as usize;
                    if var_usize < is_random_var.len() && is_random_var[var_usize] {
                        insert_sorted_unique(&mut acc, var);
                    }
                    support[idx] = acc;
                }
            }
        }
    }

    Ok(support)
}

struct XgcfSmoother<'a> {
    input: &'a Xgcf,
    is_random_var: &'a [bool],
    support: Vec<Vec<u32>>,
}

impl<'a> XgcfSmoother<'a> {
    fn new(input: &'a Xgcf, is_random_var: &'a [bool]) -> Result<Self> {
        let n = input.node_type.len();
        if input.child_offsets.len() != n + 1 {
            return Err(XlogError::Compilation(format!(
                "XGCF invariant violation: child_offsets len {} != num_nodes+1 ({})",
                input.child_offsets.len(),
                n + 1
            )));
        }
        if input.lit.len() != n
            || input.decision_var.len() != n
            || input.decision_child_false.len() != n
            || input.decision_child_true.len() != n
        {
            return Err(XlogError::Compilation(
                "XGCF invariant violation: per-node arrays length mismatch".to_string(),
            ));
        }

        let max_var = max_var_in_circuit(input)?;
        if is_random_var.len() <= (max_var as usize) {
            return Err(XlogError::Compilation(format!(
                "XGCF smoothing expects is_random_var len >= {}, got {}",
                (max_var as usize) + 1,
                is_random_var.len()
            )));
        }

        let support = compute_random_support(input, is_random_var)?;
        Ok(Self {
            input,
            is_random_var,
            support,
        })
    }

    fn smooth(&self) -> Result<Xgcf> {
        XgcfSmoothBuilder::new(self.input).smooth(self.is_random_var, &self.support)
    }
}

struct XgcfSmoothBuilder<'a> {
    input: &'a Xgcf,
    node_type: Vec<XgcfNodeType>,
    child_offsets: Vec<u32>,
    child_indices: Vec<u32>,
    lit: Vec<i32>,
    decision_var: Vec<u32>,
    decision_child_false: Vec<u32>,
    decision_child_true: Vec<u32>,
    old_to_new: Vec<Option<u32>>,
    lit_cache: HashMap<i32, u32>,
    tautology_cache: HashMap<u32, u32>,
    const0: u32,
    const1: u32,
}

impl<'a> XgcfSmoothBuilder<'a> {
    fn new(input: &'a Xgcf) -> Self {
        let mut b = Self {
            input,
            node_type: Vec::new(),
            child_offsets: Vec::new(),
            child_indices: Vec::new(),
            lit: Vec::new(),
            decision_var: Vec::new(),
            decision_child_false: Vec::new(),
            decision_child_true: Vec::new(),
            old_to_new: Vec::new(),
            lit_cache: HashMap::new(),
            tautology_cache: HashMap::new(),
            const0: 0,
            const1: 0,
        };

        b.const0 = b.push_const(false);
        b.const1 = b.push_const(true);
        b
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

    fn tautology_decision(&mut self, var: u32) -> Result<u32> {
        if let Some(&idx) = self.tautology_cache.get(&var) {
            return Ok(idx);
        }
        let idx = self.push_decision(var, self.const1, self.const1)?;
        self.tautology_cache.insert(var, idx);
        Ok(idx)
    }

    fn smooth(mut self, is_random_var: &[bool], support: &[Vec<u32>]) -> Result<Xgcf> {
        let n = self.input.node_type.len();
        self.old_to_new = vec![None; n];

        let num_levels = self.input.level_offsets.len().saturating_sub(1);
        for level in 0..num_levels {
            let start = self.input.level_offsets[level] as usize;
            let end = self.input.level_offsets[level + 1] as usize;
            for &node_u32 in &self.input.level_nodes[start..end] {
                let idx = node_u32 as usize;

                let new_idx = match self.input.node_type[idx] {
                    XgcfNodeType::Const0 => self.const0,
                    XgcfNodeType::Const1 => self.const1,
                    XgcfNodeType::Lit => {
                        let lit = self.input.lit[idx];
                        self.get_lit_node(lit)?
                    }
                    XgcfNodeType::And => {
                        let c0 = self.input.child_offsets[idx] as usize;
                        let c1 = self.input.child_offsets[idx + 1] as usize;
                        let mut children: Vec<u32> = Vec::with_capacity(c1 - c0);
                        for &child in &self.input.child_indices[c0..c1] {
                            let child_idx = child as usize;
                            let mapped = self.old_to_new[child_idx].ok_or_else(|| {
                                XlogError::Compilation(format!(
                                    "XGCF smoothing error: missing mapped child {} for AND node {}",
                                    child_idx, idx
                                ))
                            })?;
                            children.push(mapped);
                        }
                        self.push_and(children)?
                    }
                    XgcfNodeType::Or => {
                        let parent_support = &support[idx];
                        let c0 = self.input.child_offsets[idx] as usize;
                        let c1 = self.input.child_offsets[idx + 1] as usize;
                        let mut wrapped_children: Vec<u32> = Vec::with_capacity(c1 - c0);
                        let mut missing: Vec<u32> = Vec::new();
                        for &child in &self.input.child_indices[c0..c1] {
                            let child_idx = child as usize;
                            let child_new = self.old_to_new[child_idx].ok_or_else(|| {
                                XlogError::Compilation(format!(
                                    "XGCF smoothing error: missing mapped child {} for OR node {}",
                                    child_idx, idx
                                ))
                            })?;

                            let child_support = &support[child_idx];
                            sorted_difference(parent_support, child_support, &mut missing);

                            if missing.is_empty() {
                                wrapped_children.push(child_new);
                            } else {
                                let mut and_children: Vec<u32> =
                                    Vec::with_capacity(1 + missing.len());
                                and_children.push(child_new);
                                for &var in &missing {
                                    let var_usize = var as usize;
                                    if var_usize < is_random_var.len() && is_random_var[var_usize] {
                                        and_children.push(self.tautology_decision(var)?);
                                    }
                                }
                                wrapped_children.push(self.push_and(and_children)?);
                            }
                        }
                        self.push_or(wrapped_children)?
                    }
                    XgcfNodeType::Decision => {
                        let var = self.input.decision_var[idx];
                        let child_false_old = self.input.decision_child_false[idx] as usize;
                        let child_true_old = self.input.decision_child_true[idx] as usize;

                        let child_false_new = self.old_to_new[child_false_old].ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "XGCF smoothing error: missing mapped decision false child {} for node {}",
                                child_false_old, idx
                            ))
                        })?;
                        let child_true_new = self.old_to_new[child_true_old].ok_or_else(|| {
                            XlogError::Compilation(format!(
                                "XGCF smoothing error: missing mapped decision true child {} for node {}",
                                child_true_old, idx
                            ))
                        })?;

                        let mut union_children: Vec<u32> = Vec::new();
                        merge_union_sorted(
                            &support[child_false_old],
                            &support[child_true_old],
                            &mut union_children,
                        );

                        if union_children.binary_search(&var).is_ok() {
                            return Err(XlogError::Compilation(format!(
                                "XGCF smoothing error: decision var {} appears in child support at node {}",
                                var, idx
                            )));
                        }

                        let mut missing: Vec<u32> = Vec::new();

                        sorted_difference(&union_children, &support[child_false_old], &mut missing);
                        let new_false = if missing.is_empty() {
                            child_false_new
                        } else {
                            let mut and_children: Vec<u32> = Vec::with_capacity(1 + missing.len());
                            and_children.push(child_false_new);
                            for &v in &missing {
                                let v_usize = v as usize;
                                if v_usize < is_random_var.len() && is_random_var[v_usize] {
                                    and_children.push(self.tautology_decision(v)?);
                                }
                            }
                            self.push_and(and_children)?
                        };

                        sorted_difference(&union_children, &support[child_true_old], &mut missing);
                        let new_true = if missing.is_empty() {
                            child_true_new
                        } else {
                            let mut and_children: Vec<u32> = Vec::with_capacity(1 + missing.len());
                            and_children.push(child_true_new);
                            for &v in &missing {
                                let v_usize = v as usize;
                                if v_usize < is_random_var.len() && is_random_var[v_usize] {
                                    and_children.push(self.tautology_decision(v)?);
                                }
                            }
                            self.push_and(and_children)?
                        };

                        self.push_decision(var, new_false, new_true)?
                    }
                };

                self.old_to_new[idx] = Some(new_idx);
            }
        }

        // Finalize offsets (sentinel).
        self.child_offsets.push(self.child_indices.len() as u32);

        let mut roots: Vec<u32> = Vec::with_capacity(self.input.roots.len());
        for &root in &self.input.roots {
            let idx = root as usize;
            let mapped = self.old_to_new[idx].ok_or_else(|| {
                XlogError::Compilation(format!(
                    "XGCF smoothing error: missing mapped root node {}",
                    idx
                ))
            })?;
            roots.push(mapped);
        }

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
}

/// Builds bottom-up levels for the reachable nodes of a validated XGCF circuit.
pub fn levelize_xgcf(
    node_type: &[XgcfNodeType],
    child_offsets: &[u32],
    child_indices: &[u32],
    decision_child_false: &[u32],
    decision_child_true: &[u32],
    roots: &[u32],
) -> Result<(Vec<u32>, Vec<u32>)> {
    let n = node_type.len();
    let mut levels: Vec<Option<u32>> = vec![None; n];
    let mut visiting: Vec<bool> = vec![false; n];

    #[expect(
        clippy::too_many_arguments,
        reason = "recursive levelization keeps the immutable circuit tables and its two mutable traversal-state tables explicit"
    )]
    fn level_of(
        idx: usize,
        node_type: &[XgcfNodeType],
        child_offsets: &[u32],
        child_indices: &[u32],
        decision_child_false: &[u32],
        decision_child_true: &[u32],
        levels: &mut [Option<u32>],
        visiting: &mut [bool],
    ) -> Result<u32> {
        if let Some(lvl) = levels[idx] {
            return Ok(lvl);
        }
        if visiting[idx] {
            return Err(XlogError::Compilation(format!(
                "XGCF levelize error: cycle detected at node {}",
                idx
            )));
        }
        visiting[idx] = true;

        let lvl = match node_type[idx] {
            XgcfNodeType::Const0 | XgcfNodeType::Const1 | XgcfNodeType::Lit => 0,
            XgcfNodeType::And | XgcfNodeType::Or => {
                let c0 = child_offsets[idx] as usize;
                let c1 = child_offsets[idx + 1] as usize;
                let mut max_child = 0u32;
                for &child in &child_indices[c0..c1] {
                    max_child = max_child.max(level_of(
                        child as usize,
                        node_type,
                        child_offsets,
                        child_indices,
                        decision_child_false,
                        decision_child_true,
                        levels,
                        visiting,
                    )?);
                }
                max_child + 1
            }
            XgcfNodeType::Decision => {
                let lf = level_of(
                    decision_child_false[idx] as usize,
                    node_type,
                    child_offsets,
                    child_indices,
                    decision_child_false,
                    decision_child_true,
                    levels,
                    visiting,
                )?;
                let lt = level_of(
                    decision_child_true[idx] as usize,
                    node_type,
                    child_offsets,
                    child_indices,
                    decision_child_false,
                    decision_child_true,
                    levels,
                    visiting,
                )?;
                lf.max(lt) + 1
            }
        };

        visiting[idx] = false;
        levels[idx] = Some(lvl);
        Ok(lvl)
    }

    for &root in roots {
        level_of(
            root as usize,
            node_type,
            child_offsets,
            child_indices,
            decision_child_false,
            decision_child_true,
            &mut levels,
            &mut visiting,
        )?;
    }

    let max_level = levels.iter().flatten().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); (max_level as usize) + 1];
    for (i, lvl) in levels.iter().enumerate().take(n) {
        let Some(lvl) = lvl else {
            continue;
        };
        buckets[*lvl as usize].push(i as u32);
    }

    let mut level_offsets: Vec<u32> = Vec::with_capacity(buckets.len() + 1);
    let mut level_nodes: Vec<u32> = Vec::new();
    level_offsets.push(0);
    for bucket in buckets {
        level_nodes.extend(bucket);
        level_offsets.push(level_nodes.len() as u32);
    }
    Ok((level_offsets, level_nodes))
}
