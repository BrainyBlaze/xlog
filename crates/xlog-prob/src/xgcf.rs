//! XLOG GPU Circuit Format (XGCF) - host-side representation and CPU evaluator.

use std::collections::HashMap;

use xlog_core::{Result, XlogError};

use crate::kc::ddnnf::{DdnnfEdge, DdnnfNodeKind, DecisionDnnf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum XgcfNodeType {
    Const0 = 0,
    Const1 = 1,
    Lit = 2,
    And = 3,
    Or = 4,
    Decision = 5,
}

#[derive(Debug, Clone)]
pub struct Xgcf {
    pub node_type: Vec<XgcfNodeType>,
    pub child_offsets: Vec<u32>,
    pub child_indices: Vec<u32>,
    pub lit: Vec<i32>,
    pub decision_var: Vec<u32>,
    pub decision_child_false: Vec<u32>,
    pub decision_child_true: Vec<u32>,
    pub roots: Vec<u32>,
    pub level_offsets: Vec<u32>,
    pub level_nodes: Vec<u32>,
}

impl Xgcf {
    pub fn from_ddnnf(ddnnf: &DecisionDnnf) -> Result<Self> {
        XgcfBuilder::new(ddnnf).build()
    }

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
                        let (t, f) = var_log_weights(var);
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
                        logsumexp(&branch_vals)
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
                        let (t, f) = var_log_weights(var);
                        logsumexp(&[f + values[child_false], t + values[child_true]])
                    }
                };
                values[idx] = v;
            }
        }

        Ok(values[self.roots[0] as usize])
    }

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
                        let (t, f) = var_log_weights[var as usize];
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
                        logsumexp(&branch_vals)
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
                        let (t, f) = var_log_weights[var as usize];
                        logsumexp(&[f + values[child_false], t + values[child_true]])
                    }
                };
                values[idx] = v;
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

        let (level_offsets, level_nodes) = XgcfBuilder::levelize(
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
        let (level_offsets, level_nodes) = Self::levelize(
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

    fn levelize(
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

        #[allow(clippy::too_many_arguments)]
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
        let xgcf = Xgcf::from_ddnnf(&ddnnf).unwrap();

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
        let xgcf = Xgcf::from_ddnnf(&ddnnf).unwrap();

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
}
