//! Exact probabilistic inference via Decision-DNNF (D4) + weighted model counting.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use xlog_core::{Result, XlogError};

use crate::cnf::encode_cnf;
use crate::kc::d4::D4Compiler;
use crate::kc::ddnnf::DecisionDnnf;
use crate::provenance::{extract_from_source, GroundAtom, Provenance};
use crate::xgcf::{Xgcf, XgcfNodeType};

#[derive(Debug, Clone)]
pub struct QueryProbability {
    pub atom: GroundAtom,
    pub log_prob: f64,
    pub prob: f64,
}

#[derive(Debug, Clone)]
pub struct ExactResult {
    pub log_z_e: f64,
    pub query_probs: Vec<QueryProbability>,
}

#[derive(Debug, Clone)]
struct QuerySpec {
    atom: GroundAtom,
    var: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ExactDdnnfProgram {
    circuit: Option<Xgcf>,
    base_log_weights: Vec<(f64, f64)>,
    evidence_assign: Vec<u8>,
    evidence_vars: Vec<(u32, u8)>,
    free_vars: Vec<u32>,
    queries: Vec<QuerySpec>,
}

impl ExactDdnnfProgram {
    pub fn compile_source(source: &str) -> Result<Self> {
        let provenance = extract_from_source(source)?;
        Self::compile_provenance(provenance)
    }

    pub fn evaluate(&self) -> Result<ExactResult> {
        let Some(_circuit) = &self.circuit else {
            let mut query_probs: Vec<QueryProbability> = Vec::with_capacity(self.queries.len());
            for query in &self.queries {
                query_probs.push(QueryProbability {
                    atom: query.atom.clone(),
                    log_prob: f64::NEG_INFINITY,
                    prob: 0.0,
                });
            }
            return Ok(ExactResult {
                log_z_e: 0.0,
                query_probs,
            });
        };

        let log_z_e = self.eval_log_z(None)?;
        if log_z_e.is_infinite() && log_z_e.is_sign_negative() {
            return Err(XlogError::Execution(
                "Exact inference error: evidence is inconsistent (P(E)=0)".to_string(),
            ));
        }

        let mut query_probs: Vec<QueryProbability> = Vec::with_capacity(self.queries.len());
        for query in &self.queries {
            let (log_prob, prob) = match query.var {
                None => (f64::NEG_INFINITY, 0.0),
                Some(var) => {
                    let log_z_eq = self.eval_log_z(Some(var))?;
                    let log_prob = log_z_eq - log_z_e;
                    let mut prob = if log_prob.is_infinite() && log_prob.is_sign_negative() {
                        0.0
                    } else {
                        log_prob.exp()
                    };
                    if prob.is_nan() {
                        return Err(XlogError::Execution(
                            "Exact inference error: NaN probability encountered".to_string(),
                        ));
                    }
                    if prob < 0.0 {
                        prob = 0.0;
                    } else if prob > 1.0 {
                        prob = 1.0;
                    }
                    (log_prob, prob)
                }
            };

            query_probs.push(QueryProbability {
                atom: query.atom.clone(),
                log_prob,
                prob,
            });
        }

        Ok(ExactResult { log_z_e, query_probs })
    }

    fn compile_provenance(provenance: Provenance) -> Result<Self> {
        let d4 = D4Compiler::detect()?;

        let mut roots_set: HashSet<crate::pir::PirNodeId> = HashSet::new();

        let mut evidence_formulas: Vec<(crate::pir::PirNodeId, bool, GroundAtom)> = Vec::new();
        for (atom, value) in &provenance.evidence {
            let formula = provenance.query_formula(&atom.predicate, &atom.args);
            match formula {
                Some(id) => {
                    roots_set.insert(id);
                    evidence_formulas.push((id, *value, atom.clone()));
                }
                None => {
                    if *value {
                        return Err(XlogError::Execution(format!(
                            "Exact inference error: evidence atom is never derivable: {}",
                            display_atom(atom)
                        )));
                    }
                }
            }
        }

        let mut queries: Vec<QuerySpec> = Vec::new();
        for atom in &provenance.queries {
            let formula = provenance.query_formula(&atom.predicate, &atom.args);
            if let Some(id) = formula {
                roots_set.insert(id);
            }
            queries.push(QuerySpec {
                atom: atom.clone(),
                var: None,
            });
        }

        let mut roots: Vec<crate::pir::PirNodeId> = roots_set.into_iter().collect();
        roots.sort();

        if roots.is_empty() {
            return Ok(Self {
                circuit: None,
                base_log_weights: vec![(0.0, 0.0)],
                evidence_assign: vec![0],
                evidence_vars: Vec::new(),
                free_vars: Vec::new(),
                queries,
            });
        }

        let encoding = encode_cnf(&provenance.pir, &roots)?;

        let mut base_log_weights: Vec<(f64, f64)> =
            vec![(0.0, 0.0); (encoding.cnf.num_vars() as usize) + 1];
        for (leaf, var) in &encoding.leaf_var {
            let p = *provenance.leaf_probs.get(leaf).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Exact inference error: missing probability for leaf {:?}",
                    leaf
                ))
            })?;
            let t = ln_prob(p);
            let f = ln_prob(1.0 - p);
            base_log_weights[*var as usize] = (t, f);
        }
        for (choice, var) in &encoding.choice_var {
            let (pt, pf) = *provenance.choice_probs.get(choice).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Exact inference error: missing probability for choice {:?}",
                    choice
                ))
            })?;
            base_log_weights[*var as usize] = (ln_prob(pt), ln_prob(pf));
        }

        let mut evidence_assign: Vec<u8> = vec![0u8; base_log_weights.len()];
        for (formula, value, atom) in evidence_formulas {
            let var = *encoding.node_var.get(&formula).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Exact inference error: missing CNF variable for evidence formula {:?}",
                    formula
                ))
            })?;

            let idx = var as usize;
            if idx >= evidence_assign.len() {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: evidence var {} out of bounds (len={})",
                    var,
                    evidence_assign.len()
                )));
            }
            let enc = if value { 1u8 } else { 2u8 };
            match evidence_assign[idx] {
                0 => evidence_assign[idx] = enc,
                prev if prev == enc => {}
                _ => {
                    return Err(XlogError::Execution(format!(
                        "Exact inference error: conflicting evidence for {}",
                        display_atom(&atom)
                    )));
                }
            }
        }

        let mut evidence_vars: Vec<(u32, u8)> = Vec::new();
        for (idx, &enc) in evidence_assign.iter().enumerate().skip(1) {
            if enc != 0 {
                evidence_vars.push((idx as u32, enc));
            }
        }

        for query in &mut queries {
            let formula = provenance.query_formula(&query.atom.predicate, &query.atom.args);
            let Some(formula) = formula else {
                query.var = None;
                continue;
            };
            let var = *encoding.node_var.get(&formula).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Exact inference error: missing CNF variable for query formula {:?}",
                    formula
                ))
            })?;
            query.var = Some(var);
        }

        let dir = TempDirGuard::new("xlog-prob-exact-ddnnf")?;
        let cnf_path = dir.path().join("in.cnf");
        let out_path = dir.path().join("out.nnf");

        fs::write(&cnf_path, encoding.cnf.to_dimacs()).map_err(|e| {
            XlogError::Execution(format!(
                "Exact inference error: failed to write CNF file {}: {}",
                cnf_path.display(),
                e
            ))
        })?;

        d4.compile_ddnnf(&cnf_path, &out_path)?;

        let nnf = fs::read_to_string(&out_path).map_err(|e| {
            XlogError::Execution(format!(
                "Exact inference error: failed to read DDNNF output {}: {}",
                out_path.display(),
                e
            ))
        })?;
        let ddnnf = DecisionDnnf::parse_str(&nnf)?;
        if ddnnf.max_var() > encoding.cnf.num_vars() {
            return Err(XlogError::Compilation(format!(
                "Exact inference error: DDNNF references var {} but CNF has only {} vars",
                ddnnf.max_var(),
                encoding.cnf.num_vars()
            )));
        }

        if (ddnnf.max_var() as usize) >= base_log_weights.len() {
            return Err(XlogError::Compilation(format!(
                "Exact inference error: var {} out of bounds for weight table (len={})",
                ddnnf.max_var(),
                base_log_weights.len()
            )));
        }
        if evidence_assign.len() != base_log_weights.len() {
            return Err(XlogError::Compilation(format!(
                "Exact inference error: evidence table len {} != weight table len {}",
                evidence_assign.len(),
                base_log_weights.len()
            )));
        }

        let circuit = Xgcf::from_ddnnf(&ddnnf)?;

        let num_vars = encoding.cnf.num_vars() as usize;
        let mut vars_in_clauses: Vec<bool> = vec![false; num_vars + 1];
        for clause in encoding.cnf.clauses() {
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                if var == 0 || var > num_vars {
                    return Err(XlogError::Compilation(format!(
                        "Exact inference error: CNF clause references invalid var {} (num_vars={})",
                        var, num_vars
                    )));
                }
                vars_in_clauses[var] = true;
            }
        }

        let mut vars_in_circuit: Vec<bool> = vec![false; num_vars + 1];
        for (idx, node_type) in circuit.node_type.iter().enumerate() {
            match node_type {
                XgcfNodeType::Lit => {
                    let lit = circuit.lit[idx];
                    let var = lit.unsigned_abs() as usize;
                    if var == 0 || var > num_vars {
                        return Err(XlogError::Compilation(format!(
                            "Exact inference error: circuit literal references invalid var {} (num_vars={})",
                            var, num_vars
                        )));
                    }
                    vars_in_circuit[var] = true;
                }
                XgcfNodeType::Decision => {
                    let var = circuit.decision_var[idx] as usize;
                    if var == 0 || var > num_vars {
                        return Err(XlogError::Compilation(format!(
                            "Exact inference error: circuit decision references invalid var {} (num_vars={})",
                            var, num_vars
                        )));
                    }
                    vars_in_circuit[var] = true;
                }
                _ => {}
            }
        }

        let mut free_vars: Vec<u32> = Vec::new();
        for var in 1..=num_vars {
            if vars_in_circuit[var] {
                continue;
            }
            if vars_in_clauses[var] {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: DDNNF/circuit omitted var {} which appears in CNF clauses",
                    var
                )));
            }
            free_vars.push(var as u32);
        }

        Ok(Self {
            circuit: Some(circuit),
            base_log_weights,
            evidence_assign,
            evidence_vars,
            free_vars,
            queries,
        })
    }

    fn eval_log_z(&self, query_true: Option<u32>) -> Result<f64> {
        let Some(circuit) = &self.circuit else {
            return Ok(0.0);
        };

        let weights = &self.base_log_weights;
        let evidence = &self.evidence_assign;

        if let Some(q) = query_true {
            let idx = q as usize;
            if idx < evidence.len() && evidence[idx] == 2 {
                return Ok(f64::NEG_INFINITY);
            }
        }

        let mut forced_random: std::collections::HashMap<u32, bool> = std::collections::HashMap::new();
        for &(var, enc) in &self.evidence_vars {
            let idx = var as usize;
            if idx >= weights.len() {
                continue;
            }
            if weights[idx] == (0.0, 0.0) {
                continue;
            }
            if self.free_vars.binary_search(&var).is_ok() {
                continue;
            }
            let is_true = enc == 1;
            match forced_random.insert(var, is_true) {
                None => {}
                Some(prev) if prev == is_true => {}
                Some(_) => return Ok(f64::NEG_INFINITY),
            }
        }
        if let Some(q) = query_true {
            let idx = q as usize;
            if idx < weights.len() && weights[idx] != (0.0, 0.0) && self.free_vars.binary_search(&q).is_err() {
                match forced_random.insert(q, true) {
                    None => {}
                    Some(prev) if prev => {}
                    Some(_) => return Ok(f64::NEG_INFINITY),
                }
            }
        }

        let base_log_z = if forced_random.is_empty() {
            circuit.eval_log_wmc(|var| {
                let idx = var as usize;
                let (mut t, mut f) = weights[idx];
                match evidence[idx] {
                    1 => f = f64::NEG_INFINITY,
                    2 => t = f64::NEG_INFINITY,
                    _ => {}
                }
                if let Some(q) = query_true {
                    if var == q {
                        f = f64::NEG_INFINITY;
                    }
                }
                (t, f)
            })?
        } else {
            eval_log_wmc_forced(
                circuit,
                weights,
                evidence,
                query_true,
                &forced_random,
            )?
        };

        let mut log_z = base_log_z;

        for &var in &self.free_vars {
            let idx = var as usize;
            let (mut t, mut f) = weights[idx];
            match evidence[idx] {
                1 => f = f64::NEG_INFINITY,
                2 => t = f64::NEG_INFINITY,
                _ => {}
            }
            if let Some(q) = query_true {
                if var == q {
                    f = f64::NEG_INFINITY;
                }
            }
            log_z += logsumexp2(t, f);
        }

        Ok(log_z)
    }
}

fn eval_log_wmc_forced(
    circuit: &Xgcf,
    weights: &[(f64, f64)],
    evidence: &[u8],
    query_true: Option<u32>,
    forced_random: &std::collections::HashMap<u32, bool>,
) -> Result<f64> {
    let n = circuit.node_type.len();
    if circuit.roots.len() != 1 {
        return Err(XlogError::Compilation(format!(
            "Exact inference error: expected exactly 1 circuit root, got {}",
            circuit.roots.len()
        )));
    }

    let mut forced_vars: Vec<(u32, bool)> = forced_random.iter().map(|(&v, &b)| (v, b)).collect();
    forced_vars.sort_by_key(|(v, _)| *v);

    let k = forced_vars.len();
    let words = (k + 63) / 64;
    let mut masks: Vec<u64> = vec![0; n.saturating_mul(words)];
    let mut values: Vec<f64> = vec![0.0; n];

    let mut var_to_bit: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::with_capacity(k.saturating_mul(2));
    let mut forced_log_weight: Vec<f64> = vec![0.0; k];
    for (bit, (var, is_true)) in forced_vars.iter().copied().enumerate() {
        var_to_bit.insert(var, bit);
        let idx = var as usize;
        let (mut t, mut f) = weights[idx];
        match evidence[idx] {
            1 => f = f64::NEG_INFINITY,
            2 => t = f64::NEG_INFINITY,
            _ => {}
        }
        if let Some(q) = query_true {
            if var == q {
                f = f64::NEG_INFINITY;
            }
        }
        forced_log_weight[bit] = if is_true { t } else { f };
    }

    let var_weights = |var: u32| {
        let idx = var as usize;
        let (mut t, mut f) = weights[idx];
        match evidence[idx] {
            1 => f = f64::NEG_INFINITY,
            2 => t = f64::NEG_INFINITY,
            _ => {}
        }
        if let Some(q) = query_true {
            if var == q {
                f = f64::NEG_INFINITY;
            }
        }
        (t, f)
    };

    fn sum_missing_weights(
        masks: &[u64],
        words: usize,
        node_base: usize,
        child_base: usize,
        forced_log_weight: &[f64],
    ) -> f64 {
        let mut acc = 0.0;
        for word_idx in 0..words {
            let m_node = masks[node_base + word_idx];
            let m_child = masks[child_base + word_idx];
            let mut missing = m_node & !m_child;
            while missing != 0 {
                let bit = missing.trailing_zeros() as usize;
                let idx = word_idx * 64 + bit;
                acc += forced_log_weight[idx];
                missing &= missing - 1;
            }
        }
        acc
    }

    for level in 0..(circuit.level_offsets.len().saturating_sub(1)) {
        let start = circuit.level_offsets[level] as usize;
        let end = circuit.level_offsets[level + 1] as usize;
        for &node_u32 in &circuit.level_nodes[start..end] {
            let idx = node_u32 as usize;
            let base = idx.saturating_mul(words);
            for w in 0..words {
                masks[base + w] = 0;
            }

            let v = match circuit.node_type[idx] {
                XgcfNodeType::Const0 => f64::NEG_INFINITY,
                XgcfNodeType::Const1 => 0.0,
                XgcfNodeType::Lit => {
                    let lit = circuit.lit[idx];
                    if lit == 0 {
                        return Err(XlogError::Compilation(format!(
                            "Exact inference error: circuit LIT node {} has lit=0",
                            idx
                        )));
                    }
                    let var = lit.unsigned_abs();
                    if let Some(&bit) = var_to_bit.get(&var) {
                        masks[base + (bit / 64)] |= 1u64 << (bit % 64);
                    }
                    let (t, f) = var_weights(var);
                    if lit > 0 { t } else { f }
                }
                XgcfNodeType::And => {
                    let c0 = circuit.child_offsets[idx] as usize;
                    let c1 = circuit.child_offsets[idx + 1] as usize;
                    if c0 == c1 {
                        return Err(XlogError::Compilation(format!(
                            "Exact inference error: circuit AND node {} has no children",
                            idx
                        )));
                    }
                    let mut acc = 0.0;
                    for &child in &circuit.child_indices[c0..c1] {
                        let child_idx = child as usize;
                        acc += values[child_idx];
                        let child_base = child_idx.saturating_mul(words);
                        for w in 0..words {
                            masks[base + w] |= masks[child_base + w];
                        }
                    }
                    acc
                }
                XgcfNodeType::Or => {
                    let c0 = circuit.child_offsets[idx] as usize;
                    let c1 = circuit.child_offsets[idx + 1] as usize;
                    if c0 == c1 {
                        return Err(XlogError::Compilation(format!(
                            "Exact inference error: circuit OR node {} has no children",
                            idx
                        )));
                    }
                    for &child in &circuit.child_indices[c0..c1] {
                        let child_idx = child as usize;
                        let child_base = child_idx.saturating_mul(words);
                        for w in 0..words {
                            masks[base + w] |= masks[child_base + w];
                        }
                    }

                    let mut max = f64::NEG_INFINITY;
                    for &child in &circuit.child_indices[c0..c1] {
                        let child_idx = child as usize;
                        let child_base = child_idx.saturating_mul(words);
                        let branch = values[child_idx]
                            + sum_missing_weights(&masks, words, base, child_base, &forced_log_weight);
                        if branch > max {
                            max = branch;
                        }
                    }
                    if max.is_infinite() && max.is_sign_negative() {
                        max
                    } else {
                        let mut sum = 0.0;
                        for &child in &circuit.child_indices[c0..c1] {
                            let child_idx = child as usize;
                            let child_base = child_idx.saturating_mul(words);
                            let branch = values[child_idx]
                                + sum_missing_weights(&masks, words, base, child_base, &forced_log_weight);
                            sum += (branch - max).exp();
                        }
                        max + sum.ln()
                    }
                }
                XgcfNodeType::Decision => {
                    let var = circuit.decision_var[idx];
                    if var == 0 {
                        return Err(XlogError::Compilation(format!(
                            "Exact inference error: circuit DECISION node {} has var=0",
                            idx
                        )));
                    }
                    let child_false = circuit.decision_child_false[idx] as usize;
                    let child_true = circuit.decision_child_true[idx] as usize;
                    let base_false = child_false.saturating_mul(words);
                    let base_true = child_true.saturating_mul(words);
                    for w in 0..words {
                        masks[base + w] = masks[base_false + w] | masks[base_true + w];
                    }

                    let missing_false =
                        sum_missing_weights(&masks, words, base, base_false, &forced_log_weight);
                    let missing_true =
                        sum_missing_weights(&masks, words, base, base_true, &forced_log_weight);

                    if let Some(&bit) = var_to_bit.get(&var) {
                        masks[base + (bit / 64)] |= 1u64 << (bit % 64);
                    }

                    let (t, f) = var_weights(var);
                    let v_false = f + values[child_false] + missing_false;
                    let v_true = t + values[child_true] + missing_true;
                    logsumexp2(v_false, v_true)
                }
            };

            values[idx] = v;
        }
    }

    Ok(values[circuit.roots[0] as usize])
}

fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = if a > b { a } else { b };
    if m.is_infinite() && m.is_sign_negative() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn ln_prob(p: f64) -> f64 {
    if p == 0.0 {
        f64::NEG_INFINITY
    } else {
        p.ln()
    }
}

fn make_temp_dir(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("{}-{}-{}", prefix, pid, nanos));
    fs::create_dir_all(&dir).expect("failed to create temp dir");
    dir
}

fn display_atom(atom: &GroundAtom) -> String {
    if atom.args.is_empty() {
        format!("{}()", atom.predicate)
    } else {
        format!("{}({} args)", atom.predicate, atom.args.len())
    }
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Result<Self> {
        let path = make_temp_dir(prefix);
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
