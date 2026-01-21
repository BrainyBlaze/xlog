//! Exact probabilistic inference via Decision-DNNF (D4) + weighted model counting.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use xlog_core::{MemoryBudget, Result, XlogError};

use crate::cnf::encode_cnf;
use crate::gpu::GpuXgcf;
use crate::kc::d4::D4Compiler;
use crate::kc::ddnnf::DecisionDnnf;
use crate::provenance::{extract_from_source, GroundAtom, Provenance};
use crate::xgcf::{Xgcf, XgcfNodeType};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

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
pub struct QueryGradients {
    pub atom: GroundAtom,
    pub log_prob: f64,
    pub prob: f64,
    pub grad_true: Vec<f64>,
    pub grad_false: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ExactResultWithGrads {
    pub log_z_e: f64,
    pub query_grads: Vec<QueryGradients>,
}

#[derive(Debug, Clone)]
struct QuerySpec {
    atom: GroundAtom,
    var: Option<u32>,
}

struct GpuExactState {
    provider: CudaKernelProvider,
    circuit: Mutex<GpuXgcf>,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuConfig {
    pub device_ordinal: usize,
    pub memory_bytes: u64,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            memory_bytes: 1024 * 1024 * 1024,
        }
    }
}

impl GpuExactState {
    fn new(circuit: &Xgcf, config: GpuConfig) -> Result<Self> {
        let device_count = cudarc::driver::CudaDevice::count().unwrap_or(0) as usize;
        if device_count == 0 {
            return Err(XlogError::Kernel("No CUDA device available".to_string()));
        }
        if config.device_ordinal >= device_count {
            return Err(XlogError::Kernel(format!(
                "CUDA device ordinal {} out of range (count={})",
                config.device_ordinal, device_count
            )));
        }
        if config.memory_bytes == 0 {
            return Err(XlogError::Kernel(
                "GPU memory budget must be non-zero".to_string(),
            ));
        }

        let device = Arc::new(CudaDevice::new(config.device_ordinal)?);
        let memory = Arc::new(GpuMemoryManager::new(
            device.clone(),
            MemoryBudget::with_limit(config.memory_bytes),
        ));
        let provider = CudaKernelProvider::new(device, memory)?;
        let gpu_xgcf = GpuXgcf::upload(&provider, circuit)?;
        Ok(Self {
            provider,
            circuit: Mutex::new(gpu_xgcf),
        })
    }
}

#[derive(Clone)]
pub struct ExactDdnnfProgram {
    circuit: Option<Xgcf>,
    evidence_log_weights: Vec<(f64, f64)>,
    free_vars: Vec<u32>,
    queries: Vec<QuerySpec>,
    gpu_config: GpuConfig,
    gpu: Arc<OnceLock<GpuExactState>>,
}

impl ExactDdnnfProgram {
    pub fn compile_source(source: &str) -> Result<Self> {
        let provenance = extract_from_source(source)?;
        Self::compile_provenance(provenance)
    }

    pub fn compile_source_with_gpu(source: &str, config: GpuConfig) -> Result<Self> {
        let mut program = Self::compile_source(source)?;
        program.gpu_config = config;
        program.gpu = Arc::new(OnceLock::new());
        Ok(program)
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

    pub fn num_vars(&self) -> usize {
        self.evidence_log_weights.len()
    }

    pub fn evaluate_gpu_with_grads(&self) -> Result<ExactResultWithGrads> {
        let Some(_circuit) = &self.circuit else {
            return Ok(ExactResultWithGrads {
                log_z_e: 0.0,
                query_grads: Vec::new(),
            });
        };

        let mut weights: Vec<(f64, f64)> = self.evidence_log_weights.clone();

        let (log_z_e, grad_true_e, grad_false_e) =
            self.eval_log_z_and_grads_gpu(&weights)?;

        if log_z_e.is_infinite() && log_z_e.is_sign_negative() {
            return Err(XlogError::Execution(
                "Exact inference error: evidence is inconsistent (P(E)=0)".to_string(),
            ));
        }

        let mut query_grads: Vec<QueryGradients> = Vec::with_capacity(self.queries.len());

        for query in &self.queries {
            let Some(var) = query.var else {
                query_grads.push(QueryGradients {
                    atom: query.atom.clone(),
                    log_prob: f64::NEG_INFINITY,
                    prob: 0.0,
                    grad_true: vec![0.0; weights.len()],
                    grad_false: vec![0.0; weights.len()],
                });
                continue;
            };

            let idx = var as usize;
            if idx >= weights.len() {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: query var {} out of bounds (len={})",
                    var,
                    weights.len()
                )));
            }

            let prev = weights[idx];
            weights[idx].1 = f64::NEG_INFINITY;

            let (log_z_eq, grad_true_eq, grad_false_eq) =
                self.eval_log_z_and_grads_gpu(&weights)?;

            weights[idx] = prev;

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

            if grad_true_eq.len() != grad_true_e.len() || grad_false_eq.len() != grad_false_e.len() {
                return Err(XlogError::Execution(
                    "Exact inference error: gradient length mismatch".to_string(),
                ));
            }

            let mut grad_true: Vec<f64> = grad_true_eq;
            let mut grad_false: Vec<f64> = grad_false_eq;
            for i in 0..grad_true.len() {
                grad_true[i] -= grad_true_e[i];
                grad_false[i] -= grad_false_e[i];
            }

            query_grads.push(QueryGradients {
                atom: query.atom.clone(),
                log_prob,
                prob,
                grad_true,
                grad_false,
            });
        }

        Ok(ExactResultWithGrads { log_z_e, query_grads })
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
                evidence_log_weights: vec![(0.0, 0.0)],
                free_vars: Vec::new(),
                queries,
                gpu_config: GpuConfig::default(),
                gpu: Arc::new(OnceLock::new()),
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

        let mut evidence_log_weights: Vec<(f64, f64)> = base_log_weights.clone();
        for (idx, &enc) in evidence_assign.iter().enumerate().skip(1) {
            if enc == 1 {
                evidence_log_weights[idx].1 = f64::NEG_INFINITY;
            } else if enc == 2 {
                evidence_log_weights[idx].0 = f64::NEG_INFINITY;
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

        let mut is_random_var: Vec<bool> = vec![false; base_log_weights.len()];
        for (idx, &(t, f)) in base_log_weights.iter().enumerate().skip(1) {
            if (t, f) != (0.0, 0.0) {
                is_random_var[idx] = true;
            }
        }

        let circuit = Xgcf::from_ddnnf(&ddnnf)?.smooth_random_vars(&is_random_var)?;

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
            evidence_log_weights,
            free_vars,
            queries,
            gpu_config: GpuConfig::default(),
            gpu: Arc::new(OnceLock::new()),
        })
    }

    fn eval_log_z(&self, query_true: Option<u32>) -> Result<f64> {
        let Some(circuit) = &self.circuit else {
            return Ok(0.0);
        };

        let weights = &self.evidence_log_weights;

        let base_log_z = circuit.eval_log_wmc(|var| {
            let idx = var as usize;
            let (t, mut f) = weights[idx];
            if let Some(q) = query_true {
                if var == q {
                    f = f64::NEG_INFINITY;
                }
            }
            (t, f)
        })?;

        let mut log_z = base_log_z;

        for &var in &self.free_vars {
            let idx = var as usize;
            let (t, mut f) = weights[idx];
            if let Some(q) = query_true {
                if var == q {
                    f = f64::NEG_INFINITY;
                }
            }
            log_z += logsumexp2(t, f);
        }

        Ok(log_z)
    }

    fn gpu_state(&self) -> Result<&GpuExactState> {
        let Some(circuit) = &self.circuit else {
            return Err(XlogError::Execution(
                "Exact inference GPU error: program has no compiled circuit".to_string(),
            ));
        };

        if let Some(state) = self.gpu.get() {
            return Ok(state);
        }

        let state = GpuExactState::new(circuit, self.gpu_config)?;
        let _ = self.gpu.set(state);
        Ok(self.gpu.get().expect("OnceLock set failed"))
    }

    fn eval_log_z_and_grads_gpu(
        &self,
        weights: &[(f64, f64)],
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let Some(_circuit) = &self.circuit else {
            return Ok((0.0, vec![0.0], vec![0.0]));
        };
        let state = self.gpu_state()?;

        let (base_log_z, grad_true_base, grad_false_base) = {
            let mut gpu_xgcf = state
                .circuit
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            gpu_xgcf.eval_log_wmc_and_grads(&state.provider, weights)?
        };

        if grad_true_base.len() > weights.len() || grad_false_base.len() > weights.len() {
            return Err(XlogError::Execution(
                "Exact inference error: circuit gradient exceeds weight table".to_string(),
            ));
        }

        let mut log_z = base_log_z;

        let mut grad_true: Vec<f64> = vec![0.0; weights.len()];
        let mut grad_false: Vec<f64> = vec![0.0; weights.len()];
        grad_true[..grad_true_base.len()].copy_from_slice(&grad_true_base);
        grad_false[..grad_false_base.len()].copy_from_slice(&grad_false_base);

        for &var in &self.free_vars {
            let idx = var as usize;
            if idx >= weights.len() {
                return Err(XlogError::Execution(format!(
                    "Exact inference error: free var {} out of bounds (len={})",
                    var,
                    weights.len()
                )));
            }
            let (t, f) = weights[idx];
            let (ls, pt, pf) = logsumexp2_with_grads(t, f);
            log_z += ls;
            grad_true[idx] += pt;
            grad_false[idx] += pf;
        }

        Ok((log_z, grad_true, grad_false))
    }
}

fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = if a > b { a } else { b };
    if m.is_infinite() && m.is_sign_negative() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn logsumexp2_with_grads(a: f64, b: f64) -> (f64, f64, f64) {
    let m = if a > b { a } else { b };
    if m.is_infinite() && m.is_sign_negative() {
        return (m, 0.0, 0.0);
    }
    if a.is_infinite() && a.is_sign_negative() {
        return (b, 0.0, 1.0);
    }
    if b.is_infinite() && b.is_sign_negative() {
        return (a, 1.0, 0.0);
    }

    let ea = (a - m).exp();
    let eb = (b - m).exp();
    let sum = ea + eb;
    let ls = m + sum.ln();
    let pa = ea / sum;
    let pb = eb / sum;
    (ls, pa, pb)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_negation_probability() {
        // 0.3::rain(). dry() :- not rain().
        // P(dry) = P(not rain) = 1 - 0.3 = 0.7
        let source = r#"
0.3::rain().
dry() :- not rain().
query(dry()).
"#;

        let program = ExactDdnnfProgram::compile_source(source).unwrap();
        let result = program.evaluate().unwrap();

        assert_eq!(result.query_probs.len(), 1);
        let dry_prob = result.query_probs[0].prob;
        assert!(
            (dry_prob - 0.7).abs() < 1e-6,
            "P(dry) should be 0.7, got {}",
            dry_prob
        );
    }

    #[test]
    fn test_exact_multi_layer_negation() {
        // 0.4::c(). b() :- not c(). a() :- not b().
        // P(b) = P(not c) = 0.6
        // P(a) = P(not b) = 0.4
        let source = r#"
0.4::c().
b() :- not c().
a() :- not b().
query(a()).
"#;

        let program = ExactDdnnfProgram::compile_source(source).unwrap();
        let result = program.evaluate().unwrap();

        assert_eq!(result.query_probs.len(), 1);
        let a_prob = result.query_probs[0].prob;
        assert!(
            (a_prob - 0.4).abs() < 1e-6,
            "P(a) should be 0.4, got {}",
            a_prob
        );
    }
}
