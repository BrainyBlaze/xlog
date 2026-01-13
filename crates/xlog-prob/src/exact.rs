//! Exact probabilistic inference via Decision-DNNF (D4) + weighted model counting.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use xlog_core::{Result, XlogError};

use crate::cnf::encode_cnf;
use crate::kc::d4::D4Compiler;
use crate::kc::ddnnf::DecisionDnnf;
use crate::provenance::{extract_from_source, GroundAtom, Provenance};
use crate::xgcf::Xgcf;

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
    d4: D4Compiler,
    cnf: Option<crate::cnf::CnfFormula>,
    base_log_weights: Vec<(f64, f64)>,
    evidence_units: Vec<i32>,
    queries: Vec<QuerySpec>,
}

impl ExactDdnnfProgram {
    pub fn compile_source(source: &str) -> Result<Self> {
        let provenance = extract_from_source(source)?;
        Self::compile_provenance(provenance)
    }

    pub fn evaluate(&self) -> Result<ExactResult> {
        let Some(cnf) = &self.cnf else {
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

        let dir = TempDirGuard::new("xlog-prob-exact")?;

        let log_z_e = self.eval_log_z(dir.path(), "e", cnf, &self.evidence_units)?;
        if log_z_e.is_infinite() && log_z_e.is_sign_negative() {
            return Err(XlogError::Execution(
                "Exact inference error: evidence is inconsistent (P(E)=0)".to_string(),
            ));
        }

        let mut query_probs: Vec<QueryProbability> = Vec::with_capacity(self.queries.len());
        for (idx, query) in self.queries.iter().enumerate() {
            let (log_prob, prob) = match query.var {
                None => (f64::NEG_INFINITY, 0.0),
                Some(var) => {
                    let mut units: Vec<i32> = Vec::with_capacity(self.evidence_units.len() + 1);
                    units.extend_from_slice(&self.evidence_units);
                    units.push(var as i32);

                    let log_z_eq =
                        self.eval_log_z(dir.path(), &format!("eq{}", idx), cnf, &units)?;
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
                d4,
                cnf: None,
                base_log_weights: vec![(0.0, 0.0)],
                evidence_units: Vec::new(),
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

        let mut evidence_units: Vec<i32> = Vec::new();
        let mut evidence_seen: std::collections::HashMap<u32, bool> = std::collections::HashMap::new();
        for (formula, value, atom) in evidence_formulas {
            let var = *encoding.node_var.get(&formula).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "Exact inference error: missing CNF variable for evidence formula {:?}",
                    formula
                ))
            })?;
            if let Some(prev) = evidence_seen.insert(var, value) {
                if prev != value {
                    return Err(XlogError::Execution(format!(
                        "Exact inference error: conflicting evidence for {}",
                        display_atom(&atom)
                    )));
                }
            }
            evidence_units.push(if value { var as i32 } else { -(var as i32) });
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

        Ok(Self {
            d4,
            cnf: Some(encoding.cnf),
            base_log_weights,
            evidence_units,
            queries,
        })
    }
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

impl ExactDdnnfProgram {
    fn eval_log_z(&self, dir: &Path, tag: &str, cnf: &crate::cnf::CnfFormula, units: &[i32]) -> Result<f64> {
        let cnf_path = dir.join(format!("{}.cnf", tag));
        let out_path = dir.join(format!("{}.nnf", tag));

        let dimacs = dimacs_with_units(cnf, units);
        fs::write(&cnf_path, dimacs).map_err(|e| {
            XlogError::Execution(format!(
                "Exact inference error: failed to write CNF file {}: {}",
                cnf_path.display(),
                e
            ))
        })?;

        self.d4.compile_ddnnf(&cnf_path, &out_path)?;

        let nnf = fs::read_to_string(&out_path).map_err(|e| {
            XlogError::Execution(format!(
                "Exact inference error: failed to read DDNNF output {}: {}",
                out_path.display(),
                e
            ))
        })?;
        let ddnnf = DecisionDnnf::parse_str(&nnf)?;
        if ddnnf.max_var() > cnf.num_vars() {
            return Err(XlogError::Compilation(format!(
                "Exact inference error: DDNNF references var {} but CNF has only {} vars",
                ddnnf.max_var(),
                cnf.num_vars()
            )));
        }
        let circuit = Xgcf::from_ddnnf(&ddnnf)?;

        let weights = &self.base_log_weights;
        if (ddnnf.max_var() as usize) >= weights.len() {
            return Err(XlogError::Compilation(format!(
                "Exact inference error: var {} out of bounds for weight table (len={})",
                ddnnf.max_var(),
                weights.len()
            )));
        }

        circuit.eval_log_wmc(|var| weights[var as usize])
    }
}

fn dimacs_with_units(cnf: &crate::cnf::CnfFormula, units: &[i32]) -> String {
    let mut out = String::new();
    out.push_str("c xlog-prob cnf\n");
    out.push_str(&format!(
        "p cnf {} {}\n",
        cnf.num_vars(),
        cnf.clauses().len() + units.len()
    ));
    for clause in cnf.clauses() {
        for lit in clause {
            out.push_str(&format!("{} ", lit));
        }
        out.push_str("0\n");
    }
    for &u in units {
        out.push_str(&format!("{} 0\n", u));
    }
    out
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
