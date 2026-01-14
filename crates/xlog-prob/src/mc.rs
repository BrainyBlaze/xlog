//! Approximate probabilistic inference via Monte Carlo sampling (P3).
//!
//! This engine samples probabilistic facts / annotated disjunction decisions on the GPU and
//! evaluates the deterministic core in each sampled world.
//!
//! For programs with non-monotone recursion (cycles through `not` and/or aggregates), Phase 4
//! requires explicit P3 opt-in. The deterministic evaluation uses a bounded, cycle-aware semantics:
//!
//! - If an SCC reaches a fixpoint under synchronous iteration, that fixpoint is used.
//! - If the SCC enters a cycle, the interpretation is the intersection of all states in the cycle
//!   (skeptical, invariant tuples only). This avoids parity/oscillation dependence on iteration
//!   count while remaining fully deterministic and explicit.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::ast::{
    AggExpr, AggOp, AnnotatedDisjunction, Atom, BodyLiteral, Evidence, ProbFact, ProbQuery, Program,
    Rule, Term,
};
use xlog_logic::stratify::{build_dependency_graph, find_sccs_for_lowering};

use crate::exact::GpuConfig;
use crate::provenance::{
    atom_key_from_ground_atom, eval_arith_expr, eval_comparison, unify_atom, validate_prob,
    value_from_term, GroundAtom, Value,
};

/// Phase 4 semantics for non-monotone SCC evaluation inside MC sampling.
pub const NONMONOTONE_SEMANTICS: &str = "Synchronous iteration per SCC; if a fixpoint is reached, use it; if a cycle is detected, use the intersection of all states in the cycle (skeptical tuples only); if the iteration budget is exceeded, use the intersection across all visited states (conservative).";

#[derive(Debug, Clone)]
pub struct McEvalConfig {
    /// Number of Monte Carlo samples.
    pub samples: usize,
    /// RNG seed (deterministic).
    pub seed: u64,
    /// Two-sided confidence level in (0,1) (e.g., 0.95).
    pub confidence: f64,
    /// Maximum SCC iteration steps for non-monotone cycle detection.
    pub max_nonmonotone_iterations: usize,
}

impl Default for McEvalConfig {
    fn default() -> Self {
        Self {
            samples: 10000,
            seed: 0,
            confidence: 0.95,
            max_nonmonotone_iterations: 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct McQueryEstimate {
    pub atom: GroundAtom,
    pub prob: f64,
    pub log_prob: f64,
    pub stderr: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

#[derive(Debug, Clone)]
pub struct McResult {
    pub total_samples: usize,
    pub evidence_samples: usize,
    pub seed: u64,
    pub confidence: f64,
    pub query_estimates: Vec<McQueryEstimate>,
    pub nonmonotone_sccs: usize,
    pub nonmonotone_cycles: usize,
    pub nonmonotone_iteration_limit_hits: usize,
}

#[derive(Debug, Clone)]
struct ProbFactSpec {
    var_idx: usize,
    atom: GroundAtom,
}

#[derive(Debug, Clone)]
struct AdSpec {
    decision_vars: Vec<usize>,
    choices: Vec<GroundAtom>,
    has_none: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Relation {
    tuples: HashSet<Vec<Value>>,
}

impl Relation {
    fn insert_tuple(&mut self, tuple: Vec<Value>) {
        self.tuples.insert(tuple);
    }

    fn contains(&self, tuple: &[Value]) -> bool {
        self.tuples.contains(tuple)
    }

    fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }
}

#[derive(Debug, Clone)]
enum SccKind {
    MonotoneNonRecursive,
    MonotoneRecursive,
    NonMonotone,
}

#[derive(Debug, Clone)]
struct SccPlan {
    predicates: Vec<String>,
    rules: Vec<Rule>,
    kind: SccKind,
}

#[derive(Debug, Clone, Default)]
struct EvalStats {
    nonmonotone_sccs: usize,
    nonmonotone_cycles: usize,
    nonmonotone_iteration_limit_hits: usize,
}

#[derive(Clone)]
pub struct McProgram {
    gpu_config: GpuConfig,
    base_store: HashMap<String, Relation>,
    scc_plans: Vec<SccPlan>,
    queries: Vec<GroundAtom>,
    evidence: Vec<(GroundAtom, bool)>,
    bernoulli_probs: Vec<f32>,
    prob_facts: Vec<ProbFactSpec>,
    annotated_disjunctions: Vec<AdSpec>,
}

impl McProgram {
    pub fn compile_source(source: &str) -> Result<Self> {
        let program = xlog_logic::parse_program(source)?;
        Self::compile_program(&program)
    }

    pub fn compile_source_with_gpu(source: &str, config: GpuConfig) -> Result<Self> {
        let mut program = Self::compile_source(source)?;
        program.gpu_config = config;
        Ok(program)
    }

    pub fn num_vars(&self) -> usize {
        self.bernoulli_probs.len()
    }

    pub fn evaluate(&self, cfg: McEvalConfig) -> Result<McResult> {
        if cfg.samples == 0 {
            return Err(XlogError::Execution(
                "MC inference requires samples > 0".to_string(),
            ));
        }
        if !(0.0 < cfg.confidence && cfg.confidence < 1.0) || cfg.confidence.is_nan() {
            return Err(XlogError::Execution(format!(
                "MC inference requires 0 < confidence < 1, got {}",
                cfg.confidence
            )));
        }
        if cfg.max_nonmonotone_iterations == 0 {
            return Err(XlogError::Execution(
                "MC inference requires max_nonmonotone_iterations > 0".to_string(),
            ));
        }

        let mut n_evidence: usize = 0;
        let mut n_query_true: Vec<usize> = vec![0; self.queries.len()];
        let mut stats = EvalStats::default();

        let num_vars = self.bernoulli_probs.len();
        let samples_matrix: Vec<u8> = if num_vars == 0 {
            Vec::new()
        } else {
            let provider = self.provider()?;
            provider.sample_bernoulli_matrix(&self.bernoulli_probs, cfg.samples, cfg.seed)?
        };

        for sample_idx in 0..cfg.samples {
            let sample_bits = if num_vars == 0 {
                &[][..]
            } else {
                let start = sample_idx * num_vars;
                let end = start + num_vars;
                &samples_matrix[start..end]
            };

            let mut store = self.base_store.clone();
            self.apply_sample_facts(&mut store, sample_bits)?;

            let sample_stats =
                evaluate_program_inplace(&self.scc_plans, &mut store, cfg.max_nonmonotone_iterations)?;
            stats.nonmonotone_sccs += sample_stats.nonmonotone_sccs;
            stats.nonmonotone_cycles += sample_stats.nonmonotone_cycles;
            stats.nonmonotone_iteration_limit_hits += sample_stats.nonmonotone_iteration_limit_hits;

            if !evidence_satisfied(&store, &self.evidence) {
                continue;
            }

            n_evidence += 1;
            for (i, q) in self.queries.iter().enumerate() {
                if atom_holds(&store, q) {
                    n_query_true[i] += 1;
                }
            }
        }

        if !self.evidence.is_empty() && n_evidence == 0 {
            return Err(XlogError::Execution(format!(
                "MC inference error: evidence was never satisfied across {} samples (seed={})",
                cfg.samples, cfg.seed
            )));
        }

        // If there is no evidence, treat all samples as evidence-satisfying.
        let denom = if self.evidence.is_empty() {
            cfg.samples
        } else {
            n_evidence
        };

        let z = normal_quantile(0.5 + cfg.confidence / 2.0);

        let mut query_estimates: Vec<McQueryEstimate> = Vec::with_capacity(self.queries.len());
        for (i, atom) in self.queries.iter().enumerate() {
            let k = n_query_true[i];
            let (p, stderr, ci_low, ci_high) = binomial_estimate(k, denom, z);
            let log_prob = if p == 0.0 { f64::NEG_INFINITY } else { p.ln() };

            query_estimates.push(McQueryEstimate {
                atom: atom.clone(),
                prob: p,
                log_prob,
                stderr,
                ci_low,
                ci_high,
            });
        }

        Ok(McResult {
            total_samples: cfg.samples,
            evidence_samples: denom,
            seed: cfg.seed,
            confidence: cfg.confidence,
            query_estimates,
            nonmonotone_sccs: stats.nonmonotone_sccs,
            nonmonotone_cycles: stats.nonmonotone_cycles,
            nonmonotone_iteration_limit_hits: stats.nonmonotone_iteration_limit_hits,
        })
    }

    fn compile_program(program: &Program) -> Result<Self> {
        let mut queries: Vec<GroundAtom> = Vec::new();
        for ProbQuery { atom } in &program.prob_queries {
            queries.push(atom_key_from_ground_atom(atom)?);
        }

        let mut evidence: Vec<(GroundAtom, bool)> = Vec::new();
        for Evidence { atom, value } in &program.evidence {
            evidence.push((atom_key_from_ground_atom(atom)?, *value));
        }

        let (bernoulli_probs, prob_facts, annotated_disjunctions) =
            compile_sampling_plan(&program.prob_facts, &program.annotated_disjunctions)?;

        let mut base_store: HashMap<String, Relation> = HashMap::new();

        // Deterministic facts.
        for fact in program.facts() {
            let atom = atom_key_from_ground_atom(&fact.head)?;
            base_store
                .entry(atom.predicate.clone())
                .or_default()
                .insert_tuple(atom.args);
        }

        // Ensure relations exist for all referenced predicates so evaluation treats missing as empty,
        // but never errors due to an unknown predicate.
        let mut referenced: HashSet<String> = HashSet::new();
        for rule in &program.rules {
            referenced.insert(rule.head.predicate.clone());
            for lit in &rule.body {
                match lit {
                    BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => {
                        referenced.insert(a.predicate.clone());
                    }
                    BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => {}
                }
            }
        }
        for pf in &program.prob_facts {
            referenced.insert(pf.atom.predicate.clone());
        }
        for ad in &program.annotated_disjunctions {
            for pf in &ad.choices {
                referenced.insert(pf.atom.predicate.clone());
            }
        }
        for q in &queries {
            referenced.insert(q.predicate.clone());
        }
        for (e, _) in &evidence {
            referenced.insert(e.predicate.clone());
        }
        for pred in referenced {
            base_store.entry(pred).or_default();
        }

        let scc_plans = build_scc_plans(program)?;

        Ok(Self {
            gpu_config: GpuConfig::default(),
            base_store,
            scc_plans,
            queries,
            evidence,
            bernoulli_probs,
            prob_facts,
            annotated_disjunctions,
        })
    }

    fn provider(&self) -> Result<CudaKernelProvider> {
        let device_count = cudarc::driver::CudaDevice::count().unwrap_or(0) as usize;
        if device_count == 0 {
            return Err(XlogError::Kernel("No CUDA device available".to_string()));
        }
        if self.gpu_config.device_ordinal >= device_count {
            return Err(XlogError::Kernel(format!(
                "CUDA device ordinal {} out of range (count={})",
                self.gpu_config.device_ordinal, device_count
            )));
        }
        if self.gpu_config.memory_bytes == 0 {
            return Err(XlogError::Kernel(
                "GPU memory budget must be non-zero".to_string(),
            ));
        }

        let device = Arc::new(CudaDevice::new(self.gpu_config.device_ordinal)?);
        let memory = Arc::new(GpuMemoryManager::new(
            device.clone(),
            MemoryBudget::with_limit(self.gpu_config.memory_bytes),
        ));
        CudaKernelProvider::new(device, memory)
    }

    fn apply_sample_facts(&self, store: &mut HashMap<String, Relation>, bits: &[u8]) -> Result<()> {
        for pf in &self.prob_facts {
            if bits.get(pf.var_idx).copied().unwrap_or(0) == 0 {
                continue;
            }
            store
                .entry(pf.atom.predicate.clone())
                .or_default()
                .insert_tuple(pf.atom.args.clone());
        }

        for ad in &self.annotated_disjunctions {
            let mut selected: Option<usize> = None;
            for (idx, &var_idx) in ad.decision_vars.iter().enumerate() {
                if bits.get(var_idx).copied().unwrap_or(0) != 0 {
                    selected = Some(idx);
                    break;
                }
            }

            let outcome = match selected {
                Some(i) => Some(i),
                None => {
                    if ad.has_none {
                        None
                    } else {
                        Some(ad.choices.len().saturating_sub(1))
                    }
                }
            };

            if let Some(outcome_idx) = outcome {
                let atom = ad.choices.get(outcome_idx).ok_or_else(|| {
                    XlogError::Compilation("Annotated disjunction outcome index out of range".to_string())
                })?;
                store
                    .entry(atom.predicate.clone())
                    .or_default()
                    .insert_tuple(atom.args.clone());
            }
        }

        Ok(())
    }
}

fn compile_sampling_plan(
    prob_facts: &[ProbFact],
    annotated_disjunctions: &[AnnotatedDisjunction],
) -> Result<(Vec<f32>, Vec<ProbFactSpec>, Vec<AdSpec>)> {
    let mut probs: Vec<f32> = Vec::new();
    let mut fact_specs: Vec<ProbFactSpec> = Vec::new();
    let mut ad_specs: Vec<AdSpec> = Vec::new();

    for pf in prob_facts {
        validate_prob(pf.prob, "probabilistic fact")?;
        let atom = atom_key_from_ground_atom(&pf.atom)?;
        let var_idx = probs.len();
        probs.push(pf.prob as f32);
        fact_specs.push(ProbFactSpec { var_idx, atom });
    }

    for ad in annotated_disjunctions {
        if ad.choices.is_empty() {
            return Err(XlogError::Compilation(
                "Annotated disjunction must contain at least one choice".to_string(),
            ));
        }

        let mut choice_atoms: Vec<GroundAtom> = Vec::with_capacity(ad.choices.len());
        let mut choice_probs: Vec<f64> = Vec::with_capacity(ad.choices.len());
        for pf in &ad.choices {
            validate_prob(pf.prob, "annotated disjunction choice")?;
            choice_atoms.push(atom_key_from_ground_atom(&pf.atom)?);
            choice_probs.push(pf.prob);
        }

        let sum: f64 = choice_probs.iter().copied().sum();
        let eps = 1e-12;
        if sum > 1.0 + eps {
            return Err(XlogError::Compilation(format!(
                "Annotated disjunction probabilities sum to {} (> 1.0)",
                sum
            )));
        }

        let has_none = (1.0 - sum) > eps;
        let mut probs_full: Vec<f64> = choice_probs.clone();
        if has_none {
            probs_full.push((1.0 - sum).max(0.0));
        }

        // Encode categorical choice as a chain of Bernoulli decisions (same as provenance lowering).
        let m = probs_full.len();
        let mut decision_vars: Vec<usize> = Vec::new();
        if m > 1 {
            let mut remaining = 1.0f64;
            for i in 0..(m - 1) {
                let p_i = probs_full[i];
                let cond_true = if remaining <= 0.0 { 0.0 } else { p_i / remaining };
                validate_prob(cond_true, "annotated disjunction conditional")?;
                probs.push(cond_true as f32);
                decision_vars.push(probs.len() - 1);
                remaining -= p_i;
            }
        }

        ad_specs.push(AdSpec {
            decision_vars,
            choices: choice_atoms,
            has_none,
        });
    }

    Ok((probs, fact_specs, ad_specs))
}

fn build_scc_plans(program: &Program) -> Result<Vec<SccPlan>> {
    let graph = build_dependency_graph(program);
    let sccs = find_sccs_for_lowering(&graph);

    let mut rules_by_head: HashMap<String, Vec<Rule>> = HashMap::new();
    for rule in program.proper_rules() {
        rules_by_head
            .entry(rule.head.predicate.clone())
            .or_default()
            .push(rule.clone());
    }

    let mut plans: Vec<SccPlan> = Vec::new();
    for scc in sccs {
        let mut scc_rules: Vec<Rule> = Vec::new();
        for pred in &scc {
            if let Some(rules) = rules_by_head.get(pred) {
                scc_rules.extend(rules.iter().cloned());
            }
        }
        if scc_rules.is_empty() {
            continue;
        }

        let predicate_set: HashSet<String> = scc.iter().cloned().collect();

        let nonmonotone = scc_rules.iter().any(|rule| {
            rule.body.iter().any(|lit| match lit {
                BodyLiteral::Negated(atom) => predicate_set.contains(&atom.predicate),
                BodyLiteral::Positive(atom) if rule.has_aggregation() => predicate_set.contains(&atom.predicate),
                _ => false,
            })
        });

        let kind = if nonmonotone {
            SccKind::NonMonotone
        } else if is_recursive_scc(&scc, &scc_rules) {
            SccKind::MonotoneRecursive
        } else {
            SccKind::MonotoneNonRecursive
        };

        plans.push(SccPlan {
            predicates: scc,
            rules: scc_rules,
            kind,
        });
    }

    Ok(plans)
}

fn is_recursive_scc(scc: &[String], rules: &[Rule]) -> bool {
    if scc.len() > 1 {
        return true;
    }
    let Some(only) = scc.first() else {
        return false;
    };
    for rule in rules {
        for lit in &rule.body {
            if let BodyLiteral::Positive(atom) = lit {
                if &atom.predicate == only {
                    return true;
                }
            }
        }
    }
    false
}

fn evaluate_program_inplace(
    scc_plans: &[SccPlan],
    store: &mut HashMap<String, Relation>,
    max_nonmonotone_iterations: usize,
) -> Result<EvalStats> {
    let mut stats = EvalStats::default();

    for plan in scc_plans {
        match plan.kind {
            SccKind::MonotoneNonRecursive => {
                for rule in &plan.rules {
                    let derived = eval_rule(rule, store, &HashMap::new(), None)?;
                    let rel = store.entry(rule.head.predicate.clone()).or_default();
                    for tuple in derived {
                        rel.insert_tuple(tuple);
                    }
                }
            }
            SccKind::MonotoneRecursive => {
                eval_monotone_recursive_scc(&plan.predicates, &plan.rules, store)?;
            }
            SccKind::NonMonotone => {
                stats.nonmonotone_sccs += 1;
                let (cycle, hit_limit) = eval_nonmonotone_scc(
                    &plan.predicates,
                    &plan.rules,
                    store,
                    max_nonmonotone_iterations,
                )?;
                if cycle {
                    stats.nonmonotone_cycles += 1;
                }
                if hit_limit {
                    stats.nonmonotone_iteration_limit_hits += 1;
                }
            }
        }
    }

    Ok(stats)
}

fn eval_monotone_recursive_scc(
    scc: &[String],
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
) -> Result<()> {
    const MAX_ITERS: usize = 1024;

    let scc_set: HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();

    let mut full: HashMap<String, Relation> = HashMap::new();
    for pred in scc {
        let rel = store.get(pred).cloned().unwrap_or_default();
        full.insert(pred.clone(), rel);
    }

    let mut delta: HashMap<String, Relation> = HashMap::new();
    for rule in rules {
        let derived = eval_rule(rule, store, &full, None)?;
        if derived.is_empty() {
            continue;
        }
        let head = rule.head.predicate.clone();
        let delta_rel = delta.entry(head.clone()).or_default();
        let full_rel = full.entry(head).or_default();
        for tuple in derived {
            if full_rel.tuples.insert(tuple.clone()) {
                delta_rel.insert_tuple(tuple);
            }
        }
    }

    let mut reached_fixpoint = false;
    for _ in 0..MAX_ITERS {
        let any_delta = delta.values().any(|r| !r.is_empty());
        if !any_delta {
            reached_fixpoint = true;
            break;
        }

        let full_prev = full.clone();
        let delta_prev = delta.clone();
        delta.clear();

        for rule in rules {
            let body_indices: Vec<usize> = rule
                .body
                .iter()
                .enumerate()
                .filter_map(|(i, lit)| match lit {
                    BodyLiteral::Positive(atom) if scc_set.contains(atom.predicate.as_str()) => {
                        let non_empty = delta_prev
                            .get(&atom.predicate)
                            .map(|r| !r.is_empty())
                            .unwrap_or(false);
                        non_empty.then_some(i)
                    }
                    _ => None,
                })
                .collect();
            if body_indices.is_empty() {
                continue;
            }

            let mut derived_all: HashSet<Vec<Value>> = HashSet::new();
            for idx in body_indices {
                let derived = eval_rule(rule, store, &full_prev, Some((idx, &delta_prev)))?;
                derived_all.extend(derived);
            }
            if derived_all.is_empty() {
                continue;
            }

            let head = rule.head.predicate.clone();
            let delta_rel = delta.entry(head.clone()).or_default();
            let full_rel = full.entry(head).or_default();
            for tuple in derived_all {
                if full_rel.tuples.insert(tuple.clone()) {
                    delta_rel.insert_tuple(tuple);
                }
            }
        }
    }

    if !reached_fixpoint {
        return Err(XlogError::Execution(format!(
            "Monotone SCC fixpoint iteration limit ({}) exceeded for SCC {:?}",
            MAX_ITERS, scc
        )));
    }

    for (pred, rel) in full {
        store.insert(pred, rel);
    }
    Ok(())
}

fn eval_nonmonotone_scc(
    scc: &[String],
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
    max_iters: usize,
) -> Result<(bool, bool)> {
    let mut base: HashMap<String, Relation> = HashMap::new();
    for pred in scc {
        base.insert(pred.clone(), store.get(pred).cloned().unwrap_or_default());
    }

    let mut history: Vec<HashMap<String, Relation>> = Vec::new();
    history.push(base.clone());

    let mut seen: HashMap<u64, Vec<usize>> = HashMap::new();
    seen.entry(hash_scc_state(&history[0])).or_default().push(0);

    for _iter in 0..max_iters {
        let current = history
            .last()
            .expect("history non-empty")
            .clone();

        let mut next: HashMap<String, Relation> = base.clone();

        for rule in rules {
            let derived = eval_rule(rule, store, &current, None)?;
            let rel = next.entry(rule.head.predicate.clone()).or_default();
            for tuple in derived {
                rel.insert_tuple(tuple);
            }
        }

        if next == current {
            for (pred, rel) in next {
                store.insert(pred, rel);
            }
            return Ok((false, false));
        }

        let h = hash_scc_state(&next);
        if let Some(candidates) = seen.get(&h) {
            for &idx in candidates {
                if history.get(idx) == Some(&next) {
                    // Cycle detected: history[idx..] repeats.
                    let cycle_states = &history[idx..];
                    let final_state = intersect_states(cycle_states);
                    for (pred, rel) in final_state {
                        store.insert(pred, rel);
                    }
                    return Ok((true, false));
                }
            }
        }

        let next_index = history.len();
        history.push(next);
        seen.entry(h).or_default().push(next_index);
    }

    // Iteration budget exhausted: fall back to a conservative invariant set.
    let final_state = intersect_states(&history);
    for (pred, rel) in final_state {
        store.insert(pred, rel);
    }
    Ok((false, true))
}

fn intersect_states(states: &[HashMap<String, Relation>]) -> HashMap<String, Relation> {
    let mut out: HashMap<String, Relation> = HashMap::new();
    let Some(first) = states.first() else {
        return out;
    };

    for (pred, rel0) in first {
        let mut intersection: HashSet<Vec<Value>> = rel0.tuples.clone();
        for state in &states[1..] {
            if let Some(rel) = state.get(pred) {
                intersection.retain(|t| rel.tuples.contains(t));
            } else {
                intersection.clear();
                break;
            }
        }
        out.insert(pred.clone(), Relation { tuples: intersection });
    }

    out
}

fn hash_scc_state(state: &HashMap<String, Relation>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();

    let mut preds: Vec<&String> = state.keys().collect();
    preds.sort();
    for pred in preds {
        pred.hash(&mut hasher);
        if let Some(rel) = state.get(pred) {
            let mut tuple_hashes: Vec<u64> = Vec::with_capacity(rel.tuples.len());
            for tuple in &rel.tuples {
                let mut th = DefaultHasher::new();
                tuple.hash(&mut th);
                tuple_hashes.push(th.finish());
            }
            tuple_hashes.sort_unstable();
            tuple_hashes.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Evaluate a single rule and produce derived head tuples.
///
/// `full_scc` is a per-SCC snapshot (used for SCC predicates), and `delta_scc` optionally provides
/// a delta relation for a specific body literal index (semi-naive).
fn eval_rule(
    rule: &Rule,
    global: &HashMap<String, Relation>,
    full_scc: &HashMap<String, Relation>,
    delta_scc: Option<(usize, &HashMap<String, Relation>)>,
) -> Result<Vec<Vec<Value>>> {
    let mut states: Vec<HashMap<String, Value>> = Vec::new();
    states.push(HashMap::new());

    for (idx, lit) in rule.body.iter().enumerate() {
        let mut next_states: Vec<HashMap<String, Value>> = Vec::new();
        match lit {
            BodyLiteral::Positive(atom) => {
                let rel = select_relation(atom, idx, global, full_scc, delta_scc)?;
                for binding in states {
                    for tuple in &rel.tuples {
                        let mut binding2 = binding.clone();
                        if unify_atom(atom, tuple, &mut binding2)? {
                            next_states.push(binding2);
                        }
                    }
                }
            }
            BodyLiteral::Negated(atom) => {
                let rel = select_relation(atom, idx, global, full_scc, delta_scc)?;
                for binding in states {
                    if negated_atom_holds(atom, rel, &binding)? {
                        next_states.push(binding);
                    }
                }
            }
            BodyLiteral::Comparison(cmp) => {
                for binding in states {
                    if eval_comparison(cmp.op, &cmp.left, &cmp.right, &binding)? {
                        next_states.push(binding);
                    }
                }
            }
            BodyLiteral::IsExpr(is_expr) => {
                for mut binding in states {
                    if binding.contains_key(&is_expr.target) {
                        return Err(XlogError::Compilation(format!(
                            "Is-expression target {} is already bound",
                            is_expr.target
                        )));
                    }
                    let v = eval_arith_expr(&is_expr.expr, &binding)?;
                    binding.insert(is_expr.target.clone(), v);
                    next_states.push(binding);
                }
            }
        }
        states = next_states;
        if states.is_empty() {
            break;
        }
    }

    if rule.has_aggregation() {
        eval_aggregate_head(&rule.head, states)
    } else {
        let mut out: HashSet<Vec<Value>> = HashSet::new();
        for binding in states {
            let head_tuple = materialize_head_non_aggregate(&rule.head, &binding)?;
            out.insert(head_tuple);
        }
        Ok(out.into_iter().collect())
    }
}

fn select_relation<'a>(
    atom: &Atom,
    body_index: usize,
    global: &'a HashMap<String, Relation>,
    full_scc: &'a HashMap<String, Relation>,
    delta_scc: Option<(usize, &'a HashMap<String, Relation>)>,
) -> Result<&'a Relation> {
    if let Some((delta_index, delta_map)) = delta_scc {
        if delta_index == body_index {
            return delta_map.get(&atom.predicate).ok_or_else(|| {
                XlogError::Compilation(format!("Missing delta relation for predicate {}", atom.predicate))
            });
        }
    }
    if let Some(rel) = full_scc.get(&atom.predicate) {
        return Ok(rel);
    }
    global.get(&atom.predicate).ok_or_else(|| {
        XlogError::Compilation(format!("Unknown predicate {}", atom.predicate))
    })
}

fn negated_atom_holds(atom: &Atom, rel: &Relation, binding: &HashMap<String, Value>) -> Result<bool> {
    // Safety: all variables in a negated atom must be bound already (otherwise domain is unknown).
    for term in &atom.terms {
        if let Term::Variable(name) = term {
            if !binding.contains_key(name) {
                return Err(XlogError::UnsafeVariable(name.clone()));
            }
        }
    }

    for tuple in &rel.tuples {
        if atom_matches_bound(atom, tuple, binding)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn atom_matches_bound(atom: &Atom, tuple: &[Value], binding: &HashMap<String, Value>) -> Result<bool> {
    if atom.terms.len() != tuple.len() {
        return Err(XlogError::Compilation(format!(
            "Arity mismatch for {}: atom has {}, tuple has {}",
            atom.predicate,
            atom.terms.len(),
            tuple.len()
        )));
    }
    for (term, value) in atom.terms.iter().zip(tuple.iter()) {
        match term {
            Term::Variable(name) => {
                let Some(bound) = binding.get(name) else {
                    return Err(XlogError::UnsafeVariable(name.clone()));
                };
                if bound != value {
                    return Ok(false);
                }
            }
            Term::Anonymous => {}
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                if &value_from_term(term)? != value {
                    return Ok(false);
                }
            }
            Term::Aggregate(AggExpr { .. }) => {
                return Err(XlogError::Compilation(
                    "Aggregate not allowed in body atom".to_string(),
                ));
            }
        }
    }
    Ok(true)
}

fn materialize_head_non_aggregate(head: &Atom, binding: &HashMap<String, Value>) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(head.terms.len());
    for term in &head.terms {
        match term {
            Term::Variable(name) => {
                let v = binding.get(name).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "Unbound head variable {} in {}",
                        name, head.predicate
                    ))
                })?;
                out.push(v.clone());
            }
            Term::Anonymous => {
                return Err(XlogError::Compilation(format!(
                    "Anonymous wildcard '_' not allowed in rule head (predicate {})",
                    head.predicate
                )));
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                out.push(value_from_term(term)?);
            }
            Term::Aggregate(_) => {
                return Err(XlogError::Compilation(
                    "Aggregate term in non-aggregate rule head".to_string(),
                ));
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
enum AggState {
    Count(u64),
    SumI128(i128),
    SumF64(f64),
    Min(Option<Value>),
    Max(Option<Value>),
    LogSumExp { max: f64, sumexp: f64, init: bool },
}

impl AggState {
    fn new(op: AggOp) -> Self {
        match op {
            AggOp::Count => AggState::Count(0),
            AggOp::Sum => AggState::SumI128(0),
            AggOp::Min => AggState::Min(None),
            AggOp::Max => AggState::Max(None),
            AggOp::LogSumExp => AggState::LogSumExp {
                max: f64::NEG_INFINITY,
                sumexp: 0.0,
                init: false,
            },
        }
    }

    fn update(&mut self, op: AggOp, v: &Value) -> Result<()> {
        match op {
            AggOp::Count => match self {
                AggState::Count(c) => {
                    *c = c.saturating_add(1);
                    Ok(())
                }
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
            AggOp::Sum => match self {
                AggState::SumI128(acc) => match v {
                    Value::I64(i) => {
                        *acc += *i as i128;
                        Ok(())
                    }
                    Value::F64(bits) => {
                        let f = f64::from_bits(*bits);
                        let acc_f = *acc as f64;
                        *self = AggState::SumF64(acc_f + f);
                        Ok(())
                    }
                    _ => Err(XlogError::Compilation(
                        "sum() aggregate requires numeric input".to_string(),
                    )),
                },
                AggState::SumF64(acc) => match v {
                    Value::I64(i) => {
                        *acc += *i as f64;
                        Ok(())
                    }
                    Value::F64(bits) => {
                        *acc += f64::from_bits(*bits);
                        Ok(())
                    }
                    _ => Err(XlogError::Compilation(
                        "sum() aggregate requires numeric input".to_string(),
                    )),
                },
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
            AggOp::Min => match self {
                AggState::Min(current) => {
                    match current {
                        None => *current = Some(v.clone()),
                        Some(c) => {
                            if value_le(v, c)? {
                                *current = Some(v.clone());
                            }
                        }
                    }
                    Ok(())
                }
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
            AggOp::Max => match self {
                AggState::Max(current) => {
                    match current {
                        None => *current = Some(v.clone()),
                        Some(c) => {
                            if value_le(c, v)? {
                                *current = Some(v.clone());
                            }
                        }
                    }
                    Ok(())
                }
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
            AggOp::LogSumExp => match self {
                AggState::LogSumExp { max, sumexp, init } => {
                    let x = match v {
                        Value::I64(i) => *i as f64,
                        Value::F64(bits) => f64::from_bits(*bits),
                        _ => {
                            return Err(XlogError::Compilation(
                                "logsumexp() aggregate requires numeric input".to_string(),
                            ))
                        }
                    };
                    if x.is_nan() {
                        return Err(XlogError::Compilation(
                            "logsumexp() aggregate encountered NaN".to_string(),
                        ));
                    }
                    if !*init {
                        *max = x;
                        *sumexp = 1.0;
                        *init = true;
                        return Ok(());
                    }
                    if x > *max {
                        *sumexp = *sumexp * (*max - x).exp() + 1.0;
                        *max = x;
                    } else {
                        *sumexp += (x - *max).exp();
                    }
                    Ok(())
                }
                _ => Err(XlogError::Compilation(
                    "Internal aggregate state mismatch".to_string(),
                )),
            },
        }
    }

    fn finish(&self, op: AggOp) -> Result<Value> {
        match (op, self) {
            (AggOp::Count, AggState::Count(c)) => {
                let v: i64 = (*c).try_into().map_err(|_| {
                    XlogError::Compilation("count() overflowed i64".to_string())
                })?;
                Ok(Value::I64(v))
            }
            (AggOp::Sum, AggState::SumI128(acc)) => {
                let v: i64 = (*acc).try_into().map_err(|_| {
                    XlogError::Compilation("sum() overflowed i64".to_string())
                })?;
                Ok(Value::I64(v))
            }
            (AggOp::Sum, AggState::SumF64(v)) => Ok(Value::F64(v.to_bits())),
            (AggOp::Min, AggState::Min(v)) => v.clone().ok_or_else(|| {
                XlogError::Compilation("min() aggregate produced no value".to_string())
            }),
            (AggOp::Max, AggState::Max(v)) => v.clone().ok_or_else(|| {
                XlogError::Compilation("max() aggregate produced no value".to_string())
            }),
            (AggOp::LogSumExp, AggState::LogSumExp { max, sumexp, init }) => {
                if !*init {
                    return Ok(Value::F64(f64::NEG_INFINITY.to_bits()));
                }
                Ok(Value::F64((max + sumexp.ln()).to_bits()))
            }
            _ => Err(XlogError::Compilation(
                "Internal aggregate state mismatch".to_string(),
            )),
        }
    }
}

fn value_le(a: &Value, b: &Value) -> Result<bool> {
    match (a, b) {
        (Value::I64(x), Value::I64(y)) => Ok(x <= y),
        (Value::F64(x), Value::F64(y)) => Ok(f64::from_bits(*x) <= f64::from_bits(*y)),
        (Value::Symbol(x), Value::Symbol(y)) => Ok(x <= y),
        (Value::String(x), Value::String(y)) => Ok(x <= y),
        _ => Err(XlogError::Compilation(
            "min/max aggregate requires consistent comparable types".to_string(),
        )),
    }
}

fn eval_aggregate_head(head: &Atom, states: Vec<HashMap<String, Value>>) -> Result<Vec<Vec<Value>>> {
    // Collect unique group keys (variables) in head order.
    let mut key_vars: Vec<String> = Vec::new();
    let mut key_var_to_pos: HashMap<String, usize> = HashMap::new();

    // Collect unique aggregate specs (op, var) in head order.
    let mut agg_specs: Vec<(AggOp, String)> = Vec::new();
    let mut agg_to_pos: HashMap<(AggOp, String), usize> = HashMap::new();

    for term in &head.terms {
        match term {
            Term::Variable(name) => {
                if !key_var_to_pos.contains_key(name) {
                    let pos = key_vars.len();
                    key_vars.push(name.clone());
                    key_var_to_pos.insert(name.clone(), pos);
                }
            }
            Term::Aggregate(agg) => {
                let key = (agg.op, agg.variable.clone());
                if !agg_to_pos.contains_key(&key) {
                    let pos = agg_specs.len();
                    agg_specs.push(key.clone());
                    agg_to_pos.insert(key, pos);
                }
            }
            Term::Anonymous => {
                return Err(XlogError::Compilation(
                    "Anonymous wildcard '_' not allowed in aggregate rule head".to_string(),
                ));
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {}
        }
    }

    #[derive(Debug)]
    struct GroupState {
        key: Vec<Value>,
        aggs: Vec<AggState>,
    }

    let mut groups: HashMap<Vec<Value>, GroupState> = HashMap::new();

    for binding in states {
        let mut key: Vec<Value> = Vec::with_capacity(key_vars.len());
        for name in &key_vars {
            let v = binding.get(name).ok_or_else(|| XlogError::UnsafeVariable(name.clone()))?;
            key.push(v.clone());
        }

        let entry = groups.entry(key.clone()).or_insert_with(|| GroupState {
            key,
            aggs: agg_specs.iter().map(|(op, _)| AggState::new(*op)).collect(),
        });

        for (idx, (op, var)) in agg_specs.iter().enumerate() {
            let v = binding.get(var).ok_or_else(|| XlogError::UnsafeVariable(var.clone()))?;
            entry.aggs[idx].update(*op, v)?;
        }
    }

    let mut out: Vec<Vec<Value>> = Vec::with_capacity(groups.len());
    for group in groups.values() {
        let mut tuple: Vec<Value> = Vec::with_capacity(head.terms.len());
        for term in &head.terms {
            match term {
                Term::Variable(name) => {
                    let pos = *key_var_to_pos.get(name).ok_or_else(|| {
                        XlogError::Compilation(format!(
                            "Aggregate head variable {} is not a group key",
                            name
                        ))
                    })?;
                    tuple.push(group.key[pos].clone());
                }
                Term::Aggregate(AggExpr { op, variable }) => {
                    let idx = *agg_to_pos
                        .get(&(*op, variable.clone()))
                        .expect("agg_to_pos missing");
                    tuple.push(group.aggs[idx].finish(*op)?);
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    tuple.push(value_from_term(term)?);
                }
                Term::Anonymous => unreachable!(),
            }
        }
        out.push(tuple);
    }

    Ok(out)
}

fn atom_holds(store: &HashMap<String, Relation>, atom: &GroundAtom) -> bool {
    store
        .get(&atom.predicate)
        .map(|rel| rel.contains(&atom.args))
        .unwrap_or(false)
}

fn evidence_satisfied(store: &HashMap<String, Relation>, evidence: &[(GroundAtom, bool)]) -> bool {
    for (atom, value) in evidence {
        let holds = atom_holds(store, atom);
        if holds != *value {
            return false;
        }
    }
    true
}

fn binomial_estimate(k: usize, n: usize, z: f64) -> (f64, f64, f64, f64) {
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n_f = n as f64;
    let p = (k as f64) / n_f;
    let stderr = (p * (1.0 - p) / n_f).sqrt();

    let z2 = z * z;
    let denom = 1.0 + z2 / n_f;
    let center = (p + z2 / (2.0 * n_f)) / denom;
    let half = z * ((p * (1.0 - p) / n_f) + (z2 / (4.0 * n_f * n_f))).sqrt() / denom;

    let mut lo = center - half;
    let mut hi = center + half;
    if lo < 0.0 {
        lo = 0.0;
    }
    if hi > 1.0 {
        hi = 1.0;
    }
    (p, stderr, lo, hi)
}

// Acklam's inverse normal CDF approximation.
fn normal_quantile(p: f64) -> f64 {
    if !(0.0 < p && p < 1.0) || p.is_nan() {
        return f64::NAN;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > P_HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }

    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}
