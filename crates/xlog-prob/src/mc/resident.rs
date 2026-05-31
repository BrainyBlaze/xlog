//! GPU-resident Datalog/MC execution engine.
//!
//! This module replaces the host-orchestrated per-sample MC loop with a single
//! device megakernel (`mc_resident_engine`) that evaluates *all* Monte Carlo
//! worlds to a device-side fixpoint and counts query/evidence satisfaction with
//! **zero host interaction inside the measured region**: no host loop over
//! samples, no per-sample host kernel sequencing, no host metadata reads
//! (tracked or untracked).
//!
//! Supported fragment (bounded-domain positive Datalog), checked structurally —
//! never by predicate name:
//!   * predicate arity <= [`MAX_ARITY`];
//!   * rule body length <= [`MAX_BODY`] positive literals (no negation,
//!     comparison, arithmetic, epistemic, aggregate, or univ literals);
//!   * <= [`MAX_VARS`] distinct variables per rule;
//!   * all terms are variables or ground constants (no lists/compounds/functors);
//!   * bounded Herbrand universe: `domain^arity` slots, with the total universe
//!     and per-predicate domain capped so one world fits one CUDA block.
//!
//! Anything outside the fragment is rejected **before execution** with a typed
//! [`ResidentRejection`] (fail-closed; no CPU fallback).

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::LaunchConfig;
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{mc_resident_kernels, MC_RESIDENT_MODULE};
use xlog_cuda::{CudaKernelProvider, LaunchAsync};
use xlog_logic::ast::{Atom, BodyLiteral, Term};

use super::{McEvalConfig, McProgram, McSamplingMethod};
use crate::provenance::{GroundAtom, Value};

/// Maximum supported predicate arity.
pub const MAX_ARITY: usize = 2;
/// Maximum supported rule body length (positive literals).
pub const MAX_BODY: usize = 2;
/// Maximum supported distinct variables per rule.
pub const MAX_VARS: usize = 3;
/// Maximum supported bounded-universe slot count (one world must fit one block's
/// working set comfortably).
pub const MAX_UNIVERSE: usize = 1 << 16;
/// Maximum supported domain size (distinct constants).
pub const MAX_DOMAIN: usize = 256;
/// u32 width of one device atom record: base, arity, arg0, arg1, stride0.
const ATOM_REC: usize = 5;
/// u32 width of one device rule record: n_body, n_vars, domain, head, body0, body1.
const RULE_REC: usize = 3 + 3 * ATOM_REC;
/// Device encoding flag marking an arg spec as a bound constant.
const CONST_FLAG: u32 = 0x8000_0000;

/// Kind of a fail-closed rejection of an MC program by the resident engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentRejectKind {
    /// A body literal uses negation.
    Negation,
    /// A body literal is epistemic (`know`/`possible`).
    EpistemicLiteral,
    /// A body literal is a comparison / arithmetic / univ (non-relational).
    NonRelationalLiteral,
    /// A predicate arity exceeds [`MAX_ARITY`].
    ArityTooHigh,
    /// A rule body has more than [`MAX_BODY`] literals.
    BodyTooLong,
    /// A rule uses more than [`MAX_VARS`] distinct variables.
    TooManyVars,
    /// A term is not a variable or ground constant (list/compound/functor/agg).
    UnboundedTerm,
    /// The bounded domain exceeds [`MAX_DOMAIN`].
    DomainTooLarge,
    /// The bounded universe exceeds [`MAX_UNIVERSE`].
    UniverseTooLarge,
    /// A predicate appears with inconsistent arity.
    InconsistentArity,
    /// Annotated disjunctions are not yet supported by the resident engine.
    AnnotatedDisjunctionUnsupported,
}

impl ResidentRejectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ResidentRejectKind::Negation => "negation",
            ResidentRejectKind::EpistemicLiteral => "epistemic_literal",
            ResidentRejectKind::NonRelationalLiteral => "non_relational_literal",
            ResidentRejectKind::ArityTooHigh => "arity_too_high",
            ResidentRejectKind::BodyTooLong => "body_too_long",
            ResidentRejectKind::TooManyVars => "too_many_vars",
            ResidentRejectKind::UnboundedTerm => "unbounded_term",
            ResidentRejectKind::DomainTooLarge => "domain_too_large",
            ResidentRejectKind::UniverseTooLarge => "universe_too_large",
            ResidentRejectKind::InconsistentArity => "inconsistent_arity",
            ResidentRejectKind::AnnotatedDisjunctionUnsupported => {
                "annotated_disjunction_unsupported"
            }
        }
    }
}

/// A typed fail-closed rejection: which rule was violated, the offending
/// construct, and the surrounding context (for diagnostics).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentRejection {
    pub kind: ResidentRejectKind,
    pub construct: String,
    pub context: String,
}

impl ResidentRejection {
    fn err(
        kind: ResidentRejectKind,
        construct: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        ResidentRejection {
            kind,
            construct: construct.into(),
            context: context.into(),
        }
    }

    /// Convert to the engine's unified error type while preserving the kind +
    /// construct + context in the message for callers that only see `XlogError`.
    pub fn into_error(self) -> XlogError {
        XlogError::Compilation(format!(
            "resident MC engine rejected program [kind={}] construct=`{}` context=`{}`",
            self.kind.as_str(),
            self.construct,
            self.context
        ))
    }
}

/// Canonical key for a ground constant, unifying `Term` literals and `Value`s so
/// the same constant in a rule and in a fact map to one domain index.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ConstKey {
    Int(i64),
    Sym(u32),
    Str(String),
    FloatBits(u64),
}

impl ConstKey {
    fn from_value(v: &Value) -> ConstKey {
        match v {
            Value::I64(i) => ConstKey::Int(*i),
            Value::Symbol(s) => ConstKey::Sym(*s),
            Value::String(s) => ConstKey::Str(s.clone()),
            Value::F64(bits) => ConstKey::FloatBits(*bits),
        }
    }

    /// Map a rule term to either a constant key or a variable name.
    fn from_term(t: &Term) -> std::result::Result<TermClass, ResidentRejection> {
        match t {
            Term::Variable(name) => Ok(TermClass::Var(name.clone())),
            Term::Integer(i) => Ok(TermClass::Const(ConstKey::Int(*i))),
            Term::Symbol(s) => Ok(TermClass::Const(ConstKey::Sym(*s))),
            Term::String(s) => Ok(TermClass::Const(ConstKey::Str(s.clone()))),
            Term::Float(f) => Ok(TermClass::Const(ConstKey::FloatBits(f.to_bits()))),
            other => Err(ResidentRejection::err(
                ResidentRejectKind::UnboundedTerm,
                format!("{:?}", other),
                "rule term must be a variable or ground constant",
            )),
        }
    }
}

enum TermClass {
    Var(String),
    Const(ConstKey),
}

/// No-host-interaction instrumentation for the measured engine region.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McNoHostStats {
    /// Tracked data-plane host-to-device calls inside the measured region.
    pub tracked_htod_calls: u64,
    /// Tracked data-plane device-to-host calls inside the measured region.
    pub tracked_dtoh_calls: u64,
    /// Untracked control-plane metadata reads inside the measured region.
    pub untracked_metadata_reads: u64,
    /// Number of kernel launches the engine issued inside the measured region.
    pub engine_launches: u64,
    /// Number of host-side per-sample loop iterations inside the measured region
    /// (structurally zero: the engine has no host sample loop).
    pub host_loop_iterations: u64,
    /// Number of per-sample host launches inside the measured region
    /// (structurally zero: one global launch covers all worlds).
    pub per_sample_host_launches: u64,
}

impl McNoHostStats {
    /// True iff the measured region had **no host interaction**: no tracked
    /// transfers, no untracked metadata reads, no host sample loop, no
    /// per-sample host launches. (A single global engine launch is permitted
    /// and is *not* per-sample; see [`Self::engine_launches`].)
    pub fn is_no_host(&self) -> bool {
        self.tracked_htod_calls == 0
            && self.tracked_dtoh_calls == 0
            && self.untracked_metadata_reads == 0
            && self.host_loop_iterations == 0
            && self.per_sample_host_launches == 0
    }
}

/// Device-resident result of a GPU-resident MC run. Counts stay on device; the
/// caller decides whether/when to download them (after the measured region).
pub struct McResidentResult {
    pub query_counts: TrackedCudaSlice<u32>,
    pub evidence_count: TrackedCudaSlice<u32>,
    /// Per-world fixpoint iteration count (device-resident).
    pub iter_trace: TrackedCudaSlice<u32>,
    pub total_samples: usize,
    pub seed: u64,
    pub confidence: f64,
    pub sampling_method: McSamplingMethod,
    /// Number of query atoms (== `query_counts.len()`).
    pub num_queries: usize,
    pub no_host: McNoHostStats,
}

/// Compiled, device-uploadable plan for the resident engine.
#[derive(Debug, Clone)]
pub struct ResidentPlan {
    pub universe_size: u32,
    pub domain_size: u32,
    pub max_iters: u32,
    edb_slots: Vec<u32>,
    pf_slot: Vec<u32>,
    pf_var: Vec<u32>,
    rule_data: Vec<u32>,
    num_rules: u32,
    q_slot: Vec<u32>,
    ev_slot: Vec<u32>,
    ev_expected: Vec<u8>,
    ad_data: Vec<u32>,
    num_ads: u32,
    pub num_vars: usize,
    bernoulli_probs: Vec<f32>,
}

struct PredInfo {
    arity: usize,
    base: u32,
}

struct Universe {
    domain: BTreeMap<ConstKey, u32>,
    preds: BTreeMap<String, PredInfo>,
    domain_size: u32,
}

impl Universe {
    fn stride0(&self, arity: usize) -> u32 {
        if arity >= 2 {
            self.domain_size
        } else {
            1
        }
    }

    /// Slot id of a ground atom (predicate + constant args).
    fn ground_slot(&self, atom: &GroundAtom) -> std::result::Result<u32, ResidentRejection> {
        let info = self.preds.get(&atom.predicate).ok_or_else(|| {
            ResidentRejection::err(
                ResidentRejectKind::InconsistentArity,
                atom.predicate.clone(),
                "ground atom references unknown predicate",
            )
        })?;
        if atom.args.len() != info.arity {
            return Err(ResidentRejection::err(
                ResidentRejectKind::InconsistentArity,
                atom.predicate.clone(),
                format!("expected arity {} got {}", info.arity, atom.args.len()),
            ));
        }
        let mut slot = info.base;
        let stride0 = self.stride0(info.arity);
        for (i, v) in atom.args.iter().enumerate() {
            let key = ConstKey::from_value(v);
            let idx = *self.domain.get(&key).ok_or_else(|| {
                ResidentRejection::err(
                    ResidentRejectKind::UnboundedTerm,
                    format!("{:?}", v),
                    "ground constant absent from bounded domain",
                )
            })?;
            slot += idx * if i == 0 { stride0 } else { 1 };
        }
        Ok(slot)
    }
}

/// Compile an [`McProgram`] into a resident-engine plan, or return a typed
/// fail-closed rejection for anything outside the supported fragment.
pub fn compile_resident_plan(
    mc: &McProgram,
) -> std::result::Result<ResidentPlan, ResidentRejection> {
    let program = &mc.program;

    // --- 1. Gather predicate arities (consistency-checked). ---
    let mut arities: BTreeMap<String, usize> = BTreeMap::new();
    let mut note_pred = |pred: &str, arity: usize| -> std::result::Result<(), ResidentRejection> {
        if arity > MAX_ARITY {
            return Err(ResidentRejection::err(
                ResidentRejectKind::ArityTooHigh,
                pred.to_string(),
                format!("arity {} exceeds max {}", arity, MAX_ARITY),
            ));
        }
        match arities.get(pred) {
            Some(&existing) if existing != arity => Err(ResidentRejection::err(
                ResidentRejectKind::InconsistentArity,
                pred.to_string(),
                format!("arity {} vs {}", existing, arity),
            )),
            _ => {
                arities.insert(pred.to_string(), arity);
                Ok(())
            }
        }
    };

    // --- 2. Collect constants for the bounded domain. ---
    let mut domain: BTreeMap<ConstKey, u32> = BTreeMap::new();
    let mut note_const = |key: ConstKey, domain: &mut BTreeMap<ConstKey, u32>| {
        let next = domain.len() as u32;
        domain.entry(key).or_insert(next);
    };

    // Deterministic facts.
    for fact in program.facts() {
        note_pred(&fact.head.predicate, fact.head.terms.len())?;
        for t in &fact.head.terms {
            match ConstKey::from_term(t)? {
                TermClass::Const(k) => note_const(k, &mut domain),
                TermClass::Var(_) => {
                    return Err(ResidentRejection::err(
                        ResidentRejectKind::UnboundedTerm,
                        fact.head.predicate.clone(),
                        "fact head contains a variable",
                    ))
                }
            }
        }
    }
    // Prob facts (ground).
    for pf in &mc.prob_facts {
        note_pred(&pf.atom.predicate, pf.atom.args.len())?;
        for v in &pf.atom.args {
            note_const(ConstKey::from_value(v), &mut domain);
        }
    }
    // Queries + evidence (ground).
    for q in &mc.queries {
        note_pred(&q.predicate, q.args.len())?;
        for v in &q.args {
            note_const(ConstKey::from_value(v), &mut domain);
        }
    }
    for (e, _) in &mc.evidence {
        note_pred(&e.predicate, e.args.len())?;
        for v in &e.args {
            note_const(ConstKey::from_value(v), &mut domain);
        }
    }
    // Annotated-disjunction choice atoms (ground).
    for ad in &mc.annotated_disjunctions {
        for atom in &ad.choices {
            note_pred(&atom.predicate, atom.args.len())?;
            for v in &atom.args {
                note_const(ConstKey::from_value(v), &mut domain);
            }
        }
    }
    // Rules: collect predicates + constants (variables ranged over the domain).
    for rule in &program.rules {
        if rule.is_fact() {
            continue;
        }
        note_pred(&rule.head.predicate, rule.head.terms.len())?;
        collect_atom_consts(&rule.head, &mut domain, &mut note_const)?;
        if rule.body.len() > MAX_BODY {
            return Err(ResidentRejection::err(
                ResidentRejectKind::BodyTooLong,
                rule.head.predicate.clone(),
                format!("body length {} exceeds max {}", rule.body.len(), MAX_BODY),
            ));
        }
        for lit in &rule.body {
            let atom = classify_body_literal(lit, &rule.head.predicate)?;
            note_pred(&atom.predicate, atom.terms.len())?;
            collect_atom_consts(atom, &mut domain, &mut note_const)?;
        }
    }

    if domain.len() > MAX_DOMAIN {
        return Err(ResidentRejection::err(
            ResidentRejectKind::DomainTooLarge,
            format!("{} constants", domain.len()),
            format!("domain exceeds max {}", MAX_DOMAIN),
        ));
    }
    let domain_size = domain.len() as u32;

    // --- 3. Assign predicate slot blocks (deterministic order). ---
    let mut preds: BTreeMap<String, PredInfo> = BTreeMap::new();
    let mut base: u64 = 0;
    for (pred, &arity) in &arities {
        let slot_count: u64 = if arity == 0 {
            1
        } else {
            (domain_size as u64).pow(arity as u32)
        };
        preds.insert(
            pred.clone(),
            PredInfo {
                arity,
                base: base as u32,
            },
        );
        base += slot_count;
        if base > MAX_UNIVERSE as u64 {
            return Err(ResidentRejection::err(
                ResidentRejectKind::UniverseTooLarge,
                format!("{} slots", base),
                format!("universe exceeds max {}", MAX_UNIVERSE),
            ));
        }
    }
    let universe_size = base as u32;

    let universe = Universe {
        domain,
        preds,
        domain_size,
    };

    // --- 4. Lower EDB facts, prob facts, queries, evidence to slots. ---
    let mut edb_slots = Vec::new();
    for fact in program.facts() {
        let ga = ground_atom_from_atom(&fact.head)?;
        edb_slots.push(universe.ground_slot(&ga)?);
    }
    let mut pf_slot = Vec::new();
    let mut pf_var = Vec::new();
    for pf in &mc.prob_facts {
        pf_slot.push(universe.ground_slot(&pf.atom)?);
        pf_var.push(pf.var_idx as u32);
    }
    let mut q_slot = Vec::new();
    for q in &mc.queries {
        q_slot.push(universe.ground_slot(q)?);
    }
    let mut ev_slot = Vec::new();
    let mut ev_expected = Vec::new();
    for (e, v) in &mc.evidence {
        ev_slot.push(universe.ground_slot(e)?);
        ev_expected.push(if *v { 1u8 } else { 0u8 });
    }

    // --- 5. Lower rules to device records. ---
    let mut rule_data = Vec::new();
    let mut num_rules = 0u32;
    for rule in &program.rules {
        if rule.is_fact() {
            continue;
        }
        let rec = lower_rule(rule, &universe)?;
        rule_data.extend_from_slice(&rec);
        num_rules += 1;
    }

    // --- 6. Lower annotated disjunctions to device records. ---
    // Each record: [n_choices, n_dvars, dvar_0.., slot_0..]. The kernel walks the
    // conditional-Bernoulli chain (first firing decision wins; residual = last
    // choice or "none").
    let mut ad_data: Vec<u32> = Vec::new();
    let mut num_ads = 0u32;
    for ad in &mc.annotated_disjunctions {
        let n_choices = ad.choices.len() as u32;
        let n_dvars = ad.decision_vars.len() as u32;
        ad_data.push(n_choices);
        ad_data.push(n_dvars);
        for &dv in &ad.decision_vars {
            ad_data.push(u32::try_from(dv).map_err(|_| {
                ResidentRejection::err(
                    ResidentRejectKind::UnboundedTerm,
                    "decision_var",
                    "AD decision var index exceeds u32",
                )
            })?);
        }
        for atom in &ad.choices {
            ad_data.push(universe.ground_slot(atom)?);
        }
        num_ads += 1;
    }

    // Iteration cap: at most `universe_size` distinct atoms can be derived, so a
    // monotone fixpoint must converge within that many passes. +1 for the final
    // no-change confirmation pass.
    let max_iters = universe_size.saturating_add(1).max(1);

    Ok(ResidentPlan {
        universe_size,
        domain_size,
        max_iters,
        edb_slots,
        pf_slot,
        pf_var,
        rule_data,
        num_rules,
        q_slot,
        ev_slot,
        ev_expected,
        ad_data,
        num_ads,
        num_vars: mc.bernoulli_probs.len(),
        bernoulli_probs: mc.bernoulli_probs.clone(),
    })
}

fn collect_atom_consts<F: FnMut(ConstKey, &mut BTreeMap<ConstKey, u32>)>(
    atom: &Atom,
    domain: &mut BTreeMap<ConstKey, u32>,
    note_const: &mut F,
) -> std::result::Result<(), ResidentRejection> {
    if atom.terms.len() > MAX_ARITY {
        return Err(ResidentRejection::err(
            ResidentRejectKind::ArityTooHigh,
            atom.predicate.clone(),
            format!("arity {} exceeds max {}", atom.terms.len(), MAX_ARITY),
        ));
    }
    for t in &atom.terms {
        if let TermClass::Const(k) = ConstKey::from_term(t)? {
            note_const(k, domain);
        }
    }
    Ok(())
}

/// Classify a body literal: only positive relational atoms are supported.
fn classify_body_literal<'a>(
    lit: &'a BodyLiteral,
    rule_ctx: &str,
) -> std::result::Result<&'a Atom, ResidentRejection> {
    match lit {
        BodyLiteral::Positive(a) => Ok(a),
        BodyLiteral::Negated(a) => Err(ResidentRejection::err(
            ResidentRejectKind::Negation,
            a.predicate.clone(),
            format!("negated literal in rule for `{}`", rule_ctx),
        )),
        BodyLiteral::Epistemic(l) => Err(ResidentRejection::err(
            ResidentRejectKind::EpistemicLiteral,
            l.atom.predicate.clone(),
            format!("epistemic literal in rule for `{}`", rule_ctx),
        )),
        BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {
            Err(ResidentRejection::err(
                ResidentRejectKind::NonRelationalLiteral,
                "comparison/is/univ",
                format!("non-relational literal in rule for `{}`", rule_ctx),
            ))
        }
    }
}

/// Lower one rule to its fixed-width device record.
fn lower_rule(
    rule: &xlog_logic::ast::Rule,
    universe: &Universe,
) -> std::result::Result<Vec<u32>, ResidentRejection> {
    // Assign variable ids by first occurrence across head + body.
    let mut var_ids: BTreeMap<String, u32> = BTreeMap::new();
    let assign_var = |name: &str,
                      var_ids: &mut BTreeMap<String, u32>|
     -> std::result::Result<u32, ResidentRejection> {
        if let Some(&id) = var_ids.get(name) {
            return Ok(id);
        }
        let id = var_ids.len() as u32;
        if id as usize >= MAX_VARS {
            return Err(ResidentRejection::err(
                ResidentRejectKind::TooManyVars,
                rule.head.predicate.clone(),
                format!("more than {} distinct variables", MAX_VARS),
            ));
        }
        var_ids.insert(name.to_string(), id);
        Ok(id)
    };

    // First pass: assign var ids deterministically (head then body).
    let body_atoms: Vec<&Atom> = {
        let mut v = Vec::new();
        for lit in &rule.body {
            v.push(classify_body_literal(lit, &rule.head.predicate)?);
        }
        v
    };
    for t in &rule.head.terms {
        if let TermClass::Var(name) = ConstKey::from_term(t)? {
            assign_var(&name, &mut var_ids)?;
        }
    }
    for atom in &body_atoms {
        for t in &atom.terms {
            if let TermClass::Var(name) = ConstKey::from_term(t)? {
                assign_var(&name, &mut var_ids)?;
            }
        }
    }
    let n_vars = var_ids.len() as u32;

    let encode_atom = |atom: &Atom| -> std::result::Result<[u32; ATOM_REC], ResidentRejection> {
        let info = universe.preds.get(&atom.predicate).ok_or_else(|| {
            ResidentRejection::err(
                ResidentRejectKind::InconsistentArity,
                atom.predicate.clone(),
                "rule atom references unknown predicate",
            )
        })?;
        let arity = info.arity as u32;
        let mut rec = [0u32; ATOM_REC];
        rec[0] = info.base;
        rec[1] = arity;
        rec[4] = universe.stride0(info.arity);
        for (i, t) in atom.terms.iter().enumerate() {
            let spec = match ConstKey::from_term(t)? {
                TermClass::Var(name) => *var_ids.get(&name).expect("var assigned above"),
                TermClass::Const(k) => {
                    let idx = *universe.domain.get(&k).ok_or_else(|| {
                        ResidentRejection::err(
                            ResidentRejectKind::UnboundedTerm,
                            format!("{:?}", k),
                            "rule constant absent from bounded domain",
                        )
                    })?;
                    CONST_FLAG | idx
                }
            };
            rec[2 + i] = spec;
        }
        Ok(rec)
    };

    let mut rec = vec![0u32; RULE_REC];
    rec[0] = body_atoms.len() as u32;
    rec[1] = n_vars;
    rec[2] = universe.domain_size;
    let head_rec = encode_atom(&rule.head)?;
    rec[3..3 + ATOM_REC].copy_from_slice(&head_rec);
    for (bi, atom) in body_atoms.iter().enumerate() {
        let a = encode_atom(atom)?;
        let off = 3 + ATOM_REC + bi * ATOM_REC;
        rec[off..off + ATOM_REC].copy_from_slice(&a);
    }
    Ok(rec)
}

fn ground_atom_from_atom(atom: &Atom) -> std::result::Result<GroundAtom, ResidentRejection> {
    let mut args = Vec::with_capacity(atom.terms.len());
    for t in &atom.terms {
        let v = match t {
            Term::Integer(i) => Value::I64(*i),
            Term::Symbol(s) => Value::Symbol(*s),
            Term::String(s) => Value::String(s.clone()),
            Term::Float(f) => Value::F64(f.to_bits()),
            other => {
                return Err(ResidentRejection::err(
                    ResidentRejectKind::UnboundedTerm,
                    format!("{:?}", other),
                    "fact term must be a ground constant",
                ))
            }
        };
        args.push(v);
    }
    Ok(GroundAtom {
        predicate: atom.predicate.clone(),
        args,
    })
}

impl McProgram {
    /// Evaluate this program with the GPU-resident engine. Returns device-resident
    /// counts plus no-host instrumentation for the measured region. Fails closed
    /// (typed [`ResidentRejection`] wrapped into [`XlogError`]) for programs
    /// outside the supported fragment.
    pub fn evaluate_resident_with_provider(
        &self,
        cfg: McEvalConfig,
        provider: Arc<CudaKernelProvider>,
    ) -> Result<McResidentResult> {
        cfg.validate()?;
        let plan = compile_resident_plan(self).map_err(ResidentRejection::into_error)?;
        run_resident(&plan, &cfg, self, provider)
    }

    /// Convenience: evaluate with a fresh provider.
    pub fn evaluate_resident(&self, cfg: McEvalConfig) -> Result<McResidentResult> {
        let provider = Arc::new(self.provider()?);
        self.evaluate_resident_with_provider(cfg, provider)
    }
}

fn run_resident(
    plan: &ResidentPlan,
    cfg: &McEvalConfig,
    mc: &McProgram,
    provider: Arc<CudaKernelProvider>,
) -> Result<McResidentResult> {
    let (method, forcing) = mc.resolve_sampling_method(cfg.sampling_method)?;
    let num_worlds = u32::try_from(cfg.samples)
        .map_err(|_| XlogError::Execution("MC samples exceed u32::MAX".to_string()))?;
    let num_vars = plan.num_vars;

    // ---------------- Static setup (BEFORE the measured region) ----------------
    // All device allocations, the seeded sample matrix, force arrays, and every
    // plan array are uploaded here. Allocation syncs; nothing below the measured
    // marker may transfer or read back.
    let dev = provider.device();

    // Sample matrix (device-resident), forced for clamped evidence.
    let mut d_force_mask = provider.memory().alloc::<u8>(num_vars.max(1))?;
    let mut d_forced_value = provider.memory().alloc::<u8>(num_vars.max(1))?;
    if method == McSamplingMethod::EvidenceClamping && num_vars > 0 {
        provider.htod_sync_copy_into_tracked(&forcing.force_mask, &mut d_force_mask)?;
        provider.htod_sync_copy_into_tracked(&forcing.forced_value, &mut d_forced_value)?;
    } else {
        dev.inner()
            .memset_zeros(&mut d_force_mask)
            .map_err(|e| XlogError::Kernel(format!("zero force_mask: {e}")))?;
        dev.inner()
            .memset_zeros(&mut d_forced_value)
            .map_err(|e| XlogError::Kernel(format!("zero forced_value: {e}")))?;
    }
    let samples_device = if num_vars == 0 || cfg.samples == 0 {
        provider.memory().alloc::<u8>(1)?
    } else {
        provider.sample_bernoulli_matrix_device(
            &plan.bernoulli_probs,
            cfg.samples,
            cfg.seed,
            &d_force_mask.slice(..),
            &d_forced_value.slice(..),
        )?
    };

    // Working relation buffer [num_worlds * 2 * U], pre-zeroed. The kernel always
    // strides worlds by 2*U (double buffer A|B per world) regardless of rule
    // count, so the host must allocate 2*U per world unconditionally.
    let u = plan.universe_size.max(1) as usize;
    let rel_len = (num_worlds as usize)
        .saturating_mul(u)
        .saturating_mul(2)
        .max(1);
    let mut d_rel = provider.memory().alloc::<u8>(rel_len)?;
    dev.inner()
        .memset_zeros(&mut d_rel)
        .map_err(|e| XlogError::Kernel(format!("zero rel: {e}")))?;

    // Pack every read-only plan array into one contiguous `meta` u32 buffer and
    // record offsets in the `cfg` header. This keeps the kernel under the launch
    // arg-arity limit (7 params) and uses one HtoD per buffer.
    let q_count = plan.q_slot.len();
    let ev_expected_u32: Vec<u32> = plan.ev_expected.iter().map(|&b| b as u32).collect();

    let mut meta: Vec<u32> = Vec::new();
    let push_meta = |data: &[u32], meta: &mut Vec<u32>| -> u32 {
        let off = meta.len() as u32;
        meta.extend_from_slice(data);
        off
    };
    let edb_off = push_meta(&plan.edb_slots, &mut meta);
    let pf_slot_off = push_meta(&plan.pf_slot, &mut meta);
    let pf_var_off = push_meta(&plan.pf_var, &mut meta);
    let rules_off = push_meta(&plan.rule_data, &mut meta);
    let q_off = push_meta(&plan.q_slot, &mut meta);
    let ev_slot_off = push_meta(&plan.ev_slot, &mut meta);
    let ev_exp_off = push_meta(&ev_expected_u32, &mut meta);
    let ad_off = push_meta(&plan.ad_data, &mut meta);

    // cfg header, indices mirror the `CFG_*` defines in mc_resident.cu.
    let cfg_host: [u32; 18] = [
        num_worlds,
        plan.universe_size,
        num_vars as u32,
        plan.max_iters,
        edb_off,
        plan.edb_slots.len() as u32,
        pf_slot_off,
        pf_var_off,
        plan.pf_slot.len() as u32,
        rules_off,
        plan.num_rules,
        q_off,
        q_count as u32,
        ev_slot_off,
        ev_exp_off,
        plan.ev_slot.len() as u32,
        ad_off,
        plan.num_ads,
    ];

    let mut d_cfg = provider.memory().alloc::<u32>(cfg_host.len())?;
    provider.htod_sync_copy_into_tracked(&cfg_host, &mut d_cfg)?;
    let mut d_meta = provider.memory().alloc::<u32>(meta.len().max(1))?;
    if !meta.is_empty() {
        provider.htod_sync_copy_into_tracked(&meta, &mut d_meta)?;
    }

    let mut d_query_counts = provider.memory().alloc::<u32>(q_count.max(1))?;
    dev.inner()
        .memset_zeros(&mut d_query_counts)
        .map_err(|e| XlogError::Kernel(format!("zero query_counts: {e}")))?;
    let mut d_evidence_count = provider.memory().alloc::<u32>(1)?;
    dev.inner()
        .memset_zeros(&mut d_evidence_count)
        .map_err(|e| XlogError::Kernel(format!("zero evidence_count: {e}")))?;
    let mut d_iter_trace = provider.memory().alloc::<u32>(num_worlds.max(1) as usize)?;
    dev.inner()
        .memset_zeros(&mut d_iter_trace)
        .map_err(|e| XlogError::Kernel(format!("zero iter_trace: {e}")))?;

    let engine_fn = dev
        .inner()
        .get_func(MC_RESIDENT_MODULE, mc_resident_kernels::MC_RESIDENT_ENGINE)
        .ok_or_else(|| XlogError::Kernel("mc_resident_engine kernel not found".to_string()))?;

    // Ensure all setup work is complete before we start measuring.
    dev.synchronize()?;

    // ---------------- Measured region (ZERO host interaction) ----------------
    let pre = provider.host_transfer_stats();
    let pre_untracked = provider.untracked_metadata_dtoh_count();
    let mut engine_launches = 0u64;

    let block_dim = 128u32;
    let grid_dim = num_worlds.max(1);
    let launch_cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: argument list matches the `mc_resident_engine` PTX signature; every
    // device buffer was allocated with sufficient length above.
    unsafe {
        engine_fn
            .launch(
                launch_cfg,
                (
                    &d_cfg,
                    &d_meta,
                    &mut d_rel,
                    &samples_device,
                    &mut d_query_counts,
                    &mut d_evidence_count,
                    &mut d_iter_trace,
                ),
            )
            .map_err(|e| XlogError::Kernel(format!("mc_resident_engine launch failed: {e}")))?;
    }
    engine_launches += 1;
    dev.synchronize()?;

    let post = provider.host_transfer_stats();
    let post_untracked = provider.untracked_metadata_dtoh_count();
    // ---------------- End measured region ----------------

    let no_host = McNoHostStats {
        tracked_htod_calls: post.htod_calls.saturating_sub(pre.htod_calls),
        tracked_dtoh_calls: post.dtoh_calls.saturating_sub(pre.dtoh_calls),
        untracked_metadata_reads: post_untracked.saturating_sub(pre_untracked),
        engine_launches,
        host_loop_iterations: 0,
        per_sample_host_launches: 0,
    };

    Ok(McResidentResult {
        query_counts: d_query_counts,
        evidence_count: d_evidence_count,
        iter_trace: d_iter_trace,
        total_samples: cfg.samples,
        seed: cfg.seed,
        confidence: cfg.confidence,
        sampling_method: method,
        num_queries: q_count,
        no_host,
    })
}
