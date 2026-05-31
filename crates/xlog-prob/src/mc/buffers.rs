//! Sampling-plan compilation for the GPU-resident MC engine.
//!
//! After the move to the GPU-resident megakernel engine
//! (`crates/xlog-prob/src/mc/resident.rs`), the only host-side buffer work that
//! remains is compiling probabilistic facts and annotated disjunctions into the
//! flat Bernoulli sampling plan (`bernoulli_probs` + `ProbFactSpec` + `AdSpec`).
//! All the legacy per-sample relation building, reset planning, dense-table
//! upload, and schema/type inference helpers were removed with the
//! host-orchestrated loop.

use std::collections::HashSet;

use xlog_core::{Result, XlogError};
use xlog_logic::ast::{BodyLiteral, ProbFact, Program, Term};

use crate::provenance::{atom_key_from_ground_atom, validate_prob, GroundAtom};

use super::{AdSpec, ProbFactSpec};

pub(super) fn extend_prob_facts_with_coin(
    program: &Program,
    prob_facts: &mut Vec<ProbFact>,
) -> Result<()> {
    let mut seen: HashSet<GroundAtom> = HashSet::new();
    for pf in prob_facts.iter() {
        seen.insert(atom_key_from_ground_atom(&pf.atom)?);
    }

    for rule in &program.rules {
        for lit in &rule.body {
            let BodyLiteral::Positive(atom) = lit else {
                continue;
            };
            if atom.predicate != "coin" || atom.terms.len() != 1 {
                continue;
            }
            let Term::Float(prob) = atom.terms[0] else {
                continue;
            };
            let key = atom_key_from_ground_atom(atom)?;
            if seen.insert(key) {
                prob_facts.push(ProbFact {
                    prob,
                    atom: atom.clone(),
                });
            }
        }
    }

    Ok(())
}

pub(super) fn compile_sampling_plan(
    prob_facts: &[ProbFact],
    annotated_disjunctions: &[xlog_logic::ast::AnnotatedDisjunction],
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
            for &p_i in probs_full.iter().take(m - 1) {
                let cond_true = if remaining <= 0.0 {
                    0.0
                } else {
                    p_i / remaining
                };
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
