//! Evidence forcing for Monte Carlo sampling.

use xlog_core::{Result, XlogError};

use super::{McProgram, McSamplingMethod};

/// Why evidence may or may not be forceable to root Bernoulli variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForceabilityReason {
    AllForceable,
    ContainsDerivedEvidence,
    ContainsNegativeAdHeadEvidence,
    NoEvidence,
}

/// Compiled evidence forcing for the MC sampler.
#[derive(Debug, Clone)]
pub struct EvidenceForcing {
    pub force_mask: Vec<u8>,
    pub forced_value: Vec<u8>,
    pub forceable: bool,
    pub reason: ForceabilityReason,
}

impl McProgram {
    /// Resolve the sampling method from config + evidence forceability.
    pub(super) fn resolve_sampling_method(
        &self,
        requested: Option<McSamplingMethod>,
    ) -> Result<(McSamplingMethod, EvidenceForcing)> {
        let forcing = self.compile_evidence_forcing()?;
        let method = match requested {
            Some(McSamplingMethod::EvidenceClamping) => {
                if !forcing.forceable {
                    return Err(XlogError::Execution(format!(
                        "Cannot use EvidenceClamping: {:?}",
                        forcing.reason
                    )));
                }
                McSamplingMethod::EvidenceClamping
            }
            Some(McSamplingMethod::Rejection) => McSamplingMethod::Rejection,
            None => {
                if forcing.forceable {
                    McSamplingMethod::EvidenceClamping
                } else {
                    McSamplingMethod::Rejection
                }
            }
        };
        Ok((method, forcing))
    }

    pub fn compile_evidence_forcing(&self) -> Result<EvidenceForcing> {
        let num_vars = self.bernoulli_probs.len();
        let mut force_mask = vec![0u8; num_vars];
        let mut forced_value = vec![0u8; num_vars];

        if self.evidence.is_empty() {
            return Ok(EvidenceForcing {
                force_mask,
                forced_value,
                forceable: false,
                reason: ForceabilityReason::NoEvidence,
            });
        }

        for (atom, expected) in &self.evidence {
            // Try to match against prob fact specs
            if let Some(spec) = self.prob_facts.iter().find(|s| &s.atom == atom) {
                force_mask[spec.var_idx] = 1;
                forced_value[spec.var_idx] = if *expected { 1 } else { 0 };
                continue;
            }

            // Try to match against AD choice atoms (positive evidence only)
            let mut found_ad = false;
            for ad in &self.annotated_disjunctions {
                if let Some(choice_idx) = ad.choices.iter().position(|c| c == atom) {
                    if !*expected {
                        // evidence(ad_head, false) — not forceable in v0.5.1
                        return Ok(EvidenceForcing {
                            force_mask: vec![0u8; num_vars],
                            forced_value: vec![0u8; num_vars],
                            forceable: false,
                            reason: ForceabilityReason::ContainsNegativeAdHeadEvidence,
                        });
                    }

                    let num_decision_vars = ad.decision_vars.len();
                    if choice_idx < num_decision_vars {
                        // Force d_i = 0 for all i < choice_idx, d_{choice_idx} = 1
                        for i in 0..choice_idx {
                            force_mask[ad.decision_vars[i]] = 1;
                            forced_value[ad.decision_vars[i]] = 0;
                        }
                        force_mask[ad.decision_vars[choice_idx]] = 1;
                        forced_value[ad.decision_vars[choice_idx]] = 1;
                    } else {
                        // Last head (no none branch): force all decision vars to 0
                        for &dv in &ad.decision_vars {
                            force_mask[dv] = 1;
                            forced_value[dv] = 0;
                        }
                    }
                    found_ad = true;
                    break;
                }
            }
            if found_ad {
                continue;
            }

            // Evidence atom not found in prob facts or AD choices → derived/deterministic
            return Ok(EvidenceForcing {
                force_mask: vec![0u8; num_vars],
                forced_value: vec![0u8; num_vars],
                forceable: false,
                reason: ForceabilityReason::ContainsDerivedEvidence,
            });
        }

        Ok(EvidenceForcing {
            force_mask,
            forced_value,
            forceable: true,
            reason: ForceabilityReason::AllForceable,
        })
    }
}
