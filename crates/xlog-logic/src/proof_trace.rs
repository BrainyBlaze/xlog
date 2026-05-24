//! Differentiable proof trace records for proof-path training surfaces.

use std::collections::BTreeMap;

/// Input used to create one differentiable proof trace.
#[derive(Clone, Debug, PartialEq)]
pub struct ProofTraceSpec {
    /// Stable answer key, for example `root(case_1, primary_root)`.
    pub answer_key: String,
    /// Stable symbolic clause identifier.
    pub clause_id: String,
    /// Atoms supporting the proof path.
    pub support_atoms: Vec<String>,
    /// Initial symbolic clause weight.
    pub initial_weight: f64,
}

/// Exported differentiable proof trace.
#[derive(Clone, Debug, PartialEq)]
pub struct ProofTrace {
    /// Stable proof identifier derived from answer, clause, and support atoms.
    pub proof_id: u64,
    /// Stable answer key.
    pub answer_key: String,
    /// Symbolic clause identifier.
    pub clause_id: String,
    /// Atoms supporting the proof path.
    pub support_atoms: Vec<String>,
    /// Current symbolic clause weight.
    pub weight: f64,
    /// Accumulated gradient for the current training step.
    pub gradient: f64,
}

/// Collection of differentiable proof traces keyed by stable proof ID.
#[derive(Clone, Debug, Default)]
pub struct DifferentiableProofTraceMap {
    traces: BTreeMap<u64, ProofTrace>,
}

impl DifferentiableProofTraceMap {
    /// Create an empty proof trace map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a trace and return its stable proof ID.
    pub fn insert(&mut self, spec: ProofTraceSpec) -> u64 {
        let mut proof_id = stable_proof_id(&spec.answer_key, &spec.clause_id, &spec.support_atoms);
        while let Some(existing) = self.traces.get(&proof_id) {
            if existing.answer_key == spec.answer_key
                && existing.clause_id == spec.clause_id
                && existing.support_atoms == spec.support_atoms
            {
                return proof_id;
            }
            proof_id = proof_id.wrapping_add(1);
        }
        self.traces.insert(
            proof_id,
            ProofTrace {
                proof_id,
                answer_key: spec.answer_key,
                clause_id: spec.clause_id,
                support_atoms: spec.support_atoms,
                weight: spec.initial_weight,
                gradient: 0.0,
            },
        );
        proof_id
    }

    /// Get one exported trace by proof ID.
    pub fn trace(&self, proof_id: u64) -> Option<&ProofTrace> {
        self.traces.get(&proof_id)
    }

    /// Iterate over exported traces.
    pub fn traces(&self) -> impl Iterator<Item = &ProofTrace> {
        self.traces.values()
    }

    /// Accumulate binary logistic gradients grouped by answer key.
    pub fn accumulate_binary_logistic_gradients(&mut self, targets: &[(String, f64)]) -> f64 {
        for trace in self.traces.values_mut() {
            trace.gradient = 0.0;
        }

        let mut loss = 0.0;
        for (answer_key, target) in targets {
            let score: f64 = self
                .traces
                .values()
                .filter(|trace| &trace.answer_key == answer_key)
                .map(|trace| trace.weight)
                .sum();
            let prediction = sigmoid(score);
            let clamped = prediction.clamp(1e-12, 1.0 - 1e-12);
            loss += -target * clamped.ln() - (1.0 - target) * (1.0 - clamped).ln();
            let gradient = prediction - target;
            for trace in self
                .traces
                .values_mut()
                .filter(|trace| &trace.answer_key == answer_key)
            {
                trace.gradient += gradient;
            }
        }
        loss
    }

    /// Apply accumulated gradients to trace weights.
    pub fn apply_gradients(&mut self, learning_rate: f64) {
        for trace in self.traces.values_mut() {
            trace.weight -= learning_rate * trace.gradient;
        }
    }
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn stable_proof_id(answer_key: &str, clause_id: &str, support_atoms: &[String]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for part in std::iter::once(answer_key)
        .chain(std::iter::once(clause_id))
        .chain(support_atoms.iter().map(String::as_str))
    {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
