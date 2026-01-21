//! Well-Founded Semantics for non-monotone probabilistic programs.
//!
//! WFS handles programs with cycles through negation using three-valued logic:
//! - True: definitely derivable
//! - False: definitely not derivable
//! - Undefined: in cycle, neither provable
//!
//! # Algorithm Overview
//!
//! The WFS alternating fixed-point algorithm alternates between:
//! 1. **Unfounded set computation**: Find atoms that cannot be supported
//! 2. **Consequence derivation**: Derive atoms that must be true
//!
//! This continues until a fixed point is reached.
//!
//! # Gradient Treatment
//!
//! - True atoms: Normal probability and gradient computation
//! - False atoms: Probability = 0, gradient = 0
//! - Undefined atoms: Probability = 0, gradient = 0 (conservative)
//!
//! This matches ProbLog's behavior for non-stratified programs.

use std::collections::{HashMap, HashSet};
use xlog_core::{Result, XlogError};
use crate::pir::{PirGraph, PirNodeId};
use crate::provenance::Value;

/// Ground atom representation for WFS
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WfsAtom {
    pub predicate: String,
    pub args: Vec<Value>,
}

/// Three-valued truth value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthValue {
    True,
    False,
    Undefined,
}

/// Result of WFS evaluation for an SCC
#[derive(Debug, Clone)]
pub struct WfsResult {
    /// Atoms known to be true with their provenance
    pub true_set: HashMap<WfsAtom, PirNodeId>,
    /// Atoms known to be false
    pub false_set: HashSet<WfsAtom>,
    // Atoms not in either set are undefined
}

impl WfsResult {
    pub fn new() -> Self {
        Self {
            true_set: HashMap::new(),
            false_set: HashSet::new(),
        }
    }

    pub fn truth_value(&self, atom: &WfsAtom) -> TruthValue {
        if self.true_set.contains_key(atom) {
            TruthValue::True
        } else if self.false_set.contains(atom) {
            TruthValue::False
        } else {
            TruthValue::Undefined
        }
    }
}

impl Default for WfsResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for WFS evaluation
#[derive(Debug, Clone)]
pub struct WfsConfig {
    /// Maximum iterations before giving up
    pub max_iterations: usize,
}

impl Default for WfsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
        }
    }
}

/// Compute the unfounded set: atoms that cannot be supported
///
/// An atom is unfounded if every rule that could derive it either:
/// - Has a body literal that is false, or
/// - Depends on atoms in the unfounded set itself (positive cycle)
///
/// This is a simplified placeholder - full implementation requires rule access.
fn compute_unfounded_set(
    _true_set: &HashMap<WfsAtom, PirNodeId>,
    _false_set: &HashSet<WfsAtom>,
    _scc_atoms: &HashSet<WfsAtom>,
) -> HashSet<WfsAtom> {
    // TODO: Implement greatest fixed-point computation for unfounded sets
    // For now, return empty set (conservative - nothing is unfounded)
    HashSet::new()
}

/// Derive new true atoms using the immediate consequence operator
///
/// An atom becomes true if there exists a rule where:
/// - All positive body literals are in true_set
/// - All negative body literals are in false_set
///
/// This is a simplified placeholder - full implementation requires rule access.
fn derive_consequences(
    _true_set: &HashMap<WfsAtom, PirNodeId>,
    _false_set: &HashSet<WfsAtom>,
    _pir: &mut PirGraph,
) -> HashMap<WfsAtom, PirNodeId> {
    // TODO: Implement immediate consequence operator
    // For now, return empty map (no new consequences)
    HashMap::new()
}

/// Evaluate a non-monotone SCC using Well-Founded Semantics
///
/// The alternating fixed-point algorithm:
/// 1. Initialize: all atoms undefined
/// 2. Loop until fixed point:
///    a. Compute unfounded set (atoms that cannot be true)
///    b. Add unfounded atoms to false_set
///    c. Derive consequences (atoms that must be true given false_set)
///    d. Add derived atoms to true_set
/// 3. Remaining atoms stay undefined
///
/// # Current Status
///
/// WFS is not yet fully implemented. Non-monotone programs should use
/// the MC engine (`prob_engine=mc`) which handles them via sampling.
pub fn evaluate_wfs_scc(
    scc_predicates: &[String],
    pir: &mut PirGraph,
) -> Result<WfsResult> {
    evaluate_wfs_scc_with_config(scc_predicates, pir, &WfsConfig::default())
}

/// Evaluate WFS with custom configuration
pub fn evaluate_wfs_scc_with_config(
    scc_predicates: &[String],
    _pir: &mut PirGraph,
    config: &WfsConfig,
) -> Result<WfsResult> {
    let true_set: HashMap<WfsAtom, PirNodeId> = HashMap::new();
    let mut false_set: HashSet<WfsAtom> = HashSet::new();

    // Collect all ground atoms in SCC (simplified - no actual grounding)
    let scc_atoms: HashSet<WfsAtom> = scc_predicates
        .iter()
        .map(|p| WfsAtom { predicate: p.clone(), args: vec![] })
        .collect();

    for iteration in 0..config.max_iterations {
        // Step 1: Compute unfounded set
        let unfounded = compute_unfounded_set(&true_set, &false_set, &scc_atoms);
        let new_false: Vec<_> = unfounded.into_iter()
            .filter(|a| !false_set.contains(a))
            .collect();

        // Step 2: Derive consequences
        // Note: This is where we'd need access to rules and proper grounding
        // For now, this is a placeholder
        let _new_true = derive_consequences(&true_set, &false_set, _pir);

        if new_false.is_empty() {
            // Fixed point reached
            break;
        }

        false_set.extend(new_false);

        if iteration == config.max_iterations - 1 {
            return Err(XlogError::Execution(format!(
                "WFS evaluation did not converge after {} iterations",
                config.max_iterations
            )));
        }
    }

    // For now, still return an error since the algorithm isn't fully connected
    // to the rule evaluation infrastructure
    Err(XlogError::Compilation(
        "Well-Founded Semantics not yet fully implemented for non-monotone SCCs. \
         Use prob_engine=mc for programs with cycles through negation.".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::LeafId;

    #[test]
    fn test_wfs_config_default() {
        let config = WfsConfig::default();
        assert_eq!(config.max_iterations, 1000);
    }

    #[test]
    fn test_wfs_result_default() {
        let result = WfsResult::default();
        assert!(result.true_set.is_empty());
        assert!(result.false_set.is_empty());
    }

    #[test]
    fn test_wfs_result_truth_value() {
        let mut result = WfsResult::new();
        let atom = WfsAtom {
            predicate: "p".to_string(),
            args: vec![],
        };

        // Initially undefined
        assert_eq!(result.truth_value(&atom), TruthValue::Undefined);

        // After adding to false_set
        result.false_set.insert(atom.clone());
        assert_eq!(result.truth_value(&atom), TruthValue::False);

        // After moving to true_set
        result.false_set.remove(&atom);
        // Create a PirNodeId via PirGraph
        let mut pir = PirGraph::new();
        let node_id = pir.lit(LeafId::new(0));
        result.true_set.insert(atom.clone(), node_id);
        assert_eq!(result.truth_value(&atom), TruthValue::True);
    }

    #[test]
    fn test_wfs_atom_equality() {
        let atom1 = WfsAtom {
            predicate: "p".to_string(),
            args: vec![Value::I64(1)],
        };
        let atom2 = WfsAtom {
            predicate: "p".to_string(),
            args: vec![Value::I64(1)],
        };
        let atom3 = WfsAtom {
            predicate: "p".to_string(),
            args: vec![Value::I64(2)],
        };

        assert_eq!(atom1, atom2);
        assert_ne!(atom1, atom3);
    }
}
