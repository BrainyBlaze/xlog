//! Well-Founded Semantics for non-monotone probabilistic programs.
//!
//! WFS handles programs with cycles through negation using three-valued logic:
//! - True: definitely derivable
//! - False: definitely not derivable
//! - Undefined: in cycle, neither provable

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

/// Evaluate a non-monotone SCC using Well-Founded Semantics
///
/// Currently returns an error as WFS is not yet fully implemented.
/// Non-monotone programs should use the MC engine instead.
pub fn evaluate_wfs_scc(
    _scc_predicates: &[String],
    _pir: &mut PirGraph,
) -> Result<WfsResult> {
    // TODO: Implement alternating fixed-point algorithm
    Err(XlogError::Compilation(
        "Well-Founded Semantics not yet implemented for non-monotone SCCs. \
         Use prob_engine=mc for programs with cycles through negation.".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::LeafId;

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
