//! Well-Founded Semantics for non-monotone probabilistic programs.
//!
//! WFS handles programs with cycles through negation using three-valued logic:
//! - True: definitely derivable
//! - False: definitely not derivable
//! - Undefined: in cycle, neither provable nor refutable
//!
//! # Algorithm Overview
//!
//! The WFS alternating fixed-point algorithm works as follows:
//!
//! 1. **Initialize**: All atoms start as undefined
//! 2. **Loop until fixed point**:
//!    a. **Unfounded set computation**: Find atoms that cannot be supported
//!       - An atom is unfounded if every rule that derives it either:
//!         - Has a body literal that is known false, or
//!         - Depends positively on an unfounded atom
//!    b. **Mark unfounded atoms as false**
//!    c. **Consequence derivation**: Find atoms that must be true
//!       - An atom is a consequence if some rule has:
//!         - All positive body literals true
//!         - All negative body literals false
//!    d. **Mark consequences as true**
//! 3. **Remaining atoms stay undefined**
//!
//! # Gradient Treatment
//!
//! - True atoms: Normal probability and gradient computation
//! - False atoms: Probability = 0, gradient = 0
//! - Undefined atoms: Probability = 0, gradient = 0 (conservative)
//!
//! This matches ProbLog's behavior for non-stratified programs.
//!
//! # Integration
//!
//! WFS is invoked during provenance extraction when a non-monotone SCC is detected.
//! It receives ground rules (after variable substitution) and returns the well-founded
//! model with provenance formulas for true atoms.

use crate::pir::{PirGraph, PirNodeId};
use crate::provenance::Value;
use std::collections::{HashMap, HashSet};
use xlog_core::{Result, XlogError};

/// Ground atom representation for WFS
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WfsAtom {
    pub predicate: String,
    pub args: Vec<Value>,
}

impl WfsAtom {
    /// Create a new WFS atom
    pub fn new(predicate: impl Into<String>, args: Vec<Value>) -> Self {
        Self {
            predicate: predicate.into(),
            args,
        }
    }
}

impl std::fmt::Display for WfsAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.predicate)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match arg {
                Value::I64(v) => write!(f, "{}", v)?,
                Value::F64(v) => write!(f, "{}", f64::from_bits(*v))?,
                Value::Symbol(v) => write!(f, "sym{}", v)?,
                Value::String(v) => write!(f, "\"{}\"", v)?,
            }
        }
        write!(f, ")")
    }
}

/// Three-valued truth value for WFS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthValue {
    True,
    False,
    Undefined,
}

impl TruthValue {
    /// Check if this is a definite value (not undefined)
    pub fn is_defined(&self) -> bool {
        matches!(self, TruthValue::True | TruthValue::False)
    }
}

/// A ground literal in a rule body
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WfsLiteral {
    /// Positive literal: atom must be true
    Positive(WfsAtom),
    /// Negative literal: atom must be false
    Negative(WfsAtom),
}

impl WfsLiteral {
    /// Get the underlying atom
    pub fn atom(&self) -> &WfsAtom {
        match self {
            WfsLiteral::Positive(a) | WfsLiteral::Negative(a) => a,
        }
    }

    /// Check if this is a positive literal
    pub fn is_positive(&self) -> bool {
        matches!(self, WfsLiteral::Positive(_))
    }

    /// Check if this is a negative literal
    pub fn is_negative(&self) -> bool {
        matches!(self, WfsLiteral::Negative(_))
    }
}

/// A ground rule for WFS evaluation
///
/// A ground rule has no variables - all terms are concrete values.
/// Each ground rule also carries a provenance formula for probabilistic tracking.
#[derive(Debug, Clone)]
pub struct WfsRule {
    /// The head atom this rule derives
    pub head: WfsAtom,
    /// The body literals (all must be satisfied for rule to fire)
    pub body: Vec<WfsLiteral>,
    /// Provenance formula for this specific ground instance
    /// This is the AND of all non-SCC body literal provenances
    pub provenance: PirNodeId,
}

impl WfsRule {
    /// Create a new ground rule
    pub fn new(head: WfsAtom, body: Vec<WfsLiteral>, provenance: PirNodeId) -> Self {
        Self {
            head,
            body,
            provenance,
        }
    }

    /// Check if rule body is satisfied under current interpretation
    ///
    /// For external atoms (not in SCC):
    /// - Positive: must be in true_set (external facts provided)
    /// - Negative: if not in true_set, assumed false (closed-world)
    fn is_satisfied(
        &self,
        true_set: &HashSet<WfsAtom>,
        false_set: &HashSet<WfsAtom>,
        scc_atoms: &HashSet<WfsAtom>,
    ) -> Option<bool> {
        // Rule is satisfied if all body literals are satisfied
        // Returns None if any literal is undefined
        for lit in &self.body {
            match lit {
                WfsLiteral::Positive(atom) => {
                    if false_set.contains(atom) {
                        return Some(false);
                    }
                    if true_set.contains(atom) {
                        continue; // Satisfied
                    }
                    // Not true and not false
                    if scc_atoms.contains(atom) {
                        return None; // SCC atom is undefined
                    }
                    // External atom not in true_set - will never become true
                    return Some(false);
                }
                WfsLiteral::Negative(atom) => {
                    if true_set.contains(atom) {
                        return Some(false);
                    }
                    if false_set.contains(atom) {
                        continue; // Satisfied (not X where X is false)
                    }
                    // Not true and not false
                    if scc_atoms.contains(atom) {
                        return None; // SCC atom is undefined
                    }
                    // External atom not in true_set - closed world says it's false
                    // So "not external_atom" succeeds
                    continue;
                }
            }
        }
        Some(true)
    }

    /// Check if rule body is definitely unsatisfiable.
    ///
    /// This can be used for early pruning in more advanced WFS implementations.
    #[allow(dead_code)]
    pub fn is_definitely_unsatisfiable(
        &self,
        true_set: &HashSet<WfsAtom>,
        false_set: &HashSet<WfsAtom>,
    ) -> bool {
        for lit in &self.body {
            match lit {
                WfsLiteral::Positive(atom) => {
                    if false_set.contains(atom) {
                        return true;
                    }
                }
                WfsLiteral::Negative(atom) => {
                    if true_set.contains(atom) {
                        return true;
                    }
                }
            }
        }
        false
    }
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
    /// Create an empty WFS result
    pub fn new() -> Self {
        Self {
            true_set: HashMap::new(),
            false_set: HashSet::new(),
        }
    }

    /// Get the truth value of an atom
    pub fn truth_value(&self, atom: &WfsAtom) -> TruthValue {
        if self.true_set.contains_key(atom) {
            TruthValue::True
        } else if self.false_set.contains(atom) {
            TruthValue::False
        } else {
            TruthValue::Undefined
        }
    }

    /// Get the provenance for a true atom
    pub fn provenance(&self, atom: &WfsAtom) -> Option<PirNodeId> {
        self.true_set.get(atom).copied()
    }

    /// Check if an atom is undefined
    pub fn is_undefined(&self, atom: &WfsAtom) -> bool {
        !self.true_set.contains_key(atom) && !self.false_set.contains(atom)
    }

    /// Get all undefined atoms from a set of atoms
    pub fn undefined_atoms<'a>(
        &self,
        atoms: impl Iterator<Item = &'a WfsAtom>,
    ) -> Vec<&'a WfsAtom> {
        atoms.filter(|a| self.is_undefined(a)).collect()
    }
}

impl Default for WfsResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for Well-Founded Semantics evaluation.
///
/// Controls the convergence budget for the alternating fixed-point loop.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WfsConfig {
    /// Maximum iterations before giving up.
    pub max_iterations: usize,
}

impl Default for WfsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
        }
    }
}

/// Context for WFS evaluation
///
/// This holds all the data needed during WFS computation.
pub struct WfsContext<'a> {
    /// Ground rules for the SCC
    rules: &'a [WfsRule],
    /// All atoms in the SCC (derived from rule heads)
    scc_atoms: HashSet<WfsAtom>,
    /// Rules indexed by head predicate for fast lookup
    rules_by_head: HashMap<WfsAtom, Vec<usize>>,
    /// PIR graph for building provenance
    pir: &'a mut PirGraph,
}

impl<'a> WfsContext<'a> {
    /// Create a new WFS context
    fn new(rules: &'a [WfsRule], pir: &'a mut PirGraph) -> Self {
        // Collect all head atoms
        let mut scc_atoms = HashSet::new();
        let mut rules_by_head: HashMap<WfsAtom, Vec<usize>> = HashMap::new();

        for (i, rule) in rules.iter().enumerate() {
            scc_atoms.insert(rule.head.clone());
            rules_by_head.entry(rule.head.clone()).or_default().push(i);
        }

        Self {
            rules,
            scc_atoms,
            rules_by_head,
            pir,
        }
    }
}

/// Compute the unfounded set using a greatest fixed-point iteration.
///
/// An atom A is unfounded with respect to (T, F) if for every rule "A :- B":
/// - B contains a positive literal that is in the unfounded set, or
/// - B contains a literal that is known to be false given (T, F)
///
/// This computes the greatest fixed-point of the unfounded set operator.
fn compute_unfounded_set(
    ctx: &WfsContext,
    true_set: &HashSet<WfsAtom>,
    false_set: &HashSet<WfsAtom>,
) -> HashSet<WfsAtom> {
    // Start with all non-true atoms as potentially unfounded
    let mut unfounded: HashSet<WfsAtom> = ctx
        .scc_atoms
        .iter()
        .filter(|a| !true_set.contains(*a))
        .cloned()
        .collect();

    // Greatest fixed-point: keep removing atoms that have a supporting rule
    loop {
        let mut changed = false;
        let mut to_remove = Vec::new();

        for atom in &unfounded {
            // Check if any rule can support this atom
            if let Some(rule_indices) = ctx.rules_by_head.get(atom) {
                for &rule_idx in rule_indices {
                    let rule = &ctx.rules[rule_idx];
                    if rule_is_potentially_supporting(
                        rule,
                        true_set,
                        false_set,
                        &unfounded,
                        &ctx.scc_atoms,
                    ) {
                        to_remove.push(atom.clone());
                        changed = true;
                        break;
                    }
                }
            }
        }

        for atom in to_remove {
            unfounded.remove(&atom);
        }

        if !changed {
            break;
        }
    }

    unfounded
}

/// Check if a rule can potentially support its head atom.
///
/// A rule is potentially supporting if all body literals can potentially be satisfied:
/// - Positive literals must be either:
///   - Already true, or
///   - In the SCC and not in the unfounded set (could become true)
/// - Negative literals must not be known true
///
/// External atoms (not in SCC) that are not already true cannot become true,
/// so rules depending on them positively cannot fire.
fn rule_is_potentially_supporting(
    rule: &WfsRule,
    true_set: &HashSet<WfsAtom>,
    false_set: &HashSet<WfsAtom>,
    unfounded: &HashSet<WfsAtom>,
    scc_atoms: &HashSet<WfsAtom>,
) -> bool {
    for lit in &rule.body {
        match lit {
            WfsLiteral::Positive(atom) => {
                // If the positive literal is known false, rule can't fire
                if false_set.contains(atom) {
                    return false;
                }
                // If already true, this literal is satisfied
                if true_set.contains(atom) {
                    continue;
                }
                // If the atom is in the SCC and in the unfounded set,
                // it creates a circular dependency and can't support the head
                if scc_atoms.contains(atom) {
                    if unfounded.contains(atom) {
                        return false;
                    }
                    // Atom is in SCC and not unfounded - could potentially become true
                    continue;
                }
                // External atom (not in SCC) that is not true - can never become true
                // So this rule can never fire
                return false;
            }
            WfsLiteral::Negative(atom) => {
                // If the negative literal is known true, rule can't fire
                if true_set.contains(atom) {
                    return false;
                }
            }
        }
    }
    true
}

/// Derive consequences using the immediate consequence operator.
///
/// An atom becomes true if there exists a rule where:
/// - All positive body literals are in true_set or are external facts
/// - All negative body literals are in false_set
///
/// Returns new atoms to add to true_set with their provenances.
fn derive_consequences(
    ctx: &mut WfsContext,
    true_set: &HashSet<WfsAtom>,
    false_set: &HashSet<WfsAtom>,
    existing_provenance: &HashMap<WfsAtom, PirNodeId>,
) -> HashMap<WfsAtom, PirNodeId> {
    let mut new_true: HashMap<WfsAtom, PirNodeId> = HashMap::new();

    for rule in ctx.rules.iter() {
        // Skip if head is already true or false
        if true_set.contains(&rule.head) || false_set.contains(&rule.head) {
            continue;
        }

        // Check if rule fires
        if let Some(true) = rule.is_satisfied(true_set, false_set, &ctx.scc_atoms) {
            // Build provenance: rule's base provenance AND all positive body provenances
            let mut prov_parts = vec![rule.provenance];

            for lit in &rule.body {
                if let WfsLiteral::Positive(atom) = lit {
                    if let Some(&atom_prov) = existing_provenance.get(atom) {
                        prov_parts.push(atom_prov);
                    }
                    // External atoms (not in SCC) have their provenance already
                    // folded into rule.provenance during grounding
                }
            }

            let rule_prov = if prov_parts.len() == 1 {
                prov_parts[0]
            } else {
                ctx.pir.and(prov_parts)
            };

            // Combine with any existing provenance for this atom (multiple rules)
            let entry = new_true
                .entry(rule.head.clone())
                .or_insert_with(|| ctx.pir.const_false());
            *entry = ctx.pir.or(vec![*entry, rule_prov]);
        }
    }

    new_true
}

/// Evaluate a non-monotone SCC using Well-Founded Semantics.
///
/// This function takes pre-grounded rules for the SCC and computes the
/// well-founded model using the alternating fixed-point algorithm.
///
/// # Arguments
///
/// * `rules` - Ground rules for atoms in this SCC
/// * `pir` - PIR graph for building provenance formulas
/// * `config` - Configuration options
///
/// # Returns
///
/// WfsResult containing true atoms (with provenance), false atoms, and
/// implicitly undefined atoms (in neither set).
///
/// # Algorithm
///
/// The alternating fixed-point works by interleaving two operators:
/// 1. **Unfounded set computation** (Φ): Find atoms with no possible support
/// 2. **Consequence derivation** (Ψ): Find atoms that must be true
///
/// Starting from (T={}, F={}):
/// - Compute unfounded set U = Φ(T, F)
/// - Add U to F
/// - Compute consequences C = Ψ(T, F)
/// - Add C to T
/// - Repeat until no changes
pub fn evaluate_wfs_rules(
    rules: &[WfsRule],
    pir: &mut PirGraph,
    config: &WfsConfig,
) -> Result<WfsResult> {
    if rules.is_empty() {
        return Ok(WfsResult::new());
    }

    let mut ctx = WfsContext::new(rules, pir);

    let mut true_set: HashSet<WfsAtom> = HashSet::new();
    let mut true_provenance: HashMap<WfsAtom, PirNodeId> = HashMap::new();
    let mut false_set: HashSet<WfsAtom> = HashSet::new();

    for iteration in 0..config.max_iterations {
        // Step 1: Compute unfounded set
        let unfounded = compute_unfounded_set(&ctx, &true_set, &false_set);

        // Add unfounded atoms to false set
        let new_false: Vec<WfsAtom> = unfounded
            .into_iter()
            .filter(|a| !false_set.contains(a))
            .collect();
        let new_false_count = new_false.len();
        false_set.extend(new_false);

        // Step 2: Derive consequences
        let new_true = derive_consequences(&mut ctx, &true_set, &false_set, &true_provenance);
        let new_true_count = new_true.len();

        // Add new true atoms
        for (atom, prov) in new_true {
            if !true_set.contains(&atom) {
                true_set.insert(atom.clone());
                // Combine provenance if atom becomes true via multiple rules
                let entry = true_provenance
                    .entry(atom)
                    .or_insert_with(|| ctx.pir.const_false());
                *entry = ctx.pir.or(vec![*entry, prov]);
            }
        }

        // Check for fixed point
        if new_false_count == 0 && new_true_count == 0 {
            break;
        }

        if iteration == config.max_iterations - 1 {
            return Err(XlogError::Execution(format!(
                "WFS evaluation did not converge after {} iterations",
                config.max_iterations
            )));
        }
    }

    Ok(WfsResult {
        true_set: true_provenance,
        false_set,
    })
}

/// Legacy interface: Evaluate WFS given SCC predicates.
///
/// This is the original interface that takes predicate names. It creates
/// propositional atoms (no arguments) for simple testing. For full integration
/// with the provenance extractor, use `evaluate_wfs_rules` with proper grounding.
pub(crate) fn evaluate_wfs_scc(scc_predicates: &[String], pir: &mut PirGraph) -> Result<WfsResult> {
    evaluate_wfs_scc_with_config(scc_predicates, pir, &WfsConfig::default())
}

/// Evaluate WFS with custom configuration using predicate names.
///
/// Creates propositional atoms (no arguments) from predicate names.
/// This is suitable for testing but real programs should use `evaluate_wfs_rules`
/// with properly grounded rules from the provenance extractor.
pub(crate) fn evaluate_wfs_scc_with_config(
    scc_predicates: &[String],
    pir: &mut PirGraph,
    config: &WfsConfig,
) -> Result<WfsResult> {
    // For the predicate-name interface, we cannot compute WFS without rules.
    // If someone calls this without rules, we treat each predicate as a
    // propositional atom with no rules - making them all unfounded (false).

    if scc_predicates.is_empty() {
        return Ok(WfsResult::new());
    }

    // With no rules, all atoms are unfounded and thus false
    let false_set: HashSet<WfsAtom> = scc_predicates
        .iter()
        .map(|p| WfsAtom::new(p.clone(), vec![]))
        .collect();

    // Verify convergence by simulating one iteration
    // (Since we have no rules, this is trivially stable)
    let _ = config; // config would be used if we had rules
    let _ = pir; // pir would be used for provenance if we had rules

    Ok(WfsResult {
        true_set: HashMap::new(),
        false_set,
    })
}

/// Evaluate WFS with provided ground rules.
///
/// This is the main entry point for WFS evaluation during provenance extraction.
/// The caller (provenance extractor) is responsible for:
/// 1. Detecting non-monotone SCCs
/// 2. Grounding the rules in those SCCs
/// 3. Providing the ground rules with their provenances
pub fn evaluate_wfs_with_rules(rules: Vec<WfsRule>, pir: &mut PirGraph) -> Result<WfsResult> {
    evaluate_wfs_rules(&rules, pir, &WfsConfig::default())
}

/// Evaluate WFS with rules and custom configuration.
pub fn evaluate_wfs_with_rules_config(
    rules: Vec<WfsRule>,
    pir: &mut PirGraph,
    config: &WfsConfig,
) -> Result<WfsResult> {
    evaluate_wfs_rules(&rules, pir, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pir::LeafId;

    /// Helper to create a simple propositional atom (no args)
    fn prop(name: &str) -> WfsAtom {
        WfsAtom::new(name, vec![])
    }

    /// Helper to create a ground atom with integer args
    fn atom(name: &str, args: &[i64]) -> WfsAtom {
        WfsAtom::new(name, args.iter().map(|&i| Value::I64(i)).collect())
    }

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
        let atom = prop("p");

        // Initially undefined
        assert_eq!(result.truth_value(&atom), TruthValue::Undefined);

        // After adding to false_set
        result.false_set.insert(atom.clone());
        assert_eq!(result.truth_value(&atom), TruthValue::False);

        // After moving to true_set
        result.false_set.remove(&atom);
        let mut pir = PirGraph::new();
        let node_id = pir.lit(LeafId::new(0));
        result.true_set.insert(atom.clone(), node_id);
        assert_eq!(result.truth_value(&atom), TruthValue::True);
    }

    #[test]
    fn test_wfs_atom_equality() {
        let atom1 = WfsAtom::new("p", vec![Value::I64(1)]);
        let atom2 = WfsAtom::new("p", vec![Value::I64(1)]);
        let atom3 = WfsAtom::new("p", vec![Value::I64(2)]);

        assert_eq!(atom1, atom2);
        assert_ne!(atom1, atom3);
    }

    #[test]
    fn test_wfs_empty_rules() {
        let mut pir = PirGraph::new();
        let result = evaluate_wfs_rules(&[], &mut pir, &WfsConfig::default()).unwrap();
        assert!(result.true_set.is_empty());
        assert!(result.false_set.is_empty());
    }

    #[test]
    fn test_wfs_simple_fact() {
        // Test: p. (fact)
        // p :- (empty body, always true)
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![WfsRule::new(prop("p"), vec![], const_true)];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("p")), TruthValue::True);
        assert!(result.provenance(&prop("p")).is_some());
    }

    #[test]
    fn test_wfs_simple_negation() {
        // Test: p :- not q. q :- not p.
        // Classic non-stratifiable program
        // Under WFS, both p and q are undefined
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("p"), vec![WfsLiteral::Negative(prop("q"))], const_true),
            WfsRule::new(prop("q"), vec![WfsLiteral::Negative(prop("p"))], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        // Both should be undefined (not in true_set, not in false_set)
        assert_eq!(result.truth_value(&prop("p")), TruthValue::Undefined);
        assert_eq!(result.truth_value(&prop("q")), TruthValue::Undefined);
    }

    #[test]
    fn test_wfs_asymmetric_negation() {
        // Test: p :- not q. q.
        // q is a fact, so it's true. Therefore p is false (not q is false).
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("p"), vec![WfsLiteral::Negative(prop("q"))], const_true),
            WfsRule::new(prop("q"), vec![], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("q")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("p")), TruthValue::False);
    }

    #[test]
    fn test_wfs_three_way_cycle() {
        // Test: p :- not q. q :- not r. r :- not p.
        // This creates a 3-way cycle where all are undefined
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("p"), vec![WfsLiteral::Negative(prop("q"))], const_true),
            WfsRule::new(prop("q"), vec![WfsLiteral::Negative(prop("r"))], const_true),
            WfsRule::new(prop("r"), vec![WfsLiteral::Negative(prop("p"))], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        // All three are in a cycle through negation, so all undefined
        assert_eq!(result.truth_value(&prop("p")), TruthValue::Undefined);
        assert_eq!(result.truth_value(&prop("q")), TruthValue::Undefined);
        assert_eq!(result.truth_value(&prop("r")), TruthValue::Undefined);
    }

    #[test]
    fn test_wfs_win_lose() {
        // Classic example: win(X) :- move(X,Y), not win(Y).
        // Grounded for a simple two-node game: move(1,2). move(2,1).
        //
        // win(1) :- move(1,2), not win(2).
        // win(2) :- move(2,1), not win(1).
        //
        // Both win(1) and win(2) are undefined (mutual negation through positive deps)
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        // We model move as already resolved (external to SCC)
        let rules = vec![
            WfsRule::new(
                atom("win", &[1]),
                vec![WfsLiteral::Negative(atom("win", &[2]))],
                const_true, // move(1,2) is true
            ),
            WfsRule::new(
                atom("win", &[2]),
                vec![WfsLiteral::Negative(atom("win", &[1]))],
                const_true, // move(2,1) is true
            ),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        // Both win(1) and win(2) are undefined
        assert_eq!(
            result.truth_value(&atom("win", &[1])),
            TruthValue::Undefined
        );
        assert_eq!(
            result.truth_value(&atom("win", &[2])),
            TruthValue::Undefined
        );
    }

    #[test]
    fn test_wfs_win_with_base_case() {
        // win(X) :- move(X,Y), not win(Y).
        // Grounded: move(1,2). (no move from 2)
        //
        // This demonstrates that atoms NOT in the SCC (external atoms) are treated
        // as having their truth value determined by the external context.
        //
        // In a real integration:
        // - win(2) would have no rule, so during grounding it wouldn't be added
        // - The "not win(2)" would be evaluated against the external store
        // - If win(2) doesn't exist externally, closed-world assumes it's false
        //
        // For this isolated test, we simulate the scenario where:
        // - win(2) has an empty rule set (meaning it's unfounded)
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        // We need to ensure win(2) is tracked as an SCC atom that gets resolved
        // The simplest way: include a "failing" rule for win(2) that can never fire
        // Or: include win(2) with no rules (it will be unfounded -> false)

        // To properly test this, we add win(2) as a head that depends on something impossible
        // This makes win(2) unfounded and thus false
        let rules = vec![
            WfsRule::new(
                atom("win", &[1]),
                vec![WfsLiteral::Negative(atom("win", &[2]))],
                const_true,
            ),
            // win(2) :- win(3). but win(3) doesn't exist, so win(2) is unfounded
            WfsRule::new(
                atom("win", &[2]),
                vec![WfsLiteral::Positive(atom("win", &[3]))],
                const_true,
            ),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        // win(3) is not in SCC (no head), so the positive dependency on it fails
        // This means win(2)'s only rule can't fire, making win(2) unfounded -> false
        // With win(2) false, "not win(2)" succeeds, so win(1) becomes true
        assert_eq!(result.truth_value(&atom("win", &[1])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("win", &[2])), TruthValue::False);
    }

    #[test]
    fn test_wfs_chain_with_grounding() {
        // Test a longer chain that resolves:
        // a. b :- not a. c :- not b.
        //
        // a is a fact (true)
        // b :- not a fails (a is true, so not a is false) -> b is false
        // c :- not b succeeds (b is false, so not b is true) -> c is true
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("a"), vec![], const_true),
            WfsRule::new(prop("b"), vec![WfsLiteral::Negative(prop("a"))], const_true),
            WfsRule::new(prop("c"), vec![WfsLiteral::Negative(prop("b"))], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("a")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("b")), TruthValue::False);
        assert_eq!(result.truth_value(&prop("c")), TruthValue::True);
    }

    #[test]
    fn test_wfs_multiple_rules_same_head() {
        // Test: p :- q. p :- not r. q. r.
        // p can be derived from q (which is true)
        // p's second rule fails because r is true
        // Final: p is true (via first rule), q is true, r is true
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("q"), vec![], const_true),
            WfsRule::new(prop("r"), vec![], const_true),
            WfsRule::new(prop("p"), vec![WfsLiteral::Positive(prop("q"))], const_true),
            WfsRule::new(prop("p"), vec![WfsLiteral::Negative(prop("r"))], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("q")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("r")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("p")), TruthValue::True);
    }

    #[test]
    fn test_wfs_provenance_tracking() {
        // Test that provenance is correctly tracked
        // p :- (with leaf probability)
        let mut pir = PirGraph::new();
        let leaf = pir.lit(LeafId::new(42));

        let rules = vec![WfsRule::new(prop("p"), vec![], leaf)];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        let prov = result.provenance(&prop("p"));
        assert!(prov.is_some());
        // The provenance is built through OR combination, so it may not be the exact same node
        // but it should exist and have a valid ID
        assert!(prov.unwrap().as_u32() > 0);
    }

    #[test]
    fn test_wfs_combined_provenance() {
        // Test: p :- q. where q has a probabilistic provenance
        // p's provenance should combine the rule's provenance with q's provenance
        let mut pir = PirGraph::new();
        let leaf_q = pir.lit(LeafId::new(1));
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("q"), vec![], leaf_q),
            WfsRule::new(prop("p"), vec![WfsLiteral::Positive(prop("q"))], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("q")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("p")), TruthValue::True);

        // p's provenance should be AND(const_true, leaf_q)
        // which simplifies to leaf_q in a smarter builder
        let p_prov = result.provenance(&prop("p")).unwrap();
        // The exact node ID depends on builder implementation
        assert!(p_prov.as_u32() > 0);
    }

    #[test]
    fn test_wfs_no_rules_for_predicate() {
        // When predicates have no rules, they should be unfounded (false)
        let result = evaluate_wfs_scc_with_config(
            &["p".to_string(), "q".to_string()],
            &mut PirGraph::new(),
            &WfsConfig::default(),
        )
        .unwrap();

        assert_eq!(result.truth_value(&prop("p")), TruthValue::False);
        assert_eq!(result.truth_value(&prop("q")), TruthValue::False);
    }

    #[test]
    fn test_wfs_positive_cycle() {
        // Test positive cycle (not through negation): p :- q. q :- p.
        // Without any base case, both are unfounded and thus false.
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("p"), vec![WfsLiteral::Positive(prop("q"))], const_true),
            WfsRule::new(prop("q"), vec![WfsLiteral::Positive(prop("p"))], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        // Pure positive cycle with no external support -> both unfounded -> false
        assert_eq!(result.truth_value(&prop("p")), TruthValue::False);
        assert_eq!(result.truth_value(&prop("q")), TruthValue::False);
    }

    #[test]
    fn test_wfs_positive_cycle_with_base() {
        // Test: p :- q. q :- p. p.
        // p has a fact, so p is true, q depends on p, so q is true
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("p"), vec![], const_true), // Base fact
            WfsRule::new(prop("p"), vec![WfsLiteral::Positive(prop("q"))], const_true),
            WfsRule::new(prop("q"), vec![WfsLiteral::Positive(prop("p"))], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("p")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("q")), TruthValue::True);
    }

    #[test]
    fn test_wfs_mixed_dependencies() {
        // Complex test: a. b :- a. c :- not b. d :- c, not e. e :- not d.
        //
        // a is true (fact)
        // b is true (depends on a)
        // c is false (not b fails because b is true)
        // Now d :- c, not e and e :- not d
        // c is false, so d can't fire from that path
        // d has no way to be true, so d is unfounded -> false
        // e :- not d fires (d is false) -> e is true
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("a"), vec![], const_true),
            WfsRule::new(prop("b"), vec![WfsLiteral::Positive(prop("a"))], const_true),
            WfsRule::new(prop("c"), vec![WfsLiteral::Negative(prop("b"))], const_true),
            WfsRule::new(
                prop("d"),
                vec![
                    WfsLiteral::Positive(prop("c")),
                    WfsLiteral::Negative(prop("e")),
                ],
                const_true,
            ),
            WfsRule::new(prop("e"), vec![WfsLiteral::Negative(prop("d"))], const_true),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("a")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("b")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("c")), TruthValue::False);
        assert_eq!(result.truth_value(&prop("d")), TruthValue::False);
        assert_eq!(result.truth_value(&prop("e")), TruthValue::True);
    }

    #[test]
    fn test_wfs_undefined_atoms_method() {
        let mut result = WfsResult::new();
        let p = prop("p");
        let q = prop("q");
        let r = prop("r");

        result.true_set.insert(p.clone(), PirNodeId::from(0));
        result.false_set.insert(q.clone());
        // r is undefined

        let atoms = vec![&p, &q, &r];
        let undefined = result.undefined_atoms(atoms.into_iter());

        assert_eq!(undefined.len(), 1);
        assert_eq!(undefined[0], &r);
    }

    // Helper for PirNodeId construction in tests
    impl From<u32> for PirNodeId {
        fn from(v: u32) -> Self {
            // This is just for testing; normally PirNodeId comes from PirGraph
            unsafe { std::mem::transmute(v) }
        }
    }

    #[test]
    fn test_wfs_even_odd_cycle() {
        // Classic even/odd example with self-negation:
        // even(0).
        // even(X) :- succ(Y, X), odd(Y).
        // odd(X) :- succ(Y, X), even(Y).
        //
        // Grounded for 0, 1, 2:
        // even(0) is a fact
        // even(1) :- odd(0). odd(0) :- even(0). so odd(0) is true
        // therefore even(1) is true
        // etc.
        //
        // This tests positive recursion through multiple predicates
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            // even(0) is a fact
            WfsRule::new(atom("even", &[0]), vec![], const_true),
            // odd(0) :- even(0)  (succ(0, 0+1) doesn't exist, so this is from even(0))
            // Actually let's simplify: odd(N) :- even(N-1) for N > 0
            // odd(1) :- even(0)
            WfsRule::new(
                atom("odd", &[1]),
                vec![WfsLiteral::Positive(atom("even", &[0]))],
                const_true,
            ),
            // even(2) :- odd(1)
            WfsRule::new(
                atom("even", &[2]),
                vec![WfsLiteral::Positive(atom("odd", &[1]))],
                const_true,
            ),
            // odd(3) :- even(2)
            WfsRule::new(
                atom("odd", &[3]),
                vec![WfsLiteral::Positive(atom("even", &[2]))],
                const_true,
            ),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&atom("even", &[0])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("odd", &[1])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("even", &[2])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("odd", &[3])), TruthValue::True);
    }

    #[test]
    fn test_wfs_default_logic() {
        // Default logic example: bird(X) :- penguin(X). flies(X) :- bird(X), not ab(X). ab(X) :- penguin(X).
        // penguin(tweety).
        //
        // Grounded:
        // bird(tweety) :- penguin(tweety). (penguin is external/fact)
        // flies(tweety) :- bird(tweety), not ab(tweety).
        // ab(tweety) :- penguin(tweety). (penguin is external/fact)
        //
        // If penguin(tweety) is true (external):
        // - bird(tweety) becomes true
        // - ab(tweety) becomes true
        // - flies(tweety) fails because ab(tweety) is true
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        // Simulate penguin(tweety) being true externally by including it as a fact in provenance
        let rules = vec![
            // penguin(tweety) is a fact (external, but we include it for completeness)
            WfsRule::new(atom("penguin", &[1]), vec![], const_true), // tweety = 1
            // bird(tweety) :- penguin(tweety)
            WfsRule::new(
                atom("bird", &[1]),
                vec![WfsLiteral::Positive(atom("penguin", &[1]))],
                const_true,
            ),
            // ab(tweety) :- penguin(tweety)
            WfsRule::new(
                atom("ab", &[1]),
                vec![WfsLiteral::Positive(atom("penguin", &[1]))],
                const_true,
            ),
            // flies(tweety) :- bird(tweety), not ab(tweety)
            WfsRule::new(
                atom("flies", &[1]),
                vec![
                    WfsLiteral::Positive(atom("bird", &[1])),
                    WfsLiteral::Negative(atom("ab", &[1])),
                ],
                const_true,
            ),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&atom("penguin", &[1])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("bird", &[1])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("ab", &[1])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("flies", &[1])), TruthValue::False);
    }

    #[test]
    fn test_wfs_non_ground_birds() {
        // Test multiple individuals with the bird/flies/ab pattern
        // bird(X) :- penguin(X). bird(X) :- eagle(X).
        // flies(X) :- bird(X), not ab(X).
        // ab(X) :- penguin(X).
        //
        // Facts: penguin(tweety). eagle(sam).
        //
        // Results:
        // - tweety: bird, ab, not flies
        // - sam: bird, not ab, flies
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            // Facts
            WfsRule::new(atom("penguin", &[1]), vec![], const_true), // tweety = 1
            WfsRule::new(atom("eagle", &[2]), vec![], const_true),   // sam = 2
            // bird(X) :- penguin(X)
            WfsRule::new(
                atom("bird", &[1]),
                vec![WfsLiteral::Positive(atom("penguin", &[1]))],
                const_true,
            ),
            // bird(X) :- eagle(X)
            WfsRule::new(
                atom("bird", &[2]),
                vec![WfsLiteral::Positive(atom("eagle", &[2]))],
                const_true,
            ),
            // ab(X) :- penguin(X)
            WfsRule::new(
                atom("ab", &[1]),
                vec![WfsLiteral::Positive(atom("penguin", &[1]))],
                const_true,
            ),
            // flies(tweety) :- bird(tweety), not ab(tweety)
            WfsRule::new(
                atom("flies", &[1]),
                vec![
                    WfsLiteral::Positive(atom("bird", &[1])),
                    WfsLiteral::Negative(atom("ab", &[1])),
                ],
                const_true,
            ),
            // flies(sam) :- bird(sam), not ab(sam)
            // Note: ab(sam) has no rule, so it's unfounded and false
            WfsRule::new(
                atom("flies", &[2]),
                vec![
                    WfsLiteral::Positive(atom("bird", &[2])),
                    WfsLiteral::Negative(atom("ab", &[2])),
                ],
                const_true,
            ),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        // tweety (1)
        assert_eq!(result.truth_value(&atom("penguin", &[1])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("bird", &[1])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("ab", &[1])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("flies", &[1])), TruthValue::False);

        // sam (2)
        assert_eq!(result.truth_value(&atom("eagle", &[2])), TruthValue::True);
        assert_eq!(result.truth_value(&atom("bird", &[2])), TruthValue::True);
        // ab(2) has no rule, so it's not in the SCC atoms and treated as external false
        // Actually, ab(2) is referenced in a negative literal but has no rule
        // Let's check what happens
        assert_eq!(result.truth_value(&atom("flies", &[2])), TruthValue::True);
    }

    #[test]
    fn test_wfs_probabilistic_provenance_or() {
        // Test that provenance correctly combines when multiple rules derive same atom
        // p :- q. p :- r. q. r.
        // p's provenance should be OR(q_prov, r_prov)
        let mut pir = PirGraph::new();
        let leaf_q = pir.lit(LeafId::new(1));
        let leaf_r = pir.lit(LeafId::new(2));

        let rules = vec![
            WfsRule::new(prop("q"), vec![], leaf_q),
            WfsRule::new(prop("r"), vec![], leaf_r),
            WfsRule::new(
                prop("p"),
                vec![WfsLiteral::Positive(prop("q"))],
                pir.const_true(),
            ),
            WfsRule::new(
                prop("p"),
                vec![WfsLiteral::Positive(prop("r"))],
                pir.const_true(),
            ),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("p")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("q")), TruthValue::True);
        assert_eq!(result.truth_value(&prop("r")), TruthValue::True);

        // p's provenance should exist and be a combination
        let p_prov = result.provenance(&prop("p")).unwrap();
        // The exact structure depends on builder, but it should exist
        assert!(p_prov.as_u32() > 0);
    }

    #[test]
    fn test_wfs_stable_model_unique() {
        // Test a program with a unique stable model:
        // a :- not b, not c. b :- not a, not c. c :- not a, not b. d.
        //
        // With d as a fact, this doesn't change the a/b/c cycle.
        // Under WFS, a, b, c are all undefined (three-way cycle).
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            WfsRule::new(prop("d"), vec![], const_true),
            WfsRule::new(
                prop("a"),
                vec![
                    WfsLiteral::Negative(prop("b")),
                    WfsLiteral::Negative(prop("c")),
                ],
                const_true,
            ),
            WfsRule::new(
                prop("b"),
                vec![
                    WfsLiteral::Negative(prop("a")),
                    WfsLiteral::Negative(prop("c")),
                ],
                const_true,
            ),
            WfsRule::new(
                prop("c"),
                vec![
                    WfsLiteral::Negative(prop("a")),
                    WfsLiteral::Negative(prop("b")),
                ],
                const_true,
            ),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&prop("d")), TruthValue::True);
        // All three are in a cycle, so undefined under WFS
        assert_eq!(result.truth_value(&prop("a")), TruthValue::Undefined);
        assert_eq!(result.truth_value(&prop("b")), TruthValue::Undefined);
        assert_eq!(result.truth_value(&prop("c")), TruthValue::Undefined);
    }

    #[test]
    fn test_wfs_hamiltonian_cycle_like() {
        // A pattern similar to what appears in Hamiltonian cycle encodings:
        // in(1,2) :- edge(1,2), not out(1,2).
        // out(1,2) :- edge(1,2), not in(1,2).
        //
        // If edge(1,2) is true, both in(1,2) and out(1,2) are undefined
        // (classic non-stratified choice)
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        let rules = vec![
            // edge(1,2) as a fact
            WfsRule::new(atom("edge", &[1, 2]), vec![], const_true),
            // in(1,2) :- edge(1,2), not out(1,2)
            WfsRule::new(
                atom("in", &[1, 2]),
                vec![
                    WfsLiteral::Positive(atom("edge", &[1, 2])),
                    WfsLiteral::Negative(atom("out", &[1, 2])),
                ],
                const_true,
            ),
            // out(1,2) :- edge(1,2), not in(1,2)
            WfsRule::new(
                atom("out", &[1, 2]),
                vec![
                    WfsLiteral::Positive(atom("edge", &[1, 2])),
                    WfsLiteral::Negative(atom("in", &[1, 2])),
                ],
                const_true,
            ),
        ];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        assert_eq!(result.truth_value(&atom("edge", &[1, 2])), TruthValue::True);
        // Both in and out are in a cycle through negation
        assert_eq!(
            result.truth_value(&atom("in", &[1, 2])),
            TruthValue::Undefined
        );
        assert_eq!(
            result.truth_value(&atom("out", &[1, 2])),
            TruthValue::Undefined
        );
    }

    #[test]
    fn test_wfs_partial_grounding() {
        // Test that atoms with no rules at all are not tracked
        // (they're external and handled by the provenance extractor)
        let mut pir = PirGraph::new();
        let const_true = pir.const_true();

        // p :- q. (q has no rule, so it's external)
        let rules = vec![WfsRule::new(
            prop("p"),
            vec![WfsLiteral::Positive(prop("q"))],
            const_true,
        )];

        let result = evaluate_wfs_rules(&rules, &mut pir, &WfsConfig::default()).unwrap();

        // p depends positively on q, which is external and not true
        // Therefore p's rule can never fire, making p unfounded and false
        assert_eq!(result.truth_value(&prop("p")), TruthValue::False);
        // q is not tracked (not an SCC atom)
        assert_eq!(result.truth_value(&prop("q")), TruthValue::Undefined);
    }
}
