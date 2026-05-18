//! Epistemic Intermediate Representation.

/// Epistemic semantics mode selected for an EIR program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EirEpistemicMode {
    /// G91 compatibility semantics.
    G91,
    /// Founded Autoepistemic Equilibrium Logic semantics.
    Faeel,
}

/// Epistemic operator attached to an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EirEpistemicOp {
    /// The atom is known/believed true in the selected epistemic mode.
    Know,
    /// The atom is possible/consistent in the selected epistemic mode.
    Possible,
}

/// Atom summary carried across the EIR boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EirAtom {
    /// Predicate name.
    pub predicate: String,
    /// Predicate arity.
    pub arity: usize,
}

/// Explicit epistemic body literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EirEpistemicLiteral {
    /// Epistemic operator.
    pub op: EirEpistemicOp,
    /// Whether the epistemic literal is explicitly negated.
    pub negated: bool,
    /// Atom under the epistemic operator.
    pub atom: EirAtom,
}

/// Body literal at the epistemic boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EirBodyLiteral {
    /// Non-epistemic relational atom.
    Relational {
        /// Whether the relational atom is negated.
        negated: bool,
        /// Atom summary.
        atom: EirAtom,
    },
    /// Explicit epistemic atom.
    Epistemic(EirEpistemicLiteral),
    /// Non-relational constraint or comparison.
    Constraint,
    /// Variable binding expression.
    Binding,
}

/// Rule represented at the EIR boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EirRule {
    /// Rule head.
    pub head: EirAtom,
    /// Rule body.
    pub body: Vec<EirBodyLiteral>,
}

/// Program represented at the EIR boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EirProgram {
    /// Selected epistemic semantics mode.
    pub mode: EirEpistemicMode,
    /// Rules in source order.
    pub rules: Vec<EirRule>,
}
