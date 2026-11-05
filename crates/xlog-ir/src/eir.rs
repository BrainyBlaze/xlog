//! Epistemic Intermediate Representation.

/// Epistemic semantics mode selected for an EIR program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EirEpistemicMode {
    /// Gelfond-1991-style compatibility semantics, selected by `g91`.
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

/// Term representation preserved at the epistemic boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EirTerm {
    /// Named logic variable.
    Variable(String),
    /// Anonymous wildcard.
    Anonymous,
    /// Integer literal in the inclusive range `i64::MIN..=u64::MAX`.
    Integer(i128),
    /// Floating-point literal represented by its IEEE-754 bits.
    FloatBits(u64),
    /// Quoted string literal.
    String(String),
    /// Interned symbol identifier.
    Symbol(u32),
    /// Finite list literal.
    List(Vec<EirTerm>),
    /// Finite cons pattern.
    Cons {
        /// Head term.
        head: Box<EirTerm>,
        /// Tail term.
        tail: Box<EirTerm>,
    },
    /// Finite compound term.
    Compound {
        /// Functor name.
        functor: String,
        /// Compound arguments.
        args: Vec<EirTerm>,
    },
    /// Static predicate reference.
    PredRef(String),
    /// Aggregate term preserved from a rule head.
    Aggregate {
        /// Aggregate operator name.
        op: String,
        /// Variable being aggregated.
        variable: String,
    },
}

impl EirTerm {
    /// Checked heap geometry of the original term clone. The inline enum is
    /// already part of its containing vector or box; scalar terms add no heap
    /// allocation here.
    pub fn owned_allocation_bytes(&self) -> xlog_core::Result<u64> {
        let overflow = || xlog_core::XlogError::AllocationRefused {
            operation: "epistemic term clone",
            reason: xlog_core::error::AllocationRefusal::ExtentOverflow,
        };
        let text = |text: &str| u64::try_from(text.len()).map_err(|_| overflow());
        let terms = |terms: &[Self]| -> xlog_core::Result<u64> {
            let mut bytes = terms
                .len()
                .checked_mul(std::mem::size_of::<Self>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(overflow)?;
            for term in terms {
                bytes = bytes
                    .checked_add(term.owned_allocation_bytes()?)
                    .ok_or_else(overflow)?;
            }
            Ok(bytes)
        };
        match self {
            Self::Variable(value) | Self::String(value) | Self::PredRef(value) => text(value),
            Self::List(items) => terms(items),
            Self::Cons { head, tail } => (2 * std::mem::size_of::<Self>() as u64)
                .checked_add(head.owned_allocation_bytes()?)
                .and_then(|bytes| bytes.checked_add(tail.owned_allocation_bytes().ok()?))
                .ok_or_else(overflow),
            Self::Compound { functor, args } => text(functor)?
                .checked_add(terms(args)?)
                .ok_or_else(overflow),
            Self::Aggregate { op, variable } => {
                text(op)?.checked_add(text(variable)?).ok_or_else(overflow)
            }
            Self::Anonymous | Self::Integer(_) | Self::FloatBits(_) | Self::Symbol(_) => Ok(0),
        }
    }
}

/// Atom summary carried across the EIR boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EirAtom {
    /// Predicate name.
    pub predicate: String,
    /// Predicate arity.
    pub arity: usize,
    /// Source atom terms preserved for tuple-key matching.
    pub terms: Vec<EirTerm>,
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

/// Integrity constraint represented at the EIR boundary.
///
/// A constraint has no head: the conjunction of its body literals must never
/// be satisfiable. Epistemic body literals are preserved first-class so that
/// `know`/`possible` integrity constraints constrain accepted world views,
/// rather than being silently rewritten into ordinary relational constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EirConstraint {
    /// Stable index of this constraint in the complete authored program.
    pub authored_index: Option<usize>,
    /// Constraint body whose conjunction must never hold in an accepted world view.
    pub body: Vec<EirBodyLiteral>,
}

/// Program represented at the EIR boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EirProgram {
    /// Selected epistemic semantics mode.
    pub mode: EirEpistemicMode,
    /// Rules in source order.
    pub rules: Vec<EirRule>,
    /// Integrity constraints in source order.
    pub constraints: Vec<EirConstraint>,
}
