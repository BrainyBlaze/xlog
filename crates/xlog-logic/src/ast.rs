//! Abstract Syntax Tree for XLOG programs

use xlog_core::ScalarType;

/// A term in an atom
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Variable(String),
    /// Anonymous wildcard `_` - each occurrence is a fresh unnamed variable
    Anonymous,
    Integer(i64),
    Float(f64),
    String(String),
    /// Interned symbol ID - use `xlog_core::symbol::resolve(id)` to get the string
    Symbol(u32),
    Aggregate(AggExpr),
}

impl Term {
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }

    /// Returns true if this is an anonymous wildcard `_`
    pub fn is_anonymous(&self) -> bool {
        matches!(self, Term::Anonymous)
    }

    /// Returns true if this is any kind of variable (named or anonymous)
    pub fn is_any_variable(&self) -> bool {
        matches!(self, Term::Variable(_) | Term::Anonymous)
    }

    pub fn is_constant(&self) -> bool {
        !self.is_any_variable() && !matches!(self, Term::Aggregate(_))
    }

    /// Returns the variable name, or None for anonymous/constants
    pub fn variable_name(&self) -> Option<&str> {
        match self {
            Term::Variable(name) => Some(name),
            _ => None,
        }
    }
}

/// Aggregate expression
#[derive(Debug, Clone, PartialEq)]
pub struct AggExpr {
    pub op: AggOp,
    pub variable: String,
}

/// Aggregation operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggOp {
    Count,
    Sum,
    Min,
    Max,
    LogSumExp,
}

/// Arithmetic expression tree
#[derive(Debug, Clone, PartialEq)]
pub enum ArithExpr {
    Variable(String),
    Integer(i64),
    Float(f64),

    // Binary operations
    Add(Box<ArithExpr>, Box<ArithExpr>),
    Sub(Box<ArithExpr>, Box<ArithExpr>),
    Mul(Box<ArithExpr>, Box<ArithExpr>),
    Div(Box<ArithExpr>, Box<ArithExpr>),
    Mod(Box<ArithExpr>, Box<ArithExpr>),

    // Built-in functions
    Abs(Box<ArithExpr>),
    Min(Box<ArithExpr>, Box<ArithExpr>),
    Max(Box<ArithExpr>, Box<ArithExpr>),
    Pow(Box<ArithExpr>, Box<ArithExpr>),

    // Type cast
    Cast(Box<ArithExpr>, ScalarType),
}

impl ArithExpr {
    /// Get all variable names used in this expression
    pub fn variables(&self) -> Vec<&str> {
        match self {
            ArithExpr::Variable(name) => vec![name.as_str()],
            ArithExpr::Integer(_) | ArithExpr::Float(_) => vec![],
            ArithExpr::Add(l, r) | ArithExpr::Sub(l, r) | ArithExpr::Mul(l, r)
            | ArithExpr::Div(l, r) | ArithExpr::Mod(l, r)
            | ArithExpr::Min(l, r) | ArithExpr::Max(l, r) | ArithExpr::Pow(l, r) => {
                let mut vars = l.variables();
                vars.extend(r.variables());
                vars
            }
            ArithExpr::Abs(e) | ArithExpr::Cast(e, _) => e.variables(),
        }
    }
}

/// Is-expression for variable binding: Z is X + Y
#[derive(Debug, Clone, PartialEq)]
pub struct IsExpr {
    pub target: String,      // Must be fresh (unbound) variable
    pub expr: ArithExpr,
}

/// An atom (predicate applied to terms)
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    pub predicate: String,
    pub terms: Vec<Term>,
}

impl Atom {
    pub fn arity(&self) -> usize {
        self.terms.len()
    }

    pub fn variables(&self) -> Vec<&str> {
        self.terms
            .iter()
            .filter_map(|t| t.variable_name())
            .collect()
    }
}

/// Comparison operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    Eq, Ne, Lt, Le, Gt, Ge,
}

/// A comparison expression
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub left: Term,
    pub op: CompOp,
    pub right: Term,
}

/// A literal in the body of a rule
#[derive(Debug, Clone, PartialEq)]
pub enum BodyLiteral {
    Positive(Atom),
    Negated(Atom),
    Comparison(Comparison),
    IsExpr(IsExpr),
}

impl BodyLiteral {
    pub fn is_positive(&self) -> bool {
        matches!(self, BodyLiteral::Positive(_))
    }

    pub fn is_negated(&self) -> bool {
        matches!(self, BodyLiteral::Negated(_))
    }

    pub fn atom(&self) -> Option<&Atom> {
        match self {
            BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => Some(a),
            BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => None,
        }
    }

    pub fn variables(&self) -> Vec<&str> {
        match self {
            BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => a.variables(),
            BodyLiteral::Comparison(c) => {
                let mut vars = vec![];
                if let Some(v) = c.left.variable_name() { vars.push(v); }
                if let Some(v) = c.right.variable_name() { vars.push(v); }
                vars
            }
            BodyLiteral::IsExpr(is_expr) => {
                let mut vars = is_expr.expr.variables();
                vars.push(is_expr.target.as_str());
                vars
            }
        }
    }
}

/// A rule (head :- body)
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<BodyLiteral>,
}

impl Rule {
    pub fn is_fact(&self) -> bool { self.body.is_empty() }

    pub fn has_negation(&self) -> bool {
        self.body.iter().any(|l| l.is_negated())
    }

    pub fn has_aggregation(&self) -> bool {
        self.head.terms.iter().any(|t| matches!(t, Term::Aggregate(_)))
    }

    pub fn body_predicates(&self) -> Vec<&str> {
        self.body.iter().filter_map(|l| l.atom().map(|a| a.predicate.as_str())).collect()
    }

    pub fn head_variables(&self) -> Vec<&str> { self.head.variables() }

    pub fn body_variables(&self) -> Vec<&str> {
        self.body.iter().flat_map(|l| l.variables()).collect()
    }
}

/// A constraint (:- body)
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub body: Vec<BodyLiteral>,
}

/// A query (?- atom)
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub atom: Atom,
}

/// Probabilistic engine selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbEngine {
    ExactDdnnf,
    Mc,
}

/// Probabilistic compilation caching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbCache {
    On,
    Off,
}

/// Compilation/evaluation directives (e.g., `#pragma ...`)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Directives {
    pub prob_engine: Option<ProbEngine>,
    pub prob_cache: Option<ProbCache>,
}

impl Directives {
    pub fn prob_engine_or_default(&self) -> ProbEngine {
        self.prob_engine.unwrap_or(ProbEngine::ExactDdnnf)
    }
}

/// A probabilistic fact (`p::atom.`)
#[derive(Debug, Clone, PartialEq)]
pub struct ProbFact {
    pub prob: f64,
    pub atom: Atom,
}

/// Annotated disjunction (`p1::a1; p2::a2.`)
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotatedDisjunction {
    pub choices: Vec<ProbFact>,
}

/// Evidence statement (`evidence(atom, true|false).`)
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    pub atom: Atom,
    pub value: bool,
}

/// Probabilistic query statement (`query(atom).`)
#[derive(Debug, Clone, PartialEq)]
pub struct ProbQuery {
    pub atom: Atom,
}

/// Import statement: use module. or use module::{pred1, pred2}.
#[derive(Debug, Clone, PartialEq)]
pub struct UseDecl {
    /// Module path segments, e.g., ["utils", "math"]
    pub module_path: Vec<String>,
    /// Specific imports (None = import all public)
    pub imports: Option<Vec<String>>,
}

/// Domain declaration
#[derive(Debug, Clone, PartialEq)]
pub struct DomainDecl {
    pub name: String,
    pub typ: ScalarType,
}

/// Predicate declaration
#[derive(Debug, Clone, PartialEq)]
pub struct PredDecl {
    pub name: String,
    pub types: Vec<ScalarType>,
    pub is_private: bool,
}

/// A complete XLOG program
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub imports: Vec<UseDecl>,
    pub domains: Vec<DomainDecl>,
    pub predicates: Vec<PredDecl>,
    pub rules: Vec<Rule>,
    pub constraints: Vec<Constraint>,
    pub queries: Vec<Query>,
    pub prob_facts: Vec<ProbFact>,
    pub annotated_disjunctions: Vec<AnnotatedDisjunction>,
    pub evidence: Vec<Evidence>,
    pub prob_queries: Vec<ProbQuery>,
    pub directives: Directives,
}

impl Program {
    pub fn new() -> Self { Self::default() }

    pub fn facts(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter().filter(|r| r.is_fact())
    }

    pub fn proper_rules(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter().filter(|r| !r.is_fact())
    }

    pub fn defined_predicates(&self) -> Vec<&str> {
        self.rules
            .iter()
            .map(|r| r.head.predicate.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn is_probabilistic_profile(&self) -> bool {
        !self.prob_facts.is_empty()
            || !self.annotated_disjunctions.is_empty()
            || !self.evidence.is_empty()
            || !self.prob_queries.is_empty()
            || self.directives.prob_engine.is_some()
            || self.directives.prob_cache.is_some()
    }

    pub fn prob_engine(&self) -> ProbEngine {
        self.directives.prob_engine_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_variable() {
        let term = Term::Variable("X".to_string());
        assert!(term.is_variable());
        assert!(!term.is_constant());
    }

    #[test]
    fn test_term_constant() {
        let term = Term::Integer(42);
        assert!(!term.is_variable());
        assert!(term.is_constant());
    }

    #[test]
    fn test_atom_arity() {
        let atom = Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Integer(1), Term::Integer(2)],
        };
        assert_eq!(atom.arity(), 2);
    }

    #[test]
    fn test_atom_variables() {
        let atom = Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Variable("X".to_string()), Term::Integer(2)],
        };
        let vars = atom.variables();
        assert_eq!(vars, vec!["X"]);
    }

    #[test]
    fn test_rule_is_fact() {
        let fact = Rule {
            head: Atom { predicate: "edge".to_string(), terms: vec![Term::Integer(1), Term::Integer(2)] },
            body: vec![],
        };
        assert!(fact.is_fact());
    }

    #[test]
    fn test_rule_has_negation() {
        let rule = Rule {
            head: Atom { predicate: "isolated".to_string(), terms: vec![Term::Variable("X".to_string())] },
            body: vec![
                BodyLiteral::Positive(Atom { predicate: "node".to_string(), terms: vec![Term::Variable("X".to_string())] }),
                BodyLiteral::Negated(Atom { predicate: "edge".to_string(), terms: vec![Term::Variable("X".to_string()), Term::Variable("Y".to_string())] }),
            ],
        };
        assert!(rule.has_negation());
    }

    #[test]
    fn test_program_facts() {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom { predicate: "edge".to_string(), terms: vec![Term::Integer(1), Term::Integer(2)] },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom { predicate: "reach".to_string(), terms: vec![Term::Variable("X".to_string()), Term::Variable("Y".to_string())] },
            body: vec![BodyLiteral::Positive(Atom { predicate: "edge".to_string(), terms: vec![Term::Variable("X".to_string()), Term::Variable("Y".to_string())] })],
        });
        assert_eq!(program.facts().count(), 1);
        assert_eq!(program.proper_rules().count(), 1);
    }

    #[test]
    fn test_arith_expr_structure() {
        let expr = ArithExpr::Add(
            Box::new(ArithExpr::Variable("X".to_string())),
            Box::new(ArithExpr::Integer(1)),
        );
        assert!(matches!(expr, ArithExpr::Add(_, _)));
    }

    #[test]
    fn test_is_expr_structure() {
        let is_expr = IsExpr {
            target: "Z".to_string(),
            expr: ArithExpr::Variable("Y".to_string()),
        };
        assert_eq!(is_expr.target, "Z");
    }
}
