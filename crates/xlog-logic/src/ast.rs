//! Abstract Syntax Tree for XLOG programs

use xlog_core::ScalarType;

/// A term in an atom
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// Named logic variable (e.g. `X`).
    Variable(String),
    /// Anonymous wildcard `_` -- each occurrence is a fresh unnamed variable.
    Anonymous,
    /// Integer literal.
    Integer(i64),
    /// Floating-point literal.
    Float(f64),
    /// Quoted string literal.
    String(String),
    /// Interned symbol ID -- use `xlog_core::symbol::resolve(id)` to get the string.
    Symbol(u32),
    /// Aggregate expression (e.g. `count(X)`).
    Aggregate(AggExpr),
}

impl Term {
    /// Returns true if this is a named variable.
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

    /// Returns true if this is a ground (non-variable, non-aggregate) term.
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
    /// The aggregation operator.
    pub op: AggOp,
    /// The variable being aggregated.
    pub variable: String,
}

/// Aggregation operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggOp {
    /// Count aggregation.
    Count,
    /// Sum aggregation.
    Sum,
    /// Minimum aggregation.
    Min,
    /// Maximum aggregation.
    Max,
    /// Log-sum-exp aggregation.
    LogSumExp,
}

/// Arithmetic expression tree
#[derive(Debug, Clone, PartialEq)]
pub enum ArithExpr {
    /// Variable reference.
    Variable(String),
    /// Integer literal.
    Integer(i64),
    /// Float literal.
    Float(f64),

    /// Addition.
    Add(Box<ArithExpr>, Box<ArithExpr>),
    /// Subtraction.
    Sub(Box<ArithExpr>, Box<ArithExpr>),
    /// Multiplication.
    Mul(Box<ArithExpr>, Box<ArithExpr>),
    /// Division.
    Div(Box<ArithExpr>, Box<ArithExpr>),
    /// Modulo.
    Mod(Box<ArithExpr>, Box<ArithExpr>),

    /// Absolute value.
    Abs(Box<ArithExpr>),
    /// Minimum of two values.
    Min(Box<ArithExpr>, Box<ArithExpr>),
    /// Maximum of two values.
    Max(Box<ArithExpr>, Box<ArithExpr>),
    /// Power (base, exponent).
    Pow(Box<ArithExpr>, Box<ArithExpr>),

    /// Type cast to the given scalar type.
    Cast(Box<ArithExpr>, ScalarType),

    /// User-defined function call
    FuncCall {
        /// Function name being invoked.
        name: String,
        /// Positional arguments supplied to the function.
        args: Vec<ArithExpr>,
    },

    /// Conditional expression (for expanded function bodies)
    Conditional {
        /// Left operand of the condition.
        cond_left: Box<ArithExpr>,
        /// Comparison operator used in the condition.
        cond_op: CompOp,
        /// Right operand of the condition.
        cond_right: Box<ArithExpr>,
        /// Expression evaluated when the condition is true.
        then_expr: Box<ArithExpr>,
        /// Expression evaluated when the condition is false.
        else_expr: Box<ArithExpr>,
    },
}

impl ArithExpr {
    /// Get all variable names used in this expression
    pub fn variables(&self) -> Vec<&str> {
        match self {
            ArithExpr::Variable(name) => vec![name.as_str()],
            ArithExpr::Integer(_) | ArithExpr::Float(_) => vec![],
            ArithExpr::Add(l, r)
            | ArithExpr::Sub(l, r)
            | ArithExpr::Mul(l, r)
            | ArithExpr::Div(l, r)
            | ArithExpr::Mod(l, r)
            | ArithExpr::Min(l, r)
            | ArithExpr::Max(l, r)
            | ArithExpr::Pow(l, r) => {
                let mut vars = l.variables();
                vars.extend(r.variables());
                vars
            }
            ArithExpr::Abs(e) | ArithExpr::Cast(e, _) => e.variables(),
            ArithExpr::FuncCall { args, .. } => args.iter().flat_map(|a| a.variables()).collect(),
            ArithExpr::Conditional {
                cond_left,
                cond_right,
                then_expr,
                else_expr,
                ..
            } => {
                let mut vars = cond_left.variables();
                vars.extend(cond_right.variables());
                vars.extend(then_expr.variables());
                vars.extend(else_expr.variables());
                vars
            }
        }
    }
}

/// Is-expression for variable binding: Z is X + Y
#[derive(Debug, Clone, PartialEq)]
pub struct IsExpr {
    /// Target variable (must be a fresh, unbound variable).
    pub target: String,
    /// Arithmetic expression to evaluate.
    pub expr: ArithExpr,
}

/// An atom (predicate applied to terms)
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// Predicate name.
    pub predicate: String,
    /// Argument terms.
    pub terms: Vec<Term>,
}

impl Atom {
    /// Number of arguments.
    pub fn arity(&self) -> usize {
        self.terms.len()
    }

    /// Collect all named variables in this atom.
    pub fn variables(&self) -> Vec<&str> {
        self.terms
            .iter()
            .filter_map(|t| t.variable_name())
            .collect()
    }
}

/// Epistemic operator on an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicOp {
    /// Known/believed true in the selected epistemic mode.
    Know,
    /// Possible/consistent in the selected epistemic mode.
    Possible,
}

/// Epistemic atom literal in a rule body.
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicLiteral {
    /// Epistemic operator.
    pub op: EpistemicOp,
    /// Whether this epistemic literal is explicitly negated.
    pub negated: bool,
    /// Atom under the epistemic operator.
    pub atom: Atom,
}

/// Comparison operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// A comparison expression
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// Left operand.
    pub left: Term,
    /// Comparison operator.
    pub op: CompOp,
    /// Right operand.
    pub right: Term,
}

/// A literal in the body of a rule
#[derive(Debug, Clone, PartialEq)]
pub enum BodyLiteral {
    /// Positive atom.
    Positive(Atom),
    /// Negated atom (`not p(...)`).
    Negated(Atom),
    /// Epistemic atom (`know p(...)`, `possible p(...)`, or negated form).
    Epistemic(EpistemicLiteral),
    /// Arithmetic comparison (e.g. `X < Y`).
    Comparison(Comparison),
    /// Is-expression binding (e.g. `Z is X + Y`).
    IsExpr(IsExpr),
}

impl BodyLiteral {
    /// Returns true if this is a positive literal.
    pub fn is_positive(&self) -> bool {
        matches!(self, BodyLiteral::Positive(_))
    }

    /// Returns true if this is a negated literal.
    pub fn is_negated(&self) -> bool {
        matches!(self, BodyLiteral::Negated(_))
    }

    /// Returns the atom if this is a positive or negated literal.
    pub fn atom(&self) -> Option<&Atom> {
        match self {
            BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => Some(a),
            BodyLiteral::Epistemic(lit) => Some(&lit.atom),
            BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => None,
        }
    }

    /// Collect all named variables referenced by this literal.
    pub fn variables(&self) -> Vec<&str> {
        match self {
            BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => a.variables(),
            BodyLiteral::Epistemic(lit) => lit.atom.variables(),
            BodyLiteral::Comparison(c) => {
                let mut vars = vec![];
                if let Some(v) = c.left.variable_name() {
                    vars.push(v);
                }
                if let Some(v) = c.right.variable_name() {
                    vars.push(v);
                }
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
    /// Head atom of the rule.
    pub head: Atom,
    /// Body literals (empty for facts).
    pub body: Vec<BodyLiteral>,
}

impl Rule {
    /// Returns true if this rule is a ground fact (empty body).
    pub fn is_fact(&self) -> bool {
        self.body.is_empty()
    }

    /// Returns true if any body literal is negated.
    pub fn has_negation(&self) -> bool {
        self.body.iter().any(|l| l.is_negated())
    }

    /// Returns true if the head contains an aggregate term.
    pub fn has_aggregation(&self) -> bool {
        self.head
            .terms
            .iter()
            .any(|t| matches!(t, Term::Aggregate(_)))
    }

    /// Collect predicate names from the body.
    pub fn body_predicates(&self) -> Vec<&str> {
        self.body
            .iter()
            .filter_map(|l| l.atom().map(|a| a.predicate.as_str()))
            .collect()
    }

    /// Collect named variables from the head.
    pub fn head_variables(&self) -> Vec<&str> {
        self.head.variables()
    }

    /// Collect all named variables from the body.
    pub fn body_variables(&self) -> Vec<&str> {
        self.body.iter().flat_map(|l| l.variables()).collect()
    }
}

/// A constraint (:- body)
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    /// Body literals whose conjunction must never be satisfiable.
    pub body: Vec<BodyLiteral>,
}

/// A query (`?- atom.`)
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// Query atom.
    pub atom: Atom,
}

/// Probabilistic engine selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbEngine {
    /// Exact inference via d-DNNF compilation.
    ExactDdnnf,
    /// Approximate inference via Monte Carlo sampling.
    Mc,
}

/// Probabilistic compilation caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbCache {
    /// Enable circuit caching.
    On,
    /// Disable circuit caching.
    Off,
}

/// Epistemic semantics mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicMode {
    /// G91 compatibility semantics.
    G91,
    /// Founded Autoepistemic Equilibrium Logic.
    Faeel,
}

/// Compilation/evaluation directives (e.g., `#pragma ...`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Directives {
    /// Override for the probabilistic inference engine.
    pub prob_engine: Option<ProbEngine>,
    /// Override for circuit caching.
    pub prob_cache: Option<ProbCache>,
    /// Maximum UDF recursion depth.
    pub max_recursion_depth: Option<u32>,
    /// Override for epistemic semantics.
    pub epistemic_mode: Option<EpistemicMode>,
}

impl Directives {
    /// Return the configured prob engine, defaulting to ExactDdnnf.
    pub fn prob_engine_or_default(&self) -> ProbEngine {
        self.prob_engine.unwrap_or(ProbEngine::ExactDdnnf)
    }

    /// Return the configured max recursion depth, defaulting to 1000.
    pub fn max_recursion_depth_or_default(&self) -> u32 {
        self.max_recursion_depth.unwrap_or(1000)
    }

    /// Return the configured epistemic mode, defaulting to FAEEL.
    pub fn epistemic_mode_or_default(&self) -> EpistemicMode {
        self.epistemic_mode.unwrap_or(EpistemicMode::Faeel)
    }
}

/// A probabilistic fact (`p::atom.`)
#[derive(Debug, Clone, PartialEq)]
pub struct ProbFact {
    /// Probability weight.
    pub prob: f64,
    /// Ground atom.
    pub atom: Atom,
}

/// Neural predicate declaration
///
/// Neural predicates connect neural networks to probabilistic logic.
/// Syntax: `nn(network, [inputs], output, [labels]) :: pred(args).`
///
/// The neural network produces probability distributions over labels,
/// which become probabilistic facts in the logic program.
///
/// # Examples
/// ```text
/// nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
/// nn(encoder, [Text], Embedding) :: encode(Text, Embedding).
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralPredDecl {
    /// Name of the registered neural network
    pub network: String,
    /// Input variable names (bind to tensor sources)
    pub inputs: Vec<String>,
    /// Output variable name
    pub output: String,
    /// Optional classification labels (for classification networks)
    /// If None, the network produces embeddings
    pub labels: Option<Vec<NeuralLabel>>,
    /// The predicate this neural network defines
    pub predicate: Atom,
}

/// A label in a neural predicate classification
///
/// Labels can be integers or symbols (identifiers).
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralLabel {
    /// Integer label value.
    Integer(i64),
    /// Symbolic (string) label value.
    Symbol(String),
}

/// A learnable rule template parameterized by a named tensor mask.
/// Used for differentiable ILP — the mask selects which (body1, body2, head)
/// combinations are active during execution.
#[derive(Debug, Clone)]
pub struct LearnableRule {
    /// Name of the tensor mask controlling rule activation.
    pub mask_name: String,
    /// Head atom of the rule template.
    pub head: Atom,
    /// Body literals of the rule template.
    pub body: Vec<BodyLiteral>,
}

/// Annotated disjunction (`p1::a1; p2::a2.`)
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotatedDisjunction {
    /// Disjunctive choices with their probability weights.
    pub choices: Vec<ProbFact>,
}

/// Evidence statement (`evidence(atom, true|false).`)
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    /// The observed atom.
    pub atom: Atom,
    /// Whether the atom is observed true or false.
    pub value: bool,
}

/// Probabilistic query statement (`query(atom).`)
#[derive(Debug, Clone, PartialEq)]
pub struct ProbQuery {
    /// The atom whose probability is being queried.
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
    /// Domain name.
    pub name: String,
    /// Scalar type for the domain.
    pub typ: ScalarType,
}

/// Predicate declaration
#[derive(Debug, Clone, PartialEq)]
pub struct PredDecl {
    /// Predicate name.
    pub name: String,
    /// Column types.
    pub types: Vec<ScalarType>,
    /// Whether this predicate is module-private.
    pub is_private: bool,
}

/// Function parameter with optional type annotation
#[derive(Debug, Clone, PartialEq)]
pub struct FuncParam {
    /// Parameter name.
    pub name: String,
    /// Optional type annotation.
    pub typ: Option<ScalarType>,
}

/// Conditional expression: if X < 0 then A else B
#[derive(Debug, Clone, PartialEq)]
pub struct CondExpr {
    /// Left side of condition
    pub cond_left: ArithExpr,
    /// Comparison operator
    pub cond_op: CompOp,
    /// Right side of condition
    pub cond_right: ArithExpr,
    /// Value if condition is true
    pub then_branch: Box<FuncBody>,
    /// Value if condition is false
    pub else_branch: Box<FuncBody>,
}

/// Function body - arithmetic, conditional, or predicate-based
#[derive(Debug, Clone, PartialEq)]
pub enum FuncBody {
    /// Pure arithmetic expression: X * X
    Arithmetic(ArithExpr),
    /// Conditional expression: if X < 0 then ...
    Conditional(CondExpr),
    /// Predicate-based: P :- parent(X, P)
    Predicate {
        /// Result variable
        result: String,
        /// Body literals
        body: Vec<BodyLiteral>,
    },
}

/// User-defined function
#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    /// Function name
    pub name: String,
    /// Parameters
    pub params: Vec<FuncParam>,
    /// Optional return type annotation
    pub return_type: Option<ScalarType>,
    /// Function body
    pub body: FuncBody,
    /// Is this function private?
    pub is_private: bool,
}

/// A complete XLOG program
#[derive(Debug, Clone, Default)]
pub struct Program {
    /// Import declarations (`use ...`).
    pub imports: Vec<UseDecl>,
    /// User-defined function definitions.
    pub functions: Vec<FuncDef>,
    /// Domain declarations.
    pub domains: Vec<DomainDecl>,
    /// Predicate type declarations.
    pub predicates: Vec<PredDecl>,
    /// Rules and facts.
    pub rules: Vec<Rule>,
    /// Integrity constraints (`:- ...`).
    pub constraints: Vec<Constraint>,
    /// Queries (`?- ...`).
    pub queries: Vec<Query>,
    /// Probabilistic facts (`p::atom.`).
    pub prob_facts: Vec<ProbFact>,
    /// Annotated disjunctions.
    pub annotated_disjunctions: Vec<AnnotatedDisjunction>,
    /// Evidence statements.
    pub evidence: Vec<Evidence>,
    /// Probabilistic queries (`query(atom).`).
    pub prob_queries: Vec<ProbQuery>,
    /// Neural predicate declarations.
    pub neural_predicates: Vec<NeuralPredDecl>,
    /// Learnable rule templates (ILP).
    pub learnable_rules: Vec<LearnableRule>,
    /// Compilation directives.
    pub directives: Directives,
}

impl Program {
    /// Create an empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Iterate over ground facts (rules with empty bodies).
    pub fn facts(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter().filter(|r| r.is_fact())
    }

    /// Iterate over proper rules (non-fact rules with bodies).
    pub fn proper_rules(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter().filter(|r| !r.is_fact())
    }

    /// Collect the set of predicate names defined (appearing as rule heads).
    pub fn defined_predicates(&self) -> Vec<&str> {
        self.rules
            .iter()
            .map(|r| r.head.predicate.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Returns true if this program uses probabilistic features.
    pub fn is_probabilistic_profile(&self) -> bool {
        !self.prob_facts.is_empty()
            || !self.annotated_disjunctions.is_empty()
            || !self.evidence.is_empty()
            || !self.prob_queries.is_empty()
            || self.directives.prob_engine.is_some()
            || self.directives.prob_cache.is_some()
    }

    /// Return the probabilistic engine (from directives, or the default).
    pub fn prob_engine(&self) -> ProbEngine {
        self.directives.prob_engine_or_default()
    }

    /// Merge another program's exports into this program.
    /// Used for importing modules - adds predicates, functions, rules from the imported module.
    /// Only merges public items (private items are not exported).
    ///
    /// # Arguments
    /// * `other` - The program to merge from
    /// * `imported_items` - Optional set of specific items to import. If None, imports all public items.
    pub fn merge_from(
        &mut self,
        other: &Program,
        imported_items: Option<&std::collections::HashSet<String>>,
    ) {
        use std::collections::HashSet;

        // Track which predicates are private in the source
        let private_preds: HashSet<&str> = other
            .predicates
            .iter()
            .filter(|p| p.is_private)
            .map(|p| p.name.as_str())
            .collect();

        let _private_funcs: HashSet<&str> = other
            .functions
            .iter()
            .filter(|f| f.is_private)
            .map(|f| f.name.as_str())
            .collect();

        // Merge predicate declarations (only public ones)
        for pred in &other.predicates {
            if pred.is_private {
                continue;
            }
            // Check if this is in the import list (if specified)
            if let Some(items) = imported_items {
                if !items.contains(&pred.name) {
                    continue;
                }
            }
            // Avoid duplicate declarations
            if !self.predicates.iter().any(|p| p.name == pred.name) {
                self.predicates.push(pred.clone());
            }
        }

        // Merge functions (only public ones)
        for func in &other.functions {
            if func.is_private {
                continue;
            }
            if let Some(items) = imported_items {
                if !items.contains(&func.name) {
                    continue;
                }
            }
            // Avoid duplicate functions
            if !self.functions.iter().any(|f| f.name == func.name) {
                self.functions.push(func.clone());
            }
        }

        // Merge rules (facts and rules for public predicates)
        for rule in &other.rules {
            // Skip if the head predicate is private
            if private_preds.contains(rule.head.predicate.as_str()) {
                continue;
            }
            // Check import list for facts/rules
            if let Some(items) = imported_items {
                if !items.contains(&rule.head.predicate) {
                    continue;
                }
            }
            self.rules.push(rule.clone());
        }

        // Merge domains
        for domain in &other.domains {
            if !self.domains.iter().any(|d| d.name == domain.name) {
                self.domains.push(domain.clone());
            }
        }
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
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        };
        assert!(fact.is_fact());
    }

    #[test]
    fn test_rule_has_negation() {
        let rule = Rule {
            head: Atom {
                predicate: "isolated".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "node".to_string(),
                    terms: vec![Term::Variable("X".to_string())],
                }),
                BodyLiteral::Negated(Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("Y".to_string()),
                    ],
                }),
            ],
        };
        assert!(rule.has_negation());
    }

    #[test]
    fn test_program_facts() {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "reach".to_string(),
                terms: vec![
                    Term::Variable("X".to_string()),
                    Term::Variable("Y".to_string()),
                ],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".to_string(),
                terms: vec![
                    Term::Variable("X".to_string()),
                    Term::Variable("Y".to_string()),
                ],
            })],
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
