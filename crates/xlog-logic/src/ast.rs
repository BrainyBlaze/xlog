//! Abstract Syntax Tree for XLOG programs

use xlog_core::{Result, ScalarType, XlogError};

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
    /// Finite list literal.
    List(Vec<Term>),
    /// Finite cons pattern `[Head | Tail]`.
    Cons {
        /// Head term.
        head: Box<Term>,
        /// Tail term.
        tail: Box<Term>,
    },
    /// Finite compound term.
    Compound {
        /// Functor name.
        functor: String,
        /// Compound arguments.
        args: Vec<Term>,
    },
    /// Static predicate reference.
    PredRef(String),
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
        !self.is_any_variable()
            && !matches!(
                self,
                Term::Aggregate(_)
                    | Term::List(_)
                    | Term::Cons { .. }
                    | Term::Compound { .. }
                    | Term::PredRef(_)
            )
    }

    /// Returns the variable name, or None for anonymous/constants
    pub fn variable_name(&self) -> Option<&str> {
        match self {
            Term::Variable(name) => Some(name),
            _ => None,
        }
    }

    /// Infer the scalar storage type used when no predicate declaration
    /// supplies a schema for this term.
    pub fn inferred_scalar_type(&self) -> ScalarType {
        match self {
            Term::Variable(_) | Term::Anonymous => ScalarType::U64,
            Term::Integer(value) => {
                if *value >= 0 && *value <= u32::MAX as i64 {
                    ScalarType::U32
                } else {
                    ScalarType::I64
                }
            }
            Term::Float(_) => ScalarType::F64,
            Term::String(_) | Term::Symbol(_) => ScalarType::Symbol,
            Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
                ScalarType::U64
            }
            Term::Aggregate(aggregate) => aggregate.default_result_type(),
        }
    }

    /// Return all named variables referenced by this term.
    pub fn variables(&self) -> Vec<&str> {
        match self {
            Term::Variable(name) => vec![name.as_str()],
            Term::List(items) => items.iter().flat_map(Term::variables).collect(),
            Term::Cons { head, tail } => {
                let mut vars = head.variables();
                vars.extend(tail.variables());
                vars
            }
            Term::Compound { args, .. } => args.iter().flat_map(Term::variables).collect(),
            Term::Anonymous
            | Term::Integer(_)
            | Term::Float(_)
            | Term::String(_)
            | Term::Symbol(_)
            | Term::PredRef(_)
            | Term::Aggregate(_) => vec![],
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

impl AggExpr {
    /// Return the runtime result type for a known aggregate input type.
    ///
    /// `None` means the execution provider does not support that operator for the
    /// supplied input type.
    pub(crate) fn result_type_for_input(&self, input: ScalarType) -> Option<ScalarType> {
        match self.op {
            AggOp::Count => Some(ScalarType::U64),
            AggOp::Sum if matches!(input, ScalarType::U32 | ScalarType::U64) => {
                Some(ScalarType::U64)
            }
            AggOp::Min | AggOp::Max if matches!(input, ScalarType::U32 | ScalarType::U64) => {
                Some(input)
            }
            AggOp::LogSumExp if input == ScalarType::F64 => Some(ScalarType::F64),
            AggOp::Sum | AggOp::Min | AggOp::Max | AggOp::LogSumExp => None,
        }
    }

    /// Return a result type that is independent of a not-yet-known input.
    pub(crate) fn input_independent_result_type(&self) -> Option<ScalarType> {
        match self.op {
            AggOp::Count | AggOp::Sum => Some(ScalarType::U64),
            AggOp::LogSumExp => Some(ScalarType::F64),
            AggOp::Min | AggOp::Max => None,
        }
    }

    fn default_result_type(&self) -> ScalarType {
        self.input_independent_result_type()
            .unwrap_or(ScalarType::U64)
    }
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
        self.terms.iter().flat_map(Term::variables).collect()
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

/// A finite univ expression (`Term =.. Parts`) in a rule body.
#[derive(Debug, Clone, PartialEq)]
pub struct Univ {
    /// Term side of the univ relation.
    pub term: Term,
    /// Parts-list side of the univ relation.
    pub parts: Term,
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
    /// Finite univ relation (`Term =.. Parts`).
    Univ(Univ),
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
            BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => None,
        }
    }

    /// Collect all named variables referenced by this literal.
    pub fn variables(&self) -> Vec<&str> {
        match self {
            BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => a.variables(),
            BodyLiteral::Epistemic(lit) => lit.atom.variables(),
            BodyLiteral::Comparison(c) => {
                let mut vars = vec![];
                vars.extend(c.left.variables());
                vars.extend(c.right.variables());
                vars
            }
            BodyLiteral::IsExpr(is_expr) => {
                let mut vars = is_expr.expr.variables();
                vars.push(is_expr.target.as_str());
                vars
            }
            BodyLiteral::Univ(univ) => {
                let mut vars = univ.term.variables();
                vars.extend(univ.parts.variables());
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

    /// Infer a head variable's type from ordinary body atoms in source order.
    ///
    /// The caller supplies a column lookup so each compilation context can use
    /// its own predicate identity and decide whether negated atoms provide type
    /// evidence. The first known type from an accepted body occurrence is
    /// returned; other literal kinds are not considered here.
    pub(crate) fn inferred_head_variable_type<F>(
        &self,
        variable: &str,
        mut column_type: F,
    ) -> Option<ScalarType>
    where
        F: FnMut(&Atom, usize, bool) -> Option<ScalarType>,
    {
        for literal in &self.body {
            let (atom, negated) = match literal {
                BodyLiteral::Positive(atom) => (atom, false),
                BodyLiteral::Negated(atom) => (atom, true),
                BodyLiteral::Epistemic(_)
                | BodyLiteral::Comparison(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => continue,
            };
            for (index, term) in atom.terms.iter().enumerate() {
                if matches!(term, Term::Variable(name) if name == variable) {
                    if let Some(typ) = column_type(atom, index, negated) {
                        return Some(typ);
                    }
                }
            }
        }
        None
    }
}

/// A constraint (:- body)
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    /// Stable index of this constraint in the complete authored program.
    pub authored_index: Option<usize>,
    /// Body literals whose conjunction must never be satisfiable.
    pub body: Vec<BodyLiteral>,
}

impl Constraint {
    /// Return the authored identity required by prepared compilation paths.
    pub fn require_authored_index(&self) -> Result<usize> {
        self.authored_index.ok_or_else(|| {
            XlogError::Compilation(
                "prepared constraint compilation requires authored identities".to_string(),
            )
        })
    }
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
    /// Gelfond-1991-style compatibility semantics, selected by `g91`.
    G91,
    /// Founded Autoepistemic Equilibrium Logic.
    Faeel,
}

/// Monte Carlo sampling method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbMethod {
    /// Rejection sampling.
    Rejection,
    /// Forceable evidence clamping.
    EvidenceClamping,
}

/// Magic-set rewrite mode for bound recursive deterministic queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagicSetsMode {
    /// Apply the rewrite when the compiler can prove the supported safe subset.
    Auto,
    /// Require the rewrite and fail with a typed diagnostic if it is unsafe.
    On,
    /// Disable magic-set rewriting.
    Off,
}

/// Compilation/evaluation directives (e.g., `#pragma ...`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Directives {
    /// Override for the probabilistic inference engine.
    pub prob_engine: Option<ProbEngine>,
    /// Override for circuit caching.
    pub prob_cache: Option<ProbCache>,
    /// Monte Carlo sample count.
    pub prob_samples: Option<usize>,
    /// Monte Carlo deterministic RNG seed.
    pub prob_seed: Option<u64>,
    /// Monte Carlo confidence level.
    pub prob_confidence: Option<f64>,
    /// Monte Carlo sampling method.
    pub prob_method: Option<ProbMethod>,
    /// Maximum nonmonotone MC iterations.
    pub prob_max_nonmonotone_iterations: Option<usize>,
    /// Maximum UDF recursion depth.
    pub max_recursion_depth: Option<u32>,
    /// Override for epistemic semantics.
    pub epistemic_mode: Option<EpistemicMode>,
    /// Magic-set rewrite mode.
    pub magic_sets: Option<MagicSetsMode>,
}

impl Directives {
    /// Names of the pragmas explicitly set on this program, using the
    /// source spelling (`#pragma <name> = ...`), in declaration-struct
    /// order.
    pub fn set_pragma_names(&self) -> Vec<&'static str> {
        // Destructure so adding a `Directives` field without extending this
        // list is a compile error, not a pragma that silently vanishes from
        // the ignored-import warnings.
        let Directives {
            prob_engine,
            prob_cache,
            prob_samples,
            prob_seed,
            prob_confidence,
            prob_method,
            prob_max_nonmonotone_iterations,
            max_recursion_depth,
            epistemic_mode,
            magic_sets,
        } = self;
        let mut names = Vec::new();
        if prob_engine.is_some() {
            names.push("prob_engine");
        }
        if prob_cache.is_some() {
            names.push("prob_cache");
        }
        if prob_samples.is_some() {
            names.push("prob_samples");
        }
        if prob_seed.is_some() {
            names.push("prob_seed");
        }
        if prob_confidence.is_some() {
            names.push("prob_confidence");
        }
        if prob_method.is_some() {
            names.push("prob_method");
        }
        if prob_max_nonmonotone_iterations.is_some() {
            names.push("prob_max_nonmonotone_iterations");
        }
        if max_recursion_depth.is_some() {
            names.push("max_recursion_depth");
        }
        if epistemic_mode.is_some() {
            names.push("epistemic_mode");
        }
        if magic_sets.is_some() {
            names.push("magic_sets");
        }
        names
    }

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

    /// Return the configured MC sample count, defaulting to 10000.
    pub fn prob_samples_or_default(&self) -> usize {
        self.prob_samples.unwrap_or(10000)
    }

    /// Return the configured MC seed, defaulting to 0.
    pub fn prob_seed_or_default(&self) -> u64 {
        self.prob_seed.unwrap_or(0)
    }

    /// Return the configured MC confidence, defaulting to 0.95.
    pub fn prob_confidence_or_default(&self) -> f64 {
        self.prob_confidence.unwrap_or(0.95)
    }

    /// Return the configured nonmonotone MC iteration cap, defaulting to 1024.
    pub fn prob_max_nonmonotone_iterations_or_default(&self) -> usize {
        self.prob_max_nonmonotone_iterations.unwrap_or(1024)
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

/// A type reference in source declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeRef {
    /// Built-in scalar type.
    Scalar(ScalarType),
    /// Domain alias resolved during semantic analysis.
    Domain(String),
    /// Finite homogeneous list type.
    List(Box<TypeRef>),
    /// Finite term type.
    Term,
    /// Finite compound term type.
    Compound,
    /// Static predicate reference type.
    PredRef,
}

/// Predicate declaration column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredColumn {
    /// Optional source-level column name.
    pub name: Option<String>,
    /// Column type reference.
    pub typ: TypeRef,
}

/// Predicate declaration
#[derive(Debug, Clone, PartialEq)]
pub struct PredDecl {
    /// Predicate name.
    pub name: String,
    /// Column types.
    pub types: Vec<TypeRef>,
    /// Declared columns, including optional names.
    pub columns: Vec<PredColumn>,
    /// Whether this predicate is module-private.
    pub is_private: bool,
}

impl PredDecl {
    /// Return the effective declared columns, including optional source names.
    ///
    /// Parsed declarations populate `columns`; callers constructing the AST
    /// directly may supply the equivalent unnamed schema through `types`.
    pub fn schema_columns(&self) -> Vec<PredColumn> {
        if self.columns.is_empty() {
            self.types
                .iter()
                .cloned()
                .map(|typ| PredColumn { name: None, typ })
                .collect()
        } else {
            self.columns.clone()
        }
    }

    /// Return the arity of the effective declared schema.
    pub fn arity(&self) -> usize {
        if self.columns.is_empty() {
            self.types.len()
        } else {
            self.columns.len()
        }
    }
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
    /// Number of integrity constraints in the authored source program.
    ///
    /// This is carried through transforms so sparse constraint subsets can
    /// validate their authored identities without being locally re-enumerated.
    pub authored_constraint_source_bound: Option<usize>,
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProgramMergeReport {
    pub domains: Vec<usize>,
    pub predicates: Vec<usize>,
    pub functions: Vec<usize>,
    pub rules: Vec<usize>,
}

impl Program {
    /// Create an empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign or validate stable authored identities before any program transforms.
    pub fn prepare_authored_constraint_identity(
        &mut self,
        authored_source_constraint_count: usize,
    ) -> Result<()> {
        if let Some(existing_bound) = self.authored_constraint_source_bound {
            if existing_bound != authored_source_constraint_count {
                return Err(XlogError::Compilation(format!(
                    "authored constraint source bound {existing_bound} does not match requested bound {authored_source_constraint_count}"
                )));
            }
        }

        let assigned = self
            .constraints
            .iter()
            .filter(|constraint| constraint.authored_index.is_some())
            .count();

        if assigned == 0 {
            if self.constraints.len() != authored_source_constraint_count {
                return Err(XlogError::Compilation(format!(
                    "unassigned constraint count {} does not match authored source bound {}",
                    self.constraints.len(),
                    authored_source_constraint_count
                )));
            }
            for (authored_index, constraint) in self.constraints.iter_mut().enumerate() {
                constraint.authored_index = Some(authored_index);
            }
            self.authored_constraint_source_bound = Some(authored_source_constraint_count);
            return Ok(());
        }

        if assigned != self.constraints.len() {
            return Err(XlogError::Compilation(
                "mixed assigned and unassigned authored constraint identities".to_string(),
            ));
        }

        let mut seen = std::collections::HashSet::with_capacity(self.constraints.len());
        for constraint in &self.constraints {
            let authored_index = constraint
                .authored_index
                .expect("all constraint identities were checked as assigned");
            if authored_index >= authored_source_constraint_count {
                return Err(XlogError::Compilation(format!(
                    "authored constraint index {authored_index} is outside source bound {authored_source_constraint_count}"
                )));
            }
            if !seen.insert(authored_index) {
                return Err(XlogError::Compilation(format!(
                    "duplicate authored constraint index {authored_index}"
                )));
            }
        }
        self.authored_constraint_source_bound = Some(authored_source_constraint_count);
        Ok(())
    }

    /// Assign dense authored identities at the outer full-program boundary.
    pub fn prepare_authored_constraint_identity_at_root(&mut self) -> Result<()> {
        let authored_source_constraint_count = self.constraints.len();
        self.prepare_authored_constraint_identity(authored_source_constraint_count)
    }

    /// Validate identities on a program already prepared at the outer boundary.
    pub fn validate_prepared_authored_constraint_identity(&self) -> Result<()> {
        if self.constraints.is_empty() && self.authored_constraint_source_bound.is_none() {
            return Ok(());
        }
        let authored_source_constraint_count =
            self.authored_constraint_source_bound.ok_or_else(|| {
                XlogError::Compilation(
                "prepared constraint compilation requires authored identities and a source bound"
                    .to_string(),
            )
            })?;
        if self
            .constraints
            .iter()
            .any(|constraint| constraint.authored_index.is_none())
        {
            return Err(XlogError::Compilation(
                "prepared constraint compilation requires authored identities".to_string(),
            ));
        }

        let mut seen = std::collections::HashSet::with_capacity(self.constraints.len());
        for constraint in &self.constraints {
            let authored_index = constraint
                .authored_index
                .expect("all prepared constraint identities were checked as assigned");
            if authored_index >= authored_source_constraint_count {
                return Err(XlogError::Compilation(format!(
                    "authored constraint index {authored_index} is outside source bound {authored_source_constraint_count}"
                )));
            }
            if !seen.insert(authored_index) {
                return Err(XlogError::Compilation(format!(
                    "duplicate authored constraint index {authored_index}"
                )));
            }
        }
        Ok(())
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
            || self.directives.prob_samples.is_some()
            || self.directives.prob_seed.is_some()
            || self.directives.prob_confidence.is_some()
            || self.directives.prob_method.is_some()
            || self.directives.prob_max_nonmonotone_iterations.is_some()
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
        self.merge_from_with_report(other, imported_items);
    }

    pub(crate) fn merge_from_with_report(
        &mut self,
        other: &Program,
        imported_items: Option<&std::collections::HashSet<String>>,
    ) -> ProgramMergeReport {
        use std::collections::HashSet;

        let mut report = ProgramMergeReport::default();

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
        for (source_index, pred) in other.predicates.iter().enumerate() {
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
                report.predicates.push(source_index);
            }
        }

        // Merge functions (only public ones)
        for (source_index, func) in other.functions.iter().enumerate() {
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
                report.functions.push(source_index);
            }
        }

        // Merge rules (facts and rules for public predicates)
        for (source_index, rule) in other.rules.iter().enumerate() {
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
            if !self.rules.iter().any(|existing| existing == rule) {
                self.rules.push(rule.clone());
                report.rules.push(source_index);
            }
        }

        // Merge domains
        for (source_index, domain) in other.domains.iter().enumerate() {
            if !self.domains.iter().any(|d| d.name == domain.name) {
                self.domains.push(domain.clone());
                report.domains.push(source_index);
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directives_set_pragma_names() {
        let mut directives = Directives::default();
        assert!(directives.set_pragma_names().is_empty());

        directives.prob_seed = Some(7);
        directives.magic_sets = Some(MagicSetsMode::Auto);
        assert_eq!(
            directives.set_pragma_names(),
            vec!["prob_seed", "magic_sets"]
        );
    }

    #[test]
    fn test_directives_set_pragma_names_covers_all_ten_pragmas() {
        let directives = Directives {
            prob_engine: Some(ProbEngine::Mc),
            prob_cache: Some(ProbCache::On),
            prob_samples: Some(20000),
            prob_seed: Some(7),
            prob_confidence: Some(0.9),
            prob_method: Some(ProbMethod::Rejection),
            prob_max_nonmonotone_iterations: Some(64),
            max_recursion_depth: Some(100),
            epistemic_mode: Some(EpistemicMode::G91),
            magic_sets: Some(MagicSetsMode::Auto),
        };
        assert_eq!(
            directives.set_pragma_names(),
            vec![
                "prob_engine",
                "prob_cache",
                "prob_samples",
                "prob_seed",
                "prob_confidence",
                "prob_method",
                "prob_max_nonmonotone_iterations",
                "max_recursion_depth",
                "epistemic_mode",
                "magic_sets",
            ]
        );
    }

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
    fn predicate_declaration_uses_its_effective_schema_representation() {
        let types_only = PredDecl {
            name: "types_only".to_string(),
            types: vec![TypeRef::Scalar(ScalarType::U64)],
            columns: vec![],
            is_private: false,
        };
        assert_eq!(types_only.arity(), 1);
        assert_eq!(
            types_only.schema_columns(),
            vec![PredColumn {
                name: None,
                typ: TypeRef::Scalar(ScalarType::U64),
            }]
        );

        let columns_only = PredDecl {
            name: "columns_only".to_string(),
            types: vec![],
            columns: vec![PredColumn {
                name: Some("value".to_string()),
                typ: TypeRef::Scalar(ScalarType::Symbol),
            }],
            is_private: false,
        };
        assert_eq!(columns_only.arity(), 1);
        assert_eq!(
            columns_only.schema_columns(),
            vec![PredColumn {
                name: Some("value".to_string()),
                typ: TypeRef::Scalar(ScalarType::Symbol),
            }]
        );
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
