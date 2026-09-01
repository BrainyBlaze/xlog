#![allow(
    missing_docs,
    reason = "portable extraction DTOs are described by their serialized field names"
)]

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Serialize;
use xlog_core::ScalarType;

use crate::ast::{
    AggOp, ArithExpr, Atom, BodyLiteral, CompOp, EpistemicOp, FuncBody, NeuralLabel, Program,
    ProgramMergeReport, Term, TypeRef,
};
use crate::stratify::analyze_stratification;

use super::{
    ModuleResolver, ResolvedProgramManifest, ResolvedProgramManifestError, ResolvedSourceObject,
    ResolvedSourceObjectKind,
};

const EXTRACTION_SCHEMA_VERSION: &str = "xlog.resolved-program-extraction.v1";

/// Source inventory and executable dependency structure for one resolved program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedProgramExtraction {
    pub schema_version: String,
    pub source_manifest: ResolvedProgramManifest,
    pub executable_program: ExecutableProgram,
}

/// Executable rule graph after module visibility and selection are applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableProgram {
    pub domains: Vec<ExecutableDomain>,
    pub functions: Vec<ExecutableFunction>,
    pub relations: Vec<ExecutableRelation>,
    pub rules: Vec<ExecutableRule>,
    pub constraints: Vec<ExecutableConstraint>,
    pub queries: Vec<ExecutableQuery>,
    pub probabilistic_facts: Vec<ExecutableProbabilisticFact>,
    pub annotated_disjunctions: Vec<ExecutableAnnotatedDisjunction>,
    pub evidence: Vec<ExecutableEvidence>,
    pub probabilistic_queries: Vec<ExecutableProbabilisticQuery>,
    pub neural_predicates: Vec<ExecutableNeuralPredicate>,
    pub learnable_rules: Vec<ExecutableLearnableRule>,
    pub dependencies: Vec<RelationDependency>,
    pub sccs: Vec<ExecutableScc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableScalarType {
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
    Bool,
    Symbol,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutableTypeReference {
    Scalar { scalar_type: ExecutableScalarType },
    Domain { name: String },
    List { element: Box<Self> },
    Term,
    Compound,
    PredicateReference,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutablePredicateColumn {
    pub name: Option<String>,
    pub type_reference: ExecutableTypeReference,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableDomain {
    pub domain_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub name: String,
    pub scalar_type: ExecutableScalarType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableFunctionParameter {
    pub name: String,
    pub scalar_type: Option<ExecutableScalarType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutableFunctionBody {
    Arithmetic {
        expression: ExecutableArithmeticExpression,
    },
    Conditional {
        condition_left: ExecutableArithmeticExpression,
        condition_operator: ComparisonOperator,
        condition_right: ExecutableArithmeticExpression,
        then_body: Box<Self>,
        else_body: Box<Self>,
    },
    Predicate {
        result: String,
        body: Vec<ExecutableBodyLiteral>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableFunction {
    pub function_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub name: String,
    pub parameters: Vec<ExecutableFunctionParameter>,
    pub return_type: Option<ExecutableScalarType>,
    pub body: ExecutableFunctionBody,
    pub is_private: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExecutableProbability {
    pub ieee754_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableProbabilisticFact {
    pub probabilistic_fact_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub probability: ExecutableProbability,
    pub atom: ExecutableAtom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableWeightedAtom {
    pub probability: ExecutableProbability,
    pub atom: ExecutableAtom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableAnnotatedDisjunction {
    pub annotated_disjunction_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub choices: Vec<ExecutableWeightedAtom>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableEvidence {
    pub evidence_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub atom: ExecutableAtom,
    pub value: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableProbabilisticQuery {
    pub probabilistic_query_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub atom: ExecutableAtom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum ExecutableNeuralLabel {
    Integer { value: i64 },
    Symbol { value: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableNeuralPredicate {
    pub neural_predicate_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub network: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub labels: Option<Vec<ExecutableNeuralLabel>>,
    pub predicate: ExecutableAtom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableLearnableRule {
    pub learnable_rule_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub mask_name: String,
    pub head: ExecutableAtom,
    pub body: Vec<ExecutableBodyLiteral>,
}

/// One predicate signature participating in the executable program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableRelation {
    pub relation_id: String,
    pub name: String,
    pub arity: usize,
    pub schema: Option<Vec<ExecutablePredicateColumn>>,
    pub definitions: Vec<ExecutableRelationDefinition>,
    pub declaration_source_object_ids: Vec<String>,
    pub scc_id: Option<String>,
    pub stratum: Option<usize>,
    pub non_monotone: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableRelationDefinition {
    pub source_object_id: String,
    pub kind: ExecutableRelationDefinitionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableRelationDefinitionKind {
    Rule,
    ProbabilisticFact,
    AnnotatedDisjunction,
    NeuralPredicate,
    LearnableRule,
}

/// One source-authored executable fact or rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableRule {
    pub rule_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub head: ExecutableAtom,
    pub body: Vec<ExecutableBodyLiteral>,
    pub scc_id: Option<String>,
    pub stratum: Option<usize>,
}

/// One source-authored integrity constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableConstraint {
    pub constraint_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub body: Vec<ExecutableBodyLiteral>,
}

/// One source-authored deterministic query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableQuery {
    pub query_id: String,
    pub module_id: String,
    pub source_object_id: String,
    pub goal: ExecutableAtom,
}

/// One relation dependency contributed by a rule body literal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RelationDependency {
    pub producer_id: String,
    pub producer_kind: RelationDependencyProducerKind,
    pub dependent_relation_id: String,
    pub dependency_relation_id: String,
    pub body_ordinal: usize,
    pub kind: RelationDependencyKind,
    pub epistemic_operator: Option<EpistemicOperator>,
    pub epistemic_negated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationDependencyProducerKind {
    Rule,
    LearnableRule,
}

/// Semantic kind of a relation dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationDependencyKind {
    Positive,
    Negative,
    Aggregate,
    Epistemic,
}

/// One strongly connected predicate component from the production stratifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableScc {
    pub scc_id: String,
    pub relation_ids: Vec<String>,
    pub non_monotone: bool,
    pub stratum: Option<usize>,
}

/// A predicate application with a stable relation identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutableAtom {
    pub relation_id: String,
    pub name: String,
    pub terms: Vec<ExecutableTerm>,
}

/// Complete source term representation used by executable atoms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutableTerm {
    Variable {
        name: String,
    },
    Anonymous,
    Integer {
        value: i64,
    },
    Float {
        ieee754_bits: u64,
    },
    String {
        value: String,
    },
    Symbol {
        value: String,
    },
    List {
        items: Vec<ExecutableTerm>,
    },
    Cons {
        head: Box<ExecutableTerm>,
        tail: Box<ExecutableTerm>,
    },
    Compound {
        functor: String,
        arguments: Vec<ExecutableTerm>,
    },
    PredicateReference {
        name: String,
    },
    Aggregate {
        operator: AggregateOperator,
        variable: String,
    },
}

/// Aggregate operator preserved from a rule head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregateOperator {
    Count,
    Sum,
    Min,
    Max,
    LogSumExp,
}

/// Complete executable rule-body literal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutableBodyLiteral {
    Positive {
        atom: ExecutableAtom,
    },
    Negative {
        atom: ExecutableAtom,
    },
    Epistemic {
        operator: EpistemicOperator,
        negated: bool,
        atom: ExecutableAtom,
    },
    Comparison {
        left: ExecutableTerm,
        operator: ComparisonOperator,
        right: ExecutableTerm,
    },
    IsExpression {
        target: String,
        expression: ExecutableArithmeticExpression,
    },
    Univ {
        term: ExecutableTerm,
        parts: ExecutableTerm,
    },
}

/// Epistemic operator preserved from the source program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicOperator {
    Know,
    Possible,
}

/// Comparison operator preserved from a body literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

/// Arithmetic expression preserved without string rendering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutableArithmeticExpression {
    Variable {
        name: String,
    },
    Integer {
        value: i64,
    },
    Float {
        ieee754_bits: u64,
    },
    Add {
        left: Box<Self>,
        right: Box<Self>,
    },
    Subtract {
        left: Box<Self>,
        right: Box<Self>,
    },
    Multiply {
        left: Box<Self>,
        right: Box<Self>,
    },
    Divide {
        left: Box<Self>,
        right: Box<Self>,
    },
    Modulo {
        left: Box<Self>,
        right: Box<Self>,
    },
    AbsoluteValue {
        value: Box<Self>,
    },
    Minimum {
        left: Box<Self>,
        right: Box<Self>,
    },
    Maximum {
        left: Box<Self>,
        right: Box<Self>,
    },
    Power {
        base: Box<Self>,
        exponent: Box<Self>,
    },
    Cast {
        value: Box<Self>,
        scalar_type: ExecutableScalarType,
    },
    FunctionCall {
        name: String,
        arguments: Vec<Self>,
    },
    Conditional {
        condition_left: Box<Self>,
        condition_operator: ComparisonOperator,
        condition_right: Box<Self>,
        then_expression: Box<Self>,
        else_expression: Box<Self>,
    },
}

/// Failure while constructing the structured executable program.
#[derive(Debug)]
pub enum ResolvedProgramExtractionError {
    Manifest(ResolvedProgramManifestError),
    ModuleValidation {
        message: String,
    },
    MissingEntry,
    MissingModule {
        path: PathBuf,
    },
    MissingSourceObject {
        module_id: String,
        kind: ResolvedSourceObjectKind,
        ordinal: usize,
    },
    RelationArityConflict {
        name: String,
        first: usize,
        second: usize,
    },
    MissingRelationArity {
        name: String,
    },
}

impl fmt::Display for ResolvedProgramExtractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(formatter, "{error}"),
            Self::ModuleValidation { message } => {
                write!(formatter, "resolved module validation failed: {message}")
            }
            Self::MissingEntry => write!(formatter, "resolver has no loaded entry program"),
            Self::MissingModule { path } => write!(
                formatter,
                "resolved source {} is absent from the source manifest",
                path.display()
            ),
            Self::MissingSourceObject {
                module_id,
                kind,
                ordinal,
            } => write!(
                formatter,
                "module {module_id} has no {kind:?} source object at ordinal {ordinal}"
            ),
            Self::RelationArityConflict {
                name,
                first,
                second,
            } => write!(
                formatter,
                "relation {name} appears with incompatible arities {first} and {second}"
            ),
            Self::MissingRelationArity { name } => {
                write!(formatter, "relation {name} has no recoverable arity")
            }
        }
    }
}

impl std::error::Error for ResolvedProgramExtractionError {}

impl From<ResolvedProgramManifestError> for ResolvedProgramExtractionError {
    fn from(error: ResolvedProgramManifestError) -> Self {
        Self::Manifest(error)
    }
}

#[derive(Debug, Clone)]
struct AstOrigin {
    module_id: String,
    source_object_id: String,
}

struct MergedProgram {
    program: Program,
    domain_origins: Vec<AstOrigin>,
    function_origins: Vec<AstOrigin>,
    predicate_origins: Vec<AstOrigin>,
    rule_origins: Vec<AstOrigin>,
    constraint_origins: Vec<AstOrigin>,
    query_origins: Vec<AstOrigin>,
    probabilistic_fact_origins: Vec<AstOrigin>,
    annotated_disjunction_origins: Vec<AstOrigin>,
    evidence_origins: Vec<AstOrigin>,
    probabilistic_query_origins: Vec<AstOrigin>,
    neural_predicate_origins: Vec<AstOrigin>,
    learnable_rule_origins: Vec<AstOrigin>,
}

struct OriginCatalog {
    modules: BTreeMap<PathBuf, OriginModule>,
}

struct OriginModule {
    module_id: String,
    objects: BTreeMap<ResolvedSourceObjectKind, Vec<ResolvedSourceObject>>,
}

impl OriginCatalog {
    fn new(
        source_root: &Path,
        manifest: &ResolvedProgramManifest,
    ) -> Result<Self, ResolvedProgramExtractionError> {
        let source_root = std::fs::canonicalize(source_root).map_err(|error| {
            ResolvedProgramExtractionError::ModuleValidation {
                message: format!("cannot canonicalize source root: {error}"),
            }
        })?;
        let mut modules = BTreeMap::new();
        for module in &manifest.modules {
            let source =
                std::fs::canonicalize(source_root.join(&module.source_path)).map_err(|error| {
                    ResolvedProgramExtractionError::ModuleValidation {
                        message: format!(
                            "cannot canonicalize resolved source {}: {error}",
                            module.source_path
                        ),
                    }
                })?;
            let mut objects = BTreeMap::<ResolvedSourceObjectKind, Vec<_>>::new();
            for object in &module.source_objects {
                objects.entry(object.kind).or_default().push(object.clone());
            }
            modules.insert(
                source,
                OriginModule {
                    module_id: module.module_id.clone(),
                    objects,
                },
            );
        }
        Ok(Self { modules })
    }

    fn origin(
        &self,
        source: &Path,
        kind: ResolvedSourceObjectKind,
        ordinal: usize,
    ) -> Result<AstOrigin, ResolvedProgramExtractionError> {
        let module = self.modules.get(source).ok_or_else(|| {
            ResolvedProgramExtractionError::MissingModule {
                path: source.to_path_buf(),
            }
        })?;
        let object = module
            .objects
            .get(&kind)
            .and_then(|objects| objects.get(ordinal))
            .ok_or_else(|| ResolvedProgramExtractionError::MissingSourceObject {
                module_id: module.module_id.clone(),
                kind,
                ordinal,
            })?;
        Ok(AstOrigin {
            module_id: module.module_id.clone(),
            source_object_id: object.object_id.clone(),
        })
    }
}

impl ModuleResolver {
    /// Extract the exact executable program selected by this resolved entry closure.
    pub fn resolved_program_extraction(
        &self,
        source_root: &Path,
    ) -> Result<ResolvedProgramExtraction, ResolvedProgramExtractionError> {
        let source_manifest = self.resolved_program_manifest(source_root)?;
        let catalog = OriginCatalog::new(source_root, &source_manifest)?;
        let merged = self.merge_program_with_origins(&catalog)?;
        let executable_program = build_executable_program(merged)?;
        Ok(ResolvedProgramExtraction {
            schema_version: EXTRACTION_SCHEMA_VERSION.to_string(),
            source_manifest,
            executable_program,
        })
    }

    fn merge_program_with_origins(
        &self,
        catalog: &OriginCatalog,
    ) -> Result<MergedProgram, ResolvedProgramExtractionError> {
        let entry = self
            .entry
            .as_ref()
            .ok_or(ResolvedProgramExtractionError::MissingEntry)?;
        let entry_source = self
            .entry_source
            .as_ref()
            .ok_or(ResolvedProgramExtractionError::MissingEntry)?;
        let mut program = entry.program.clone();
        let imports = self
            .resolved_imports_for_program(&program)
            .map_err(|error| ResolvedProgramExtractionError::ModuleValidation {
                message: error.to_string(),
            })?;
        self.validate_resolved_imports(&imports).map_err(|error| {
            ResolvedProgramExtractionError::ModuleValidation {
                message: error.to_string(),
            }
        })?;
        self.validate_program_against_imports(&program, &imports)
            .map_err(|error| ResolvedProgramExtractionError::ModuleValidation {
                message: error.to_string(),
            })?;

        let entry_rules = std::mem::take(&mut program.rules);
        let mut domain_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::Domain,
            program.domains.len(),
        )?;
        let mut function_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::Function,
            program.functions.len(),
        )?;
        let mut predicate_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::Predicate,
            program.predicates.len(),
        )?;
        let constraint_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::Constraint,
            program.constraints.len(),
        )?;
        let query_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::Query,
            program.queries.len(),
        )?;
        let probabilistic_fact_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::ProbabilisticFact,
            program.prob_facts.len(),
        )?;
        let annotated_disjunction_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::AnnotatedDisjunction,
            program.annotated_disjunctions.len(),
        )?;
        let evidence_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::Evidence,
            program.evidence.len(),
        )?;
        let probabilistic_query_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::ProbabilisticQuery,
            program.prob_queries.len(),
        )?;
        let neural_predicate_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::NeuralPredicate,
            program.neural_predicates.len(),
        )?;
        let learnable_rule_origins = origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::LearnableRule,
            program.learnable_rules.len(),
        )?;

        let mut merge_reports = Vec::<(PathBuf, ProgramMergeReport)>::new();
        let mut merged_imports = HashSet::new();
        self.merge_import_group_with_report(
            &mut program,
            &imports,
            &mut merged_imports,
            &mut |source, report| merge_reports.push((source.to_path_buf(), report.clone())),
        )
        .map_err(|error| ResolvedProgramExtractionError::ModuleValidation {
            message: error.to_string(),
        })?;

        let mut rule_origins = Vec::new();
        for (source, report) in merge_reports {
            for ordinal in report.domains {
                domain_origins.push(catalog.origin(
                    &source,
                    ResolvedSourceObjectKind::Domain,
                    ordinal,
                )?);
            }
            for ordinal in report.functions {
                function_origins.push(catalog.origin(
                    &source,
                    ResolvedSourceObjectKind::Function,
                    ordinal,
                )?);
            }
            for ordinal in report.predicates {
                predicate_origins.push(catalog.origin(
                    &source,
                    ResolvedSourceObjectKind::Predicate,
                    ordinal,
                )?);
            }
            for ordinal in report.rules {
                rule_origins.push(catalog.origin(
                    &source,
                    ResolvedSourceObjectKind::Rule,
                    ordinal,
                )?);
            }
        }
        program.rules.extend(entry_rules);
        rule_origins.extend(origins_for_count(
            catalog,
            entry_source,
            ResolvedSourceObjectKind::Rule,
            entry.program.rules.len(),
        )?);

        Ok(MergedProgram {
            program,
            domain_origins,
            function_origins,
            predicate_origins,
            rule_origins,
            constraint_origins,
            query_origins,
            probabilistic_fact_origins,
            annotated_disjunction_origins,
            evidence_origins,
            probabilistic_query_origins,
            neural_predicate_origins,
            learnable_rule_origins,
        })
    }
}

fn origins_for_count(
    catalog: &OriginCatalog,
    source: &Path,
    kind: ResolvedSourceObjectKind,
    count: usize,
) -> Result<Vec<AstOrigin>, ResolvedProgramExtractionError> {
    (0..count)
        .map(|ordinal| catalog.origin(source, kind, ordinal))
        .collect()
}

fn build_executable_program(
    merged: MergedProgram,
) -> Result<ExecutableProgram, ResolvedProgramExtractionError> {
    let analysis = analyze_stratification(&merged.program);
    let mut scc_by_predicate = BTreeMap::new();
    let mut sccs = Vec::with_capacity(analysis.sccs.len());
    for (index, predicates) in analysis.sccs.iter().enumerate() {
        let scc_id = format!("scc:{index}");
        let non_monotone = analysis.non_monotone_sccs.contains(&index);
        let stratum = predicates
            .iter()
            .filter_map(|predicate| analysis.strata.get(predicate).copied())
            .next();
        let mut relation_ids = predicates
            .iter()
            .map(|predicate| {
                relation_arity(&merged.program, predicate)
                    .map(|arity| relation_id(predicate, arity))
            })
            .collect::<Result<Vec<_>, _>>()?;
        relation_ids.sort();
        for predicate in predicates {
            scc_by_predicate.insert(predicate.clone(), (scc_id.clone(), stratum, non_monotone));
        }
        sccs.push(ExecutableScc {
            scc_id,
            relation_ids,
            non_monotone,
            stratum,
        });
    }

    let mut rules = Vec::with_capacity(merged.program.rules.len());
    let mut dependencies = Vec::new();
    for (rule, origin) in merged.program.rules.iter().zip(&merged.rule_origins) {
        let head = executable_atom(&rule.head);
        let (scc_id, stratum, _) = scc_by_predicate
            .get(&rule.head.predicate)
            .cloned()
            .map_or((None, None, false), |(scc_id, stratum, non_monotone)| {
                (Some(scc_id), stratum, non_monotone)
            });
        let rule_id = origin.source_object_id.clone();
        for (body_ordinal, literal) in rule.body.iter().enumerate() {
            if let Some(dependency) = relation_dependency(
                &rule_id,
                RelationDependencyProducerKind::Rule,
                &head.relation_id,
                body_ordinal,
                rule.has_aggregation(),
                literal,
            ) {
                dependencies.push(dependency);
            }
        }
        rules.push(ExecutableRule {
            rule_id,
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            head,
            body: rule.body.iter().map(executable_literal).collect(),
            scc_id,
            stratum,
        });
    }

    let constraints = merged
        .program
        .constraints
        .iter()
        .zip(&merged.constraint_origins)
        .map(|(constraint, origin)| ExecutableConstraint {
            constraint_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            body: constraint.body.iter().map(executable_literal).collect(),
        })
        .collect();
    let queries = merged
        .program
        .queries
        .iter()
        .zip(&merged.query_origins)
        .map(|(query, origin)| ExecutableQuery {
            query_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            goal: executable_atom(&query.atom),
        })
        .collect();
    let domains = merged
        .program
        .domains
        .iter()
        .zip(&merged.domain_origins)
        .map(|(domain, origin)| ExecutableDomain {
            domain_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            name: domain.name.clone(),
            scalar_type: executable_scalar_type(domain.typ),
        })
        .collect();
    let functions = merged
        .program
        .functions
        .iter()
        .zip(&merged.function_origins)
        .map(|(function, origin)| ExecutableFunction {
            function_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            name: function.name.clone(),
            parameters: function
                .params
                .iter()
                .map(|parameter| ExecutableFunctionParameter {
                    name: parameter.name.clone(),
                    scalar_type: parameter.typ.map(executable_scalar_type),
                })
                .collect(),
            return_type: function.return_type.map(executable_scalar_type),
            body: executable_function_body(&function.body),
            is_private: function.is_private,
        })
        .collect();
    let probabilistic_facts = merged
        .program
        .prob_facts
        .iter()
        .zip(&merged.probabilistic_fact_origins)
        .map(|(fact, origin)| ExecutableProbabilisticFact {
            probabilistic_fact_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            probability: executable_probability(fact.prob),
            atom: executable_atom(&fact.atom),
        })
        .collect();
    let annotated_disjunctions = merged
        .program
        .annotated_disjunctions
        .iter()
        .zip(&merged.annotated_disjunction_origins)
        .map(|(disjunction, origin)| ExecutableAnnotatedDisjunction {
            annotated_disjunction_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            choices: disjunction
                .choices
                .iter()
                .map(|choice| ExecutableWeightedAtom {
                    probability: executable_probability(choice.prob),
                    atom: executable_atom(&choice.atom),
                })
                .collect(),
        })
        .collect();
    let evidence = merged
        .program
        .evidence
        .iter()
        .zip(&merged.evidence_origins)
        .map(|(evidence, origin)| ExecutableEvidence {
            evidence_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            atom: executable_atom(&evidence.atom),
            value: evidence.value,
        })
        .collect();
    let probabilistic_queries = merged
        .program
        .prob_queries
        .iter()
        .zip(&merged.probabilistic_query_origins)
        .map(|(query, origin)| ExecutableProbabilisticQuery {
            probabilistic_query_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            atom: executable_atom(&query.atom),
        })
        .collect();
    let neural_predicates = merged
        .program
        .neural_predicates
        .iter()
        .zip(&merged.neural_predicate_origins)
        .map(|(declaration, origin)| ExecutableNeuralPredicate {
            neural_predicate_id: origin.source_object_id.clone(),
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            network: declaration.network.clone(),
            inputs: declaration.inputs.clone(),
            output: declaration.output.clone(),
            labels: declaration
                .labels
                .as_ref()
                .map(|labels| labels.iter().map(executable_neural_label).collect()),
            predicate: executable_atom(&declaration.predicate),
        })
        .collect();
    let mut learnable_rules = Vec::with_capacity(merged.program.learnable_rules.len());
    for (rule, origin) in merged
        .program
        .learnable_rules
        .iter()
        .zip(&merged.learnable_rule_origins)
    {
        let rule_id = origin.source_object_id.clone();
        let head = executable_atom(&rule.head);
        let has_aggregation = rule
            .head
            .terms
            .iter()
            .any(|term| matches!(term, Term::Aggregate { .. }));
        for (body_ordinal, literal) in rule.body.iter().enumerate() {
            if let Some(dependency) = relation_dependency(
                &rule_id,
                RelationDependencyProducerKind::LearnableRule,
                &head.relation_id,
                body_ordinal,
                has_aggregation,
                literal,
            ) {
                dependencies.push(dependency);
            }
        }
        learnable_rules.push(ExecutableLearnableRule {
            learnable_rule_id: rule_id,
            module_id: origin.module_id.clone(),
            source_object_id: origin.source_object_id.clone(),
            mask_name: rule.mask_name.clone(),
            head,
            body: rule.body.iter().map(executable_literal).collect(),
        });
    }
    let relations = build_relations(&merged, &rules, &scc_by_predicate)?;

    Ok(ExecutableProgram {
        domains,
        functions,
        relations,
        rules,
        constraints,
        queries,
        probabilistic_facts,
        annotated_disjunctions,
        evidence,
        probabilistic_queries,
        neural_predicates,
        learnable_rules,
        dependencies,
        sccs,
    })
}

fn build_relations(
    merged: &MergedProgram,
    rules: &[ExecutableRule],
    scc_by_predicate: &BTreeMap<String, (String, Option<usize>, bool)>,
) -> Result<Vec<ExecutableRelation>, ResolvedProgramExtractionError> {
    let mut arities = BTreeMap::<String, usize>::new();
    for declaration in &merged.program.predicates {
        register_arity(
            &mut arities,
            &declaration.name,
            declaration.schema_columns().len(),
        )?;
    }
    for rule in &merged.program.rules {
        register_atom_arities(&mut arities, &rule.head)?;
        for literal in &rule.body {
            if let Some(atom) = literal.atom() {
                register_atom_arities(&mut arities, atom)?;
            } else if let BodyLiteral::Epistemic(literal) = literal {
                register_atom_arities(&mut arities, &literal.atom)?;
            }
        }
    }
    for constraint in &merged.program.constraints {
        for literal in &constraint.body {
            if let Some(atom) = literal.atom() {
                register_atom_arities(&mut arities, atom)?;
            } else if let BodyLiteral::Epistemic(literal) = literal {
                register_atom_arities(&mut arities, &literal.atom)?;
            }
        }
    }
    for query in &merged.program.queries {
        register_atom_arities(&mut arities, &query.atom)?;
    }
    for fact in &merged.program.prob_facts {
        register_atom_arities(&mut arities, &fact.atom)?;
    }
    for disjunction in &merged.program.annotated_disjunctions {
        for choice in &disjunction.choices {
            register_atom_arities(&mut arities, &choice.atom)?;
        }
    }
    for evidence in &merged.program.evidence {
        register_atom_arities(&mut arities, &evidence.atom)?;
    }
    for query in &merged.program.prob_queries {
        register_atom_arities(&mut arities, &query.atom)?;
    }
    for declaration in &merged.program.neural_predicates {
        register_atom_arities(&mut arities, &declaration.predicate)?;
    }
    for rule in &merged.program.learnable_rules {
        register_atom_arities(&mut arities, &rule.head)?;
        for literal in &rule.body {
            if let Some(atom) = literal.atom() {
                register_atom_arities(&mut arities, atom)?;
            } else if let BodyLiteral::Epistemic(literal) = literal {
                register_atom_arities(&mut arities, &literal.atom)?;
            }
        }
    }

    let mut definitions = BTreeMap::<String, Vec<ExecutableRelationDefinition>>::new();
    for rule in rules {
        definitions
            .entry(rule.head.name.clone())
            .or_default()
            .push(ExecutableRelationDefinition {
                source_object_id: rule.rule_id.clone(),
                kind: ExecutableRelationDefinitionKind::Rule,
            });
    }
    for (fact, origin) in merged
        .program
        .prob_facts
        .iter()
        .zip(&merged.probabilistic_fact_origins)
    {
        definitions
            .entry(fact.atom.predicate.clone())
            .or_default()
            .push(ExecutableRelationDefinition {
                source_object_id: origin.source_object_id.clone(),
                kind: ExecutableRelationDefinitionKind::ProbabilisticFact,
            });
    }
    for (disjunction, origin) in merged
        .program
        .annotated_disjunctions
        .iter()
        .zip(&merged.annotated_disjunction_origins)
    {
        for choice in &disjunction.choices {
            definitions
                .entry(choice.atom.predicate.clone())
                .or_default()
                .push(ExecutableRelationDefinition {
                    source_object_id: origin.source_object_id.clone(),
                    kind: ExecutableRelationDefinitionKind::AnnotatedDisjunction,
                });
        }
    }
    for (declaration, origin) in merged
        .program
        .neural_predicates
        .iter()
        .zip(&merged.neural_predicate_origins)
    {
        definitions
            .entry(declaration.predicate.predicate.clone())
            .or_default()
            .push(ExecutableRelationDefinition {
                source_object_id: origin.source_object_id.clone(),
                kind: ExecutableRelationDefinitionKind::NeuralPredicate,
            });
    }
    for (rule, origin) in merged
        .program
        .learnable_rules
        .iter()
        .zip(&merged.learnable_rule_origins)
    {
        definitions
            .entry(rule.head.predicate.clone())
            .or_default()
            .push(ExecutableRelationDefinition {
                source_object_id: origin.source_object_id.clone(),
                kind: ExecutableRelationDefinitionKind::LearnableRule,
            });
    }
    let mut declarations = BTreeMap::<String, Vec<String>>::new();
    let mut schemas = BTreeMap::<String, Vec<ExecutablePredicateColumn>>::new();
    for (declaration, origin) in merged
        .program
        .predicates
        .iter()
        .zip(&merged.predicate_origins)
    {
        declarations
            .entry(declaration.name.clone())
            .or_default()
            .push(origin.source_object_id.clone());
        schemas.entry(declaration.name.clone()).or_insert_with(|| {
            declaration
                .schema_columns()
                .iter()
                .map(|column| ExecutablePredicateColumn {
                    name: column.name.clone(),
                    type_reference: executable_type_reference(&column.typ),
                })
                .collect()
        });
    }

    Ok(arities
        .into_iter()
        .map(|(name, arity)| {
            let relation_id = relation_id(&name, arity);
            let (scc_id, stratum, non_monotone) = scc_by_predicate
                .get(&name)
                .cloned()
                .map_or((None, None, false), |(id, stratum, non_monotone)| {
                    (Some(id), stratum, non_monotone)
                });
            ExecutableRelation {
                relation_id,
                name: name.clone(),
                arity,
                schema: schemas.remove(&name),
                definitions: definitions.remove(&name).unwrap_or_default(),
                declaration_source_object_ids: declarations.remove(&name).unwrap_or_default(),
                scc_id,
                stratum,
                non_monotone,
            }
        })
        .collect())
}

fn register_atom_arities(
    arities: &mut BTreeMap<String, usize>,
    atom: &Atom,
) -> Result<(), ResolvedProgramExtractionError> {
    register_arity(arities, &atom.predicate, atom.terms.len())
}

fn register_arity(
    arities: &mut BTreeMap<String, usize>,
    name: &str,
    arity: usize,
) -> Result<(), ResolvedProgramExtractionError> {
    if let Some(existing) = arities.insert(name.to_string(), arity) {
        if existing != arity {
            return Err(ResolvedProgramExtractionError::RelationArityConflict {
                name: name.to_string(),
                first: existing,
                second: arity,
            });
        }
    }
    Ok(())
}

fn relation_arity(
    program: &Program,
    predicate: &str,
) -> Result<usize, ResolvedProgramExtractionError> {
    let defined_arity = program
        .rules
        .iter()
        .map(|rule| &rule.head)
        .chain(program.prob_facts.iter().map(|fact| &fact.atom))
        .chain(
            program
                .annotated_disjunctions
                .iter()
                .flat_map(|disjunction| disjunction.choices.iter().map(|choice| &choice.atom)),
        )
        .chain(
            program
                .neural_predicates
                .iter()
                .map(|declaration| &declaration.predicate),
        )
        .chain(program.learnable_rules.iter().map(|rule| &rule.head))
        .find(|atom| atom.predicate == predicate)
        .map(|atom| atom.terms.len());
    let declared_arity = program
        .predicates
        .iter()
        .find(|declaration| declaration.name == predicate)
        .map(|declaration| declaration.schema_columns().len());
    let referenced_arity = program
        .rules
        .iter()
        .flat_map(|rule| &rule.body)
        .chain(
            program
                .constraints
                .iter()
                .flat_map(|constraint| &constraint.body),
        )
        .chain(program.learnable_rules.iter().flat_map(|rule| &rule.body))
        .find_map(|literal| literal_atom_arity(literal, predicate));
    let observed_arity = program
        .queries
        .iter()
        .map(|query| &query.atom)
        .chain(program.evidence.iter().map(|evidence| &evidence.atom))
        .chain(program.prob_queries.iter().map(|query| &query.atom))
        .find(|atom| atom.predicate == predicate)
        .map(|atom| atom.terms.len());

    defined_arity
        .or(declared_arity)
        .or(referenced_arity)
        .or(observed_arity)
        .ok_or_else(|| ResolvedProgramExtractionError::MissingRelationArity {
            name: predicate.to_string(),
        })
}

fn literal_atom_arity(literal: &BodyLiteral, predicate: &str) -> Option<usize> {
    match literal {
        BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) if atom.predicate == predicate => {
            Some(atom.terms.len())
        }
        BodyLiteral::Epistemic(literal) if literal.atom.predicate == predicate => {
            Some(literal.atom.terms.len())
        }
        _ => None,
    }
}

fn relation_dependency(
    producer_id: &str,
    producer_kind: RelationDependencyProducerKind,
    dependent_relation_id: &str,
    body_ordinal: usize,
    aggregate_head: bool,
    literal: &BodyLiteral,
) -> Option<RelationDependency> {
    let (atom, kind, epistemic_operator, epistemic_negated) = match literal {
        BodyLiteral::Positive(atom) => (
            atom,
            if aggregate_head {
                RelationDependencyKind::Aggregate
            } else {
                RelationDependencyKind::Positive
            },
            None,
            false,
        ),
        BodyLiteral::Negated(atom) => (atom, RelationDependencyKind::Negative, None, false),
        BodyLiteral::Epistemic(literal) => (
            &literal.atom,
            RelationDependencyKind::Epistemic,
            Some(epistemic_operator(literal.op)),
            literal.negated,
        ),
        BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {
            return None;
        }
    };
    Some(RelationDependency {
        producer_id: producer_id.to_string(),
        producer_kind,
        dependent_relation_id: dependent_relation_id.to_string(),
        dependency_relation_id: relation_id(&atom.predicate, atom.terms.len()),
        body_ordinal,
        kind,
        epistemic_operator,
        epistemic_negated,
    })
}

fn executable_literal(literal: &BodyLiteral) -> ExecutableBodyLiteral {
    match literal {
        BodyLiteral::Positive(atom) => ExecutableBodyLiteral::Positive {
            atom: executable_atom(atom),
        },
        BodyLiteral::Negated(atom) => ExecutableBodyLiteral::Negative {
            atom: executable_atom(atom),
        },
        BodyLiteral::Epistemic(literal) => ExecutableBodyLiteral::Epistemic {
            operator: epistemic_operator(literal.op),
            negated: literal.negated,
            atom: executable_atom(&literal.atom),
        },
        BodyLiteral::Comparison(comparison) => ExecutableBodyLiteral::Comparison {
            left: executable_term(&comparison.left),
            operator: comparison_operator(comparison.op),
            right: executable_term(&comparison.right),
        },
        BodyLiteral::IsExpr(expression) => ExecutableBodyLiteral::IsExpression {
            target: expression.target.clone(),
            expression: executable_arithmetic_expression(&expression.expr),
        },
        BodyLiteral::Univ(univ) => ExecutableBodyLiteral::Univ {
            term: executable_term(&univ.term),
            parts: executable_term(&univ.parts),
        },
    }
}

fn executable_scalar_type(scalar_type: ScalarType) -> ExecutableScalarType {
    match scalar_type {
        ScalarType::U32 => ExecutableScalarType::U32,
        ScalarType::U64 => ExecutableScalarType::U64,
        ScalarType::I32 => ExecutableScalarType::I32,
        ScalarType::I64 => ExecutableScalarType::I64,
        ScalarType::F32 => ExecutableScalarType::F32,
        ScalarType::F64 => ExecutableScalarType::F64,
        ScalarType::Bool => ExecutableScalarType::Bool,
        ScalarType::Symbol => ExecutableScalarType::Symbol,
    }
}

fn executable_type_reference(type_reference: &TypeRef) -> ExecutableTypeReference {
    match type_reference {
        TypeRef::Scalar(scalar_type) => ExecutableTypeReference::Scalar {
            scalar_type: executable_scalar_type(*scalar_type),
        },
        TypeRef::Domain(name) => ExecutableTypeReference::Domain { name: name.clone() },
        TypeRef::List(element) => ExecutableTypeReference::List {
            element: Box::new(executable_type_reference(element)),
        },
        TypeRef::Term => ExecutableTypeReference::Term,
        TypeRef::Compound => ExecutableTypeReference::Compound,
        TypeRef::PredRef => ExecutableTypeReference::PredicateReference,
    }
}

fn executable_function_body(body: &FuncBody) -> ExecutableFunctionBody {
    match body {
        FuncBody::Arithmetic(expression) => ExecutableFunctionBody::Arithmetic {
            expression: executable_arithmetic_expression(expression),
        },
        FuncBody::Conditional(conditional) => ExecutableFunctionBody::Conditional {
            condition_left: executable_arithmetic_expression(&conditional.cond_left),
            condition_operator: comparison_operator(conditional.cond_op),
            condition_right: executable_arithmetic_expression(&conditional.cond_right),
            then_body: Box::new(executable_function_body(&conditional.then_branch)),
            else_body: Box::new(executable_function_body(&conditional.else_branch)),
        },
        FuncBody::Predicate { result, body } => ExecutableFunctionBody::Predicate {
            result: result.clone(),
            body: body.iter().map(executable_literal).collect(),
        },
    }
}

fn executable_atom(atom: &Atom) -> ExecutableAtom {
    ExecutableAtom {
        relation_id: relation_id(&atom.predicate, atom.terms.len()),
        name: atom.predicate.clone(),
        terms: atom.terms.iter().map(executable_term).collect(),
    }
}

fn executable_probability(probability: f64) -> ExecutableProbability {
    ExecutableProbability {
        ieee754_bits: probability.to_bits(),
    }
}

fn executable_neural_label(label: &NeuralLabel) -> ExecutableNeuralLabel {
    match label {
        NeuralLabel::Integer(value) => ExecutableNeuralLabel::Integer { value: *value },
        NeuralLabel::Symbol(value) => ExecutableNeuralLabel::Symbol {
            value: value.clone(),
        },
    }
}

fn executable_term(term: &Term) -> ExecutableTerm {
    match term {
        Term::Variable(name) => ExecutableTerm::Variable { name: name.clone() },
        Term::Anonymous => ExecutableTerm::Anonymous,
        Term::Integer(value) => ExecutableTerm::Integer { value: *value },
        Term::Float(value) => ExecutableTerm::Float {
            ieee754_bits: value.to_bits(),
        },
        Term::String(value) => ExecutableTerm::String {
            value: value.clone(),
        },
        Term::Symbol(id) => ExecutableTerm::Symbol {
            value: xlog_core::symbol::resolve(*id),
        },
        Term::List(items) => ExecutableTerm::List {
            items: items.iter().map(executable_term).collect(),
        },
        Term::Cons { head, tail } => ExecutableTerm::Cons {
            head: Box::new(executable_term(head)),
            tail: Box::new(executable_term(tail)),
        },
        Term::Compound { functor, args } => ExecutableTerm::Compound {
            functor: functor.clone(),
            arguments: args.iter().map(executable_term).collect(),
        },
        Term::PredRef(name) => ExecutableTerm::PredicateReference { name: name.clone() },
        Term::Aggregate(aggregate) => ExecutableTerm::Aggregate {
            operator: aggregate_operator(aggregate.op),
            variable: aggregate.variable.clone(),
        },
    }
}

fn executable_arithmetic_expression(expression: &ArithExpr) -> ExecutableArithmeticExpression {
    match expression {
        ArithExpr::Variable(name) => {
            ExecutableArithmeticExpression::Variable { name: name.clone() }
        }
        ArithExpr::Integer(value) => ExecutableArithmeticExpression::Integer { value: *value },
        ArithExpr::Float(value) => ExecutableArithmeticExpression::Float {
            ieee754_bits: value.to_bits(),
        },
        ArithExpr::Add(left, right) => ExecutableArithmeticExpression::Add {
            left: Box::new(executable_arithmetic_expression(left)),
            right: Box::new(executable_arithmetic_expression(right)),
        },
        ArithExpr::Sub(left, right) => ExecutableArithmeticExpression::Subtract {
            left: Box::new(executable_arithmetic_expression(left)),
            right: Box::new(executable_arithmetic_expression(right)),
        },
        ArithExpr::Mul(left, right) => ExecutableArithmeticExpression::Multiply {
            left: Box::new(executable_arithmetic_expression(left)),
            right: Box::new(executable_arithmetic_expression(right)),
        },
        ArithExpr::Div(left, right) => ExecutableArithmeticExpression::Divide {
            left: Box::new(executable_arithmetic_expression(left)),
            right: Box::new(executable_arithmetic_expression(right)),
        },
        ArithExpr::Mod(left, right) => ExecutableArithmeticExpression::Modulo {
            left: Box::new(executable_arithmetic_expression(left)),
            right: Box::new(executable_arithmetic_expression(right)),
        },
        ArithExpr::Abs(value) => ExecutableArithmeticExpression::AbsoluteValue {
            value: Box::new(executable_arithmetic_expression(value)),
        },
        ArithExpr::Min(left, right) => ExecutableArithmeticExpression::Minimum {
            left: Box::new(executable_arithmetic_expression(left)),
            right: Box::new(executable_arithmetic_expression(right)),
        },
        ArithExpr::Max(left, right) => ExecutableArithmeticExpression::Maximum {
            left: Box::new(executable_arithmetic_expression(left)),
            right: Box::new(executable_arithmetic_expression(right)),
        },
        ArithExpr::Pow(base, exponent) => ExecutableArithmeticExpression::Power {
            base: Box::new(executable_arithmetic_expression(base)),
            exponent: Box::new(executable_arithmetic_expression(exponent)),
        },
        ArithExpr::Cast(value, scalar_type) => ExecutableArithmeticExpression::Cast {
            value: Box::new(executable_arithmetic_expression(value)),
            scalar_type: executable_scalar_type(*scalar_type),
        },
        ArithExpr::FuncCall { name, args } => ExecutableArithmeticExpression::FunctionCall {
            name: name.clone(),
            arguments: args.iter().map(executable_arithmetic_expression).collect(),
        },
        ArithExpr::Conditional {
            cond_left,
            cond_op,
            cond_right,
            then_expr,
            else_expr,
        } => ExecutableArithmeticExpression::Conditional {
            condition_left: Box::new(executable_arithmetic_expression(cond_left)),
            condition_operator: comparison_operator(*cond_op),
            condition_right: Box::new(executable_arithmetic_expression(cond_right)),
            then_expression: Box::new(executable_arithmetic_expression(then_expr)),
            else_expression: Box::new(executable_arithmetic_expression(else_expr)),
        },
    }
}

fn relation_id(name: &str, arity: usize) -> String {
    format!("relation:{name}/{arity}")
}

fn aggregate_operator(operator: AggOp) -> AggregateOperator {
    match operator {
        AggOp::Count => AggregateOperator::Count,
        AggOp::Sum => AggregateOperator::Sum,
        AggOp::Min => AggregateOperator::Min,
        AggOp::Max => AggregateOperator::Max,
        AggOp::LogSumExp => AggregateOperator::LogSumExp,
    }
}

fn epistemic_operator(operator: EpistemicOp) -> EpistemicOperator {
    match operator {
        EpistemicOp::Know => EpistemicOperator::Know,
        EpistemicOp::Possible => EpistemicOperator::Possible,
    }
}

fn comparison_operator(operator: CompOp) -> ComparisonOperator {
    match operator {
        CompOp::Eq => ComparisonOperator::Equal,
        CompOp::Ne => ComparisonOperator::NotEqual,
        CompOp::Lt => ComparisonOperator::LessThan,
        CompOp::Le => ComparisonOperator::LessThanOrEqual,
        CompOp::Gt => ComparisonOperator::GreaterThan,
        CompOp::Ge => ComparisonOperator::GreaterThanOrEqual,
    }
}
