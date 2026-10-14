//! Contextual scalar encodings, independently scoped from program compilation.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;
use xlog_core::{ScalarType, Schema};
use xlog_ir::ConstValue;

use crate::ast::{Atom, BodyLiteral, PredColumn, Program, Rule, Term, TypeRef};
use crate::compile::prepare_program_frontend;
use crate::lower::{
    comparison_constant_value, resolve_pred_column_type, term_to_typed_const_value, Lowerer,
};
use crate::{normalize_program_for_execution, stratify};

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextualValueProjection {
    pub schema_version: String,
    pub entry_module_id: String,
    pub normalization: ValueProjectionStage,
    pub frontend: ValueProjectionStage,
    pub schema_validation: ValueProjectionStage,
    pub ordinary_rir: ValueProjectionStage,
    pub occurrences: Vec<NativeTermOccurrence>,
    pub source_diagnostics: Vec<NativeSourceDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ValueProjectionStage {
    Accepted,
    Rejected { reason: String },
    Unavailable { reason: String },
}

impl ValueProjectionStage {
    fn unavailable(reason: &str) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeValueOwnership {
    Authored,
    Retained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeValueScope {
    DeclaredScalarCarrier,
    InferredScalarCarrier,
    ComparisonOperand,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeTermPath {
    Head {
        term_index: usize,
    },
    BodyAtom {
        body_ordinal: usize,
        term_index: usize,
    },
    Comparison {
        body_ordinal: usize,
        operand: ComparisonOperand,
    },
    Atom {
        term_index: usize,
    },
    Choice {
        choice_ordinal: usize,
        term_index: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonOperand {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeValueContext {
    RelationColumn {
        relation_id: String,
        column_index: usize,
        column_name: Option<String>,
        authored_column_name: Option<String>,
        /// Effective declaration selected by this entrypoint's merge.
        type_reference: Option<ExecutableTypeReference>,
        /// This provider's own declaration, before compatible declarations coalesce.
        authored_type_reference: Option<ExecutableTypeReference>,
    },
    ComparisonOperand {
        peer_variable: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NativeTermOccurrence {
    pub source_object_id: String,
    pub executable_source_object_id: String,
    pub ownership: NativeValueOwnership,
    pub path: NativeTermPath,
    pub authored_term: ExecutableTerm,
    pub context: NativeValueContext,
    pub scalar_type: Option<ExecutableScalarType>,
    pub scope: NativeValueScope,
    pub outcome: NativeTermOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeTermOutcome {
    Constant { value: NativeScalarValue },
    Variable { name: String },
    Anonymous,
    Unavailable { reason_code: String, reason: String },
}

impl NativeTermOutcome {
    fn unavailable(code: &str, reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason_code: code.into(),
            reason: reason.into(),
        }
    }
}

/// Decimal integer strings and hexadecimal float bits avoid JSON numeric loss.
/// Equality of same-scalar values preserves native total-order float equality,
/// including signed zero. This is not a mixed-type comparison evaluator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeScalarValue {
    U32 { value: String },
    U64 { value: String },
    I32 { value: String },
    I64 { value: String },
    F32 { bits: String },
    F64 { bits: String },
    Bool { value: bool },
    Symbol { value: String },
}

impl NativeScalarValue {
    fn from_native(value: ConstValue) -> (ScalarType, Self) {
        match value {
            ConstValue::U32(value) => (
                ScalarType::U32,
                Self::U32 {
                    value: value.to_string(),
                },
            ),
            ConstValue::U64(value) => (
                ScalarType::U64,
                Self::U64 {
                    value: value.to_string(),
                },
            ),
            ConstValue::I32(value) => (
                ScalarType::I32,
                Self::I32 {
                    value: value.to_string(),
                },
            ),
            ConstValue::I64(value) => (
                ScalarType::I64,
                Self::I64 {
                    value: value.to_string(),
                },
            ),
            ConstValue::F32(value) => (
                ScalarType::F32,
                Self::F32 {
                    bits: format!("{:08x}", value.to_bits()),
                },
            ),
            ConstValue::F64(value) => (
                ScalarType::F64,
                Self::F64 {
                    bits: format!("{:016x}", value.to_bits()),
                },
            ),
            ConstValue::Bool(value) => (ScalarType::Bool, Self::Bool { value }),
            ConstValue::Symbol(value) => (ScalarType::Symbol, Self::Symbol { value }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NativeSourceDiagnostic {
    pub source_object_id: String,
    pub ownership: NativeValueOwnership,
    pub body_ordinal: Option<usize>,
    pub reason_code: String,
    pub reason: String,
}

struct Carrier {
    name: Option<String>,
    reference: TypeRef,
    scalar: Result<ScalarType, String>,
}

struct ProjectionBuilder {
    report: ContextualValueProjection,
    carriers: HashMap<String, Vec<Carrier>>,
    schemas: HashMap<String, Schema>,
    lowerer: Lowerer,
}

struct SourceContext<'a> {
    source_id: &'a str,
    executable_id: &'a str,
    ownership: NativeValueOwnership,
    declarations: &'a HashMap<String, Vec<PredColumn>>,
}

pub(super) fn build_native_values(
    resolver: &ModuleResolver,
    catalog: &OriginCatalog,
    merged: &MergedProgram,
    dispositions: &[SourceObjectDisposition],
    entry_module_id: &str,
) -> Result<ContextualValueProjection, ResolvedProgramExtractionError> {
    let mut builder = ProjectionBuilder::new(&merged.program, entry_module_id);
    let selected: BTreeMap<_, _> = dispositions
        .iter()
        .filter_map(|row| {
            let target = match &row.disposition {
                SourceObjectDispositionKind::Admitted {
                    executable_source_object_id,
                }
                | SourceObjectDispositionKind::Coalesced {
                    executable_source_object_id,
                    ..
                } => executable_source_object_id,
                _ => return None,
            };
            Some((row.source_object_id.as_str(), target.as_str()))
        })
        .collect();

    // Read native ASTs already owned by the resolver. Never reparse or rematch
    // providers: a coalesced authored float can differ from its retained owner.
    for source in catalog.modules.keys() {
        let module = if resolver.entry_source.as_ref() == Some(source) {
            resolver.entry.as_ref()
        } else {
            resolver.loaded.get(source)
        }
        .ok_or_else(|| ResolvedProgramExtractionError::MissingModule {
            path: source.clone(),
        })?;
        builder.visit_program(
            &module.program,
            NativeValueOwnership::Authored,
            |kind, index| {
                let origin = catalog.origin(source, kind, index)?;
                Ok(selected
                    .get(origin.source_object_id.as_str())
                    .map(|target| (origin.source_object_id, (*target).to_string())))
            },
        )?;
    }
    builder.visit_program(
        &merged.program,
        NativeValueOwnership::Retained,
        |kind, index| {
            let origins = match kind {
                ResolvedSourceObjectKind::Rule => &merged.rule_origins,
                ResolvedSourceObjectKind::Constraint => &merged.constraint_origins,
                ResolvedSourceObjectKind::Query => &merged.query_origins,
                ResolvedSourceObjectKind::ProbabilisticFact => &merged.probabilistic_fact_origins,
                ResolvedSourceObjectKind::AnnotatedDisjunction => {
                    &merged.annotated_disjunction_origins
                }
                ResolvedSourceObjectKind::Evidence => &merged.evidence_origins,
                ResolvedSourceObjectKind::ProbabilisticQuery => &merged.probabilistic_query_origins,
                ResolvedSourceObjectKind::NeuralPredicate => &merged.neural_predicate_origins,
                ResolvedSourceObjectKind::LearnableRule => &merged.learnable_rule_origins,
                _ => unreachable!("only executable term-bearing objects are visited"),
            };
            let id = origins[index].source_object_id.clone();
            Ok(Some((id.clone(), id)))
        },
    )?;
    Ok(builder.report)
}

impl ProjectionBuilder {
    fn new(program: &Program, entry_module_id: &str) -> Self {
        let domains = program
            .domains
            .iter()
            .map(|domain| (domain.name.clone(), domain.typ))
            .collect();
        let carriers = program
            .predicates
            .iter()
            .map(|declaration| {
                let columns = declaration
                    .schema_columns()
                    .into_iter()
                    .enumerate()
                    .map(|(index, column)| Carrier {
                        name: column.name,
                        scalar: resolve_pred_column_type(
                            &declaration.name,
                            index,
                            &column.typ,
                            &domains,
                        )
                        .map_err(|error| error.to_string()),
                        reference: column.typ,
                    })
                    .collect();
                (declaration.name.clone(), columns)
            })
            .collect();
        let mut report = ContextualValueProjection {
            schema_version: "xlog.contextual-values.v1".into(),
            entry_module_id: entry_module_id.into(),
            normalization: ValueProjectionStage::Accepted,
            frontend: ValueProjectionStage::unavailable("normalization did not complete"),
            schema_validation: ValueProjectionStage::unavailable(
                "frontend preparation did not complete",
            ),
            ordinary_rir: ValueProjectionStage::unavailable("schema validation did not complete"),
            occurrences: Vec::new(),
            source_diagnostics: Vec::new(),
        };
        let mut lowerer = Lowerer::new();
        let mut schemas = HashMap::new();
        match normalize_program_for_execution(program.clone()) {
            Err(error) => {
                report.normalization = ValueProjectionStage::Rejected {
                    reason: error.to_string(),
                }
            }
            Ok(normalized) => match prepare_program_frontend(normalized) {
                Err(error) => {
                    report.frontend = ValueProjectionStage::Rejected {
                        reason: error.to_string(),
                    }
                }
                Ok((prepared, _)) => {
                    report.frontend = ValueProjectionStage::Accepted;
                    match lowerer.infer_and_validate_schemas(&prepared) {
                        Err(error) => {
                            report.schema_validation = ValueProjectionStage::Rejected {
                                reason: error.to_string(),
                            }
                        }
                        Ok(()) => {
                            report.schema_validation = ValueProjectionStage::Accepted;
                            schemas = lowerer.schemas().clone();
                            report.ordinary_rir = match stratify(&prepared) {
                                Err(error) => ValueProjectionStage::Rejected {
                                    reason: error.to_string(),
                                },
                                Ok(strata) => {
                                    lowerer.set_strata(
                                        strata
                                            .into_iter()
                                            .map(|stratum| stratum.predicates)
                                            .collect(),
                                    );
                                    match lowerer.lower_program_with_inferred_schemas(&prepared) {
                                        Ok(_) => ValueProjectionStage::Accepted,
                                        Err(error) => ValueProjectionStage::Rejected {
                                            reason: error.to_string(),
                                        },
                                    }
                                }
                            };
                        }
                    }
                }
            },
        }
        Self {
            report,
            carriers,
            schemas,
            lowerer,
        }
    }

    fn column_type(&self, predicate: &str, index: usize) -> Option<ScalarType> {
        self.carriers
            .get(predicate)
            .and_then(|columns| columns.get(index))
            .and_then(|carrier| carrier.scalar.as_ref().ok().copied())
            .or_else(|| {
                self.schemas
                    .get(predicate)
                    .and_then(|schema| schema.column_type(index))
            })
    }

    fn diagnostic(
        &mut self,
        source: &SourceContext<'_>,
        ordinal: Option<usize>,
        code: &str,
        reason: String,
    ) {
        self.report.source_diagnostics.push(NativeSourceDiagnostic {
            source_object_id: source.source_id.into(),
            ownership: source.ownership,
            body_ordinal: ordinal,
            reason_code: code.into(),
            reason,
        });
    }

    fn record(
        &mut self,
        source: &SourceContext<'_>,
        path: NativeTermPath,
        term: &Term,
        context: NativeValueContext,
        scope: NativeValueScope,
        value: (Option<ScalarType>, NativeTermOutcome),
    ) {
        let (scalar, outcome) = value;
        self.report.occurrences.push(NativeTermOccurrence {
            source_object_id: source.source_id.into(),
            executable_source_object_id: source.executable_id.into(),
            ownership: source.ownership,
            path,
            authored_term: executable_term(term),
            context,
            scalar_type: scalar.map(executable_scalar_type),
            scope,
            outcome,
        });
    }

    fn atom(
        &mut self,
        source: &SourceContext<'_>,
        atom: &Atom,
        path: impl Fn(usize) -> NativeTermPath,
    ) {
        for (index, term) in atom.terms.iter().enumerate() {
            let carrier = self
                .carriers
                .get(&atom.predicate)
                .and_then(|columns| columns.get(index));
            let context = NativeValueContext::RelationColumn {
                relation_id: relation_id(&atom.predicate, atom.terms.len()),
                column_index: index,
                column_name: carrier.and_then(|carrier| carrier.name.clone()),
                authored_column_name: source
                    .declarations
                    .get(&atom.predicate)
                    .and_then(|columns| columns.get(index))
                    .and_then(|column| column.name.clone()),
                type_reference: carrier
                    .map(|carrier| executable_type_reference(&carrier.reference)),
                authored_type_reference: source
                    .declarations
                    .get(&atom.predicate)
                    .and_then(|columns| columns.get(index))
                    .map(|column| executable_type_reference(&column.typ)),
            };
            let scalar = self.column_type(&atom.predicate, index);
            let scope = if carrier.is_some() {
                NativeValueScope::DeclaredScalarCarrier
            } else {
                NativeValueScope::InferredScalarCarrier
            };
            let outcome = match carrier {
                Some(Carrier { reference: TypeRef::List(_) | TypeRef::Term | TypeRef::Compound | TypeRef::PredRef, .. }) =>
                    NativeTermOutcome::unavailable("transformed_context", "This declared carrier requires native term normalization; no authored-to-prepared occurrence correspondence is supplied"),
                Some(Carrier { scalar: Err(reason), .. }) => NativeTermOutcome::unavailable("schema_rejected", reason.clone()),
                _ => match scalar {
                    None => NativeTermOutcome::unavailable("missing_schema", format!("No validated native scalar for {} column {index}", atom.predicate)),
                    Some(typ) => scalar_outcome(term, term_to_typed_const_value(term, typ)),
                },
            };
            let scope = if matches!(outcome, NativeTermOutcome::Unavailable { .. }) {
                NativeValueScope::Unavailable
            } else {
                scope
            };
            self.record(source, path(index), term, context, scope, (scalar, outcome));
        }
    }

    fn body(&mut self, source: &SourceContext<'_>, rule: &Rule) {
        let types = self.lowerer.infer_rule_variable_types(rule, |atom, index| {
            self.column_type(&atom.predicate, index)
        });
        if let Err(error) = &types {
            self.diagnostic(
                source,
                None,
                "variable_type_inference_rejected",
                error.to_string(),
            );
        }
        for (ordinal, literal) in rule.body.iter().enumerate() {
            match literal {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    self.atom(source, atom, |term_index| NativeTermPath::BodyAtom {
                        body_ordinal: ordinal,
                        term_index,
                    });
                }
                BodyLiteral::Epistemic(literal) => {
                    self.atom(source, &literal.atom, |term_index| {
                        NativeTermPath::BodyAtom {
                            body_ordinal: ordinal,
                            term_index,
                        }
                    });
                }
                BodyLiteral::Comparison(comparison) => {
                    for (side, term, peer) in [
                        (ComparisonOperand::Left, &comparison.left, &comparison.right),
                        (
                            ComparisonOperand::Right,
                            &comparison.right,
                            &comparison.left,
                        ),
                    ] {
                        let variable = match (term, peer) {
                            (Term::Variable(name), _) | (_, Term::Variable(name)) => Some(name),
                            _ => None,
                        };
                        let scalar = variable
                            .and_then(|name| types.as_ref().ok()?.get(name).map(|(typ, _)| *typ));
                        let context = NativeValueContext::ComparisonOperand {
                            peer_variable: match peer {
                                Term::Variable(name) => Some(name.clone()),
                                _ => None,
                            },
                        };
                        let (scalar, outcome) = if variable.is_some() && scalar.is_none() {
                            let reason = match &types {
                                Err(error) => error.to_string(),
                                Ok(_) => {
                                    "No native scalar type evidence for this comparison variable"
                                        .into()
                                }
                            };
                            (
                                None,
                                NativeTermOutcome::unavailable(
                                    "comparison_type_unavailable",
                                    reason,
                                ),
                            )
                        } else {
                            match comparison_constant_value(term, scalar) {
                                Ok(Some(value)) => {
                                    let (typ, value) = NativeScalarValue::from_native(value);
                                    (Some(typ), NativeTermOutcome::Constant { value })
                                }
                                result => (scalar, scalar_outcome(term, result)),
                            }
                        };
                        let scope = if matches!(outcome, NativeTermOutcome::Unavailable { .. }) {
                            NativeValueScope::Unavailable
                        } else {
                            NativeValueScope::ComparisonOperand
                        };
                        self.record(
                            source,
                            NativeTermPath::Comparison {
                                body_ordinal: ordinal,
                                operand: side,
                            },
                            term,
                            context,
                            scope,
                            (scalar, outcome),
                        );
                    }
                }
                BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {
                    self.diagnostic(source, Some(ordinal), "transformed_context",
                        "Expression or meta-literal requires a native generated-occurrence correspondence; the authored construct remains in executable_program".into());
                }
            }
        }
    }

    fn visit_program(
        &mut self,
        program: &Program,
        ownership: NativeValueOwnership,
        mut identity: impl FnMut(
            ResolvedSourceObjectKind,
            usize,
        )
            -> Result<Option<(String, String)>, ResolvedProgramExtractionError>,
    ) -> Result<(), ResolvedProgramExtractionError> {
        let declarations = program
            .predicates
            .iter()
            .map(|declaration| (declaration.name.clone(), declaration.schema_columns()))
            .collect();
        macro_rules! visit {
            ($items:expr, $kind:ident, $item:ident, $source:ident, $body:block) => {
                for (index, $item) in $items.iter().enumerate() {
                    if let Some((id, target)) = identity(ResolvedSourceObjectKind::$kind, index)? {
                        let $source = SourceContext {
                            source_id: &id,
                            executable_id: &target,
                            ownership,
                            declarations: &declarations,
                        };
                        $body
                    }
                }
            };
        }
        visit!(&program.rules, Rule, rule, source, {
            self.atom(&source, &rule.head, |term_index| NativeTermPath::Head {
                term_index,
            });
            self.body(&source, rule);
        });
        visit!(&program.constraints, Constraint, constraint, source, {
            self.body(
                &source,
                &Rule {
                    head: Atom {
                        predicate: "<constraint>".into(),
                        terms: Vec::new(),
                    },
                    body: constraint.body.clone(),
                },
            );
        });
        visit!(&program.queries, Query, query, source, {
            self.atom(&source, &query.atom, |term_index| NativeTermPath::Atom {
                term_index,
            });
        });
        visit!(&program.prob_facts, ProbabilisticFact, fact, source, {
            self.atom(&source, &fact.atom, |term_index| NativeTermPath::Atom {
                term_index,
            });
        });
        visit!(
            &program.annotated_disjunctions,
            AnnotatedDisjunction,
            disjunction,
            source,
            {
                for (choice_ordinal, choice) in disjunction.choices.iter().enumerate() {
                    self.atom(&source, &choice.atom, |term_index| NativeTermPath::Choice {
                        choice_ordinal,
                        term_index,
                    });
                }
            }
        );
        visit!(&program.evidence, Evidence, evidence, source, {
            self.atom(&source, &evidence.atom, |term_index| NativeTermPath::Atom {
                term_index,
            });
        });
        visit!(&program.prob_queries, ProbabilisticQuery, query, source, {
            self.atom(&source, &query.atom, |term_index| NativeTermPath::Atom {
                term_index,
            });
        });
        visit!(
            &program.neural_predicates,
            NeuralPredicate,
            neural,
            source,
            {
                self.atom(&source, &neural.predicate, |term_index| {
                    NativeTermPath::Atom { term_index }
                });
            }
        );
        visit!(&program.learnable_rules, LearnableRule, rule, source, {
            self.atom(&source, &rule.head, |term_index| NativeTermPath::Head {
                term_index,
            });
            self.body(
                &source,
                &Rule {
                    head: rule.head.clone(),
                    body: rule.body.clone(),
                },
            );
        });
        Ok(())
    }
}

fn scalar_outcome(
    term: &Term,
    lowered: xlog_core::Result<Option<ConstValue>>,
) -> NativeTermOutcome {
    match lowered {
        Err(error) => NativeTermOutcome::unavailable("constant_lowering_rejected", error.to_string()),
        Ok(Some(value)) => NativeTermOutcome::Constant { value: NativeScalarValue::from_native(value).1 },
        Ok(None) => match term {
            Term::Variable(name) => NativeTermOutcome::Variable { name: name.clone() },
            Term::Anonymous => NativeTermOutcome::Anonymous,
            Term::Aggregate(_) => NativeTermOutcome::unavailable("computed_value", "An aggregate result is computed, not a scalar constant"),
            _ => NativeTermOutcome::unavailable("transformed_context", "This term requires native normalization; no authored-to-prepared occurrence correspondence is supplied"),
        }
    }
}
