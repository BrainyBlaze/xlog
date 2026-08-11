//! Generated-rule diagnostic row analysis and JSON presentation.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use xlog_core::{symbol, Result, ScalarType, XlogError};
use xlog_logic::ast::{BodyLiteral, CompOp, Program, Term};
use xlog_logic::{
    compare_arithmetic_values, evaluate_arithmetic_expression, format_atom, format_term,
    generated_function_variable_sources, source_format_normalized_alternative, ArithmeticValue,
    Lowerer,
};

use super::{json_escape, json_string_array};

pub(super) struct GeneratedRuleDiagnostic {
    rule_head: String,
    source_relation: String,
    row_decisions: Vec<GeneratedRuleRowDecision>,
}

struct GeneratedRuleRowDecision {
    row_key: String,
    accepted: bool,
    failed_predicates: Vec<String>,
    threshold_comparisons: Vec<ThresholdComparison>,
    aggregate_inputs: Vec<String>,
}

#[derive(Clone)]
struct ThresholdComparison {
    predicate: String,
    left: String,
    op: String,
    right: String,
    left_value: String,
    right_value: String,
    passed: bool,
}

struct GeneratedRuleEvaluation {
    accepted: bool,
    failed_predicates: Vec<String>,
    threshold_comparisons: Vec<ThresholdComparison>,
}

enum GeneratedRuleSearchTask {
    Continue {
        literal_index: usize,
        bindings: DiagnosticBindings,
        threshold_comparisons: Vec<ThresholdComparison>,
    },
    TryPositiveRows {
        literal_index: usize,
        row_index: usize,
        bindings: DiagnosticBindings,
        threshold_comparisons: Vec<ThresholdComparison>,
    },
}

#[derive(Clone)]
struct DiagnosticScalar {
    value: ArithmeticValue,
    scalar_type: ScalarType,
    label: String,
}

type DiagnosticBindings = HashMap<String, DiagnosticScalar>;
type DiagnosticRow = Vec<DiagnosticScalar>;
type DiagnosticRelationRows = HashMap<(String, usize), Vec<DiagnosticRow>>;

pub(super) fn explain_generated_rule_diagnostics(
    source_program: &Program,
    analysis_program: &Program,
    source_path: Option<&Path>,
) -> Result<Vec<GeneratedRuleDiagnostic>> {
    let function_variable_sources =
        generated_function_variable_sources(source_program, analysis_program);
    let source_predicates = analysis_program
        .rules
        .iter()
        .filter(|rule| !rule.body.is_empty() && generated_rule_candidate(rule))
        .filter_map(diagnostic_source_atom)
        .map(|atom| atom.predicate.clone())
        .collect::<HashSet<_>>();
    let external_rows = source_path
        .map(|path| load_external_relation_rows(analysis_program, path, &source_predicates))
        .transpose()?
        .unwrap_or_default();
    let relation_rows = diagnostic_relation_rows(analysis_program, &external_rows)?;
    let mut diagnostics = Vec::new();
    for rule in analysis_program
        .rules
        .iter()
        .filter(|rule| !rule.body.is_empty() && generated_rule_candidate(rule))
    {
        let Some(source_atom) = diagnostic_source_atom(rule) else {
            continue;
        };
        ensure_extensional_generated_rule_support(analysis_program, rule)?;
        let mut row_decisions = Vec::new();
        for source_row in relation_rows_for_atom(&relation_rows, source_atom) {
            let Some(bindings) = bindings_for_source_row(source_atom, source_row)? else {
                continue;
            };
            let evaluation = evaluate_generated_rule(
                &relation_rows,
                rule,
                source_atom,
                bindings,
                &function_variable_sources,
            )?;
            row_decisions.push(GeneratedRuleRowDecision {
                row_key: source_row
                    .first()
                    .map(|value| value.label.clone())
                    .unwrap_or_else(|| source_atom.predicate.clone()),
                accepted: evaluation.accepted,
                failed_predicates: evaluation.failed_predicates,
                threshold_comparisons: evaluation.threshold_comparisons,
                aggregate_inputs: vec![format!(
                    "{}({})",
                    source_atom.predicate,
                    source_row
                        .iter()
                        .map(|value| value.label.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                )],
            });
        }

        if !row_decisions.is_empty() {
            diagnostics.push(GeneratedRuleDiagnostic {
                rule_head: rule.head.predicate.clone(),
                source_relation: source_atom.predicate.clone(),
                row_decisions,
            });
        }
    }
    Ok(diagnostics)
}

fn ensure_extensional_generated_rule_support(
    program: &Program,
    rule: &xlog_logic::ast::Rule,
) -> Result<()> {
    let probabilistic_predicates = program
        .prob_facts
        .iter()
        .map(|fact| fact.atom.predicate.as_str())
        .chain(
            program
                .annotated_disjunctions
                .iter()
                .flat_map(|disjunction| {
                    disjunction
                        .choices
                        .iter()
                        .map(|choice| choice.atom.predicate.as_str())
                }),
        )
        .collect::<HashSet<_>>();
    let intensional_predicates = program
        .rules
        .iter()
        .filter(|candidate| !candidate.body.is_empty())
        .map(|candidate| candidate.head.predicate.as_str())
        .collect::<HashSet<_>>();
    let body_predicates = || {
        rule.body.iter().filter_map(|literal| match literal {
            BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                Some(atom.predicate.as_str())
            }
            BodyLiteral::Epistemic(_)
            | BodyLiteral::Comparison(_)
            | BodyLiteral::IsExpr(_)
            | BodyLiteral::Univ(_) => None,
        })
    };
    if let Some(predicate) =
        body_predicates().find(|predicate| probabilistic_predicates.contains(predicate))
    {
        return Err(XlogError::Compilation(format!(
            "generated-rule diagnostics do not assign deterministic row decisions to probabilistic predicate '{predicate}'"
        )));
    }
    if let Some(predicate) =
        body_predicates().find(|predicate| intensional_predicates.contains(predicate))
    {
        return Err(XlogError::Compilation(format!(
            "generated-rule diagnostics require materialized rows for derived predicate '{predicate}'"
        )));
    }
    Ok(())
}

fn generated_rule_candidate(rule: &xlog_logic::ast::Rule) -> bool {
    rule.head.predicate.starts_with("generated_")
        || rule.head.predicate.starts_with("xlog_accepted_")
        || rule.head.predicate.starts_with("xlog_rejected_")
        || rule.body.iter().any(|literal| match literal {
            BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                diagnostic_source_predicate(&atom.predicate)
            }
            BodyLiteral::Epistemic(_) => false,
            BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => false,
        })
}

fn diagnostic_source_atom(rule: &xlog_logic::ast::Rule) -> Option<&xlog_logic::ast::Atom> {
    rule.body.iter().find_map(|literal| match literal {
        BodyLiteral::Positive(atom) if diagnostic_source_predicate(&atom.predicate) => Some(atom),
        _ => None,
    })
}

fn diagnostic_source_predicate(predicate: &str) -> bool {
    predicate.starts_with("generated_")
        || predicate.ends_with("_candidate_input")
        || (predicate.contains("candidate") && predicate.ends_with("_input"))
}

fn diagnostic_relation_rows(
    program: &Program,
    external_rows: &HashMap<String, Vec<DiagnosticRow>>,
) -> Result<DiagnosticRelationRows> {
    let mut lowerer = Lowerer::new();
    lowerer.infer_and_validate_schemas(program)?;
    let schemas = lowerer.schemas();
    let mut relation_rows = DiagnosticRelationRows::new();
    for fact in program.rules.iter().filter(|rule| rule.body.is_empty()) {
        let row = diagnostic_row(&fact.head.predicate, &fact.head.terms, schemas)?;
        relation_rows
            .entry((fact.head.predicate.clone(), fact.head.terms.len()))
            .or_default()
            .push(row);
    }
    for (predicate, rows) in external_rows {
        for row in rows {
            relation_rows
                .entry((predicate.clone(), row.len()))
                .or_default()
                .push(row.clone());
        }
    }
    Ok(relation_rows)
}

fn diagnostic_row(
    predicate: &str,
    terms: &[Term],
    schemas: &HashMap<String, xlog_core::Schema>,
) -> Result<DiagnosticRow> {
    let schema = schemas.get(predicate).ok_or_else(|| {
        XlogError::Compilation(format!(
            "generated-rule diagnostics require a schema for predicate '{predicate}'"
        ))
    })?;
    if schema.arity() != terms.len() {
        return Err(XlogError::Compilation(format!(
            "generated-rule diagnostic row for '{predicate}' has {} values but its schema has {}",
            terms.len(),
            schema.arity()
        )));
    }
    terms
        .iter()
        .enumerate()
        .map(|(index, term)| {
            let scalar_type = schema.column_type(index).ok_or_else(|| {
                XlogError::Compilation(format!(
                    "generated-rule diagnostics require a type for '{predicate}' column {}",
                    index + 1
                ))
            })?;
            let value = ArithmeticValue::from_typed_term(term, scalar_type)?;
            Ok(DiagnosticScalar {
                label: arithmetic_value_label(&value),
                value,
                scalar_type,
            })
        })
        .collect()
}

fn relation_rows_for_atom<'a>(
    relation_rows: &'a DiagnosticRelationRows,
    atom: &xlog_logic::ast::Atom,
) -> &'a [DiagnosticRow] {
    relation_rows
        .get(&(atom.predicate.clone(), atom.terms.len()))
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn bindings_for_source_row(
    atom: &xlog_logic::ast::Atom,
    row: &[DiagnosticScalar],
) -> Result<Option<DiagnosticBindings>> {
    extend_bindings_for_row(atom, row, &HashMap::new())
}

fn extend_bindings_for_row(
    atom: &xlog_logic::ast::Atom,
    row: &[DiagnosticScalar],
    initial: &DiagnosticBindings,
) -> Result<Option<DiagnosticBindings>> {
    if atom.terms.len() != row.len() {
        return Ok(None);
    }
    let mut bindings = initial.clone();
    for (pattern, value) in atom.terms.iter().zip(row) {
        match pattern {
            Term::Variable(name) => {
                if let Some(existing) = bindings.get(name) {
                    if existing.scalar_type != value.scalar_type
                        || !compare_arithmetic_values(&existing.value, CompOp::Eq, &value.value)?
                    {
                        return Ok(None);
                    }
                } else {
                    bindings.insert(name.clone(), value.clone());
                }
            }
            Term::Anonymous => {}
            _ => {
                let pattern_value = ArithmeticValue::from_typed_term(pattern, value.scalar_type)?;
                if !compare_arithmetic_values(&pattern_value, CompOp::Eq, &value.value)? {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(bindings))
}

fn load_external_relation_rows(
    program: &Program,
    source_path: &Path,
    source_predicates: &HashSet<String>,
) -> Result<HashMap<String, Vec<DiagnosticRow>>> {
    let mut lowerer = Lowerer::new();
    lowerer.infer_and_validate_schemas(program)?;
    let manifest = external_relation_manifest(source_path);
    if manifest.is_some() && source_predicates.len() > 1 {
        let mut predicates = source_predicates.iter().cloned().collect::<Vec<_>>();
        predicates.sort();
        return Err(XlogError::Compilation(format!(
            "candidate relation manifest is ambiguous for generated-rule source predicates: {}",
            predicates.join(", ")
        )));
    }
    let manifest_predicate = source_predicates.iter().next();
    let mut loaded = HashMap::new();
    for decl in &program.predicates {
        let manifest_source = manifest_predicate
            .filter(|predicate| *predicate == &decl.name)
            .and(manifest.as_ref());
        let Some((relation_path, columns)) =
            external_relation_source(source_path, decl, manifest_source)
        else {
            continue;
        };
        if columns.len() != decl.arity() {
            continue;
        }
        let schema = lowerer.schemas().get(&decl.name).ok_or_else(|| {
            XlogError::Compilation(format!(
                "generated-rule diagnostics require a schema for external predicate '{}'",
                decl.name
            ))
        })?;
        let source = std::fs::read_to_string(&relation_path).map_err(|error| {
            XlogError::Compilation(format!(
                "cannot read external relation '{}': {error}",
                relation_path.display()
            ))
        })?;
        let json = serde_json::from_str::<serde_json::Value>(&source).map_err(|error| {
            XlogError::Compilation(format!(
                "invalid JSON in external relation '{}': {error}",
                relation_path.display()
            ))
        })?;
        let rows = json
            .get("rows")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                XlogError::Compilation(format!(
                    "external relation '{}' must contain a rows array",
                    relation_path.display()
                ))
            })?;
        let mut relation_rows = Vec::new();
        for (row_index, row) in rows.iter().enumerate() {
            let object = row.as_object().ok_or_else(|| {
                XlogError::Compilation(format!(
                    "external relation '{}' row {} must be an object",
                    relation_path.display(),
                    row_index + 1
                ))
            })?;
            let mut values = Vec::with_capacity(columns.len());
            for (column_index, column) in columns.iter().enumerate() {
                let scalar_type = schema.column_type(column_index).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "external predicate '{}' has no type for column '{}'",
                        decl.name, column
                    ))
                })?;
                let value = object.get(column).ok_or_else(|| {
                    XlogError::Compilation(format!(
                        "external relation '{}' row {} is missing column '{}'",
                        relation_path.display(),
                        row_index + 1,
                        column
                    ))
                })?;
                values.push(json_diagnostic_scalar(value, scalar_type).map_err(|error| {
                    XlogError::Compilation(format!(
                        "external relation '{}' row {} column '{}': {error}",
                        relation_path.display(),
                        row_index + 1,
                        column
                    ))
                })?);
            }
            relation_rows.push(values);
        }
        if !relation_rows.is_empty() {
            loaded.insert(decl.name.clone(), relation_rows);
        }
    }
    Ok(loaded)
}

fn external_relation_source(
    source_path: &Path,
    decl: &xlog_logic::ast::PredDecl,
    manifest: Option<&(PathBuf, Vec<String>)>,
) -> Option<(PathBuf, Vec<String>)> {
    if let Some((relation_path, columns)) = manifest {
        if columns.len() == decl.arity() {
            return Some((relation_path.clone(), columns.clone()));
        }
    }
    let columns = declared_column_names(decl)?;
    let source_dir = source_path.parent()?;
    for candidate in relation_json_candidates(source_dir, &decl.name) {
        if candidate.exists() {
            return Some((candidate, columns));
        }
    }
    None
}

fn external_relation_manifest(source_path: &Path) -> Option<(PathBuf, Vec<String>)> {
    let source_dir = source_path.parent()?;
    let mut manifests = vec![source_dir.join("xlog_hypothesis_execution.json")];
    if let Some(parent) = source_dir.parent() {
        manifests.push(parent.join("xlog_hypothesis_execution.json"));
    }
    for manifest_path in manifests {
        let Ok(source) = std::fs::read_to_string(&manifest_path) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&source) else {
            continue;
        };
        let Some(columns) = json
            .get("relation_input_columns")
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
        else {
            continue;
        };
        let Some(path_value) = json
            .get("relation_input_path")
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let relation_path = PathBuf::from(path_value);
        let relation_path = if relation_path.is_absolute() {
            relation_path
        } else {
            manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(relation_path)
        };
        if relation_path.exists() {
            return Some((relation_path, columns));
        }
    }
    None
}

fn declared_column_names(decl: &xlog_logic::ast::PredDecl) -> Option<Vec<String>> {
    decl.schema_columns()
        .into_iter()
        .map(|column| column.name)
        .collect()
}

fn relation_json_candidates(source_dir: &Path, predicate: &str) -> Vec<PathBuf> {
    let mut candidates = vec![source_dir.join(format!("{predicate}.json"))];
    if let Some(stem) = predicate.strip_suffix("_input") {
        candidates.push(source_dir.join(format!("{stem}_relation.json")));
    }
    candidates
}

fn json_diagnostic_scalar(
    value: &serde_json::Value,
    scalar_type: ScalarType,
) -> Result<DiagnosticScalar> {
    let arithmetic = match scalar_type {
        ScalarType::I32 => ArithmeticValue::I32(
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| XlogError::Compilation("expected an i32 JSON value".to_string()))?,
        ),
        ScalarType::I64 => ArithmeticValue::I64(
            value
                .as_i64()
                .ok_or_else(|| XlogError::Compilation("expected an i64 JSON value".to_string()))?,
        ),
        ScalarType::U32 => ArithmeticValue::U32(
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| XlogError::Compilation("expected a u32 JSON value".to_string()))?,
        ),
        ScalarType::U64 => ArithmeticValue::U64(
            value
                .as_u64()
                .ok_or_else(|| XlogError::Compilation("expected a u64 JSON value".to_string()))?,
        ),
        ScalarType::F32 => ArithmeticValue::F32(
            value
                .as_f64()
                .filter(|value| value.is_finite() && value.abs() <= f64::from(f32::MAX))
                .ok_or_else(|| {
                    XlogError::Compilation("expected a finite f32 JSON value".to_string())
                })? as f32,
        ),
        ScalarType::F64 => ArithmeticValue::F64(
            value
                .as_f64()
                .ok_or_else(|| XlogError::Compilation("expected an f64 JSON value".to_string()))?,
        ),
        ScalarType::Bool => {
            ArithmeticValue::Bool(value.as_bool().ok_or_else(|| {
                XlogError::Compilation("expected a boolean JSON value".to_string())
            })?)
        }
        ScalarType::Symbol => {
            ArithmeticValue::Symbol(symbol::intern(value.as_str().ok_or_else(|| {
                XlogError::Compilation("expected a string JSON value".to_string())
            })?))
        }
    };
    Ok(DiagnosticScalar {
        label: match value {
            serde_json::Value::String(value) => value.clone(),
            _ => value.to_string(),
        },
        value: arithmetic,
        scalar_type,
    })
}

fn evaluate_generated_rule(
    relation_rows: &DiagnosticRelationRows,
    rule: &xlog_logic::ast::Rule,
    source_atom: &xlog_logic::ast::Atom,
    bindings: DiagnosticBindings,
    function_variable_sources: &HashMap<String, String>,
) -> Result<GeneratedRuleEvaluation> {
    let mut tasks = vec![GeneratedRuleSearchTask::Continue {
        literal_index: 0,
        bindings,
        threshold_comparisons: Vec::new(),
    }];
    let mut first_failure = None;

    while let Some(task) = tasks.pop() {
        match task {
            GeneratedRuleSearchTask::Continue {
                literal_index,
                mut bindings,
                mut threshold_comparisons,
            } => {
                let Some(literal) = rule.body.get(literal_index) else {
                    return Ok(GeneratedRuleEvaluation {
                        accepted: true,
                        failed_predicates: Vec::new(),
                        threshold_comparisons,
                    });
                };
                match literal {
                    BodyLiteral::Positive(atom) if std::ptr::eq(atom, source_atom) => {
                        tasks.push(GeneratedRuleSearchTask::Continue {
                            literal_index: literal_index + 1,
                            bindings,
                            threshold_comparisons,
                        });
                    }
                    BodyLiteral::Positive(_) => {
                        tasks.push(GeneratedRuleSearchTask::TryPositiveRows {
                            literal_index,
                            row_index: 0,
                            bindings,
                            threshold_comparisons,
                        });
                    }
                    BodyLiteral::Negated(atom) => {
                        let mut matched = false;
                        for row in relation_rows_for_atom(relation_rows, atom) {
                            if extend_bindings_for_row(atom, row, &bindings)?.is_some() {
                                matched = true;
                                break;
                            }
                        }
                        if matched {
                            first_failure.get_or_insert_with(|| GeneratedRuleEvaluation {
                                accepted: false,
                                failed_predicates: vec![source_format_normalized_alternative(
                                    &format!("not {}", format_atom(atom)),
                                    function_variable_sources,
                                )],
                                threshold_comparisons,
                            });
                        } else {
                            tasks.push(GeneratedRuleSearchTask::Continue {
                                literal_index: literal_index + 1,
                                bindings,
                                threshold_comparisons,
                            });
                        }
                    }
                    BodyLiteral::Comparison(comparison) => {
                        let report =
                            threshold_comparison(comparison, &bindings, function_variable_sources)?;
                        let passed = report.passed;
                        let predicate = report.predicate.clone();
                        threshold_comparisons.push(report);
                        if passed {
                            tasks.push(GeneratedRuleSearchTask::Continue {
                                literal_index: literal_index + 1,
                                bindings,
                                threshold_comparisons,
                            });
                        } else {
                            first_failure.get_or_insert(GeneratedRuleEvaluation {
                                accepted: false,
                                failed_predicates: vec![predicate],
                                threshold_comparisons,
                            });
                        }
                    }
                    BodyLiteral::IsExpr(binding) => {
                        let value = arithmetic_expression_value(&binding.expr, &bindings)?;
                        let compatible = match bindings.get(&binding.target) {
                            Some(existing) => {
                                existing.scalar_type == value.scalar_type
                                    && compare_arithmetic_values(
                                        &existing.value,
                                        CompOp::Eq,
                                        &value.value,
                                    )?
                            }
                            None => {
                                bindings.insert(binding.target.clone(), value);
                                true
                            }
                        };
                        if compatible {
                            tasks.push(GeneratedRuleSearchTask::Continue {
                                literal_index: literal_index + 1,
                                bindings,
                                threshold_comparisons,
                            });
                        } else {
                            first_failure.get_or_insert(GeneratedRuleEvaluation {
                                accepted: false,
                                failed_predicates: vec![source_format_normalized_alternative(
                                    &binding.target,
                                    function_variable_sources,
                                )],
                                threshold_comparisons,
                            });
                        }
                    }
                    BodyLiteral::Epistemic(_) => {
                        return Err(XlogError::Compilation(
                            "generated-rule diagnostics do not evaluate epistemic literals"
                                .to_string(),
                        ));
                    }
                    BodyLiteral::Univ(_) => {
                        return Err(XlogError::Compilation(
                            "generated-rule diagnostics do not evaluate univ literals".to_string(),
                        ));
                    }
                }
            }
            GeneratedRuleSearchTask::TryPositiveRows {
                literal_index,
                mut row_index,
                bindings,
                threshold_comparisons,
            } => {
                let BodyLiteral::Positive(atom) = &rule.body[literal_index] else {
                    return Err(XlogError::Compilation(
                        "invalid generated-rule diagnostic search state".to_string(),
                    ));
                };
                let rows = relation_rows_for_atom(relation_rows, atom);
                let mut matched = None;
                while let Some(row) = rows.get(row_index) {
                    row_index += 1;
                    if let Some(next_bindings) = extend_bindings_for_row(atom, row, &bindings)? {
                        matched = Some(next_bindings);
                        break;
                    }
                }
                if let Some(next_bindings) = matched {
                    tasks.push(GeneratedRuleSearchTask::TryPositiveRows {
                        literal_index,
                        row_index,
                        bindings,
                        threshold_comparisons: threshold_comparisons.clone(),
                    });
                    tasks.push(GeneratedRuleSearchTask::Continue {
                        literal_index: literal_index + 1,
                        bindings: next_bindings,
                        threshold_comparisons,
                    });
                } else {
                    first_failure.get_or_insert_with(|| GeneratedRuleEvaluation {
                        accepted: false,
                        failed_predicates: vec![source_format_normalized_alternative(
                            &format_atom(atom),
                            function_variable_sources,
                        )],
                        threshold_comparisons,
                    });
                }
            }
        }
    }

    Ok(first_failure.unwrap_or(GeneratedRuleEvaluation {
        accepted: false,
        failed_predicates: Vec::new(),
        threshold_comparisons: Vec::new(),
    }))
}

fn threshold_comparison(
    comparison: &xlog_logic::ast::Comparison,
    bindings: &DiagnosticBindings,
    function_variable_sources: &HashMap<String, String>,
) -> Result<ThresholdComparison> {
    let left = source_format_normalized_alternative(
        &format_term(&comparison.left),
        function_variable_sources,
    );
    let right = source_format_normalized_alternative(
        &format_term(&comparison.right),
        function_variable_sources,
    );
    let bound_left = bound_scalar(&comparison.left, bindings);
    let bound_right = bound_scalar(&comparison.right, bindings);
    let left_value = comparison_scalar(
        &comparison.left,
        bound_left.as_ref(),
        bound_right.as_ref().map(|value| value.scalar_type),
    )?;
    let right_value = comparison_scalar(
        &comparison.right,
        bound_right.as_ref(),
        bound_left.as_ref().map(|value| value.scalar_type),
    )?;
    let passed = compare_arithmetic_values(&left_value.value, comparison.op, &right_value.value)?;
    Ok(ThresholdComparison {
        predicate: format!("{left} {} {right}", comp_op_label(comparison.op)),
        left,
        op: comp_op_label(comparison.op).to_string(),
        right,
        left_value: left_value.label,
        right_value: right_value.label,
        passed,
    })
}

fn arithmetic_expression_value(
    expression: &xlog_logic::ast::ArithExpr,
    bindings: &DiagnosticBindings,
) -> Result<DiagnosticScalar> {
    let evaluator_bindings = bindings
        .iter()
        .map(|(name, value)| (name.clone(), value.value.clone()))
        .collect::<HashMap<_, _>>();
    let value = evaluate_arithmetic_expression(expression, &evaluator_bindings)?;
    let scalar_type = value.scalar_type().ok_or_else(|| {
        XlogError::Compilation(
            "generated-rule arithmetic produced a value without a runtime scalar type".to_string(),
        )
    })?;
    Ok(DiagnosticScalar {
        label: arithmetic_value_label(&value),
        scalar_type,
        value,
    })
}

fn bound_scalar(term: &Term, bindings: &DiagnosticBindings) -> Option<DiagnosticScalar> {
    match term {
        Term::Variable(name) => bindings.get(name).cloned(),
        _ => None,
    }
}

fn comparison_scalar(
    term: &Term,
    bound: Option<&DiagnosticScalar>,
    peer_type: Option<ScalarType>,
) -> Result<DiagnosticScalar> {
    if let Some(bound) = bound {
        return Ok(bound.clone());
    }
    if matches!(term, Term::Variable(_)) {
        return Err(XlogError::Compilation(format!(
            "Unbound variable {} in generated-rule comparison",
            format_term(term)
        )));
    }
    let value = if let Some(expected) = peer_type {
        ArithmeticValue::from_typed_term(term, expected)?
    } else {
        ArithmeticValue::from_term(term)?
    };
    let scalar_type = value.scalar_type().ok_or_else(|| {
        XlogError::Compilation(
            "generated-rule comparison requires a runtime scalar type".to_string(),
        )
    })?;
    Ok(DiagnosticScalar {
        value,
        scalar_type,
        label: format_term(term),
    })
}

fn arithmetic_value_label(value: &ArithmeticValue) -> String {
    match value {
        ArithmeticValue::I32(value) => value.to_string(),
        ArithmeticValue::I64(value) => value.to_string(),
        ArithmeticValue::U32(value) => value.to_string(),
        ArithmeticValue::U64(value) => value.to_string(),
        ArithmeticValue::F32(value) => value.to_string(),
        ArithmeticValue::F64(value) => value.to_string(),
        ArithmeticValue::Bool(value) => value.to_string(),
        ArithmeticValue::Symbol(value) => symbol::resolve(*value),
        ArithmeticValue::String(value) => value.clone(),
    }
}

fn comp_op_label(op: CompOp) -> &'static str {
    match op {
        CompOp::Eq => "==",
        CompOp::Ne => "!=",
        CompOp::Lt => "<",
        CompOp::Le => "<=",
        CompOp::Gt => ">",
        CompOp::Ge => ">=",
    }
}

pub(super) fn print_generated_rule_diagnostics_json(entries: &[GeneratedRuleDiagnostic]) {
    println!("  \"generated_rule_diagnostics\": [");
    for (idx, entry) in entries.iter().enumerate() {
        let suffix = if idx + 1 == entries.len() { "" } else { "," };
        println!("    {{");
        println!(
            "      \"rule_head\": \"{}\",",
            json_escape(&entry.rule_head)
        );
        println!(
            "      \"source_relation\": \"{}\",",
            json_escape(&entry.source_relation)
        );
        println!("      \"row_decisions\": [");
        for (row_idx, row) in entry.row_decisions.iter().enumerate() {
            let row_suffix = if row_idx + 1 == entry.row_decisions.len() {
                ""
            } else {
                ","
            };
            println!("        {{");
            println!("          \"row_key\": \"{}\",", json_escape(&row.row_key));
            println!("          \"accepted\": {},", row.accepted);
            println!(
                "          \"failed_predicates\": {},",
                json_string_array(&row.failed_predicates)
            );
            println!("          \"threshold_comparisons\": [");
            for (comparison_idx, comparison) in row.threshold_comparisons.iter().enumerate() {
                let comparison_suffix = if comparison_idx + 1 == row.threshold_comparisons.len() {
                    ""
                } else {
                    ","
                };
                println!("            {{");
                println!(
                    "              \"predicate\": \"{}\",",
                    json_escape(&comparison.predicate)
                );
                println!(
                    "              \"left\": \"{}\",",
                    json_escape(&comparison.left)
                );
                println!("              \"op\": \"{}\",", json_escape(&comparison.op));
                println!(
                    "              \"right\": \"{}\",",
                    json_escape(&comparison.right)
                );
                println!(
                    "              \"left_value\": \"{}\",",
                    json_escape(&comparison.left_value)
                );
                println!(
                    "              \"right_value\": \"{}\",",
                    json_escape(&comparison.right_value)
                );
                println!("              \"passed\": {}", comparison.passed);
                println!("            }}{}", comparison_suffix);
            }
            println!("          ],");
            println!(
                "          \"aggregate_inputs\": {}",
                json_string_array(&row.aggregate_inputs)
            );
            println!("        }}{}", row_suffix);
        }
        println!("      ]");
        println!("    }}{}", suffix);
    }
    println!("  ]");
}
