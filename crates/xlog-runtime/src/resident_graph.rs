//! Device-resident execution for bounded ordinary Datalog plans.
//!
//! The resident route is deliberately fail-closed.  Plan inspection records
//! every physical route occurrence before setup allocates workspace or
//! enqueues CUDA work.  Unsupported shapes remain on the existing executor.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::Arc;

#[cfg(all(test, feature = "resident-graph-tests"))]
use std::cell::Cell;

use xlog_core::{RelId, Result, ScalarType, Schema};
use xlog_ir::{ConstValue, ExecutionPlan, Expr, JoinType, ProjectExpr, RirNode};

const RESIDENT_GRAPH_MAX_INTERMEDIATE_ARITY: usize = 17;

/// Schemas keyed by the relation identities carried by compiled scan nodes.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphSchemaCatalog {
    by_relation: HashMap<RelId, Vec<(String, Schema)>>,
}

impl ResidentGraphSchemaCatalog {
    /// Builds a catalog from compiler-assigned names, relation ids, and schemas.
    pub fn from_named_schemas(entries: impl IntoIterator<Item = (String, RelId, Schema)>) -> Self {
        let mut by_relation: HashMap<RelId, Vec<(String, Schema)>> = HashMap::new();
        for (name, relation, schema) in entries {
            by_relation
                .entry(relation)
                .or_default()
                .push((name, schema));
        }
        for aliases in by_relation.values_mut() {
            aliases.sort_by(|left, right| left.0.cmp(&right.0));
            aliases.dedup();
        }
        Self { by_relation }
    }

    fn descriptor(&self, relation: RelId) -> Option<String> {
        let aliases = self.by_relation.get(&relation)?;
        Some(
            aliases
                .iter()
                .map(|(name, schema)| format!("{name}={schema:#?}"))
                .collect::<Vec<_>>()
                .join("|"),
        )
    }

    pub(crate) fn schema(&self, relation: RelId) -> Option<&Schema> {
        let aliases = self.by_relation.get(&relation)?;
        let (_, first) = aliases.first()?;
        aliases
            .iter()
            .all(|(_, schema)| schema == first)
            .then_some(first)
    }
}

/// A reason why a complete plan cannot use the resident conditional graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentGraphDeclineReason {
    /// A scan relation has no compiler schema identity.
    MissingScanSchema {
        /// Relation whose scan lacks a compiler schema identity.
        relation: RelId,
    },
    /// A physical node has no resident implementation.
    UnsupportedNode {
        /// Stable path to the unsupported node within the physical plan.
        path: String,
        /// Physical node kind that has no resident implementation.
        node: &'static str,
    },
    /// The resident route supports only inner and semi joins.
    UnsupportedJoin {
        /// Stable path to the unsupported join within the physical plan.
        path: String,
        /// Join semantics that the resident route cannot execute.
        join_type: JoinType,
    },
    /// The caller requested a complete relation store, which cannot be staged
    /// as a query-only transaction.
    FullStoreRequested,
    /// The compiled program uses epistemic or another non-ordinary semantics.
    NonOrdinaryPlan,
    /// A caller input is imported or belongs to a different memory manager, so
    /// its lifetime cannot be bound to this resident transaction.
    ImportedInputUnsupported {
        /// Imported relation whose ownership cannot join the transaction.
        relation: String,
    },
    /// A scanned source lacks a current provider-produced full-row set proof.
    SourceSetUncertified {
        /// Source relation without a current full-row set certificate.
        relation: String,
    },
    /// The CUDA driver cannot construct a conditional WHILE graph.
    ConditionalGraphUnavailable {
        /// Driver capability failure reported during conditional-graph setup.
        detail: String,
    },
    /// Setup cannot reserve a bounded workspace without exceeding the budget.
    WorkspaceUnbounded {
        /// Workspace bound or reservation failure reported during preflight.
        detail: String,
    },
}

/// Complete prelaunch proof of the physical routes selected for a plan.
#[derive(Debug, Clone)]
pub struct ResidentGraphRouteCertificate {
    covered_route_descriptors: BTreeSet<String>,
    declines: Vec<ResidentGraphDeclineReason>,
    schema_catalog: ResidentGraphSchemaCatalog,
    plan_fingerprint: u64,
}

/// A route certificate sealed to the exact immutable plan allocation it inspected.
///
/// Preparation through this value cannot accidentally pair a certificate with a
/// different plan. The owned [`Arc`] also prevents safe mutation of the plan while
/// the seal remains live.
#[derive(Debug, Clone)]
pub struct ResidentGraphCertifiedPlan {
    plan: Arc<ExecutionPlan>,
    certificate: ResidentGraphRouteCertificate,
}

impl ResidentGraphCertifiedPlan {
    /// Inspect and seal one immutable plan allocation.
    pub fn inspect(plan: Arc<ExecutionPlan>, catalog: &ResidentGraphSchemaCatalog) -> Result<Self> {
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, catalog)?;
        Ok(Self { plan, certificate })
    }

    /// Return the exact plan inspected by this seal.
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Return the certificate produced for the sealed plan.
    pub fn certificate(&self) -> &ResidentGraphRouteCertificate {
        &self.certificate
    }
}

#[cfg(all(test, feature = "resident-graph-tests"))]
thread_local! {
    static RESIDENT_ROUTE_INSPECTION_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(test, feature = "resident-graph-tests"))]
pub(crate) fn reset_resident_route_inspection_count() {
    RESIDENT_ROUTE_INSPECTION_COUNT.with(|count| count.set(0));
}

#[cfg(all(test, feature = "resident-graph-tests"))]
pub(crate) fn resident_route_inspection_count() -> usize {
    RESIDENT_ROUTE_INSPECTION_COUNT.with(Cell::get)
}

impl ResidentGraphRouteCertificate {
    /// Inspects every explicit and implicit route in deterministic plan order.
    pub fn inspect(plan: &ExecutionPlan, catalog: &ResidentGraphSchemaCatalog) -> Result<Self> {
        #[cfg(all(test, feature = "resident-graph-tests"))]
        RESIDENT_ROUTE_INSPECTION_COUNT.with(|count| count.set(count.get() + 1));
        let mut certificate = Self {
            covered_route_descriptors: BTreeSet::new(),
            declines: Vec::new(),
            schema_catalog: catalog.clone(),
            plan_fingerprint: 0,
        };
        certificate
            .covered_route_descriptors
            .insert(format!("plan;scc_count={}", plan.sccs.len()));
        for (scc_index, scc) in plan.sccs.iter().enumerate() {
            certificate.covered_route_descriptors.insert(format!(
                "plan;scc={scc_index};id={};recursive={};predicate_count={}",
                scc.id,
                scc.is_recursive,
                scc.predicates.len()
            ));
            for (predicate_index, predicate) in scc.predicates.iter().enumerate() {
                certificate.covered_route_descriptors.insert(format!(
                    "plan;scc={scc_index};predicate={predicate_index};name={predicate:?}"
                ));
            }
        }
        certificate
            .covered_route_descriptors
            .insert(format!("plan;stratum_count={}", plan.strata.len()));
        for (stratum_index, stratum) in plan.strata.iter().enumerate() {
            certificate.covered_route_descriptors.insert(format!(
                "plan;stratum={stratum_index};id={};scc_count={}",
                stratum.id,
                stratum.sccs.len()
            ));
            for (scc_position, scc_id) in stratum.sccs.iter().enumerate() {
                certificate.covered_route_descriptors.insert(format!(
                    "plan;stratum={stratum_index};scc={scc_position};id={scc_id}"
                ));
            }
        }
        certificate.covered_route_descriptors.insert(format!(
            "plan;rules_by_scc_count={}",
            plan.rules_by_scc.len()
        ));
        certificate.covered_route_descriptors.insert(format!(
            "plan;generated_query_rule_count={}",
            plan.generated_query_rules.len()
        ));
        for (position, provenance) in plan.generated_query_rules.iter().enumerate() {
            certificate.covered_route_descriptors.insert(format!(
                "plan;generated_query_rule={position};query_index={};scc={};rule={}",
                provenance.query_index, provenance.scc_index, provenance.rule_index
            ));
        }
        for (scc_index, rules) in plan.rules_by_scc.iter().enumerate() {
            certificate.covered_route_descriptors.insert(format!(
                "plan;rules_by_scc={scc_index};rule_count={}",
                rules.len()
            ));
            for (rule_index, rule) in rules.iter().enumerate() {
                certificate.covered_route_descriptors.insert(format!(
                    "plan;rules_by_scc={scc_index};rule={rule_index};head={:?};meta={:#?};body={:#?}",
                    rule.head, rule.meta, rule.body
                ));
            }
        }
        certificate
            .covered_route_descriptors
            .insert(format!("plan;est_memory_peak={}", plan.est_memory_peak));
        certificate
            .covered_route_descriptors
            .insert(format!("plan;rel_arity_count={}", plan.rel_arities.len()));
        let mut rel_arities = plan
            .rel_arities
            .iter()
            .map(|(&relation, &arity)| (relation, arity))
            .collect::<Vec<_>>();
        rel_arities.sort_unstable_by_key(|(relation, _)| *relation);
        for (position, (relation, arity)) in rel_arities.into_iter().enumerate() {
            certificate.covered_route_descriptors.insert(format!(
                "plan;rel_arity={position};relation={};arity={arity}",
                relation.0
            ));
        }
        for (scc_index, scc) in plan.sccs.iter().enumerate() {
            let Some(rules) = plan.rules_by_scc.get(scc_index) else {
                certificate
                    .declines
                    .push(ResidentGraphDeclineReason::UnsupportedNode {
                        path: format!("scc={scc_index}"),
                        node: "missing_rule_vector",
                    });
                continue;
            };
            for (rule_index, rule) in rules.iter().enumerate() {
                certificate.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    scc.is_recursive,
                    &rule.body,
                    "primary/root",
                );
                let identity = format!(
                    "scc={scc_index};rule={rule_index};head={};schema={:#?}",
                    rule.head, rule.meta.schema
                );
                certificate
                    .covered_route_descriptors
                    .insert(format!("{identity};implicit=rule_result_union"));
                certificate
                    .covered_route_descriptors
                    .insert(format!("{identity};implicit=full_row_dedup"));
                if scc.is_recursive {
                    certificate
                        .covered_route_descriptors
                        .insert(format!("{identity};implicit=novel_tuple_difference"));
                    certificate
                        .covered_route_descriptors
                        .insert(format!("{identity};implicit=device_convergence"));
                }
            }
        }
        certificate.plan_fingerprint = stable_descriptor_fingerprint(
            certificate
                .covered_route_descriptors
                .iter()
                .map(String::as_bytes),
        );
        Ok(certificate)
    }

    /// Whether every inspected route has a resident implementation.
    pub fn is_supported(&self) -> bool {
        self.declines.is_empty()
    }

    /// Deterministically ordered route occurrences covered by this proof.
    pub fn covered_route_descriptors(&self) -> &BTreeSet<String> {
        &self.covered_route_descriptors
    }

    /// Fail-closed reasons collected during inspection.
    pub fn declines(&self) -> &[ResidentGraphDeclineReason] {
        &self.declines
    }

    /// Stable binding between this certificate and every inspected physical
    /// route, expression, key position, schema, and implicit set operation.
    pub fn plan_fingerprint(&self) -> u64 {
        self.plan_fingerprint
    }

    /// Re-inspects a plan and proves that it is the exact plan certified by
    /// this value. Preparation performs this check before workspace allocation.
    pub fn matches_plan(&self, plan: &ExecutionPlan) -> Result<bool> {
        let inspected = Self::inspect(plan, &self.schema_catalog)?;
        Ok(inspected.plan_fingerprint == self.plan_fingerprint
            && inspected.covered_route_descriptors == self.covered_route_descriptors
            && inspected.declines == self.declines)
    }

    pub(crate) fn schema_for(&self, relation: RelId) -> Option<&Schema> {
        self.schema_catalog.schema(relation)
    }

    pub(crate) fn node_schema(&self, node: &RirNode) -> Option<Schema> {
        node_schema(&self.schema_catalog, node)
    }

    fn visit(
        &mut self,
        catalog: &ResidentGraphSchemaCatalog,
        scc_index: usize,
        rule_index: usize,
        recursive: bool,
        node: &RirNode,
        path: &str,
    ) {
        self.validate_node_route(catalog, node, path);
        let scan_schema = match node {
            RirNode::Scan { rel } => match catalog.descriptor(*rel) {
                Some(descriptor) => descriptor,
                None => {
                    self.declines
                        .push(ResidentGraphDeclineReason::MissingScanSchema { relation: *rel });
                    String::new()
                }
            },
            _ => String::new(),
        };
        self.covered_route_descriptors.insert(format!(
            "scc={scc_index};rule={rule_index};recursive={recursive};path={path};node={node:#?};scan_schema={scan_schema}"
        ));
        match node {
            RirNode::Unit | RirNode::Scan { .. } => {}
            RirNode::Filter { input, .. }
            | RirNode::Project { input, .. }
            | RirNode::Distinct { input, .. } => self.visit(
                catalog,
                scc_index,
                rule_index,
                recursive,
                input,
                &format!("{path}/input"),
            ),
            RirNode::Join {
                left,
                right,
                join_type,
                ..
            } => {
                if !matches!(join_type, JoinType::Inner | JoinType::Semi) {
                    self.declines
                        .push(ResidentGraphDeclineReason::UnsupportedJoin {
                            path: path.to_string(),
                            join_type: *join_type,
                        });
                }
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    left,
                    &format!("{path}/left"),
                );
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    right,
                    &format!("{path}/right"),
                );
            }
            RirNode::ChainJoin {
                left,
                right,
                fallback,
                ..
            } => {
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    left,
                    &format!("{path}/primary/left"),
                );
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    right,
                    &format!("{path}/primary/right"),
                );
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    fallback,
                    &format!("{path}/alternative/captured_fallback"),
                );
            }
            RirNode::Union { inputs } => {
                for (index, input) in inputs.iter().enumerate() {
                    self.visit(
                        catalog,
                        scc_index,
                        rule_index,
                        recursive,
                        input,
                        &format!("{path}/input[{index}]"),
                    );
                }
            }
            RirNode::Diff { left, right } => {
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    left,
                    &format!("{path}/left"),
                );
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    right,
                    &format!("{path}/right"),
                );
            }
            RirNode::Fixpoint {
                base,
                recursive: step,
                ..
            } => {
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    base,
                    &format!("{path}/base"),
                );
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    step,
                    &format!("{path}/recursive"),
                );
            }
            RirNode::MultiWayJoin {
                inputs, fallback, ..
            } => {
                self.declines
                    .push(ResidentGraphDeclineReason::UnsupportedNode {
                        path: path.to_string(),
                        node: "multi_way_join",
                    });
                for (index, input) in inputs.iter().enumerate() {
                    self.visit(
                        catalog,
                        scc_index,
                        rule_index,
                        recursive,
                        input,
                        &format!("{path}/primary/input[{index}]"),
                    );
                }
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    fallback,
                    &format!("{path}/alternative/captured_fallback"),
                );
            }
            RirNode::GroupBy { input, .. } => {
                self.declines
                    .push(ResidentGraphDeclineReason::UnsupportedNode {
                        path: path.to_string(),
                        node: "group_by",
                    });
                self.visit(
                    catalog,
                    scc_index,
                    rule_index,
                    recursive,
                    input,
                    &format!("{path}/input"),
                );
            }
            RirNode::TensorMaskedJoin { .. } => {
                self.declines
                    .push(ResidentGraphDeclineReason::UnsupportedNode {
                        path: path.to_string(),
                        node: "tensor_masked_join",
                    })
            }
        }
    }

    fn validate_node_route(
        &mut self,
        catalog: &ResidentGraphSchemaCatalog,
        node: &RirNode,
        path: &str,
    ) {
        let schema = node_schema(catalog, node);
        let unsupported = schema
            .as_ref()
            .is_some_and(|schema| schema.arity() > RESIDENT_GRAPH_MAX_INTERMEDIATE_ARITY)
            .then_some("intermediate_arity")
            .or_else(|| {
                schema
                    .as_ref()
                    .is_some_and(|schema| {
                        schema
                            .columns
                            .iter()
                            .any(|(_, scalar)| !resident_scalar_supported(*scalar))
                    })
                    .then_some("scalar_type")
            })
            .or_else(|| match node {
                RirNode::Filter { input, predicate } => node_schema(catalog, input)
                    .filter(|schema| resident_predicate_supported(predicate, schema))
                    .is_none()
                    .then_some("filter_expression"),
                RirNode::Project { input, columns } => node_schema(catalog, input)
                    .filter(|schema| resident_projection_supported(columns, schema))
                    .is_none()
                    .then_some("project_expression"),
                RirNode::Join {
                    left,
                    right,
                    left_keys,
                    right_keys,
                    join_type: JoinType::Inner | JoinType::Semi,
                } => {
                    let compatible = (|| {
                        let left_schema = node_schema(catalog, left)?;
                        let right_schema = node_schema(catalog, right)?;
                        if left_keys.len() != 1 || right_keys.len() != 1 {
                            return None;
                        }
                        let left_ty = left_schema.column_type(left_keys[0])?;
                        let right_ty = right_schema.column_type(right_keys[0])?;
                        (left_ty == right_ty && resident_scalar_supported(left_ty)).then_some(())
                    })()
                    .is_some();
                    (!compatible).then_some("join_key_layout")
                }
                RirNode::ChainJoin {
                    left,
                    right,
                    left_key,
                    right_key,
                    output_columns,
                    ..
                } => {
                    let compatible = (|| {
                        let left_schema = node_schema(catalog, left)?;
                        let right_schema = node_schema(catalog, right)?;
                        let left_ty = left_schema.column_type(*left_key)?;
                        let right_ty = right_schema.column_type(*right_key)?;
                        if left_ty != right_ty || !resident_scalar_supported(left_ty) {
                            return None;
                        }
                        let joined = joined_schema(&left_schema, &right_schema);
                        resident_projection_supported(output_columns, &joined).then_some(())
                    })()
                    .is_some();
                    (!compatible).then_some("chain_join_layout")
                }
                RirNode::Distinct { input, key_cols } => node_schema(catalog, input)
                    .filter(|schema| key_cols.iter().copied().eq(0..schema.arity()))
                    .is_none()
                    .then_some("partial_row_distinct"),
                _ => None,
            });
        if let Some(node) = unsupported {
            self.declines
                .push(ResidentGraphDeclineReason::UnsupportedNode {
                    path: path.to_string(),
                    node,
                });
        }
    }
}

fn resident_scalar_supported(scalar: ScalarType) -> bool {
    matches!(
        scalar,
        ScalarType::Symbol | ScalarType::U32 | ScalarType::U64
    )
}

fn constant_scalar(value: &ConstValue) -> ScalarType {
    match value {
        ConstValue::U32(_) => ScalarType::U32,
        ConstValue::U64(_) => ScalarType::U64,
        ConstValue::I32(_) => ScalarType::I32,
        ConstValue::I64(_) => ScalarType::I64,
        ConstValue::F32(_) => ScalarType::F32,
        ConstValue::F64(_) => ScalarType::F64,
        ConstValue::Bool(_) => ScalarType::Bool,
        ConstValue::Symbol(_) => ScalarType::Symbol,
    }
}

fn scalar_expression_type(expression: &Expr, schema: &Schema) -> Option<ScalarType> {
    match expression {
        Expr::Column(index) => schema.column_type(*index),
        Expr::Const(value) => Some(constant_scalar(value)),
        _ => None,
    }
}

fn resident_predicate_supported(expression: &Expr, schema: &Schema) -> bool {
    match expression {
        Expr::Compare { left, right, .. } => {
            let Some(left_ty) = scalar_expression_type(left, schema) else {
                return false;
            };
            let Some(right_ty) = scalar_expression_type(right, schema) else {
                return false;
            };
            left_ty == right_ty && resident_scalar_supported(left_ty)
        }
        Expr::And(expressions) => {
            !expressions.is_empty()
                && expressions
                    .iter()
                    .all(|expression| resident_predicate_supported(expression, schema))
        }
        _ => false,
    }
}

fn resident_projection_supported(expressions: &[ProjectExpr], schema: &Schema) -> bool {
    expressions.iter().all(|expression| match expression {
        ProjectExpr::Column(index) => schema
            .column_type(*index)
            .is_some_and(resident_scalar_supported),
        ProjectExpr::Computed(Expr::Const(value), declared) => {
            *declared == constant_scalar(value) && resident_scalar_supported(*declared)
        }
        _ => false,
    })
}

fn joined_schema(left: &Schema, right: &Schema) -> Schema {
    Schema::new(
        left.columns
            .iter()
            .chain(&right.columns)
            .enumerate()
            .map(|(index, (_, ty))| (format!("column_{index}"), *ty))
            .collect(),
    )
}

pub(crate) fn node_schema(catalog: &ResidentGraphSchemaCatalog, node: &RirNode) -> Option<Schema> {
    match node {
        RirNode::Unit => Some(Schema::new(Vec::new())),
        RirNode::Scan { rel } => catalog.schema(*rel).cloned(),
        RirNode::Filter { input, .. } | RirNode::Distinct { input, .. } => {
            node_schema(catalog, input)
        }
        RirNode::Project { input, columns } => {
            let input = node_schema(catalog, input)?;
            let output = columns
                .iter()
                .enumerate()
                .map(|(index, expression)| {
                    let scalar = match expression {
                        ProjectExpr::Column(column) => input.column_type(*column),
                        ProjectExpr::Computed(_, scalar) => Some(*scalar),
                    };
                    scalar.map(|scalar| (format!("column_{index}"), scalar))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Schema::new(output))
        }
        RirNode::Join {
            left,
            right,
            join_type,
            ..
        } => {
            let left = node_schema(catalog, left)?;
            if matches!(join_type, JoinType::Semi | JoinType::Anti) {
                Some(left)
            } else {
                Some(joined_schema(&left, &node_schema(catalog, right)?))
            }
        }
        RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
            node_schema(catalog, fallback)
        }
        RirNode::Union { inputs } => inputs.first().and_then(|input| node_schema(catalog, input)),
        RirNode::Diff { left, .. } => node_schema(catalog, left),
        RirNode::Fixpoint { base, .. } => node_schema(catalog, base),
        RirNode::GroupBy { .. } | RirNode::TensorMaskedJoin { .. } => None,
    }
}

fn stable_descriptor_fingerprint<'a>(descriptors: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for descriptor in descriptors {
        for byte in (descriptor.len() as u64)
            .to_le_bytes()
            .iter()
            .chain(descriptor)
        {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Device-written terminal status for a resident graph transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentGraphDeviceStatus {
    /// The graph converged and staged output is valid.
    Success {
        /// Device-observed recursive iterations completed before convergence.
        iterations: u32,
    },
    /// The exact configured iteration limit was exhausted.
    IterationLimit {
        /// Configured maximum recursive iterations.
        limit: u32,
        /// Iterations completed when the device stopped the graph.
        completed: u32,
    },
    /// An operator's exact output exceeded its reserved row capacity.
    CapacityOverflow {
        /// Physical operator that exceeded its reserved row count.
        op_id: u32,
        /// Exact row count required by the operator.
        required: u64,
        /// Reserved row capacity available to the operator.
        capacity: u64,
    },
    /// A bounded device resource was insufficient.
    ResourceExhausted {
        /// Physical operator that exhausted the bounded resource.
        op_id: u32,
        /// Stable name of the exhausted device resource.
        resource: &'static str,
        /// Exact resource quantity required by the operator.
        required: u64,
        /// Reserved resource quantity available to the operator.
        capacity: u64,
    },
}

/// Typed error decoded from the graph's single terminal receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentGraphExecutionError {
    /// A complete prelaunch inspection selected the existing GPU route instead.
    Declined(ResidentGraphDeclineReason),
    /// The exact configured iteration limit was exhausted.
    IterationLimit {
        /// Configured maximum recursive iterations.
        limit: u32,
        /// Iterations completed before the device reported exhaustion.
        completed: u32,
    },
    /// An operator's exact output exceeded its reserved row capacity.
    CapacityOverflow {
        /// Physical operator that exceeded its reserved row count.
        op_id: u32,
        /// Exact row count required by the operator.
        required: u64,
        /// Reserved row capacity available to the operator.
        capacity: u64,
    },
    /// A bounded device resource was insufficient.
    ResourceExhausted {
        /// Physical operator that exhausted the bounded resource.
        op_id: u32,
        /// Stable name of the exhausted device resource.
        resource: &'static str,
        /// Exact resource quantity required by the operator.
        required: u64,
        /// Reserved resource quantity available to the operator.
        capacity: u64,
    },
    /// Setup or execution failed before a valid terminal status existed.
    Runtime(String),
}

impl fmt::Display for ResidentGraphExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ResidentGraphExecutionError {}

/// Test-only device-kernel status request.  Production execution never writes
/// terminal status from the host.
#[derive(Debug, Clone)]
pub struct ResidentGraphDeviceStatusTestInjection {
    pub(crate) after_op: u32,
    pub(crate) status: ResidentGraphDeviceStatus,
}

impl ResidentGraphDeviceStatusTestInjection {
    /// Requests a device status-writer kernel after the indexed physical op.
    pub fn device_kernel_after_op(after_op: u32, status: ResidentGraphDeviceStatus) -> Self {
        Self { after_op, status }
    }
}

/// Prelaunch options for a resident transaction.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphPrepareOptions {
    pub(crate) test_device_status: Option<ResidentGraphDeviceStatusTestInjection>,
    pub(crate) latency_diagnostic_sample: Option<u64>,
}

impl ResidentGraphPrepareOptions {
    /// Adds a device-written test status without permitting host injection.
    pub fn with_test_device_status(
        mut self,
        injection: ResidentGraphDeviceStatusTestInjection,
    ) -> Self {
        self.test_device_status = Some(injection);
        self
    }

    /// Enables cold-path prepare timing for one explicitly numbered diagnostic sample.
    #[doc(hidden)]
    pub fn with_latency_diagnostic_sample(mut self, sample: u64) -> Self {
        self.latency_diagnostic_sample = Some(sample);
        self
    }
}

/// Copyable, opt-in cold-path timings captured before a resident graph launches.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentGraphPrepareDiagnosticSnapshot {
    pub(crate) sample: u64,
    pub(crate) total_ns: u64,
    pub(crate) admission_and_source_snapshot_ns: u64,
    pub(crate) execution_domain_and_build_setup_ns: u64,
    pub(crate) logical_schedule_planning_ns: u64,
    pub(crate) manifest_compact_construction_ns: u64,
    pub(crate) schedule_lowering_ns: u64,
    pub(crate) reservation_ns: u64,
    pub(crate) relation_preparation_ns: u64,
    pub(crate) count_initialization_ns: u64,
    pub(crate) workspace_preparation_ns: u64,
    pub(crate) metadata_binding_construction_ns: u64,
    pub(crate) metadata_preparation_ns: u64,
    pub(crate) reservation_validation_and_release_ns: u64,
    pub(crate) pinned_receipt_ns: u64,
    pub(crate) graph_body_capture_ns: u64,
    pub(crate) graph_instantiate_ns: u64,
    pub(crate) validation_owner_assembly_ns: u64,
    pub(crate) unattributed_ns: u64,
    pub(crate) required_reservation_bytes: u64,
    pub(crate) logical_relation_values: u64,
    pub(crate) physical_relation_slots: u64,
    pub(crate) relation_device_allocation_calls: u64,
    pub(crate) relation_reserved_bytes: u64,
    pub(crate) relation_slot_preparation_ns_max: u64,
    pub(crate) count_memset_calls: u64,
    pub(crate) count_memset_bytes: u64,
    pub(crate) count_initialization_ns_max: u64,
    pub(crate) workspace_provider_calls: u64,
    pub(crate) workspace_reserved_bytes: u64,
    pub(crate) filter_scratch_preparation_ns: u64,
    pub(crate) filter_scratch_reserved_bytes: u64,
    pub(crate) set_workspace_preparation_ns: u64,
    pub(crate) set_workspace_reserved_bytes: u64,
    pub(crate) join_workspace_preparation_ns: u64,
    pub(crate) join_workspace_reserved_bytes: u64,
    pub(crate) control_preparation_ns: u64,
    pub(crate) control_reserved_bytes: u64,
    pub(crate) metadata_provider_calls: u64,
    pub(crate) metadata_reserved_bytes: u64,
    pub(crate) metadata_initial_htod_calls: u64,
    pub(crate) metadata_initial_htod_bytes: u64,
    pub(crate) device_trace_preparation_ns: u64,
    pub(crate) device_trace_reserved_bytes: u64,
    pub(crate) device_trace_initial_htod_calls: u64,
    pub(crate) device_trace_initial_htod_bytes: u64,
    pub(crate) schema_winners_preparation_ns: u64,
    pub(crate) schema_winners_reserved_bytes: u64,
    pub(crate) schema_winners_initial_htod_calls: u64,
    pub(crate) schema_winners_initial_htod_bytes: u64,
    pub(crate) receipt_preparation_ns: u64,
    pub(crate) receipt_reserved_bytes: u64,
    pub(crate) receipt_initial_htod_calls: u64,
    pub(crate) receipt_initial_htod_bytes: u64,
    pub(crate) schedule_program_preparation_ns: u64,
    pub(crate) schedule_program_reserved_bytes: u64,
    pub(crate) schedule_program_initial_htod_calls: u64,
    pub(crate) schedule_program_initial_htod_bytes: u64,
    pub(crate) compact_ops: u64,
    pub(crate) compact_waves: u64,
    pub(crate) compact_regions: u64,
    pub(crate) conditional_regions: u64,
    pub(crate) parent_graph_nodes: u64,
    pub(crate) conditional_body_nodes: u64,
}

impl ResidentGraphPrepareDiagnosticSnapshot {
    /// Format one stable, machine-tokenizable diagnostic line without emitting it.
    #[doc(hidden)]
    pub fn format_line(self) -> String {
        let Self {
            sample,
            total_ns,
            admission_and_source_snapshot_ns,
            execution_domain_and_build_setup_ns,
            logical_schedule_planning_ns,
            manifest_compact_construction_ns,
            schedule_lowering_ns,
            reservation_ns,
            relation_preparation_ns,
            count_initialization_ns,
            workspace_preparation_ns,
            metadata_binding_construction_ns,
            metadata_preparation_ns,
            reservation_validation_and_release_ns,
            pinned_receipt_ns,
            graph_body_capture_ns,
            graph_instantiate_ns,
            validation_owner_assembly_ns,
            unattributed_ns,
            required_reservation_bytes,
            logical_relation_values,
            physical_relation_slots,
            relation_device_allocation_calls,
            relation_reserved_bytes,
            relation_slot_preparation_ns_max,
            count_memset_calls,
            count_memset_bytes,
            count_initialization_ns_max,
            workspace_provider_calls,
            workspace_reserved_bytes,
            filter_scratch_preparation_ns,
            filter_scratch_reserved_bytes,
            set_workspace_preparation_ns,
            set_workspace_reserved_bytes,
            join_workspace_preparation_ns,
            join_workspace_reserved_bytes,
            control_preparation_ns,
            control_reserved_bytes,
            metadata_provider_calls,
            metadata_reserved_bytes,
            metadata_initial_htod_calls,
            metadata_initial_htod_bytes,
            device_trace_preparation_ns,
            device_trace_reserved_bytes,
            device_trace_initial_htod_calls,
            device_trace_initial_htod_bytes,
            schema_winners_preparation_ns,
            schema_winners_reserved_bytes,
            schema_winners_initial_htod_calls,
            schema_winners_initial_htod_bytes,
            receipt_preparation_ns,
            receipt_reserved_bytes,
            receipt_initial_htod_calls,
            receipt_initial_htod_bytes,
            schedule_program_preparation_ns,
            schedule_program_reserved_bytes,
            schedule_program_initial_htod_calls,
            schedule_program_initial_htod_bytes,
            compact_ops,
            compact_waves,
            compact_regions,
            conditional_regions,
            parent_graph_nodes,
            conditional_body_nodes,
        } = self;
        format!(
            "resident prepare phases: sample={sample} total_ns={total_ns} admission_and_source_snapshot_ns={admission_and_source_snapshot_ns} execution_domain_and_build_setup_ns={execution_domain_and_build_setup_ns} logical_schedule_planning_ns={logical_schedule_planning_ns} manifest_compact_construction_ns={manifest_compact_construction_ns} schedule_lowering_ns={schedule_lowering_ns} reservation_ns={reservation_ns} relation_preparation_ns={relation_preparation_ns} count_initialization_ns={count_initialization_ns} workspace_preparation_ns={workspace_preparation_ns} metadata_binding_construction_ns={metadata_binding_construction_ns} metadata_preparation_ns={metadata_preparation_ns} reservation_validation_and_release_ns={reservation_validation_and_release_ns} pinned_receipt_ns={pinned_receipt_ns} graph_body_capture_ns={graph_body_capture_ns} graph_instantiate_ns={graph_instantiate_ns} validation_owner_assembly_ns={validation_owner_assembly_ns} unattributed_ns={unattributed_ns} required_reservation_bytes={required_reservation_bytes} logical_relation_values={logical_relation_values} physical_relation_slots={physical_relation_slots} relation_device_allocation_calls={relation_device_allocation_calls} relation_reserved_bytes={relation_reserved_bytes} relation_slot_preparation_ns_max={relation_slot_preparation_ns_max} count_memset_calls={count_memset_calls} count_memset_bytes={count_memset_bytes} count_initialization_ns_max={count_initialization_ns_max} workspace_provider_calls={workspace_provider_calls} workspace_reserved_bytes={workspace_reserved_bytes} filter_scratch_preparation_ns={filter_scratch_preparation_ns} filter_scratch_reserved_bytes={filter_scratch_reserved_bytes} set_workspace_preparation_ns={set_workspace_preparation_ns} set_workspace_reserved_bytes={set_workspace_reserved_bytes} join_workspace_preparation_ns={join_workspace_preparation_ns} join_workspace_reserved_bytes={join_workspace_reserved_bytes} control_preparation_ns={control_preparation_ns} control_reserved_bytes={control_reserved_bytes} metadata_provider_calls={metadata_provider_calls} metadata_reserved_bytes={metadata_reserved_bytes} metadata_initial_htod_calls={metadata_initial_htod_calls} metadata_initial_htod_bytes={metadata_initial_htod_bytes} device_trace_preparation_ns={device_trace_preparation_ns} device_trace_reserved_bytes={device_trace_reserved_bytes} device_trace_initial_htod_calls={device_trace_initial_htod_calls} device_trace_initial_htod_bytes={device_trace_initial_htod_bytes} schema_winners_preparation_ns={schema_winners_preparation_ns} schema_winners_reserved_bytes={schema_winners_reserved_bytes} schema_winners_initial_htod_calls={schema_winners_initial_htod_calls} schema_winners_initial_htod_bytes={schema_winners_initial_htod_bytes} receipt_preparation_ns={receipt_preparation_ns} receipt_reserved_bytes={receipt_reserved_bytes} receipt_initial_htod_calls={receipt_initial_htod_calls} receipt_initial_htod_bytes={receipt_initial_htod_bytes} schedule_program_preparation_ns={schedule_program_preparation_ns} schedule_program_reserved_bytes={schedule_program_reserved_bytes} schedule_program_initial_htod_calls={schedule_program_initial_htod_calls} schedule_program_initial_htod_bytes={schedule_program_initial_htod_bytes} compact_ops={compact_ops} compact_waves={compact_waves} compact_regions={compact_regions} conditional_regions={conditional_regions} parent_graph_nodes={parent_graph_nodes} conditional_body_nodes={conditional_body_nodes} deallocation_calls=unavailable"
        )
    }
}

#[cfg(test)]
fn resident_graph_prepare_diagnostic_line(
    snapshot: Option<ResidentGraphPrepareDiagnosticSnapshot>,
) -> Option<String> {
    snapshot.map(ResidentGraphPrepareDiagnosticSnapshot::format_line)
}

/// Runtime route selected for one ordinary evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentGraphSelectionKind {
    /// The existing host-dispatched GPU executor ran the plan.
    ExistingGpu,
    /// A device-controlled conditional CUDA graph ran the plan.
    ResidentConditionalGraph,
}

/// Core-loop host transfer counters.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphCoreTransferStats {
    /// Host-to-device calls observed by the tracked memory runtime.
    pub tracked_htod_calls: u64,
    /// Host-to-device bytes observed by the tracked memory runtime.
    pub tracked_htod_bytes: u64,
    /// Device-to-host calls observed by the tracked memory runtime.
    pub tracked_dtoh_calls: u64,
    /// Device-to-host bytes observed by the tracked memory runtime.
    pub tracked_dtoh_bytes: u64,
    /// Device-to-host calls issued through provider metadata operations.
    pub provider_dtoh_calls: u64,
    /// Metadata device-to-host calls not attributable to the tracked runtime.
    pub untracked_metadata_dtoh_calls: u64,
}

/// The one bounded observation after the terminal synchronization.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphFinalObservationStats {
    /// Device-to-host calls used to read the terminal observation.
    pub dtoh_calls: u64,
    /// Device-to-host bytes used to read the terminal observation.
    pub dtoh_bytes: u64,
    /// Pinned host receipts used for the terminal observation.
    pub pinned_receipts: u64,
}

/// CUDA-event timing resolved after the graph completes.
#[derive(Debug, Clone, Default)]
pub struct ResidentGraphDeferredProfile {
    /// Resident scan and filter invocations covered by deferred CUDA timing.
    pub timed_scan_filter_invocations: u64,
    /// Device elapsed time resolved from CUDA events after completion.
    pub device_elapsed_ns: u64,
    /// Host synchronization time excluded from device execution timing.
    pub final_sync_misattributed_ns: u64,
}

/// Truthful telemetry for resident selection, execution, and decline.
#[derive(Debug, Clone)]
pub struct ResidentGraphExecutionStats {
    /// Runtime route selected for the evaluation.
    pub selection: ResidentGraphSelectionKind,
    /// Preflight decline reason when the existing GPU route was selected.
    pub decline: Option<ResidentGraphDeclineReason>,
    /// Conditional resident graph launches performed by this evaluation.
    pub conditional_graph_launches: u64,
    /// Host synchronizations performed to obtain terminal status.
    pub terminal_synchronizations: u64,
    /// Fixpoint iterations controlled by the host during the core loop.
    pub host_iterations: u64,
    /// Host allocations performed during the resident core loop.
    pub host_allocations: u64,
    /// Terminal statuses injected by the host rather than written by a device kernel.
    pub host_status_injections: u64,
    /// Deterministic device-to-host transfer contract violations.
    pub deterministic_d2h_violations: u64,
    /// Scan operations dispatched individually by the host.
    pub host_dispatched_scan_ops: u64,
    /// Filter operations dispatched individually by the host.
    pub host_dispatched_filter_ops: u64,
    /// Physical Scan nodes executed by the resident device graph.
    pub device_scan_invocations: u64,
    /// Physical Filter nodes executed by the resident device graph.
    pub device_filter_invocations: u64,
    /// Logical Scan count for the selected dependency-closed plan after
    /// excluding recursive variants whose input delta was empty.
    pub semantic_scan_invocations: u64,
    /// Logical Filter count for the selected dependency-closed plan after
    /// excluding recursive variants whose input delta was empty.
    pub semantic_filter_invocations: u64,
    /// Relation-store mutations staged until terminal success is authoritative.
    pub staged_store_mutations: u64,
    /// CUDA-event measurements resolved after the graph completes.
    pub deferred_profile: ResidentGraphDeferredProfile,
    /// Host/device transfers observed inside the resident core loop.
    pub core_transfers: ResidentGraphCoreTransferStats,
    /// Single bounded terminal observation made after synchronization.
    pub final_observation: ResidentGraphFinalObservationStats,
}

impl ResidentGraphExecutionStats {
    /// Telemetry for a call that deliberately remained on the existing GPU path.
    pub fn declined(reason: ResidentGraphDeclineReason) -> Self {
        Self {
            selection: ResidentGraphSelectionKind::ExistingGpu,
            decline: Some(reason),
            conditional_graph_launches: 0,
            terminal_synchronizations: 0,
            host_iterations: 0,
            host_allocations: 0,
            host_status_injections: 0,
            deterministic_d2h_violations: 0,
            host_dispatched_scan_ops: 0,
            host_dispatched_filter_ops: 0,
            device_scan_invocations: 0,
            device_filter_invocations: 0,
            semantic_scan_invocations: 0,
            semantic_filter_invocations: 0,
            staged_store_mutations: 0,
            deferred_profile: ResidentGraphDeferredProfile::default(),
            core_transfers: ResidentGraphCoreTransferStats::default(),
            final_observation: ResidentGraphFinalObservationStats::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use xlog_ir::{CompiledRule, GeneratedQueryRuleProvenance, RirMeta, Scc, Stratum};

    fn schema(columns: &[(&str, ScalarType)]) -> Schema {
        Schema::new(
            columns
                .iter()
                .map(|(name, scalar)| ((*name).to_owned(), *scalar))
                .collect(),
        )
    }

    fn rule(head: &str, body: RirNode, columns: &[(&str, ScalarType)]) -> CompiledRule {
        CompiledRule {
            head: head.to_owned(),
            body,
            meta: RirMeta {
                schema: schema(columns),
                ..RirMeta::default()
            },
        }
    }

    fn chain_join() -> RirNode {
        RirNode::ChainJoin {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(1) }),
            left_key: 0,
            right_key: 0,
            output_columns: vec![ProjectExpr::Column(0)],
            fallback: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(1) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
        }
    }

    fn representative_plan() -> ExecutionPlan {
        let mut rel_arities = HashMap::new();
        rel_arities.insert(RelId(9), 0);
        rel_arities.insert(RelId(1), 2);
        ExecutionPlan {
            sccs: vec![
                Scc {
                    id: 10,
                    predicates: vec!["alpha".to_owned(), "beta".to_owned()],
                    is_recursive: true,
                },
                Scc {
                    id: 20,
                    predicates: Vec::new(),
                    is_recursive: false,
                },
                Scc {
                    id: 30,
                    predicates: vec!["gamma".to_owned()],
                    is_recursive: false,
                },
                Scc {
                    id: 40,
                    predicates: Vec::new(),
                    is_recursive: false,
                },
            ],
            strata: vec![
                Stratum {
                    id: 0,
                    sccs: vec![10],
                },
                Stratum {
                    id: 1,
                    sccs: Vec::new(),
                },
                Stratum {
                    id: 2,
                    sccs: vec![20, 30, 40],
                },
            ],
            rules_by_scc: vec![
                vec![
                    rule(
                        "alpha",
                        RirNode::Scan { rel: RelId(1) },
                        &[("left", ScalarType::U32), ("right", ScalarType::Symbol)],
                    ),
                    rule("beta", chain_join(), &[("value", ScalarType::U64)]),
                ],
                Vec::new(),
                vec![rule("gamma", RirNode::Unit, &[])],
                Vec::new(),
            ],
            generated_query_rules: vec![],
            est_memory_peak: 4_096,
            rel_arities,
        }
    }

    fn catalog() -> ResidentGraphSchemaCatalog {
        ResidentGraphSchemaCatalog::from_named_schemas([(
            "input".to_owned(),
            RelId(1),
            schema(&[("left", ScalarType::U32), ("right", ScalarType::Symbol)]),
        )])
    }

    fn assert_mutation_rejected(mutate: impl FnOnce(&mut ExecutionPlan)) {
        let plan = representative_plan();
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        let mut mutated = plan.clone();
        mutate(&mut mutated);
        assert!(!certificate.matches_plan(&mutated).unwrap());
    }

    fn assert_generated_query_provenance_mutation_rejected(
        mutate: impl FnOnce(&mut GeneratedQueryRuleProvenance),
    ) {
        let mut plan = representative_plan();
        plan.sccs[0].predicates[0] = "__xlog_query_0".to_owned();
        plan.rules_by_scc[0][0].head = "__xlog_query_0".to_owned();
        plan.generated_query_rules = vec![GeneratedQueryRuleProvenance {
            query_index: 0,
            scc_index: 0,
            rule_index: 0,
        }];
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        mutate(&mut plan.generated_query_rules[0]);
        assert!(!certificate.matches_plan(&plan).unwrap());
    }

    #[test]
    fn certified_plan_seals_the_exact_arc_and_certificate() {
        let plan = Arc::new(representative_plan());
        let certified = ResidentGraphCertifiedPlan::inspect(Arc::clone(&plan), &catalog()).unwrap();

        assert!(std::ptr::eq(Arc::as_ptr(&plan), certified.plan()));
        assert!(certified
            .certificate()
            .matches_plan(certified.plan())
            .unwrap());

        let mut independent_mutation = (*plan).clone();
        independent_mutation.est_memory_peak += 1;
        assert!(!certified
            .certificate()
            .matches_plan(&independent_mutation)
            .unwrap());
    }

    #[test]
    fn prepare_latency_diagnostics_are_explicit_and_sample_bound() {
        let disabled = ResidentGraphPrepareOptions::default();
        assert_eq!(disabled.latency_diagnostic_sample, None);

        let enabled = disabled.with_latency_diagnostic_sample(7);
        assert_eq!(enabled.latency_diagnostic_sample, Some(7));
    }

    #[test]
    fn prepare_latency_diagnostic_format_is_aligned_and_default_off_is_silent() {
        let mut snapshot = ResidentGraphPrepareDiagnosticSnapshot {
            sample: 17,
            total_ns: 101,
            admission_and_source_snapshot_ns: 102,
            execution_domain_and_build_setup_ns: 109,
            metadata_binding_construction_ns: 110,
            reservation_validation_and_release_ns: 111,
            relation_device_allocation_calls: 103,
            workspace_provider_calls: 104,
            metadata_initial_htod_calls: 105,
            metadata_initial_htod_bytes: 106,
            graph_instantiate_ns: 107,
            unattributed_ns: 108,
            ..ResidentGraphPrepareDiagnosticSnapshot::default()
        };
        snapshot.schema_winners_initial_htod_calls = 1;
        snapshot.schema_winners_initial_htod_bytes = 12;

        let line = resident_graph_prepare_diagnostic_line(Some(snapshot))
            .expect("enabled diagnostics produce one line");
        for expected in [
            "sample=17",
            "total_ns=101",
            "admission_and_source_snapshot_ns=102",
            "execution_domain_and_build_setup_ns=109",
            "metadata_binding_construction_ns=110",
            "reservation_validation_and_release_ns=111",
            "relation_device_allocation_calls=103",
            "workspace_provider_calls=104",
            "metadata_initial_htod_calls=105",
            "metadata_initial_htod_bytes=106",
            "schema_winners_initial_htod_calls=1",
            "schema_winners_initial_htod_bytes=12",
            "graph_instantiate_ns=107",
            "unattributed_ns=108",
        ] {
            assert!(line.split_ascii_whitespace().any(|field| field == expected));
        }
        assert_eq!(line.matches("sample=17").count(), 1);
        assert!(!line.contains("admission_source_validation_ns="));
        assert_eq!(resident_graph_prepare_diagnostic_line(None), None);
    }

    #[test]
    fn certificate_binds_strata_vector_length() {
        assert_mutation_rejected(|plan| {
            plan.strata.push(Stratum {
                id: 3,
                sccs: Vec::new(),
            });
        });
    }

    #[test]
    fn certificate_binds_empty_stratum_position() {
        assert_mutation_rejected(|plan| plan.strata.swap(1, 2));
    }

    #[test]
    fn certificate_binds_stratum_id() {
        assert_mutation_rejected(|plan| plan.strata[1].id = 99);
    }

    #[test]
    fn certificate_binds_ordered_stratum_scc_membership() {
        assert_mutation_rejected(|plan| plan.strata[2].sccs.swap(0, 1));
    }

    #[test]
    fn certificate_binds_scc_vector_length_including_empty_sccs() {
        assert_mutation_rejected(|plan| {
            plan.sccs.pop();
        });
    }

    #[test]
    fn certificate_binds_scc_id() {
        assert_mutation_rejected(|plan| plan.sccs[1].id = 99);
    }

    #[test]
    fn certificate_binds_ordered_scc_predicate_membership() {
        assert_mutation_rejected(|plan| plan.sccs[0].predicates.swap(0, 1));
    }

    #[test]
    fn certificate_binds_scc_recursive_flag() {
        assert_mutation_rejected(|plan| plan.sccs[1].is_recursive = true);
    }

    #[test]
    fn certificate_binds_rules_by_scc_vector_length() {
        assert_mutation_rejected(|plan| plan.rules_by_scc.push(Vec::new()));
    }

    #[test]
    fn certificate_binds_rule_occurrence_count() {
        assert_mutation_rejected(|plan| {
            plan.rules_by_scc[0].pop();
        });
    }

    #[test]
    fn certificate_binds_rule_order() {
        assert_mutation_rejected(|plan| plan.rules_by_scc[0].swap(0, 1));
    }

    #[test]
    fn certificate_binds_generated_query_provenance_omission() {
        let mut plan = representative_plan();
        plan.generated_query_rules = vec![GeneratedQueryRuleProvenance {
            query_index: 0,
            scc_index: 0,
            rule_index: 0,
        }];
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        plan.generated_query_rules.clear();
        assert!(!certificate.matches_plan(&plan).unwrap());
    }

    #[test]
    fn certificate_binds_generated_query_index() {
        assert_generated_query_provenance_mutation_rejected(|provenance| {
            provenance.query_index = 1;
        });
    }

    #[test]
    fn certificate_binds_generated_query_rule_position() {
        assert_generated_query_provenance_mutation_rejected(|provenance| {
            provenance.rule_index = 1;
        });
    }

    #[test]
    fn certificate_binds_rule_schema_type() {
        assert_mutation_rejected(|plan| {
            plan.rules_by_scc[0][0].meta.schema =
                schema(&[("left", ScalarType::U64), ("right", ScalarType::Symbol)]);
        });
    }

    #[test]
    fn certificate_binds_rule_schema_arity() {
        assert_mutation_rejected(|plan| {
            plan.rules_by_scc[0][0].meta.schema = schema(&[("left", ScalarType::U32)]);
        });
    }

    #[test]
    fn certificate_binds_primary_rir() {
        assert_mutation_rejected(|plan| plan.rules_by_scc[0][0].body = RirNode::Unit);
    }

    #[test]
    fn certificate_binds_captured_alternative_rir() {
        assert_mutation_rejected(|plan| {
            let RirNode::ChainJoin { fallback, .. } = &mut plan.rules_by_scc[0][1].body else {
                panic!("representative rule must carry a captured alternative");
            };
            **fallback = RirNode::Unit;
        });
    }

    #[test]
    fn certificate_binds_implicit_device_convergence_route() {
        let plan = representative_plan();
        let mut certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        let convergence = certificate
            .covered_route_descriptors
            .iter()
            .find(|descriptor| {
                descriptor.contains("scc=0;rule=0;head=alpha;")
                    && descriptor.ends_with(";implicit=device_convergence")
            })
            .cloned()
            .expect("recursive rule must certify implicit device convergence");
        assert!(certificate.covered_route_descriptors.remove(&convergence));
        certificate
            .covered_route_descriptors
            .insert(format!("{convergence};mutated"));
        certificate.plan_fingerprint = stable_descriptor_fingerprint(
            certificate
                .covered_route_descriptors
                .iter()
                .map(String::as_bytes),
        );
        assert!(!certificate.matches_plan(&plan).unwrap());
    }

    #[test]
    fn certificate_binds_rel_arities_vector_length() {
        assert_mutation_rejected(|plan| {
            plan.rel_arities.insert(RelId(7), 3);
        });
    }

    #[test]
    fn certificate_binds_relation_arity() {
        assert_mutation_rejected(|plan| {
            plan.rel_arities.insert(RelId(1), 3);
        });
    }

    #[test]
    fn certificate_rel_arities_are_order_independent() {
        let plan = representative_plan();
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog()).unwrap();
        let mut reordered = plan.clone();
        reordered.rel_arities.clear();
        reordered.rel_arities.insert(RelId(1), 2);
        reordered.rel_arities.insert(RelId(9), 0);
        assert!(certificate.matches_plan(&reordered).unwrap());
    }

    #[test]
    fn certificate_binds_estimated_memory_peak() {
        assert_mutation_rejected(|plan| plan.est_memory_peak += 1);
    }

    #[test]
    fn certificate_rejects_intermediate_arity_above_fixed_metadata_envelope() {
        let columns = (0..18)
            .map(|index| (format!("column_{index}"), ScalarType::U32))
            .collect::<Vec<_>>();
        let wide_schema = Schema::new(columns);
        let plan = ExecutionPlan {
            sccs: vec![Scc {
                id: 1,
                predicates: vec!["wide".to_owned()],
                is_recursive: false,
            }],
            strata: vec![Stratum {
                id: 0,
                sccs: vec![1],
            }],
            rules_by_scc: vec![vec![CompiledRule {
                head: "wide".to_owned(),
                body: RirNode::Scan { rel: RelId(1) },
                meta: RirMeta {
                    schema: wide_schema.clone(),
                    ..RirMeta::default()
                },
            }]],
            generated_query_rules: vec![],
            est_memory_peak: 0,
            rel_arities: HashMap::from([(RelId(1), 18)]),
        };
        let catalog = ResidentGraphSchemaCatalog::from_named_schemas([(
            "wide".to_owned(),
            RelId(1),
            wide_schema,
        )]);
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog).unwrap();
        assert!(!certificate.is_supported());
        assert!(certificate.declines().iter().any(|decline| matches!(
            decline,
            ResidentGraphDeclineReason::UnsupportedNode {
                node: "intermediate_arity",
                ..
            }
        )));
    }

    #[test]
    fn certificate_accepts_intermediate_arity_at_fixed_metadata_envelope() {
        let columns = (0..17)
            .map(|index| (format!("column_{index}"), ScalarType::U32))
            .collect::<Vec<_>>();
        let wide_schema = Schema::new(columns);
        let plan = ExecutionPlan {
            sccs: vec![Scc {
                id: 1,
                predicates: vec!["wide".to_owned()],
                is_recursive: false,
            }],
            strata: vec![Stratum {
                id: 0,
                sccs: vec![1],
            }],
            rules_by_scc: vec![vec![CompiledRule {
                head: "wide".to_owned(),
                body: RirNode::Scan { rel: RelId(1) },
                meta: RirMeta {
                    schema: wide_schema.clone(),
                    ..RirMeta::default()
                },
            }]],
            generated_query_rules: vec![],
            est_memory_peak: 0,
            rel_arities: HashMap::from([(RelId(1), 17)]),
        };
        let catalog = ResidentGraphSchemaCatalog::from_named_schemas([(
            "wide".to_owned(),
            RelId(1),
            wide_schema,
        )]);
        let certificate = ResidentGraphRouteCertificate::inspect(&plan, &catalog).unwrap();
        assert!(certificate.is_supported(), "{:#?}", certificate.declines());
    }
}
