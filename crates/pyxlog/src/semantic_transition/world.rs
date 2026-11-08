//! Immutable common-world material and its original context/source owners.
//!
//! These factories perform cold declaration parsing only. They neither execute a
//! model nor select a mutable semantic root; publication remains owned by the
//! existing native session.

use super::*;
use pyo3::types::PyBool;
use xlog_ir::semantic_value as value;
use xlog_ir::semantic_world as world;

const WORLD_TAG: &str = "xlog.semantic-world.v1";
const CONTEXT_TAG: &str = "xlog.semantic-context.v1";

pub(super) struct WorldDeclarationData {
    pub(super) native: Arc<world::SemanticWorldDeclaration>,
    pub(super) material: Arc<[u8]>,
    pub(super) identity: [u8; 32],
    pub(super) local_identities: Vec<[u8; 32]>,
    original_sources: Vec<Py<PyTuple>>,
}

pub(super) struct WorldOwnerData {
    pub(super) declaration: Arc<WorldDeclarationData>,
}

pub(super) struct ContextSourceData {
    pub(super) owner: Arc<WorldOwnerData>,
    pub(super) index: u32,
    pub(super) identity: [u8; 32],
    material: Vec<u8>,
}

impl ContextSourceData {
    pub(super) fn declaration(&self) -> &world::SemanticContextDeclaration {
        &self.owner.declaration.native.contexts[self.index as usize]
    }
}

fn text_value(value: &ColdValue) -> PyResult<&str> {
    match value {
        ColdValue::Text(text) => Ok(text),
        _ => Err(invalid("semantic declaration tags require exact text")),
    }
}

fn unsigned(value: &ColdValue) -> PyResult<u64> {
    match value {
        ColdValue::Integer(text) => {
            let number = text
                .parse::<u64>()
                .map_err(|_| invalid("semantic declaration integer exceeds u64"))?;
            if number.to_string() != *text {
                return Err(invalid("semantic declaration integer is not canonical"));
            }
            Ok(number)
        }
        _ => Err(invalid(
            "semantic declaration indices require exact unsigned integers",
        )),
    }
}

fn index(value: &ColdValue) -> PyResult<u32> {
    u32::try_from(unsigned(value)?).map_err(|_| invalid("semantic declaration index exceeds u32"))
}

fn polarity(value: &ColdValue) -> PyResult<world::SemanticSupportPolarity> {
    match text_value(value)? {
        "pro" => Ok(world::SemanticSupportPolarity::Pro),
        "contra" => Ok(world::SemanticSupportPolarity::Contra),
        _ => Err(invalid("semantic support polarity must be pro or contra")),
    }
}

fn query_kind(value: &ColdValue) -> PyResult<world::SemanticValueQueryKind> {
    match text_value(value)? {
        "assignment" => Ok(world::SemanticValueQueryKind::Assignment),
        "measure" => Ok(world::SemanticValueQueryKind::Measure),
        _ => Err(invalid("semantic query kind must be assignment or measure")),
    }
}

fn exact_immutable(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<()> {
    if depth > 32 {
        return Err(invalid(
            "semantic declaration exceeds the cold transport depth",
        ));
    }
    if value.is_exact_instance_of::<PyTuple>() {
        for child in value.cast::<PyTuple>()?.iter() {
            exact_immutable(&child, depth + 1)?;
        }
    } else if !(value.is_none()
        || value.is_exact_instance_of::<PyString>()
        || value.is_exact_instance_of::<PyInt>()
        || value.is_exact_instance_of::<PyBool>())
    {
        return Err(invalid(
            "semantic declaration requires exact immutable builtin material",
        ));
    }
    Ok(())
}

fn cold_immutable(value: &Bound<'_, PyAny>) -> PyResult<ColdValue> {
    exact_immutable(value, 0)?;
    // The complete roster is not a per-task model-metadata envelope. Its actual
    // counted material is charged when the common owner binds the session budget.
    let mut budget = usize::MAX;
    ColdValue::read(value, &mut budget, 0)
}

fn source_material(source: &Bound<'_, PyAny>) -> PyResult<ColdValue> {
    super::cold_task::ColdTaskSource::from_tuple(source)?;
    let source = source.cast::<PyTuple>()?;
    let mut fields = Vec::with_capacity(6);
    for ordinal in 0..5 {
        fields.push(cold_immutable(&source.get_item(ordinal)?)?);
    }
    let image = super::program_image::declaration(&source.get_item(5)?)?;
    fields.push(image.material_value().clone());
    Ok(ColdValue::Sequence(fields))
}

fn source_python(py: Python<'_>, source: &ColdValue) -> PyResult<Py<PyTuple>> {
    let fields = source.fields(6)?;
    let mut original = fields[..5]
        .iter()
        .map(|field| field.python_value(py))
        .collect::<PyResult<Vec<_>>>()?;
    let image = super::program_image::ProgramImageDeclaration::from_value(fields[5].clone())?;
    original.push(
        Py::new(
            py,
            super::program_image::PySemanticProgramImage {
                inner: Arc::new(image),
            },
        )?
        .into_any(),
    );
    let original = PyTuple::new(py, original)?;
    super::cold_task::ColdTaskSource::from_tuple(original.as_any())?;
    Ok(original.unbind())
}

fn parse_context(value: &ColdValue) -> PyResult<world::SemanticContextDeclaration> {
    let fields = value.fields(5)?;
    let view = fields[1].fields(4)?;
    let mut assignment = Vec::new();
    for item in view[3].sequence()? {
        let pair = item.fields(2)?;
        let ColdValue::Bool(value) = pair[1] else {
            return Err(invalid("semantic assignment values require exact booleans"));
        };
        assignment.push((index(&pair[0])?, value));
    }
    let signed = fields[2].sequence()?;
    let tag = signed
        .first()
        .map(text_value)
        .transpose()?
        .ok_or_else(|| invalid("semantic source lowering has no construction law"))?;
    let program = match (tag, signed.len()) {
        ("native_program", 1) => world::SemanticSignedProgram::NativeProgram,
        ("signed_program", 3) => {
            let mut facts = Vec::new();
            for row in signed[1].sequence()? {
                let row = row.fields(3)?;
                facts.push(world::SemanticFactEventBinding {
                    fact: index(&row[0])?,
                    pro_event: index(&row[1])?,
                    contra_event: index(&row[2])?,
                });
            }
            let mut rules = Vec::new();
            for row in signed[2].sequence()? {
                let row = row.fields(4)?;
                rules.push(world::SemanticRuleEventBinding {
                    rule: index(&row[0])?,
                    head: polarity(&row[1])?,
                    body: row[2]
                        .sequence()?
                        .iter()
                        .map(polarity)
                        .collect::<PyResult<Vec<_>>>()?,
                    gate_event: index(&row[3])?,
                });
            }
            world::SemanticSignedProgram::SignedProgram { facts, rules }
        }
        _ => {
            return Err(invalid(
                "unknown semantic source construction law or field count",
            ))
        }
    };
    let cells = fields[3].fields(2)?;
    let rule_cells = match text_value(&cells[0])? {
        "uniform" => world::SemanticInitialRuleCells::Uniform(index(&cells[1])?),
        "codes" => world::SemanticInitialRuleCells::Codes(
            cells[1]
                .sequence()?
                .iter()
                .map(index)
                .collect::<PyResult<Vec<_>>>()?,
        ),
        _ => {
            return Err(invalid(
                "initial semantic rule cells require uniform or codes",
            ))
        }
    };
    let mut queries = Vec::new();
    for row in fields[4].sequence()? {
        let row = row.fields(3)?;
        let proof = match text_value(&row[2])? {
            "structural" => world::SemanticValueProofKind::Structural,
            "certified" => world::SemanticValueProofKind::Certified,
            _ => {
                return Err(invalid(
                    "semantic query proof must be structural or certified",
                ))
            }
        };
        queries.push(world::SemanticQueryView {
            query: index(&row[0])?,
            kind: query_kind(&row[1])?,
            proof,
        });
    }
    Ok(world::SemanticContextDeclaration {
        view: world::SemanticContextView {
            world_ordinal: index(&view[0])?,
            kind: query_kind(&view[1])?,
            condition_event: index(&view[2])?,
            assignment,
        },
        program,
        rule_cells,
        queries,
    })
}

fn parse_events(value: &ColdValue) -> PyResult<Vec<world::SemanticEventNode>> {
    let mut events = Vec::new();
    for row in value.sequence()? {
        let fields = row.sequence()?;
        let tag = fields
            .first()
            .map(text_value)
            .transpose()?
            .ok_or_else(|| invalid("semantic event has no kind"))?;
        events.push(match (tag, fields.len()) {
            ("false", 1) => world::SemanticEventNode::False,
            ("true", 1) => world::SemanticEventNode::True,
            ("literal", 2) => {
                let ColdValue::Integer(text) = &fields[1] else {
                    return Err(invalid("semantic literal requires an exact signed integer"));
                };
                let literal = text
                    .parse::<i32>()
                    .map_err(|_| invalid("semantic literal exceeds i32"))?;
                if literal.to_string() != *text {
                    return Err(invalid("semantic literal is not canonical"));
                }
                world::SemanticEventNode::Literal(literal)
            }
            ("and", 2) => world::SemanticEventNode::And(
                fields[1]
                    .sequence()?
                    .iter()
                    .map(index)
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            ("or", 2) => world::SemanticEventNode::Or(
                fields[1]
                    .sequence()?
                    .iter()
                    .map(index)
                    .collect::<PyResult<Vec<_>>>()?,
            ),
            ("not", 2) => world::SemanticEventNode::Not(index(&fields[1])?),
            _ => return Err(invalid("unknown semantic event kind or field count")),
        });
    }
    Ok(events)
}

fn parse_factors(value: &ColdValue) -> PyResult<Vec<world::SemanticMeasureFactor>> {
    let mut factors = Vec::new();
    for row in value.sequence()? {
        let row = row.sequence()?;
        let tag = row
            .first()
            .map(text_value)
            .transpose()?
            .ok_or_else(|| invalid("semantic measure factor has no kind"))?;
        factors.push(match (tag, row.len()) {
            ("unit", 1) => world::SemanticMeasureFactor::Unit,
            ("bernoulli", 4) => world::SemanticMeasureFactor::Bernoulli {
                variable: index(&row[1])?,
                numerator: unsigned(&row[2])?,
                denominator: unsigned(&row[3])?,
            },
            ("event_weight", 4) => world::SemanticMeasureFactor::EventWeight {
                event: index(&row[1])?,
                numerator: unsigned(&row[2])?,
                denominator: unsigned(&row[3])?,
            },
            _ => return Err(invalid("unknown common measure factor or field count")),
        });
    }
    Ok(factors)
}

fn hash_material(domain: &[u8], material: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update((material.len() as u64).to_le_bytes());
    digest.update(material);
    digest.finalize().into()
}

fn validate_canonical_primitives(value: &ColdValue) -> PyResult<()> {
    match value {
        ColdValue::None | ColdValue::Bool(_) | ColdValue::Text(_) => Ok(()),
        ColdValue::Integer(text) => {
            let integer = text.strip_prefix('-').unwrap_or(text);
            if integer.is_empty()
                || !integer.bytes().all(|b| b.is_ascii_digit())
                || (integer.len() > 1 && integer.starts_with('0'))
                || text == "-0"
            {
                return Err(invalid("semantic material contains a noncanonical integer"));
            }
            Ok(())
        }
        ColdValue::Sequence(items) => {
            for item in items {
                validate_canonical_primitives(item)?;
            }
            Ok(())
        }
        _ => Err(invalid(
            "semantic world material requires its original immutable primitives",
        )),
    }
}

impl WorldDeclarationData {
    fn from_value(
        py: Python<'_>,
        material: ColdValue,
        original_sources: Option<Vec<Py<PyTuple>>>,
    ) -> PyResult<Self> {
        validate_canonical_primitives(&material)?;
        let fields = material.fields(6)?;
        if text_value(&fields[0])? != WORLD_TAG {
            return Err(invalid(
                "semantic world material has an unknown declaration kind",
            ));
        }
        let local_records = fields[1].sequence()?;
        if local_records.is_empty() || local_records.len() > u32::MAX as usize {
            return Err(invalid(
                "semantic world requires the full finite ordered context roster",
            ));
        }
        let law = fields[4].fields(value::SEMANTIC_VALUE_NUMERIC_LAW.len())?;
        for (actual, expected) in law.iter().zip(value::SEMANTIC_VALUE_NUMERIC_LAW) {
            if text_value(actual)? != expected {
                return Err(invalid("semantic world changed the declared numerical law"));
            }
        }
        let caps = fields[5].fields(8)?;
        let native = world::SemanticWorldDeclaration {
            contexts: local_records
                .iter()
                .map(parse_context)
                .collect::<PyResult<Vec<_>>>()?,
            event_nodes: parse_events(&fields[2])?,
            measure_factors: parse_factors(&fields[3])?,
            capacities: world::SemanticMeasureCapacities {
                nodes: index(&caps[0])?,
                edges: index(&caps[1])?,
                unique_slots: index(&caps[2])?,
                apply_entries: index(&caps[3])?,
                dependency_edges: index(&caps[4])?,
                frontier_entries: index(&caps[5])?,
                particles: index(&caps[6])?,
                variables: index(&caps[7])?,
            },
        };
        native
            .validate()
            .map_err(|error| invalid(&error.to_string()))?;
        let sources = match original_sources {
            Some(sources) if sources.len() == local_records.len() => sources,
            Some(_) => return Err(invalid("semantic world lost its original source owners")),
            None => local_records
                .iter()
                .map(|record| source_python(py, &record.fields(5)?[0]))
                .collect::<PyResult<Vec<_>>>()?,
        };
        for (row, original) in local_records.iter().zip(&sources) {
            if source_material(original.bind(py).as_any())? != row.fields(5)?[0] {
                return Err(invalid(
                    "semantic local material differs from its original source owner",
                ));
            }
        }
        let local_identities = local_records
            .iter()
            .map(|record| {
                hash_material(
                    b"xlog.semantic-local.v1\0",
                    &checkpoint_cold_value_bytes(record),
                )
            })
            .collect::<Vec<_>>();
        let mut digest = Sha256::new();
        digest.update(b"xlog.semantic-world.v1\0");
        digest.update((local_identities.len() as u64).to_le_bytes());
        for identity in &local_identities {
            digest.update(identity);
        }
        for field in &fields[2..] {
            let bytes = checkpoint_cold_value_bytes(field);
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
        }
        let identity = digest.finalize().into();
        Ok(Self {
            native: Arc::new(native),
            material: checkpoint_cold_value_bytes(&material).into(),
            identity,
            local_identities,
            original_sources: sources,
        })
    }
}

/// Complete authored run-level semantic material, independent of allocation and
/// publication. Full identities can include protected source; they are native
/// binding metadata, never an automatic learned or public-text projection.
#[pyclass(name = "SemanticWorldDeclaration", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticWorldDeclaration {
    pub(super) inner: Arc<WorldDeclarationData>,
}

#[pymethods]
impl PySemanticWorldDeclaration {
    #[new]
    #[pyo3(signature = (*, contexts, event_nodes, measure_factors, numeric_law, capacities))]
    fn new(
        py: Python<'_>,
        contexts: &Bound<'_, PyAny>,
        event_nodes: &Bound<'_, PyAny>,
        measure_factors: &Bound<'_, PyAny>,
        numeric_law: &Bound<'_, PyAny>,
        capacities: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        if !contexts.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "semantic contexts require an exact ordered immutable tuple",
            ));
        }
        let mut records = Vec::new();
        let mut original_sources = Vec::new();
        for row in contexts.cast::<PyTuple>()?.iter() {
            if !row.is_exact_instance_of::<PyTuple>() || row.cast::<PyTuple>()?.len() != 5 {
                return Err(invalid("semantic local record requires source, view, signed program, rule cells and queries"));
            }
            let row = row.cast::<PyTuple>()?;
            let original = row.get_item(0)?;
            let mut fields = vec![source_material(&original)?];
            original_sources.push(original.cast::<PyTuple>()?.clone().unbind());
            for field in row.iter().skip(1) {
                fields.push(cold_immutable(&field)?);
            }
            records.push(ColdValue::Sequence(fields));
        }
        let material = ColdValue::Sequence(vec![
            ColdValue::Text(WORLD_TAG.into()),
            ColdValue::Sequence(records),
            cold_immutable(event_nodes)?,
            cold_immutable(measure_factors)?,
            cold_immutable(numeric_law)?,
            cold_immutable(capacities)?,
        ]);
        Ok(Self {
            inner: Arc::new(WorldDeclarationData::from_value(
                py,
                material,
                Some(original_sources),
            )?),
        })
    }

    #[staticmethod]
    fn from_material(py: Python<'_>, material: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !material.is_exact_instance_of::<PyBytes>() {
            return Err(invalid(
                "semantic world restoration requires exact original material bytes",
            ));
        }
        let bytes = material.cast::<PyBytes>()?.as_bytes();
        let value = checkpoint_cold_value(bytes, bytes.len())?;
        let inner = WorldDeclarationData::from_value(py, value, None)?;
        if inner.material.as_ref() != bytes {
            return Err(invalid(
                "semantic world restoration changed its canonical material",
            ));
        }
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[getter]
    fn material(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.inner.material).unbind()
    }

    #[getter]
    fn identity(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.inner.identity).unbind()
    }

    #[getter]
    fn context_local_identities(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.inner
                .local_identities
                .iter()
                .map(|identity| PyBytes::new(py, identity)),
        )?
        .unbind())
    }

    #[getter]
    fn context_count(&self) -> u32 {
        self.inner.native.contexts.len() as u32
    }

    /// Structural schema only: no world identity or protected source enters it.
    fn feedback_schema(&self, py: Python<'_>, statement_bytes: usize) -> PyResult<Py<PyTuple>> {
        value_schema_object(py, statement_bytes)
    }
}

/// Run-level namespace and storage-lifetime owner. It has no separate CURRENT
/// selector and no authority to publish a semantic root.
#[pyclass(name = "SemanticWorldOwner", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticWorldOwner {
    pub(super) inner: Arc<WorldOwnerData>,
    declaration: Py<PySemanticWorldDeclaration>,
}

#[pymethods]
impl PySemanticWorldOwner {
    #[new]
    #[pyo3(signature = (*, declaration))]
    fn new(declaration: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !declaration.is_exact_instance_of::<PySemanticWorldDeclaration>() {
            return Err(invalid(
                "semantic world owner requires its native immutable declaration",
            ));
        }
        let inner = Arc::clone(
            &declaration
                .extract::<PyRef<'_, PySemanticWorldDeclaration>>()?
                .inner,
        );
        Ok(Self {
            inner: Arc::new(WorldOwnerData { declaration: inner }),
            declaration: declaration.extract()?,
        })
    }

    fn context_source(
        &self,
        py: Python<'_>,
        context_index: u32,
    ) -> PyResult<PySemanticContextSource> {
        let declaration = &self.inner.declaration;
        let local = declaration
            .local_identities
            .get(context_index as usize)
            .ok_or_else(|| invalid("semantic context index is outside the original declaration"))?;
        let mut digest = Sha256::new();
        digest.update(b"xlog.semantic-bound-context.v1\0");
        digest.update(declaration.identity);
        digest.update(local);
        digest.update(context_index.to_le_bytes());
        let identity: [u8; 32] = digest.finalize().into();
        let material = checkpoint_cold_value_bytes(&ColdValue::Sequence(vec![
            ColdValue::Text(CONTEXT_TAG.into()),
            ColdValue::Bytes(Arc::from(declaration.identity)),
            ColdValue::Bytes(Arc::from(*local)),
            ColdValue::Bytes(Arc::from(identity)),
            ColdValue::Integer(context_index.to_string()),
        ]));
        Ok(PySemanticContextSource {
            inner: Arc::new(ContextSourceData {
                owner: Arc::clone(&self.inner),
                index: context_index,
                identity,
                material,
            }),
            declaration: self.declaration.clone_ref(py),
        })
    }
}

/// Original context binding. Only its common owner can construct it.
#[pyclass(name = "SemanticContextSource", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticContextSource {
    pub(super) inner: Arc<ContextSourceData>,
    declaration: Py<PySemanticWorldDeclaration>,
}

#[pymethods]
impl PySemanticContextSource {
    #[getter]
    fn original_source(&self, py: Python<'_>) -> Py<PyTuple> {
        self.inner.owner.declaration.original_sources[self.inner.index as usize].clone_ref(py)
    }

    #[getter]
    fn world_declaration(&self, py: Python<'_>) -> Py<PySemanticWorldDeclaration> {
        self.declaration.clone_ref(py)
    }

    #[getter]
    fn material(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.inner.material).unbind()
    }

    #[getter]
    fn world_identity(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.inner.owner.declaration.identity).unbind()
    }

    #[getter]
    fn context_identity(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.inner.identity).unbind()
    }

    #[getter]
    fn context_index(&self) -> u32 {
        self.inner.index
    }
}

pub(super) fn context_source(value: &Bound<'_, PyAny>) -> PyResult<Arc<ContextSourceData>> {
    if !value.is_exact_instance_of::<PySemanticContextSource>() {
        return Err(invalid(
            "semantic_context requires its original native context source",
        ));
    }
    Ok(Arc::clone(
        &value.extract::<PyRef<'_, PySemanticContextSource>>()?.inner,
    ))
}

fn text(value: &str) -> ColdValue {
    ColdValue::Text(value.into())
}
fn integer(value: usize) -> ColdValue {
    ColdValue::Integer(value.to_string())
}

pub(super) fn value_schema_object(py: Python<'_>, statement_bytes: usize) -> PyResult<Py<PyTuple>> {
    let schema = value::SemanticValueFeedbackSchema::new(statement_bytes)
        .map_err(|error| invalid(&error.to_string()))?;
    let features = ColdValue::Sequence(
        schema
            .fields
            .iter()
            .map(|field| {
                ColdValue::Sequence(vec![
                    text(field.name),
                    integer(field.offset),
                    integer(field.length),
                    text(field.dtype),
                    text(field.units),
                    text(field.defined_when),
                ])
            })
            .collect(),
    );
    let receipts = ColdValue::Sequence(
        value::SEMANTIC_VALUE_RECEIPT_FIELDS
            .iter()
            .map(|field| text(field))
            .collect(),
    );
    let supports = ColdValue::Sequence(
        value::SEMANTIC_VALUE_SUPPORT_FIELDS
            .iter()
            .map(|field| text(field))
            .collect(),
    );
    let estimators = ColdValue::Sequence(
        value::SEMANTIC_VALUE_ESTIMATORS
            .iter()
            .map(|estimator| {
                ColdValue::Sequence(vec![
                    text(estimator.name),
                    ColdValue::Bool(estimator.sample_count_defined),
                    ColdValue::Bool(estimator.confidence_defined),
                    text(estimator.error_bound_kind),
                ])
            })
            .collect(),
    );
    let law = checkpoint_cold_value_bytes(&ColdValue::Sequence(
        value::SEMANTIC_VALUE_NUMERIC_LAW
            .iter()
            .map(|field| text(field))
            .collect(),
    ));
    let law_identity = hash_material(b"xlog.semantic-value.numeric-law.v1\0", &law);
    let structural = ColdValue::Sequence(vec![
        text("xlog.feedback.semantic-value.v1"),
        integer(statement_bytes),
        integer(schema.feature_width),
        features.clone(),
        receipts.clone(),
        supports.clone(),
        estimators.clone(),
        ColdValue::Bytes(Arc::from(law.clone())),
        ColdValue::Bytes(Arc::from(law_identity)),
    ]);
    let identity = hash_material(
        b"xlog.feedback.semantic-value.v1\0",
        &checkpoint_cold_value_bytes(&structural),
    );
    let mut adapter = Sha256::new();
    adapter.update(b"xlog.python.semantic-value-adapter.v1\0");
    adapter.update(identity);
    adapter.update(include_bytes!("world.rs"));
    let adapter_identity: [u8; 32] = adapter.finalize().into();
    let result = ColdValue::Sequence(vec![
        integer(2),
        ColdValue::Bytes(Arc::from(identity)),
        ColdValue::Bytes(Arc::from(adapter_identity)),
        integer(statement_bytes),
        integer(schema.feature_width),
        features,
        receipts,
        supports,
        estimators,
        ColdValue::Bytes(Arc::from(law)),
        ColdValue::Bytes(Arc::from(law_identity)),
    ])
    .python_value(py)?;
    Ok(result.bind(py).cast::<PyTuple>()?.clone().unbind())
}
