use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySequence, PyString};
use sha2::{Digest, Sha256};

use xlog_core::{ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_gpu::logic::{LogicArgumentSchema, PreparedRelationDeltaBatch, RelationDeltaDirection};

use super::types;

create_exception!(
    pyxlog,
    RelationMetadataError,
    PyValueError,
    "A relation role or whole-fact provenance value violates the compiled relation contract."
);

const FACT_IDENTITY_DOMAIN: &[u8] = b"xlog.fact.identity.v1\0";
const SCHEMA_IDENTITY_DOMAIN: &[u8] = b"xlog.relation.schema.v1\0";
const PROGRAM_EVIDENCE_DOMAIN: &[u8] = b"xlog.program.evidence.v1\0";

pub(crate) fn require_positive_metadata_arity(relation: &str, schema: &Schema) -> PyResult<()> {
    if schema.arity() == 0 {
        return Err(metadata_error(format!(
            "Relation '{relation}' provenance requires positive arity; nullary relation metadata is unsupported"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RelationRole {
    name: String,
    sort: Option<String>,
    scalar_type: ScalarType,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ProvenanceSpan {
    start: u64,
    end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ProvenanceRecord {
    source: Option<String>,
    document: Option<String>,
    span: Option<ProvenanceSpan>,
    content_hash: Option<String>,
    kind: Option<String>,
    polarity: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TypedCell {
    type_code: u8,
    bytes: Vec<u8>,
}

impl TypedCell {
    fn scalar_type(&self) -> ScalarType {
        ScalarType::from_code(self.type_code)
            .expect("typed relation cells are constructed from compiled scalar types")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FactKey {
    schema_sha256: [u8; 32],
    cells: Vec<TypedCell>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RelationMetadata {
    roles: Vec<RelationRole>,
    facts: BTreeMap<FactKey, BTreeSet<ProvenanceRecord>>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RelationMetadataStore {
    relations: BTreeMap<String, Arc<RelationMetadata>>,
}

/// Parsed, schema-bound evidence for one original insert occurrence.
///
/// Values of this type own every Python-derived value and have already passed
/// full-row membership validation against that occurrence's insert buffer.
#[derive(Debug)]
pub(crate) struct PreparedInsertEvidence {
    relation: String,
    facts: BTreeMap<FactKey, BTreeSet<ProvenanceRecord>>,
}

#[derive(Debug)]
pub(crate) struct PreparedRelationMetadataUpdate {
    relation: String,
    insert_evidence: Option<PreparedInsertEvidence>,
}

#[must_use = "prepared relation metadata has no effect until it is committed"]
pub(crate) struct PreparedRelationMetadataTransition {
    prospective_store: Option<RelationMetadataStore>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RelationSnapshot {
    relation: String,
    row_count: usize,
    schema_sha256: [u8; 32],
    schema_arity: usize,
    metadata: Option<Arc<RelationMetadata>>,
}

#[pyclass(frozen, module = "pyxlog._native")]
pub struct RelationEvidence {
    snapshot: RelationSnapshot,
}

#[pymethods]
impl RelationEvidence {
    pub fn provenance(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.snapshot.pack(py)
    }
}

impl RelationEvidence {
    pub(crate) fn new(snapshot: RelationSnapshot) -> Self {
        Self { snapshot }
    }
}

impl RelationMetadataStore {
    pub(crate) fn prepare_replacement(
        &self,
        relation: &str,
        arguments: &[LogicArgumentSchema],
        schema: &Schema,
        provider: &Arc<CudaKernelProvider>,
        relation_buffer: &CudaBuffer,
        roles: &Bound<'_, PyAny>,
        facts: &Bound<'_, PyAny>,
        row_count: usize,
    ) -> PyResult<(Self, RelationSnapshot)> {
        require_positive_metadata_arity(relation, schema)?;
        if arguments.len() != schema.arity() {
            return Err(PyRuntimeError::new_err(format!(
                "Relation '{relation}' compiled argument metadata has arity {} but its schema has arity {}",
                arguments.len(),
                schema.arity()
            )));
        }

        let parsed_roles = parse_roles(relation, arguments, roles)?;
        if arguments.iter().any(|argument| !argument.source_named()) {
            if let Some(existing) = self.relations.get(relation) {
                if existing.roles != parsed_roles {
                    return Err(metadata_error(format!(
                        "Relation '{relation}' roles do not match the registered role contract"
                    )));
                }
            }
        }

        let schema_sha256 = relation_schema_fingerprint(relation, arguments)?;
        let parsed_facts = parse_facts(relation, schema, schema_sha256, facts)?;
        validate_fact_membership(relation, schema, provider, relation_buffer, &parsed_facts)?;

        let metadata = Arc::new(RelationMetadata {
            roles: parsed_roles,
            facts: parsed_facts,
        });
        let mut prospective = self.clone();
        prospective
            .relations
            .insert(relation.to_string(), Arc::clone(&metadata));
        let snapshot = RelationSnapshot {
            relation: relation.to_string(),
            row_count,
            schema_sha256,
            schema_arity: schema.arity(),
            metadata: Some(metadata),
        };
        Ok((prospective, snapshot))
    }

    pub(crate) fn prepare_insert_evidence(
        &self,
        relation: &str,
        arguments: &[LogicArgumentSchema],
        schema: &Schema,
        provider: &Arc<CudaKernelProvider>,
        insert_buffer: &CudaBuffer,
        facts: &Bound<'_, PyAny>,
    ) -> PyResult<PreparedInsertEvidence> {
        require_positive_metadata_arity(relation, schema)?;
        if !self.relations.contains_key(relation) {
            return Err(metadata_error(format!(
                "Relation '{relation}' insert_facts requires an existing registered role contract"
            )));
        }
        if arguments.len() != schema.arity() {
            return Err(PyRuntimeError::new_err(format!(
                "Relation '{relation}' compiled argument metadata has arity {} but its schema has arity {}",
                arguments.len(),
                schema.arity()
            )));
        }

        let schema_sha256 = relation_schema_fingerprint(relation, arguments)?;
        let parsed_facts = parse_facts(relation, schema, schema_sha256, facts)?;
        validate_fact_membership(relation, schema, provider, insert_buffer, &parsed_facts)?;
        Ok(PreparedInsertEvidence {
            relation: relation.to_string(),
            facts: parsed_facts,
        })
    }

    pub(crate) fn prepare_delta_transition(
        &self,
        relation: &str,
        schema: &Schema,
        provider: &Arc<CudaKernelProvider>,
        delete_buffer: Option<&CudaBuffer>,
        insert_evidence: Option<PreparedInsertEvidence>,
    ) -> PyResult<PreparedRelationMetadataTransition> {
        if insert_evidence
            .as_ref()
            .is_some_and(|evidence| evidence.relation != relation)
        {
            return Err(PyRuntimeError::new_err(format!(
                "Prepared insert evidence for another relation was supplied while updating '{relation}'"
            )));
        }

        let Some(existing) = self.relations.get(relation) else {
            if insert_evidence.is_some() {
                return Err(PyRuntimeError::new_err(format!(
                    "Relation '{relation}' lost its registered role contract during delta preparation"
                )));
            }
            return Ok(PreparedRelationMetadataTransition::unchanged());
        };

        let mut facts = existing.facts.clone();
        if let Some(delete_buffer) = delete_buffer {
            remove_matching_facts(relation, schema, provider, delete_buffer, &mut facts)?;
        }
        if let Some(insert_evidence) = insert_evidence {
            union_fact_records(&mut facts, insert_evidence.facts);
        }

        if facts == existing.facts {
            return Ok(PreparedRelationMetadataTransition::unchanged());
        }
        let mut prospective = self.clone();
        let metadata = prospective.relations.get_mut(relation).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Relation '{relation}' metadata disappeared during delta preparation"
            ))
        })?;
        Arc::make_mut(metadata).facts = facts;
        Ok(PreparedRelationMetadataTransition::replace(prospective))
    }

    pub(crate) fn prepare_batch_transition(
        &self,
        provider: &Arc<CudaKernelProvider>,
        schemas: &BTreeMap<String, Schema>,
        updates: Vec<PreparedRelationMetadataUpdate>,
        prepared_batch: &PreparedRelationDeltaBatch,
    ) -> PyResult<PreparedRelationMetadataTransition> {
        let mut pending_inserts =
            BTreeMap::<String, BTreeMap<FactKey, BTreeSet<ProvenanceRecord>>>::new();

        for (update_index, update) in updates.into_iter().enumerate() {
            let schema = schemas.get(&update.relation).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Relation '{}' schema disappeared during batch metadata preparation",
                    update.relation
                ))
            })?;
            let cancellations = prepared_batch
                .cancellations()
                .get(&update.relation)
                .into_iter()
                .flatten()
                .filter(|cancellation| cancellation.update_index() == update_index)
                .collect::<Vec<_>>();

            if let Some(insert_evidence) = update.insert_evidence {
                if insert_evidence.relation != update.relation {
                    return Err(PyRuntimeError::new_err(format!(
                        "Prepared insert evidence relation '{}' does not match batch update relation '{}'",
                        insert_evidence.relation, update.relation
                    )));
                }
                let mut incoming_facts = insert_evidence.facts;
                for cancellation in cancellations.iter().filter(|cancellation| {
                    cancellation.incoming_direction() == RelationDeltaDirection::Insert
                }) {
                    remove_matching_facts(
                        &update.relation,
                        schema,
                        provider,
                        cancellation.tuples(),
                        &mut incoming_facts,
                    )?;
                }
                if !incoming_facts.is_empty() {
                    union_fact_records(
                        pending_inserts.entry(update.relation.clone()).or_default(),
                        incoming_facts,
                    );
                }
            }

            for cancellation in cancellations.iter().filter(|cancellation| {
                cancellation.incoming_direction() == RelationDeltaDirection::Delete
            }) {
                let Some(pending) = pending_inserts.get_mut(&update.relation) else {
                    continue;
                };
                remove_matching_facts(
                    &update.relation,
                    schema,
                    provider,
                    cancellation.tuples(),
                    pending,
                )?;
            }
        }

        let mut prospective_store: Option<RelationMetadataStore> = None;
        let mut relation_names = prepared_batch
            .net_deltas()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        relation_names.sort_unstable();
        for relation in relation_names {
            let Some(existing) = self.relations.get(relation) else {
                if pending_inserts
                    .get(relation)
                    .is_some_and(|facts| !facts.is_empty())
                {
                    return Err(PyRuntimeError::new_err(format!(
                        "Relation '{relation}' lost its registered role contract during batch preparation"
                    )));
                }
                continue;
            };
            let schema = schemas.get(relation).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Relation '{relation}' schema disappeared during batch metadata preparation"
                ))
            })?;
            let delta = prepared_batch
                .net_deltas()
                .get(relation)
                .expect("relation name came from the prepared net deltas");
            let mut facts = existing.facts.clone();
            if let Some(delete) = delta.delete.as_ref() {
                remove_matching_facts(relation, schema, provider, delete, &mut facts)?;
            }
            if let Some(inserts) = pending_inserts.remove(relation) {
                if delta.insert.is_none() && !inserts.is_empty() {
                    return Err(PyRuntimeError::new_err(format!(
                        "Relation '{relation}' has surviving provenance without a net insert"
                    )));
                }
                union_fact_records(&mut facts, inserts);
            }
            if facts == existing.facts {
                continue;
            }

            let prospective = prospective_store.get_or_insert_with(|| self.clone());
            let metadata = prospective.relations.get_mut(relation).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Relation '{relation}' metadata disappeared while staging the batch transition"
                ))
            })?;
            Arc::make_mut(metadata).facts = facts;
        }

        if pending_inserts.values().any(|facts| !facts.is_empty()) {
            return Err(PyRuntimeError::new_err(
                "Batch cancellation trace left provenance without a net relation insert",
            ));
        }

        Ok(PreparedRelationMetadataTransition { prospective_store })
    }

    pub(crate) fn clear_relation(&mut self, relation: &str) {
        self.relations.remove(relation);
    }

    pub(crate) fn clear(&mut self) {
        self.relations.clear();
    }

    pub(crate) fn snapshot(
        &self,
        relation: &str,
        row_count: usize,
        schema_sha256: [u8; 32],
        schema_arity: usize,
    ) -> RelationSnapshot {
        RelationSnapshot {
            relation: relation.to_string(),
            row_count,
            schema_sha256,
            schema_arity,
            metadata: self.relations.get(relation).cloned(),
        }
    }
}

impl PreparedInsertEvidence {
    pub(crate) fn has_fact_keys(&self) -> bool {
        !self.facts.is_empty()
    }
}

impl PreparedRelationMetadataUpdate {
    pub(crate) fn new(relation: String, insert_evidence: Option<PreparedInsertEvidence>) -> Self {
        Self {
            relation,
            insert_evidence,
        }
    }
}

impl PreparedRelationMetadataTransition {
    fn unchanged() -> Self {
        Self {
            prospective_store: None,
        }
    }

    fn replace(prospective_store: RelationMetadataStore) -> Self {
        Self {
            prospective_store: Some(prospective_store),
        }
    }

    pub(crate) fn commit(self, store: &mut RelationMetadataStore) {
        if let Some(prospective_store) = self.prospective_store {
            *store = prospective_store;
        }
    }
}

impl RelationSnapshot {
    pub(crate) fn pack(&self, py: Python<'_>) -> PyResult<PyObject> {
        let snapshot = PyDict::new(py);
        snapshot.set_item("relation", &self.relation)?;
        snapshot.set_item("metadata_present", self.metadata.is_some())?;
        snapshot.set_item("row_count", self.row_count)?;

        let roles = PyList::empty(py);
        if let Some(metadata) = &self.metadata {
            for role in &metadata.roles {
                roles.append(pack_role(py, role)?)?;
            }
        }
        snapshot.set_item("roles", roles)?;

        let facts = PyList::empty(py);
        if let Some(metadata) = &self.metadata {
            for (key, records) in &metadata.facts {
                facts.append(pack_fact(py, &self.relation, key, records)?)?;
            }
        }
        snapshot.set_item("facts", facts)?;
        Ok(snapshot.into())
    }
}

pub(crate) fn pack_session_evidence(
    py: Python<'_>,
    mut snapshots: Vec<RelationSnapshot>,
    selected_relation: Option<&str>,
) -> PyResult<PyObject> {
    snapshots.sort_by(|left, right| left.relation.cmp(&right.relation));
    let result = PyDict::new(py);
    result.set_item("program_hash", program_hash(&snapshots)?)?;
    let relations = PyDict::new(py);
    for snapshot in snapshots {
        if selected_relation.is_some_and(|selected| selected != snapshot.relation) {
            continue;
        }
        relations.set_item(&snapshot.relation, snapshot.pack(py)?)?;
    }
    result.set_item("relations", relations)?;
    Ok(result.into())
}

fn parse_roles(
    relation: &str,
    arguments: &[LogicArgumentSchema],
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<RelationRole>> {
    let sequence = sequence(value, &format!("Relation '{relation}' roles"))?;
    let items = sequence.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let actual = items.len();
    if actual != arguments.len() {
        return Err(metadata_error(format!(
            "Relation '{relation}' expected {} roles but received {actual}",
            arguments.len()
        )));
    }

    let mut parsed = Vec::with_capacity(actual);
    let mut names = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        let dict = dictionary(item, &format!("Relation '{relation}' role {index}"))?;
        reject_unknown_keys(
            dict,
            &["name", "sort", "type"],
            &format!("Relation '{relation}' role {index}"),
        )?;
        let name = required_string(dict, "name", &format!("Relation '{relation}' role {index}"))?;
        if name.is_empty() {
            return Err(metadata_error(format!(
                "Relation '{relation}' role {index} name must be non-empty"
            )));
        }
        if !names.insert(name.clone()) {
            return Err(metadata_error(format!(
                "Relation '{relation}' has duplicate role name '{name}'"
            )));
        }

        let argument = &arguments[index];
        if argument.source_named() && name != argument.name() {
            return Err(metadata_error(format!(
                "Relation '{relation}' role {index} expected name '{}' but received '{name}'",
                argument.name()
            )));
        }

        let supplied_sort =
            optional_string(dict, "sort", &format!("Relation '{relation}' role {index}"))?;
        if let Some(supplied) = supplied_sort.as_deref() {
            if Some(supplied) != argument.sort() {
                return Err(metadata_error(format!(
                    "Relation '{relation}' role {index} sort mismatch: expected {:?}, received '{supplied}'",
                    argument.sort()
                )));
            }
        }

        let expected_type = types::scalar_type_name(&argument.scalar_type());
        let supplied_type =
            optional_string(dict, "type", &format!("Relation '{relation}' role {index}"))?;
        if let Some(supplied) = supplied_type.as_deref() {
            if supplied != expected_type {
                return Err(metadata_error(format!(
                    "Relation '{relation}' role {index} type mismatch: expected '{expected_type}', received '{supplied}'"
                )));
            }
        }

        parsed.push(RelationRole {
            name,
            sort: argument.sort().map(str::to_string),
            scalar_type: argument.scalar_type(),
        });
    }
    Ok(parsed)
}

fn parse_facts(
    relation: &str,
    schema: &Schema,
    schema_sha256: [u8; 32],
    value: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<FactKey, BTreeSet<ProvenanceRecord>>> {
    let sequence = sequence(value, &format!("Relation '{relation}' facts"))?;
    let mut facts = BTreeMap::<FactKey, BTreeSet<ProvenanceRecord>>::new();
    for (fact_index, item) in sequence.try_iter()?.enumerate() {
        let item = item?;
        let dict = dictionary(&item, &format!("Relation '{relation}' fact {fact_index}"))?;
        reject_unknown_keys(
            dict,
            &["tuple", "cells", "provenance"],
            &format!("Relation '{relation}' fact {fact_index}"),
        )?;
        let tuple = dict.get_item("tuple")?;
        let cells = dict.get_item("cells")?;
        if tuple.is_some() == cells.is_some() {
            return Err(metadata_error(format!(
                "Relation '{relation}' fact {fact_index} must supply exactly one of 'tuple' or 'cells'"
            )));
        }
        let key_cells = match (tuple, cells) {
            (Some(tuple), None) => parse_friendly_tuple(relation, fact_index, schema, &tuple)?,
            (None, Some(cells)) => parse_exact_cells(relation, fact_index, schema, &cells)?,
            _ => unreachable!("exactly one tuple representation was checked"),
        };
        let provenance = dict.get_item("provenance")?.ok_or_else(|| {
            metadata_error(format!(
                "Relation '{relation}' fact {fact_index} is missing required 'provenance'"
            ))
        })?;
        let records = parse_records(relation, fact_index, &provenance)?;
        facts
            .entry(FactKey {
                schema_sha256,
                cells: key_cells,
            })
            .or_default()
            .extend(records);
    }
    Ok(facts)
}

fn parse_friendly_tuple(
    relation: &str,
    fact_index: usize,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<TypedCell>> {
    let sequence = sequence(
        value,
        &format!("Relation '{relation}' fact {fact_index} tuple"),
    )?;
    let items = sequence.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let actual = items.len();
    if actual != schema.arity() {
        return Err(metadata_error(format!(
            "Relation '{relation}' fact {fact_index} tuple arity mismatch: expected {}, received {actual}",
            schema.arity()
        )));
    }
    let mut cells = Vec::with_capacity(actual);
    for (column, item) in items.iter().enumerate() {
        let scalar_type = schema.column_type(column).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Relation '{relation}' schema is missing column {column}"
            ))
        })?;
        cells.push(parse_friendly_cell(
            relation,
            fact_index,
            column,
            scalar_type,
            item,
        )?);
    }
    Ok(cells)
}

fn parse_friendly_cell(
    relation: &str,
    fact_index: usize,
    column: usize,
    scalar_type: ScalarType,
    value: &Bound<'_, PyAny>,
) -> PyResult<TypedCell> {
    let location = format!(
        "Relation '{relation}' fact {fact_index} tuple column {column} ({})",
        types::scalar_type_name(&scalar_type)
    );
    let bytes = match scalar_type {
        ScalarType::U32 | ScalarType::Symbol => {
            require_python_int(value, &location)?;
            let parsed = value.extract::<u32>().map_err(|_| {
                metadata_error(format!(
                    "{location} must be in 0..={} for {}",
                    u32::MAX,
                    types::scalar_type_name(&scalar_type)
                ))
            })?;
            parsed.to_le_bytes().to_vec()
        }
        ScalarType::U64 => {
            require_python_int(value, &location)?;
            value
                .extract::<u64>()
                .map_err(|_| {
                    metadata_error(format!("{location} must be in 0..={} for u64", u64::MAX))
                })?
                .to_le_bytes()
                .to_vec()
        }
        ScalarType::I32 => {
            require_python_int(value, &location)?;
            value
                .extract::<i32>()
                .map_err(|_| {
                    metadata_error(format!(
                        "{location} must be in {}..={} for i32",
                        i32::MIN,
                        i32::MAX
                    ))
                })?
                .to_le_bytes()
                .to_vec()
        }
        ScalarType::I64 => {
            require_python_int(value, &location)?;
            value
                .extract::<i64>()
                .map_err(|_| {
                    metadata_error(format!(
                        "{location} must be in {}..={} for i64",
                        i64::MIN,
                        i64::MAX
                    ))
                })?
                .to_le_bytes()
                .to_vec()
        }
        ScalarType::F32 => {
            if !value.is_instance_of::<PyFloat>() {
                return Err(metadata_error(format!(
                    "{location} requires a Python float"
                )));
            }
            (value.extract::<f64>()? as f32).to_le_bytes().to_vec()
        }
        ScalarType::F64 => {
            if !value.is_instance_of::<PyFloat>() {
                return Err(metadata_error(format!(
                    "{location} requires a Python float"
                )));
            }
            value.extract::<f64>()?.to_le_bytes().to_vec()
        }
        ScalarType::Bool => {
            if !value.is_instance_of::<PyBool>() {
                return Err(metadata_error(format!("{location} requires a Python bool")));
            }
            vec![u8::from(value.extract::<bool>()?)]
        }
    };
    Ok(TypedCell {
        type_code: scalar_type.to_code(),
        bytes,
    })
}

fn parse_exact_cells(
    relation: &str,
    fact_index: usize,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<TypedCell>> {
    let sequence = sequence(
        value,
        &format!("Relation '{relation}' fact {fact_index} cells"),
    )?;
    let items = sequence.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let actual = items.len();
    if actual != schema.arity() {
        return Err(metadata_error(format!(
            "Relation '{relation}' fact {fact_index} cells arity mismatch: expected {}, received {actual}",
            schema.arity()
        )));
    }

    let mut cells = Vec::with_capacity(actual);
    for (column, item) in items.iter().enumerate() {
        let context = format!("Relation '{relation}' fact {fact_index} cell {column}");
        let dict = dictionary(item, &context)?;
        reject_unknown_keys(dict, &["type", "hex"], &context)?;
        let supplied_type = required_string(dict, "type", &context)?;
        let expected_type = schema.column_type(column).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Relation '{relation}' schema is missing column {column}"
            ))
        })?;
        let expected_name = types::scalar_type_name(&expected_type);
        if supplied_type != expected_name {
            return Err(metadata_error(format!(
                "{context} type mismatch: expected '{expected_name}', received '{supplied_type}'"
            )));
        }
        let encoded = required_string(dict, "hex", &context)?;
        if encoded.bytes().any(|byte| byte.is_ascii_uppercase()) {
            return Err(metadata_error(format!(
                "{context} hex must use lowercase characters"
            )));
        }
        if encoded.len() % 2 != 0 {
            return Err(metadata_error(format!(
                "{context} hex must contain an even number of characters"
            )));
        }
        let bytes = decode_hex(&encoded).ok_or_else(|| {
            metadata_error(format!(
                "{context} hex must contain only lowercase hexadecimal characters"
            ))
        })?;
        if bytes.len() != expected_type.size_bytes() {
            return Err(metadata_error(format!(
                "{context} must encode exactly {} bytes for {expected_name}, received {} bytes",
                expected_type.size_bytes(),
                bytes.len()
            )));
        }
        if expected_type == ScalarType::Bool && !matches!(bytes.as_slice(), [0] | [1]) {
            return Err(metadata_error(format!(
                "{context} bool hex must encode 00 or 01"
            )));
        }
        cells.push(TypedCell {
            type_code: expected_type.to_code(),
            bytes,
        });
    }
    Ok(cells)
}

fn parse_records(
    relation: &str,
    fact_index: usize,
    value: &Bound<'_, PyAny>,
) -> PyResult<BTreeSet<ProvenanceRecord>> {
    let sequence = sequence(
        value,
        &format!("Relation '{relation}' fact {fact_index} provenance"),
    )?;
    let mut records = BTreeSet::new();
    for (record_index, item) in sequence.try_iter()?.enumerate() {
        let item = item?;
        let context =
            format!("Relation '{relation}' fact {fact_index} provenance record {record_index}");
        let dict = dictionary(&item, &context)?;
        reject_unknown_keys(
            dict,
            &[
                "source",
                "document",
                "span",
                "content_hash",
                "kind",
                "polarity",
            ],
            &context,
        )?;
        let source = optional_string(dict, "source", &context)?;
        let document = optional_string(dict, "document", &context)?;
        let content_hash = optional_string(dict, "content_hash", &context)?;
        let kind = optional_string(dict, "kind", &context)?;
        let polarity = optional_string(dict, "polarity", &context)?;
        let span = match dict.get_item("span")? {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => Some(parse_span(&value, &context)?),
        };
        let record = ProvenanceRecord {
            source,
            document,
            span,
            content_hash,
            kind,
            polarity,
        };
        if record.source.is_none()
            && record.document.is_none()
            && record.span.is_none()
            && record.content_hash.is_none()
            && record.kind.is_none()
            && record.polarity.is_none()
        {
            return Err(metadata_error(format!(
                "{context} must contain at least one non-null field"
            )));
        }
        records.insert(record);
    }
    Ok(records)
}

fn parse_span(value: &Bound<'_, PyAny>, parent: &str) -> PyResult<ProvenanceSpan> {
    let context = format!("{parent} span");
    let dict = dictionary(value, &context)?;
    reject_unknown_keys(dict, &["start", "end"], &context)?;
    let start = non_negative_offset(dict, "start", &context)?;
    let end = non_negative_offset(dict, "end", &context)?;
    if start > end {
        return Err(metadata_error(format!(
            "{context} requires start <= end, received start={start}, end={end}"
        )));
    }
    Ok(ProvenanceSpan { start, end })
}

fn validate_fact_membership(
    relation: &str,
    schema: &Schema,
    provider: &Arc<CudaKernelProvider>,
    relation_buffer: &CudaBuffer,
    facts: &BTreeMap<FactKey, BTreeSet<ProvenanceRecord>>,
) -> PyResult<()> {
    if facts.is_empty() {
        return Ok(());
    }

    let keys = facts.keys().collect::<Vec<_>>();
    let membership = fact_membership_mask(relation, schema, provider, relation_buffer, &keys)?;
    if let Some(index) = membership.iter().position(|present| !present) {
        let key = keys[index];
        return Err(metadata_error(format!(
            "Relation '{relation}' evidence fact {} is not present in the uploaded relation",
            format_cells(&key.cells)
        )));
    }
    Ok(())
}

fn fact_membership_mask(
    relation: &str,
    schema: &Schema,
    provider: &Arc<CudaKernelProvider>,
    relation_buffer: &CudaBuffer,
    facts: &[&FactKey],
) -> PyResult<Vec<bool>> {
    if facts.is_empty() {
        return Ok(Vec::new());
    }

    let mut columns: Vec<Vec<u8>> = schema
        .columns
        .iter()
        .map(|(_, scalar_type)| Vec::with_capacity(facts.len() * scalar_type.size_bytes()))
        .collect();
    for (fact_index, key) in facts.iter().enumerate() {
        if key.cells.len() != schema.arity() {
            return Err(PyRuntimeError::new_err(format!(
                "Relation '{relation}' prepared fact {fact_index} has arity {} but schema arity is {}",
                key.cells.len(),
                schema.arity()
            )));
        }
        for (column, cell) in key.cells.iter().enumerate() {
            let expected = schema.column_type(column).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Relation '{relation}' schema is missing column {column}"
                ))
            })?;
            if cell.scalar_type() != expected || cell.bytes.len() != expected.size_bytes() {
                return Err(PyRuntimeError::new_err(format!(
                    "Relation '{relation}' prepared fact {fact_index} column {column} does not match the compiled scalar type"
                )));
            }
            columns[column].extend_from_slice(&cell.bytes);
        }
    }
    let slices = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let fact_buffer = provider
        .create_buffer_from_slices(&slices, schema.clone())
        .map_err(types::xlog_err)?;
    let keys = (0..schema.arity()).collect::<Vec<_>>();
    let membership = provider
        .membership_mask(&fact_buffer, relation_buffer, &keys, &keys)
        .map_err(types::xlog_err)?;
    if membership.len() != facts.len() {
        return Err(PyRuntimeError::new_err(format!(
            "Relation '{relation}' membership validation returned {} mask entries for {} fact keys",
            membership.len(),
            facts.len()
        )));
    }
    Ok(membership)
}

fn remove_matching_facts(
    relation: &str,
    schema: &Schema,
    provider: &Arc<CudaKernelProvider>,
    tuples: &CudaBuffer,
    facts: &mut BTreeMap<FactKey, BTreeSet<ProvenanceRecord>>,
) -> PyResult<()> {
    if facts.is_empty() {
        return Ok(());
    }
    let keys = facts.keys().collect::<Vec<_>>();
    let membership = fact_membership_mask(relation, schema, provider, tuples, &keys)?;
    let removed = keys
        .into_iter()
        .zip(membership)
        .filter(|(_, present)| *present)
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    for key in removed {
        facts.remove(&key);
    }
    Ok(())
}

fn union_fact_records(
    destination: &mut BTreeMap<FactKey, BTreeSet<ProvenanceRecord>>,
    incoming: BTreeMap<FactKey, BTreeSet<ProvenanceRecord>>,
) {
    for (key, records) in incoming {
        destination.entry(key).or_default().extend(records);
    }
}

fn pack_role(py: Python<'_>, role: &RelationRole) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("name", &role.name)?;
    match &role.sort {
        Some(sort) => dict.set_item("sort", sort)?,
        None => dict.set_item("sort", py.None())?,
    }
    dict.set_item("type", types::scalar_type_name(&role.scalar_type))?;
    Ok(dict.into())
}

fn pack_fact(
    py: Python<'_>,
    relation: &str,
    key: &FactKey,
    records: &BTreeSet<ProvenanceRecord>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("identity", fact_identity(relation, &key.cells)?)?;

    let tuple = PyList::empty(py);
    let cells = PyList::empty(py);
    for cell in &key.cells {
        append_friendly_cell(&tuple, cell)?;
        let exact = PyDict::new(py);
        exact.set_item("type", types::scalar_type_name(&cell.scalar_type()))?;
        exact.set_item("hex", encode_hex(&cell.bytes))?;
        cells.append(exact)?;
    }
    dict.set_item("tuple", tuple)?;
    dict.set_item("cells", cells)?;

    let provenance = PyList::empty(py);
    for record in records {
        provenance.append(pack_record(py, record)?)?;
    }
    dict.set_item("provenance", provenance)?;
    Ok(dict.into())
}

fn append_friendly_cell(list: &Bound<'_, PyList>, cell: &TypedCell) -> PyResult<()> {
    match cell.scalar_type() {
        ScalarType::U32 | ScalarType::Symbol => list.append(read_u32(&cell.bytes)),
        ScalarType::U64 => list.append(read_u64(&cell.bytes)),
        ScalarType::I32 => list.append(read_i32(&cell.bytes)),
        ScalarType::I64 => list.append(read_i64(&cell.bytes)),
        ScalarType::F32 => list.append(read_f32(&cell.bytes)),
        ScalarType::F64 => list.append(read_f64(&cell.bytes)),
        ScalarType::Bool => list.append(cell.bytes[0] != 0),
    }
}

fn pack_record(py: Python<'_>, record: &ProvenanceRecord) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    set_optional_string(&dict, py, "source", record.source.as_deref())?;
    set_optional_string(&dict, py, "document", record.document.as_deref())?;
    match &record.span {
        Some(span) => {
            let packed = PyDict::new(py);
            packed.set_item("start", span.start)?;
            packed.set_item("end", span.end)?;
            dict.set_item("span", packed)?;
        }
        None => dict.set_item("span", py.None())?,
    }
    set_optional_string(&dict, py, "content_hash", record.content_hash.as_deref())?;
    set_optional_string(&dict, py, "kind", record.kind.as_deref())?;
    set_optional_string(&dict, py, "polarity", record.polarity.as_deref())?;
    Ok(dict.into())
}

fn set_optional_string(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    key: &str,
    value: Option<&str>,
) -> PyResult<()> {
    match value {
        Some(value) => dict.set_item(key, value),
        None => dict.set_item(key, py.None()),
    }
}

fn program_hash(snapshots: &[RelationSnapshot]) -> PyResult<String> {
    let mut digest = Sha256::new();
    digest.update(PROGRAM_EVIDENCE_DOMAIN);
    write_u32_len(&mut digest, snapshots.len(), "relation snapshot count")?;
    for snapshot in snapshots {
        write_string(&mut digest, &snapshot.relation, "relation name")?;
        write_u32_len(&mut digest, snapshot.schema_arity, "relation schema arity")?;
        digest.update(snapshot.schema_sha256);
        digest.update((snapshot.row_count as u64).to_le_bytes());
        digest.update([u8::from(snapshot.metadata.is_some())]);
        match &snapshot.metadata {
            None => {
                write_u32_len(&mut digest, 0, "role count")?;
                write_u32_len(&mut digest, 0, "fact count")?;
            }
            Some(metadata) => {
                write_u32_len(&mut digest, metadata.roles.len(), "role count")?;
                for role in &metadata.roles {
                    write_string(&mut digest, &role.name, "role name")?;
                    write_optional_string(&mut digest, role.sort.as_deref(), "role sort")?;
                    digest.update([role.scalar_type.to_code()]);
                }
                write_u32_len(&mut digest, metadata.facts.len(), "fact count")?;
                for (fact, records) in &metadata.facts {
                    write_cells(&mut digest, &fact.cells)?;
                    write_u32_len(&mut digest, records.len(), "provenance record count")?;
                    for record in records {
                        write_record(&mut digest, record)?;
                    }
                }
            }
        }
    }
    Ok(prefixed_digest(digest))
}

fn fact_identity(relation: &str, cells: &[TypedCell]) -> PyResult<String> {
    let mut digest = Sha256::new();
    digest.update(FACT_IDENTITY_DOMAIN);
    write_string(&mut digest, relation, "predicate name")?;
    write_u32_len(&mut digest, cells.len(), "predicate arity")?;
    write_cells(&mut digest, cells)?;
    Ok(prefixed_digest(digest))
}

pub(crate) fn relation_schema_fingerprint(
    relation: &str,
    arguments: &[LogicArgumentSchema],
) -> PyResult<[u8; 32]> {
    let mut digest = Sha256::new();
    digest.update(SCHEMA_IDENTITY_DOMAIN);
    write_string(&mut digest, relation, "predicate name")?;
    write_u32_len(&mut digest, arguments.len(), "predicate arity")?;
    for argument in arguments {
        write_string(&mut digest, argument.name(), "compiled column name")?;
        digest.update([argument.scalar_type().to_code()]);
        match argument.sort() {
            None => digest.update([0]),
            Some(sort) => {
                digest.update([1]);
                write_string(&mut digest, sort, "source domain alias")?;
            }
        }
    }
    Ok(digest.finalize().into())
}

fn write_cells(digest: &mut Sha256, cells: &[TypedCell]) -> PyResult<()> {
    for cell in cells {
        digest.update([cell.type_code]);
        write_u32_len(digest, cell.bytes.len(), "cell byte length")?;
        digest.update(&cell.bytes);
    }
    Ok(())
}

fn write_record(digest: &mut Sha256, record: &ProvenanceRecord) -> PyResult<()> {
    write_optional_string(digest, record.source.as_deref(), "record source")?;
    write_optional_string(digest, record.document.as_deref(), "record document")?;
    match &record.span {
        None => digest.update([0]),
        Some(span) => {
            digest.update([1]);
            digest.update(span.start.to_le_bytes());
            digest.update(span.end.to_le_bytes());
        }
    }
    write_optional_string(
        digest,
        record.content_hash.as_deref(),
        "record content hash",
    )?;
    write_optional_string(digest, record.kind.as_deref(), "record kind")?;
    write_optional_string(digest, record.polarity.as_deref(), "record polarity")
}

fn write_optional_string(digest: &mut Sha256, value: Option<&str>, label: &str) -> PyResult<()> {
    match value {
        None => digest.update([0]),
        Some(value) => {
            digest.update([1]);
            write_string(digest, value, label)?;
        }
    }
    Ok(())
}

fn write_string(digest: &mut Sha256, value: &str, label: &str) -> PyResult<()> {
    write_u32_len(digest, value.len(), label)?;
    digest.update(value.as_bytes());
    Ok(())
}

fn write_u32_len(digest: &mut Sha256, value: usize, label: &str) -> PyResult<()> {
    let value = u32::try_from(value)
        .map_err(|_| metadata_error(format!("{label} exceeds the version-1 u32 length limit")))?;
    digest.update(value.to_le_bytes());
    Ok(())
}

fn sequence<'py>(
    value: &'py Bound<'py, PyAny>,
    context: &str,
) -> PyResult<&'py Bound<'py, PySequence>> {
    if value.is_instance_of::<PyString>() {
        return Err(metadata_error(format!("{context} must be a sequence")));
    }
    value
        .downcast::<PySequence>()
        .map_err(|_| metadata_error(format!("{context} must be a sequence")))
}

fn dictionary<'py>(
    value: &'py Bound<'py, PyAny>,
    context: &str,
) -> PyResult<&'py Bound<'py, PyDict>> {
    value
        .downcast::<PyDict>()
        .map_err(|_| metadata_error(format!("{context} must be a dictionary")))
}

fn reject_unknown_keys(dict: &Bound<'_, PyDict>, allowed: &[&str], context: &str) -> PyResult<()> {
    for (key, _) in dict.iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| metadata_error(format!("{context} dictionary keys must be strings")))?;
        if !allowed.contains(&key.as_str()) {
            return Err(metadata_error(format!(
                "{context} contains unknown key '{key}'"
            )));
        }
    }
    Ok(())
}

fn required_string(dict: &Bound<'_, PyDict>, key: &str, context: &str) -> PyResult<String> {
    let value = dict
        .get_item(key)?
        .ok_or_else(|| metadata_error(format!("{context} is missing required '{key}'")))?;
    if value.is_none() {
        return Err(metadata_error(format!(
            "{context} field '{key}' must be a string"
        )));
    }
    value
        .extract::<String>()
        .map_err(|_| metadata_error(format!("{context} field '{key}' must be a string")))
}

fn optional_string(dict: &Bound<'_, PyDict>, key: &str, context: &str) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    value
        .extract::<String>()
        .map(Some)
        .map_err(|_| metadata_error(format!("{context} field '{key}' must be a string or None")))
}

fn non_negative_offset(dict: &Bound<'_, PyDict>, key: &str, context: &str) -> PyResult<u64> {
    let value = dict
        .get_item(key)?
        .ok_or_else(|| metadata_error(format!("{context} is missing required '{key}'")))?;
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
        return Err(metadata_error(format!(
            "{context} '{key}' must be a non-negative integer"
        )));
    }
    value.extract::<u64>().map_err(|_| {
        metadata_error(format!(
            "{context} '{key}' must be a non-negative integer representable as u64"
        ))
    })
}

fn require_python_int(value: &Bound<'_, PyAny>, context: &str) -> PyResult<()> {
    if value.is_instance_of::<PyBool>() {
        return Err(metadata_error(format!(
            "{context} rejects Python bool; an integer is required"
        )));
    }
    if !value.is_instance_of::<PyInt>() {
        return Err(metadata_error(format!("{context} requires a Python int")));
    }
    Ok(())
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let raw = value.as_bytes();
    for pair in raw.chunks_exact(2) {
        let high = hex_digit(pair[0])?;
        let low = hex_digit(pair[1])?;
        bytes.push((high << 4) | low);
    }
    Some(bytes)
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn prefixed_digest(digest: Sha256) -> String {
    format!("sha256:{}", encode_hex(&digest.finalize()))
}

fn format_cells(cells: &[TypedCell]) -> String {
    let cells = cells
        .iter()
        .map(|cell| {
            format!(
                "{}:{}",
                types::scalar_type_name(&cell.scalar_type()),
                encode_hex(&cell.bytes)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{cells}]")
}

pub(crate) fn metadata_error(message: String) -> PyErr {
    RelationMetadataError::new_err(message)
}

fn read_u32(bytes: &[u8]) -> u32 {
    let mut array = [0u8; 4];
    array.copy_from_slice(bytes);
    u32::from_le_bytes(array)
}

fn read_u64(bytes: &[u8]) -> u64 {
    let mut array = [0u8; 8];
    array.copy_from_slice(bytes);
    u64::from_le_bytes(array)
}

fn read_i32(bytes: &[u8]) -> i32 {
    let mut array = [0u8; 4];
    array.copy_from_slice(bytes);
    i32::from_le_bytes(array)
}

fn read_i64(bytes: &[u8]) -> i64 {
    let mut array = [0u8; 8];
    array.copy_from_slice(bytes);
    i64::from_le_bytes(array)
}

fn read_f32(bytes: &[u8]) -> f32 {
    f32::from_bits(read_u32(bytes))
}

fn read_f64(bytes: &[u8]) -> f64 {
    f64::from_bits(read_u64(bytes))
}
