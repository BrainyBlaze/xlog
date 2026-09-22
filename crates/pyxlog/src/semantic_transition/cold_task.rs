//! Canonical creator-thread construction for one fresh semantic task parent.

use std::collections::BTreeSet;
use std::thread::ThreadId;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt, PyString, PyTuple};
use sha2::{Digest, Sha256};
use xlog_core::{RelId, ScalarType, Schema};
use xlog_cuda::{
    SemanticActionCatalogue, SemanticAdmissionRecords, SemanticArgument, SemanticPredicateRecord,
    SemanticRecordRole, SemanticTypedRecord,
};

use super::{invalid, xlog_err, PySemanticTransitionController, PySemanticTransitionSession};

const MAX_POSITION: u64 = 262_144;
const STATEMENT_RECORDS: (u32, u32, u32) = (0, 1, 2);

#[derive(Clone)]
struct FreshRecord {
    role: u64,
    bytes: Vec<u8>,
}

/// One immutable, XLOG-produced set of non-model arguments for a fresh parent.
///
/// The model Runtime adds its cache, model-contract and selected training-view
/// owners before forwarding these exact values to ``bind_parent``. The prefix
/// and tensor collections are always empty; no Python consumer encodes native
/// state roles or private record bytes.
#[pyclass(
    name = "SemanticTransitionFreshParent",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticTransitionFreshParent {
    owner_thread: ThreadId,
    provenance_capacity_records: u64,
    prefix_capacity: u64,
    feedback_capacity: u64,
    pad_token: u64,
    terminal_tokens: Vec<u64>,
    final_intent_payload_bytes: u64,
    intent_effect: Vec<u8>,
    intent_entry_capacity: u64,
    intent_payload_capacity_bytes: usize,
    generations: (u64, u64, u64, u64),
    training_cursor: u64,
    training_rng: [u64; 4],
    fuel: u64,
    rng: (u32, u64, u8, u32),
    role_counts: [u64; 55],
    authority_decisions_capacity_bytes: usize,
    records: Vec<FreshRecord>,
}

#[pymethods]
impl PySemanticTransitionFreshParent {
    fn require_creator(&self) -> PyResult<()> {
        super::require_creator_thread(self.owner_thread)
    }

    /// Return the exact pre-model metadata dictionary accepted by Runtime.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.require_creator()?;
        let metadata = PyDict::new(py);
        metadata.set_item("recovered_instance", py.None())?;
        metadata.set_item("ring_head", 0)?;
        metadata.set_item(
            "provenance_capacity_records",
            self.provenance_capacity_records,
        )?;
        metadata.set_item("prefix_capacity", self.prefix_capacity)?;
        metadata.set_item("feedback_capacity", self.feedback_capacity)?;
        metadata.set_item("pad_token", self.pad_token)?;
        metadata.set_item("terminal_tokens", PyTuple::new(py, &self.terminal_tokens)?)?;
        metadata.set_item(
            "final_intent_payload_bytes",
            self.final_intent_payload_bytes,
        )?;
        metadata.set_item("intent_effect", PyBytes::new(py, &self.intent_effect))?;
        metadata.set_item("intent_entry_capacity", self.intent_entry_capacity)?;
        metadata.set_item(
            "intent_payload_capacity_bytes",
            self.intent_payload_capacity_bytes,
        )?;
        metadata.set_item("model_generation", self.generations.0)?;
        metadata.set_item("policy_generation", self.generations.1)?;
        metadata.set_item("neural_generation", self.generations.2)?;
        metadata.set_item("cache_generation", self.generations.3)?;
        metadata.set_item("training_cursor", self.training_cursor)?;
        metadata.set_item("training_rng", PyTuple::new(py, self.training_rng)?)?;
        metadata.set_item("fuel", self.fuel)?;
        metadata.set_item("rng", self.rng)?;
        metadata.set_item("role_counts", PyTuple::new(py, self.role_counts)?)?;
        metadata.set_item("active_layouts", PyTuple::empty(py))?;
        metadata.set_item(
            "authority_decisions_capacity_bytes",
            self.authority_decisions_capacity_bytes,
        )?;
        Ok(metadata.unbind())
    }

    /// Return the exact 32-slot fresh source ring. Every slot is padding.
    #[getter]
    fn source(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_creator()?;
        let rows = (0..32).map(|_| (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64));
        Ok(PyTuple::new(py, rows)?.unbind())
    }

    /// Fresh construction never accepts a computed prefix.
    #[getter]
    fn prefix(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_creator()?;
        Ok(PyTuple::empty(py).unbind())
    }

    /// Return XLOG's canonical non-model control records.
    #[getter]
    fn records(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_creator()?;
        let rows = self.records.iter().map(|record| {
            (
                record.role,
                0u64,
                record.bytes.len(),
                PyBytes::new(py, &record.bytes),
            )
        });
        Ok(PyTuple::new(py, rows)?.unbind())
    }

    /// Fresh construction has no caller-owned tensors.
    #[getter]
    fn tensors(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_creator()?;
        Ok(PyTuple::empty(py).unbind())
    }

    /// Return ``(metadata, source, prefix, records, tensors)`` without adapters.
    fn parts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let metadata = self.metadata(py)?;
        let source = self.source(py)?;
        let prefix = self.prefix(py)?;
        let records = self.records(py)?;
        let tensors = self.tensors(py)?;
        Ok(PyTuple::new(
            py,
            [
                metadata.into_any(),
                source.into_any(),
                prefix.into_any(),
                records.into_any(),
                tensors.into_any(),
            ],
        )?
        .unbind())
    }
}

/// Canonical cold producer for the fixed three-statement semantic task contract.
///
/// Construction parses the complete observer and admits the exact initial
/// theory, input facts and ordered task statements,
/// allocates one CUDA Session, creates its sole Controller and retains the one
/// typed fresh-parent input set. No object is returned when any parse,
/// admission, geometry or resource-budget check fails.
#[pyclass(name = "SemanticTransitionColdTask", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticTransitionColdTask {
    session: Py<PySemanticTransitionSession>,
    controller: Py<PySemanticTransitionController>,
    parent: Py<PySemanticTransitionFreshParent>,
}

#[pymethods]
impl PySemanticTransitionColdTask {
    #[new]
    #[pyo3(signature = (*, initial_theory, input_facts, observer_program, statements, capacities, admission_limits, device_ordinal, memory_bytes, provenance_capacity_records, prefix_capacity, feedback_capacity, pad_token, terminal_tokens, final_intent_payload_bytes, intent_effect, intent_entry_capacity, intent_payload_capacity_bytes, authority_decisions_capacity_bytes, generations, training_cursor, training_rng, fuel, rng))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the cold producer receives independent native resource budgets"
    )]
    fn new(
        py: Python<'_>,
        initial_theory: &Bound<'_, PyAny>,
        input_facts: &Bound<'_, PyAny>,
        observer_program: &Bound<'_, PyAny>,
        statements: &Bound<'_, PyAny>,
        capacities: (u32, u32, u32, u32),
        admission_limits: (u32, u32, u32, usize),
        device_ordinal: usize,
        memory_bytes: u64,
        provenance_capacity_records: u64,
        prefix_capacity: u64,
        feedback_capacity: u64,
        pad_token: u64,
        terminal_tokens: &Bound<'_, PyAny>,
        final_intent_payload_bytes: u64,
        intent_effect: &Bound<'_, PyAny>,
        intent_entry_capacity: u64,
        intent_payload_capacity_bytes: usize,
        authority_decisions_capacity_bytes: usize,
        generations: (u64, u64, u64, u64),
        training_cursor: u64,
        training_rng: [u64; 4],
        fuel: u64,
        rng: (u64, u8, u32),
    ) -> PyResult<Self> {
        let initial_theory = exact_text(initial_theory, "initial_theory")?;
        let input_facts = exact_text(input_facts, "input_facts")?;
        let observer_program = exact_text(observer_program, "observer_program")?;
        let statements = exact_statements(statements)?;
        let terminal_tokens = exact_unsigned_tuple(terminal_tokens, "terminal_tokens")?;
        let intent_effect = exact_bytes(intent_effect, "intent_effect")?;
        validate_parent_geometry(
            memory_bytes,
            provenance_capacity_records,
            prefix_capacity,
            feedback_capacity,
            pad_token,
            &terminal_tokens,
            final_intent_payload_bytes,
            &intent_effect,
            intent_entry_capacity,
            intent_payload_capacity_bytes,
            authority_decisions_capacity_bytes,
            generations.0,
            rng,
        )?;
        xlog_gpu::logic::SemanticLogicTaskProgram::compile(observer_program.clone(), [0, 1, 2])
            .map_err(xlog_err)?;

        let task_digest = task_digest(
            &initial_theory,
            &input_facts,
            &observer_program,
            &statements,
        );
        let admission = task_admission(
            &initial_theory,
            &input_facts,
            &observer_program,
            &statements,
        )?;
        let role_counts = initial_role_counts();
        let records = fresh_records(
            task_digest,
            capacities,
            admission_limits,
            device_ordinal,
            memory_bytes,
            prefix_capacity,
            feedback_capacity,
            pad_token,
            &terminal_tokens,
            generations,
            training_cursor,
            training_rng,
            fuel,
            rng,
        );
        let parent = Py::new(
            py,
            PySemanticTransitionFreshParent {
                owner_thread: std::thread::current().id(),
                provenance_capacity_records,
                prefix_capacity,
                feedback_capacity,
                pad_token,
                terminal_tokens,
                final_intent_payload_bytes,
                intent_effect,
                intent_entry_capacity,
                intent_payload_capacity_bytes,
                generations,
                training_cursor,
                training_rng,
                fuel,
                rng: (generations.0 as u32, rng.0, rng.1, rng.2),
                role_counts,
                authority_decisions_capacity_bytes,
                records,
            },
        )?;
        let session = Py::new(
            py,
            PySemanticTransitionSession::from_admission(
                admission,
                capacities,
                admission_limits,
                device_ordinal,
                memory_bytes,
            )?,
        )?;
        let controller = Py::new(
            py,
            PySemanticTransitionController::new(py, session.clone_ref(py))?,
        )?;
        Ok(Self {
            session,
            controller,
            parent,
        })
    }

    #[getter]
    fn session(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionSession>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.session.clone_ref(py))
    }

    #[getter]
    fn controller(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionController>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.controller.clone_ref(py))
    }

    #[getter]
    fn parent(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionFreshParent>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.parent.clone_ref(py))
    }

    #[getter]
    fn statement_records(&self, py: Python<'_>) -> PyResult<(u32, u32, u32)> {
        self.session.borrow(py).require_creator()?;
        Ok(STATEMENT_RECORDS)
    }
}

fn exact_text(value: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    if !value.is_exact_instance_of::<PyString>() {
        return Err(invalid(&format!("{name} requires an exact builtin str")));
    }
    Ok(value.cast::<PyString>()?.to_str()?.to_owned())
}

fn exact_bytes(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u8>> {
    if !value.is_exact_instance_of::<PyBytes>() {
        return Err(invalid(&format!("{name} requires exact builtin bytes")));
    }
    Ok(value.cast::<PyBytes>()?.as_bytes().to_vec())
}

fn exact_unsigned_tuple(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u64>> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(&format!("{name} requires an exact builtin tuple")));
    }
    value
        .cast::<PyTuple>()?
        .iter()
        .map(|item| {
            if !item.is_exact_instance_of::<PyInt>() {
                return Err(invalid(&format!(
                    "{name} entries require exact builtin ints"
                )));
            }
            item.extract::<u64>()
        })
        .collect()
}

fn exact_statements(value: &Bound<'_, PyAny>) -> PyResult<[(String, u8); 3]> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid("statements requires an exact builtin tuple"));
    }
    let value = value.cast::<PyTuple>()?;
    if value.len() != 3 {
        return Err(invalid(
            "cold semantic task requires exactly three ordered statements",
        ));
    }
    value
        .iter()
        .map(|row| {
            if !row.is_exact_instance_of::<PyTuple>() {
                return Err(invalid(
                    "each statement requires an exact (text, truth_mask) tuple",
                ));
            }
            let row = row.cast::<PyTuple>()?;
            if row.len() != 2 {
                return Err(invalid("each statement requires text and truth_mask"));
            }
            let text = exact_text(&row.get_item(0)?, "statement text")?;
            if text.is_empty() || !row.get_item(1)?.is_exact_instance_of::<PyInt>() {
                return Err(invalid(
                    "statement text must be nonempty and truth_mask an exact int",
                ));
            }
            let mask = row.get_item(1)?.extract::<u8>()?;
            if !(1..=15).contains(&mask) {
                return Err(invalid(
                    "statement truth_mask must be a nonempty Truth4 mask",
                ));
            }
            Ok((text, mask))
        })
        .collect::<PyResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| invalid("cold semantic task requires exactly three ordered statements"))
}

#[expect(
    clippy::too_many_arguments,
    reason = "validation covers independent native resource budgets"
)]
fn validate_parent_geometry(
    memory_bytes: u64,
    provenance_capacity_records: u64,
    prefix_capacity: u64,
    feedback_capacity: u64,
    pad_token: u64,
    terminal_tokens: &[u64],
    final_intent_payload_bytes: u64,
    intent_effect: &[u8],
    intent_entry_capacity: u64,
    intent_payload_capacity_bytes: usize,
    authority_decisions_capacity_bytes: usize,
    model_generation: u64,
    rng: (u64, u8, u32),
) -> PyResult<()> {
    let text_cardinality = SemanticActionCatalogue::current().text_cardinality() as u64;
    let token_set = terminal_tokens.iter().copied().collect::<BTreeSet<_>>();
    if memory_bytes == 0
        || provenance_capacity_records < 32
        || prefix_capacity == 0
        || feedback_capacity < 3
        || prefix_capacity
            .checked_add(32)
            .and_then(|extent| extent.checked_add(feedback_capacity))
            .is_none_or(|extent| extent > MAX_POSITION)
    {
        return Err(invalid(
            "fresh parent resource or text geometry budget is invalid",
        ));
    }
    if pad_token >= text_cardinality
        || terminal_tokens.is_empty()
        || terminal_tokens
            .iter()
            .any(|&token| token >= text_cardinality)
        || token_set.len() != terminal_tokens.len()
    {
        return Err(invalid("fresh parent token contract is invalid"));
    }
    if intent_effect.is_empty()
        || intent_entry_capacity == 0
        || final_intent_payload_bytes == 0
        || (intent_effect.len() as u64)
            .checked_add(final_intent_payload_bytes)
            .is_none_or(|needed| needed > intent_payload_capacity_bytes as u64)
        || authority_decisions_capacity_bytes == 0
    {
        return Err(invalid(
            "fresh parent intent or authority capacity is invalid",
        ));
    }
    if model_generation > u32::MAX as u64 || rng.0 >= 1 << 56 {
        return Err(invalid(
            "fresh parent generation or RNG stream exceeds the native ABI",
        ));
    }
    Ok(())
}

fn schema(columns: Vec<(&str, ScalarType, &str)>, key_columns: Vec<usize>) -> PyResult<Schema> {
    let labels = columns
        .iter()
        .map(|(_, _, label)| (*label).to_owned())
        .collect();
    let mut schema = Schema::new(
        columns
            .into_iter()
            .map(|(name, scalar, _)| (name.to_owned(), scalar))
            .collect(),
    )
    .with_sort_labels(labels)
    .map_err(super::val_err)?;
    schema.key_columns = key_columns;
    Ok(schema)
}

fn task_admission(
    initial_theory: &str,
    input_facts: &str,
    observer_program: &str,
    statements: &[(String, u8); 3],
) -> PyResult<SemanticAdmissionRecords> {
    let statement_schema = schema(
        vec![
            ("ordinal", ScalarType::U32, "task-statement-ordinal"),
            ("statement", ScalarType::Symbol, "task-statement-text"),
            ("truth_mask", ScalarType::U32, "truth4-admissibility-mask"),
        ],
        vec![0],
    )?;
    let input_schema = schema(
        vec![
            ("kind", ScalarType::U32, "task-input-kind"),
            ("content", ScalarType::Symbol, "task-input-text"),
        ],
        vec![0],
    )?;
    let mut records = statements
        .iter()
        .enumerate()
        .map(|(ordinal, (statement, mask))| SemanticTypedRecord {
            predicate: RelId(1),
            arguments: vec![
                SemanticArgument::U32(ordinal as u32),
                SemanticArgument::Symbol(xlog_core::symbol::intern(statement)),
                SemanticArgument::U32(u32::from(*mask)),
            ],
            qualifiers: vec![3, 4, 5],
        })
        .collect::<Vec<_>>();
    records.extend(
        [initial_theory, input_facts, observer_program]
            .into_iter()
            .enumerate()
            .map(|(kind, content)| SemanticTypedRecord {
                predicate: RelId(2),
                arguments: vec![
                    SemanticArgument::U32(kind as u32),
                    SemanticArgument::Symbol(xlog_core::symbol::intern(content)),
                ],
                qualifiers: Vec::new(),
            }),
    );
    Ok(SemanticAdmissionRecords {
        predicates: vec![
            SemanticPredicateRecord {
                predicate: RelId(1),
                role: SemanticRecordRole::Statement,
                schema: statement_schema,
            },
            SemanticPredicateRecord {
                predicate: RelId(2),
                role: SemanticRecordRole::Qualifier,
                schema: input_schema,
            },
        ],
        records,
        supports: Vec::new(),
    })
}

fn append_sized(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn task_digest(
    initial_theory: &str,
    input_facts: &str,
    observer_program: &str,
    statements: &[(String, u8); 3],
) -> [u8; 32] {
    let mut bytes = b"xlog.semantic.cold-task.v1\0".to_vec();
    for value in [initial_theory, input_facts, observer_program] {
        append_sized(&mut bytes, value.as_bytes());
    }
    for (statement, mask) in statements {
        append_sized(&mut bytes, statement.as_bytes());
        bytes.push(*mask);
    }
    Sha256::digest(bytes).into()
}

fn fresh_record(role: u64, state: &str, task_digest: [u8; 32], detail: &[u8]) -> FreshRecord {
    let mut bytes = b"xlog.semantic.fresh-control.v1\0".to_vec();
    bytes.extend_from_slice(&role.to_le_bytes());
    bytes.extend_from_slice(&task_digest);
    append_sized(&mut bytes, state.as_bytes());
    append_sized(&mut bytes, detail);
    FreshRecord { role, bytes }
}

fn append_u64s<const N: usize>(bytes: &mut Vec<u8>, values: [u64; N]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the environment record binds every independent cold resource budget"
)]
fn fresh_records(
    task_digest: [u8; 32],
    capacities: (u32, u32, u32, u32),
    admission_limits: (u32, u32, u32, usize),
    device_ordinal: usize,
    memory_bytes: u64,
    prefix_capacity: u64,
    feedback_capacity: u64,
    pad_token: u64,
    terminal_tokens: &[u64],
    generations: (u64, u64, u64, u64),
    training_cursor: u64,
    training_rng: [u64; 4],
    fuel: u64,
    rng: (u64, u8, u32),
) -> Vec<FreshRecord> {
    let mut rng_detail = Vec::new();
    append_u64s(&mut rng_detail, training_rng);
    append_u64s(
        &mut rng_detail,
        [generations.0, rng.0, u64::from(rng.1), u64::from(rng.2)],
    );
    let mut environment = Vec::new();
    append_u64s(
        &mut environment,
        [
            u64::from(capacities.0),
            u64::from(capacities.1),
            u64::from(capacities.2),
            u64::from(capacities.3),
            u64::from(admission_limits.0),
            u64::from(admission_limits.1),
            u64::from(admission_limits.2),
            admission_limits.3 as u64,
            device_ordinal as u64,
            memory_bytes,
            prefix_capacity,
            feedback_capacity,
            fuel,
        ],
    );
    let mut token_contract = pad_token.to_le_bytes().to_vec();
    token_contract.extend_from_slice(&(terminal_tokens.len() as u64).to_le_bytes());
    for token in terminal_tokens {
        token_contract.extend_from_slice(&token.to_le_bytes());
    }
    let mut generation_detail = Vec::new();
    append_u64s(
        &mut generation_detail,
        [
            generations.0,
            generations.1,
            generations.2,
            generations.3,
            training_cursor,
        ],
    );
    vec![
        fresh_record(26, "rng", task_digest, &rng_detail),
        fresh_record(27, "empty-replay-index", task_digest, &[]),
        fresh_record(28, "empty-replay-payload", task_digest, &[]),
        fresh_record(32, "empty-acknowledgements", task_digest, &[]),
        fresh_record(34, "task-goal", task_digest, &generation_detail),
        fresh_record(35, "fresh-world-root", task_digest, &[]),
        fresh_record(36, "empty-edit-journal", task_digest, &[]),
        fresh_record(37, "runtime-environment", task_digest, &environment),
        fresh_record(40, "fresh-checkpoint-root", task_digest, &[]),
        fresh_record(41, "cold-task-schema-v1", task_digest, &[]),
        fresh_record(42, "token-contract", task_digest, &token_contract),
        fresh_record(49, "empty-proof-records", task_digest, &[]),
        fresh_record(50, "fresh-learning-copy", task_digest, &[]),
    ]
}

fn initial_role_counts() -> [u64; 55] {
    let mut counts = [1u64; 55];
    counts[15] = 3;
    for role in [
        4u64, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 29, 43, 44, 45, 46,
        51, 52, 53, 54,
    ] {
        counts[role as usize - 1] = 0;
    }
    counts
}
