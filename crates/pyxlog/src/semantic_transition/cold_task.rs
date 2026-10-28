//! Canonical creator-thread construction for one fresh semantic task parent.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt, PyString, PyTuple};
use sha2::{Digest, Sha256};
use xlog_core::{RelId, ScalarType, Schema};
use xlog_cuda::{
    SemanticActionCatalogue, SemanticAdmissionRecords, SemanticArgument, SemanticPredicateRecord,
    SemanticRecordRole, SemanticTypedRecord,
};

use super::{
    invalid, material_owner, xlog_err, CheckpointSources, ProposalExpense,
    PySemanticTransitionController, PySemanticTransitionSession,
};

const MAX_POSITION: u64 = 262_144;

pub(super) struct EditableTaskSource {
    pub program: Arc<xlog_cuda::SemanticProgramAdmission>,
    pub observer_source: Option<String>,
    pub initial_source: String,
    pub initial_theory: String,
    pub input_facts: String,
    pub statements: Vec<(String, u8)>,
    pub query_records: Vec<u32>,
    pub observer_record_offset: u32,
    allocations: Option<Arc<EditableTaskAllocations>>,
}

struct EditableTaskAllocations {
    _vectors: xlog_cuda::SemanticReplayMaterialReservation,
    _text: xlog_cuda::SemanticReplayMaterialReservation,
    _program: xlog_gpu::logic::PositiveTaskAllocations,
    _context: Option<(
        Arc<EditableTaskAllocations>,
        xlog_cuda::SemanticReplayMaterialReservation,
    )>,
}

pub(super) struct EditableProgramOwner {
    pub(super) program: Arc<xlog_cuda::SemanticProgramAdmission>,
    pub(super) observer_record_offset: u32,
    pub(super) rule_library: Option<Arc<xlog_cuda::SemanticEvaluationRuleLibrary>>,
    pub(super) _library_metadata: Option<Arc<xlog_cuda::SemanticReplayMaterialReservation>>,
    _allocations: Option<Arc<EditableTaskAllocations>>,
    #[cfg(feature = "semantic-policy")]
    _actor_metadata: Option<Arc<super::actor_refresh::ActorConstructionMetadata>>,
}

impl EditableProgramOwner {
    pub(super) fn with_rule_library(
        &self,
        library: Arc<xlog_cuda::SemanticEvaluationRuleLibrary>,
        metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
    ) -> Arc<Self> {
        Arc::new(Self {
            program: Arc::clone(&self.program),
            observer_record_offset: self.observer_record_offset,
            rule_library: Some(library),
            _library_metadata: Some(metadata),
            _allocations: self._allocations.clone(),
            #[cfg(feature = "semantic-policy")]
            _actor_metadata: self._actor_metadata.clone(),
        })
    }
}

impl EditableTaskSource {
    pub(super) fn program_owner(&self) -> Arc<EditableProgramOwner> {
        self.program_owner_with_library(None)
    }

    pub(super) fn program_owner_with_library(
        &self,
        rule_library: Option<Arc<xlog_cuda::SemanticEvaluationRuleLibrary>>,
    ) -> Arc<EditableProgramOwner> {
        Arc::new(EditableProgramOwner {
            program: Arc::clone(&self.program),
            observer_record_offset: self.observer_record_offset,
            rule_library,
            _library_metadata: None,
            _allocations: self.allocations.as_ref().map(Arc::clone),
            #[cfg(feature = "semantic-policy")]
            _actor_metadata: None,
        })
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn actor_program_owner(
        &self,
        metadata: Arc<super::actor_refresh::ActorConstructionMetadata>,
        rule_library: Option<Arc<xlog_cuda::SemanticEvaluationRuleLibrary>>,
    ) -> Arc<EditableProgramOwner> {
        Arc::new(EditableProgramOwner {
            program: Arc::clone(&self.program),
            observer_record_offset: self.observer_record_offset,
            rule_library,
            _library_metadata: None,
            _allocations: self.allocations.as_ref().map(Arc::clone),
            _actor_metadata: Some(metadata),
        })
    }
}

/// Recover executable source only from the canonical typed cold admission.
/// This same path is used for fresh construction and checkpoint restoration;
/// serialized rule words are never accepted as an alternative source.
pub(super) fn editable_source_from_admission(
    admission: &SemanticAdmissionRecords,
) -> PyResult<Option<EditableTaskSource>> {
    editable_source_from_admission_owned(admission, None)
}

pub(super) fn editable_source_from_admission_bound(
    admission: &SemanticAdmissionRecords,
    material_sources: &Arc<super::material_source::MaterialSourceRegistry>,
) -> PyResult<Option<EditableTaskSource>> {
    let budget = material_sources.native_budget()?;
    editable_source_from_admission_owned(admission, Some(&budget))
}

fn evaluation_scope_record(
    admission: &SemanticAdmissionRecords,
    budget: Option<&xlog_cuda::SemanticReplayMaterialBudget>,
) -> PyResult<Option<usize>> {
    if admission.predicates.len() != 3 {
        return Ok(None);
    }
    let Some(scope) = admission.records.last() else {
        return Ok(None);
    };
    if scope.predicate != RelId(2)
        || !matches!(
            scope.arguments.first(),
            Some(SemanticArgument::U32(u32::MAX))
        )
    {
        return Ok(None);
    }
    let [SemanticArgument::U32(u32::MAX), SemanticArgument::Symbol(symbol)] =
        scope.arguments.as_slice()
    else {
        return Err(invalid(
            "evaluation context changed its original typed scope",
        ));
    };
    if !scope.qualifiers.is_empty() {
        return Err(invalid(
            "evaluation context changed its original typed scope",
        ));
    }
    let (symbols, _owner) = if let Some(budget) = budget {
        let (symbols, owner) = xlog_core::symbol::snapshot_checked_with_allocation_owner(
            &[*symbol],
            1,
            64,
            &mut |count, utf8| {
                let bytes = count
                    .checked_mul(std::mem::size_of::<(u32, Box<str>)>())
                    .and_then(|bytes| bytes.checked_mul(2))
                    .and_then(|bytes| utf8.checked_mul(2).and_then(|text| bytes.checked_add(text)))
                    .ok_or_else(
                        || xlog_core::symbol::SymbolSnapshotError::AllocationRejected {
                            detail: "evaluation scope allocation extent overflow".into(),
                        },
                    )?;
                let owner = budget.reserve_auxiliary(bytes as u64).map_err(|error| {
                    xlog_core::symbol::SymbolSnapshotError::AllocationRejected {
                        detail: error.to_string(),
                    }
                })?;
                Ok(Arc::new(owner) as Arc<dyn Send + Sync>)
            },
        )
        .map_err(|error| invalid(&error.to_string()))?;
        (symbols, Some(owner))
    } else {
        (
            xlog_core::symbol::snapshot_checked(&[*symbol], 1, 64)
                .map_err(|error| invalid(&error.to_string()))?,
            None,
        )
    };
    let scope = symbols
        .entries()
        .first()
        .map(|(_, text)| text.as_ref())
        .ok_or_else(|| invalid("evaluation context scope symbol is unavailable"))?;
    if scope.len() != 64
        || !scope
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "evaluation context scope is not its canonical lowercase digest",
        ));
    }
    Ok(Some(admission.records.len() - 1))
}

fn editable_source_from_admission_owned(
    admission: &SemanticAdmissionRecords,
    budget: Option<&xlog_cuda::SemanticReplayMaterialBudget>,
) -> PyResult<Option<EditableTaskSource>> {
    if admission.predicates.len() != 3
        || admission.predicates[0].predicate != RelId(1)
        || admission.predicates[0].role != SemanticRecordRole::Statement
        || admission.predicates[1].predicate != RelId(2)
        || admission.predicates[1].role != SemanticRecordRole::Qualifier
        || admission.predicates[2].predicate != RelId(3)
        || admission.predicates[2].role != SemanticRecordRole::Qualifier
        || admission.predicates[0].schema.sort_labels()
            != [
                "task-statement-ordinal",
                "task-statement-text",
                "truth4-admissibility-mask",
            ]
        || admission.predicates[1].schema.sort_labels() != ["task-input-kind", "task-input-text"]
        || admission.predicates[2].schema.sort_labels()
            != ["task-query-ordinal", "task-statement-record"]
        || admission.predicates[0].schema.key_columns != [0]
        || admission.predicates[1].schema.key_columns != [0]
        || admission.predicates[2].schema.key_columns != [0]
        || !admission.predicates[0]
            .schema
            .columns
            .iter()
            .map(|(name, scalar)| (name.as_str(), *scalar))
            .eq([
                ("ordinal", ScalarType::U32),
                ("statement", ScalarType::Symbol),
                ("truth_mask", ScalarType::U32),
            ])
        || !admission.predicates[1]
            .schema
            .columns
            .iter()
            .map(|(name, scalar)| (name.as_str(), *scalar))
            .eq([("kind", ScalarType::U32), ("content", ScalarType::Symbol)])
        || !admission.predicates[2]
            .schema
            .columns
            .iter()
            .map(|(name, scalar)| (name.as_str(), *scalar))
            .eq([
                ("ordinal", ScalarType::U32),
                ("statement_record", ScalarType::U32),
            ])
    {
        return Ok(None);
    }
    let scope_record = evaluation_scope_record(admission, budget)?;
    let record_end = scope_record.unwrap_or(admission.records.len());
    let mut begin = 0;
    let mut segments = 0;
    let mut source: Option<EditableTaskSource> = None;
    while begin < record_end {
        let statement_end = begin
            + admission.records[begin..record_end]
                .iter()
                .take_while(|record| record.predicate == RelId(1))
                .count();
        let input_end = statement_end
            + admission.records[statement_end..record_end]
                .iter()
                .take_while(|record| record.predicate == RelId(2))
                .count();
        let end = input_end
            + admission.records[input_end..record_end]
                .iter()
                .take_while(|record| record.predicate == RelId(3))
                .count();
        if end == begin {
            return editable_source_segment(admission, begin, record_end, scope_record, budget);
        }
        segments += 1;
        if segments > 2 || (segments > 1 && scope_record.is_none()) {
            return Err(invalid(
                "editable task admission changed its original context segments",
            ));
        }
        let Some(mut target) =
            editable_source_segment(admission, begin, end, scope_record, budget)?
        else {
            return Ok(None);
        };
        if let Some(prefix) = source {
            let reservation = budget
                .map(|budget| {
                    let bytes =
                        xlog_cuda::SemanticProgramAdmission::evaluation_context_allocation_bytes(
                            &prefix.program,
                            &target.program,
                        )
                        .map_err(invalid)?;
                    budget.reserve_auxiliary(bytes as u64).map_err(xlog_err)
                })
                .transpose()?;
            target.program = Arc::new(
                xlog_cuda::SemanticProgramAdmission::evaluation_context(
                    &prefix.program,
                    &target.program,
                    u32::try_from(begin)
                        .map_err(|_| invalid("evaluation context record offset overflow"))?,
                )
                .map_err(invalid)?,
            );
            if let Some(reservation) = reservation {
                let allocations = target
                    .allocations
                    .as_mut()
                    .and_then(Arc::get_mut)
                    .ok_or_else(|| {
                        invalid("evaluation context lost its original compile allocation owner")
                    })?;
                allocations._context = Some((
                    prefix.allocations.ok_or_else(|| {
                        invalid("evaluation prefix lost its original compile allocation owner")
                    })?,
                    reservation,
                ));
            }
        }
        for record in &mut target.query_records {
            *record = record
                .checked_add(
                    u32::try_from(begin)
                        .map_err(|_| invalid("evaluation context record offset overflow"))?,
                )
                .ok_or_else(|| invalid("evaluation query record offset overflow"))?;
        }
        target.observer_record_offset = u32::try_from(begin)
            .map_err(|_| invalid("evaluation observer record offset overflow"))?;
        source = Some(target);
        begin = end;
    }
    if scope_record.is_some() && segments != 2 {
        return Err(invalid(
            "evaluation scope requires its original donor and target segments",
        ));
    }
    if source.is_none() {
        return editable_source_segment(admission, 0, 0, scope_record, budget);
    }
    Ok(source)
}

fn editable_source_segment(
    admission: &SemanticAdmissionRecords,
    record_begin: usize,
    record_end: usize,
    scope_record: Option<usize>,
    budget: Option<&xlog_cuda::SemanticReplayMaterialBudget>,
) -> PyResult<Option<EditableTaskSource>> {
    let records = &admission.records[record_begin..record_end];
    let statement_count = records
        .iter()
        .take_while(|record| record.predicate == RelId(1))
        .count();
    let source_count = records[statement_count..]
        .iter()
        .take_while(|record| record.predicate == RelId(2))
        .count();
    let selector_begin = statement_count
        .checked_add(source_count)
        .ok_or_else(|| invalid("editable task record geometry overflow"))?;
    let query_count = records.len().saturating_sub(selector_begin);
    if statement_count == 0
        || statement_count > u32::MAX as usize - 3
        || !(2..=3).contains(&source_count)
        || query_count == 0
        || admission.records.len() > u32::MAX as usize
        || !admission.supports.is_empty()
    {
        return Err(invalid(
            "editable task admission has incomplete source records",
        ));
    }
    let overflow = || invalid("editable source allocation extent overflow");
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let symbol_count = records
        .iter()
        .flat_map(|record| &record.arguments)
        .filter(|argument| matches!(argument, SemanticArgument::Symbol(_)))
        .count();
    let vector_bytes = add(
        mul(
            statement_count,
            std::mem::size_of::<String>()
                + std::mem::size_of::<u8>()
                + std::mem::size_of::<(String, u8)>()
                + std::mem::size_of::<&str>(),
        )?,
        add(
            mul(
                source_count,
                std::mem::size_of::<String>() + std::mem::size_of::<u32>(),
            )?,
            add(
                mul(
                    query_count,
                    std::mem::size_of::<u32>() + std::mem::size_of::<&str>(),
                )?,
                add(
                    mul(symbol_count, std::mem::size_of::<u32>())?,
                    std::mem::size_of::<EditableTaskAllocations>()
                        + std::mem::size_of::<EditableProgramOwner>()
                        + 4 * std::mem::size_of::<usize>(),
                )?,
            )?,
        )?,
    )?;
    let vectors = budget
        .map(|budget| {
            budget
                .reserve_auxiliary(add(
                    vector_bytes,
                    usize::from(record_begin != 0) * std::mem::size_of::<u32>(),
                )? as u64)
                .map_err(xlog_err)
        })
        .transpose()?;
    let symbol_ids = budget.map(|_| {
        records
            .iter()
            .flat_map(|record| &record.arguments)
            .filter_map(|argument| match argument {
                SemanticArgument::Symbol(id) => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>()
    });
    let snapshot = match (budget, symbol_ids.as_ref()) {
        (Some(budget), Some(ids)) => Some(
            xlog_core::symbol::snapshot_checked_with_allocation_owner(
                ids,
                ids.len(),
                usize::MAX,
                &mut |count, utf8| {
                    let bytes = count
                        .checked_mul(std::mem::size_of::<(u32, Box<str>)>())
                        .and_then(|n| n.checked_mul(2))
                        .and_then(|n| utf8.checked_mul(2).and_then(|text| n.checked_add(text)))
                        .ok_or_else(|| {
                            xlog_core::symbol::SymbolSnapshotError::AllocationRejected {
                                detail: "editable symbol allocation extent overflow".into(),
                            }
                        })?;
                    let owner = budget.reserve_auxiliary(bytes as u64).map_err(|error| {
                        xlog_core::symbol::SymbolSnapshotError::AllocationRejected {
                            detail: error.to_string(),
                        }
                    })?;
                    Ok(Arc::new(owner) as Arc<dyn Send + Sync>)
                },
            )
            .map_err(|error| invalid(&error.to_string()))?,
        ),
        _ => None,
    };
    let symbol_text = |argument: &SemanticArgument| -> PyResult<&str> {
        let SemanticArgument::Symbol(id) = argument else {
            return Err(invalid("editable task source must be an admitted symbol"));
        };
        snapshot
            .as_ref()
            .and_then(|(snapshot, _)| {
                snapshot
                    .entries()
                    .iter()
                    .find(|(original, _)| original == id)
            })
            .map(|(_, text)| text.as_ref())
            .ok_or_else(|| invalid("editable task source symbol is unavailable"))
    };
    let text_owner = if let Some(budget) = budget {
        let mut text = 0usize;
        for record in &records[..selector_begin] {
            let argument = record
                .arguments
                .get(1)
                .ok_or_else(|| invalid("editable task source has invalid typed arguments"))?;
            text = add(text, symbol_text(argument)?.len())?;
        }
        let source_text = |index: usize| {
            records[statement_count + index]
                .arguments
                .get(1)
                .ok_or_else(|| invalid("editable task input has invalid typed arguments"))
                .and_then(symbol_text)
        };
        // The source carrier and its actual Session/actor projections can own
        // three copies of each original text. The selected initial source has
        // its real ordered selector multiplicity, not all-history bytes.
        let mut initial = add(add(source_text(0)?.len(), source_text(1)?.len())?, 2)?;
        for record in &records[selector_begin..] {
            let Some(SemanticArgument::U32(statement)) = record.arguments.get(1) else {
                return Err(invalid(
                    "editable task query selector has invalid typed arguments",
                ));
            };
            let statement = (*statement as usize)
                .checked_sub(record_begin)
                .filter(|&statement| statement < statement_count)
                .and_then(|statement| records.get(statement))
                .and_then(|record| record.arguments.get(1))
                .ok_or_else(|| {
                    invalid("editable task query selection differs from its admission")
                })?;
            initial = add(initial, add(symbol_text(statement)?.len(), 5)?)?;
        }
        Some(
            budget
                .reserve_auxiliary(mul(add(text, initial)?, 3)? as u64)
                .map_err(xlog_err)?,
        )
    } else {
        None
    };
    let symbol = |argument: &SemanticArgument| -> PyResult<String> {
        let SemanticArgument::Symbol(id) = argument else {
            return Err(invalid("editable task source must be an admitted symbol"));
        };
        if snapshot.is_some() {
            return symbol_text(argument).map(str::to_owned);
        }
        xlog_core::symbol::resolve_checked(*id)
            .ok_or_else(|| invalid("editable task source symbol is unavailable"))
    };
    let scope_index = if record_begin != 0 {
        let scope = scope_record
            .ok_or_else(|| invalid("evaluation context has no original typed scope"))?;
        Some(u32::try_from(scope).map_err(|_| invalid("evaluation scope record offset overflow"))?)
    } else {
        None
    };
    let qualifiers = (record_begin + statement_count..record_begin + selector_begin)
        .map(|index| index as u32)
        .chain(scope_index)
        .collect::<Vec<_>>();
    let mut statements = Vec::with_capacity(statement_count);
    let mut masks = Vec::with_capacity(statement_count);
    for (index, record) in records[..statement_count].iter().enumerate() {
        let [SemanticArgument::U32(ordinal), content, SemanticArgument::U32(mask)] =
            record.arguments.as_slice()
        else {
            return Err(invalid(
                "editable task statement has invalid typed arguments",
            ));
        };
        if record.predicate != RelId(1)
            || *ordinal != index as u32
            || record.qualifiers != qualifiers
            || !(1..=15).contains(mask)
        {
            return Err(invalid(
                "editable task statement order or source links differ",
            ));
        }
        statements.push(symbol(content)?);
        masks.push(*mask as u8);
    }
    let mut sources = Vec::with_capacity(source_count);
    for (index, record) in records[statement_count..selector_begin].iter().enumerate() {
        let [SemanticArgument::U32(kind), content] = record.arguments.as_slice() else {
            return Err(invalid("editable task input has invalid typed arguments"));
        };
        if record.predicate != RelId(2) || *kind != index as u32 || !record.qualifiers.is_empty() {
            return Err(invalid(
                "editable task input order differs from its admission",
            ));
        }
        sources.push(symbol(content)?);
    }
    let mut query_records = Vec::with_capacity(query_count);
    for (index, record) in records[selector_begin..].iter().enumerate() {
        let [SemanticArgument::U32(ordinal), SemanticArgument::U32(statement)] =
            record.arguments.as_slice()
        else {
            return Err(invalid(
                "editable task query selector has invalid typed arguments",
            ));
        };
        if record.predicate != RelId(3)
            || *ordinal != index as u32
            || (*statement as usize)
                .checked_sub(record_begin)
                .is_none_or(|statement| statement >= statement_count)
            || !record.qualifiers.is_empty()
        {
            return Err(invalid(
                "editable task query selection differs from its admission",
            ));
        }
        query_records.push((*statement as usize - record_begin) as u32);
    }
    let statement_texts = statements.iter().map(String::as_str).collect::<Vec<_>>();
    let (program, allocations) = match budget {
        Some(budget) => {
            let (program, native) = xlog_gpu::logic::compile_positive_task_with_allocation_owner(
                &sources[0],
                &sources[1],
                &statement_texts,
                &query_records,
                &mut |bytes| {
                    budget
                        .reserve_auxiliary(bytes)
                        .map(|owner| Arc::new(owner) as Arc<dyn Send + Sync>)
                        .map_err(|error| xlog_core::XlogError::Compilation(error.to_string()))
                },
            )
            .map_err(xlog_err)?;
            (
                program,
                Some(Arc::new(EditableTaskAllocations {
                    _vectors: vectors.expect("original source vector allocation owner"),
                    _text: text_owner.expect("original source text allocation owner"),
                    _program: native,
                    _context: None,
                })),
            )
        }
        None => (
            xlog_gpu::logic::compile_positive_task(
                &sources[0],
                &sources[1],
                &statement_texts,
                &query_records,
            )
            .map_err(xlog_err)?,
            None,
        ),
    };
    Ok(Some(EditableTaskSource {
        program: Arc::new(program),
        observer_source: sources.get(2).cloned(),
        initial_source: xlog_gpu::logic::positive_task_source(
            &sources[0],
            &sources[1],
            &query_records
                .iter()
                .map(|record| statement_texts[*record as usize])
                .collect::<Vec<_>>(),
        ),
        initial_theory: sources[0].clone(),
        input_facts: sources[1].clone(),
        statements: statements.into_iter().zip(masks).collect(),
        query_records,
        observer_record_offset: 0,
        allocations,
    }))
}

#[derive(Clone, PartialEq, Eq)]
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
#[derive(PartialEq, Eq)]
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
    acknowledgement_payload_capacity_bytes: usize,
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
        metadata.set_item(
            "acknowledgement_payload_capacity_bytes",
            self.acknowledgement_payload_capacity_bytes,
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

struct ColdTaskContent {
    observation: (
        xlog_cuda::SemanticTaskContentIdentity,
        Vec<xlog_cuda::SemanticTruth>,
    ),
    statement_records: Vec<u32>,
    _reservation: Option<xlog_cuda::SemanticReplayMaterialReservation>,
}

impl PySemanticTransitionSession {
    fn prepare_evaluation_model_context(
        &self,
        donor: &Self,
        donor_owner: &xlog_cuda::SemanticTransitionSession,
        target_owner: &mut xlog_cuda::SemanticTransitionSession,
        ground: &super::ColdValue,
        parent: &PySemanticTransitionFreshParent,
    ) -> PyResult<ColdTaskContent> {
        let donor_program = donor.editable_source()?.program.as_ref().cloned();
        let target_program = self
            .editable_source()?
            .program
            .as_ref()
            .cloned()
            .ok_or_else(|| invalid("evaluation context requires its original native program"))?;
        let (program, record_offset) = target_owner
            .prepare_evaluation_model_context(
                donor_owner,
                donor_program.as_ref().map(|owner| owner.program.as_ref()),
                &target_program.program,
                xlog_cuda::SemanticAdmissionLimits {
                    max_records: self.admission_limits.0,
                    max_terms: self.admission_limits.1,
                    max_references: self.admission_limits.2,
                    max_utf8_bytes: self.admission_limits.3,
                },
            )
            .map_err(xlog_err)?;
        if !target_owner
            .evaluation_fork_target_bound()
            .map_err(xlog_err)?
        {
            let environment = &parent
                .records
                .iter()
                .find(|record| record.role == 37)
                .ok_or_else(|| invalid("evaluation target omitted its original environment"))?
                .bytes;
            let checkpoint_lineage = &parent
                .records
                .iter()
                .find(|record| record.role == 40)
                .ok_or_else(|| {
                    invalid("evaluation target omitted its original checkpoint lineage")
                })?
                .bytes;
            let bytes = parent
                .terminal_tokens
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .and_then(|bytes| bytes.checked_add(parent.intent_effect.len()))
                .and_then(|bytes| bytes.checked_add(environment.len()))
                .and_then(|bytes| bytes.checked_add(checkpoint_lineage.len()))
                .and_then(|bytes| {
                    bytes
                        .checked_add(std::mem::size_of::<xlog_cuda::SemanticEvaluationForkTarget>())
                })
                .ok_or_else(|| invalid("evaluation target envelope capacity overflow"))?;
            let _transfer_reservation = self
                .material_sources
                .native_budget()?
                .reserve_auxiliary(bytes as u64)
                .map_err(xlog_err)?;
            target_owner
                .bind_evaluation_fork_target(xlog_cuda::SemanticEvaluationForkTarget {
                    prefix_capacity: parent.prefix_capacity,
                    feedback_capacity: parent.feedback_capacity,
                    max_position: MAX_POSITION,
                    pad_token: parent.pad_token,
                    terminal_tokens: parent.terminal_tokens.clone(),
                    final_intent_payload_bytes: parent.final_intent_payload_bytes,
                    nonterminal_intent: None,
                    intent_effect: parent.intent_effect.clone(),
                    intent_entry_capacity: parent.intent_entry_capacity,
                    intent_payload_capacity_bytes: parent.intent_payload_capacity_bytes,
                    acknowledgement_payload_capacity_bytes: parent
                        .acknowledgement_payload_capacity_bytes,
                    fuel: parent.fuel,
                    environment: environment.clone(),
                    checkpoint_lineage: checkpoint_lineage.clone(),
                    authority_decisions_capacity_bytes: parent.authority_decisions_capacity_bytes,
                    rng: xlog_cuda::SemanticRngBinding {
                        model_generation: parent.rng.0,
                        stream_serial: parent.rng.1,
                        family_id: parent.rng.2,
                        proposal: parent.rng.3,
                    },
                })
                .map_err(xlog_err)?;
        }
        let parsed = editable_source_from_admission_bound(
            target_owner.cold_admission_records().map_err(xlog_err)?,
            &self.material_sources,
        )?
        .ok_or_else(|| invalid("evaluation context lost its canonical segmented source"))?;
        if parsed.program.as_ref() != program.as_ref() {
            return Err(invalid(
                "evaluation context source changed its original native program",
            ));
        }
        let bytes = parsed
            .query_records
            .len()
            .checked_mul(
                std::mem::size_of::<u32>()
                    + std::mem::size_of::<xlog_cuda::SemanticTruth>()
                    + std::mem::size_of::<usize>(),
            )
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<ColdTaskContent>()))
            .ok_or_else(|| invalid("evaluation content cache capacity overflow"))?;
        let budget = self.material_sources.native_budget()?;
        let reservation = budget.reserve_auxiliary(bytes as u64).map_err(xlog_err)?;
        let observation = match target_owner.observed_cold_task_content() {
            Ok(content) => content,
            Err(xlog_cuda::SemanticTransitionError::NotBound) => {
                let native_ground = super::task_ground_value(ground)?;
                let observer = xlog_gpu::logic::SemanticLogicTaskProgram::compile_with_allocation_owner(
                    parsed.observer_source.clone().unwrap_or_else(|| parsed.initial_source.clone()),
                    if parsed.observer_source.is_some() {
                        parsed.query_records.iter().map(|record| {
                            record.checked_sub(record_offset).map(|record| record as usize)
                                .ok_or_else(|| invalid("evaluation observer selection precedes its original segment"))
                        }).collect::<PyResult<Vec<_>>>()?
                    } else {
                        (0..parsed.query_records.len()).collect()
                    },
                    &mut |bytes| {
                        budget.reserve_auxiliary(bytes)
                            .map(|owner| Arc::new(owner) as Arc<dyn Send + Sync>)
                            .map_err(|error| xlog_core::XlogError::Compilation(error.to_string()))
                    },
                ).map_err(xlog_err)?.with_editable_program(Arc::clone(&program));
                target_owner
                    .observe_cold_task_content(
                        &parsed.query_records,
                        &[],
                        &observer,
                        &native_ground,
                    )
                    .map_err(xlog_err)?
            }
            Err(error) => return Err(xlog_err(error)),
        };
        self.install_editable_source(&parsed, target_owner.evaluation_rule_library())?;
        Ok(ColdTaskContent {
            observation,
            statement_records: parsed.query_records,
            _reservation: Some(reservation),
        })
    }
}

/// Canonical cold producer for one finite ordered semantic task.
///
/// Construction parses the complete observer and admits the exact initial
/// theory, input facts and ordered task statements, allocates one CUDA Session,
/// creates its sole Controller and retains the typed fresh-parent input set.
#[pyclass(name = "SemanticTransitionColdTask", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticTransitionColdTask {
    session: Py<PySemanticTransitionSession>,
    controller: Py<PySemanticTransitionController>,
    parent: Py<PySemanticTransitionFreshParent>,
    content: Mutex<ColdTaskContent>,
    task_ground: super::ColdValue,
    pending_model_contexts: Mutex<Option<Py<PyTuple>>>,
}

impl PySemanticTransitionColdTask {
    pub(super) fn canary_source(
        &self,
        py: Python<'_>,
    ) -> PyResult<Arc<xlog_cuda::SemanticTrainingCanarySource>> {
        let session = self.session.borrow(py);
        if let Some(source) = session.detached_training_canary_source()? {
            return Ok(source);
        }
        let source = session
            .owner()?
            .training_canary_source()
            .map_err(xlog_err)?;
        Ok(source)
    }

    pub(super) fn canary_allocation_domain(
        &self,
        py: Python<'_>,
    ) -> PyResult<super::CheckpointAllocationDomain> {
        let session = self.session.borrow(py);
        if let Some(allocation) = session.detached_allocation_domain()? {
            return Ok(allocation);
        }
        session
            .owner()?
            .training_canary_source()
            .map_err(xlog_err)?;
        let (provider, domain) = session
            .owner()?
            .checkpoint_allocation_domain()
            .map_err(xlog_err)?;
        Ok(super::CheckpointAllocationDomain {
            provider,
            domain,
            #[cfg(feature = "semantic-policy")]
            cold_work: None,
        })
    }
}

#[pymethods]
impl PySemanticTransitionColdTask {
    #[new]
    #[pyo3(signature = (*, initial_theory, input_facts, observer_program, statements, query_records, task_ground, training_canary_source, capacities, admission_limits, device_ordinal, memory_bytes, provenance_capacity_records, prefix_capacity, feedback_capacity, pad_token, terminal_tokens, final_intent_payload_bytes, intent_effect, intent_entry_capacity, intent_payload_capacity_bytes, acknowledgement_payload_capacity_bytes, authority_decisions_capacity_bytes, generations, training_cursor, training_rng, fuel, rng, material_source=None, private_replay_child=None, prepared_actor_refresh=None, actor_refresh_bank=None, rule_library=None))]
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
        query_records: &Bound<'_, PyAny>,
        task_ground: &Bound<'_, PyAny>,
        training_canary_source: &Bound<'_, PyAny>,
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
        acknowledgement_payload_capacity_bytes: usize,
        authority_decisions_capacity_bytes: usize,
        generations: (u64, u64, u64, u64),
        training_cursor: u64,
        training_rng: [u64; 4],
        fuel: u64,
        rng: (u64, u8, u32),
        material_source: Option<&Bound<'_, PyAny>>,
        private_replay_child: Option<&Bound<'_, PyAny>>,
        prepared_actor_refresh: Option<&Bound<'_, PyAny>>,
        actor_refresh_bank: Option<usize>,
        rule_library: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if rule_library.is_some()
            && (private_replay_child.is_some() || prepared_actor_refresh.is_some())
        {
            return Err(invalid(
                "rule library must be declared on its original cold task",
            ));
        }
        if private_replay_child.is_some() && prepared_actor_refresh.is_some()
            || prepared_actor_refresh.is_some() != actor_refresh_bank.is_some()
            || actor_refresh_bank.is_some_and(|bank| bank > 1)
        {
            return Err(invalid("actor construction requires one typed owner and original bank, not a historical replay child"));
        }
        #[cfg(feature = "semantic-policy")]
        let actor_custody = prepared_actor_refresh
            .map(|owner| {
                owner
                    .extract::<PyRef<'_, super::actor_refresh::PySemanticPreparedActorRefresh>>()
                    .map(|owner| Arc::clone(&owner.inner))
            })
            .transpose()?;
        #[cfg(feature = "semantic-policy")]
        let replay_custody = private_replay_child
            .map(|child| {
                child
                    .extract::<PyRef<'_, super::learning_phase::PySemanticPrivateReplayChild>>()
                    .map(|child| Arc::clone(&child.inner))
            })
            .transpose()?;
        let explicit_material_sources = material_source
            .map(material_owner::original_registry)
            .transpose()?;
        #[cfg(feature = "semantic-policy")]
        let inherited_material_sources = if let Some(child) = &replay_custody {
            Some(Arc::clone(
                &child.source_checkpoint(py)?.seed.material_sources,
            ))
        } else if let Some(owner) = &actor_custody {
            Some(Arc::clone(&owner.source_checkpoint().seed.material_sources))
        } else {
            None
        };
        #[cfg(not(feature = "semantic-policy"))]
        let inherited_material_sources: Option<
            Arc<super::material_source::MaterialSourceRegistry>,
        > = None;
        let material_sources =
            match (inherited_material_sources, explicit_material_sources) {
                (Some(original), Some(supplied)) => {
                    if !Arc::ptr_eq(&original, &supplied) {
                        return Err(invalid(
                            "dependent task requires its original process material source",
                        ));
                    }
                    original
                }
                (Some(original), None) | (None, Some(original)) => original,
                (None, None) => return Err(invalid(
                    "fresh task requires its original native material source before construction",
                )),
            };
        #[cfg(feature = "semantic-policy")]
        let allocation = replay_custody
            .as_ref()
            .map(|child| child.allocation(py))
            .transpose()?;
        #[cfg(feature = "semantic-policy")]
        let allocation = match &actor_custody {
            Some(owner) => {
                Some(owner.allocation(py, actor_refresh_bank.expect("checked actor bank"))?)
            }
            None => allocation,
        };
        #[cfg(not(feature = "semantic-policy"))]
        let allocation = {
            if private_replay_child.is_some() || prepared_actor_refresh.is_some() {
                return Err(invalid(
                    "private replay construction requires semantic-policy",
                ));
            }
            None
        };
        let (canary_source, canary_owner, _) =
            super::read_training_canary_source(py, Some(training_canary_source))?;
        let source_allocation = canary_owner
            .as_ref()
            .map(|source| source.borrow(py).canary_allocation_domain(py))
            .transpose()?;
        let allocation = match (allocation, source_allocation) {
            (Some(replay), Some(source)) => {
                if !Arc::ptr_eq(replay.provider.memory(), source.provider.memory())
                    || replay.domain.stream_id() != source.domain.stream_id()
                {
                    return Err(invalid("dependent task must retain its original canary allocation owner and stream"));
                }
                Some(replay)
            }
            (Some(allocation), None) | (None, Some(allocation)) => Some(allocation),
            (None, None) => None,
        };
        let initial_theory = exact_text(initial_theory, "initial_theory")?;
        let input_facts = exact_text(input_facts, "input_facts")?;
        let task_ground = super::read_task_ground(task_ground, &mut (16 * 1024 * 1024))?;
        let native_ground = super::task_ground_value(&task_ground)?;
        let observer_program = match &native_ground {
            xlog_cuda::SemanticTaskGround::Logical => {
                Some(exact_text(observer_program, "observer_program")?)
            }
            xlog_cuda::SemanticTaskGround::Coding { .. } if observer_program.is_none() => None,
            xlog_cuda::SemanticTaskGround::Coding { .. } => {
                return Err(invalid("coding task has no independent observer reference"))
            }
        };
        let statements = exact_statements(statements)?;
        let statement_records = exact_unsigned_tuple(query_records, "query_records")?
            .into_iter()
            .map(|record| {
                u32::try_from(record)
                    .map_err(|_| invalid("query record exceeds its native index domain"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        if statement_records.is_empty()
            || statement_records.len() > u32::MAX as usize
            || statement_records
                .iter()
                .any(|record| *record as usize >= statements.len())
        {
            return Err(invalid(
                "query_records requires a finite nonempty selection of original Statements",
            ));
        }
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
            acknowledgement_payload_capacity_bytes,
            authority_decisions_capacity_bytes,
            generations.0,
            rng,
        )?;
        let statement_texts = statement_records
            .iter()
            .map(|record| statements[*record as usize].0.as_str())
            .collect::<Vec<_>>();
        let initial_source =
            xlog_gpu::logic::positive_task_source(&initial_theory, &input_facts, &statement_texts);
        let mut program = xlog_gpu::logic::SemanticLogicTaskProgram::compile(
            observer_program.clone().unwrap_or(initial_source),
            if observer_program.is_some() {
                statement_records
                    .iter()
                    .map(|record| *record as usize)
                    .collect()
            } else {
                (0..statement_records.len()).collect()
            },
        )
        .map_err(xlog_err)?;

        let task_digest = task_digest(
            &initial_theory,
            &input_facts,
            observer_program.as_deref(),
            &statements,
            &statement_records,
            &task_ground,
        );
        let admission = task_admission(
            &initial_theory,
            &input_facts,
            observer_program.as_deref(),
            &statements,
            &statement_records,
        )?;
        let role_counts = initial_role_counts(statement_records.len());
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
                acknowledgement_payload_capacity_bytes,
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
        let import_owners = || {
            (
                Arc::new(Mutex::new(ProposalExpense::default())),
                Arc::new(Mutex::new(CheckpointSources::default())),
            )
        };
        #[cfg(feature = "semantic-policy")]
        let (proposal_expense, checkpoint_sources) = replay_custody
            .as_ref()
            .map(|child| child.shared_import_owners(py))
            .transpose()?
            .unwrap_or_else(import_owners);
        #[cfg(feature = "semantic-policy")]
        let (proposal_expense, checkpoint_sources) = match &actor_custody {
            Some(owner) => owner.shared_import_owners(py)?,
            None => (proposal_expense, checkpoint_sources),
        };
        #[cfg(not(feature = "semantic-policy"))]
        let (proposal_expense, checkpoint_sources) = import_owners();
        let canary = canary_source.zip(canary_owner);
        #[cfg(feature = "semantic-policy")]
        let actor_session = actor_custody
            .as_ref()
            .map(|owner| {
                owner.construct_session(
                    py,
                    actor_refresh_bank.expect("checked actor bank"),
                    admission.clone(),
                    &parent,
                    &task_ground,
                    capacities,
                    admission_limits,
                    device_ordinal,
                    memory_bytes,
                    Arc::clone(&proposal_expense),
                    Arc::clone(&checkpoint_sources),
                    canary.clone(),
                )
            })
            .transpose()?;
        #[cfg(not(feature = "semantic-policy"))]
        let actor_session: Option<(
            Py<PySemanticTransitionSession>,
            Py<PySemanticTransitionFreshParent>,
        )> = None;
        let (session, parent) = match actor_session {
            Some(original) => original,
            None => {
                let native_session = PySemanticTransitionSession::from_admission(
                    admission,
                    capacities,
                    admission_limits,
                    device_ordinal,
                    memory_bytes,
                    proposal_expense,
                    checkpoint_sources,
                    material_sources,
                    allocation,
                    canary,
                    None,
                )
                .map_err(|error| {
                    #[cfg(feature = "semantic-policy")]
                    if let Some(child) = &replay_custody {
                        child.fail();
                    }
                    error
                })?;
                (Py::new(py, native_session)?, parent)
            }
        };
        #[cfg(feature = "semantic-policy")]
        if let Some(child) = &replay_custody {
            child.retain_session(py, &session)?;
        }
        #[cfg(feature = "semantic-policy")]
        let cached_content = actor_custody
            .as_ref()
            .map(|owner| owner.constructed_content(actor_refresh_bank.expect("checked actor bank")))
            .transpose()?
            .flatten();
        #[cfg(not(feature = "semantic-policy"))]
        let cached_content = None;
        if let Some(library) = rule_library {
            super::evaluation_rule_library::bind_cold_library(
                &session.borrow(py),
                library,
                &initial_theory,
                &input_facts,
                &statements,
                &statement_records,
            )?;
        }
        let content = match cached_content {
            Some(content) => content,
            None => {
                let content = (|| {
                    let native_session = session.borrow(py);
                    if let Some(editable) = native_session.editable_source()?.program.as_ref() {
                        program = program.with_editable_program(Arc::clone(&editable.program));
                    }
                    #[cfg(feature = "semantic-policy")]
                    if let Some(owner) = &actor_custody {
                        return owner.observe_constructed_content(
                            py,
                            actor_refresh_bank.expect("checked actor bank"),
                            Arc::new(program),
                            &statement_records,
                            &native_ground,
                        );
                    }
                    let result = native_session
                        .owner()?
                        .observe_cold_task_content(
                            &statement_records,
                            &[],
                            &program,
                            &native_ground,
                        )
                        .map_err(xlog_err);
                    result
                })()
                .map_err(|error| {
                    #[cfg(feature = "semantic-policy")]
                    if let Some(child) = &replay_custody {
                        child.fail();
                    }
                    error
                })?;
                content
            }
        };
        #[cfg(feature = "semantic-policy")]
        if let Some(child) = &replay_custody {
            child.observed()?;
        }
        #[cfg(feature = "semantic-policy")]
        let cached_controller = actor_custody
            .as_ref()
            .map(|owner| {
                owner.constructed_controller(py, actor_refresh_bank.expect("checked actor bank"))
            })
            .transpose()?
            .flatten();
        #[cfg(not(feature = "semantic-policy"))]
        let cached_controller = None;
        let controller = match cached_controller {
            Some(controller) => controller,
            None => Py::new(
                py,
                PySemanticTransitionController::new(py, session.clone_ref(py))?,
            )?,
        };
        #[cfg(feature = "semantic-policy")]
        if let Some(child) = &replay_custody {
            child.retain_controller(&controller.borrow(py).identity)?;
        }
        #[cfg(feature = "semantic-policy")]
        if let Some(child) = &actor_custody {
            child.retain_controller(
                py,
                actor_refresh_bank.expect("checked actor bank"),
                &controller,
            )?;
        }
        Ok(Self {
            session,
            controller,
            parent,
            content: Mutex::new(ColdTaskContent {
                observation: content,
                statement_records,
                _reservation: None,
            }),
            task_ground,
            pending_model_contexts: Mutex::new(None),
        })
    }

    #[getter]
    fn session(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionSession>> {
        self.session.borrow(py).owner()?;
        Ok(self.session.clone_ref(py))
    }

    #[getter]
    fn controller(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionController>> {
        self.session.borrow(py).owner()?;
        Ok(self.controller.clone_ref(py))
    }

    #[getter]
    fn parent(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionFreshParent>> {
        self.session.borrow(py).owner()?;
        Ok(self.parent.clone_ref(py))
    }

    /// Freeze the exact ordered tuple of genuine cold tasks before importing
    /// tasks or loading the model. Unknown retains this same tuple for retry.
    /// The common model identity is then available through Session.policy_layout;
    /// TaskUse.feedback_schema retains the shared shape and each own payload.
    #[staticmethod]
    fn freeze_model_contexts(py: Python<'_>, contexts: &Bound<'_, PyAny>) -> PyResult<()> {
        if !contexts.is_exact_instance_of::<PyTuple>() {
            return Err(invalid("model contexts require an exact builtin tuple"));
        }
        let contexts = contexts.cast::<PyTuple>()?;
        if contexts.len() < 2 {
            return Err(invalid(
                "model contexts require ordered source, target and optional evaluation cold tasks",
            ));
        }
        let first = contexts.get_item(0)?;
        if !first.is_exact_instance_of::<Self>() {
            return Err(invalid(
                "model contexts require exact original ColdTask owners",
            ));
        }
        let first_cold = first.extract::<PyRef<'_, Self>>()?;
        let first_session = first_cold.session.borrow(py);
        first_session.require_creator()?;
        let budget = first_session.material_sources.native_budget()?;
        let roster_bytes = contexts
            .len()
            .checked_mul(
                std::mem::size_of::<PyRef<'_, Self>>()
                    + std::mem::size_of::<PyRef<'_, PySemanticTransitionSession>>()
                    + std::mem::size_of::<super::SemanticSessionGuard<'_>>()
                    + 3 * std::mem::size_of::<&xlog_cuda::SemanticTransitionSession>(),
            )
            .ok_or_else(|| invalid("model context roster capacity overflow"))?;
        let _roster_reservation = budget
            .reserve_auxiliary(roster_bytes as u64)
            .map_err(xlog_err)?;
        drop(first_session);
        drop(first_cold);
        let mut cold_tasks = Vec::new();
        cold_tasks
            .try_reserve_exact(contexts.len())
            .map_err(|_| invalid("model context owner roster allocation failed"))?;
        let mut sessions = Vec::new();
        sessions
            .try_reserve_exact(contexts.len())
            .map_err(|_| invalid("model context session roster allocation failed"))?;
        for context in contexts.iter() {
            if !context.is_exact_instance_of::<Self>() {
                return Err(invalid(
                    "model contexts require exact original ColdTask owners",
                ));
            }
            cold_tasks.push(context.extract::<PyRef<'_, Self>>()?);
        }
        for (member, cold) in cold_tasks.iter().enumerate() {
            let session = cold.session.borrow(py);
            session.require_creator()?;
            if sessions
                .iter()
                .any(|prior: &PyRef<'_, PySemanticTransitionSession>| {
                    Arc::ptr_eq(&prior.inner, &session.inner)
                })
            {
                return Err(invalid(
                    "model contexts require distinct original cold owners",
                ));
            }
            let pending = cold
                .pending_model_contexts
                .try_lock()
                .map_err(|_| invalid("model context tuple custody is already in use"))?;
            if pending
                .as_ref()
                .is_some_and(|original| !original.bind(py).is(contexts))
            {
                return Err(invalid(
                    "pending model contexts require the same original tuple object",
                ));
            }
            let original = cold_tasks[0].parent.borrow(py);
            let parent = cold.parent.borrow(py);
            if member != 0 {
                let first = &sessions[0];
                if session.device_ordinal != first.device_ordinal
                    || session.memory_bytes != first.memory_bytes
                    || !Arc::ptr_eq(&session.material_sources, &first.material_sources)
                    || parent.prefix_capacity != original.prefix_capacity
                    || parent.feedback_capacity != original.feedback_capacity
                    || parent.pad_token != original.pad_token
                    || parent.terminal_tokens != original.terminal_tokens
                {
                    return Err(invalid(
                        "model contexts require one original material domain, device, memory budget, model input geometry and token semantics",
                    ));
                }
            }
            sessions.push(session);
        }
        // Construct the recoverable Python custody before the first native
        // effect, so a late error cannot lose any original cold owner.
        let pending = super::SemanticModelContextFreezePending::new_err(
            "retain contexts and repeat SemanticTransitionColdTask.freeze_model_contexts(contexts) with that same tuple; do not import, replace, reorder, append, drop or abort pending owners",
        );
        pending.value(py).setattr("contexts", contexts)?;
        let mut owners = Vec::new();
        owners
            .try_reserve_exact(sessions.len())
            .map_err(|_| invalid("model context native owner roster allocation failed"))?;
        for session in &sessions {
            owners.push(session.owner()?);
        }
        for session in sessions.iter().skip(2) {
            session.retain_evaluation_source_registry(Arc::clone(&sessions[1].material_sources))?;
        }
        let mut native_contexts = Vec::new();
        native_contexts
            .try_reserve_exact(owners.len())
            .map_err(|_| invalid("model context native reference roster allocation failed"))?;
        let mut preseal_contexts = Vec::new();
        preseal_contexts
            .try_reserve_exact(owners.len())
            .map_err(|_| invalid("model context preseal reference roster allocation failed"))?;
        let mut original_contexts = Vec::new();
        original_contexts
            .try_reserve_exact(owners.len())
            .map_err(|_| invalid("model context retry reference roster allocation failed"))?;
        for cold in &cold_tasks {
            *cold
                .pending_model_contexts
                .try_lock()
                .map_err(|_| invalid("model context tuple custody is already in use"))? =
                Some(contexts.clone().unbind());
        }
        let result = (|| -> PyResult<()> {
            for owner in &mut owners {
                preseal_contexts.push(&mut **owner);
            }
            let projected =
                xlog_cuda::SemanticTransitionSession::begin_model_contexts(&mut preseal_contexts)
                    .map_err(xlog_err)?;
            drop(preseal_contexts);
            if !projected {
                let (donor, targets) = owners[1..].split_first_mut().expect("target donor checked");
                for (index, target) in targets.iter_mut().enumerate() {
                    let member = index + 2;
                    let content = sessions[member].prepare_evaluation_model_context(
                        &sessions[1],
                        &*donor,
                        &mut *target,
                        &cold_tasks[member].task_ground,
                        &cold_tasks[member].parent.borrow(py),
                    )?;
                    *cold_tasks[member]
                        .content
                        .try_lock()
                        .map_err(|_| invalid("evaluation cold content is already in use"))? =
                        content;
                }
            }
            for owner in &mut owners {
                native_contexts.push(&mut **owner);
            }
            xlog_cuda::SemanticTransitionSession::freeze_model_contexts(&mut native_contexts)
                .map_err(xlog_err)?;
            drop(native_contexts);
            Ok(())
        })();
        for owner in &owners {
            original_contexts.push(&**owner);
        }
        let is_pending =
            xlog_cuda::SemanticTransitionSession::model_contexts_pending(&original_contexts);
        if !is_pending {
            for cold in &cold_tasks {
                cold.pending_model_contexts
                    .try_lock()
                    .map_err(|_| invalid("model context tuple custody is already in use"))?
                    .take();
            }
        }
        if let Err(original) = result {
            if is_pending {
                pending.set_cause(py, Some(original));
                return Err(pending);
            }
            return Err(original);
        }
        Ok(())
    }

    /// Terminally release this observed-only fresh CUDA Session before a
    /// restored Session is allocated. Immutable task content and statement
    /// ordinals remain readable; saved Session/controller aliases cannot run.
    /// An unfinished terminal join retains this same ColdTask for another close;
    /// fatal CUDA failure requires terminating the isolated task process.
    fn close(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let cold = slf.borrow();
        if cold
            .pending_model_contexts
            .try_lock()
            .map_err(|_| invalid("model context tuple custody is already in use"))?
            .is_some()
        {
            return Err(invalid("pending model contexts must retain every original cold owner until the same tuple resolves"));
        }
        let session = cold.session.borrow(py);
        session.release_observed_cold(py, slf)
    }

    /// Return content observed by the native cold task before final import.
    /// Final import must reproduce these identities and objective truths.
    #[getter]
    fn task_content(&self, py: Python<'_>) -> PyResult<super::TaskContentRead> {
        self.session.borrow(py).require_creator()?;
        let content = self
            .content
            .try_lock()
            .map_err(|_| invalid("cold task content is already in use"))?;
        super::task_content_read(py, content.observation.clone())
    }

    #[getter]
    fn statement_records(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let content = self
            .content
            .try_lock()
            .map_err(|_| invalid("cold task content is already in use"))?;
        Ok(PyTuple::new(py, content.statement_records.iter().copied())?.unbind())
    }

    #[getter]
    fn task_ground(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        self.task_ground.python_value(py)
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

fn exact_statements(value: &Bound<'_, PyAny>) -> PyResult<Vec<(String, u8)>> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid("statements requires an exact builtin tuple"));
    }
    let value = value.cast::<PyTuple>()?;
    if value.is_empty() || value.len() > u32::MAX as usize - 3 {
        return Err(invalid(
            "cold semantic task requires a finite nonempty ordered statement roster",
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
        .collect()
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
    acknowledgement_payload_capacity_bytes: usize,
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
        || acknowledgement_payload_capacity_bytes == 0
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
    observer_program: Option<&str>,
    statements: &[(String, u8)],
    query_records: &[u32],
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
    let query_schema = schema(
        vec![
            ("ordinal", ScalarType::U32, "task-query-ordinal"),
            ("statement_record", ScalarType::U32, "task-statement-record"),
        ],
        vec![0],
    )?;
    let mut sources = vec![initial_theory, input_facts];
    sources.extend(observer_program);
    if statements
        .len()
        .checked_add(sources.len())
        .and_then(|count| count.checked_add(query_records.len()))
        .is_none_or(|count| count > u32::MAX as usize)
    {
        return Err(invalid(
            "task admission record geometry exceeds its original index domain",
        ));
    }
    let qualifiers = (statements.len()..statements.len() + sources.len())
        .map(|index| index as u32)
        .collect::<Vec<_>>();
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
            qualifiers: qualifiers.clone(),
        })
        .collect::<Vec<_>>();
    records.extend(
        sources
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
    records.extend(
        query_records
            .iter()
            .enumerate()
            .map(|(ordinal, statement)| SemanticTypedRecord {
                predicate: RelId(3),
                arguments: vec![
                    SemanticArgument::U32(ordinal as u32),
                    SemanticArgument::U32(*statement),
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
            SemanticPredicateRecord {
                predicate: RelId(3),
                role: SemanticRecordRole::Qualifier,
                schema: query_schema,
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
    observer_program: Option<&str>,
    statements: &[(String, u8)],
    query_records: &[u32],
    task_ground: &super::ColdValue,
) -> [u8; 32] {
    let mut bytes = b"xlog.semantic.cold-task.v2\0".to_vec();
    for value in [initial_theory, input_facts] {
        append_sized(&mut bytes, value.as_bytes());
    }
    bytes.push(u8::from(observer_program.is_some()));
    if let Some(observer) = observer_program {
        append_sized(&mut bytes, observer.as_bytes());
    }
    append_sized(&mut bytes, &super::checkpoint_cold_value_bytes(task_ground));
    bytes.extend((statements.len() as u64).to_le_bytes());
    for (statement, mask) in statements {
        append_sized(&mut bytes, statement.as_bytes());
        bytes.push(*mask);
    }
    bytes.extend((query_records.len() as u64).to_le_bytes());
    for record in query_records {
        bytes.extend(record.to_le_bytes());
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

fn initial_role_counts(query_count: usize) -> [u64; 55] {
    let mut counts = [1u64; 55];
    counts[15] = query_count as u64;
    for role in [
        4u64, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 29, 43, 44, 45, 46,
        51, 52, 53, 54,
    ] {
        counts[role as usize - 1] = 0;
    }
    counts
}

#[cfg(test)]
mod editable_source_tests {
    use super::{editable_source_from_admission, task_admission};

    #[test]
    fn fresh_and_restored_admissions_recover_the_same_canonical_program() {
        let theory = "pred p(u32, u32).\npred q(u32, u32).\npred t(u32, u32).\n";
        let facts = "p(1, 2).\nq(2, 3).\n";
        let observer = "pred t(u32, u32).\nt(1, 3).\n?- t(1, 3).\n?- t(1, 4).\n?- t(4, 3).\n";
        let statements = [
            ("t(1, 3)".to_owned(), 15),
            ("t(1, 4)".to_owned(), 15),
            ("t(4, 3)".to_owned(), 15),
        ];
        let admission = task_admission(theory, facts, observer, &statements).unwrap();
        let source = editable_source_from_admission(&admission).unwrap().unwrap();
        assert_eq!(source.observer_source, observer);
        assert_eq!(source.program.predicate_count, 3);
        assert_eq!(source.program.initial_facts.len(), 2);
        assert_eq!(source.program.queries[0].first, 1);

        let mut invalid = admission;
        invalid.records[0].qualifiers.clear();
        assert!(editable_source_from_admission(&invalid).is_err());
    }
}
