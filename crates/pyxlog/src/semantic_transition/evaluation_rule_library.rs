//! Compile a fixed rule library through the original positive task compiler.

use super::*;

pub(super) struct LibraryDeclaration {
    pub(super) material: xlog_cuda::SemanticReplayLoadedMaterial,
    pub(super) digest: Identity256,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

/// The exact ordered bytes are shared by all interventions. Each authored rule
/// has one newline frame; neither a digest alone nor a detached rule list binds it.
pub(super) fn declaration(
    value: &Bound<'_, PyAny>,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<LibraryDeclaration> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "rule library requires its original records and digest tuple",
        ));
    }
    let fields = value.cast::<PyTuple>()?;
    if fields.len() != 2 {
        return Err(invalid("rule library requires exactly records and digest"));
    }
    declaration_parts(&fields.get_item(0)?, &fields.get_item(1)?, budget)
}

fn declaration_parts(
    records: &Bound<'_, PyAny>,
    digest: &Bound<'_, PyAny>,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<LibraryDeclaration> {
    if !records.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "rule library records require an exact immutable tuple",
        ));
    }
    let records = records.cast::<PyTuple>()?;
    if records.len() != 32 {
        return Err(invalid(
            "rule library requires its complete ordered thirty-two rules",
        ));
    }
    if !digest.is_exact_instance_of::<PyString>() {
        return Err(invalid("rule library digest requires exact text"));
    }
    let metadata = budget
        .reserve_auxiliary(
            (std::mem::size_of::<LibraryDeclaration>() + 2 * std::mem::size_of::<usize>() + 65)
                as u64,
        )
        .map_err(xlog_err)?;
    let digest = replay_digest_identity(digest.cast::<PyString>()?.to_str()?)?;
    let mut extent = 0usize;
    for record in records.iter() {
        if !record.is_exact_instance_of::<PyBytes>() {
            return Err(invalid(
                "rule library records require exact immutable bytes",
            ));
        }
        let bytes = record.cast::<PyBytes>()?.as_bytes();
        let text =
            std::str::from_utf8(bytes).map_err(|_| invalid("rule library record is not UTF-8"))?;
        if text.is_empty() || text.contains(['\n', '\r']) {
            return Err(invalid(
                "rule library record must contain one unframed rule",
            ));
        }
        extent = extent
            .checked_add(bytes.len())
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("rule library declaration extent overflow"))?;
    }
    let reservation = budget
        .reserve(
            u64::try_from(extent).map_err(|_| invalid("rule library declaration exceeds u64"))?,
        )
        .map_err(xlog_err)?;
    let owner_metadata = budget.reserve_owned_payload_metadata().map_err(xlog_err)?;
    let mut bytes = Vec::with_capacity(extent);
    for record in records.iter() {
        bytes.extend_from_slice(record.cast::<PyBytes>()?.as_bytes());
        bytes.push(b'\n');
    }
    if <[u8; 32]>::from(Sha256::digest(&bytes)) != *digest.as_bytes() {
        return Err(invalid(
            "rule library bytes differ from their original ordered digest",
        ));
    }
    let material = reservation
        .retain_owned_payload_with_metadata(bytes, owner_metadata)
        .map_err(xlog_err)?;
    Ok(LibraryDeclaration {
        material,
        digest,
        _metadata: metadata,
    })
}

pub(super) fn intervention(
    value: &Bound<'_, PyAny>,
    library: &xlog_cuda::SemanticEvaluationRuleLibrary,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<xlog_cuda::SemanticEvaluationIntervention> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "evaluation intervention requires its mode, records and digest",
        ));
    }
    let fields = value.cast::<PyTuple>()?;
    if fields.len() != 3 {
        return Err(invalid(
            "evaluation intervention requires exactly three fields",
        ));
    }
    let mode = fields.get_item(0)?;
    if !mode.is_exact_instance_of::<PyString>() {
        return Err(invalid("evaluation intervention mode requires exact text"));
    }
    let mode = match mode.cast::<PyString>()?.to_str()? {
        "no_edit" => xlog_cuda::SemanticEvaluationIntervention::NoEdit,
        "fixed_rule_activation" => xlog_cuda::SemanticEvaluationIntervention::FixedRuleActivation,
        "structural" => xlog_cuda::SemanticEvaluationIntervention::Structural,
        _ => return Err(invalid("unknown evaluation intervention mode")),
    };
    let declaration = declaration_parts(&fields.get_item(1)?, &fields.get_item(2)?, budget)?;
    if declaration.digest != library.digest()
        || declaration.material.as_bytes() != library.declaration()
    {
        return Err(invalid(
            "evaluation intervention changed its original frozen rule library",
        ));
    }
    Ok(mode)
}

pub(super) fn intervention_material(
    py: Python<'_>,
    mode: xlog_cuda::SemanticEvaluationIntervention,
    digest: Identity256,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<Py<material_bytes::PySemanticMaterialBytes>> {
    let payload = budget.reserve(33).map_err(xlog_err)?;
    let metadata = budget.reserve_owned_payload_metadata().map_err(xlog_err)?;
    let mut bytes = Vec::with_capacity(33);
    bytes.push(match mode {
        xlog_cuda::SemanticEvaluationIntervention::NoEdit => 0,
        xlog_cuda::SemanticEvaluationIntervention::FixedRuleActivation => 1,
        xlog_cuda::SemanticEvaluationIntervention::Structural => 3,
    });
    bytes.extend_from_slice(digest.as_bytes());
    let material = payload
        .retain_owned_payload_with_metadata(bytes, metadata)
        .map_err(xlog_err)?;
    Py::new(
        py,
        material_bytes::PySemanticMaterialBytes::from_native(py, material, budget)?,
    )
}

pub(super) fn bind_cold_library(
    session: &PySemanticTransitionSession,
    value: &Bound<'_, PyAny>,
    initial_theory: &str,
    input_facts: &str,
    statements: &[(String, u8)],
    query_records: &[u32],
) -> PyResult<()> {
    session.require_creator()?;
    let budget = session.material_sources.native_budget()?;
    let declaration = declaration(value, &budget)?;
    let text = std::str::from_utf8(declaration.material.as_bytes())
        .map_err(|_| invalid("rule library declaration changed its UTF-8 encoding"))?;
    for record in text.lines() {
        xlog_gpu::logic::validate_positive_rule_library_record_with_allocation_owner(
            record,
            &mut |bytes| {
                budget
                    .reserve_auxiliary(bytes)
                    .map(|owner| Arc::new(owner) as Arc<dyn Send + Sync>)
                    .map_err(|error| xlog_core::XlogError::Compilation(error.to_string()))
            },
        )
        .map_err(xlog_err)?;
    }
    let original = session
        .editable_source()?
        .program
        .as_ref()
        .cloned()
        .ok_or_else(|| invalid("rule library requires the original positive task program"))?;
    let source_bytes = initial_theory
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_add(declaration.material.as_bytes().len()))
        .ok_or_else(|| invalid("rule library compiler source extent overflow"))?;
    let transient_bytes = source_bytes
        .checked_add(
            statements
                .len()
                .checked_mul(std::mem::size_of::<&str>())
                .ok_or_else(|| invalid("rule library statement references overflow"))?,
        )
        .and_then(|n| n.checked_add(32 * std::mem::size_of::<xlog_cuda::SemanticProgramRule>()))
        .ok_or_else(|| invalid("rule library compiler allocation extent overflow"))?;
    let _transient = budget
        .reserve_auxiliary(transient_bytes as u64)
        .map_err(xlog_err)?;
    let mut source = String::with_capacity(source_bytes);
    source.push_str(initial_theory);
    source.push('\n');
    source.push_str(
        std::str::from_utf8(declaration.material.as_bytes())
            .map_err(|_| invalid("rule library declaration changed its UTF-8 encoding"))?,
    );
    let texts = statements
        .iter()
        .map(|(text, _)| text.as_str())
        .collect::<Vec<_>>();
    let (mut compiled, _compiler_owner) =
        xlog_gpu::logic::compile_positive_task_with_allocation_owner(
            &source,
            input_facts,
            &texts,
            query_records,
            &mut |bytes| {
                budget
                    .reserve_auxiliary(bytes)
                    .map(|owner| Arc::new(owner) as Arc<dyn Send + Sync>)
                    .map_err(|error| xlog_core::XlogError::Compilation(error.to_string()))
            },
        )
        .map_err(xlog_err)?;
    let program = original.program.as_ref();
    if compiled.predicate_count != program.predicate_count
        || compiled.predicate_arities != program.predicate_arities
        || compiled.initial_facts != program.initial_facts
        || compiled.queries != program.queries
        || compiled.query_records != program.query_records
        || compiled.observation_facts != program.observation_facts
        || compiled.initial_rules.len() != program.initial_rules.len() + 32
        || !compiled.initial_rules.starts_with(&program.initial_rules)
    {
        return Err(invalid(
            "rule library changed the original vocabulary, facts, queries or program",
        ));
    }
    let rules = compiled
        .initial_rules
        .split_off(program.initial_rules.len());
    let metadata = Arc::new(
        budget
            .reserve_auxiliary(
                (std::mem::size_of::<cold_task::EditableProgramOwner>()
                    + 2 * std::mem::size_of::<usize>()) as u64,
            )
            .map_err(xlog_err)?,
    );
    let library = session
        .owner()?
        .bind_evaluation_rule_library(declaration.material, declaration.digest, rules)
        .map_err(xlog_err)?;
    let owner = original.with_rule_library(library, metadata);
    session.editable_source()?.program = Some(owner);
    Ok(())
}
