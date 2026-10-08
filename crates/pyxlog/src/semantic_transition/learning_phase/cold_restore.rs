//! Authenticate original cold phase custody in the canonical checkpoint loader.
//! Scientific comparisons are not re-run: their original native-signed result,
//! frozen program and complete delivered history must remain unchanged.

use super::*;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use phase_record::{lifecycle_payloads, RecordReader};
use rand::rngs::OsRng;
use serde_json::{Map, Value};

pub(super) const SOURCE_EVALUATION_CANCELLED: &str = "source-evaluation-cancelled";
pub(super) const PRIVATE_EVALUATION_CANCELLED: &str = "private-evaluation-cancelled";
pub(super) const PRIVATE_SEGMENT_BUDGET_REFUSED: &str = "private-segment-budget-refused";
pub(super) const PRIVATE_SEGMENT_CANCELLED: &str = "private-segment-cancelled";

pub(super) fn cancelled_evaluation_reason(
    proof: &xlog_cuda::SemanticCancelledModelEvaluation,
    cause: &str,
) -> ColdValue {
    let parent = proof.parent();
    ColdValue::Sequence(vec![
        ColdValue::Text(cause.to_owned()),
        ColdValue::Sequence(vec![
            ColdValue::Text(hex(parent.instance.as_bytes())),
            ColdValue::Integer(parent.word.to_string()),
            ColdValue::Text(hex(parent.logical_digest.as_bytes())),
            ColdValue::Text(hex(parent.state_digest.as_bytes())),
        ]),
        ColdValue::Integer(proof.model_work().to_string()),
        ColdValue::Integer(proof.model_calls().to_string()),
    ])
}

pub(in crate::semantic_transition) struct VerifiedClosure {
    pub(in crate::semantic_transition) phase_id: [u8; 32],
    pub(in crate::semantic_transition) completion_digest: [u8; 32],
    pub(in crate::semantic_transition) admission: Arc<[u8]>,
    pub(in crate::semantic_transition) program: Arc<[u8]>,
    pub(in crate::semantic_transition) history: Arc<[u8]>,
    pub(in crate::semantic_transition) acceptance: Arc<[u8]>,
    pub(in crate::semantic_transition) snapshot: AuthoritySnapshot,
    pub(in crate::semantic_transition) refusal: Option<RefusalClosure>,
}

pub(in crate::semantic_transition) struct RefusalClosure {
    pub(in crate::semantic_transition) cause: Arc<str>,
    pub(in crate::semantic_transition) record: Arc<[u8]>,
    pub(in crate::semantic_transition) expense: ProposalExpense,
    pub(super) source_native: Arc<[u8]>,
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(DIGITS[(byte >> 4) as usize] as char);
        text.push(DIGITS[(byte & 15) as usize] as char);
    }
    text
}

fn unhex(value: &Value) -> PyResult<Vec<u8>> {
    let text = value
        .as_str()
        .filter(|text| !text.is_empty() && text.len().is_multiple_of(2))
        .ok_or_else(|| invalid("signed phase instruction lacks its original byte material"))?;
    let digit = |byte| match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(invalid(
            "signed phase byte material is not canonical hexadecimal",
        )),
    };
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| Ok(digit(pair[0])? * 16 + digit(pair[1])?))
        .collect()
}

fn digest(bytes: &[u8]) -> Value {
    Value::String(hex(&Sha256::digest(bytes)))
}

/// Check the existing cold JSON transport, including Python's ASCII escapes,
/// exact integer spellings, sorted object keys and absence of duplicate keys.
/// This is not an alternate scientific-program or executable-instruction codec.
pub(super) fn json(bytes: &[u8]) -> PyResult<Value> {
    fn string(text: &str, out: &mut Vec<u8>) {
        out.push(b'"');
        for character in text.chars() {
            match character {
                '"' => out.extend_from_slice(b"\\\""),
                '\\' => out.extend_from_slice(b"\\\\"),
                '\u{8}' => out.extend_from_slice(b"\\b"),
                '\t' => out.extend_from_slice(b"\\t"),
                '\n' => out.extend_from_slice(b"\\n"),
                '\u{c}' => out.extend_from_slice(b"\\f"),
                '\r' => out.extend_from_slice(b"\\r"),
                ' '..='~' => out.push(character as u8),
                _ => {
                    for unit in character.encode_utf16(&mut [0; 2]).iter() {
                        out.extend_from_slice(format!("\\u{unit:04x}").as_bytes());
                    }
                }
            }
        }
        out.push(b'"');
    }
    fn append(value: &Value, out: &mut Vec<u8>) -> PyResult<()> {
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(true) => out.extend_from_slice(b"true"),
            Value::Bool(false) => out.extend_from_slice(b"false"),
            Value::String(text) => string(text, out),
            Value::Number(number) => {
                let text = number.to_string();
                let digits = text.strip_prefix('-').unwrap_or(&text);
                if text != "0"
                    && !(matches!(digits.as_bytes().first(), Some(b'1'..=b'9'))
                        && digits.as_bytes().iter().all(u8::is_ascii_digit))
                {
                    return Err(invalid(
                        "signed cold phase JSON requires exact integer values",
                    ));
                }
                out.extend_from_slice(text.as_bytes());
            }
            Value::Array(values) => {
                out.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    append(value, out)?;
                }
                out.push(b']');
            }
            Value::Object(fields) => {
                out.push(b'{');
                let ordered: BTreeMap<_, _> = fields.iter().collect();
                for (index, (name, value)) in ordered.into_iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    string(name, out);
                    out.push(b':');
                    append(value, out)?;
                }
                out.push(b'}');
            }
        }
        Ok(())
    }
    let value = serde_json::from_slice(bytes)
        .map_err(|_| invalid("signed phase JSON is not a complete cold value"))?;
    let mut canonical = Vec::with_capacity(bytes.len());
    append(&value, &mut canonical)?;
    if canonical != bytes {
        return Err(invalid(
            "signed phase JSON changed its original canonical representation",
        ));
    }
    Ok(value)
}

fn host(value: &ColdValue) -> PyResult<Value> {
    Ok(match value {
        ColdValue::None => Value::Null,
        ColdValue::Bool(value) => Value::Bool(*value),
        ColdValue::Integer(value) => {
            serde_json::from_str(value).map_err(|_| invalid("signed cold integer is invalid"))?
        }
        ColdValue::Text(value) => Value::String(value.clone()),
        ColdValue::Sequence(values) => {
            Value::Array(values.iter().map(host).collect::<PyResult<_>>()?)
        }
        ColdValue::Bytes(bytes) => {
            let mut fields = Map::new();
            fields.insert("bytes_hex".to_owned(), Value::String(hex(bytes)));
            Value::Object(fields)
        }
    })
}

fn identity_value(identity: xlog_cuda::SemanticPublishedIdentity) -> Value {
    Value::Array(vec![
        Value::String(hex(identity.instance.as_bytes())),
        Value::from(identity.word),
        Value::String(hex(identity.logical_digest.as_bytes())),
        Value::String(hex(identity.state_digest.as_bytes())),
    ])
}

fn identity_material(identity: xlog_cuda::SemanticPublishedIdentity) -> Vec<u8> {
    ColdValue::Sequence(vec![
        ColdValue::Bytes(Arc::from(identity.instance.as_bytes().as_slice())),
        ColdValue::Integer(identity.word.to_string()),
        ColdValue::Bytes(Arc::from(identity.logical_digest.as_bytes().as_slice())),
        ColdValue::Bytes(Arc::from(identity.state_digest.as_bytes().as_slice())),
    ])
    .canonical_bytes()
}

fn original_bytes(value: &Bound<'_, PyAny>, extent: usize) -> PyResult<Vec<u8>> {
    if !value.is_exact_instance_of::<PyBytes>() {
        return Err(invalid("cold phase custody requires exact builtin bytes"));
    }
    let bytes = value.cast::<PyBytes>()?.as_bytes();
    if bytes.len() != extent {
        return Err(invalid("cold phase custody has an incorrect fixed extent"));
    }
    Ok(bytes.to_vec())
}

fn require_fields(value: &Value, names: &[&str]) -> PyResult<()> {
    let fields = value
        .as_object()
        .ok_or_else(|| invalid("signed phase material lacks an original object"))?;
    if fields.len() != names.len() || names.iter().any(|name| !fields.contains_key(*name)) {
        return Err(invalid("signed phase material changed its original fields"));
    }
    Ok(())
}

fn require_device_certificate(certificate: &Value) -> PyResult<u64> {
    require_fields(
        certificate,
        &[
            "format",
            "observer_identity",
            "device_uuid",
            "total_physical_bytes",
            "process_id",
            "driver_version",
            "source",
            "whole_device",
        ],
    )?;
    let identity = unhex(&certificate["observer_identity"])?;
    let uuid = unhex(&certificate["device_uuid"])?;
    let capacity = certificate["total_physical_bytes"]
        .as_u64()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| invalid("original device capacity must be positive exact bytes"))?;
    if certificate["format"] != "xlog.device-capacity/1"
        || certificate["source"] != "nvml-memory-v2"
        || certificate["whole_device"] != true
        || identity.len() != 32
        || identity.iter().all(|byte| *byte == 0)
        || uuid.len() != 16
        || certificate["process_id"].as_u64().is_none_or(|id| id == 0)
        || certificate["driver_version"]
            .as_str()
            .is_none_or(|version| {
                version.is_empty()
                    || version.len() >= 32
                    || !version.as_bytes().iter().all(u8::is_ascii_graphic)
            })
    {
        return Err(invalid(
            "signed history lacks its original whole-device capacity certificate",
        ));
    }
    Ok(capacity)
}

/// One memory interpretation for native admission and original signed recovery.
/// Old physical-peak programs are not re-labelled under this format.
pub(super) fn require_program_memory(program: &Value, certificate: &Value) -> PyResult<()> {
    if program["format"] != resource_observer::PROGRAM_FORMAT
        || program["memory_unit"] != resource_observer::MEMORY_UNIT
        || program["memory_admission"] != resource_observer::MEMORY_ADMISSION
    {
        return Err(invalid("scientific program must freeze the original backing unit and separate whole-device capacity admission"));
    }
    let capacity = require_device_certificate(certificate)?;
    let schedule = program["schedule"]
        .as_array()
        .filter(|schedule| !schedule.is_empty())
        .ok_or_else(|| invalid("memory admission requires every original scheduled operation"))?;
    for limits in schedule
        .iter()
        .map(|entry| &entry["budget"])
        .chain(std::iter::once(&program["terminal_cleanup"]["budget"]))
    {
        let limits = limits
            .as_array()
            .filter(|limits| {
                limits.len() == 3 && limits.iter().all(|value| value.as_u64().is_some())
            })
            .ok_or_else(|| {
                invalid("memory admission requires every original exact whole-resource budget")
            })?;
        let ceiling = limits[1]
            .as_u64()
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| {
                invalid("memory admission requires an original positive byte ceiling")
            })?;
        if capacity > ceiling {
            return Err(invalid("whole-device physical capacity cannot prove an original operation or cleanup ceiling; refuse before allocation/admission without rewriting its budget"));
        }
    }
    Ok(())
}

/// Live signing compares to the native-retained original certificate. Recovery
/// checks the already signed certificate without substituting the new process.
pub(super) fn require_memory_history(
    program: &Value,
    history: &Value,
    expected: Option<&Value>,
) -> PyResult<()> {
    if history["format"] != resource_observer::PROGRAM_FORMAT {
        return Err(invalid(
            "original history cannot reinterpret another memory format",
        ));
    }
    let records = history["records"]
        .as_array()
        .ok_or_else(|| invalid("original memory history lacks its actual observations"))?;
    let certificate = expected
        .or_else(|| records.first().map(|record| &record["device_certificate"]))
        .ok_or_else(|| invalid("original memory history lacks its native certificate"))?;
    require_program_memory(program, certificate)?;
    let capacity = require_device_certificate(certificate)?;
    for record in records.iter().chain(
        history
            .get("terminal_cleanup")
            .filter(|value| !value.is_null()),
    ) {
        let peak = record["resource_usage"]
            .as_array()
            .filter(|values| values.len() == 3)
            .and_then(|values| values[1].as_u64())
            .ok_or_else(|| invalid("backing history lacks its actual exact peak bytes"))?;
        if record["memory_unit"] != resource_observer::MEMORY_UNIT
            || record["memory_admission"] != resource_observer::MEMORY_ADMISSION
            || record["device_certificate"] != *certificate
            || peak > capacity
        {
            return Err(invalid("original backing observation changed its unit, native certificate or whole-device bound"));
        }
    }
    Ok(())
}

fn observation_metadata(record: &Value) -> PyResult<(&Value, &Value, &Value)> {
    if record["memory_unit"] != resource_observer::MEMORY_UNIT
        || record["memory_admission"] != resource_observer::MEMORY_ADMISSION
    {
        return Err(invalid(
            "terminal observation lost its original memory interpretation",
        ));
    }
    require_device_certificate(&record["device_certificate"])?;
    Ok((
        &record["memory_unit"],
        &record["memory_admission"],
        &record["device_certificate"],
    ))
}

/// Recognize only the original negative comparison of a complete prefix.
/// A contract exception with no comparisons is not a scientific zero result.
pub(super) fn scientific_refusal_reason(
    program: &[u8],
    prefix: &[u8],
    result: &[u8],
) -> PyResult<ColdValue> {
    let program = json(program)?;
    let prefix = json(prefix)?;
    let result = json(result)?;
    require_fields(
        &result,
        &[
            "format",
            "program",
            "history",
            "outcome",
            "comparisons",
            "reason",
            "boundary",
        ],
    )?;
    if result["program"] != program
        || result["history"] != prefix
        || result["format"] != program["format"]
        || result["outcome"] != "phase-refused"
        || result["comparisons"].as_array().is_none_or(Vec::is_empty)
        || program["schedule"]
            .as_array()
            .and_then(|steps| steps.len().checked_sub(1))
            != prefix["records"].as_array().map(Vec::len)
    {
        return Err(invalid(
            "scientific refusal lacks its original complete program, prefix and comparisons",
        ));
    }
    Ok(ColdValue::Text(
        result["reason"]
            .as_str()
            .filter(|reason| !reason.is_empty())
            .ok_or_else(|| invalid("scientific refusal lost its original negative reason"))?
            .to_owned(),
    ))
}

pub(super) fn require_terminal_history(
    prefix: &[u8],
    history: &[u8],
    instruction: &[u8],
    ordinal: u64,
    branch: &str,
    reason: &ColdValue,
    usage: [u64; 3],
) -> PyResult<()> {
    let mut prefix = json(prefix)?;
    let history = json(history)?;
    let actual = history["records"]
        .as_array()
        .and_then(|records| records.last())
        .ok_or_else(|| invalid("terminal refusal lost its actual original observation"))?;
    let (memory_unit, memory_admission, device_certificate) = observation_metadata(actual)?;
    let records = prefix["records"]
        .as_array_mut()
        .ok_or_else(|| invalid("terminal refusal lost its original prefix records"))?;
    let record = serde_json::json!({
        "instruction_sha256": digest(instruction), "operation_ordinal": ordinal,
        "branch": branch, "resource_usage": usage, "refusal": host(reason)?,
        "memory_unit": memory_unit, "memory_admission": memory_admission,
        "device_certificate": device_certificate,
    });
    records.push(record);
    if prefix != history {
        return Err(invalid(
            "terminal refusal changed its original prefix or actual final observation",
        ));
    }
    Ok(())
}

pub(super) fn require_segment_terminal_history(
    prefix: &[u8],
    history: &[u8],
    ordinal: u64,
    start: u64,
    count: u64,
    branch: &str,
    cause: &str,
    usage: [u64; 3],
) -> PyResult<()> {
    let mut prefix = json(prefix)?;
    require_fields(
        &prefix,
        &["format", "program_sha256", "records", "terminal_cleanup"],
    )?;
    let history = json(history)?;
    let (memory_unit, memory_admission, device_certificate) =
        observation_metadata(&history["terminal_cleanup"])?;
    let records = prefix["records"]
        .as_array()
        .ok_or_else(|| invalid("segment terminal closure lost its complete original history"))?;
    let extent = u64::try_from(records.len()).ok();
    let original_extent = match cause {
        PRIVATE_SEGMENT_BUDGET_REFUSED => start.checked_add(count),
        PRIVATE_SEGMENT_CANCELLED => Some(start),
        _ => {
            return Err(invalid(
                "segment terminal closure has another native disposition",
            ))
        }
    };
    if !prefix["terminal_cleanup"].is_null()
        || count == 0
        || original_extent != extent
        || !matches!(branch, "real" | "control")
    {
        return Err(invalid(
            "segment terminal closure changed its original history extent",
        ));
    }
    prefix["terminal_cleanup"] = serde_json::json!({
        "operation_ordinal": ordinal, "cause": cause,
        "group_start": start, "group_count": count, "branch": branch,
        "resource_usage": usage,
        "memory_unit": memory_unit, "memory_admission": memory_admission,
        "device_certificate": device_certificate,
    });
    if prefix != history {
        return Err(invalid(
            "segment terminal cleanup must preserve every original step and its expenditure",
        ));
    }
    Ok(())
}

fn require_history(history: &Value, terminal_cleanup: bool) -> PyResult<()> {
    require_fields(
        history,
        &["format", "program_sha256", "records", "terminal_cleanup"],
    )?;
    if !terminal_cleanup && !history["terminal_cleanup"].is_null() {
        return Err(invalid(
            "ordinary phase history cannot contain a terminal cleanup observation",
        ));
    }
    Ok(())
}

fn require_operation_observation(
    entry: &Value,
    record: &Value,
    ordinal: usize,
    terminal: bool,
) -> PyResult<()> {
    let instruction = unhex(&entry["instruction"]["bytes_hex"])?;
    let budget = entry["budget"]
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| invalid("signed phase operation lost its original whole-resource budget"))?;
    let expenditure = record["resource_usage"]
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| invalid("signed phase operation lost its actual resource expenditure"))?;
    if record["operation_ordinal"].as_u64() != Some(ordinal as u64)
        || record["branch"] != entry["branch"]
        || record["instruction_sha256"] != digest(&instruction)
        || record.as_object().is_none()
        || (!terminal
            && record
                .get("refusal")
                .is_some_and(|reason| !reason.is_null()))
    {
        return Err(invalid(
            "signed phase history changed its original operation or disposition",
        ));
    }
    if matches!(
        entry["operation"].as_str(),
        Some("proposal" | "update" | "recompute")
    ) && (record["step"]["step_ordinal"] != entry["step_ordinal"]
        || record["step"]["transition"] != entry["operation"]
        || !terminal
            && (record["step"]["skipped"] != false || !record["step"]["refusal"].is_null()))
    {
        return Err(invalid(
            "signed phase history contains a skipped, refused or substituted numerical step",
        ));
    }
    for (index, (spent, limit)) in expenditure.iter().zip(budget).enumerate() {
        let spent = spent
            .as_u64()
            .ok_or_else(|| invalid("signed phase expenditure is not an exact u64"))?;
        let limit = limit
            .as_u64()
            .ok_or_else(|| invalid("signed phase budget is not an exact u64"))?;
        if (!terminal && spent > limit) || index == 1 && limit == 0 {
            return Err(invalid(
                "signed phase history exceeded its original operation budget",
            ));
        }
    }
    Ok(())
}

fn recipe_material(record: &xlog_cuda::SemanticLearningPhaseRecord) -> Vec<u8> {
    let integer = |value: u64| ColdValue::Integer(value.to_string());
    let views = record
        .recipe
        .iter()
        .map(|operation| {
            let (role, index, action, source_role, source_index) = match *operation {
                SemanticLearningCopyReset::Preserve { role, index } => {
                    (role, index, "preserve", 0, 0)
                }
                SemanticLearningCopyReset::Zero { role, index } => (role, index, "zero", 0, 0),
                SemanticLearningCopyReset::Phase { index } => (23, index, "phase", 0, 0),
                SemanticLearningCopyReset::MasterFromEffective {
                    index,
                    effective_role,
                    effective_index,
                } => (
                    21,
                    index,
                    "master-from-effective",
                    effective_role,
                    effective_index,
                ),
            };
            ColdValue::Sequence(vec![
                integer(role),
                integer(index),
                ColdValue::Text(action.to_owned()),
                integer(source_role),
                integer(source_index),
            ])
        })
        .collect();
    ColdValue::Sequence(vec![
        ColdValue::Text(phase_name(record.source).to_owned()),
        ColdValue::Text(phase_name(record.target).to_owned()),
        integer(record.phase_index),
        integer(record.completed_updates_index),
        ColdValue::Sequence(views),
    ])
    .canonical_bytes()
}

/// Live and cold custody prove the original executed native prefix. Complete
/// delivery additionally requires a transition reaching its declared final phase;
/// cancelled evaluation preserves genuine absence or partial phase progress.
pub(super) fn require_executed_lineage(
    source_native: &[u8],
    final_native: &[u8],
    admission: &[u8],
    recipe_bytes: &[u8],
    final_phase: Option<SemanticLearningPhase>,
) -> PyResult<Vec<xlog_cuda::SemanticLearningPhaseRecord>> {
    let recipe = ColdValue::from_canonical_bytes(recipe_bytes)?;
    let recipe = recipe.fields(5)?;
    let source_phase = phase(recipe[0].text()?)?;
    let original = SemanticTransitionSession::checkpoint_learning_phase_history(source_native)
        .map_err(xlog_err)?;
    let final_phases = SemanticTransitionSession::checkpoint_learning_phase_history(final_native)
        .map_err(xlog_err)?;
    let added = final_phases
        .strip_prefix(original.as_slice())
        .ok_or_else(|| invalid("phase checkpoint lacks its original executed native transition"))?;
    if added
        .iter()
        .any(|record| record.admission.as_slice() != admission)
        || added.first().is_some_and(|first| {
            first.source != source_phase || recipe_material(first) != recipe_bytes
        })
        || final_phase
            .is_some_and(|expected| added.last().map(|last| last.target) != Some(expected))
    {
        return Err(invalid(
            "phase checkpoint differs from its original executed recipe or signed admission",
        ));
    }
    Ok(added.to_vec())
}

fn require_private_expense(
    source: &SemanticCheckpointManifest,
    private: &SemanticCheckpointManifest,
) -> PyResult<ProposalExpense> {
    let (source_seed, _, _, _) = TaskCheckpointSeed::decode(&source.task)?;
    let (private_seed, _, _, _) = TaskCheckpointSeed::decode(&private.task)?;
    let expense = private_seed.proposal_expense()?;
    if source.session != private.session
        || source_seed.training_domain != private_seed.training_domain
        || source_seed.training_objective != private_seed.training_objective
        || source_seed.checkpoint_source_limits()? != private_seed.checkpoint_source_limits()?
        || source_seed.proposal_expense()?.capacity != expense.capacity
        || source_seed.proposal_expense()?.spent > expense.spent
    {
        return Err(invalid(
            "terminal phase changed original Source configuration, task or cumulative expense",
        ));
    }
    require_retained_arena_authority(&source_seed.authority, &private_seed.authority)?;
    Ok(expense)
}

impl VerifiedClosure {
    #[expect(
        clippy::too_many_arguments,
        reason = "terminal verification retains the already decoded original signed inputs and fence"
    )]
    fn read_refusal(
        phase_id: [u8; 32],
        admission: &[u8],
        terminal_record: &[u8],
        lifecycle: &phase_record::LifecyclePayloads<'_>,
        source: &[u8],
        program_bytes: &[u8],
        recipe_bytes: &[u8],
        grant: &ColdValue,
        feedback: [&[u8]; 2],
        destination: &str,
        source_phase: SemanticLearningPhase,
        target_phase: SemanticLearningPhase,
        pending_target: Option<[u8; 32]>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let mut reader = RecordReader(
            lifecycle
                .completion
                .strip_prefix(b"xlog.learning-phase.refusal.v1\0")
                .ok_or_else(|| invalid("terminal phase record has another original domain"))?,
        );
        let cause = std::str::from_utf8(reader.field()?)
            .map_err(|_| invalid("terminal phase cause is not its original text"))?;
        let retained_source = reader.field()?;
        let private = reader.field()?;
        let prefix_bytes = reader.field()?;
        let result_bytes = reader.field()?;
        let history_bytes = reader.field()?;
        let entry_material = reader.field()?;
        let reason = ColdValue::from_canonical_bytes(reader.field()?)?;
        let snapshot =
            AuthoritySnapshot::parse(&ColdValue::from_canonical_bytes(reader.field()?)?)?;
        let mut usage = RecordReader(reader.field()?);
        let usage_words = [usage.word()?, usage.word()?, usage.word()?];
        usage.finish()?;
        reader.finish()?;
        if retained_source != source {
            return Err(invalid(
                "terminal phase changed its original full Source checkpoint",
            ));
        }
        if matches!(
            cause,
            SOURCE_EVALUATION_CANCELLED
                | PRIVATE_EVALUATION_CANCELLED
                | PRIVATE_SEGMENT_BUDGET_REFUSED
                | PRIVATE_SEGMENT_CANCELLED
        ) {
            let source_only = cause == SOURCE_EVALUATION_CANCELLED;
            let cancelled_segment = cause == PRIVATE_SEGMENT_CANCELLED;
            let segment = cancelled_segment || cause == PRIVATE_SEGMENT_BUDGET_REFUSED;
            if lifecycle.preparation.is_some()
                || source_only != private.is_empty()
                || !result_bytes.is_empty()
                || pending_target.is_some()
            {
                return Err(invalid("Source evaluation cancellation contradicts absent private restoration or scientific acceptance"));
            }
            let manifest = SemanticCheckpointManifest::decode(source)?;
            let (seed, saved_snapshot, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
            let mut expense = seed.proposal_expense()?;
            let source_parent =
                SemanticTransitionSession::state_material_projection(&manifest.native)
                    .map_err(xlog_err)?
                    .publication;
            let recipe = ColdValue::from_canonical_bytes(recipe_bytes)?;
            let fields = recipe.fields(5)?;
            let values = fields
                .iter()
                .map(|field| field.python_value(py))
                .collect::<PyResult<Vec<_>>>()?;
            let original_recipe = PySemanticLearningPhaseRecipe::new(
                values[0].bind(py),
                values[1].bind(py),
                values[2].bind(py),
                values[3].bind(py),
                values[4].bind(py),
            )?;
            let program = json(program_bytes)?;
            let prefix = json(prefix_bytes)?;
            let history = json(history_bytes)?;
            require_memory_history(&program, &history, None)?;
            require_history(&prefix, false)?;
            require_history(&history, segment)?;
            let records = prefix["records"]
                .as_array()
                .ok_or_else(|| invalid("cancelled Source lacks its original actual prefix"))?;
            let prefix_count = records.len();
            let schedule = program["schedule"]
                .as_array()
                .filter(|schedule| schedule.len() > 1)
                .ok_or_else(|| invalid("cancelled Source lost its original frozen schedule"))?;
            let reason_fields = reason.fields(if segment { 5 } else { 4 })?;
            let group_start = if segment {
                usize::try_from(reason_fields[2].unsigned()?)
                    .map_err(|_| invalid("terminal segment group position overflowed"))?
            } else {
                prefix_count
            };
            let entry = schedule
                .get(group_start)
                .ok_or_else(|| invalid("terminal refusal invented an original group position"))?;
            let branch = entry["branch"]
                .as_str()
                .ok_or_else(|| invalid("terminal refusal lost its original branch"))?;
            let target = TerminalCleanupTarget::read(program_bytes)?;
            let ordinal = if segment {
                target.ordinal
            } else {
                prefix_count as u64
            };
            let cancelled_parent = if source_only {
                if branch != "source" {
                    return Err(invalid("Source cancellation changed its original branch"));
                }
                source_parent
            } else {
                if !matches!(branch, "control" | "real") {
                    return Err(invalid(
                        "private cancellation requires its actual original private branch",
                    ));
                }
                let private_manifest = SemanticCheckpointManifest::decode(private)?;
                let (_, private_snapshot, _, _) =
                    TaskCheckpointSeed::decode(&private_manifest.task)?;
                expense = require_private_expense(&manifest, &private_manifest)?;
                snapshot.newer_than(&private_snapshot)?;
                // Initial trajectory construction is an exact Source restore.
                // Only executed cold restores may add a phase. No transition
                // yet is genuine absence, never a fabricated first recipe.
                let added = require_executed_lineage(
                    &manifest.native,
                    &private_manifest.native,
                    admission,
                    recipe_bytes,
                    None,
                )?;
                let restored = schedule
                    .iter()
                    .take(prefix_count)
                    .filter(|entry| entry["branch"] == branch && entry["operation"] == "restore")
                    .count();
                if added.len() > restored
                    || !schedule.iter().take(prefix_count).any(|entry| {
                        entry["branch"] == branch && entry["operation"] == "trajectory-start"
                    })
                {
                    return Err(invalid(
                        "cancelled private checkpoint extends beyond its actual original prefix",
                    ));
                }
                SemanticTransitionSession::state_material_projection(&private_manifest.native)
                    .map_err(xlog_err)?
                    .publication
            };
            let entry_fields = ColdValue::from_canonical_bytes(entry_material)?;
            let entry_fields = entry_fields.fields(6)?;
            let format = program["format"]
                .as_str()
                .filter(|format| !format.is_empty())
                .ok_or_else(|| invalid("cancelled Source lost its original program format"))?;
            if prefix_count == 0
                || prefix_count >= schedule.len() - 1
                || source_only
                    && records
                        .iter()
                        .zip(schedule)
                        .enumerate()
                        .any(|(index, (_, operation))| {
                            operation["branch"] != "source"
                                || index == 0 && operation["operation"] != "prepare"
                                || index != 0 && operation["operation"] != "evaluation"
                        })
                || prefix["format"] != format
                || history["format"] != format
                || prefix["program_sha256"] != digest(program_bytes)
                || history["program_sha256"] != digest(program_bytes)
                || program["source_checkpoint_sha256"] != digest(source)
                || program["source_parent"] != identity_value(source_parent)
                || program["grant"] != host(grant)?
                || program["destination"] != destination
                || program["feedback_interventions"]["real"]
                    != host(&ColdValue::Bytes(Arc::from(feedback[0])))?
                || program["feedback_interventions"]["control"]
                    != host(&ColdValue::Bytes(Arc::from(feedback[1])))?
                || phase(
                    program["final_phase"].as_str().ok_or_else(|| {
                        invalid("cancelled Source lacks its original final phase")
                    })?,
                )? != target_phase
                || phase(fields[0].text()?)? != source_phase
                || program["recipe_digest"]
                    != host(&ColdValue::Bytes(Arc::from(
                        original_recipe.inner.recipe_digest().as_bytes().as_slice(),
                    )))?
                || entry_fields[1].text()? != branch
                || entry_fields[2].unsigned()? != ordinal
                || entry_fields[3].unsigned()? != 0
                || reason_fields[0].text()? != cause
                || host(&reason_fields[1])? != identity_value(cancelled_parent)
            {
                return Err(invalid("cancelled Source changed its native proof, actual position, frozen inputs or measured expense"));
            }
            if segment {
                let count = reason_fields[3].unsigned()?;
                let group_count = usize::try_from(count)
                    .map_err(|_| invalid("terminal segment group count overflowed"))?;
                let group_stop = group_start
                    .checked_add(group_count)
                    .filter(|stop| *stop < schedule.len())
                    .ok_or_else(|| {
                        invalid("terminal segment exceeds its original frozen schedule")
                    })?;
                if group_start == 0
                    || group_count == 0
                    || prefix_count
                        != if cancelled_segment {
                            group_start
                        } else {
                            group_stop
                        }
                    || schedule[0]["operation"] != "prepare"
                    || schedule[0]["branch"] != "source"
                    || entry["step_ordinal"].as_u64() != Some(0)
                    || reason_fields[4].text()? != branch
                    || entry_fields[0].text()? != "terminal-cleanup"
                    || host(&entry_fields[4])? != program["terminal_cleanup"]["budget"]
                    || entry_fields[5] != ColdValue::None
                    || schedule[group_start..group_stop].iter().enumerate().any(
                        |(index, operation)| {
                            operation["branch"] != branch
                                || !matches!(
                                    operation["operation"].as_str(),
                                    Some("proposal" | "update" | "recompute")
                                )
                                || operation["step_ordinal"].as_u64() != Some(index as u64)
                                || operation["instruction"]["bytes_hex"]
                                    != entry["instruction"]["bytes_hex"]
                        },
                    )
                    || matches!(
                        schedule[group_stop]["operation"].as_str(),
                        Some("proposal" | "update" | "recompute")
                    ) && schedule[group_stop]["step_ordinal"].as_u64() != Some(0)
                {
                    return Err(invalid("segment terminal closure changed its complete original roster or frozen reserve"));
                }
                // A cancelled roster contributes no invented numerical rows.
                // Only completed-group budget refusal requires actual rows and
                // exceedance; its cleanup uses a separate cold-only interval.
                let mut exceeded = false;
                for (index, (operation, record)) in schedule.iter().zip(records).enumerate() {
                    require_operation_observation(operation, record, index, index >= group_start)?;
                    if index >= group_start {
                        let usage = record["resource_usage"]
                            .as_array()
                            .expect("verified actual expenditure");
                        let budget = operation["budget"]
                            .as_array()
                            .expect("verified original budget");
                        exceeded |= usage.iter().zip(budget).any(|(actual, limit)| {
                            actual.as_u64().expect("verified unsigned expense")
                                > limit.as_u64().expect("verified unsigned limit")
                        });
                    }
                }
                if !cancelled_segment && !exceeded {
                    return Err(invalid("segment terminal budget refusal requires an actual original budget exceedance"));
                }
                require_segment_terminal_history(
                    prefix_bytes,
                    history_bytes,
                    target.ordinal,
                    group_start as u64,
                    count,
                    branch,
                    cause,
                    usage_words,
                )?;
            } else {
                if entry["operation"] != "evaluation"
                    || entry["step_ordinal"].as_u64() != Some(0)
                    || entry_fields[0].text()? != "evaluation"
                    || host(&entry_fields[4])? != entry["budget"]
                    || host(&entry_fields[5])? != entry["comparison"]
                    || reason_fields[2].unsigned()? > usage_words[0]
                    || reason_fields[3].unsigned()? > usage_words[2]
                {
                    return Err(invalid("cancelled evaluation changed its actual native proof or original instruction"));
                }
                let instruction = unhex(&entry["instruction"]["bytes_hex"])?;
                require_terminal_history(
                    prefix_bytes,
                    history_bytes,
                    &instruction,
                    ordinal,
                    branch,
                    &reason,
                    usage_words,
                )?;
                let full_records = history["records"]
                    .as_array()
                    .expect("verified actual terminal history");
                for (index, (operation, record)) in schedule.iter().zip(full_records).enumerate() {
                    require_operation_observation(operation, record, index, index == prefix_count)?;
                }
            }
            if records[0]["parent"] != identity_value(source_parent)
                || records[0]["source_checkpoint_sha256"] != digest(source)
            {
                return Err(invalid(
                    "cancelled Source changed its original completed preparation",
                ));
            }
            snapshot.newer_than(&saved_snapshot)?;
            return Ok(Self {
                phase_id,
                completion_digest: Sha256::digest(terminal_record).into(),
                admission: Arc::from(admission),
                program: Arc::from(program_bytes),
                history: Arc::from(history_bytes),
                acceptance: Arc::from(&[][..]),
                snapshot,
                refusal: Some(RefusalClosure {
                    cause: Arc::from(cause),
                    record: Arc::from(terminal_record),
                    expense,
                    source_native: Arc::from(manifest.native.as_slice()),
                }),
            });
        }
        if pending_target.is_some_and(|target| {
            lifecycle.preparation.is_none() || target != <[u8; 32]>::from(Sha256::digest(private))
        }) {
            return Err(invalid(
                "terminal pending fence differs from its signed original private outcome",
            ));
        }
        match (cause, lifecycle.preparation) {
            ("scientific-comparison-refused", None) => {
                if scientific_refusal_reason(program_bytes, prefix_bytes, result_bytes)? != reason {
                    return Err(invalid(
                        "terminal phase changed its original scientific reason",
                    ));
                }
            }
            ("private-candidate-abandoned" | "checkpoint-publication-absent", Some(prepared)) => {
                let mut prepared = RecordReader(prepared);
                if prepared.field()? != private
                    || prepared.field()? != prefix_bytes
                    || prepared.field()? != result_bytes
                    || reason != ColdValue::Text(cause.to_owned())
                {
                    return Err(invalid(
                        "terminal phase changed its original positive private outcome",
                    ));
                }
                prepared.finish()?;
            }
            _ => {
                return Err(invalid(
                    "terminal phase cause contradicts its signed lifecycle position",
                ))
            }
        }
        let source_manifest = SemanticCheckpointManifest::decode(source)?;
        let private_manifest = SemanticCheckpointManifest::decode(private)?;
        let (_, source_snapshot, _, _) = TaskCheckpointSeed::decode(&source_manifest.task)?;
        let (_, private_snapshot, _, _) = TaskCheckpointSeed::decode(&private_manifest.task)?;
        let expense = require_private_expense(&source_manifest, &private_manifest)?;
        let program = json(program_bytes)?;
        let declared_phase =
            phase(program["final_phase"].as_str().ok_or_else(|| {
                invalid("terminal phase lost its original declared final phase")
            })?)?;
        let phases = require_executed_lineage(
            &source_manifest.native,
            &private_manifest.native,
            admission,
            recipe_bytes,
            Some(declared_phase),
        )?;
        let source_parent =
            SemanticTransitionSession::state_material_projection(&source_manifest.native)
                .map_err(xlog_err)?
                .publication;
        let private_parent =
            SemanticTransitionSession::state_material_projection(&private_manifest.native)
                .map_err(xlog_err)?
                .publication;
        let prefix = json(prefix_bytes)?;
        let history = json(history_bytes)?;
        let result = json(result_bytes)?;
        require_memory_history(&program, &history, None)?;
        require_history(&prefix, false)?;
        require_history(&history, false)?;
        require_fields(
            &result,
            &[
                "format",
                "program",
                "history",
                "outcome",
                "comparisons",
                "reason",
                "boundary",
            ],
        )?;
        let format = program["format"]
            .as_str()
            .filter(|format| !format.is_empty())
            .ok_or_else(|| invalid("terminal program lost its original format"))?;
        if phases[0].source != source_phase
            || declared_phase != target_phase
            || prefix["format"] != format
            || history["format"] != format
            || result["format"] != format
            || prefix["program_sha256"] != digest(program_bytes)
            || history["program_sha256"] != digest(program_bytes)
            || result["program"] != program
            || result["history"] != prefix
            || result["comparisons"].as_array().is_none_or(Vec::is_empty)
            || program["source_checkpoint_sha256"] != digest(source)
            || program["source_parent"] != identity_value(source_parent)
            || program["recipe_digest"]
                != host(&ColdValue::Bytes(Arc::from(
                    phases[0].recipe_digest.as_bytes().as_slice(),
                )))?
            || program["grant"] != host(grant)?
            || program["destination"] != destination
            || program["feedback_interventions"]["real"]
                != host(&ColdValue::Bytes(Arc::from(feedback[0])))?
            || program["feedback_interventions"]["control"]
                != host(&ColdValue::Bytes(Arc::from(feedback[1])))?
            || (lifecycle.preparation.is_some()
                && (result["outcome"] != "frozen-phase-criteria-satisfied"
                    || !result["reason"].is_null()))
        {
            return Err(invalid(
                "terminal phase changed its original program, fence or comparison result",
            ));
        }
        let schedule = program["schedule"]
            .as_array()
            .filter(|schedule| schedule.len() > 1)
            .ok_or_else(|| invalid("terminal phase lost its original complete schedule"))?;
        let prefix_records = prefix["records"]
            .as_array()
            .ok_or_else(|| invalid("terminal phase lost its original complete prefix"))?;
        if prefix_records.len().checked_add(1) != Some(schedule.len()) {
            return Err(invalid(
                "terminal phase shortened its original completed prefix",
            ));
        }
        let ordinal = schedule.len() - 1;
        let last = &schedule[ordinal];
        let instruction = unhex(&last["instruction"]["bytes_hex"])?;
        require_terminal_history(
            prefix_bytes,
            history_bytes,
            &instruction,
            ordinal as u64,
            "real",
            &reason,
            usage_words,
        )?;
        let full_records = history["records"]
            .as_array()
            .expect("verified full terminal history");
        for (index, (entry, record)) in schedule.iter().zip(full_records).enumerate() {
            require_operation_observation(entry, record, index, index == ordinal)?;
        }
        let fields = ColdValue::from_canonical_bytes(entry_material)?;
        let fields = fields.fields(6)?;
        let final_checkpoint = &prefix_records[ordinal - 1];
        if last["operation"] != "delivery"
            || last["branch"] != "real"
            || last["step_ordinal"].as_u64() != Some(0)
            || !last["comparison"].is_null()
            || fields[0].text()? != "delivery"
            || fields[1].text()? != "real"
            || fields[2].unsigned()? != ordinal as u64
            || fields[3].unsigned()? != 0
            || host(&fields[4])? != last["budget"]
            || fields[5] != ColdValue::None
            || final_checkpoint["parent"] != identity_value(private_parent)
            || final_checkpoint["full_checkpoint_sha256"] != digest(private)
            || result["boundary"]["final_parent"] != final_checkpoint["parent"]
            || result["boundary"]["full_final_checkpoint_sha256"] != digest(private)
            || result["boundary"]["final_model"] != final_checkpoint["model"]
            || result["boundary"]["completed_prefix_operations"].as_u64() != Some(ordinal as u64)
            || result["boundary"]["delivery_operation_ordinal"].as_u64() != Some(ordinal as u64)
        {
            return Err(invalid("terminal phase changed its actual private checkpoint or measured final instruction"));
        }
        snapshot.newer_than(&source_snapshot)?;
        snapshot.newer_than(&private_snapshot)?;
        snapshot.newer_than(&AuthoritySnapshot::parse(&json_cold(
            &result["boundary"]["refreshed_snapshot"],
        )?)?)?;
        Ok(Self {
            phase_id,
            completion_digest: Sha256::digest(terminal_record).into(),
            admission: Arc::from(admission),
            program: Arc::from(program_bytes),
            history: Arc::from(history_bytes),
            acceptance: Arc::from(result_bytes),
            snapshot,
            refusal: Some(RefusalClosure {
                cause: Arc::from(cause),
                record: Arc::from(terminal_record),
                expense,
                source_native: Arc::from(source_manifest.native.as_slice()),
            }),
        })
    }

    pub(in crate::semantic_transition) fn read(
        checkpoint: &[u8],
        issuer: &Bound<'_, PyAny>,
        records: &Bound<'_, PyAny>,
        limits: &Bound<'_, PyAny>,
        fence: &Bound<'_, PyAny>,
        checkpoint_issuer: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let py = issuer.py();
        let issuer: [u8; 32] = original_bytes(issuer, 32)?.try_into().unwrap();
        if !records.is_exact_instance_of::<PyTuple>() || !fence.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "cold phase recovery requires the original ordered tuple and fence",
            ));
        }
        let records = records.cast::<PyTuple>()?;
        let fence = fence.cast::<PyTuple>()?;
        if !matches!(records.len(), 2 | 3)
            || fence.len() != 6
            || !limits.is_exact_instance_of::<PyTuple>()
        {
            return Err(invalid(
                "cold phase recovery requires its complete original record chain and exact limits",
            ));
        }
        let limits = RecordLimits::read(limits)?;
        let phase_id: [u8; 32] = original_bytes(&fence.get_item(0)?, 32)?.try_into().unwrap();
        let source_digest = original_bytes(&fence.get_item(1)?, 32)?;
        let target = fence.get_item(2)?;
        let destination = ColdValue::read(&fence.get_item(3)?, &mut (16 * 1024 * 1024), 0)?;
        let source_phase = ColdValue::read(&fence.get_item(4)?, &mut 128, 0)?;
        let target_phase = ColdValue::read(&fence.get_item(5)?, &mut 128, 0)?;
        let source_phase = phase(source_phase.text()?)?;
        let target_phase = phase(target_phase.text()?)?;
        let retained: Vec<_> = records.iter().collect();
        let mut raw = Vec::with_capacity(retained.len());
        for value in &retained {
            if !value.is_exact_instance_of::<PyBytes>() {
                return Err(invalid(
                    "cold phase records require exact original byte materials",
                ));
            }
            raw.push(value.cast::<PyBytes>()?.as_bytes());
        }
        let lifecycle = lifecycle_payloads(&raw, issuer, phase_id, limits)?;
        let admission = lifecycle.admission;
        let mut reader = RecordReader(
            admission
                .strip_prefix(b"xlog.learning-phase.inputs.v2\0")
                .ok_or_else(|| invalid("cold phase admission has another domain"))?,
        );
        let source = reader.field()?;
        let program_bytes = reader.field()?;
        let recipe_bytes = reader.field()?;
        let grant = ColdValue::from_canonical_bytes(reader.field()?)?;
        if reader.field()? != destination.text()?.as_bytes() {
            return Err(invalid("cold phase chain changed its original destination"));
        }
        let feedback = [reader.field()?, reader.field()?];
        let _capacity = ColdValue::from_canonical_bytes(reader.field()?)?;
        if reader.field()? != limits.encode() {
            return Err(invalid("cold phase chain changed its original bounds"));
        }
        reader.finish()?;
        if Sha256::digest(source).as_slice() != source_digest {
            return Err(invalid(
                "cold phase admission differs from its original source fence",
            ));
        }
        // The reserve is part of the original signed program on every path,
        // including ordinary Delivery. Recovery cannot add one after Admission.
        TerminalCleanupTarget::read(program_bytes)?;
        if lifecycle.refused {
            let pending_target = if target.is_none() {
                None
            } else {
                Some(original_bytes(&target, 32)?.try_into().unwrap())
            };
            let closure = Self::read_refusal(
                phase_id,
                raw[0],
                raw[raw.len() - 1],
                &lifecycle,
                source,
                program_bytes,
                recipe_bytes,
                &grant,
                feedback,
                destination.text()?,
                source_phase,
                target_phase,
                pending_target,
                py,
            )?;
            closure.require_checkpoint(checkpoint, source, checkpoint_issuer)?;
            return Ok(closure);
        }
        let target_digest = original_bytes(&target, 32)?;
        let preparation = lifecycle
            .preparation
            .expect("authenticated positive outcome");
        let delivery = lifecycle.completion;
        let mut reader = RecordReader(preparation);
        let accepted_checkpoint = reader.field()?;
        let prefix_bytes = reader.field()?;
        let acceptance_bytes = reader.field()?;
        reader.finish()?;
        let mut reader = RecordReader(
            delivery
                .strip_prefix(b"xlog.learning-phase.delivery.v1\0")
                .ok_or_else(|| invalid("cold phase delivery has another domain"))?,
        );
        let delivered_checkpoint = reader.field()?;
        let history_bytes = reader.field()?;
        let entry_material = reader.field()?;
        let parent_material = reader.field()?;
        let snapshot = ColdValue::from_canonical_bytes(reader.field()?)?;
        let usage_bytes = reader.field()?;
        reader.finish()?;
        if accepted_checkpoint != delivered_checkpoint
            || Sha256::digest(delivered_checkpoint).as_slice() != target_digest
        {
            return Err(invalid(
                "cold phase delivery changed its accepted full target checkpoint",
            ));
        }
        let mut reader = RecordReader(usage_bytes);
        let usage = [reader.word()?, reader.word()?, reader.word()?];
        reader.finish()?;
        let source_manifest = SemanticCheckpointManifest::decode(source)?;
        let final_manifest = SemanticCheckpointManifest::decode(delivered_checkpoint)?;
        let (source_seed, _, _, _) = TaskCheckpointSeed::decode(&source_manifest.task)?;
        let (final_seed, final_snapshot, _, _) = TaskCheckpointSeed::decode(&final_manifest.task)?;
        if source_seed.training_domain != final_seed.training_domain
            || source_seed.training_objective != final_seed.training_objective
            || source_seed.checkpoint_source_limits()? != final_seed.checkpoint_source_limits()?
            || source_seed.proposal_expense()?.capacity != final_seed.proposal_expense()?.capacity
            || source_seed.proposal_expense()?.spent > final_seed.proposal_expense()?.spent
        {
            return Err(invalid(
                "cold phase delivery changed original task resource or training custody",
            ));
        }
        require_retained_arena_authority(&source_seed.authority, &final_seed.authority)?;
        let source_parent =
            SemanticTransitionSession::state_material_projection(&source_manifest.native)
                .map_err(xlog_err)?
                .publication;
        let final_parent =
            SemanticTransitionSession::state_material_projection(&final_manifest.native)
                .map_err(xlog_err)?
                .publication;
        if parent_material != identity_material(final_parent) {
            return Err(invalid(
                "cold phase delivery changed its actual native publication identity",
            ));
        }
        let program = json(program_bytes)?;
        let declared_phase =
            phase(program["final_phase"].as_str().ok_or_else(|| {
                invalid("signed program lost its original declared final phase")
            })?)?;
        let added = require_executed_lineage(
            &source_manifest.native,
            &final_manifest.native,
            raw[0],
            recipe_bytes,
            Some(declared_phase),
        )?;
        let final_phase = added.last().expect("nonempty native phase lineage");
        if added[0].source != source_phase || final_phase.target != target_phase {
            return Err(invalid(
                "cold phase delivery differs from its original signed native admission or recipe",
            ));
        }
        let prefix = json(prefix_bytes)?;
        let history = json(history_bytes)?;
        let acceptance = json(acceptance_bytes)?;
        require_memory_history(&program, &history, None)?;
        require_history(&prefix, false)?;
        require_history(&history, false)?;
        require_fields(
            &acceptance,
            &[
                "format",
                "program",
                "history",
                "outcome",
                "comparisons",
                "reason",
                "boundary",
            ],
        )?;
        let format = program["format"]
            .as_str()
            .filter(|format| !format.is_empty())
            .ok_or_else(|| invalid("signed phase program has no original format"))?;
        if prefix["format"] != format
            || history["format"] != format
            || acceptance["format"] != format
            || prefix["program_sha256"] != digest(program_bytes)
            || history["program_sha256"] != digest(program_bytes)
            || acceptance["program"] != program
            || acceptance["history"] != prefix
            || acceptance["outcome"] != "frozen-phase-criteria-satisfied"
            || !acceptance["reason"].is_null()
            || acceptance["comparisons"]
                .as_array()
                .is_none_or(Vec::is_empty)
            || program["source_checkpoint_sha256"] != digest(source)
            || program["source_parent"] != identity_value(source_parent)
            || program["destination"] != destination.text()?
            || program["grant"] != host(&grant)?
            || program["recipe_digest"]
                != host(&ColdValue::Bytes(Arc::from(
                    added[0].recipe_digest.as_bytes().as_slice(),
                )))?
            || program["feedback_interventions"]["real"]
                != host(&ColdValue::Bytes(Arc::from(feedback[0])))?
            || program["feedback_interventions"]["control"]
                != host(&ColdValue::Bytes(Arc::from(feedback[1])))?
        {
            return Err(invalid(
                "cold phase recovery changed its original scientific acceptance or inputs",
            ));
        }
        let schedule = program["schedule"]
            .as_array()
            .filter(|schedule| schedule.len() > 1)
            .ok_or_else(|| invalid("signed phase program lost its complete original schedule"))?;
        let prefix_records = prefix["records"]
            .as_array()
            .ok_or_else(|| invalid("accepted phase lost its original completed prefix"))?;
        let full_records = history["records"]
            .as_array()
            .ok_or_else(|| invalid("delivered phase lost its original full history"))?;
        if prefix_records.len().checked_add(1) != Some(schedule.len())
            || full_records.len() != schedule.len()
            || !full_records.starts_with(prefix_records)
        {
            return Err(invalid(
                "cold phase delivery shortened or rewrote its accepted original history",
            ));
        }
        for (ordinal, (entry, record)) in schedule.iter().zip(full_records).enumerate() {
            require_operation_observation(entry, record, ordinal, false)?;
        }
        let ordinal = schedule.len() - 1;
        let last = &schedule[ordinal];
        let delivered = &full_records[ordinal];
        let fields = ColdValue::from_canonical_bytes(entry_material)?;
        let fields = fields.fields(6)?;
        if last["operation"] != "delivery"
            || last["branch"] != "real"
            || last["step_ordinal"].as_u64() != Some(0)
            || !last["comparison"].is_null()
            || fields[0].text()? != "delivery"
            || fields[1].text()? != "real"
            || fields[2].unsigned()? != ordinal as u64
            || fields[3].unsigned()? != 0
            || host(&fields[4])? != last["budget"]
            || fields[5] != ColdValue::None
            || delivered["parent"] != identity_value(final_parent)
            || delivered["full_checkpoint_sha256"] != digest(delivered_checkpoint)
            || delivered["resource_usage"]
                != Value::Array(usage.into_iter().map(Value::from).collect())
            || full_records[ordinal - 1]["parent"] != delivered["parent"]
            || full_records[ordinal - 1]["full_checkpoint_sha256"]
                != delivered["full_checkpoint_sha256"]
            || acceptance["boundary"]["final_parent"] != delivered["parent"]
            || acceptance["boundary"]["full_final_checkpoint_sha256"]
                != delivered["full_checkpoint_sha256"]
            || acceptance["boundary"]["completed_prefix_operations"].as_u64()
                != Some(ordinal as u64)
            || acceptance["boundary"]["delivery_operation_ordinal"].as_u64() != Some(ordinal as u64)
            || acceptance["boundary"]["final_model"] != full_records[ordinal - 1]["model"]
        {
            return Err(invalid(
                "cold phase delivery changed its actual accepted final checkpoint or measured tail",
            ));
        }
        let snapshot = AuthoritySnapshot::parse(&snapshot)?;
        snapshot.newer_than(&final_snapshot)?;
        let accepted_snapshot =
            AuthoritySnapshot::parse(&json_cold(&acceptance["boundary"]["refreshed_snapshot"])?)?;
        snapshot.newer_than(&accepted_snapshot)?;
        let closure = Self {
            phase_id,
            completion_digest: Sha256::digest(raw[2]).into(),
            admission: Arc::from(raw[0]),
            program: Arc::from(program_bytes),
            history: Arc::from(history_bytes),
            acceptance: Arc::from(acceptance_bytes),
            snapshot,
            refusal: None,
        };
        closure.require_checkpoint(checkpoint, delivered_checkpoint, checkpoint_issuer)?;
        Ok(closure)
    }
}

fn json_cold(value: &Value) -> PyResult<ColdValue> {
    Ok(match value {
        Value::Null => ColdValue::None,
        Value::Bool(value) => ColdValue::Bool(*value),
        Value::String(value) => ColdValue::Text(value.clone()),
        Value::Number(value) => ColdValue::Integer(value.to_string()),
        Value::Array(values) => {
            ColdValue::Sequence(values.iter().map(json_cold).collect::<PyResult<_>>()?)
        }
        Value::Object(_) => {
            return Err(invalid(
                "signed authority snapshot is not its original cold metadata",
            ))
        }
    })
}

const SEAL_DOMAIN: &[u8] = b"xlog.learning-phase.checkpoint.v1\0";
const SEAL_BYTES: usize = SEAL_DOMAIN.len() + 5 * 32 + 64;

/// This key belongs to the actual known Session, never to Python or a saved Q.
/// The trusted consumer pins its public key independently before subsequent
/// requests. A fresh known recovery gets one new issuer, not the old secret.
pub(in crate::semantic_transition) struct CheckpointCustody {
    pub(in crate::semantic_transition) closure: Arc<VerifiedClosure>,
    pub(in crate::semantic_transition) origin: [u8; 32],
    key: SigningKey,
}

impl CheckpointCustody {
    pub(in crate::semantic_transition) fn issue(
        closure: Arc<VerifiedClosure>,
        origin: [u8; 32],
    ) -> Self {
        Self {
            closure,
            origin,
            key: SigningKey::generate(&mut OsRng),
        }
    }

    pub(in crate::semantic_transition) fn issuer(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    pub(in crate::semantic_transition) fn seal(
        &self,
        manifest: &SemanticCheckpointManifest,
    ) -> PyResult<Vec<u8>> {
        let phases = SemanticTransitionSession::checkpoint_learning_phase_history(&manifest.native)
            .map_err(xlog_err)?;
        let valid = if let Some(refusal) = &self.closure.refusal {
            let original = SemanticTransitionSession::checkpoint_learning_phase_history(
                &refusal.source_native,
            )
            .map_err(xlog_err)?;
            let (seed, _, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
            let expense = seed.proposal_expense()?;
            phases == original
                && expense.capacity == refusal.expense.capacity
                && expense.spent >= refusal.expense.spent
        } else {
            phases
                .last()
                .is_some_and(|phase| phase.admission.as_slice() == self.closure.admission.as_ref())
        };
        if !valid {
            return Err(invalid(
                "checkpoint signing lost this Session's known native lineage or cumulative expense",
            ));
        }
        let mut bytes = SEAL_DOMAIN.to_vec();
        for field in [
            self.closure.phase_id,
            self.closure.completion_digest,
            self.origin,
            self.issuer(),
            manifest.unsigned_phase_root(),
        ] {
            bytes.extend_from_slice(&field);
        }
        bytes.extend_from_slice(&self.key.sign(&bytes).to_bytes());
        Ok(bytes)
    }
}

/// The selector is untrusted metadata. It locates an independent create-only
/// pin; it never supplies that pin's authority or verifies a checkpoint.
pub(in crate::semantic_transition) fn checkpoint_issuer_selector(
    manifest: &SemanticCheckpointManifest,
) -> PyResult<Option<[u8; 32]>> {
    if manifest.phase_seal.is_empty() {
        return Ok(None);
    }
    let mut reader = seal_reader(&manifest.phase_seal)?;
    reader.take(3 * 32)?;
    Ok(Some(reader.take(32)?.try_into().unwrap()))
}

fn seal_reader(bytes: &[u8]) -> PyResult<RecordReader<'_>> {
    if bytes.len() != SEAL_BYTES || !bytes.starts_with(SEAL_DOMAIN) {
        return Err(invalid(
            "phase checkpoint seal has another domain or extent",
        ));
    }
    Ok(RecordReader(&bytes[SEAL_DOMAIN.len()..]))
}

impl VerifiedClosure {
    pub(super) fn require_checkpoint(
        &self,
        checkpoint: &[u8],
        delivered_checkpoint: &[u8],
        checkpoint_issuer: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if checkpoint == delivered_checkpoint {
            if let Some(issuer) = checkpoint_issuer {
                original_bytes(issuer, 32)?;
            }
            return Ok(());
        }
        let manifest = SemanticCheckpointManifest::decode(checkpoint)?;
        let original = SemanticCheckpointManifest::decode(delivered_checkpoint)?;
        let issuer: [u8; 32] = original_bytes(
            checkpoint_issuer.ok_or_else(|| {
                invalid("descendant checkpoint requires its independently pinned native issuer")
            })?,
            32,
        )?
        .try_into()
        .unwrap();
        let mut reader = seal_reader(&manifest.phase_seal)?;
        if reader.take(32)? != self.phase_id || reader.take(32)? != self.completion_digest {
            return Err(invalid(
                "descendant checkpoint belongs to another known phase completion",
            ));
        }
        let _original_known_checkpoint = reader.take(32)?;
        if reader.take(32)? != issuer || reader.take(32)? != manifest.unsigned_phase_root() {
            return Err(invalid(
                "descendant checkpoint changed its independently pinned issuer or complete state",
            ));
        }
        let signature = ed25519_dalek::Signature::from_slice(reader.take(64)?)
            .map_err(|_| invalid("descendant checkpoint has an invalid signature encoding"))?;
        reader.finish()?;
        VerifyingKey::from_bytes(&issuer)
            .map_err(|_| invalid("independent checkpoint issuer has an invalid encoding"))?
            .verify_strict(&manifest.phase_seal[..SEAL_BYTES - 64], &signature)
            .map_err(|_| invalid("descendant checkpoint failed its independent native issuer"))?;
        let original_phases =
            SemanticTransitionSession::checkpoint_learning_phase_history(&original.native)
                .map_err(xlog_err)?;
        let actual_phases =
            SemanticTransitionSession::checkpoint_learning_phase_history(&manifest.native)
                .map_err(xlog_err)?;
        let (saved, _, _, _) = TaskCheckpointSeed::decode(&original.task)?;
        let (current, _, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        let saved_parent = SemanticTransitionSession::state_material_projection(&original.native)
            .map_err(xlog_err)?;
        let current_parent = SemanticTransitionSession::state_material_projection(&manifest.native)
            .map_err(xlog_err)?;
        let current_expense = current.proposal_expense()?;
        if original_phases != actual_phases
            || saved.training_domain != current.training_domain
            || saved.checkpoint_source_limits()? != current.checkpoint_source_limits()?
            || saved.proposal_expense()?.capacity != current_expense.capacity
            || saved.proposal_expense()?.spent > current_expense.spent
            || saved_parent.model_generation > current_parent.model_generation
            || self.refusal.as_ref().is_some_and(|refusal| {
                current_expense.capacity != refusal.expense.capacity
                    || current_expense.spent < refusal.expense.spent
            })
        {
            return Err(invalid(
                "descendant checkpoint changed its completed phase ancestry or cumulative custody",
            ));
        }
        require_retained_arena_authority(&saved.authority, &current.authority)?;
        Ok(())
    }

    pub(in crate::semantic_transition) fn python_value(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PyTuple>> {
        Ok((
            PyBytes::new(py, &self.phase_id),
            PyBytes::new(py, &self.program),
            PyBytes::new(py, &self.history),
            PyBytes::new(py, &self.acceptance),
        )
            .into_pyobject(py)?
            .unbind())
    }

    pub(in crate::semantic_transition) fn refusal_value(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyTuple>>> {
        self.refusal
            .as_ref()
            .map(|refusal| {
                Ok((
                    refusal.cause.as_ref(),
                    if matches!(
                        refusal.cause.as_ref(),
                        SOURCE_EVALUATION_CANCELLED
                            | PRIVATE_EVALUATION_CANCELLED
                            | PRIVATE_SEGMENT_BUDGET_REFUSED
                            | PRIVATE_SEGMENT_CANCELLED
                    ) {
                        py.None()
                    } else {
                        PyBytes::new(py, &self.acceptance).into_any().unbind()
                    },
                    PyBytes::new(py, &self.history),
                    PyBytes::new(py, &refusal.record),
                )
                    .into_pyobject(py)?
                    .unbind())
            })
            .transpose()
    }

    pub(in crate::semantic_transition) fn restored_expense(
        &self,
        checkpoint: &[u8],
    ) -> PyResult<Option<Arc<Mutex<ProposalExpense>>>> {
        let Some(refusal) = &self.refusal else {
            return Ok(None);
        };
        let manifest = SemanticCheckpointManifest::decode(checkpoint)?;
        let (seed, _, _, _) = TaskCheckpointSeed::decode(&manifest.task)?;
        let mut expense = seed.proposal_expense()?;
        if expense.capacity != refusal.expense.capacity {
            return Err(invalid(
                "terminal Source restore changed its original Proposal capacity",
            ));
        }
        expense.spent = expense.spent.max(refusal.expense.spent);
        Ok(Some(Arc::new(Mutex::new(expense))))
    }
}
