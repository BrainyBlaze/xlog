//! Authenticate original cold phase custody in the canonical checkpoint loader.
//! Scientific comparisons are not re-run: their original native-signed result,
//! frozen program and complete delivered history must remain unchanged.

use super::*;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use phase_record::{delivery_payloads, RecordReader};
use rand::rngs::OsRng;
use serde_json::{Map, Value};

pub(in crate::semantic_transition) struct VerifiedClosure {
    pub(in crate::semantic_transition) phase_id: [u8; 32],
    pub(in crate::semantic_transition) delivery_digest: [u8; 32],
    pub(in crate::semantic_transition) admission: Arc<[u8]>,
    pub(in crate::semantic_transition) program: Arc<[u8]>,
    pub(in crate::semantic_transition) history: Arc<[u8]>,
    pub(in crate::semantic_transition) acceptance: Arc<[u8]>,
    pub(in crate::semantic_transition) snapshot: AuthoritySnapshot,
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
    reason: &ColdValue,
    usage: [u64; 3],
) -> PyResult<()> {
    let mut prefix = json(prefix)?;
    let history = json(history)?;
    let records = prefix["records"]
        .as_array_mut()
        .ok_or_else(|| invalid("terminal refusal lost its original prefix records"))?;
    let record = serde_json::json!({
        "instruction_sha256": digest(instruction), "operation_ordinal": ordinal,
        "branch": "real", "resource_usage": usage, "refusal": host(reason)?,
    });
    records.push(record);
    if prefix != history {
        return Err(invalid(
            "terminal refusal changed its original prefix or actual final observation",
        ));
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

/// Both the live producer and cold recovery must prove the actual complete
/// native transition, not infer it from a frozen recipe or phase marker.
pub(super) fn require_executed_lineage(
    source_native: &[u8],
    final_native: &[u8],
    admission: &[u8],
    recipe_bytes: &[u8],
    final_phase: SemanticLearningPhase,
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
        .filter(|phases| !phases.is_empty())
        .ok_or_else(|| invalid("phase checkpoint lacks its original executed native transition"))?;
    if added
        .iter()
        .any(|record| record.admission.as_slice() != admission)
        || added[0].source != source_phase
        || added.last().expect("nonempty native phase lineage").target != final_phase
        || recipe_material(&added[0]) != recipe_bytes
    {
        return Err(invalid(
            "phase checkpoint differs from its original executed recipe or signed admission",
        ));
    }
    Ok(added.to_vec())
}

impl VerifiedClosure {
    pub(in crate::semantic_transition) fn read(
        checkpoint: &[u8],
        issuer: &Bound<'_, PyAny>,
        records: &Bound<'_, PyAny>,
        limits: &Bound<'_, PyAny>,
        fence: &Bound<'_, PyAny>,
        checkpoint_issuer: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let issuer: [u8; 32] = original_bytes(issuer, 32)?.try_into().unwrap();
        if !records.is_exact_instance_of::<PyTuple>() || !fence.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "cold phase recovery requires the original ordered tuple and fence",
            ));
        }
        let records = records.cast::<PyTuple>()?;
        let fence = fence.cast::<PyTuple>()?;
        if records.len() != 3 || fence.len() != 6 || !limits.is_exact_instance_of::<PyTuple>() {
            return Err(invalid(
                "cold phase recovery requires all three original records and exact limits",
            ));
        }
        let limits = RecordLimits::read(limits)?;
        let phase_id: [u8; 32] = original_bytes(&fence.get_item(0)?, 32)?.try_into().unwrap();
        let source_digest = original_bytes(&fence.get_item(1)?, 32)?;
        let target_digest = original_bytes(&fence.get_item(2)?, 32)?;
        let destination = ColdValue::read(&fence.get_item(3)?, &mut (16 * 1024 * 1024), 0)?;
        let source_phase = ColdValue::read(&fence.get_item(4)?, &mut 128, 0)?;
        let target_phase = ColdValue::read(&fence.get_item(5)?, &mut 128, 0)?;
        let source_phase = phase(source_phase.text()?)?;
        let target_phase = phase(target_phase.text()?)?;
        let retained = [
            records.get_item(0)?,
            records.get_item(1)?,
            records.get_item(2)?,
        ];
        let mut raw = [&[][..]; 3];
        for (index, value) in retained.iter().enumerate() {
            if !value.is_exact_instance_of::<PyBytes>() {
                return Err(invalid(
                    "cold phase records require exact original byte materials",
                ));
            }
            raw[index] = value.cast::<PyBytes>()?.as_bytes();
        }
        let [admission, preparation, delivery] = delivery_payloads(raw, issuer, phase_id, limits)?;
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
            declared_phase,
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
        require_fields(&prefix, &["format", "program_sha256", "records"])?;
        require_fields(&history, &["format", "program_sha256", "records"])?;
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
            let instruction = unhex(&entry["instruction"]["bytes_hex"])?;
            let budget = entry["budget"]
                .as_array()
                .filter(|values| values.len() == 3)
                .ok_or_else(|| {
                    invalid("signed phase operation lost its original whole-resource budget")
                })?;
            let expenditure = record["resource_usage"]
                .as_array()
                .filter(|values| values.len() == 3)
                .ok_or_else(|| {
                    invalid("signed phase operation lost its actual resource expenditure")
                })?;
            if record["operation_ordinal"].as_u64() != Some(ordinal as u64)
                || record["branch"] != entry["branch"]
                || record["instruction_sha256"] != digest(&instruction)
                || record.as_object().is_none()
                || record
                    .get("refusal")
                    .is_some_and(|refusal| !refusal.is_null())
            {
                return Err(invalid(
                    "signed phase history contains a refusal or changed original operation",
                ));
            }
            if matches!(
                entry["operation"].as_str(),
                Some("proposal" | "update" | "recompute")
            ) && (record["step"]["skipped"] != false
                || !record["step"]["refusal"].is_null()
                || record["step"]["step_ordinal"] != entry["step_ordinal"]
                || record["step"]["transition"] != entry["operation"])
            {
                return Err(invalid("signed phase history contains a skipped, refused or substituted numerical step"));
            }
            for (index, (spent, limit)) in expenditure.iter().zip(budget).enumerate() {
                let spent = spent
                    .as_u64()
                    .ok_or_else(|| invalid("signed phase expenditure is not an exact u64"))?;
                let limit = limit
                    .as_u64()
                    .ok_or_else(|| invalid("signed phase budget is not an exact u64"))?;
                if spent > limit || index == 1 && limit == 0 {
                    return Err(invalid(
                        "signed phase history exceeded its original operation budget",
                    ));
                }
            }
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
            delivery_digest: Sha256::digest(raw[2]).into(),
            admission: Arc::from(raw[0]),
            program: Arc::from(program_bytes),
            history: Arc::from(history_bytes),
            acceptance: Arc::from(acceptance_bytes),
            snapshot,
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
        if phases
            .last()
            .is_none_or(|phase| phase.admission.as_slice() != self.closure.admission.as_ref())
        {
            return Err(invalid(
                "checkpoint signing lost this Session's completed native phase lineage",
            ));
        }
        let mut bytes = SEAL_DOMAIN.to_vec();
        for field in [
            self.closure.phase_id,
            self.closure.delivery_digest,
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
        if reader.take(32)? != self.phase_id || reader.take(32)? != self.delivery_digest {
            return Err(invalid(
                "descendant checkpoint belongs to another completed phase or Delivery",
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
        if original_phases != actual_phases
            || saved.training_domain != current.training_domain
            || saved.checkpoint_source_limits()? != current.checkpoint_source_limits()?
            || saved.proposal_expense()?.capacity != current.proposal_expense()?.capacity
            || saved.proposal_expense()?.spent > current.proposal_expense()?.spent
            || saved_parent.model_generation > current_parent.model_generation
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
}
