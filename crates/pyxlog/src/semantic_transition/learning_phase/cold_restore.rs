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

#[cfg(feature = "semantic-policy")]
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
        ]),
        ColdValue::Integer(proof.model_work().to_string()),
        ColdValue::Integer(proof.model_calls().to_string()),
    ])
}

pub(in crate::semantic_transition) struct VerifiedClosure {
    pub(in crate::semantic_transition) phase_id: [u8; 32],
    pub(in crate::semantic_transition) completion_digest: [u8; 32],
    pub(in crate::semantic_transition) admission: xlog_cuda::SemanticReplayLoadedMaterial,
    pub(in crate::semantic_transition) program: Arc<[u8]>,
    pub(in crate::semantic_transition) history: Arc<[u8]>,
    pub(in crate::semantic_transition) acceptance: Arc<[u8]>,
    pub(in crate::semantic_transition) snapshot: AuthoritySnapshot,
    pub(in crate::semantic_transition) refusal: Option<RefusalClosure>,
    material_sources: Arc<material_source::MaterialSourceRegistry>,
    _metadata_reservation: xlog_cuda::SemanticReplayMaterialReservation,
}

pub(in crate::semantic_transition) struct RefusalClosure {
    pub(in crate::semantic_transition) cause: Arc<str>,
    pub(in crate::semantic_transition) record: xlog_cuda::SemanticReplayLoadedMaterial,
    pub(in crate::semantic_transition) expense: ProposalExpense,
    pub(super) source_proof: xlog_cuda::SemanticReplayStateMaterialProof,
}

struct BoundCheckpointContext<'a> {
    source: &'a Arc<material_source::MaterialSourceRegistry>,
    original: xlog_cuda::SemanticReplayLoadedMaterial,
    records: [Option<xlog_cuda::SemanticReplayLoadedMaterial>; 3],
    proof: xlog_cuda::SemanticReplayStateMaterialProof,
    native_proofs: Mutex<Option<RetainedNativeProof>>,
}

struct RetainedNativeProof {
    data: Box<NativeProofData>,
    _metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
}

struct NativeProofData {
    original_begin: usize,
    original_extent: usize,
    loaded: xlog_cuda::SemanticReplayLoadedMaterial,
    outcome: Option<
        Result<xlog_cuda::SemanticReplayStateMaterialProof, xlog_cuda::SemanticTransitionError>,
    >,
    next: Option<RetainedNativeProof>,
}

impl NativeProofData {
    fn original_proof(&self) -> PyResult<xlog_cuda::SemanticReplayStateMaterialProof> {
        match self
            .outcome
            .as_ref()
            .ok_or_else(|| invalid("original native checkpoint proof has not completed"))?
        {
            Ok(proof) => Ok(proof.clone()),
            Err(error) => Err(xlog_err(error)),
        }
    }
}

struct BoundPhaseLineage {
    proof: xlog_cuda::SemanticReplayStateMaterialProof,
    start: usize,
}

impl std::ops::Deref for BoundPhaseLineage {
    type Target = [xlog_cuda::SemanticLearningPhaseRecord];

    fn deref(&self) -> &Self::Target {
        &self.proof.learning_phase_history()[self.start..]
    }
}

impl<'a> BoundCheckpointContext<'a> {
    fn new(
        source: &'a Arc<material_source::MaterialSourceRegistry>,
        native: &mut xlog_cuda::SemanticReplayLoadedMaterial,
        records: &[Option<xlog_cuda::SemanticReplayLoadedMaterial>; 3],
    ) -> PyResult<Self> {
        native
            .require_budget(&source.native_budget()?)
            .map_err(xlog_err)?;
        let proof = native.checkpoint_proof().map_err(xlog_err)?;
        let mut retained_records = std::array::from_fn(|_| None);
        for (index, record) in records.iter().enumerate() {
            if let Some(record) = record {
                record
                    .require_budget(&source.native_budget()?)
                    .map_err(xlog_err)?;
                retained_records[index] = Some(
                    record
                        .slice_payload(0..record.as_bytes().len())
                        .map_err(xlog_err)?,
                );
            }
        }
        Ok(Self {
            source,
            original: native
                .slice_payload(0..native.as_bytes().len())
                .map_err(xlog_err)?,
            records: retained_records,
            proof,
            native_proofs: Mutex::new(None),
        })
    }

    fn original_material(&self, bytes: &[u8]) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        let begin = bytes.as_ptr() as usize;
        let end = begin
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("cold closure original slice extent overflow"))?;
        for owner in std::iter::once(&self.original).chain(self.records.iter().flatten()) {
            let original = owner.as_bytes();
            let original_begin = original.as_ptr() as usize;
            let original_end = original_begin
                .checked_add(original.len())
                .ok_or_else(|| invalid("cold closure retained material extent overflow"))?;
            if begin >= original_begin && end <= original_end {
                return owner
                    .slice_payload(begin - original_begin..end - original_begin)
                    .map_err(xlog_err);
            }
        }
        Err(invalid(
            "cold closure slice is not retained by its original loaded materials",
        ))
    }

    fn native_proof(&self, native: &[u8]) -> PyResult<xlog_cuda::SemanticReplayStateMaterialProof> {
        let loaded = self.original_material(native)?;
        loaded
            .require_budget(&self.source.native_budget()?)
            .map_err(xlog_err)?;
        let original_begin = loaded.as_bytes().as_ptr() as usize;
        let original_extent = loaded.as_bytes().len();
        if original_begin == self.original.as_bytes().as_ptr() as usize
            && original_extent == self.original.as_bytes().len()
        {
            return Ok(self.proof.clone());
        }
        let mut cache = self
            .native_proofs
            .lock()
            .map_err(|_| invalid("original native checkpoint proof mutex is poisoned"))?;
        let mut entry = cache.as_ref();
        while let Some(original) = entry {
            if original.data.original_begin == original_begin
                && original.data.original_extent == original_extent
            {
                return original.data.original_proof();
            }
            entry = original.data.next.as_ref();
        }
        let metadata_extent = std::mem::size_of::<NativeProofData>()
            .checked_add(std::mem::size_of::<
                xlog_cuda::SemanticReplayMaterialReservation,
            >())
            .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<usize>()))
            .ok_or_else(|| invalid("original native checkpoint proof metadata extent overflow"))?;
        let metadata = Arc::new(
            self.source
                .native_budget()?
                .reserve_auxiliary(u64::try_from(metadata_extent).map_err(|_| {
                    invalid("original native checkpoint proof metadata exceeds its native index")
                })?)
                .map_err(xlog_err)?,
        );
        // The receipt lives outside its Box, so data and the Box allocation
        // retire first, including when the original decode refuses.
        let mut original = RetainedNativeProof {
            data: Box::new(NativeProofData {
                original_begin,
                original_extent,
                loaded,
                outcome: None,
                next: None,
            }),
            _metadata: metadata,
        };
        original.data.next = cache.take();
        *cache = Some(original);
        let original = &mut cache
            .as_mut()
            .expect("retained native checkpoint proof")
            .data;
        // Retain the first owner before entering its one decoder. Both a proof
        // and a refusal are immutable; neither permits another decode.
        original.outcome = Some(original.loaded.checkpoint_proof());
        original.original_proof()
    }

    fn lineage(
        &self,
        source: &[u8],
        target: &[u8],
        admission: &[u8],
        recipe: &[u8],
        final_phase: Option<SemanticLearningPhase>,
    ) -> PyResult<BoundPhaseLineage> {
        let source = self.native_proof(source)?;
        let target = self.native_proof(target)?;
        require_executed_lineage_records(
            source.learning_phase_history(),
            target.learning_phase_history(),
            admission,
            recipe,
            final_phase,
        )?;
        Ok(BoundPhaseLineage {
            proof: target,
            start: source.learning_phase_history().len(),
        })
    }
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
        ColdValue::ReplayBytes(payload) => {
            let mut fields = Map::new();
            fields.insert("bytes_hex".to_owned(), Value::String(hex(payload.bytes())));
            Value::Object(fields)
        }
        ColdValue::Stored(stored) => Value::Array(vec![
            Value::String("stored".to_owned()),
            Value::Array(stored.store.0.into_iter().map(Value::from).collect()),
            Value::String(stored.digest.clone()),
            Value::from(stored.size),
        ]),
    })
}

fn identity_value(identity: xlog_cuda::SemanticPublishedIdentity) -> Value {
    Value::Array(vec![
        Value::String(hex(identity.instance.as_bytes())),
        Value::from(identity.word),
        Value::String(hex(identity.logical_digest.as_bytes())),
    ])
}

fn identity_material(identity: xlog_cuda::SemanticPublishedIdentity) -> Vec<u8> {
    ColdValue::Sequence(vec![
        ColdValue::Bytes(Arc::from(identity.instance.as_bytes().as_slice())),
        ColdValue::Integer(identity.word.to_string()),
        ColdValue::Bytes(Arc::from(identity.logical_digest.as_bytes().as_slice())),
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
        require_resource_interval(record, capacity)?;
    }
    Ok(())
}

fn require_trace_metadata(trace: &Value, capacity: u64) -> PyResult<u64> {
    const FIELDS: &[&str] = &[
        "state",
        "boundary_flags",
        "requested_coverage",
        "enabled_coverage",
        "unsupported_coverage",
        "process_dropped_records",
        "process_errors",
        "process_buffer_overflows",
        "backing_baseline_bytes",
        "backing_peak_bytes",
    ];
    require_fields(trace, FIELDS)?;
    if FIELDS.iter().any(|field| trace[*field].as_u64().is_none()) {
        return Err(invalid(
            "resource trace status lost its exact original scalar fields",
        ));
    }
    let peak = trace["backing_peak_bytes"]
        .as_u64()
        .expect("checked scalar");
    if trace["state"].as_u64().expect("checked scalar") > 1
        || trace["backing_baseline_bytes"]
            .as_u64()
            .expect("checked scalar")
            > peak
        || peak > capacity
    {
        return Err(invalid(
            "resource trace changed its original state or backing extent",
        ));
    }
    Ok(peak)
}

/// Recovery preserves the signed original clock and process, without replacing
/// them with a fresh observer or interpreting capture handles as activity IDs.
fn require_resource_interval(record: &Value, capacity: u64) -> PyResult<()> {
    let certificate = &record["device_certificate"];
    let interval = &record["resource_interval"];
    require_fields(
        interval,
        &[
            "original_attempt_nonce",
            "operation_ordinal",
            "timestamp_scale",
            "start",
            "end",
            "observer_identity",
            "process_id",
            "device_uuid",
            "cuda_context_handle",
            "cancelled_roster",
            "cancellation",
            "steps",
            "trace",
        ],
    )?;
    let word = |object: &Value, field: &str| {
        object[field]
            .as_u64()
            .ok_or_else(|| invalid("resource interval lost an exact original coordinate"))
    };
    let ordinal = word(interval, "operation_ordinal")?;
    let start = word(interval, "start")?;
    let end = word(interval, "end")?;
    let cancelled = interval["cancelled_roster"]
        .as_bool()
        .ok_or_else(|| invalid("resource interval lost its original cancellation disposition"))?;
    if unhex(&interval["original_attempt_nonce"])?.len() != 32
        || interval["timestamp_scale"] != "activity-producer"
        || interval["observer_identity"] != certificate["observer_identity"]
        || interval["process_id"] != certificate["process_id"]
        || interval["device_uuid"] != certificate["device_uuid"]
        || word(interval, "cuda_context_handle")? == 0
        || start == 0
        || end < start
    {
        return Err(invalid(
            "resource interval changed its original identity, process or clock",
        ));
    }
    let group_peak = require_trace_metadata(&interval["trace"], capacity)?;
    let steps = interval["steps"]
        .as_array()
        .ok_or_else(|| invalid("resource interval lost its original captured roster"))?;
    let mut previous_end = Some(start);
    let mut all_boundaries_known = true;
    for (index, step) in steps.iter().enumerate() {
        require_fields(
            step,
            &[
                "operation_ordinal",
                "capture_stream_handle",
                "start_confirmed",
                "end_attempted",
                "capture_finished",
                "boundaries",
                "trace",
            ],
        )?;
        let flags: Vec<_> = ["start_confirmed", "end_attempted", "capture_finished"]
            .iter()
            .map(|field| {
                step[*field]
                    .as_bool()
                    .ok_or_else(|| invalid("resource step lost its original capture status"))
            })
            .collect::<PyResult<_>>()?;
        if ordinal.checked_add(index as u64) != Some(word(step, "operation_ordinal")?)
            || word(step, "capture_stream_handle")? == 0
            || flags[2] && (!flags[0] || !flags[1])
        {
            return Err(invalid(
                "resource step changed its original capture identity",
            ));
        }
        if step["boundaries"].is_null() {
            if !step["trace"].is_null() {
                return Err(invalid(
                    "unknown step boundaries cannot carry a completed trace",
                ));
            }
            all_boundaries_known = false;
            previous_end = None;
            continue;
        }
        let boundaries = step["boundaries"]
            .as_array()
            .filter(|boundaries| boundaries.len() == 2)
            .ok_or_else(|| invalid("resource step lacks its original boundary pair"))?;
        let step_start = boundaries[0]
            .as_u64()
            .ok_or_else(|| invalid("resource step lost its original start"))?;
        let step_end = boundaries[1]
            .as_u64()
            .ok_or_else(|| invalid("resource step lost its original end"))?;
        if cancelled
            || !flags.iter().all(|flag| *flag)
            || step_start < start
            || step_end < step_start
            || step_end > end
            || previous_end.is_some_and(|previous| previous != step_start)
            || require_trace_metadata(&step["trace"], capacity)? > group_peak
        {
            return Err(invalid(
                "resource step changed its authenticated execution interval",
            ));
        }
        previous_end = Some(step_end);
    }
    if !steps.is_empty() && all_boundaries_known && previous_end != Some(end) {
        return Err(invalid(
            "resource steps lost their complete original group extent",
        ));
    }
    let cancellation = &interval["cancellation"];
    if cancelled {
        require_fields(
            cancellation,
            &[
                "operation_ordinal",
                "planned_count",
                "capture_stream_handle",
                "status",
            ],
        )?;
        let count = word(cancellation, "planned_count")?;
        let stream = word(cancellation, "capture_stream_handle")?;
        if word(cancellation, "operation_ordinal")? != ordinal
            || count == 0
            || count < steps.len() as u64
            || stream == 0
            || word(cancellation, "status")? != 0
            || steps
                .iter()
                .any(|step| step["capture_stream_handle"].as_u64() != Some(stream))
        {
            return Err(invalid(
                "resource cancellation changed its original planned roster",
            ));
        }
    } else if !cancellation.is_null() {
        return Err(invalid(
            "resource interval lacks its original known cancellation handoff",
        ));
    }
    Ok(())
}

fn observation_metadata(record: &Value) -> PyResult<(&Value, &Value, &Value, &Value)> {
    if record["memory_unit"] != resource_observer::MEMORY_UNIT
        || record["memory_admission"] != resource_observer::MEMORY_ADMISSION
    {
        return Err(invalid(
            "terminal observation lost its original memory interpretation",
        ));
    }
    let capacity = require_device_certificate(&record["device_certificate"])?;
    require_resource_interval(record, capacity)?;
    Ok((
        &record["memory_unit"],
        &record["memory_admission"],
        &record["device_certificate"],
        &record["resource_interval"],
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
    let (memory_unit, memory_admission, device_certificate, resource_interval) =
        observation_metadata(actual)?;
    let records = prefix["records"]
        .as_array_mut()
        .ok_or_else(|| invalid("terminal refusal lost its original prefix records"))?;
    let record = serde_json::json!({
        "instruction_sha256": digest(instruction), "operation_ordinal": ordinal,
        "branch": branch, "resource_usage": usage, "refusal": host(reason)?,
        "memory_unit": memory_unit, "memory_admission": memory_admission,
        "device_certificate": device_certificate,
        "resource_interval": resource_interval,
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
    let (memory_unit, memory_admission, device_certificate, resource_interval) =
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
        "resource_interval": resource_interval,
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
        .recipe()
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
pub(super) fn require_executed_lineage_bound(
    source: &xlog_cuda::SemanticReplayStateMaterialProof,
    target: &xlog_cuda::SemanticReplayStateMaterialProof,
    admission: &[u8],
    recipe_bytes: &[u8],
    final_phase: Option<SemanticLearningPhase>,
) -> PyResult<()> {
    require_executed_lineage_records(
        source.learning_phase_history(),
        target.learning_phase_history(),
        admission,
        recipe_bytes,
        final_phase,
    )?;
    Ok(())
}

fn require_executed_lineage_records<'a>(
    original: &[xlog_cuda::SemanticLearningPhaseRecord],
    final_phases: &'a [xlog_cuda::SemanticLearningPhaseRecord],
    admission: &[u8],
    recipe_bytes: &[u8],
    final_phase: Option<SemanticLearningPhase>,
) -> PyResult<&'a [xlog_cuda::SemanticLearningPhaseRecord]> {
    let recipe = ColdValue::from_canonical_bytes(recipe_bytes)?;
    let recipe = recipe.fields(5)?;
    let source_phase = phase(recipe[0].text()?)?;
    let added = final_phases
        .strip_prefix(original)
        .ok_or_else(|| invalid("phase checkpoint lacks its original executed native transition"))?;
    if added
        .iter()
        .any(|record| record.admission().as_bytes() != admission)
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
    Ok(added)
}

fn require_private_expense(
    source: &impl CheckpointManifestReader,
    private: &impl CheckpointManifestReader,
    material_sources: &Arc<material_source::MaterialSourceRegistry>,
) -> PyResult<ProposalExpense> {
    let source = source.borrowed();
    let private = private.borrowed();
    let (source_seed, _, _, _) = TaskCheckpointSeed::decode_bound(&source.task, material_sources)?;
    let (private_seed, _, _, _) =
        TaskCheckpointSeed::decode_bound(&private.task, material_sources)?;
    let expense = private_seed.proposal_expense()?;
    if source.session != private.session
        || source_seed.training_domain != private_seed.training_domain
        || source_seed.replay_capacity != private_seed.replay_capacity
        || source_seed.training_objective != private_seed.training_objective
        || source_seed.original_training_roster()? != private_seed.original_training_roster()?
        || source_seed.checkpoint_source_limits()? != private_seed.checkpoint_source_limits()?
        || source_seed.proposal_expense()?.capacity != expense.capacity
        || source_seed.proposal_expense()?.original_admission != expense.original_admission
        || source_seed.proposal_expense()?.spent > expense.spent
    {
        return Err(invalid(
            "terminal phase changed original Source configuration, task or cumulative expense",
        ));
    }
    require_retained_arena_authority(&source_seed.authority, &private_seed.authority)?;
    Ok(expense)
}

fn closure_metadata_bound(lifecycle: &phase_record::LifecyclePayloads<'_>) -> PyResult<u64> {
    fn add(total: &mut u64, bytes: u64) -> PyResult<()> {
        *total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("cold closure metadata extent overflow"))?;
        Ok(())
    }
    fn canonical(total: &mut u64, bytes: &[u8]) -> PyResult<()> {
        let (_, owned, nodes) = ColdValue::checked_canonical_bytes(bytes, false)?;
        // Original cold decoding, authority projection, Python recipe fields,
        // JSON host projection, and their canonical comparison coexist here.
        let extent = (owned as u64)
            .checked_add(bytes.len() as u64)
            .and_then(|extent| {
                extent.checked_add((nodes as u64).checked_mul(
                    (std::mem::size_of::<ColdValue>() + std::mem::size_of::<Value>()) as u64,
                )?)
            })
            .and_then(|extent| extent.checked_mul(6))
            .ok_or_else(|| invalid("cold closure canonical metadata extent overflow"))?;
        add(total, extent)
    }
    fn json_material(total: &mut u64, bytes: &[u8]) -> PyResult<()> {
        if !bytes.is_empty() {
            // The signed-program check and terminal-target readers use the
            // same JSON core while the original program remains retained.
            let extent = xlog_cuda::SemanticReplayMaterialBudget::json_allocation_bound(bytes)
                .map_err(xlog_err)?
                .checked_mul(3)
                .ok_or_else(|| invalid("cold closure JSON metadata extent overflow"))?;
            add(total, extent)?;
            add(
                total,
                (bytes.len() as u64)
                    .checked_add((2 * std::mem::size_of::<usize>()) as u64)
                    .ok_or_else(|| invalid("cold closure retained JSON extent overflow"))?,
            )?;
        }
        Ok(())
    }
    let mut total =
        (std::mem::size_of::<VerifiedClosure>() + 2 * std::mem::size_of::<usize>()) as u64;
    let mut inputs = RecordReader(
        lifecycle
            .admission
            .strip_prefix(b"xlog.learning-phase.inputs.v2\0")
            .ok_or_else(|| invalid("cold phase admission has another domain"))?,
    );
    inputs.field()?;
    json_material(&mut total, inputs.field()?)?;
    canonical(&mut total, inputs.field()?)?;
    canonical(&mut total, inputs.field()?)?;
    add(
        &mut total,
        (inputs.field()?.len() as u64)
            .checked_mul(2)
            .ok_or_else(|| invalid("cold closure destination extent overflow"))?,
    )?;
    for _ in 0..2 {
        add(
            &mut total,
            (inputs.field()?.len() as u64)
                .checked_mul(4)
                .ok_or_else(|| invalid("cold closure feedback projection extent overflow"))?,
        )?;
    }
    inputs.field()?;
    inputs.finish()?;
    if let Some(preparation) = lifecycle.preparation {
        let mut reader = RecordReader(preparation);
        reader.field()?;
        json_material(&mut total, reader.field()?)?;
        json_material(&mut total, reader.field()?)?;
        reader.finish()?;
    }
    if lifecycle.refused {
        let mut reader = RecordReader(
            lifecycle
                .completion
                .strip_prefix(b"xlog.learning-phase.refusal.v1\0")
                .ok_or_else(|| invalid("terminal phase record has another original domain"))?,
        );
        add(
            &mut total,
            (reader.field()?.len() as u64)
                .checked_add((2 * std::mem::size_of::<usize>()) as u64)
                .ok_or_else(|| invalid("cold closure refusal cause extent overflow"))?,
        )?;
        for _ in 0..2 {
            reader.field()?;
        }
        for _ in 0..3 {
            json_material(&mut total, reader.field()?)?;
        }
        for _ in 0..3 {
            canonical(&mut total, reader.field()?)?;
        }
        reader.field()?;
        reader.finish()?;
    } else {
        let mut reader = RecordReader(
            lifecycle
                .completion
                .strip_prefix(b"xlog.learning-phase.delivery.v1\0")
                .ok_or_else(|| invalid("cold phase delivery has another domain"))?,
        );
        reader.field()?;
        json_material(&mut total, reader.field()?)?;
        canonical(&mut total, reader.field()?)?;
        reader.field()?;
        canonical(&mut total, reader.field()?)?;
        reader.field()?;
        reader.finish()?;
    }
    // Fixed issuer/digest/fence byte copies and exact three-record stack roster.
    add(&mut total, 8 * 32)?;
    Ok(total)
}

impl VerifiedClosure {
    #[expect(
        clippy::too_many_arguments,
        reason = "the original live phase retains its signed chain and independent checkpoint issuer"
    )]
    pub(super) fn read_retained(
        py: Python<'_>,
        checkpoint: &[u8],
        checkpoint_issuer: &[u8; 32],
        original_records: &[Option<xlog_cuda::SemanticReplayLoadedMaterial>; 3],
        record_count: usize,
        issuer: [u8; 32],
        phase_id: [u8; 32],
        limits: RecordLimits,
        source_checkpoint: &[u8],
        target: Option<[u8; 32]>,
        destination: &str,
        source_phase: SemanticLearningPhase,
        target_phase: SemanticLearningPhase,
        material_sources: &Arc<material_source::MaterialSourceRegistry>,
    ) -> PyResult<Self> {
        if !matches!(record_count, 2 | 3) {
            return Err(invalid(
                "live phase closure lost its complete original signed chain",
            ));
        }
        let mut raw = [&[][..]; 3];
        for (index, material) in original_records.iter().take(record_count).enumerate() {
            raw[index] = material
                .as_ref()
                .ok_or_else(|| invalid("live phase closure lost an original signed record"))?
                .as_bytes();
        }
        let lifecycle = lifecycle_payloads(&raw[..record_count], issuer, phase_id, limits)?;
        fn nested_native(
            owner: &xlog_cuda::SemanticReplayLoadedMaterial,
            original: &[u8],
            checkpoint: &[u8],
        ) -> PyResult<Option<xlog_cuda::SemanticReplayLoadedMaterial>> {
            if original != checkpoint {
                return Ok(None);
            }
            let manifest = SemanticCheckpointManifest::decode_borrowed(original)?;
            let begin = (manifest.native.as_ptr() as usize)
                .checked_sub(owner.as_bytes().as_ptr() as usize)
                .ok_or_else(|| {
                    invalid("live phase checkpoint is outside its original signed owner")
                })?;
            let end = begin
                .checked_add(manifest.native.len())
                .filter(|end| *end <= owner.as_bytes().len())
                .ok_or_else(|| {
                    invalid("live phase native slice exceeds its original signed owner")
                })?;
            owner.slice_payload(begin..end).map(Some).map_err(xlog_err)
        }
        let mut admission = RecordReader(
            lifecycle
                .admission
                .strip_prefix(b"xlog.learning-phase.inputs.v2\0")
                .ok_or_else(|| invalid("live phase admission has another original domain"))?,
        );
        let source = admission.field()?;
        let mut native = nested_native(original_records[0].as_ref().unwrap(), source, checkpoint)?;
        if native.is_none() {
            if let Some(preparation) = lifecycle.preparation {
                let prepared = RecordReader(preparation).field()?;
                native =
                    nested_native(original_records[1].as_ref().unwrap(), prepared, checkpoint)?;
            }
        }
        if native.is_none() {
            let domain = if lifecycle.refused {
                &b"xlog.learning-phase.refusal.v1\0"[..]
            } else {
                &b"xlog.learning-phase.delivery.v1\0"[..]
            };
            let mut completion = RecordReader(
                lifecycle
                    .completion
                    .strip_prefix(domain)
                    .ok_or_else(|| invalid("live phase completion has another original domain"))?,
            );
            if lifecycle.refused {
                completion.field()?;
                completion.field()?;
            }
            let completed = completion.field()?;
            native = nested_native(
                original_records[record_count - 1].as_ref().unwrap(),
                completed,
                checkpoint,
            )?;
        }
        let mut native = native.ok_or_else(|| {
            invalid("live phase checkpoint is not retained by its original signed lifecycle")
        })?;
        let budget = material_sources.native_budget()?;
        let fixed_bytes_count = if target.is_some() { 5 } else { 4 };
        let mut metadata = material_bytes::python_instance_extent::<PyBytes>(py)?
            .checked_add(33)
            .and_then(|bytes| bytes.checked_mul(fixed_bytes_count))
            .ok_or_else(|| invalid("live phase fixed-byte metadata extent overflow"))?;
        metadata = metadata
            .checked_add(
                material_bytes::python_instance_extent::<PyTuple>(py)?
                    .checked_mul(3)
                    .and_then(|bytes| {
                        bytes.checked_add((record_count + 9) * std::mem::size_of::<usize>())
                    })
                    .ok_or_else(|| invalid("live phase tuple metadata extent overflow"))?,
            )
            .ok_or_else(|| invalid("live phase metadata extent overflow"))?;
        let integer_type = <pyo3::types::PyInt as pyo3::PyTypeInfo>::type_object_raw(py);
        // Every limb stores at least one bit. This derives a finite u64
        // instance bound from the actual interpreter's integer item extent.
        let integer_item = usize::try_from(unsafe { (*integer_type).tp_itemsize })
            .map_err(|_| invalid("live phase integer type has an invalid item extent"))?;
        metadata =
            metadata
                .checked_add(
                    material_bytes::python_instance_extent::<pyo3::types::PyInt>(py)?
                        .checked_add((u64::BITS as usize).checked_mul(integer_item).ok_or_else(
                            || invalid("live phase integer metadata extent overflow"),
                        )?)
                        .and_then(|bytes| bytes.checked_mul(3))
                        .ok_or_else(|| invalid("live phase limit metadata extent overflow"))?,
                )
                .ok_or_else(|| invalid("live phase metadata extent overflow"))?;
        for text in [
            destination,
            phase_name(source_phase),
            phase_name(target_phase),
        ] {
            metadata = metadata
                .checked_add(
                    material_bytes::python_instance_extent::<PyString>(py)?
                        .checked_add(
                            text.len()
                                .checked_mul(4)
                                .and_then(|bytes| bytes.checked_add(4))
                                .ok_or_else(|| {
                                    invalid("live phase text metadata extent overflow")
                                })?,
                        )
                        .ok_or_else(|| invalid("live phase text instance extent overflow"))?,
                )
                .ok_or_else(|| invalid("live phase metadata extent overflow"))?;
        }
        let _metadata = material_bytes::reserve_material_metadata(&budget, metadata)?;
        let mut wrappers: [Option<Py<material_bytes::PySemanticMaterialBytes>>; 3] =
            std::array::from_fn(|_| None);
        for (index, original) in original_records.iter().take(record_count).enumerate() {
            let original = original.as_ref().unwrap();
            wrappers[index] = Some(Py::new(
                py,
                material_bytes::PySemanticMaterialBytes::from_native(
                    py,
                    original
                        .slice_payload(0..original.as_bytes().len())
                        .map_err(xlog_err)?,
                    &budget,
                )?,
            )?);
        }
        let records = PyTuple::new(
            py,
            wrappers
                .iter()
                .take(record_count)
                .map(|value| value.as_ref().unwrap().clone_ref(py)),
        )?;
        let issuer = PyBytes::new(py, &issuer);
        let checkpoint_issuer = PyBytes::new(py, checkpoint_issuer);
        let limits = (limits.record_bytes, limits.total_bytes, limits.records).into_pyobject(py)?;
        let target = target
            .map(|digest| PyBytes::new(py, &digest).into_any().unbind())
            .unwrap_or_else(|| py.None());
        let fence = (
            PyBytes::new(py, &phase_id),
            PyBytes::new(py, &Sha256::digest(source_checkpoint)),
            target,
            destination,
            phase_name(source_phase),
            phase_name(target_phase),
        )
            .into_pyobject(py)?;
        Self::read(
            checkpoint,
            issuer.as_any(),
            records.as_any(),
            limits.as_any(),
            fence.as_any(),
            Some(checkpoint_issuer.as_any()),
            material_sources,
            &mut native,
        )
    }

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
        context: &BoundCheckpointContext<'_>,
        metadata_reservation: xlog_cuda::SemanticReplayMaterialReservation,
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
            let manifest = SemanticCheckpointManifest::decode_borrowed(source)?;
            let (seed, saved_snapshot, _, _) =
                TaskCheckpointSeed::decode_bound(&manifest.task, context.source)?;
            let mut expense = seed.proposal_expense()?;
            let source_proof = context.native_proof(manifest.native)?;
            let source_parent = source_proof.original_identity();
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
                let private_manifest = SemanticCheckpointManifest::decode_borrowed(private)?;
                let (_, private_snapshot, _, _) =
                    TaskCheckpointSeed::decode_bound(&private_manifest.task, context.source)?;
                expense = require_private_expense(&manifest, &private_manifest, context.source)?;
                snapshot.newer_than(&private_snapshot)?;
                // Initial trajectory construction is an exact Source restore.
                // Only executed cold restores may add a phase. No transition
                // yet is genuine absence, never a fabricated first recipe.
                let added = context.lineage(
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
                context
                    .native_proof(private_manifest.native)?
                    .original_identity()
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
                admission: context.original_material(admission)?,
                program: Arc::from(program_bytes),
                history: Arc::from(history_bytes),
                acceptance: Arc::from(&[][..]),
                snapshot,
                refusal: Some(RefusalClosure {
                    cause: Arc::from(cause),
                    record: context.original_material(terminal_record)?,
                    expense,
                    source_proof,
                }),
                material_sources: context.source.clone(),
                _metadata_reservation: metadata_reservation,
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
        let source_manifest = SemanticCheckpointManifest::decode_borrowed(source)?;
        let private_manifest = SemanticCheckpointManifest::decode_borrowed(private)?;
        let (_, source_snapshot, _, _) =
            TaskCheckpointSeed::decode_bound(&source_manifest.task, context.source)?;
        let (_, private_snapshot, _, _) =
            TaskCheckpointSeed::decode_bound(&private_manifest.task, context.source)?;
        let expense = require_private_expense(&source_manifest, &private_manifest, context.source)?;
        let program = json(program_bytes)?;
        let declared_phase =
            phase(program["final_phase"].as_str().ok_or_else(|| {
                invalid("terminal phase lost its original declared final phase")
            })?)?;
        let phases = context.lineage(
            &source_manifest.native,
            &private_manifest.native,
            admission,
            recipe_bytes,
            Some(declared_phase),
        )?;
        let source_proof = context.native_proof(source_manifest.native)?;
        let source_parent = source_proof.original_identity();
        let private_parent = context
            .native_proof(private_manifest.native)?
            .original_identity();
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
            admission: context.original_material(admission)?,
            program: Arc::from(program_bytes),
            history: Arc::from(history_bytes),
            acceptance: Arc::from(result_bytes),
            snapshot,
            refusal: Some(RefusalClosure {
                cause: Arc::from(cause),
                record: context.original_material(terminal_record)?,
                expense,
                source_proof,
            }),
            material_sources: context.source.clone(),
            _metadata_reservation: metadata_reservation,
        })
    }

    pub(in crate::semantic_transition) fn read(
        checkpoint: &[u8],
        issuer: &Bound<'_, PyAny>,
        records: &Bound<'_, PyAny>,
        limits: &Bound<'_, PyAny>,
        fence: &Bound<'_, PyAny>,
        checkpoint_issuer: Option<&Bound<'_, PyAny>>,
        material_sources: &Arc<material_source::MaterialSourceRegistry>,
        native: &mut xlog_cuda::SemanticReplayLoadedMaterial,
    ) -> PyResult<Self> {
        let py = issuer.py();
        let manifest = SemanticCheckpointManifest::decode_borrowed(checkpoint)?;
        if native.as_bytes() != manifest.native {
            return Err(invalid(
                "cold closure requires this original loaded checkpoint material",
            ));
        }
        if !issuer.is_exact_instance_of::<PyBytes>() {
            return Err(invalid(
                "cold phase issuer requires its exact original bytes",
            ));
        }
        let issuer: [u8; 32] = issuer
            .cast::<PyBytes>()?
            .as_bytes()
            .try_into()
            .map_err(|_| invalid("cold phase issuer requires its original 32-byte extent"))?;
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
        let phase_identifier = fence.get_item(0)?;
        if !phase_identifier.is_exact_instance_of::<PyBytes>() {
            return Err(invalid(
                "cold phase identifier requires its exact original bytes",
            ));
        }
        let phase_id: [u8; 32] = phase_identifier
            .cast::<PyBytes>()?
            .as_bytes()
            .try_into()
            .map_err(|_| invalid("cold phase identifier requires its original 32-byte extent"))?;
        let budget = material_sources.native_budget()?;
        let mut retained: [Option<xlog_cuda::SemanticReplayLoadedMaterial>; 3] =
            std::array::from_fn(|_| None);
        for (index, value) in records.iter().enumerate() {
            if !value.is_exact_instance_of::<material_bytes::PySemanticMaterialBytes>() {
                return Err(invalid(
                    "cold phase records require their original guarded byte materials",
                ));
            }
            retained[index] = Some(
                value
                    .cast::<material_bytes::PySemanticMaterialBytes>()?
                    .borrow()
                    .original_loaded(&budget)?,
            );
        }
        let mut raw = [&[][..]; 3];
        for (index, value) in retained.iter().take(records.len()).enumerate() {
            let value = value.as_ref().expect("original record tuple extent");
            raw[index] = value.as_bytes();
        }
        let raw = &raw[..records.len()];
        let context = BoundCheckpointContext::new(material_sources, native, &retained)?;
        let lifecycle = lifecycle_payloads(raw, issuer, phase_id, limits)?;
        let metadata_reservation = material_sources
            .native_budget()?
            .reserve_auxiliary(closure_metadata_bound(&lifecycle)?)
            .map_err(xlog_err)?;
        let source_digest = original_bytes(&fence.get_item(1)?, 32)?;
        let target = fence.get_item(2)?;
        let destination = fence.get_item(3)?;
        let source_phase = fence.get_item(4)?;
        let target_phase = fence.get_item(5)?;
        if [&destination, &source_phase, &target_phase]
            .iter()
            .any(|value| !value.is_exact_instance_of::<PyString>())
        {
            return Err(invalid(
                "cold phase fence requires its exact original text fields",
            ));
        }
        let destination = destination.cast::<PyString>()?.to_str()?;
        let source_phase = phase(source_phase.cast::<PyString>()?.to_str()?)?;
        let target_phase = phase(target_phase.cast::<PyString>()?.to_str()?)?;
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
        if reader.field()? != destination.as_bytes() {
            return Err(invalid("cold phase chain changed its original destination"));
        }
        let feedback = [reader.field()?, reader.field()?];
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
                destination,
                source_phase,
                target_phase,
                pending_target,
                py,
                &context,
                metadata_reservation,
            )?;
            closure.require_checkpoint_bound(checkpoint, source, checkpoint_issuer, &context)?;
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
        let source_manifest = SemanticCheckpointManifest::decode_borrowed(source)?;
        let final_manifest = SemanticCheckpointManifest::decode_borrowed(delivered_checkpoint)?;
        let (source_seed, _, _, _) =
            TaskCheckpointSeed::decode_bound(&source_manifest.task, material_sources)?;
        let (final_seed, final_snapshot, _, _) =
            TaskCheckpointSeed::decode_bound(&final_manifest.task, material_sources)?;
        if source_seed.training_domain != final_seed.training_domain
            || source_seed.replay_capacity != final_seed.replay_capacity
            || source_seed.training_objective != final_seed.training_objective
            || source_seed.original_training_roster()? != final_seed.original_training_roster()?
            || source_seed.checkpoint_source_limits()? != final_seed.checkpoint_source_limits()?
            || source_seed.proposal_expense()?.capacity != final_seed.proposal_expense()?.capacity
            || source_seed.proposal_expense()?.original_admission
                != final_seed.proposal_expense()?.original_admission
            || source_seed.proposal_expense()?.spent > final_seed.proposal_expense()?.spent
        {
            return Err(invalid(
                "cold phase delivery changed original task resource or training custody",
            ));
        }
        require_retained_arena_authority(&source_seed.authority, &final_seed.authority)?;
        let source_parent = context
            .native_proof(source_manifest.native)?
            .original_identity();
        let final_parent = context
            .native_proof(final_manifest.native)?
            .original_identity();
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
        let added = context.lineage(
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
            || program["destination"] != destination
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
            admission: context.original_material(raw[0])?,
            program: Arc::from(program_bytes),
            history: Arc::from(history_bytes),
            acceptance: Arc::from(acceptance_bytes),
            snapshot,
            refusal: None,
            material_sources: material_sources.clone(),
            _metadata_reservation: metadata_reservation,
        };
        closure.require_checkpoint_bound(
            checkpoint,
            delivered_checkpoint,
            checkpoint_issuer,
            &context,
        )?;
        Ok(closure)
    }
}

/// Reuse the canonical restore trust boundary without allocating a native Session.
/// An absent closure still requires the Session signature, not phase completion.
pub(in crate::semantic_transition) fn read_checkpoint_restore_closure_bound(
    checkpoint: &[u8],
    snapshot: Option<&Bound<'_, PyAny>>,
    issuer: Option<&Bound<'_, PyAny>>,
    records: Option<&Bound<'_, PyAny>>,
    limits: Option<&Bound<'_, PyAny>>,
    fence: Option<&Bound<'_, PyAny>>,
    checkpoint_issuer: Option<&Bound<'_, PyAny>>,
    material_sources: &Arc<material_source::MaterialSourceRegistry>,
    native: &mut xlog_cuda::SemanticReplayLoadedMaterial,
) -> PyResult<Option<Arc<VerifiedClosure>>> {
    let manifest = SemanticCheckpointManifest::decode_borrowed(checkpoint)?;
    native
        .require_budget(&material_sources.native_budget()?)
        .map_err(xlog_err)?;
    if native.as_bytes() != manifest.native {
        return Err(invalid(
            "cold closure requires this original loaded checkpoint material",
        ));
    }
    let closure = match (issuer, records, limits, fence) {
        (None, None, None, None) => {
            let manifest = SemanticCheckpointManifest::decode_borrowed(checkpoint)?;
            let checkpoint_issuer = checkpoint_issuer.ok_or_else(|| {
                invalid("checkpoint restoration requires its independently pinned Session issuer")
            })?;
            if verify_checkpoint_signature(&manifest, checkpoint_issuer)?
                || !native
                    .checkpoint_proof()
                    .map_err(xlog_err)?
                    .learning_phase_history()
                    .is_empty()
            {
                return Err(invalid(
                    "phase-marked checkpoint requires its original complete signed lifecycle",
                ));
            }
            None
        }
        (Some(issuer), Some(records), Some(limits), Some(fence)) => {
            let verified = VerifiedClosure::read(
                checkpoint,
                issuer,
                records,
                limits,
                fence,
                checkpoint_issuer,
                material_sources,
                native,
            )?;
            let snapshot = snapshot.ok_or_else(|| {
                invalid("cold phase restoration requires its current authority snapshot")
            })?;
            let budget = material_sources.native_budget()?;
            let _preflight_reservation = budget
                .reserve_auxiliary(
                    (std::mem::size_of::<pyo3::ffi::PyVarObject>()
                        + 2 * std::mem::size_of::<usize>()
                        + std::mem::size_of::<Bound<'_, PyAny>>()) as u64,
                )
                .map_err(xlog_err)?;
            let extent = ColdValue::read_allocation_bound(snapshot, &mut (16 * 1024 * 1024), 0)?;
            let _snapshot_reservation =
                budget
                    .reserve_auxiliary(extent.checked_mul(3).ok_or_else(|| {
                        invalid("cold authority snapshot metadata extent overflow")
                    })?)
                    .map_err(xlog_err)?;
            let current =
                AuthoritySnapshot::parse(&ColdValue::read(snapshot, &mut (16 * 1024 * 1024), 0)?)?;
            current.newer_than(&verified.snapshot)?;
            Some(Arc::new(verified))
        }
        _ => {
            return Err(invalid(
                "cold phase restoration requires all original proof fields together",
            ))
        }
    };
    Ok(closure)
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

const SEAL_DOMAIN: &[u8] = b"xlog.semantic.checkpoint.v2\0";

struct CheckpointPhaseSeal {
    phase_id: [u8; 32],
    completion_digest: [u8; 32],
}

struct CheckpointSeal {
    issuer: [u8; 32],
    root: [u8; 32],
    phase: Option<CheckpointPhaseSeal>,
    signature: ed25519_dalek::Signature,
}

/// This key belongs to the actual known Session, never to Python or a saved Q.
/// The trusted consumer pins its public key independently before subsequent
/// requests. A fresh known recovery gets one new issuer, not the old secret.
pub(in crate::semantic_transition) struct CheckpointCustody {
    pub(in crate::semantic_transition) closure: Option<Arc<VerifiedClosure>>,
    pub(in crate::semantic_transition) origin: Option<[u8; 32]>,
    key: SigningKey,
    pub(in crate::semantic_transition) pending_export:
        Option<Arc<Mutex<material_bytes::OriginalCheckpointExport>>>,
}

impl CheckpointCustody {
    pub(in crate::semantic_transition) fn issue() -> Self {
        Self {
            closure: None,
            origin: None,
            key: SigningKey::generate(&mut OsRng),
            pending_export: None,
        }
    }

    pub(in crate::semantic_transition) fn issuer(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    pub(in crate::semantic_transition) fn seal_extent(&self) -> usize {
        SEAL_DOMAIN.len() + 2 * 32 + 8 + if self.closure.is_some() { 3 * 32 } else { 0 } + 64
    }

    pub(in crate::semantic_transition) fn seal(
        &self,
        manifest: &impl CheckpointManifestReader,
        phases: &[xlog_cuda::SemanticLearningPhaseRecord],
    ) -> PyResult<Vec<u8>> {
        let manifest = manifest.borrowed();
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.seal_extent())
            .map_err(|_| invalid("checkpoint signing buffer allocation failed"))?;
        bytes.extend_from_slice(SEAL_DOMAIN);
        for field in [self.issuer(), manifest.unsigned_phase_root()] {
            bytes.extend_from_slice(&field);
        }
        if let Some(closure) = &self.closure {
            let valid = if let Some(refusal) = &closure.refusal {
                let original = refusal.source_proof.learning_phase_history();
                let (seed, _, _, _) =
                    TaskCheckpointSeed::decode_bound(manifest.task, &closure.material_sources)?;
                let expense = seed.proposal_expense()?;
                phases == original
                    && expense.capacity == refusal.expense.capacity
                    && expense.original_admission == refusal.expense.original_admission
                    && expense.spent >= refusal.expense.spent
            } else {
                phases
                    .last()
                    .is_some_and(|phase| phase.admission().as_bytes() == closure.admission.as_ref())
            };
            if !valid {
                return Err(invalid(
                    "checkpoint signing lost this Session's known native lineage or cumulative expense",
                ));
            }
            bytes.extend_from_slice(&1u64.to_le_bytes());
            for field in [
                closure.phase_id,
                closure.completion_digest,
                self.origin
                    .ok_or_else(|| invalid("known checkpoint phase lost its original source"))?,
            ] {
                bytes.extend_from_slice(&field);
            }
        } else {
            // A Session signature does not claim that a private phase completed.
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
        bytes.extend_from_slice(&self.key.sign(&bytes).to_bytes());
        Ok(bytes)
    }
}

/// The selector is untrusted metadata. It locates an independent create-only
/// pin; it never supplies that pin's authority or verifies a checkpoint.
pub(in crate::semantic_transition) fn checkpoint_issuer_selector(
    manifest: &impl CheckpointManifestReader,
) -> PyResult<Option<[u8; 32]>> {
    let manifest = manifest.borrowed();
    if manifest.phase_seal.is_empty() {
        return Ok(None);
    }
    Ok(Some(checkpoint_seal(&manifest.phase_seal)?.issuer))
}

fn checkpoint_seal(bytes: &[u8]) -> PyResult<CheckpointSeal> {
    if !bytes.starts_with(SEAL_DOMAIN) {
        return Err(invalid("checkpoint seal has another domain or version"));
    }
    let mut reader = RecordReader(&bytes[SEAL_DOMAIN.len()..]);
    let issuer = reader.take(32)?.try_into().unwrap();
    let root = reader.take(32)?.try_into().unwrap();
    let phase = match reader.word()? {
        0 => None,
        1 => {
            let phase_id = reader.take(32)?.try_into().unwrap();
            let completion_digest = reader.take(32)?.try_into().unwrap();
            // The signature covers the origin; the retained restore owner
            // authenticates the exact source material when it is consumed.
            reader.take(32)?;
            Some(CheckpointPhaseSeal {
                phase_id,
                completion_digest,
            })
        }
        _ => return Err(invalid("checkpoint seal has another ancestry tag")),
    };
    let signature = ed25519_dalek::Signature::from_slice(reader.take(64)?)
        .map_err(|_| invalid("checkpoint has an invalid signature encoding"))?;
    reader.finish()?;
    Ok(CheckpointSeal {
        issuer,
        root,
        phase,
        signature,
    })
}

fn authenticated_checkpoint_seal(
    manifest: &impl CheckpointManifestReader,
    checkpoint_issuer: &Bound<'_, PyAny>,
) -> PyResult<CheckpointSeal> {
    let manifest = manifest.borrowed();
    let issuer: [u8; 32] = original_bytes(checkpoint_issuer, 32)?.try_into().unwrap();
    let seal = checkpoint_seal(&manifest.phase_seal)?;
    if seal.issuer != issuer || seal.root != manifest.unsigned_phase_root() {
        return Err(invalid(
            "checkpoint changed its independently pinned issuer or complete state",
        ));
    }
    VerifyingKey::from_bytes(&issuer)
        .map_err(|_| invalid("independent checkpoint issuer has an invalid encoding"))?
        .verify_strict(
            &manifest.phase_seal[..manifest.phase_seal.len() - 64],
            &seal.signature,
        )
        .map_err(|_| invalid("checkpoint failed its independent native issuer"))?;
    Ok(seal)
}

/// Verify the one canonical Session signature against an independently supplied pin.
/// The returned phase tag is metadata, not proof of lifecycle completion.
pub(in crate::semantic_transition) fn verify_checkpoint_signature(
    manifest: &impl CheckpointManifestReader,
    checkpoint_issuer: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    Ok(authenticated_checkpoint_seal(manifest, checkpoint_issuer)?
        .phase
        .is_some())
}

impl VerifiedClosure {
    fn require_checkpoint_bound(
        &self,
        checkpoint: &[u8],
        delivered_checkpoint: &[u8],
        checkpoint_issuer: Option<&Bound<'_, PyAny>>,
        context: &BoundCheckpointContext<'_>,
    ) -> PyResult<()> {
        let manifest = SemanticCheckpointManifest::decode_borrowed(checkpoint)?;
        let seal = authenticated_checkpoint_seal(
            &manifest,
            checkpoint_issuer.ok_or_else(|| {
                invalid("checkpoint requires its independently pinned native issuer")
            })?,
        )?;
        if checkpoint == delivered_checkpoint {
            return Ok(());
        }
        let original = SemanticCheckpointManifest::decode_borrowed(delivered_checkpoint)?;
        let phase = seal.phase.ok_or_else(|| {
            invalid("descendant checkpoint lacks its original known phase completion")
        })?;
        if phase.phase_id != self.phase_id || phase.completion_digest != self.completion_digest {
            return Err(invalid(
                "descendant checkpoint belongs to another known phase completion",
            ));
        }
        let saved_parent = context.native_proof(original.native)?;
        let current_parent = context.native_proof(manifest.native)?;
        let original_phases = saved_parent.learning_phase_history();
        let actual_phases = current_parent.learning_phase_history();
        let (saved, _, _, _) =
            TaskCheckpointSeed::decode_bound(&original.task, &self.material_sources)?;
        let (current, _, _, _) =
            TaskCheckpointSeed::decode_bound(&manifest.task, &self.material_sources)?;
        let current_expense = current.proposal_expense()?;
        if original_phases != actual_phases
            || saved.training_domain != current.training_domain
            || saved.replay_capacity != current.replay_capacity
            || saved.checkpoint_source_limits()? != current.checkpoint_source_limits()?
            || saved.proposal_expense()?.capacity != current_expense.capacity
            || saved.proposal_expense()?.original_admission != current_expense.original_admission
            || saved.proposal_expense()?.spent > current_expense.spent
            || saved_parent.model_generation() > current_parent.model_generation()
            || self.refusal.as_ref().is_some_and(|refusal| {
                current_expense.capacity != refusal.expense.capacity
                    || current_expense.original_admission != refusal.expense.original_admission
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
                    material_bytes::PySemanticMaterialBytes::from_native(
                        py,
                        refusal
                            .record
                            .slice_payload(0..refusal.record.as_bytes().len())
                            .map_err(xlog_err)?,
                        &self.material_sources.native_budget()?,
                    )?,
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
        let manifest = SemanticCheckpointManifest::decode_borrowed(checkpoint)?;
        let (seed, _, _, _) =
            TaskCheckpointSeed::decode_bound(&manifest.task, &self.material_sources)?;
        let mut expense = seed.proposal_expense()?;
        if expense.capacity != refusal.expense.capacity
            || expense.original_admission != refusal.expense.original_admission
        {
            return Err(invalid(
                "terminal Source restore changed its original Proposal capacity",
            ));
        }
        expense.spent = expense.spent.max(refusal.expense.spent);
        Ok(Some(Arc::new(Mutex::new(expense))))
    }
}
