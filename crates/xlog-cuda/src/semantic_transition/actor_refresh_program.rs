use super::*;
use serde_json::Value;
use serde::Deserialize;

const PROGRAM_BYTES: usize = 16 * 1024 * 1024;

struct UniqueJson(Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueJson;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("bounded JSON without duplicate object fields")
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> { Ok(UniqueJson(Value::Null)) }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> { Ok(UniqueJson(value.into())) }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> { Ok(UniqueJson(value.into())) }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> { Ok(UniqueJson(value.into())) }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value).map(|value| UniqueJson(Value::Number(value)))
                    .ok_or_else(|| E::custom("nonfinite JSON number"))
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> { Ok(UniqueJson(value.into())) }
            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> { Ok(UniqueJson(value.into())) }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(UniqueJson(value)) = sequence.next_element()? { values.push(value); }
                Ok(UniqueJson(Value::Array(values)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut fields: A) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = fields.next_key::<String>()? {
                    if values.contains_key(&key) { return Err(serde::de::Error::custom("duplicate JSON field")); }
                    let UniqueJson(value) = fields.next_value()?;
                    values.insert(key, value);
                }
                Ok(UniqueJson(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

fn json_value(bytes: &[u8]) -> Result<Value, SemanticTransitionError> {
    if bytes.is_empty() || bytes.len() > PROGRAM_BYTES { return Err(invalid_program()); }
    // Keep serde's original finite recursion limit, in addition to the ingress
    // byte bound. Exact original bytes, not this comparison encoding, identify
    // the admitted program.
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let UniqueJson(value) = UniqueJson::deserialize(&mut decoder).map_err(|_| invalid_program())?;
    decoder.end().map_err(|_| invalid_program())?;
    Ok(value)
}

#[derive(Clone)]
pub(super) struct ProgramStep {
    pub(super) ordinal: u64,
    pub(super) kind: Option<SemanticTransitionKind>,
    pub(super) segment_begin: u64,
    pub(super) segment_end: u64,
    pub(super) logical_update: Option<u64>,
    pub(super) actor_slot: Option<u64>,
    pub(super) before_proposals: u64,
    pub(super) phase: Identity256,
    pub(super) instruction: Vec<u8>,
}

pub(super) struct ActorRefreshProgram {
    pub(super) bytes: Vec<u8>,
    pub(super) identity: Identity256,
    pub(super) proposals: u64,
    pub(super) updates: u64,
    pub(super) logical_updates: u64,
    pub(super) steps: Vec<ProgramStep>,
    pub(super) original_rng: Option<SemanticRngBinding>,
    pub(super) main_origins: Vec<MainActorAssignment>,
}

pub(super) struct MainActorAssignment {
    pub(super) ordinal: u64,
    pub(super) origin: SemanticTrainingViewOriginRecord,
    pub(super) slot: u64,
    pub(super) batch: Identity256,
    pub(super) batch_bytes: Vec<u8>,
    pub(super) replay: AssignmentReplayProof,
}

#[derive(Clone, Copy)]
pub(super) enum AssignmentReplayProof { Unresolved, Joined }

fn structural_bytes(value: &Value) -> Result<Vec<u8>, SemanticTransitionError> {
    serde_json::to_vec(value).map_err(|_| invalid_program())
}

fn instruction_projection(bytes: &[u8]) -> Result<Vec<u8>, SemanticTransitionError> {
    if bytes.is_empty() || bytes.len() > PROGRAM_BYTES { return Err(invalid_program()); }
    let mut value = json_value(bytes)?;
    let instruction = value.as_object_mut().ok_or_else(invalid_program)?;
    if let Some(recipe) = instruction.get_mut("recipe").filter(|recipe| !recipe.is_null()) {
        let recipe = recipe.as_object_mut().ok_or_else(invalid_program)?;
        if recipe.len() != 5 || ["source", "target", "phase_index", "completed_updates_index", "views"]
            .iter().any(|field| !recipe.contains_key(*field)) { return Err(invalid_program()); }
        for field in ["phase_index", "completed_updates_index", "views"] { recipe.remove(field); }
    }
    structural_bytes(&value)
}

fn validate_segment(value: &Value) -> Result<(), SemanticTransitionError> {
    let segment = object(value, &["operation", "transitions", "capacity_bytes", "tensor_content_capacity",
        "model_work_capacity", "other_external_cuda_bytes", "entropy_threshold", "distance_weight", "proposal_views"])?;
    let mode = operation(&segment["operation"])?;
    if !matches!(mode, "inference" | "training") { return Err(invalid_program()); }
    let transitions = array(&segment["transitions"])?;
    if transitions.is_empty() { return Err(invalid_program()); }
    let mut proposals = 0usize;
    let mut updates = 0usize;
    for transition in transitions {
        match operation(transition)? {
            "proposal" => proposals += 1,
            "recompute" => (),
            "update" if mode == "training" => updates += 1,
            _ => return Err(invalid_program()),
        }
    }
    if mode == "training" && updates == 0 { return Err(invalid_program()); }
    if integer(&segment["capacity_bytes"])? <= 512 { return Err(invalid_program()); }
    for field in ["tensor_content_capacity", "model_work_capacity", "other_external_cuda_bytes"] {
        if integer(&segment[field])? == 0 { return Err(invalid_program()); }
    }
    for field in ["entropy_threshold", "distance_weight"] {
        let value = segment[field].as_f64().filter(|value| value.is_finite()).ok_or_else(invalid_program)?;
        if field == "distance_weight" && value < 0.0 { return Err(invalid_program()); }
    }
    let views = array(&segment["proposal_views"])?;
    if views.len() != proposals { return Err(invalid_program()); }
    for view in views {
        let view = array(view)?;
        if view.len() != 2 { return Err(invalid_program()); }
        for coordinate in view { integer(coordinate)?; }
    }
    Ok(())
}

fn invalid_program() -> SemanticTransitionError {
    publication_input_error("actor refresh requires its complete original ordered program")
}

fn object<'a>(value: &'a Value, fields: &[&str]) -> Result<&'a serde_json::Map<String, Value>, SemanticTransitionError> {
    let object = value.as_object().ok_or_else(invalid_program)?;
    if object.len() != fields.len() || fields.iter().any(|field| !object.contains_key(*field)) {
        return Err(invalid_program());
    }
    Ok(object)
}

fn array(value: &Value) -> Result<&[Value], SemanticTransitionError> {
    value.as_array().map(Vec::as_slice).ok_or_else(invalid_program)
}

fn integer(value: &Value) -> Result<u64, SemanticTransitionError> {
    value.as_u64().ok_or_else(invalid_program)
}

fn operation(value: &Value) -> Result<&str, SemanticTransitionError> {
    value.as_str().filter(|value| !value.is_empty()).ok_or_else(invalid_program)
}

fn kind(operation: &str) -> Option<SemanticTransitionKind> {
    match operation {
        "proposal" => Some(SemanticTransitionKind::Proposal),
        "recompute" => Some(SemanticTransitionKind::Recompute),
        "update" => Some(SemanticTransitionKind::Update),
        "drain" => Some(SemanticTransitionKind::Drain),
        _ => None,
    }
}

fn digest(value: &Value) -> Result<(), SemanticTransitionError> {
    let value = operation(value)?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || value.bytes().all(|byte| byte == b'0') {
        return Err(invalid_program());
    }
    Ok(())
}

impl ActorRefreshProgram {
    fn decode(bytes: &[u8]) -> Result<Self, SemanticTransitionError> {
        if bytes.is_empty() || bytes.len() > PROGRAM_BYTES { return Err(invalid_program()); }
        let value = json_value(bytes)?;
        let root = object(&value, &["kind", "run_sha256", "resources_sha256", "fresh_restore_before", "frozen_run_sha256", "schedule"])?;
        if operation(&root["kind"])? != "dlm-new.original-self-source-program/1" { return Err(invalid_program()); }
        for field in ["run_sha256", "resources_sha256", "frozen_run_sha256"] { digest(&root[field])?; }
        let schedule = object(&root["schedule"], &["requests", "updates"])?;
        let requests = array(&schedule["requests"])?;
        if requests.is_empty() || integer(&root["fresh_restore_before"])? >= requests.len() as u64 {
            return Err(invalid_program());
        }
        let mut steps = Vec::new();
        let mut expected_updates = Vec::new();
        let mut proposals = 0u64;
        let mut updates = 0u64;
        let mut logical_updates = 0u64;
        let mut actor_slots = 0u64;
        for (request_ordinal, request) in requests.iter().enumerate() {
            let request_object = request.as_object().ok_or_else(invalid_program)?;
            let request_kind = operation(request_object.get("operation").ok_or_else(invalid_program)?)?;
            let phase = Identity256::from_bytes(Sha256::digest(structural_bytes(request)?).into());
            let mut paired_updates: BTreeMap<u64, (u64, Vec<u8>)> = BTreeMap::new();
            let mut paired_proposals: BTreeMap<u64, (u64, Vec<u8>)> = BTreeMap::new();
            let mut branch_updates: BTreeMap<String, u64> = BTreeMap::new();
            let mut branch_proposals: BTreeMap<String, u64> = BTreeMap::new();
            let mut branches = BTreeSet::new();
            let mut paired_instructions: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();
            let entries: Vec<(Value, Option<String>, u64, String)> = if request_kind == "learning-phase" {
                let phase_request = object(request, &["operation", "source", "target", "final_phase", "scientific_phase",
                    "feedback_interventions", "schedule", "terminal_cleanup", "learning_grant_ref"])?;
                let source = operation(&phase_request["source"])?;
                let target = operation(&phase_request["target"])?;
                if !matches!((source, target), ("alignment", "fast") | ("fast", "consolidation") | ("consolidation", "fast"))
                    || !matches!(operation(&phase_request["final_phase"])?, "fast" | "consolidation")
                    || (source != "fast" && phase_request["final_phase"] != "fast")
                    || operation(&phase_request["scientific_phase"])? != if source == "alignment" { "alignment" } else { "consolidation" }
                { return Err(invalid_program()); }
                operation(&phase_request["learning_grant_ref"])?;
                let interventions = array(&phase_request["feedback_interventions"])?;
                if interventions.len() != 2 { return Err(invalid_program()); }
                for (mode, intervention) in interventions.iter().enumerate() {
                    let mut original = b"xlog.learning-feedback-intervention.v1\0".to_vec();
                    original.push(mode as u8);
                    let expected: String = original.iter().map(|byte| format!("{byte:02x}")).collect();
                    if operation(intervention)? != expected { return Err(invalid_program()); }
                }
                let cleanup = object(&phase_request["terminal_cleanup"], &["operation_ordinal", "budget"])?;
                let budget = array(&cleanup["budget"])?;
                if integer(&cleanup["operation_ordinal"])? != array(&phase_request["schedule"])?.len() as u64
                    || budget.len() != 3 || integer(&budget[1])? == 0 { return Err(invalid_program()); }
                for value in budget { integer(value)?; }
                array(request_object.get("schedule").ok_or_else(invalid_program)?)?.iter().map(|entry| {
                    let entry = object(entry, &["instruction", "step_ordinal", "operation", "branch", "budget", "comparison"])?;
                    let branch = operation(&entry["branch"])?;
                    if !matches!(branch, "source" | "control" | "real") { return Err(invalid_program()); }
                    let budget = array(&entry["budget"])?;
                    if budget.len() != 3 || integer(&budget[1])? == 0 { return Err(invalid_program()); }
                    for value in budget { integer(value)?; }
                    let entry_operation = operation(&entry["operation"])?;
                    if entry_operation == "evaluation" {
                        let comparison = object(&entry["comparison"], &["case"])?;
                        if comparison["case"] != entry["instruction"]["cohort_case"] { return Err(invalid_program()); }
                    } else if !entry["comparison"].is_null() { return Err(invalid_program()); }
                    Ok((entry["instruction"].clone(), Some(branch.to_owned()), integer(&entry["step_ordinal"])?, operation(&entry["operation"])?.to_owned()))
                }).collect::<Result<_, _>>()?
            } else {
                validate_segment(request)?;
                array(request_object.get("transitions").ok_or_else(invalid_program)?)?.iter().enumerate()
                    .map(|(ordinal, transition)| Ok((request.clone(), None, ordinal as u64, operation(transition)?.to_owned())))
                    .collect::<Result<_, SemanticTransitionError>>()?
            };
            let request_actor_start = actor_slots;
            let mut segment_begin = steps.len() as u64;
            let mut active_segment: Option<(Vec<u8>, Option<String>, usize)> = None;
            for (instruction, branch, step_ordinal, operation_name) in entries {
                let instruction_object = instruction.as_object().ok_or_else(invalid_program)?;
                let instruction_bytes = structural_bytes(&instruction)?;
                let transition = kind(&operation_name);
                let mut segment_end = steps.len() as u64 + 1;
                if transition.is_some() {
                    validate_segment(&instruction)?;
                    let transitions = array(instruction_object.get("transitions").ok_or_else(invalid_program)?)?;
                    if transitions.is_empty() || step_ordinal as usize >= transitions.len()
                        || operation(&transitions[step_ordinal as usize])? != operation_name {
                        return Err(invalid_program());
                    }
                    if step_ordinal == 0 {
                        if active_segment.is_some() { return Err(invalid_program()); }
                        segment_begin = steps.len() as u64;
                        active_segment = Some((instruction_bytes.clone(), branch.clone(), transitions.len()));
                    }
                    let (original, original_branch, count) = active_segment.as_ref().ok_or_else(invalid_program)?;
                    if *original != instruction_bytes || *original_branch != branch
                        || steps.len() as u64 != segment_begin + step_ordinal {
                        return Err(invalid_program());
                    }
                    segment_end = segment_begin.checked_add(*count as u64).ok_or_else(invalid_program)?;
                    if step_ordinal as usize + 1 == *count { active_segment = None; }
                } else {
                    if active_segment.is_some() || step_ordinal != 0 { return Err(invalid_program()); }
                    match operation_name.as_str() {
                        "evaluation" => {
                            let evaluation = object(&instruction, &["cohort_case", "memory_capacity_bytes", "model_work_capacity", "forward_calls", "other_external_cuda_bytes"])?;
                            operation(&evaluation["cohort_case"])?;
                            if integer(&evaluation["forward_calls"])? != 1 { return Err(invalid_program()); }
                            for field in ["memory_capacity_bytes", "model_work_capacity", "other_external_cuda_bytes"] {
                                if integer(&evaluation[field])? == 0 { return Err(invalid_program()); }
                            }
                        }
                        "trajectory-start" | "restore" => {
                            let restore = object(&instruction, &["checkpoint", "recipe", "forward_calls", "other_external_cuda_bytes"])?;
                            if integer(&restore["forward_calls"])? != 0
                                || integer(&restore["other_external_cuda_bytes"])? == 0
                                || operation(&restore["checkpoint"])? != if operation_name == "trajectory-start" { "source" } else { "current" } {
                                return Err(invalid_program());
                            }
                            if operation_name == "trajectory-start" {
                                if !restore["recipe"].is_null() { return Err(invalid_program()); }
                            } else {
                                let recipe = object(&restore["recipe"], &["source", "target"])?;
                                if !matches!((operation(&recipe["source"])?, operation(&recipe["target"])?),
                                    ("alignment", "fast") | ("fast", "consolidation") | ("consolidation", "fast")) { return Err(invalid_program()); }
                            }
                        }
                        "prepare" | "checkpoint" | "retire" | "delivery" => {
                            if operation_name == "prepare" {
                                let prepare = object(&instruction, &["operation", "cold_model_work_capacity"])?;
                                if integer(&prepare["cold_model_work_capacity"])? == 0 { return Err(invalid_program()); }
                            } else { object(&instruction, &["operation"])?; }
                            if operation(instruction_object.get("operation").ok_or_else(invalid_program)?)? != operation_name {
                                return Err(invalid_program());
                            }
                        }
                        _ => return Err(invalid_program()),
                    }
                    segment_begin = steps.len() as u64;
                }
                let mut logical_update = None;
                let mut actor_slot = None;
                let paired = matches!(branch.as_deref(), Some("control" | "real"));
                let branch_name = branch.clone().unwrap_or_default();
                if paired && !matches!(operation_name.as_str(), "retire" | "delivery") {
                    paired_instructions.entry(branch_name.clone()).or_default().push(instruction_bytes.clone());
                }
                if let Some(branch) = &branch { branches.insert(branch.clone()); }
                if transition == Some(SemanticTransitionKind::Update) {
                    expected_updates.push(serde_json::json!({"request": request_ordinal, "branch": branch,
                        "step_ordinal": step_ordinal, "program_ordinal": steps.len()}));
                    let ordinal = branch_updates.entry(branch_name.clone()).or_default();
                    let index = if paired {
                        if let Some((index, original)) = paired_updates.get(ordinal) {
                            if original != &instruction_bytes { return Err(invalid_program()); }
                            *index
                        } else {
                            let index = logical_updates;
                            paired_updates.insert(*ordinal, (index, instruction_bytes.clone()));
                            logical_updates = logical_updates.checked_add(1).ok_or_else(invalid_program)?;
                            index
                        }
                    } else {
                        let index = logical_updates;
                        logical_updates = logical_updates.checked_add(1).ok_or_else(invalid_program)?;
                        index
                    };
                    *ordinal = ordinal.checked_add(1).ok_or_else(invalid_program)?;
                    logical_update = Some(index);
                    updates = updates.checked_add(1).ok_or_else(invalid_program)?;
                }
                if transition == Some(SemanticTransitionKind::Proposal) {
                    let ordinal = branch_proposals.entry(branch_name.clone()).or_default();
                    let index = if paired {
                        if let Some((index, original)) = paired_proposals.get(ordinal) {
                            if original != &instruction_bytes { return Err(invalid_program()); }
                            *index
                        } else {
                            let index = actor_slots;
                            paired_proposals.insert(*ordinal, (index, instruction_bytes.clone()));
                            actor_slots = actor_slots.checked_add(1).ok_or_else(invalid_program)?;
                            index
                        }
                    } else {
                        let index = actor_slots;
                        actor_slots = actor_slots.checked_add(1).ok_or_else(invalid_program)?;
                        index
                    };
                    *ordinal = ordinal.checked_add(1).ok_or_else(invalid_program)?;
                    actor_slot = Some(index);
                    proposals = proposals.checked_add(1).ok_or_else(invalid_program)?;
                }
                let before_proposals = if paired {
                    request_actor_start + branch_proposals.get(&branch_name).copied().unwrap_or(0)
                        - u64::from(transition == Some(SemanticTransitionKind::Proposal))
                } else { actor_slot.unwrap_or(actor_slots) };
                steps.push(ProgramStep { ordinal: steps.len() as u64, kind: transition,
                    segment_begin, segment_end, logical_update, actor_slot, before_proposals, phase,
                    instruction: instruction_bytes });
            }
            if active_segment.is_some() { return Err(invalid_program()); }
            if request_kind == "learning-phase" && (!branches.contains("control") || !branches.contains("real")
                || branch_updates.get("control") != branch_updates.get("real")
                || branch_proposals.get("control") != branch_proposals.get("real")
                || paired_instructions.get("control") != paired_instructions.get("real")) {
                return Err(invalid_program());
            }
        }
        if schedule["updates"] != Value::Array(expected_updates) { return Err(invalid_program()); }
        Ok(Self { bytes: bytes.to_vec(), identity: Identity256::from_bytes(Sha256::digest(bytes).into()),
            proposals, updates, logical_updates, steps, original_rng: None, main_origins: Vec::new() })
    }
}

impl SemanticTransitionSession {
    pub fn bind_actor_refresh_program(&mut self, program_bytes: &[u8]) -> Result<[u64; 2], SemanticTransitionError> {
        self.ensure_quiescent()?;
        if let Some(program) = &self.actor_refresh_program {
            if program.bytes != program_bytes { return Err(invalid_program()); }
            return Ok([program.proposals, program.updates]);
        }
        if self.prepared_segment.is_some() { return Err(invalid_program()); }
        let program = ActorRefreshProgram::decode(program_bytes)?;
        let quantities = [program.proposals, program.updates];
        self.actor_refresh_program = Some(program);
        Ok(quantities)
    }

    /// Authenticate the complete actual native segment against its original
    /// place in the frozen program, before any callback or draw is recorded.
    pub fn bind_prepared_actor_refresh_program(
        &mut self,
        steps: &[SemanticPreparedStep],
        released_parent: &SemanticPublishedLease,
        first_program_ordinal: u64,
        instruction_bytes: &[u8],
    ) -> Result<(), SemanticTransitionError> {
        let first = steps.first().ok_or_else(invalid_program)?;
        self.require_prepared_program_parent(first, released_parent)?;
        let program = self.actor_refresh_program.as_ref().ok_or_else(invalid_program)?;
        let start = usize::try_from(first_program_ordinal).map_err(|_| invalid_program())?;
        let original = program.steps.get(start).ok_or_else(invalid_program)?;
        if instruction_projection(instruction_bytes)? != original.instruction { return Err(invalid_program()); }
        let end = start.checked_add(steps.len()).ok_or_else(invalid_program)?;
        if original.segment_begin != first_program_ordinal || original.segment_end != end as u64 {
            return Err(invalid_program());
        }
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        if build.tokens.len() != steps.len() || !build.program_steps.is_empty() {
            return Err(invalid_program());
        }
        let instruction = json_value(instruction_bytes)?;
        for (field, actual) in [
            ("capacity_bytes", build.cold_capacity.segment_capacity_bytes),
            ("tensor_content_capacity", build.cold_capacity.tensor_content_capacity as u64),
            ("model_work_capacity", build.cold_capacity.model_work_capacity as u64),
            ("other_external_cuda_bytes", build.cold_capacity.other_external_cuda_bytes),
        ] {
            if integer(&instruction[field])? != actual { return Err(invalid_program()); }
        }
        for ((step, token), expected) in steps.iter().zip(&build.tokens).zip(&program.steps[start..end]) {
            build.check_retained(step, &self.publication_issuer)?;
            if step.token != *token || Some(build.requested_kind(step, &self.publication_issuer)?) != expected.kind {
                return Err(invalid_program());
            }
        }
        let mut rng = released_parent.header.rng_binding()?;
        rng.proposal = u32::try_from(u64::from(rng.proposal).checked_sub(original.before_proposals)
            .ok_or_else(invalid_program)?).map_err(|_| invalid_program())?;
        let program = self.actor_refresh_program.as_mut().expect("checked original program");
        if let Some(previous) = program.original_rng {
            if previous.family_id != rng.family_id || previous.stream_serial != rng.stream_serial
                || previous.proposal != rng.proposal { return Err(invalid_program()); }
        } else { program.original_rng = Some(rng); }
        let mapping = steps.iter().zip(&program.steps[start..end])
            .map(|(step, expected)| (step.token, expected.clone())).collect();
        self.prepared_segment.as_mut().expect("checked original scope").program_steps = mapping;
        Ok(())
    }

    pub(super) fn register_prepared_actor_refresh_origin(
        &mut self,
        step: &SemanticPreparedStep,
        origin: SemanticTrainingViewOriginRecord,
        batch: Identity256,
        batch_bytes: Vec<u8>,
    ) -> Result<(), SemanticTransitionError> {
        let Some(program) = self.actor_refresh_program.as_mut() else { return Ok(()); };
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        let address = build.program_steps.get(&step.token).ok_or_else(invalid_program)?;
        let slot = address.actor_slot.ok_or_else(invalid_program)?;
        if let Some(previous) = program.main_origins.iter().find(|previous| previous.ordinal == address.ordinal) {
            if previous.slot != slot || previous.origin != origin || previous.batch != batch
                || previous.batch_bytes != batch_bytes { return Err(invalid_program()); }
        } else {
            if program.main_origins.iter().any(|previous| previous.origin == origin)
                || program.main_origins.len() as u64 >= program.proposals { return Err(invalid_program()); }
            if batch_bytes.len() != size_of::<SemanticActionBatchReceipt>()
                || Identity256::from_bytes(Sha256::digest(&batch_bytes).into()) != batch { return Err(invalid_program()); }
            program.main_origins.push(MainActorAssignment { ordinal: address.ordinal, origin, slot, batch,
                batch_bytes, replay: AssignmentReplayProof::Joined });
        }
        Ok(())
    }

    /// Native-issued assignments retained by the original authenticated task
    /// checkpoint. Complete replay materials remain in their existing custody.
    pub fn actor_refresh_assignment_material(&self) -> Result<Vec<u8>, SemanticTransitionError> {
        let program = self.actor_refresh_program.as_ref().ok_or_else(invalid_program)?;
        let initial = self.actor_refresh_initial.as_ref().ok_or_else(invalid_program)?;
        let mut bytes = b"XLOG-ACTOR-REFRESH-ASSIGNMENTS\0".to_vec();
        material_u64(&mut bytes, 2);
        bytes.extend_from_slice(program.identity.as_bytes());
        bytes.extend_from_slice(initial.index.as_bytes());
        material_u64(&mut bytes, u64::from(program.original_rng.is_some()));
        if let Some(rng) = program.original_rng {
            for value in [rng.stream_serial, u64::from(rng.family_id), u64::from(rng.proposal)] {
                material_u64(&mut bytes, value);
            }
        }
        material_u64(&mut bytes, program.main_origins.len() as u64);
        let mut assignments: Vec<_> = program.main_origins.iter().collect();
        assignments.sort_by_key(|assignment| assignment.ordinal);
        for assignment in assignments {
            material_u64(&mut bytes, assignment.ordinal);
            bytes.extend_from_slice(&publication_abi_bytes(&[assignment.origin]));
            bytes.extend_from_slice(assignment.batch.as_bytes());
            bytes.extend_from_slice(&assignment.batch_bytes);
        }
        Ok(bytes)
    }

    /// Restore every native-issued assignment from the authenticated checkpoint.
    /// An unavailable episode remains unresolved, not dropped or authenticated.
    /// Actor preparation joins its complete original replay before any draw.
    pub fn restore_actor_refresh_assignments(
        &mut self,
        bytes: &[u8],
        episodes: &[(SemanticReplayMaterial, Identity256, Vec<u8>)],
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        let program = self.actor_refresh_program.as_ref().ok_or_else(invalid_program)?;
        let initial = self.actor_refresh_initial.as_ref().ok_or_else(invalid_program)?;
        if !program.main_origins.is_empty() || self.prepared_segment.is_some() { return Err(invalid_program()); }
        let mut reader = SemanticMaterialReader::new(bytes);
        let magic = b"XLOG-ACTOR-REFRESH-ASSIGNMENTS\0";
        let semantic = SemanticTransitionError::Semantic;
        if reader.take(magic.len()).map_err(semantic)? != magic
            || reader.u64().map_err(semantic)? != 2
            || reader.take(32).map_err(semantic)? != program.identity.as_bytes()
            || reader.take(32).map_err(semantic)? != initial.index.as_bytes() { return Err(invalid_program()); }
        let rng = match reader.u64().map_err(semantic)? {
            0 => None,
            1 => {
                let stream_serial = reader.u64().map_err(semantic)?;
                let family_id = u8::try_from(reader.u64().map_err(semantic)?).map_err(|_| invalid_program())?;
                let proposal = u32::try_from(reader.u64().map_err(semantic)?).map_err(|_| invalid_program())?;
                if stream_serial >= 1u64 << 56 { return Err(invalid_program()); }
                Some(SemanticRngBinding { model_generation: 0, stream_serial, family_id, proposal })
            }
            _ => return Err(invalid_program()),
        };
        let count = reader.u64().map_err(semantic)?;
        if count > program.proposals || (count != 0 && rng.is_none()) { return Err(invalid_program()); }
        let mut assignments = Vec::new();
        for _ in 0..count {
            let ordinal = reader.u64().map_err(semantic)?;
            let raw = reader.take(size_of::<SemanticTrainingViewOriginRecord>()).map_err(semantic)?;
            // SAFETY: fixed integer-only ABI with its exact extent checked.
            let origin = unsafe { std::ptr::read_unaligned(raw.as_ptr().cast::<SemanticTrainingViewOriginRecord>()) };
            let batch = Identity256::from_bytes(reader.take(32).map_err(semantic)?.try_into().map_err(|_| invalid_program())?);
            let batch_bytes = reader.take(size_of::<SemanticActionBatchReceipt>()).map_err(semantic)?.to_vec();
            if Identity256::from_bytes(Sha256::digest(&batch_bytes).into()) != batch { return Err(invalid_program()); }
            let address = usize::try_from(ordinal).ok().and_then(|ordinal| program.steps.get(ordinal)).ok_or_else(invalid_program)?;
            let slot = address.actor_slot.ok_or_else(invalid_program)?;
            if assignments.last().is_some_and(|previous: &MainActorAssignment| previous.ordinal >= ordinal)
                || assignments.iter().any(|previous| previous.origin == origin) { return Err(invalid_program()); }
            let mut matches = 0usize;
            for (material, identity, bytes) in episodes {
                if *identity != batch { continue; }
                let (actual, actor) = actor_refresh::initial_episode_actor(material, *identity, bytes)?;
                if actual == origin && actor { matches += 1; }
            }
            if matches > 1 { return Err(invalid_program()); }
            assignments.push(MainActorAssignment { ordinal, origin, slot, batch, batch_bytes,
                replay: if matches == 1 { AssignmentReplayProof::Joined } else { AssignmentReplayProof::Unresolved } });
        }
        reader.finish().map_err(semantic)?;
        let program = self.actor_refresh_program.as_mut().expect("checked original program");
        program.main_origins = assignments;
        program.original_rng = rng;
        Ok(())
    }
}
