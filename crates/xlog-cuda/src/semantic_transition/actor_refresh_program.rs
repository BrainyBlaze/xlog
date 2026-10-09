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
    pub(super) budget: [u64; 3],
    request: u64,
    branch: Option<String>,
}

/// One original instruction, admitted while its actual publication reader is
/// still active. Only the originating Session can issue or consume this handle.
#[derive(Clone)]
pub struct SemanticSegmentInstructionAdmission {
    issuer: Arc<()>,
    issuance: Arc<()>,
    first: u64,
}

impl SemanticSegmentInstructionAdmission {
    pub fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.issuer, &other.issuer)
            && Arc::ptr_eq(&self.issuance, &other.issuance)
            && self.first == other.first
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum InstructionState {
    Admitted = 0,
    HandoffEntered = 1,
    Released = 2,
    BuildEntered = 3,
    Bound = 4,
    Completed = 5,
}

pub(super) struct InstructionAdmission {
    issuance: Arc<()>,
    first: u64,
    end: u64,
    pub(super) instruction: Arc<[u8]>,
    instruction_identity: Identity256,
    parent: SemanticPublishedIdentity,
    parent_token: u64,
    pub(super) state: InstructionState,
    pub(super) cold_work: Option<SemanticColdModelWork>,
    pub(super) cold_result: Option<SemanticColdModelWorkResult>,
    pub(super) retirement_work: Option<SemanticColdModelWork>,
    pub(super) retirement_result: Option<SemanticColdModelWorkResult>,
    pub(super) prepared_cold_results: Vec<(u64, Option<SemanticColdModelWorkResult>)>,
    handoff_streams: Option<Vec<u64>>,
    parent_fully_retired: bool,
    resources_retired: bool,
    // Reader retirement does not retire the original numerical generation or
    // the allocation catalogue needed by historical and backward consumers.
    _publication: Option<Arc<PublicationStorage>>,
    _model_owners: Vec<Arc<ModelGenerationOwner>>,
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
    admissions: Vec<InstructionAdmission>,
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

fn require_completed_instruction_prefix(program: &ActorRefreshProgram, start: usize,
    admissions: &[InstructionAdmission]) -> Result<(), SemanticTransitionError> {
    let original = program.steps.get(start).ok_or_else(invalid_program)?;
    if admissions.iter().any(|claim| claim.first == original.ordinal
        || claim.state != InstructionState::Completed) { return Err(invalid_program()); }
    for previous in &program.steps[..start] {
        if previous.kind.is_none()
            || (previous.branch.as_deref() == Some("control")
                && (previous.request != original.request || original.branch.as_deref() != Some("control")))
            || (previous.request == original.request && previous.branch.as_deref() == Some("real")
                && original.branch.as_deref() != Some("real")) { continue; }
        if !admissions.iter().any(|claim| claim.state == InstructionState::Completed
            && claim.first <= previous.ordinal && previous.ordinal < claim.end) {
            return Err(publication_input_error("original program progression has an uncompleted instruction"));
        }
    }
    Ok(())
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

fn resource_budget(value: &Value) -> Result<[u64; 3], SemanticTransitionError> {
    let values = array(value)?;
    if values.len() != 3 { return Err(invalid_program()); }
    let result = [integer(&values[0])?, integer(&values[1])?, integer(&values[2])?];
    if result[1] == 0 { return Err(invalid_program()); }
    Ok(result)
}

fn write_instruction_cold_result(bytes: &mut Vec<u8>, result: Option<SemanticColdModelWorkResult>) {
    material_u64(bytes, u64::from(result.is_some()));
    if let Some(result) = result {
        for value in [result.model_work, result.operation_count, result.work_bound,
            result.model_calls, result.native_work].into_iter().chain(result.native_events) {
            material_u64(bytes, value);
        }
    }
}

fn read_instruction_cold_result(reader: &mut SemanticMaterialReader<'_>) -> Result<Option<SemanticColdModelWorkResult>, SemanticTransitionError> {
    let semantic = SemanticTransitionError::Semantic;
    match reader.u64().map_err(semantic)? {
        0 => Ok(None),
        1 => {
            let model_work = reader.u64().map_err(semantic)?;
            let operation_count = reader.u64().map_err(semantic)?;
            let work_bound = reader.u64().map_err(semantic)?;
            let model_calls = reader.u64().map_err(semantic)?;
            let native_work = reader.u64().map_err(semantic)?;
            let mut native_events = [0; 9];
            for value in &mut native_events { *value = reader.u64().map_err(semantic)?; }
            if model_work > work_bound || model_calls > operation_count
                || native_events.iter().try_fold(0u64, |sum, value| sum.checked_add(*value)) != Some(native_work)
                || model_work.checked_add(native_work).is_none() { return Err(invalid_program()); }
            Ok(Some(SemanticColdModelWorkResult { model_work, operation_count, work_bound,
                model_calls, native_work, native_events }))
        }
        _ => Err(invalid_program()),
    }
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
        let root = object(&value, &["kind", "run_sha256", "resources_sha256", "fresh_restore_before", "frozen_run_sha256", "schedule", "budgets"])?;
        if operation(&root["kind"])? != "dlm-new.original-self-source-program/1" { return Err(invalid_program()); }
        for field in ["run_sha256", "resources_sha256", "frozen_run_sha256"] { digest(&root[field])?; }
        let budgets = object(&root["budgets"], &["prepare", "trajectory-start", "restore", "proposal", "recompute", "update",
            "evaluation", "checkpoint", "retire", "delivery", "terminal-cleanup"])?;
        let budgets: BTreeMap<_, _> = budgets.iter().map(|(name, value)| {
            if name.is_empty() { return Err(invalid_program()); }
            Ok((name.as_str(), resource_budget(value)?))
        }).collect::<Result<_, SemanticTransitionError>>()?;
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
            let entries: Vec<(Value, Option<String>, u64, String, [u64; 3])> = if request_kind == "learning-phase" {
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
                if resource_budget(&cleanup["budget"])? != *budgets.get("terminal-cleanup").ok_or_else(invalid_program)? { return Err(invalid_program()); }
                array(request_object.get("schedule").ok_or_else(invalid_program)?)?.iter().map(|entry| {
                    let entry = object(entry, &["instruction", "step_ordinal", "operation", "branch", "budget", "comparison"])?;
                    let branch = operation(&entry["branch"])?;
                    if !matches!(branch, "source" | "control" | "real") { return Err(invalid_program()); }
                    let budget = array(&entry["budget"])?;
                    if budget.len() != 3 || integer(&budget[1])? == 0 { return Err(invalid_program()); }
                    for value in budget { integer(value)?; }
                    let entry_operation = operation(&entry["operation"])?;
                    let budget = resource_budget(&entry["budget"])?;
                    if budget != *budgets.get(entry_operation).ok_or_else(invalid_program)? { return Err(invalid_program()); }
                    if entry_operation == "evaluation" {
                        let comparison = object(&entry["comparison"], &["case"])?;
                        if comparison["case"] != entry["instruction"]["cohort_case"] { return Err(invalid_program()); }
                    } else if !entry["comparison"].is_null() { return Err(invalid_program()); }
                    Ok((entry["instruction"].clone(), Some(branch.to_owned()), integer(&entry["step_ordinal"])?, entry_operation.to_owned(), budget))
                }).collect::<Result<_, _>>()?
            } else {
                let mut instruction = request.clone();
                instruction.as_object_mut().ok_or_else(invalid_program)?.remove("budgets").ok_or_else(invalid_program)?;
                validate_segment(&instruction)?;
                let ordinary_budgets = array(request_object.get("budgets").ok_or_else(invalid_program)?)?;
                if ordinary_budgets.len() != array(request_object.get("transitions").ok_or_else(invalid_program)?)?.len() { return Err(invalid_program()); }
                array(request_object.get("transitions").ok_or_else(invalid_program)?)?.iter().enumerate()
                    .map(|(ordinal, transition)| {
                        let operation = operation(transition)?;
                        let budget = resource_budget(&ordinary_budgets[ordinal])?;
                        if budget != *budgets.get(operation).ok_or_else(invalid_program)? { return Err(invalid_program()); }
                        Ok((instruction.clone(), None, ordinal as u64, operation.to_owned(), budget))
                    })
                    .collect::<Result<_, SemanticTransitionError>>()?
            };
            let request_actor_start = actor_slots;
            let mut segment_begin = steps.len() as u64;
            let mut active_segment: Option<(Vec<u8>, Option<String>, usize)> = None;
            for (instruction, branch, step_ordinal, operation_name, budget) in entries {
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
                    instruction: instruction_bytes, budget, request: request_ordinal as u64, branch });
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
            proposals, updates, logical_updates, steps, original_rng: None, main_origins: Vec::new(),
            admissions: Vec::new() })
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

    /// Authenticate the original live reader and instruction before any cold
    /// callback, historical import, nominal charge or prepared allocation.
    pub fn admit_segment_instruction(
        &mut self,
        parent: &SemanticPublishedLease,
        first_program_ordinal: u64,
        instruction_bytes: &[u8],
    ) -> Result<SemanticSegmentInstructionAdmission, SemanticTransitionError> {
        self.ensure_rebindable()?;
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        if self.prepared_segment.is_some() || self.readers.len() != 1 {
            return Err(invalid_program());
        }
        let program = self.actor_refresh_program.as_ref().ok_or_else(invalid_program)?;
        let start = usize::try_from(first_program_ordinal).map_err(|_| invalid_program())?;
        let original = program.steps.get(start).ok_or_else(invalid_program)?;
        if instruction_projection(instruction_bytes)? != original.instruction { return Err(invalid_program()); }
        let end = usize::try_from(original.segment_end).map_err(|_| invalid_program())?;
        if original.kind.is_none() || original.segment_begin != first_program_ordinal || end > program.steps.len() {
            return Err(invalid_program());
        }
        // A SOURCE checkpoint forks independent control/real execution. Their
        // equal logical assignments do not make either branch's effects receipts
        // for the other. Prior real/ordinary work, and this branch's prefix, must
        // have completed under native-issued claims before its next segment.
        require_completed_instruction_prefix(program, start, &program.admissions)?;
        let mut rng = parent.header.rng_binding()?;
        rng.proposal = u32::try_from(u64::from(rng.proposal).checked_sub(original.before_proposals)
            .ok_or_else(invalid_program)?).map_err(|_| invalid_program())?;
        if let Some(previous) = program.original_rng {
            if previous.family_id != rng.family_id || previous.stream_serial != rng.stream_serial
                || previous.proposal != rng.proposal { return Err(invalid_program()); }
        } else if original.before_proposals != 0 {
            return Err(invalid_program());
        }
        let issuance = Arc::new(());
        let admission = SemanticSegmentInstructionAdmission {
            issuer: Arc::clone(&self.publication_issuer), issuance: Arc::clone(&issuance), first: first_program_ordinal,
        };
        let claim = InstructionAdmission {
            issuance, first: first_program_ordinal, end: end as u64,
            instruction: Arc::from(instruction_bytes), instruction_identity: Identity256::from_bytes(Sha256::digest(instruction_bytes).into()),
            parent: parent.identity, parent_token: parent.token, state: InstructionState::Admitted,
            cold_work: None, cold_result: None,
            retirement_work: None, retirement_result: None,
            prepared_cold_results: Vec::new(),
            handoff_streams: None,
            parent_fully_retired: false,
            resources_retired: false,
            _publication: Some(Arc::clone(self.publication.as_ref().ok_or(SemanticTransitionError::NotBound)?)),
            _model_owners: parent._model_owners.clone(),
        };
        let program = self.actor_refresh_program.as_mut().expect("checked original program");
        program.admissions.try_reserve(1).map_err(|error| runtime_error("original instruction custody", error))?;
        program.original_rng = Some(rng);
        program.admissions.push(claim);
        Ok(admission)
    }

    pub(super) fn instruction_admission(&self, admission: &SemanticSegmentInstructionAdmission) -> Result<&InstructionAdmission, SemanticTransitionError> {
        if !Arc::ptr_eq(&admission.issuer, &self.publication_issuer) { return Err(invalid_program()); }
        self.actor_refresh_program.as_ref().ok_or_else(invalid_program)?.admissions.iter()
            .find(|claim| claim.first == admission.first && Arc::ptr_eq(&claim.issuance, &admission.issuance))
            .ok_or_else(invalid_program)
    }

    pub(super) fn instruction_admission_mut(&mut self, admission: &SemanticSegmentInstructionAdmission) -> Result<&mut InstructionAdmission, SemanticTransitionError> {
        if !Arc::ptr_eq(&admission.issuer, &self.publication_issuer) { return Err(invalid_program()); }
        self.actor_refresh_program.as_mut().ok_or_else(invalid_program)?.admissions.iter_mut()
            .find(|claim| claim.first == admission.first && Arc::ptr_eq(&claim.issuance, &admission.issuance))
            .ok_or_else(invalid_program)
    }

    pub fn require_segment_instruction_admission(&self, admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease) -> Result<(), SemanticTransitionError> {
        let claim = self.instruction_admission(admission)?;
        if !Arc::ptr_eq(&parent.issuer, &self.publication_issuer) || parent.token != claim.parent_token
            || parent.identity != claim.parent { return Err(invalid_program()); }
        // This metadata-only custody check intentionally remains available when
        // the original effect has unknown completion. It grants no tensor read.
        match self.steps.get(&parent.token) {
            Some(step) if step.identity == Some(claim.parent) => (),
            None if claim.parent_fully_retired => (),
            _ => return Err(invalid_program()),
        }
        Ok(())
    }

    pub fn admitted_segment_parent_identity(&self, admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        Ok(self.instruction_admission(admission)?.parent)
    }

    pub fn admitted_segment_first_program_ordinal(&self, admission: &SemanticSegmentInstructionAdmission) -> Result<u64, SemanticTransitionError> {
        Ok(self.instruction_admission(admission)?.first)
    }

    pub fn admitted_segment_transitions(&self, admission: &SemanticSegmentInstructionAdmission) -> Result<Vec<SemanticTransitionKind>, SemanticTransitionError> {
        let claim = self.instruction_admission(admission)?;
        self.actor_refresh_program.as_ref().expect("original program").steps[claim.first as usize..claim.end as usize]
            .iter().map(|step| step.kind.ok_or_else(invalid_program)).collect()
    }

    pub fn admitted_segment_cold_capacity(&self, admission: &SemanticSegmentInstructionAdmission) -> Result<SemanticSegmentColdCapacity, SemanticTransitionError> {
        let instruction = json_value(&self.instruction_admission(admission)?.instruction)?;
        Ok(SemanticSegmentColdCapacity {
            tensor_content_capacity: usize::try_from(integer(&instruction["tensor_content_capacity"])?).map_err(|_| invalid_program())?,
            model_work_capacity: usize::try_from(integer(&instruction["model_work_capacity"])?).map_err(|_| invalid_program())?,
            segment_capacity_bytes: integer(&instruction["capacity_bytes"])?,
            other_external_cuda_bytes: integer(&instruction["other_external_cuda_bytes"])?,
        })
    }

    /// Join and retire only the original reader. Its numerical allocations,
    /// autograd producers and instruction custody remain retained.
    pub fn handoff_admitted_segment_parent(&mut self, admission: &SemanticSegmentInstructionAdmission,
        parent: &mut SemanticPublishedLease, streams: &[u64]) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        if self.instruction_admission(admission)?.state != InstructionState::Admitted { return Err(invalid_program()); }
        self.checked_step(parent)?;
        let reader = self.checked_reader(parent)?;
        if Arc::strong_count(&reader.aliases) != 1 || self.admitted_transition.is_some() { return Err(invalid_program()); }
        self.require_closed_evaluations()?;
        let streams: Vec<_> = self.step_consumer_streams(parent.token, streams)?.into_iter().collect();
        self.instruction_admission_mut(admission)?.handoff_streams = Some(streams);
        self.instruction_admission_mut(admission)?.state = InstructionState::HandoffEntered;
        self.resolve_admitted_segment_parent(admission, parent)
    }

    /// Observe the already entered original retirement, without another Release
    /// command, reader acquisition, callback or instruction admission.
    pub fn resolve_admitted_segment_parent(&mut self, admission: &SemanticSegmentInstructionAdmission,
        parent: &mut SemanticPublishedLease) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        if self.instruction_admission(admission)?.state == InstructionState::Released { return Ok(()); }
        if self.instruction_admission(admission)?.state != InstructionState::HandoffEntered { return Err(invalid_program()); }
        let streams = self.instruction_admission(admission)?.handoff_streams.clone().ok_or_else(invalid_program)?;
        // The canonical retirement retains its exact event edges, Release
        // command and observation before entry. A retry resolves those owners;
        // an inactive value alone cannot prove a successful submission.
        self.handoff_published_reader(parent, &streams)?;
        self.instruction_admission_mut(admission)?.state = InstructionState::Released;
        Ok(())
    }

    pub fn prepare_admitted_segment_steps(&mut self, admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        transitions: impl ExactSizeIterator<Item = SemanticTransitionKind> + Clone,
        cold_capacity: SemanticSegmentColdCapacity) -> Result<Vec<SemanticPreparedStep>, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        self.checked_step(parent)?;
        if self.instruction_admission(admission)?.state != InstructionState::Released || parent.active
            || transitions.clone().collect::<Vec<_>>() != self.admitted_segment_transitions(admission)?
            || cold_capacity != self.admitted_segment_cold_capacity(admission)? { return Err(invalid_program()); }
        self.instruction_admission_mut(admission)?.state = InstructionState::BuildEntered;
        let steps = self.prepare_segment_steps(transitions, cold_capacity)?;
        let first = steps.first().ok_or_else(invalid_program)?;
        self.require_prepared_program_parent(first, parent)?;
        let claim = self.instruction_admission(admission)?;
        let program = self.actor_refresh_program.as_ref().expect("original program");
        let mapping = steps.iter().zip(&program.steps[claim.first as usize..claim.end as usize])
            .map(|(step, expected)| (step.token, expected.clone())).collect();
        let build = self.prepared_segment.as_mut().expect("original scope");
        build.program_steps = mapping;
        build.program_admission = Some(admission.clone());
        build.parent_quiescent = true;
        self.instruction_admission_mut(admission)?.state = InstructionState::Bound;
        Ok(steps)
    }

    pub(super) fn complete_segment_instruction(&mut self) -> Result<(), SemanticTransitionError> {
        if let Some(admission) = self.prepared_segment.as_ref().and_then(|build| build.program_admission.clone()) {
            if self.instruction_admission(&admission)?.state != InstructionState::Bound { return Err(invalid_program()); }
            self.instruction_admission_mut(&admission)?.state = InstructionState::Completed;
        }
        Ok(())
    }

    pub(super) fn note_instruction_parent_retirement(&mut self, token: u64) {
        if let Some(program) = self.actor_refresh_program.as_mut() {
            for claim in &mut program.admissions {
                if claim.parent_token == token {
                    claim.parent_fully_retired = true;
                    if claim.resources_retired {
                        claim._publication = None;
                        claim._model_owners.clear();
                    }
                }
            }
        }
    }

    pub(super) fn retire_instruction_resources(&mut self) -> Result<(), SemanticTransitionError> {
        let Some(admission) = self.prepared_segment.as_ref().and_then(|build| build.program_admission.clone()) else { return Ok(()); };
        let claim = self.instruction_admission_mut(&admission)?;
        claim.resources_retired = true;
        if claim.parent_fully_retired {
            claim._publication = None;
            claim._model_owners.clear();
        }
        Ok(())
    }

    pub fn admitted_segment_budgets(&self, admission: &SemanticSegmentInstructionAdmission) -> Result<Vec<[u64; 3]>, SemanticTransitionError> {
        let claim = self.instruction_admission(admission)?;
        Ok(self.actor_refresh_program.as_ref().expect("original program").steps[claim.first as usize..claim.end as usize]
            .iter().map(|step| step.budget).collect())
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
        material_u64(&mut bytes, 3);
        bytes.extend_from_slice(program.identity.as_bytes());
        bytes.extend_from_slice(initial.index.as_bytes());
        material_u64(&mut bytes, u64::from(program.original_rng.is_some()));
        if let Some(rng) = program.original_rng {
            for value in [rng.stream_serial, u64::from(rng.family_id), u64::from(rng.proposal)] {
                material_u64(&mut bytes, value);
            }
        }
        material_u64(&mut bytes, program.admissions.len() as u64);
        for claim in &program.admissions {
            for value in [claim.first, claim.end, claim.state as u64, claim.parent.word, claim.parent_token,
                u64::from(claim.parent_fully_retired), u64::from(claim.resources_retired)] {
                material_u64(&mut bytes, value);
            }
            bytes.extend_from_slice(claim.parent.instance.as_bytes());
            bytes.extend_from_slice(claim.instruction_identity.as_bytes());
            material_u64(&mut bytes, claim.instruction.len() as u64);
            bytes.extend_from_slice(&claim.instruction);
            for result in [claim.cold_result, claim.retirement_result] {
                write_instruction_cold_result(&mut bytes, result);
            }
            material_u64(&mut bytes, claim.prepared_cold_results.len() as u64);
            for (ordinal, result) in &claim.prepared_cold_results {
                material_u64(&mut bytes, *ordinal);
                write_instruction_cold_result(&mut bytes, *result);
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
        if !program.main_origins.is_empty() || !program.admissions.is_empty() || self.prepared_segment.is_some() { return Err(invalid_program()); }
        let mut reader = SemanticMaterialReader::new(bytes);
        let magic = b"XLOG-ACTOR-REFRESH-ASSIGNMENTS\0";
        let semantic = SemanticTransitionError::Semantic;
        if reader.take(magic.len()).map_err(semantic)? != magic
            || reader.u64().map_err(semantic)? != 3
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
        let claims = reader.u64().map_err(semantic)?;
        if claims > program.steps.len() as u64 || (claims != 0 && rng.is_none()) { return Err(invalid_program()); }
        let mut admissions = Vec::new();
        for _ in 0..claims {
            let first = reader.u64().map_err(semantic)?;
            let end = reader.u64().map_err(semantic)?;
            let state = match reader.u64().map_err(semantic)? {
                0 => InstructionState::Admitted,
                1 => InstructionState::HandoffEntered,
                2 => InstructionState::Released,
                3 => InstructionState::BuildEntered,
                4 => InstructionState::Bound,
                5 => InstructionState::Completed,
                _ => return Err(invalid_program()),
            };
            let word = reader.u64().map_err(semantic)?;
            let parent_token = reader.u64().map_err(semantic)?;
            let parent_fully_retired = match reader.u64().map_err(semantic)? { 0 => false, 1 => true, _ => return Err(invalid_program()) };
            let resources_retired = match reader.u64().map_err(semantic)? { 0 => false, 1 => true, _ => return Err(invalid_program()) };
            let instance = Identity256::from_bytes(reader.take(32).map_err(semantic)?.try_into().map_err(|_| invalid_program())?);
            let instruction_identity = Identity256::from_bytes(reader.take(32).map_err(semantic)?.try_into().map_err(|_| invalid_program())?);
            let instruction_len = usize::try_from(reader.u64().map_err(semantic)?).map_err(|_| invalid_program())?;
            if instruction_len == 0 || instruction_len > PROGRAM_BYTES { return Err(invalid_program()); }
            let instruction: Arc<[u8]> = Arc::from(reader.take(instruction_len).map_err(semantic)?);
            let original = usize::try_from(first).ok().and_then(|index| program.steps.get(index)).ok_or_else(invalid_program)?;
            if original.kind.is_none() || original.segment_begin != first || original.segment_end != end
                || instruction_projection(&instruction)? != original.instruction
                || Identity256::from_bytes(Sha256::digest(&instruction).into()) != instruction_identity { return Err(invalid_program()); }
            require_completed_instruction_prefix(program, first as usize, &admissions)?;
            let cold_result = read_instruction_cold_result(&mut reader)?;
            let retirement_result = read_instruction_cold_result(&mut reader)?;
            let prepared_count = reader.u64().map_err(semantic)?;
            if prepared_count > end - first { return Err(invalid_program()); }
            let mut prepared_cold_results = Vec::new();
            for _ in 0..prepared_count {
                let ordinal = reader.u64().map_err(semantic)?;
                if ordinal < first || ordinal >= end
                    || program.steps[ordinal as usize].kind != Some(SemanticTransitionKind::Update)
                    || prepared_cold_results.iter().any(|(previous, _)| *previous == ordinal) { return Err(invalid_program()); }
                prepared_cold_results.push((ordinal, read_instruction_cold_result(&mut reader)?));
            }
            // An incomplete restored attempt is retained, not re-issued. Only
            // the original live handle can resolve its asynchronous owners.
            admissions.push(InstructionAdmission { issuance: Arc::new(()), first, end,
                instruction, instruction_identity,
                parent: SemanticPublishedIdentity { instance, word }, parent_token, state,
                cold_work: None, cold_result, retirement_work: None, retirement_result,
                prepared_cold_results,
                handoff_streams: None, parent_fully_retired, resources_retired,
                _publication: None, _model_owners: Vec::new() });
        }
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
        for claim in &admissions {
            if claim.state == InstructionState::Completed {
                for step in &program.steps[claim.first as usize..claim.end as usize] {
                    if step.actor_slot.is_some() && !assignments.iter().any(|assignment| assignment.ordinal == step.ordinal) {
                        return Err(publication_input_error("completed instruction lacks its original native Proposal assignment"));
                    }
                }
            }
        }
        let program = self.actor_refresh_program.as_mut().expect("checked original program");
        program.main_origins = assignments;
        program.original_rng = rng;
        program.admissions = admissions;
        Ok(())
    }
}
