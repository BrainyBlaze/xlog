use super::*;
use serde_json::Value;

const PROGRAM_BYTES: usize = 16 * 1024 * 1024;
const PROGRAM_JSON_DEPTH: u8 = 128;
const CONTEXT_CONTINUATION_MAGIC: &[u8] = b"XLOG-ORIGINAL-MODEL-CONTEXT-CONTINUATION\0";
const CONTEXT_CONTINUATION_BYTES: usize = CONTEXT_CONTINUATION_MAGIC.len() + 8 + 64 + 5 * 8;

struct UniqueJson {
    remaining_depth: u8,
}

impl UniqueJson {
    fn whitespace(input: &mut &str) {
        *input = input.trim_start_matches([' ', '\n', '\r', '\t']);
    }

    fn scalar<'a>(input: &mut &'a str) -> Result<&'a str, SemanticTransitionError> {
        let text = *input;
        if matches!(text.as_bytes().first(), Some(b'{' | b'[')) {
            return Err(invalid_program());
        }
        let mut decoder =
            serde_json::Deserializer::from_str(text).into_iter::<&serde_json::value::RawValue>();
        let raw = decoder
            .next()
            .ok_or_else(invalid_program)?
            .map_err(|_| invalid_program())?;
        *input = &text[decoder.byte_offset()..];
        Ok(raw.get())
    }

    fn read(&self, input: &mut &str) -> Result<Value, SemanticTransitionError> {
        Self::whitespace(input);
        if !matches!(input.as_bytes().first(), Some(b'{' | b'[')) {
            // Only original scalar tokens reach Value's deserializer. Its
            // optional synthetic number/raw-value maps cannot reinterpret a
            // literal object, and integers never pass through f64.
            return serde_json::from_str(Self::scalar(input)?).map_err(|_| invalid_program());
        }
        // Match serde's original finite recursion limit, shared across every
        // container. Never scan a whole RawValue container: that scanner has no
        // recursion check and would repeatedly scan nested subtrees.
        let remaining_depth = self
            .remaining_depth
            .checked_sub(1)
            .filter(|depth| *depth != 0)
            .ok_or_else(invalid_program)?;
        let child = Self { remaining_depth };
        let object = input.starts_with('{');
        let end = if object { '}' } else { ']' };
        *input = &input[1..];
        Self::whitespace(input);
        let mut fields = serde_json::Map::new();
        let mut values = Vec::new();
        if let Some(rest) = input.strip_prefix(end) {
            *input = rest;
        } else {
            loop {
                if object {
                    if !input.starts_with('"') {
                        return Err(invalid_program());
                    }
                    let key: String = serde_json::from_str(Self::scalar(input)?)
                        .map_err(|_| invalid_program())?;
                    if fields.contains_key(&key) {
                        return Err(invalid_program());
                    }
                    Self::whitespace(input);
                    *input = input.strip_prefix(':').ok_or_else(invalid_program)?;
                    fields.insert(key, child.read(input)?);
                } else {
                    values.push(child.read(input)?);
                }
                Self::whitespace(input);
                if let Some(rest) = input.strip_prefix(end) {
                    *input = rest;
                    break;
                }
                *input = input.strip_prefix(',').ok_or_else(invalid_program)?;
                Self::whitespace(input);
            }
        }
        Ok(if object {
            Value::Object(fields)
        } else {
            Value::Array(values)
        })
    }
}

fn json_value(bytes: &[u8]) -> Result<Value, SemanticTransitionError> {
    if bytes.is_empty() || bytes.len() > PROGRAM_BYTES {
        return Err(invalid_program());
    }
    // Exact original bytes, not this comparison encoding, identify the admitted
    // program. One finite depth budget covers the complete original input.
    let mut input = std::str::from_utf8(bytes).map_err(|_| invalid_program())?;
    let value = UniqueJson {
        remaining_depth: PROGRAM_JSON_DEPTH,
    }
    .read(&mut input)?;
    UniqueJson::whitespace(&mut input);
    if !input.is_empty() {
        return Err(invalid_program());
    }
    Ok(value)
}

#[derive(Clone, PartialEq, Eq)]
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
    evaluation: Option<(usize, Option<SemanticEvaluationIntervention>)>,
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
    Cancelled = 6,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct InstructionCheckpointSource {
    native: Identity256,
    checkpoint: Identity256,
    proposal_spent: u64,
}

pub(super) struct InstructionAdmission {
    // Checkpoint restoration retains durable parent identities; live handoff
    // additionally retains the authentic issuing identity with the claim.
    _issuer: Option<Arc<()>>,
    issuance: Arc<()>,
    first: u64,
    end: u64,
    pub(super) instruction: Arc<[u8]>,
    instruction_identity: Identity256,
    assignment_prefix: Identity256,
    assignment_prefix_has_rng: bool,
    checkpoint_source: Option<InstructionCheckpointSource>,
    pub(super) parent: SemanticPublishedIdentity,
    pub(super) parent_token: u64,
    pub(super) state: InstructionState,
    pub(super) cold_work: Option<SemanticColdModelWork>,
    pub(super) cold_result: Option<SemanticColdModelWorkResult>,
    pub(super) retirement_work: Option<SemanticColdModelWork>,
    pub(super) retirement_result: Option<SemanticColdModelWorkResult>,
    pub(super) retirement_cancellation: Option<SemanticPreparedSegmentNonSubmission>,
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
    owner: Arc<()>,
    pub(super) bytes: Vec<u8>,
    pub(super) identity: Identity256,
    pub(super) proposals: u64,
    pub(super) updates: u64,
    pub(super) logical_updates: u64,
    pub(super) steps: Vec<ProgramStep>,
    pub(super) original_rng: Option<SemanticRngBinding>,
    pub(super) main_origins: Vec<MainActorAssignment>,
    admissions: Vec<InstructionAdmission>,
    _evaluation_source_reservation: Option<Arc<SemanticReplayMaterialReservation>>,
}

pub(super) struct PreparedEvaluationProgramAssignments {
    program: Arc<()>,
    initial_index: Identity256,
    rng: Option<SemanticRngBinding>,
    admissions: Vec<InstructionAdmission>,
    assignments: Vec<MainActorAssignment>,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

impl PreparedEvaluationProgramAssignments {
    pub(super) fn require_target(
        &self,
        session: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        let program = session
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let initial = session
            .actor_refresh_initial
            .as_ref()
            .ok_or_else(invalid_program)?;
        if !Arc::ptr_eq(&self.program, &program.owner)
            || initial.index != self.initial_index
            || !program.admissions.is_empty()
            || !program.main_origins.is_empty()
            || program.original_rng.is_some()
        {
            return Err(invalid_program());
        }
        Ok(())
    }

    pub(super) fn install(self, session: &mut SemanticTransitionSession) {
        let program = session
            .actor_refresh_program
            .as_mut()
            .expect("validated original program");
        program.original_rng = self.rng;
        program.admissions = self.admissions;
        program.main_origins = self.assignments;
        program._evaluation_source_reservation = Some(self._reservation);
    }
}

pub(super) struct InitialPrefillProgramAuthority {
    pub(super) owner: Arc<()>,
    pub(super) identity: Identity256,
    pub(super) instruction: Arc<[u8]>,
    pub(super) budget: [u64; 3],
    _reservation: SemanticReplayMaterialReservation,
}

fn original_source_has_model_context_successor(
    program: &ActorRefreshProgram,
    claim: &InstructionAdmission,
) -> bool {
    let Some(first) = program.steps.iter().find(|step| step.kind.is_some()) else {
        return false;
    };
    let Some(source) = program.steps.get(claim.first as usize..claim.end as usize) else {
        return false;
    };
    let Some(next) = program.steps.get(claim.end as usize) else {
        return false;
    };
    let Some(target) = program
        .steps
        .get(claim.end as usize..next.segment_end as usize)
    else {
        return false;
    };
    claim.first == first.ordinal
        && source.len() >= 2
        && source
            .last()
            .is_some_and(|step| step.kind == Some(SemanticTransitionKind::Recompute))
        && source[..source.len() - 1]
            .iter()
            .all(|step| step.kind == Some(SemanticTransitionKind::Proposal))
        && source.iter().all(|step| step.branch.is_none())
        && next.segment_begin == next.ordinal
        && first.request.checked_add(1) == Some(next.request)
        && target.len() == 1
        && next.kind == Some(SemanticTransitionKind::Recompute)
        && next.branch.is_none()
}

impl ActorRefreshProgram {
    pub(super) fn checkpoint_owner_binding(&self) -> (Arc<()>, Identity256) {
        (Arc::clone(&self.owner), self.identity)
    }

    pub(super) fn require_completed_checkpoint_prefix(
        &self,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .admissions
            .iter()
            .any(|claim| claim.state != InstructionState::Completed)
        {
            return Err(publication_input_error(
                "public checkpoint provenance requires its completed original instruction prefix",
            ));
        }
        Ok(())
    }
}

enum OriginalProgramContextSource {
    Live(SemanticSegmentInstructionAdmission),
    Restored(RestoredOriginalProgramContextSource),
}

struct RestoredOriginalProgramContextSource {
    restoration: Arc<state_restoration::OriginalStateMaterialRestore>,
    claim: Arc<()>,
    first: u64,
    native_digest: Identity256,
    assignments_digest: Identity256,
    checkpoint_parent: SemanticPublishedIdentity,
    parent: SemanticPublishedIdentity,
}

/// Immutable future instruction geometry, never a live admission. The original
/// Program moves once; the captured TARGET scope and its tokens remain fixed.
pub(super) struct OriginalProgramContextContinuation {
    source_issuer: Arc<()>,
    target_issuer: Arc<()>,
    source_program: Arc<()>,
    target_program: Arc<()>,
    program_identity: Identity256,
    initial_index: Identity256,
    source: OriginalProgramContextSource,
    source_end: u64,
    source_proposals: u64,
    source_proposal: u64,
    target_steps: Vec<ProgramStep>,
    cold_capacity: SemanticSegmentColdCapacity,
    prepared_steps: Vec<SemanticPreparedStep>,
    transferred: bool,
    admission: Option<SemanticSegmentInstructionAdmission>,
    // Retire payload owners before releasing their original pool charges.
    _frozen_reservation: Option<SemanticReplayMaterialReservation>,
    _mapping_reservation: Option<SemanticReplayMaterialReservation>,
}

impl OriginalProgramContextContinuation {
    pub(super) fn first_program_ordinal(&self) -> u64 {
        self.target_steps[0].ordinal
    }

    pub(super) fn expected_source_proposals(&self) -> u64 {
        self.source_proposals
    }

    pub(super) fn expected_source_proposal(&self) -> u64 {
        self.source_proposal
    }

    pub(super) fn transitions(
        &self,
    ) -> impl ExactSizeIterator<Item = SemanticTransitionKind> + Clone + '_ {
        self.target_steps
            .iter()
            .map(|step| step.kind.expect("frozen original transition"))
    }

    pub(super) fn cold_capacity(&self) -> SemanticSegmentColdCapacity {
        self.cold_capacity
    }

    pub(super) fn cold_budget(&self) -> [u64; 3] {
        self.target_steps[0].budget
    }

    pub(super) fn instruction(&self) -> &[u8] {
        &self.target_steps[0].instruction
    }

    fn source_first(&self) -> u64 {
        match &self.source {
            OriginalProgramContextSource::Live(source) => source.first,
            OriginalProgramContextSource::Restored(source) => source.first,
        }
    }
}

fn decode_original_program_context_mapping(
    program: &ActorRefreshProgram,
    initial_index: Identity256,
    bytes: &[u8],
) -> Result<[u64; 5], SemanticTransitionError> {
    if bytes.len() != CONTEXT_CONTINUATION_BYTES {
        return Err(invalid_program());
    }
    let mut reader = SemanticMaterialReader::new(bytes);
    let semantic = SemanticTransitionError::Semantic;
    if reader
        .take(CONTEXT_CONTINUATION_MAGIC.len())
        .map_err(semantic)?
        != CONTEXT_CONTINUATION_MAGIC
        || reader.u64().map_err(semantic)? != 1
        || reader.take(32).map_err(semantic)? != program.identity.as_bytes()
        || reader.take(32).map_err(semantic)? != initial_index.as_bytes()
    {
        return Err(invalid_program());
    }
    let mut mapping = [0; 5];
    for value in &mut mapping {
        *value = reader.u64().map_err(semantic)?;
    }
    Ok(mapping)
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
pub(super) enum AssignmentReplayProof {
    Unresolved,
    Joined,
}

fn structural_bytes(value: &Value) -> Result<Vec<u8>, SemanticTransitionError> {
    serde_json::to_vec(value).map_err(|_| invalid_program())
}

fn program_step_payload_allocation_bound(
    steps: &[ProgramStep],
) -> Result<usize, SemanticTransitionError> {
    steps.iter().try_fold(0usize, |bytes, step| {
        bytes
            .checked_add(step.instruction.len())
            .and_then(|bytes| bytes.checked_add(step.branch.as_ref().map_or(0, String::len)))
            .ok_or(SemanticTransitionError::GenerationExhausted)
    })
}

fn instruction_projection(bytes: &[u8]) -> Result<Vec<u8>, SemanticTransitionError> {
    if bytes.is_empty() || bytes.len() > PROGRAM_BYTES {
        return Err(invalid_program());
    }
    let mut value = json_value(bytes)?;
    let instruction = value.as_object_mut().ok_or_else(invalid_program)?;
    if let Some(recipe) = instruction
        .get_mut("recipe")
        .filter(|recipe| !recipe.is_null())
    {
        let recipe = recipe.as_object_mut().ok_or_else(invalid_program)?;
        if recipe.len() != 5
            || [
                "source",
                "target",
                "phase_index",
                "completed_updates_index",
                "views",
            ]
            .iter()
            .any(|field| !recipe.contains_key(*field))
        {
            return Err(invalid_program());
        }
        for field in ["phase_index", "completed_updates_index", "views"] {
            recipe.remove(field);
        }
    }
    structural_bytes(&value)
}

fn require_completed_instruction_prefix(
    program: &ActorRefreshProgram,
    start: usize,
    admissions: &[InstructionAdmission],
) -> Result<(), SemanticTransitionError> {
    let original = program.steps.get(start).ok_or_else(invalid_program)?;
    if admissions
        .iter()
        .any(|claim| claim.first == original.ordinal || claim.state != InstructionState::Completed)
    {
        return Err(invalid_program());
    }
    for previous in &program.steps[..start] {
        if previous.kind.is_none()
            || (original.evaluation.is_some()
                && previous.evaluation.is_some()
                && previous.evaluation != original.evaluation)
            || (previous.branch.as_deref() == Some("control")
                && (previous.request != original.request
                    || original.branch.as_deref() != Some("control")))
            || (previous.request == original.request
                && previous.branch.as_deref() == Some("real")
                && original.branch.as_deref() != Some("real"))
        {
            continue;
        }
        if !admissions.iter().any(|claim| {
            claim.state == InstructionState::Completed
                && claim.first <= previous.ordinal
                && previous.ordinal < claim.end
        }) {
            return Err(publication_input_error(
                "original program progression has an uncompleted instruction",
            ));
        }
    }
    Ok(())
}

fn evaluation_instruction_scope(
    value: &Value,
) -> Result<Option<(usize, Option<SemanticEvaluationIntervention>)>, SemanticTransitionError> {
    let Some(sealed) = value.as_object().ok_or_else(invalid_program)?.get("sealed") else {
        return Ok(None);
    };
    let sealed = array(sealed)?;
    if sealed.len() != 2 {
        return Err(invalid_program());
    }
    let member = usize::try_from(integer(&sealed[0])?)
        .ok()
        .and_then(|member| member.checked_add(2))
        .ok_or_else(invalid_program)?;
    let mode = match operation(&sealed[1])? {
        "coding" => None,
        "v0" => Some(SemanticEvaluationIntervention::NoEdit),
        "v1" => Some(SemanticEvaluationIntervention::FixedRuleActivation),
        "v3" => Some(SemanticEvaluationIntervention::Structural),
        _ => return Err(invalid_program()),
    };
    Ok(Some((member, mode)))
}

fn validate_segment(value: &Value) -> Result<(), SemanticTransitionError> {
    let evaluation = evaluation_instruction_scope(value)?;
    let fields = [
        "operation",
        "transitions",
        "capacity_bytes",
        "other_external_cuda_bytes",
        "entropy_threshold",
        "distance_weight",
        "proposal_views",
        "sealed",
    ];
    let segment = object(value, &fields[..if evaluation.is_some() { 8 } else { 7 }])?;
    let mode = operation(&segment["operation"])?;
    if !matches!(mode, "inference" | "training") {
        return Err(invalid_program());
    }
    let transitions = array(&segment["transitions"])?;
    if transitions.is_empty() {
        return Err(invalid_program());
    }
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
    if mode == "training" && updates == 0 {
        return Err(invalid_program());
    }
    if evaluation.is_some()
        && (mode != "inference"
            || proposals == 0
            || updates != 0
            || transitions.len() != proposals + 1
            || operation(transitions.last().ok_or_else(invalid_program)?)? != "recompute"
            || transitions[..proposals]
                .iter()
                .any(|step| step.as_str() != Some("proposal")))
    {
        return Err(invalid_program());
    }
    if integer(&segment["capacity_bytes"])? <= 512 {
        return Err(invalid_program());
    }
    for field in ["other_external_cuda_bytes"] {
        if integer(&segment[field])? == 0 {
            return Err(invalid_program());
        }
    }
    for field in ["entropy_threshold", "distance_weight"] {
        let value = segment[field]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(invalid_program)?;
        if field == "distance_weight" && value < 0.0 {
            return Err(invalid_program());
        }
    }
    let views = array(&segment["proposal_views"])?;
    if views.len() != proposals {
        return Err(invalid_program());
    }
    for view in views {
        let view = array(view)?;
        if view.len() != 2 {
            return Err(invalid_program());
        }
        for coordinate in view {
            integer(coordinate)?;
        }
    }
    Ok(())
}

fn invalid_program() -> SemanticTransitionError {
    publication_input_error("actor refresh requires its complete original ordered program")
}

fn object<'a>(
    value: &'a Value,
    fields: &[&str],
) -> Result<&'a serde_json::Map<String, Value>, SemanticTransitionError> {
    let object = value.as_object().ok_or_else(invalid_program)?;
    if object.len() != fields.len() || fields.iter().any(|field| !object.contains_key(*field)) {
        return Err(invalid_program());
    }
    Ok(object)
}

fn array(value: &Value) -> Result<&[Value], SemanticTransitionError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(invalid_program)
}

fn integer(value: &Value) -> Result<u64, SemanticTransitionError> {
    value.as_u64().ok_or_else(invalid_program)
}

fn resource_budget(value: &Value) -> Result<[u64; 3], SemanticTransitionError> {
    let values = array(value)?;
    if values.len() != 3 {
        return Err(invalid_program());
    }
    let result = [
        integer(&values[0])?,
        integer(&values[1])?,
        integer(&values[2])?,
    ];
    if result[1] == 0 {
        return Err(invalid_program());
    }
    Ok(result)
}

fn write_instruction_cold_result(bytes: &mut Vec<u8>, result: Option<SemanticColdModelWorkResult>) {
    material_u64(bytes, u64::from(result.is_some()));
    if let Some(result) = result {
        for value in [
            result.model_work,
            result.operation_count,
            result.work_bound,
            result.model_calls,
            result.native_work,
        ]
        .into_iter()
        .chain(result.native_events)
        {
            material_u64(bytes, value);
        }
    }
}

fn read_instruction_cold_result(
    reader: &mut SemanticMaterialReader<'_>,
) -> Result<Option<SemanticColdModelWorkResult>, SemanticTransitionError> {
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
            for value in &mut native_events {
                *value = reader.u64().map_err(semantic)?;
            }
            if model_work > work_bound
                || model_calls > operation_count
                || native_events
                    .iter()
                    .try_fold(0u64, |sum, value| sum.checked_add(*value))
                    != Some(native_work)
                || model_work.checked_add(native_work).is_none()
            {
                return Err(invalid_program());
            }
            Ok(Some(SemanticColdModelWorkResult {
                model_work,
                operation_count,
                work_bound,
                model_calls,
                native_work,
                native_events,
            }))
        }
        _ => Err(invalid_program()),
    }
}

fn operation(value: &Value) -> Result<&str, SemanticTransitionError> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(invalid_program)
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
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || value.bytes().all(|byte| byte == b'0')
    {
        return Err(invalid_program());
    }
    Ok(())
}

impl ActorRefreshProgram {
    fn decode(bytes: &[u8]) -> Result<Self, SemanticTransitionError> {
        if bytes.is_empty() || bytes.len() > PROGRAM_BYTES {
            return Err(invalid_program());
        }
        let value = json_value(bytes)?;
        let root = object(
            &value,
            &[
                "kind",
                "run_sha256",
                "resources_sha256",
                "fresh_restore_before",
                "frozen_run_sha256",
                "schedule",
                "budgets",
            ],
        )?;
        if operation(&root["kind"])? != "dlm-new.original-self-source-program/1" {
            return Err(invalid_program());
        }
        for field in ["run_sha256", "resources_sha256", "frozen_run_sha256"] {
            digest(&root[field])?;
        }
        let budgets = object(
            &root["budgets"],
            &[
                "prepare",
                "trajectory-start",
                "restore",
                "proposal",
                "recompute",
                "update",
                "evaluation",
                "checkpoint",
                "retire",
                "delivery",
                "terminal-cleanup",
            ],
        )?;
        let budgets: BTreeMap<_, _> = budgets
            .iter()
            .map(|(name, value)| {
                if name.is_empty() {
                    return Err(invalid_program());
                }
                Ok((name.as_str(), resource_budget(value)?))
            })
            .collect::<Result<_, SemanticTransitionError>>()?;
        let schedule = object(&root["schedule"], &["requests", "updates"])?;
        let requests = array(&schedule["requests"])?;
        if requests.is_empty() {
            return Err(invalid_program());
        }
        match &root["fresh_restore_before"] {
            Value::Array(points) => {
                let mut previous = None;
                if points.is_empty() {
                    return Err(invalid_program());
                }
                for point in points {
                    let point = integer(point)?;
                    if point >= requests.len() as u64
                        || previous.is_some_and(|previous| previous >= point)
                    {
                        return Err(invalid_program());
                    }
                    previous = Some(point);
                }
            }
            point if integer(point)? < requests.len() as u64 => {}
            _ => return Err(invalid_program()),
        }
        let mut steps = Vec::new();
        let mut expected_updates = Vec::new();
        let mut proposals = 0u64;
        let mut updates = 0u64;
        let mut logical_updates = 0u64;
        let mut actor_slots = 0u64;
        let mut evaluation_requests = BTreeSet::new();
        for (request_ordinal, request) in requests.iter().enumerate() {
            let request_object = request.as_object().ok_or_else(invalid_program)?;
            let request_kind = operation(
                request_object
                    .get("operation")
                    .ok_or_else(invalid_program)?,
            )?;
            let phase = Identity256::from_bytes(Sha256::digest(structural_bytes(request)?).into());
            let mut paired_updates: BTreeMap<u64, (u64, Vec<u8>)> = BTreeMap::new();
            let mut paired_proposals: BTreeMap<u64, (u64, Vec<u8>)> = BTreeMap::new();
            let mut branch_updates: BTreeMap<String, u64> = BTreeMap::new();
            let mut branch_proposals: BTreeMap<String, u64> = BTreeMap::new();
            let mut branches = BTreeSet::new();
            let mut paired_instructions: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();
            let entries: Vec<(Value, Option<String>, u64, String, [u64; 3])> = if request_kind
                == "learning-phase"
            {
                let phase_request = object(
                    request,
                    &[
                        "operation",
                        "source",
                        "target",
                        "final_phase",
                        "scientific_phase",
                        "feedback_interventions",
                        "schedule",
                        "terminal_cleanup",
                        "learning_grant_ref",
                    ],
                )?;
                let source = operation(&phase_request["source"])?;
                let target = operation(&phase_request["target"])?;
                if !matches!(
                    (source, target),
                    ("alignment", "fast") | ("fast", "consolidation") | ("consolidation", "fast")
                ) || !matches!(
                    operation(&phase_request["final_phase"])?,
                    "fast" | "consolidation"
                ) || (source != "fast" && phase_request["final_phase"] != "fast")
                    || operation(&phase_request["scientific_phase"])?
                        != if source == "alignment" {
                            "alignment"
                        } else {
                            "consolidation"
                        }
                {
                    return Err(invalid_program());
                }
                operation(&phase_request["learning_grant_ref"])?;
                let interventions = array(&phase_request["feedback_interventions"])?;
                if interventions.len() != 2 {
                    return Err(invalid_program());
                }
                for (mode, intervention) in interventions.iter().enumerate() {
                    let mut original = b"xlog.learning-feedback-intervention.v1\0".to_vec();
                    original.push(mode as u8);
                    let expected: String =
                        original.iter().map(|byte| format!("{byte:02x}")).collect();
                    if operation(intervention)? != expected {
                        return Err(invalid_program());
                    }
                }
                let cleanup = object(
                    &phase_request["terminal_cleanup"],
                    &["operation_ordinal", "budget"],
                )?;
                let budget = array(&cleanup["budget"])?;
                if integer(&cleanup["operation_ordinal"])?
                    != array(&phase_request["schedule"])?.len() as u64
                    || budget.len() != 3
                    || integer(&budget[1])? == 0
                {
                    return Err(invalid_program());
                }
                for value in budget {
                    integer(value)?;
                }
                if resource_budget(&cleanup["budget"])?
                    != *budgets
                        .get("terminal-cleanup")
                        .ok_or_else(invalid_program)?
                {
                    return Err(invalid_program());
                }
                array(request_object.get("schedule").ok_or_else(invalid_program)?)?
                    .iter()
                    .map(|entry| {
                        let entry = object(
                            entry,
                            &[
                                "instruction",
                                "step_ordinal",
                                "operation",
                                "branch",
                                "budget",
                                "comparison",
                            ],
                        )?;
                        let branch = operation(&entry["branch"])?;
                        if !matches!(branch, "source" | "control" | "real") {
                            return Err(invalid_program());
                        }
                        let budget = array(&entry["budget"])?;
                        if budget.len() != 3 || integer(&budget[1])? == 0 {
                            return Err(invalid_program());
                        }
                        for value in budget {
                            integer(value)?;
                        }
                        let entry_operation = operation(&entry["operation"])?;
                        let budget = resource_budget(&entry["budget"])?;
                        if budget != *budgets.get(entry_operation).ok_or_else(invalid_program)? {
                            return Err(invalid_program());
                        }
                        if entry_operation == "evaluation" {
                            let comparison = object(&entry["comparison"], &["case"])?;
                            if comparison["case"] != entry["instruction"]["cohort_case"] {
                                return Err(invalid_program());
                            }
                        } else if !entry["comparison"].is_null() {
                            return Err(invalid_program());
                        }
                        Ok((
                            entry["instruction"].clone(),
                            Some(branch.to_owned()),
                            integer(&entry["step_ordinal"])?,
                            entry_operation.to_owned(),
                            budget,
                        ))
                    })
                    .collect::<Result<_, _>>()?
            } else {
                let mut instruction = request.clone();
                instruction
                    .as_object_mut()
                    .ok_or_else(invalid_program)?
                    .remove("budgets")
                    .ok_or_else(invalid_program)?;
                validate_segment(&instruction)?;
                if let Some((member, mode)) = evaluation_instruction_scope(&instruction)? {
                    if !evaluation_requests.insert((member, mode.map(|mode| mode as u64))) {
                        return Err(invalid_program());
                    }
                }
                let ordinary_budgets =
                    array(request_object.get("budgets").ok_or_else(invalid_program)?)?;
                if ordinary_budgets.len()
                    != array(
                        request_object
                            .get("transitions")
                            .ok_or_else(invalid_program)?,
                    )?
                    .len()
                {
                    return Err(invalid_program());
                }
                array(
                    request_object
                        .get("transitions")
                        .ok_or_else(invalid_program)?,
                )?
                .iter()
                .enumerate()
                .map(|(ordinal, transition)| {
                    let operation = operation(transition)?;
                    let budget = resource_budget(&ordinary_budgets[ordinal])?;
                    if budget != *budgets.get(operation).ok_or_else(invalid_program)? {
                        return Err(invalid_program());
                    }
                    Ok((
                        instruction.clone(),
                        None,
                        ordinal as u64,
                        operation.to_owned(),
                        budget,
                    ))
                })
                .collect::<Result<_, SemanticTransitionError>>()?
            };
            let request_actor_start = actor_slots;
            let mut segment_begin = steps.len() as u64;
            let mut active_segment: Option<(Vec<u8>, Option<String>, usize)> = None;
            for (instruction, branch, step_ordinal, operation_name, budget) in entries {
                let instruction_object = instruction.as_object().ok_or_else(invalid_program)?;
                let instruction_bytes = structural_bytes(&instruction)?;
                let evaluation = evaluation_instruction_scope(&instruction)?;
                let transition = kind(&operation_name);
                let mut segment_end = steps.len() as u64 + 1;
                if transition.is_some() {
                    validate_segment(&instruction)?;
                    let transitions = array(
                        instruction_object
                            .get("transitions")
                            .ok_or_else(invalid_program)?,
                    )?;
                    if transitions.is_empty()
                        || step_ordinal as usize >= transitions.len()
                        || operation(&transitions[step_ordinal as usize])? != operation_name
                    {
                        return Err(invalid_program());
                    }
                    if step_ordinal == 0 {
                        if active_segment.is_some() {
                            return Err(invalid_program());
                        }
                        segment_begin = steps.len() as u64;
                        active_segment =
                            Some((instruction_bytes.clone(), branch.clone(), transitions.len()));
                    }
                    let (original, original_branch, count) =
                        active_segment.as_ref().ok_or_else(invalid_program)?;
                    if *original != instruction_bytes
                        || *original_branch != branch
                        || steps.len() as u64 != segment_begin + step_ordinal
                    {
                        return Err(invalid_program());
                    }
                    segment_end = segment_begin
                        .checked_add(*count as u64)
                        .ok_or_else(invalid_program)?;
                    if step_ordinal as usize + 1 == *count {
                        active_segment = None;
                    }
                } else {
                    if active_segment.is_some() || step_ordinal != 0 {
                        return Err(invalid_program());
                    }
                    match operation_name.as_str() {
                        "evaluation" => {
                            let evaluation = object(
                                &instruction,
                                &[
                                    "cohort_case",
                                    "memory_capacity_bytes",
                                    "forward_calls",
                                    "other_external_cuda_bytes",
                                ],
                            )?;
                            operation(&evaluation["cohort_case"])?;
                            if integer(&evaluation["forward_calls"])? != 1 {
                                return Err(invalid_program());
                            }
                            for field in ["memory_capacity_bytes", "other_external_cuda_bytes"] {
                                if integer(&evaluation[field])? == 0 {
                                    return Err(invalid_program());
                                }
                            }
                        }
                        "trajectory-start" | "restore" => {
                            let restore = object(
                                &instruction,
                                &[
                                    "checkpoint",
                                    "recipe",
                                    "forward_calls",
                                    "other_external_cuda_bytes",
                                ],
                            )?;
                            if integer(&restore["forward_calls"])? != 0
                                || integer(&restore["other_external_cuda_bytes"])? == 0
                                || operation(&restore["checkpoint"])?
                                    != if operation_name == "trajectory-start" {
                                        "source"
                                    } else {
                                        "current"
                                    }
                            {
                                return Err(invalid_program());
                            }
                            if operation_name == "trajectory-start" {
                                if !restore["recipe"].is_null() {
                                    return Err(invalid_program());
                                }
                            } else {
                                let recipe = object(&restore["recipe"], &["source", "target"])?;
                                if !matches!(
                                    (operation(&recipe["source"])?, operation(&recipe["target"])?),
                                    ("alignment", "fast")
                                        | ("fast", "consolidation")
                                        | ("consolidation", "fast")
                                ) {
                                    return Err(invalid_program());
                                }
                            }
                        }
                        "prepare" | "checkpoint" | "retire" | "delivery" => {
                            object(&instruction, &["operation"])?;
                            if operation(
                                instruction_object
                                    .get("operation")
                                    .ok_or_else(invalid_program)?,
                            )? != operation_name
                            {
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
                    paired_instructions
                        .entry(branch_name.clone())
                        .or_default()
                        .push(instruction_bytes.clone());
                }
                if let Some(branch) = &branch {
                    branches.insert(branch.clone());
                }
                if transition == Some(SemanticTransitionKind::Update) {
                    expected_updates.push(
                        serde_json::json!({"request": request_ordinal, "branch": branch,
                        "step_ordinal": step_ordinal, "program_ordinal": steps.len()}),
                    );
                    let ordinal = branch_updates.entry(branch_name.clone()).or_default();
                    let index = if paired {
                        if let Some((index, original)) = paired_updates.get(ordinal) {
                            if original != &instruction_bytes {
                                return Err(invalid_program());
                            }
                            *index
                        } else {
                            let index = logical_updates;
                            paired_updates.insert(*ordinal, (index, instruction_bytes.clone()));
                            logical_updates =
                                logical_updates.checked_add(1).ok_or_else(invalid_program)?;
                            index
                        }
                    } else {
                        let index = logical_updates;
                        logical_updates =
                            logical_updates.checked_add(1).ok_or_else(invalid_program)?;
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
                            if original != &instruction_bytes {
                                return Err(invalid_program());
                            }
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
                let before_proposals = if evaluation.is_some() {
                    step_ordinal
                } else if paired {
                    request_actor_start + branch_proposals.get(&branch_name).copied().unwrap_or(0)
                        - u64::from(transition == Some(SemanticTransitionKind::Proposal))
                } else {
                    actor_slot.unwrap_or(actor_slots)
                };
                steps.push(ProgramStep {
                    ordinal: steps.len() as u64,
                    kind: transition,
                    segment_begin,
                    segment_end,
                    logical_update,
                    actor_slot,
                    before_proposals,
                    phase,
                    instruction: instruction_bytes,
                    budget,
                    request: request_ordinal as u64,
                    branch,
                    evaluation,
                });
            }
            if active_segment.is_some() {
                return Err(invalid_program());
            }
            if request_kind == "learning-phase"
                && (!branches.contains("control")
                    || !branches.contains("real")
                    || branch_updates.get("control") != branch_updates.get("real")
                    || branch_proposals.get("control") != branch_proposals.get("real")
                    || paired_instructions.get("control") != paired_instructions.get("real"))
            {
                return Err(invalid_program());
            }
        }
        if schedule["updates"] != Value::Array(expected_updates) {
            return Err(invalid_program());
        }
        Ok(Self {
            owner: Arc::new(()),
            bytes: bytes.to_vec(),
            identity: Identity256::from_bytes(Sha256::digest(bytes).into()),
            proposals,
            updates,
            logical_updates,
            steps,
            original_rng: None,
            main_origins: Vec::new(),
            admissions: Vec::new(),
            _evaluation_source_reservation: None,
        })
    }
}

fn require_retired_instruction_expense(
    claim: &InstructionAdmission,
) -> Result<(), SemanticTransitionError> {
    if !claim.parent_fully_retired
        || !claim.resources_retired
        || claim.cold_result.is_none()
        || claim.retirement_result.is_none()
        || claim
            .prepared_cold_results
            .iter()
            .any(|(_, result)| result.is_none())
    {
        return Err(publication_input_error(
            "cancelled instruction lacks its complete original retirement and expense",
        ));
    }
    Ok(())
}

fn require_cancelled_instruction_material(
    claim: &InstructionAdmission,
) -> Result<(), SemanticTransitionError> {
    require_retired_instruction_expense(claim)?;
    if claim.checkpoint_source.is_none() {
        return Err(invalid_program());
    }
    Ok(())
}

fn instruction_evaluation(
    program: &ActorRefreshProgram,
    claim: &InstructionAdmission,
) -> Option<(usize, Option<SemanticEvaluationIntervention>)> {
    program.steps[claim.first as usize].evaluation
}

fn evaluation_proposal_capacity(
    program: &ActorRefreshProgram,
    member: usize,
) -> Result<u64, SemanticTransitionError> {
    program
        .steps
        .iter()
        .filter(|step| {
            step.evaluation
                .is_some_and(|(original, _)| original == member)
                && step.kind == Some(SemanticTransitionKind::Proposal)
        })
        .try_fold(0u64, |count, _| {
            count.checked_add(1).ok_or_else(invalid_program)
        })
}

fn same_instruction_expense_scope(
    program: &ActorRefreshProgram,
    current: &InstructionAdmission,
    previous: &InstructionAdmission,
) -> bool {
    instruction_evaluation(program, current)
        .is_none_or(|scope| instruction_evaluation(program, previous) == Some(scope))
}

fn instruction_expense_capacity(
    program: &ActorRefreshProgram,
    claim: &InstructionAdmission,
) -> Result<u64, SemanticTransitionError> {
    match instruction_evaluation(program, claim) {
        Some((member, _)) => evaluation_proposal_capacity(program, member),
        None => Ok(program.proposals),
    }
}

fn instruction_proposals(program: &ActorRefreshProgram, claim: &InstructionAdmission) -> u64 {
    program.steps[claim.first as usize..claim.end as usize]
        .iter()
        .filter(|step| step.kind == Some(SemanticTransitionKind::Proposal))
        .count() as u64
}

fn assignment_material(
    program: Identity256,
    initial_index: Identity256,
    rng: Option<SemanticRngBinding>,
    admissions: &[InstructionAdmission],
    assignments: &[&MainActorAssignment],
) -> Vec<u8> {
    let mut bytes = b"XLOG-ACTOR-REFRESH-ASSIGNMENTS\0".to_vec();
    material_u64(&mut bytes, 6);
    bytes.extend_from_slice(program.as_bytes());
    bytes.extend_from_slice(initial_index.as_bytes());
    material_u64(&mut bytes, u64::from(rng.is_some()));
    if let Some(rng) = rng {
        for value in [
            rng.stream_serial,
            u64::from(rng.family_id),
            u64::from(rng.proposal),
        ] {
            material_u64(&mut bytes, value);
        }
    }
    material_u64(&mut bytes, admissions.len() as u64);
    for claim in admissions {
        for value in [
            claim.first,
            claim.end,
            claim.state as u64,
            claim.parent.word,
            claim.parent_token,
            u64::from(claim.parent_fully_retired),
            u64::from(claim.resources_retired),
        ] {
            material_u64(&mut bytes, value);
        }
        bytes.extend_from_slice(claim.parent.instance.as_bytes());
        bytes.extend_from_slice(claim.parent.logical_digest.as_bytes());
        bytes.extend_from_slice(claim.instruction_identity.as_bytes());
        material_u64(&mut bytes, claim.instruction.len() as u64);
        bytes.extend_from_slice(&claim.instruction);
        bytes.extend_from_slice(claim.assignment_prefix.as_bytes());
        material_u64(&mut bytes, u64::from(claim.assignment_prefix_has_rng));
        material_u64(&mut bytes, u64::from(claim.checkpoint_source.is_some()));
        if let Some(source) = claim.checkpoint_source {
            bytes.extend_from_slice(source.native.as_bytes());
            bytes.extend_from_slice(source.checkpoint.as_bytes());
            material_u64(&mut bytes, source.proposal_spent);
        }
        for result in [claim.cold_result, claim.retirement_result] {
            write_instruction_cold_result(&mut bytes, result);
        }
        material_u64(&mut bytes, claim.prepared_cold_results.len() as u64);
        for (ordinal, result) in &claim.prepared_cold_results {
            material_u64(&mut bytes, *ordinal);
            write_instruction_cold_result(&mut bytes, *result);
        }
    }
    material_u64(&mut bytes, assignments.len() as u64);
    let mut assignments = assignments.to_vec();
    assignments.sort_by_key(|assignment| assignment.ordinal);
    for assignment in assignments {
        material_u64(&mut bytes, assignment.ordinal);
        bytes.extend_from_slice(&publication_abi_bytes(&[assignment.origin]));
        bytes.extend_from_slice(assignment.batch.as_bytes());
        bytes.extend_from_slice(&assignment.batch_bytes);
    }
    bytes
}

fn assignment_material_allocation_bound(
    program: &ActorRefreshProgram,
) -> Result<u64, SemanticTransitionError> {
    let mut bytes = b"XLOG-ACTOR-REFRESH-ASSIGNMENTS\0".len() as u64 + 8 + 64 + 8 + 8 + 8;
    let mut add = |amount: u64| -> Result<(), SemanticTransitionError> {
        bytes = bytes
            .checked_add(amount)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        Ok(())
    };
    let cold_result =
        |result: Option<SemanticColdModelWorkResult>| -> Result<u64, SemanticTransitionError> {
            Ok(8 + if let Some(result) = result {
                (size_of::<[u64; 5]>() + std::mem::size_of_val(&result.native_events)) as u64
            } else {
                0
            })
        };
    if program.original_rng.is_some() {
        add(3 * 8)?;
    }
    for claim in &program.admissions {
        add(7 * 8 + 3 * 32 + 8 + 32 + 8 + 8 + 8)?;
        add(u64::try_from(claim.instruction.len())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?)?;
        if claim.checkpoint_source.is_some() {
            add(2 * 32 + 8)?;
        }
        add(cold_result(claim.cold_result)?)?;
        add(cold_result(claim.retirement_result)?)?;
        for (_, result) in &claim.prepared_cold_results {
            add(8 + cold_result(*result)?)?;
        }
    }
    for assignment in &program.main_origins {
        add(8 + size_of::<SemanticTrainingViewOriginRecord>() as u64 + 32)?;
        add(u64::try_from(assignment.batch_bytes.len())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?)?;
    }
    let pointer_bytes = program
        .main_origins
        .len()
        .checked_mul(3 * size_of::<&MainActorAssignment>())
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(pointer_bytes as u64))
        .and_then(|bytes| {
            bytes.checked_add(
                (size_of::<SemanticTrainingViewOriginRecord>()
                    + size_of::<PublicationMaterialBytesOwner>()
                    + 2 * size_of::<usize>()) as u64,
            )
        })
        .ok_or(SemanticTransitionError::GenerationExhausted)
}

fn require_cancelled_assignment_prefix(
    program: &ActorRefreshProgram,
    initial_index: Identity256,
    rng: Option<SemanticRngBinding>,
    admissions: &[InstructionAdmission],
    assignments: &[MainActorAssignment],
) -> Result<(), SemanticTransitionError> {
    let (claim, previous) = admissions.split_last().ok_or_else(invalid_program)?;
    let source = claim.checkpoint_source.ok_or_else(invalid_program)?;
    let rng = if claim.assignment_prefix_has_rng {
        Some(rng.ok_or_else(invalid_program)?)
    } else {
        if !previous.is_empty() {
            return Err(invalid_program());
        }
        None
    };
    let prefix_assignments: Vec<_> = assignments
        .iter()
        .filter(|assignment| {
            previous.iter().any(|original| {
                original.first <= assignment.ordinal && assignment.ordinal < original.end
            })
        })
        .collect();
    let prefix = assignment_material(
        program.identity,
        initial_index,
        rng,
        previous,
        &prefix_assignments,
    );
    if Identity256::from_bytes(Sha256::digest(&prefix).into()) != claim.assignment_prefix {
        return Err(publication_input_error(
            "cancelled instruction changed its original assignment prefix",
        ));
    }
    let previous_proposals = previous
        .iter()
        .filter(|original| same_instruction_expense_scope(program, claim, original))
        .try_fold(0u64, |spent, original| {
            spent
                .checked_add(instruction_proposals(program, original))
                .ok_or(SemanticTransitionError::GenerationExhausted)
        })?;
    if source.proposal_spent < previous_proposals
        || source.proposal_spent > instruction_expense_capacity(program, claim)?
    {
        return Err(publication_input_error(
            "cancelled instruction changed its original cumulative Proposal expense",
        ));
    }
    Ok(())
}

// One canonical parser serves both live restoration and terminal preflight.
// It authenticates every original instruction and assignment before GPU entry.
fn decode_assignment_material(
    program: &ActorRefreshProgram,
    initial_index: Identity256,
    bytes: &[u8],
) -> Result<
    (
        Option<SemanticRngBinding>,
        Vec<InstructionAdmission>,
        Vec<MainActorAssignment>,
    ),
    SemanticTransitionError,
> {
    let mut reader = SemanticMaterialReader::new(bytes);
    let magic = b"XLOG-ACTOR-REFRESH-ASSIGNMENTS\0";
    let semantic = SemanticTransitionError::Semantic;
    if reader.take(magic.len()).map_err(semantic)? != magic
        || reader.u64().map_err(semantic)? != 6
        || reader.take(32).map_err(semantic)? != program.identity.as_bytes()
        || reader.take(32).map_err(semantic)? != initial_index.as_bytes()
    {
        return Err(invalid_program());
    }
    let rng = match reader.u64().map_err(semantic)? {
        0 => None,
        1 => {
            let stream_serial = reader.u64().map_err(semantic)?;
            let family_id =
                u8::try_from(reader.u64().map_err(semantic)?).map_err(|_| invalid_program())?;
            let proposal =
                u32::try_from(reader.u64().map_err(semantic)?).map_err(|_| invalid_program())?;
            if stream_serial >= 1u64 << 56 {
                return Err(invalid_program());
            }
            Some(SemanticRngBinding {
                model_generation: 0,
                stream_serial,
                family_id,
                proposal,
            })
        }
        _ => return Err(invalid_program()),
    };
    let claims = reader.u64().map_err(semantic)?;
    if claims > program.steps.len() as u64 || (claims != 0 && rng.is_none()) {
        return Err(invalid_program());
    }
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
            6 => InstructionState::Cancelled,
            _ => return Err(invalid_program()),
        };
        let word = reader.u64().map_err(semantic)?;
        let parent_token = reader.u64().map_err(semantic)?;
        let parent_fully_retired = match reader.u64().map_err(semantic)? {
            0 => false,
            1 => true,
            _ => return Err(invalid_program()),
        };
        let resources_retired = match reader.u64().map_err(semantic)? {
            0 => false,
            1 => true,
            _ => return Err(invalid_program()),
        };
        let instance = Identity256::from_bytes(
            reader
                .take(32)
                .map_err(semantic)?
                .try_into()
                .map_err(|_| invalid_program())?,
        );
        let logical_digest = Identity256::from_bytes(
            reader
                .take(32)
                .map_err(semantic)?
                .try_into()
                .map_err(|_| invalid_program())?,
        );
        let instruction_identity = Identity256::from_bytes(
            reader
                .take(32)
                .map_err(semantic)?
                .try_into()
                .map_err(|_| invalid_program())?,
        );
        let instruction_len =
            usize::try_from(reader.u64().map_err(semantic)?).map_err(|_| invalid_program())?;
        if instruction_len == 0 || instruction_len > PROGRAM_BYTES {
            return Err(invalid_program());
        }
        let instruction: Arc<[u8]> = Arc::from(reader.take(instruction_len).map_err(semantic)?);
        let assignment_prefix = Identity256::from_bytes(
            reader
                .take(32)
                .map_err(semantic)?
                .try_into()
                .map_err(|_| invalid_program())?,
        );
        let assignment_prefix_has_rng = match reader.u64().map_err(semantic)? {
            0 => false,
            1 => true,
            _ => return Err(invalid_program()),
        };
        let checkpoint_source = match reader.u64().map_err(semantic)? {
            0 => None,
            1 => Some(InstructionCheckpointSource {
                native: Identity256::from_bytes(
                    reader
                        .take(32)
                        .map_err(semantic)?
                        .try_into()
                        .map_err(|_| invalid_program())?,
                ),
                checkpoint: Identity256::from_bytes(
                    reader
                        .take(32)
                        .map_err(semantic)?
                        .try_into()
                        .map_err(|_| invalid_program())?,
                ),
                proposal_spent: reader.u64().map_err(semantic)?,
            }),
            _ => return Err(invalid_program()),
        };
        let original = usize::try_from(first)
            .ok()
            .and_then(|index| program.steps.get(index))
            .ok_or_else(invalid_program)?;
        if original.kind.is_none()
            || original.segment_begin != first
            || original.segment_end != end
            || instruction_projection(&instruction)? != original.instruction
            || Identity256::from_bytes(Sha256::digest(&instruction).into()) != instruction_identity
        {
            return Err(invalid_program());
        }
        require_completed_instruction_prefix(program, first as usize, &admissions)?;
        let cold_result = read_instruction_cold_result(&mut reader)?;
        let retirement_result = read_instruction_cold_result(&mut reader)?;
        let prepared_count = reader.u64().map_err(semantic)?;
        if prepared_count > end - first {
            return Err(invalid_program());
        }
        let mut prepared_cold_results = Vec::new();
        for _ in 0..prepared_count {
            let ordinal = reader.u64().map_err(semantic)?;
            if ordinal < first
                || ordinal >= end
                || program.steps[ordinal as usize].kind != Some(SemanticTransitionKind::Update)
                || prepared_cold_results
                    .iter()
                    .any(|(previous, _)| *previous == ordinal)
            {
                return Err(invalid_program());
            }
            prepared_cold_results.push((ordinal, read_instruction_cold_result(&mut reader)?));
        }
        // An incomplete restored attempt is retained, not re-issued. Only
        // the original live handle can resolve its asynchronous owners.
        admissions.push(InstructionAdmission {
            _issuer: None,
            issuance: Arc::new(()),
            first,
            end,
            instruction,
            instruction_identity,
            assignment_prefix,
            assignment_prefix_has_rng,
            checkpoint_source,
            parent: SemanticPublishedIdentity {
                instance,
                word,
                logical_digest,
            },
            parent_token,
            state,
            cold_work: None,
            cold_result,
            retirement_work: None,
            retirement_result,
            retirement_cancellation: None,
            prepared_cold_results,
            handoff_streams: None,
            parent_fully_retired,
            resources_retired,
            _publication: None,
            _model_owners: Vec::new(),
        });
    }
    let count = reader.u64().map_err(semantic)?;
    if count > program.proposals || (count != 0 && rng.is_none()) {
        return Err(invalid_program());
    }
    let mut assignments = Vec::new();
    for _ in 0..count {
        let ordinal = reader.u64().map_err(semantic)?;
        let raw = reader
            .take(size_of::<SemanticTrainingViewOriginRecord>())
            .map_err(semantic)?;
        // SAFETY: fixed integer-only ABI with its exact extent checked.
        let origin = unsafe {
            std::ptr::read_unaligned(raw.as_ptr().cast::<SemanticTrainingViewOriginRecord>())
        };
        let batch = Identity256::from_bytes(
            reader
                .take(32)
                .map_err(semantic)?
                .try_into()
                .map_err(|_| invalid_program())?,
        );
        let batch_bytes = reader
            .take(size_of::<SemanticActionBatchReceipt>())
            .map_err(semantic)?
            .to_vec();
        if Identity256::from_bytes(Sha256::digest(&batch_bytes).into()) != batch {
            return Err(invalid_program());
        }
        let address = usize::try_from(ordinal)
            .ok()
            .and_then(|ordinal| program.steps.get(ordinal))
            .ok_or_else(invalid_program)?;
        let slot = address.actor_slot.ok_or_else(invalid_program)?;
        if assignments
            .last()
            .is_some_and(|previous: &MainActorAssignment| previous.ordinal >= ordinal)
            || assignments.iter().any(|previous| previous.origin == origin)
        {
            return Err(invalid_program());
        }
        assignments.push(MainActorAssignment {
            ordinal,
            origin,
            slot,
            batch,
            batch_bytes,
            replay: AssignmentReplayProof::Unresolved,
        });
    }
    reader.finish().map_err(semantic)?;
    for (index, claim) in admissions.iter().enumerate() {
        if claim.state == InstructionState::Cancelled {
            require_cancelled_instruction_material(claim)?;
            if index + 1 != admissions.len()
                || admissions[..index]
                    .iter()
                    .any(|previous| previous.state != InstructionState::Completed)
            {
                return Err(invalid_program());
            }
            for previous in &admissions[..index] {
                require_retired_instruction_expense(previous)?;
            }
            require_cancelled_assignment_prefix(
                program,
                initial_index,
                rng,
                &admissions,
                &assignments,
            )?;
        }
        if claim.state == InstructionState::Completed {
            for step in &program.steps[claim.first as usize..claim.end as usize] {
                if step.actor_slot.is_some()
                    && !assignments
                        .iter()
                        .any(|assignment| assignment.ordinal == step.ordinal)
                {
                    return Err(publication_input_error(
                        "completed instruction lacks its original native Proposal assignment",
                    ));
                }
            }
        }
    }
    Ok((rng, admissions, assignments))
}

impl SemanticTransitionSession {
    pub(super) fn freeze_original_program_context_continuation(
        &mut self,
        target: &Self,
        source: &SemanticSegmentInstructionAdmission,
    ) -> Result<OriginalProgramContextContinuation, SemanticTransitionError> {
        self.freeze_original_program_context_source(
            target,
            OriginalProgramContextSource::Live(source.clone()),
        )
    }

    /// Persist immutable original instruction geometry, not a live admission
    /// or an assertion that SOURCE has already completed or retired.
    pub(super) fn original_program_context_continuation_mapping(
        &self,
        frozen: &OriginalProgramContextContinuation,
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let claim = self.original_program_context_source_claim(&frozen.source)?;
        if frozen.transferred
            || !Arc::ptr_eq(&self.publication_issuer, &frozen.source_issuer)
            || !Arc::ptr_eq(&program.owner, &frozen.source_program)
            || program.identity != frozen.program_identity
            || Identity256::from_bytes(Sha256::digest(&program.bytes).into())
                != frozen.program_identity
            || claim.end != frozen.source_end
            || instruction_proposals(program, claim) != frozen.source_proposals
            || self
                .actor_refresh_initial
                .as_ref()
                .is_none_or(|initial| initial.index != frozen.initial_index)
            || program
                .steps
                .get(frozen.source_end as usize..frozen.target_steps[0].segment_end as usize)
                != Some(frozen.target_steps.as_slice())
        {
            return Err(invalid_program());
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or_else(invalid_program)?;
        if frozen
            ._frozen_reservation
            .as_ref()
            .is_none_or(|reservation| !budget.same_pool(&reservation.budget))
        {
            return Err(invalid_program());
        }
        let allocation_bytes = CONTEXT_CONTINUATION_BYTES
            .checked_add(size_of::<PublicationMaterialBytesOwner>())
            .and_then(|bytes| bytes.checked_add(2 * size_of::<usize>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(allocation_bytes)?;
        budget.require_host_allocation(allocation_bytes)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(CONTEXT_CONTINUATION_BYTES)
            .map_err(|error| runtime_error("original context mapping custody", error))?;
        bytes.extend_from_slice(CONTEXT_CONTINUATION_MAGIC);
        material_u64(&mut bytes, 1);
        bytes.extend_from_slice(frozen.program_identity.as_bytes());
        bytes.extend_from_slice(frozen.initial_index.as_bytes());
        for value in [
            frozen.source_first(),
            frozen.source_end,
            frozen.first_program_ordinal(),
            frozen.target_steps[0].segment_end,
            frozen.source_proposals,
        ] {
            material_u64(&mut bytes, value);
        }
        Ok(SemanticReplayLoadedMaterial {
            bytes: PublicationMaterialBytes::retained(bytes, Some(reservation)),
            decoded: None,
        })
    }

    pub(super) fn require_original_program_context_checkpoint(
        &self,
        frozen: &OriginalProgramContextContinuation,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let claim = self.original_program_context_source_claim(&frozen.source)?;
        if frozen.transferred
            || !Arc::ptr_eq(&self.publication_issuer, &frozen.source_issuer)
            || !Arc::ptr_eq(&program.owner, &frozen.source_program)
            || program.identity != frozen.program_identity
            || Identity256::from_bytes(Sha256::digest(&program.bytes).into())
                != frozen.program_identity
            || claim.state != InstructionState::Completed
            || claim.end != frozen.source_end
            || program
                .admissions
                .iter()
                .any(|claim| claim.first >= frozen.source_end)
            || instruction_proposals(program, claim) != frozen.source_proposals
            || u64::from(parent.header.rng_binding()?.proposal) != frozen.source_proposal
            || self.prepared_segment.is_some()
            || self.readers.len() != 1
            || self.pending_replay_delivery.is_some()
            || self.pending_training_materialization.is_some()
        {
            return Err(invalid_program());
        }
        require_retired_instruction_expense(claim)?;
        require_completed_instruction_prefix(
            program,
            frozen.source_end as usize,
            &program.admissions,
        )
    }

    /// Reconstitute only dormant continuation geometry from the canonical full
    /// SOURCE restore. Durable completion and retirement expense remain exactly
    /// as decoded; a fresh process cannot promote older unfinished claims.
    pub(super) fn freeze_restored_original_program_context_continuation(
        &mut self,
        target: &Self,
        parent: &SemanticPublishedLease,
        native_source: &mut SemanticReplayLoadedMaterial,
        assignments: &SemanticReplayLoadedMaterial,
        mapping: &SemanticReplayLoadedMaterial,
    ) -> Result<OriginalProgramContextContinuation, SemanticTransitionError> {
        self.ensure_quiescent()?;
        target.ensure_quiescent()?;
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        if self.readers.len() != 1 || self.prepared_segment.is_some() {
            return Err(invalid_program());
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or_else(invalid_program)?;
        assignments.require_budget(budget)?;
        mapping.require_budget(budget)?;
        let (native_digest, restored) =
            self.verify_checkpoint_loaded_source_material(parent, native_source)?;
        if !restored {
            return Err(invalid_program());
        }
        let header = native_source.complete()?.bank.header;
        let checkpoint_parent = SemanticPublishedIdentity {
            instance: header.instance,
            word: header.publication_word,
            logical_digest: header.logical_digest,
        };
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let initial_index = self
            .actor_refresh_initial
            .as_ref()
            .ok_or_else(invalid_program)?
            .index;
        let saved =
            decode_original_program_context_mapping(program, initial_index, mapping.as_bytes())?;
        let claim = program
            .admissions
            .iter()
            .find(|claim| claim.first == saved[0] && claim.end == saved[1])
            .ok_or_else(invalid_program)?;
        require_retired_instruction_expense(claim)?;
        if claim.state != InstructionState::Completed
            || claim._issuer.is_some()
            || program
                .admissions
                .iter()
                .any(|claim| claim.first >= saved[1])
        {
            return Err(invalid_program());
        }
        require_completed_instruction_prefix(
            program,
            usize::try_from(saved[1]).map_err(|_| invalid_program())?,
            &program.admissions,
        )?;
        let mut rng = parent.header.rng_binding()?;
        let next = program
            .steps
            .get(usize::try_from(saved[2]).map_err(|_| invalid_program())?)
            .ok_or_else(invalid_program)?;
        rng.proposal = u32::try_from(
            u64::from(rng.proposal)
                .checked_sub(next.before_proposals)
                .ok_or_else(invalid_program)?,
        )
        .map_err(|_| invalid_program())?;
        if program.original_rng.is_none_or(|original| {
            original.family_id != rng.family_id
                || original.stream_serial != rng.stream_serial
                || original.proposal != rng.proposal
        }) {
            return Err(invalid_program());
        }
        let assignments_digest =
            Identity256::from_bytes(Sha256::digest(assignments.as_bytes()).into());
        let (current_assignments, _reservation) = self.actor_refresh_assignment_bytes()?;
        if Identity256::from_bytes(Sha256::digest(&current_assignments).into())
            != assignments_digest
        {
            return Err(invalid_program());
        }
        let source = RestoredOriginalProgramContextSource {
            restoration: Arc::clone(
                self.state_material_restore
                    .as_ref()
                    .ok_or_else(invalid_program)?,
            ),
            claim: Arc::clone(&claim.issuance),
            first: claim.first,
            native_digest,
            assignments_digest,
            checkpoint_parent,
            parent: parent.identity,
        };
        let frozen = self.freeze_original_program_context_source(
            target,
            OriginalProgramContextSource::Restored(source),
        )?;
        if saved
            != [
                frozen.source_first(),
                frozen.source_end,
                frozen.first_program_ordinal(),
                frozen.target_steps[0].segment_end,
                frozen.source_proposals,
            ]
        {
            return Err(invalid_program());
        }
        Ok(frozen)
    }

    fn original_program_context_source_claim<'a>(
        &'a self,
        source: &OriginalProgramContextSource,
    ) -> Result<&'a InstructionAdmission, SemanticTransitionError> {
        match source {
            OriginalProgramContextSource::Live(source) => self.instruction_admission(source),
            OriginalProgramContextSource::Restored(source) => {
                if self
                    .state_material_restore
                    .as_ref()
                    .is_none_or(|restoration| !Arc::ptr_eq(restoration, &source.restoration))
                {
                    return Err(invalid_program());
                }
                self.actor_refresh_program
                    .as_ref()
                    .ok_or_else(invalid_program)?
                    .admissions
                    .iter()
                    .find(|claim| {
                        claim.first == source.first && Arc::ptr_eq(&claim.issuance, &source.claim)
                    })
                    .ok_or_else(invalid_program)
            }
        }
    }

    fn freeze_original_program_context_source(
        &mut self,
        target: &Self,
        source: OriginalProgramContextSource,
    ) -> Result<OriginalProgramContextContinuation, SemanticTransitionError> {
        self.ensure_quiescent()?;
        target.ensure_quiescent()?;
        let claim = self.original_program_context_source_claim(&source)?;
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let target_program = target
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let initial_index = self
            .actor_refresh_initial
            .as_ref()
            .ok_or_else(invalid_program)?
            .index;
        let next = program
            .steps
            .get(claim.end as usize)
            .ok_or_else(invalid_program)?;
        let end = usize::try_from(next.segment_end).map_err(|_| invalid_program())?;
        let target_steps = program
            .steps
            .get(claim.end as usize..end)
            .ok_or_else(invalid_program)?;
        let source_owned = match &source {
            OriginalProgramContextSource::Live(_) => {
                claim.state == InstructionState::Admitted
                    && claim
                        ._issuer
                        .as_ref()
                        .is_some_and(|issuer| Arc::ptr_eq(issuer, &self.publication_issuer))
                    && self.readers.contains_key(&claim.parent_token)
                    && self.prepared_segment.as_ref().is_none_or(|build| {
                        build.structural_only
                            && !build.capturing
                            && !build.submitted
                            && !build.completed
                    })
            }
            OriginalProgramContextSource::Restored(_) => {
                claim.state == InstructionState::Completed
                    && claim._issuer.is_none()
                    && self.prepared_segment.is_none()
                    && claim._publication.is_none()
                    && claim._model_owners.is_empty()
                    && claim.cold_work.is_none()
                    && claim.retirement_work.is_none()
                    && claim.retirement_cancellation.is_none()
                    && require_retired_instruction_expense(claim).is_ok()
            }
        };
        if Arc::ptr_eq(&self.publication_issuer, &target.publication_issuer)
            || !source_owned
            || !original_source_has_model_context_successor(program, claim)
            || program.bytes != target_program.bytes
            || program.identity != target_program.identity
            || !target_program.admissions.is_empty()
            || !target_program.main_origins.is_empty()
            || target_program.original_rng.is_some()
            || target.prepared_segment.is_some()
            || target
                .actor_refresh_initial
                .as_ref()
                .is_none_or(|initial| initial.index != initial_index)
        {
            return Err(invalid_program());
        }
        // One absolute original RNG cursor serves live and restored SOURCE.
        // Restoration has already authenticated this arithmetic against its
        // actual current reader; no completed Proposal is added a second time.
        let source_proposal = u64::from(program.original_rng.ok_or_else(invalid_program)?.proposal)
            .checked_add(next.before_proposals)
            .filter(|proposal| u32::try_from(*proposal).is_ok())
            .ok_or_else(invalid_program)?;
        let budget = self.replay_material_budget.as_ref().ok_or_else(|| {
            publication_input_error("original context continuation requires its material budget")
        })?;
        if target
            .replay_material_budget
            .as_ref()
            .is_none_or(|target_budget| !budget.same_pool(target_budget))
        {
            return Err(publication_input_error(
                "original context continuation requires one shared material pool",
            ));
        }
        let admission_count = program
            .admissions
            .len()
            .checked_add(1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let admission_peak = if program.admissions.capacity() < admission_count {
            admission_count
                .checked_mul(size_of::<InstructionAdmission>())
                .ok_or(SemanticTransitionError::GenerationExhausted)?
        } else {
            0
        };
        let payload_bytes = program_step_payload_allocation_bound(target_steps)?;
        let frozen_bytes = target_steps
            .len()
            .checked_mul(size_of::<ProgramStep>())
            .and_then(|bytes| bytes.checked_add(payload_bytes))
            .and_then(|bytes| bytes.checked_add(size_of::<OriginalProgramContextContinuation>()))
            .and_then(|bytes| bytes.checked_add(admission_peak))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let frozen_reservation = budget.reserve_auxiliary(frozen_bytes)?;
        let parse_reservation = budget.reserve_json_parse(&next.instruction)?;
        budget.require_host_allocation(
            frozen_bytes
                .checked_add(parse_reservation.charged_bytes)
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
        )?;
        let instruction = json_value(&next.instruction)?;
        let frozen = OriginalProgramContextContinuation {
            _frozen_reservation: Some(frozen_reservation),
            _mapping_reservation: None,
            source_issuer: Arc::clone(&self.publication_issuer),
            target_issuer: Arc::clone(&target.publication_issuer),
            source_program: Arc::clone(&program.owner),
            target_program: Arc::clone(&target_program.owner),
            program_identity: program.identity,
            initial_index,
            source_end: claim.end,
            source_proposals: instruction_proposals(program, claim),
            source_proposal,
            source,
            target_steps: target_steps.to_vec(),
            cold_capacity: SemanticSegmentColdCapacity {
                segment_capacity_bytes: integer(&instruction["capacity_bytes"])?,
                other_external_cuda_bytes: integer(&instruction["other_external_cuda_bytes"])?,
            },
            prepared_steps: Vec::new(),
            transferred: false,
            admission: None,
        };
        self.actor_refresh_program
            .as_mut()
            .expect("original Program owner")
            .admissions
            .try_reserve_exact(1)
            .map_err(|error| runtime_error("future original instruction custody", error))?;
        Ok(frozen)
    }

    pub(super) fn bind_future_original_program_preparation(
        &mut self,
        frozen: &mut OriginalProgramContextContinuation,
        steps: &[SemanticPreparedStep],
    ) -> Result<(), SemanticTransitionError> {
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        self.require_original_prepared_structure()?;
        if !Arc::ptr_eq(&self.publication_issuer, &frozen.target_issuer)
            || !Arc::ptr_eq(&program.owner, &frozen.target_program)
            || program.identity != frozen.program_identity
            || Identity256::from_bytes(Sha256::digest(&program.bytes).into())
                != frozen.program_identity
            || program
                .steps
                .get(frozen.source_end as usize..frozen.target_steps[0].segment_end as usize)
                != Some(frozen.target_steps.as_slice())
            || frozen.transferred
            || !frozen.prepared_steps.is_empty()
            || !build.program_steps.is_empty()
            || build.program_admission.is_some()
            || build.cold_capacity != frozen.cold_capacity
        {
            return Err(invalid_program());
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or_else(invalid_program)?;
        if frozen
            ._frozen_reservation
            .as_ref()
            .is_none_or(|original| !budget.same_pool(&original.budget))
            || frozen._mapping_reservation.is_some()
        {
            return Err(invalid_program());
        }
        let node =
            OwnedPublicationMaterial::tree_node_bound(size_of::<u64>(), size_of::<ProgramStep>())?;
        let payload_bytes = program_step_payload_allocation_bound(&frozen.target_steps)?;
        let mapping_bytes = steps
            .len()
            .checked_mul(node)
            .and_then(|bytes| bytes.checked_add(size_of::<BTreeMap<u64, ProgramStep>>()))
            .and_then(|bytes| {
                steps
                    .len()
                    .checked_mul(size_of::<SemanticPreparedStep>())
                    .and_then(|handles| bytes.checked_add(handles))
            })
            .and_then(|bytes| bytes.checked_add(payload_bytes))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mapping_reservation = budget.reserve_auxiliary(mapping_bytes)?;
        budget.require_host_allocation(mapping_bytes)?;
        let mapping = self.original_segment_step_mapping(&frozen.target_steps, steps)?;
        let mut retained = Vec::new();
        retained
            .try_reserve_exact(steps.len())
            .map_err(|error| runtime_error("future original prepared custody", error))?;
        retained.extend_from_slice(steps);
        self.prepared_segment
            .as_mut()
            .expect("original future scope")
            .program_steps = mapping;
        frozen.prepared_steps = retained;
        frozen._mapping_reservation = Some(mapping_reservation);
        Ok(())
    }

    pub(super) fn require_frozen_original_program_scope(
        &self,
        frozen: &OriginalProgramContextContinuation,
    ) -> Result<(), SemanticTransitionError> {
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or_else(invalid_program)?;
        if !Arc::ptr_eq(&self.publication_issuer, &frozen.target_issuer)
            || frozen
                ._frozen_reservation
                .as_ref()
                .is_none_or(|original| !budget.same_pool(&original.budget))
            || frozen
                ._mapping_reservation
                .as_ref()
                .is_none_or(|original| !budget.same_pool(&original.budget))
            || frozen.prepared_steps.is_empty()
            || build.tokens.len() != frozen.prepared_steps.len()
            || build.program_steps.len() != frozen.target_steps.len()
            || build.cold_capacity != frozen.cold_capacity
            || build.cancelled.load(Ordering::Acquire)
            || frozen
                .prepared_steps
                .iter()
                .zip(&frozen.target_steps)
                .zip(&build.tokens)
                .any(|((step, expected), token)| {
                    step.token != *token
                        || build
                            .check_retained(step, &self.publication_issuer)
                            .is_err()
                        || build.program_steps.get(token) != Some(expected)
                        || build.requested_kind(step, &self.publication_issuer).ok()
                            != expected.kind
                })
        {
            return Err(invalid_program());
        }
        Ok(())
    }

    pub(super) fn validate_transfer_original_program_context_continuation(
        &self,
        source: &Self,
        frozen: &OriginalProgramContextContinuation,
        source_parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        source.checked_step(source_parent)?;
        source.checked_reader(source_parent)?;
        self.require_frozen_original_program_scope(frozen)?;
        let claim = source.original_program_context_source_claim(&frozen.source)?;
        let anchor = source.require_model_context_acknowledgement_parent(self, source_parent)?;
        if let OriginalProgramContextSource::Restored(restored) = &frozen.source {
            if restored.parent != anchor {
                return Err(invalid_program());
            }
            source.require_model_context_checkpoint_anchor_digest(
                anchor,
                restored.checkpoint_parent,
                restored.native_digest,
            )?;
            let (assignments, _reservation) = source.actor_refresh_assignment_bytes()?;
            if Identity256::from_bytes(Sha256::digest(&assignments).into())
                != restored.assignments_digest
            {
                return Err(invalid_program());
            }
            require_retired_instruction_expense(claim)?;
        }
        let program = source
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let target_program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let target_build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        let mut rng = source_parent.header.rng_binding()?;
        let source_proposal = u64::from(rng.proposal);
        rng.proposal = u32::try_from(
            u64::from(rng.proposal)
                .checked_sub(frozen.target_steps[0].before_proposals)
                .ok_or_else(invalid_program)?,
        )
        .map_err(|_| invalid_program())?;
        if frozen.transferred
            || source_proposal != frozen.source_proposal
            || source.replay_material_budget.as_ref().is_none_or(|budget| {
                frozen
                    ._frozen_reservation
                    .as_ref()
                    .is_none_or(|original| !budget.same_pool(&original.budget))
            })
            || !Arc::ptr_eq(&source.publication_issuer, &frozen.source_issuer)
            || !Arc::ptr_eq(&program.owner, &frozen.source_program)
            || !Arc::ptr_eq(&target_program.owner, &frozen.target_program)
            || program.identity != frozen.program_identity
            || Identity256::from_bytes(Sha256::digest(&program.bytes).into())
                != frozen.program_identity
            || program.bytes != target_program.bytes
            || program
                .steps
                .get(frozen.source_end as usize..frozen.target_steps[0].segment_end as usize)
                != Some(frozen.target_steps.as_slice())
            || claim.state != InstructionState::Completed
            || claim.end != frozen.source_end
            || program
                .admissions
                .iter()
                .any(|claim| claim.first >= frozen.source_end)
            || instruction_proposals(program, claim) != frozen.source_proposals
            || source.prepared_segment.is_some()
            || !claim.resources_retired
            || target_build.program_admission.is_some()
            || !target_build.finished
            || target_build.capturing
            || target_build.submitted
            || target_build.completed
            || program.original_rng.is_none_or(|original| {
                original.family_id != rng.family_id
                    || original.stream_serial != rng.stream_serial
                    || original.proposal != rng.proposal
            })
            || !target_program.admissions.is_empty()
            || !target_program.main_origins.is_empty()
            || target_program.original_rng.is_some()
            || self
                .actor_refresh_initial
                .as_ref()
                .is_none_or(|initial| initial.index != frozen.initial_index)
            || source
                .actor_refresh_initial
                .as_ref()
                .is_none_or(|initial| initial.index != frozen.initial_index)
        {
            return Err(invalid_program());
        }
        require_completed_instruction_prefix(
            program,
            frozen.source_end as usize,
            &program.admissions,
        )?;
        Ok(())
    }

    // The ordinary continuation owner keeps these exact Sessions and frozen
    // scope exclusive from validation through the known CUDA publication
    // outcome. All refusals precede that publication; this move cannot refuse.
    pub(super) fn transfer_original_program_context_continuation(
        &mut self,
        source: &mut Self,
        frozen: &mut OriginalProgramContextContinuation,
    ) {
        self.actor_refresh_program = source.actor_refresh_program.take();
        frozen.transferred = true;
    }

    pub(super) fn require_frozen_original_program_checkpoint_source(
        &self,
        frozen: &OriginalProgramContextContinuation,
        admission: Option<&SemanticSegmentInstructionAdmission>,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.require_closed_evaluations()?;
        self.require_frozen_original_program_scope(frozen)?;
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        if !frozen.transferred
            || !Arc::ptr_eq(&program.owner, &frozen.source_program)
            || program.identity != frozen.program_identity
            || Identity256::from_bytes(Sha256::digest(&program.bytes).into())
                != frozen.program_identity
            || program
                .steps
                .get(frozen.source_end as usize..frozen.target_steps[0].segment_end as usize)
                != Some(frozen.target_steps.as_slice())
            || self.initial_prefill.is_some()
            || self.task_observed
            || self.policy.as_ref().is_some_and(|policy| policy.tape_live)
            || self
                .steps
                .iter()
                .any(|(token, step)| step.identity.is_none() && !build.tokens.contains(token))
            || !build.finished
            || build.capturing
            || build.submitted
            || build.completed
        {
            return Err(invalid_program());
        }
        match admission {
            None => {
                if frozen.admission.is_some() || build.program_admission.is_some() {
                    return Err(invalid_program());
                }
            }
            Some(admission) => {
                if frozen
                    .admission
                    .as_ref()
                    .is_none_or(|original| !original.same_handle(admission))
                    || build
                        .program_admission
                        .as_ref()
                        .is_none_or(|original| !original.same_handle(admission))
                    || self.instruction_admission(admission)?.state != InstructionState::Admitted
                {
                    return Err(invalid_program());
                }
            }
        }
        Ok(())
    }

    pub(super) fn admit_future_original_program_instruction(
        &mut self,
        frozen: &mut OriginalProgramContextContinuation,
        parent: &SemanticPublishedLease,
        instruction_bytes: &[u8],
    ) -> Result<SemanticSegmentInstructionAdmission, SemanticTransitionError> {
        self.require_frozen_original_program_checkpoint_source(frozen, None)?;
        let admission = self.issue_original_segment_instruction(
            parent,
            frozen.first_program_ordinal(),
            instruction_bytes,
        )?;
        self.prepared_segment
            .as_mut()
            .expect("same captured future scope")
            .program_admission = Some(admission.clone());
        frozen.admission = Some(admission.clone());
        Ok(admission)
    }

    pub(super) fn enter_future_original_program_execution(
        &mut self,
        frozen: &OriginalProgramContextContinuation,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.require_frozen_original_program_scope(frozen)?;
        self.require_segment_instruction_admission(admission, parent)?;
        self.checked_step(parent)?;
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        if frozen
            .admission
            .as_ref()
            .is_none_or(|original| !original.same_handle(admission))
            || build
                .program_admission
                .as_ref()
                .is_none_or(|original| !original.same_handle(admission))
            || self.instruction_admission(admission)?.state != InstructionState::Released
            || parent.active
            || self.readers.contains_key(&parent.token)
            || !build.finished
            || build.capturing
            || build.submitted
            || build.completed
        {
            return Err(invalid_program());
        }
        self.instruction_admission_mut(admission)?.state = InstructionState::Bound;
        self.prepared_segment
            .as_mut()
            .expect("same captured future scope")
            .parent_quiescent = true;
        Ok(())
    }

    pub(super) fn prepare_evaluation_program_assignments(
        &self,
        source: Option<&[u8]>,
        source_rng: SemanticRngBinding,
    ) -> Result<Option<PreparedEvaluationProgramAssignments>, SemanticTransitionError> {
        let Some(program) = self.actor_refresh_program.as_ref() else {
            if source.is_some() {
                return Err(invalid_program());
            }
            return Ok(None);
        };
        let source = source.ok_or_else(invalid_program)?;
        let initial = self
            .actor_refresh_initial
            .as_ref()
            .ok_or_else(invalid_program)?;
        if !program.admissions.is_empty()
            || !program.main_origins.is_empty()
            || program.original_rng.is_some()
        {
            return Err(invalid_program());
        }
        let bytes = source
            .len()
            .checked_mul(8)
            .and_then(|bytes| {
                bytes.checked_add(size_of::<PreparedEvaluationProgramAssignments>() + 64)
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = Arc::new(
            self.replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .reserve_auxiliary(bytes)?,
        );
        let (rng, admissions, assignments) =
            decode_assignment_material(program, initial.index, source)?;
        if admissions.is_empty()
            || admissions.iter().any(|claim| {
                claim.state != InstructionState::Completed
                    || instruction_evaluation(program, claim).is_some()
            })
        {
            return Err(invalid_program());
        }
        let evaluation_start = program
            .steps
            .iter()
            .find(|step| step.evaluation.is_some())
            .ok_or_else(invalid_program)?
            .ordinal as usize;
        require_completed_instruction_prefix(program, evaluation_start, &admissions)?;
        let last = admissions.last().ok_or_else(invalid_program)?;
        let last_step = program
            .steps
            .get(last.end as usize - 1)
            .ok_or_else(invalid_program)?;
        let original_rng = rng.ok_or_else(invalid_program)?;
        if last_step.kind != Some(SemanticTransitionKind::Recompute)
            || original_rng.stream_serial != source_rng.stream_serial
            || original_rng.family_id != source_rng.family_id
            || u64::from(original_rng.proposal).checked_add(last_step.before_proposals)
                != Some(u64::from(source_rng.proposal))
        {
            return Err(publication_input_error(
                "evaluation source assignments differ from their selected donor cursor",
            ));
        }
        Ok(Some(PreparedEvaluationProgramAssignments {
            program: Arc::clone(&program.owner),
            initial_index: initial.index,
            rng,
            admissions,
            assignments,
            _reservation: reservation,
        }))
    }

    fn evaluation_program_member(&self) -> Option<usize> {
        self.codebooks
            .model_catalogue
            .as_ref()
            .and_then(|(_, member)| (*member >= 2).then_some(*member))
    }

    fn program_quantities(
        &self,
        program: &ActorRefreshProgram,
    ) -> Result<[u64; 2], SemanticTransitionError> {
        if let Some(member) = self.evaluation_program_member() {
            let proposals = evaluation_proposal_capacity(program, member)?;
            if proposals == 0 {
                return Err(invalid_program());
            }
            Ok([proposals, 0])
        } else {
            Ok([program.proposals, program.updates])
        }
    }

    fn require_program_evaluation_scope(
        &self,
        step: &ProgramStep,
    ) -> Result<(), SemanticTransitionError> {
        match self.evaluation_program_member() {
            Some(member) => {
                let mode = self
                    .task
                    .as_ref()
                    .ok_or_else(invalid_program)?
                    .0
                    .spec
                    .evaluation_intervention;
                if step.evaluation != Some((member, mode))
                    || self.evaluation_fork.as_ref().is_none_or(|origin| {
                        origin.member != member || origin.intervention() != mode
                    })
                {
                    return Err(invalid_program());
                }
            }
            None if step.evaluation.is_some() => return Err(invalid_program()),
            None => {}
        }
        Ok(())
    }

    pub fn bind_actor_refresh_program(
        &mut self,
        program_bytes: &[u8],
    ) -> Result<[u64; 2], SemanticTransitionError> {
        self.ensure_quiescent()?;
        if let Some(program) = &self.actor_refresh_program {
            if program.bytes != program_bytes {
                return Err(invalid_program());
            }
            return self.program_quantities(program);
        }
        if self.prepared_segment.is_some() {
            return Err(invalid_program());
        }
        let program = ActorRefreshProgram::decode(program_bytes)?;
        let quantities = self.program_quantities(&program)?;
        self.actor_refresh_program = Some(program);
        Ok(quantities)
    }

    pub(super) fn initial_prefill_program_authority(
        &self,
        memory_capacity_bytes: u64,
        other_external_cuda_bytes: u64,
    ) -> Result<InitialPrefillProgramAuthority, SemanticTransitionError> {
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let first = program
            .steps
            .iter()
            .find(|step| step.kind.is_some())
            .ok_or_else(invalid_program)?;
        if !program.admissions.is_empty()
            || program.original_rng.is_some()
            || !program.main_origins.is_empty()
            || first.segment_begin != first.ordinal
            || first.branch.is_some()
        {
            return Err(publication_input_error(
                "initial prefill requires its original program before instruction execution",
            ));
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let _parse = budget.reserve_json_parse(&program.bytes)?;
        let value = json_value(&program.bytes)?;
        let preparation = resource_budget(&value["budgets"]["prepare"])?;
        let instruction = json_value(&first.instruction)?;
        let capacity = SemanticSegmentColdCapacity {
            segment_capacity_bytes: integer(&instruction["capacity_bytes"])?,
            other_external_cuda_bytes: integer(&instruction["other_external_cuda_bytes"])?,
        };
        if capacity.segment_capacity_bytes != memory_capacity_bytes
            || capacity.other_external_cuda_bytes != other_external_cuda_bytes
        {
            return Err(publication_input_error(
                "initial prefill capacity differs from its original source instruction",
            ));
        }
        let retained = size_of::<InitialPrefillProgramAuthority>()
            .checked_add(first.instruction.len())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = budget.reserve_auxiliary(retained)?;
        Ok(InitialPrefillProgramAuthority {
            owner: Arc::clone(&program.owner),
            identity: program.identity,
            instruction: Arc::from(first.instruction.as_slice()),
            budget: preparation,
            _reservation: reservation,
        })
    }

    pub(super) fn instruction_prepares_original_model_context(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<bool, SemanticTransitionError> {
        let original = self.instruction_admission(admission)?;
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        Ok(original_source_has_model_context_successor(
            program, original,
        ))
    }

    pub(super) fn require_initial_prefill_program_authority(
        &self,
        original: &InitialPrefillProgramAuthority,
    ) -> Result<(), SemanticTransitionError> {
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        if !Arc::ptr_eq(&program.owner, &original.owner) || program.identity != original.identity {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
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
        if self.prepared_segment.is_some() {
            return Err(invalid_program());
        }
        self.issue_original_segment_instruction(parent, first_program_ordinal, instruction_bytes)
    }

    fn issue_original_segment_instruction(
        &mut self,
        parent: &SemanticPublishedLease,
        first_program_ordinal: u64,
        instruction_bytes: &[u8],
    ) -> Result<SemanticSegmentInstructionAdmission, SemanticTransitionError> {
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        if self.readers.len() != 1 {
            return Err(invalid_program());
        }
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let start = usize::try_from(first_program_ordinal).map_err(|_| invalid_program())?;
        let original = program.steps.get(start).ok_or_else(invalid_program)?;
        self.require_program_evaluation_scope(original)?;
        if instruction_projection(instruction_bytes)? != original.instruction {
            return Err(invalid_program());
        }
        let end = usize::try_from(original.segment_end).map_err(|_| invalid_program())?;
        if original.kind.is_none()
            || original.segment_begin != first_program_ordinal
            || end > program.steps.len()
        {
            return Err(invalid_program());
        }
        if program
            .admissions
            .iter()
            .any(|claim| claim.state == InstructionState::Cancelled)
        {
            return Err(publication_input_error(
                "cancelled instruction cannot admit another segment",
            ));
        }
        // A SOURCE checkpoint forks independent control/real execution. Their
        // equal logical assignments do not make either branch's effects receipts
        // for the other. Prior real/ordinary work, and this branch's prefix, must
        // have completed under native-issued claims before its next segment.
        require_completed_instruction_prefix(program, start, &program.admissions)?;
        let mut rng = parent.header.rng_binding()?;
        rng.proposal = u32::try_from(
            u64::from(rng.proposal)
                .checked_sub(original.before_proposals)
                .ok_or_else(invalid_program)?,
        )
        .map_err(|_| invalid_program())?;
        let expected_rng = if original.evaluation.is_some() {
            Some(
                self.evaluation_fork
                    .as_ref()
                    .ok_or_else(invalid_program)?
                    .rng_basis(),
            )
        } else {
            program.original_rng
        };
        if let Some(previous) = expected_rng {
            if previous.family_id != rng.family_id
                || previous.stream_serial != rng.stream_serial
                || previous.proposal != rng.proposal
                || (original.evaluation.is_some()
                    && previous.model_generation != rng.model_generation)
            {
                return Err(invalid_program());
            }
        } else if original.before_proposals != 0 {
            return Err(invalid_program());
        }
        let evaluation = original.evaluation;
        // Freeze the actual source prefix before the first admission installs
        // its RNG binding or any admission changes the canonical assignment map.
        let (assignment_bytes, _assignment_reservation) = self.actor_refresh_assignment_bytes()?;
        let assignment_prefix = Identity256::from_bytes(Sha256::digest(assignment_bytes).into());
        let assignment_prefix_has_rng = program.original_rng.is_some();
        let issuance = Arc::new(());
        let admission = SemanticSegmentInstructionAdmission {
            issuer: Arc::clone(&self.publication_issuer),
            issuance: Arc::clone(&issuance),
            first: first_program_ordinal,
        };
        let claim = InstructionAdmission {
            _issuer: Some(Arc::clone(&self.publication_issuer)),
            issuance,
            first: first_program_ordinal,
            end: end as u64,
            instruction: Arc::from(instruction_bytes),
            instruction_identity: Identity256::from_bytes(Sha256::digest(instruction_bytes).into()),
            assignment_prefix,
            assignment_prefix_has_rng,
            checkpoint_source: None,
            parent: parent.identity,
            parent_token: parent.token,
            state: InstructionState::Admitted,
            cold_work: None,
            cold_result: None,
            retirement_work: None,
            retirement_result: None,
            retirement_cancellation: None,
            prepared_cold_results: Vec::new(),
            handoff_streams: None,
            parent_fully_retired: false,
            resources_retired: false,
            _publication: Some(Arc::clone(
                self.publication
                    .as_ref()
                    .ok_or(SemanticTransitionError::NotBound)?,
            )),
            _model_owners: parent._model_owners.clone(),
        };
        let program = self
            .actor_refresh_program
            .as_mut()
            .expect("checked original program");
        program
            .admissions
            .try_reserve(1)
            .map_err(|error| runtime_error("original instruction custody", error))?;
        if evaluation.is_none() {
            program.original_rng = Some(rng);
        }
        program.admissions.push(claim);
        Ok(admission)
    }

    pub(super) fn instruction_admission(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<&InstructionAdmission, SemanticTransitionError> {
        if !Arc::ptr_eq(&admission.issuer, &self.publication_issuer) {
            return Err(invalid_program());
        }
        self.actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?
            .admissions
            .iter()
            .find(|claim| {
                claim.first == admission.first && Arc::ptr_eq(&claim.issuance, &admission.issuance)
            })
            .ok_or_else(invalid_program)
    }

    pub(super) fn instruction_admission_mut(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<&mut InstructionAdmission, SemanticTransitionError> {
        if !Arc::ptr_eq(&admission.issuer, &self.publication_issuer) {
            return Err(invalid_program());
        }
        self.actor_refresh_program
            .as_mut()
            .ok_or_else(invalid_program)?
            .admissions
            .iter_mut()
            .find(|claim| {
                claim.first == admission.first && Arc::ptr_eq(&claim.issuance, &admission.issuance)
            })
            .ok_or_else(invalid_program)
    }

    /// Bind the original sealed source to this native claim before any charge,
    /// handoff, cold callback or initializer. The authenticated checkpoint owner
    /// supplies its full digest and actual cumulative expense; this native owner
    /// compares the exact assignment prefix it retained before admission.
    /// A repeated binding is identical.
    pub fn bind_segment_checkpoint_source(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        source_material: &[u8],
        source_checkpoint: Identity256,
        source_proposal_spent: u64,
        source_assignments: &[u8],
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let (digest, _) = self.verify_checkpoint_source_material(parent, source_material)?;
        self.bind_segment_checkpoint_source_digest(
            admission,
            digest,
            source_checkpoint,
            source_proposal_spent,
            source_assignments,
        )
    }

    /// Bind an authenticated source from the original material pool, without a copy.
    pub fn bind_segment_loaded_checkpoint_source(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        source_material: &mut SemanticReplayLoadedMaterial,
        source_checkpoint: Identity256,
        source_proposal_spent: u64,
        source_assignments: &[u8],
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let (digest, _) = self.verify_checkpoint_loaded_source_material(parent, source_material)?;
        self.bind_segment_checkpoint_source_digest(
            admission,
            digest,
            source_checkpoint,
            source_proposal_spent,
            source_assignments,
        )
    }

    pub(super) fn bind_segment_checkpoint_source_digest(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        digest: Identity256,
        source_checkpoint: Identity256,
        source_proposal_spent: u64,
        source_assignments: &[u8],
    ) -> Result<(), SemanticTransitionError> {
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let current = self.instruction_admission(admission)?;
        let capacity = instruction_expense_capacity(program, current)?;
        let spent = program
            .admissions
            .iter()
            .take_while(|claim| !Arc::ptr_eq(&claim.issuance, &admission.issuance))
            .filter(|claim| same_instruction_expense_scope(program, current, claim))
            .try_fold(0u64, |spent, claim| {
                // Every serialized report and retirement bit in the retained
                // prefix is final before this source can authorize a charge.
                require_retired_instruction_expense(claim)?;
                spent
                    .checked_add(instruction_proposals(program, claim))
                    .ok_or(SemanticTransitionError::GenerationExhausted)
            })?;
        if source_proposal_spent < spent || source_proposal_spent > capacity {
            return Err(invalid_program());
        }
        let source = InstructionCheckpointSource {
            native: digest,
            checkpoint: source_checkpoint,
            proposal_spent: source_proposal_spent,
        };
        let claim = self.instruction_admission_mut(admission)?;
        if claim.state != InstructionState::Admitted
            || Identity256::from_bytes(Sha256::digest(source_assignments).into())
                != claim.assignment_prefix
            || claim
                .checkpoint_source
                .is_some_and(|original| original != source)
        {
            return Err(invalid_program());
        }
        claim.checkpoint_source = Some(source);
        Ok(())
    }

    pub fn require_segment_instruction_admission(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        let claim = self.instruction_admission(admission)?;
        if !Arc::ptr_eq(&parent.issuer, &self.publication_issuer)
            || parent.token != claim.parent_token
            || parent.identity != claim.parent
        {
            return Err(invalid_program());
        }
        // This metadata-only custody check intentionally remains available when
        // the original effect has unknown completion. It grants no tensor read.
        match self.steps.get(&parent.token) {
            Some(step) if step.identity == Some(claim.parent) => (),
            None if claim.parent_fully_retired => (),
            _ => return Err(invalid_program()),
        }
        Ok(())
    }

    pub fn admitted_segment_parent_identity(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> Result<SemanticPublishedIdentity, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        Ok(self.instruction_admission(admission)?.parent)
    }

    pub fn admitted_segment_first_program_ordinal(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<u64, SemanticTransitionError> {
        Ok(self.instruction_admission(admission)?.first)
    }

    pub fn admitted_segment_transitions(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<Vec<SemanticTransitionKind>, SemanticTransitionError> {
        self.admitted_segment_steps(admission)?
            .iter()
            .map(|step| step.kind.ok_or_else(invalid_program))
            .collect()
    }

    pub(super) fn admitted_segment_steps(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<&[ProgramStep], SemanticTransitionError> {
        let claim = self.instruction_admission(admission)?;
        let first = usize::try_from(claim.first).map_err(|_| invalid_program())?;
        let end = usize::try_from(claim.end).map_err(|_| invalid_program())?;
        self.actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?
            .steps
            .get(first..end)
            .ok_or_else(invalid_program)
    }

    pub fn admitted_segment_cold_capacity(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<SemanticSegmentColdCapacity, SemanticTransitionError> {
        let instruction = json_value(&self.instruction_admission(admission)?.instruction)?;
        Ok(SemanticSegmentColdCapacity {
            segment_capacity_bytes: integer(&instruction["capacity_bytes"])?,
            other_external_cuda_bytes: integer(&instruction["other_external_cuda_bytes"])?,
        })
    }

    /// Join and retire only the original reader. Its numerical allocations,
    /// autograd producers and instruction custody remain retained.
    pub fn handoff_admitted_segment_parent(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &mut SemanticPublishedLease,
        streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        if self.instruction_admission(admission)?.state != InstructionState::Admitted {
            return Err(invalid_program());
        }
        self.checked_step(parent)?;
        let reader = self.checked_reader(parent)?;
        if Arc::strong_count(&reader.aliases) != 1 || self.admitted_transition.is_some() {
            return Err(invalid_program());
        }
        self.require_closed_evaluations()?;
        let streams: Vec<_> = self
            .step_consumer_streams(parent.token, streams)?
            .into_iter()
            .collect();
        self.instruction_admission_mut(admission)?.handoff_streams = Some(streams);
        self.instruction_admission_mut(admission)?.state = InstructionState::HandoffEntered;
        self.resolve_admitted_segment_parent(admission, parent)
    }

    /// Observe the already entered original retirement, without another Release
    /// command, reader acquisition, callback or instruction admission.
    pub fn resolve_admitted_segment_parent(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &mut SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        if self.instruction_admission(admission)?.state == InstructionState::Released {
            return Ok(());
        }
        if self.instruction_admission(admission)?.state != InstructionState::HandoffEntered {
            return Err(invalid_program());
        }
        let streams = self
            .instruction_admission(admission)?
            .handoff_streams
            .clone()
            .ok_or_else(invalid_program)?;
        // The canonical retirement retains its exact event edges, Release
        // command and observation before entry. A retry resolves those owners;
        // an inactive value alone cannot prove a successful submission.
        self.handoff_published_reader(parent, &streams)?;
        self.instruction_admission_mut(admission)?.state = InstructionState::Released;
        Ok(())
    }

    /// Issue this instruction's exact allocation roster while retaining its
    /// original source reader. No initializer, release or outcome is entered.
    pub fn prepare_admitted_segment_structure(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        transitions: impl ExactSizeIterator<Item = SemanticTransitionKind> + Clone,
        cold_capacity: SemanticSegmentColdCapacity,
    ) -> Result<Vec<SemanticPreparedStep>, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        self.checked_step(parent)?;
        let state = self.instruction_admission(admission)?.state;
        if !matches!(
            state,
            InstructionState::Admitted | InstructionState::Released
        ) || (state == InstructionState::Admitted) != parent.active
            || transitions.clone().collect::<Vec<_>>()
                != self.admitted_segment_transitions(admission)?
            || cold_capacity != self.admitted_segment_cold_capacity(admission)?
        {
            return Err(invalid_program());
        }
        self.prepare_segment_steps_from_source(
            transitions,
            cold_capacity,
            PreparedSegmentSource::AdmittedPublication,
            Some((admission, parent)),
        )
    }

    /// Admit recording only after positive retirement of the original source.
    /// Resolve the same retained initializer and return the same issued handles.
    pub fn enter_admitted_segment_recording(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
    ) -> Result<Vec<SemanticPreparedStep>, SemanticTransitionError> {
        self.require_segment_instruction_admission(admission, parent)?;
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        if parent.active
            || self.readers.contains_key(&parent.token)
            || build
                .program_admission
                .as_ref()
                .is_none_or(|original| !original.same_handle(admission))
            || !matches!(
                self.instruction_admission(admission)?.state,
                InstructionState::Released
                    | InstructionState::BuildEntered
                    | InstructionState::Bound
            )
        {
            return Err(invalid_program());
        }
        let handles = self.enter_segment_recording()?;
        self.prepared_segment
            .as_mut()
            .expect("original initialized instruction")
            .parent_quiescent = true;
        self.instruction_admission_mut(admission)?.state = InstructionState::Bound;
        Ok(handles)
    }

    // Bind the issued complete allocation roster before the first initializer
    // effect. This is original claim custody, not execution or completion.
    pub(super) fn bind_original_segment_preparation(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
        parent: &SemanticPublishedLease,
        steps: &[SemanticPreparedStep],
    ) -> Result<(), SemanticTransitionError> {
        let first = steps.first().ok_or_else(invalid_program)?;
        self.require_original_prepared_program_parent(first, parent)?;
        let claim = self.instruction_admission(admission)?;
        if !matches!(
            claim.state,
            InstructionState::Admitted | InstructionState::Released
        ) {
            return Err(invalid_program());
        }
        let program = self
            .actor_refresh_program
            .as_ref()
            .expect("original program");
        let expected = &program.steps[claim.first as usize..claim.end as usize];
        let mapping = self.original_segment_step_mapping(expected, steps)?;
        let build = self.prepared_segment.as_mut().expect("original scope");
        build.program_steps = mapping;
        build.program_admission = Some(admission.clone());
        Ok(())
    }

    fn original_segment_step_mapping(
        &self,
        expected: &[ProgramStep],
        steps: &[SemanticPreparedStep],
    ) -> Result<BTreeMap<u64, ProgramStep>, SemanticTransitionError> {
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        if steps.len() != expected.len()
            || build.tokens.len() != steps.len()
            || steps
                .iter()
                .zip(expected)
                .zip(&build.tokens)
                .any(|((step, expected), token)| {
                    step.token != *token
                        || build
                            .check_retained(step, &self.publication_issuer)
                            .is_err()
                        || build.requested_kind(step, &self.publication_issuer).ok()
                            != expected.kind
                })
        {
            return Err(invalid_program());
        }
        Ok(steps
            .iter()
            .zip(expected)
            .map(|(step, expected)| (step.token, expected.clone()))
            .collect())
    }

    pub(super) fn complete_segment_instruction(&mut self) -> Result<(), SemanticTransitionError> {
        if let Some(admission) = self
            .prepared_segment
            .as_ref()
            .and_then(|build| build.program_admission.clone())
        {
            if self.instruction_admission(&admission)?.state != InstructionState::Bound {
                return Err(invalid_program());
            }
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
        let Some(admission) = self
            .prepared_segment
            .as_ref()
            .and_then(|build| build.program_admission.clone())
        else {
            return Ok(());
        };
        let claim = self.instruction_admission_mut(&admission)?;
        claim.resources_retired = true;
        if claim.parent_fully_retired {
            claim._publication = None;
            claim._model_owners.clear();
        }
        Ok(())
    }

    pub fn admitted_segment_budgets(
        &self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<Vec<[u64; 3]>, SemanticTransitionError> {
        let claim = self.instruction_admission(admission)?;
        Ok(self
            .actor_refresh_program
            .as_ref()
            .expect("original program")
            .steps[claim.first as usize..claim.end as usize]
            .iter()
            .map(|step| step.budget)
            .collect())
    }

    pub(super) fn register_prepared_actor_refresh_origin(
        &mut self,
        step: &SemanticPreparedStep,
        origin: SemanticTrainingViewOriginRecord,
        batch: Identity256,
        batch_bytes: Vec<u8>,
    ) -> Result<(), SemanticTransitionError> {
        let Some(program) = self.actor_refresh_program.as_mut() else {
            return Ok(());
        };
        let build = self.prepared_segment.as_ref().ok_or_else(invalid_program)?;
        let address = build
            .program_steps
            .get(&step.token)
            .ok_or_else(invalid_program)?;
        let slot = address.actor_slot.ok_or_else(invalid_program)?;
        if let Some(previous) = program
            .main_origins
            .iter()
            .find(|previous| previous.ordinal == address.ordinal)
        {
            if previous.slot != slot
                || previous.origin != origin
                || previous.batch != batch
                || previous.batch_bytes != batch_bytes
            {
                return Err(invalid_program());
            }
        } else {
            if program
                .main_origins
                .iter()
                .any(|previous| previous.origin == origin)
                || program.main_origins.len() as u64 >= program.proposals
            {
                return Err(invalid_program());
            }
            if batch_bytes.len() != size_of::<SemanticActionBatchReceipt>()
                || Identity256::from_bytes(Sha256::digest(&batch_bytes).into()) != batch
            {
                return Err(invalid_program());
            }
            program.main_origins.push(MainActorAssignment {
                ordinal: address.ordinal,
                origin,
                slot,
                batch,
                batch_bytes,
                replay: AssignmentReplayProof::Joined,
            });
        }
        Ok(())
    }

    /// Native-issued assignments retained by the original authenticated task
    /// checkpoint. Complete replay materials remain in their existing custody.
    pub fn actor_refresh_assignment_material(&self) -> Result<Vec<u8>, SemanticTransitionError> {
        if self.replay_material_budget.is_some() {
            return Err(publication_input_error(
                "bounded actor assignment export requires its original loaded owner",
            ));
        }
        Ok(self.actor_refresh_assignment_bytes()?.0)
    }

    pub fn actor_refresh_loaded_assignment_material(
        &self,
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        if self.replay_material_budget.is_none() {
            return Err(publication_input_error(
                "loaded actor assignments require their original capacity pool",
            ));
        }
        let (bytes, reservation) = self.actor_refresh_assignment_bytes()?;
        Ok(SemanticReplayLoadedMaterial {
            bytes: PublicationMaterialBytes::retained(bytes, reservation),
            decoded: None,
        })
    }

    fn actor_refresh_assignment_bytes(
        &self,
    ) -> Result<(Vec<u8>, Option<SemanticReplayMaterialReservation>), SemanticTransitionError> {
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let initial = self
            .actor_refresh_initial
            .as_ref()
            .ok_or_else(invalid_program)?;
        let reservation = self
            .replay_material_budget
            .as_ref()
            .map(|budget| budget.reserve_auxiliary(assignment_material_allocation_bound(program)?))
            .transpose()?;
        let assignments: Vec<_> = program.main_origins.iter().collect();
        Ok((
            assignment_material(
                program.identity,
                initial.index,
                program.original_rng,
                &program.admissions,
                &assignments,
            ),
            reservation,
        ))
    }

    /// Seal the original non-submitted instruction after actual owner retirement
    /// and all native preparation/retirement reports are positively complete.
    /// This metadata operation cannot acquire a reader or publish a successor.
    pub fn cancelled_segment_assignment_material(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<Vec<u8>, SemanticTransitionError> {
        if self.replay_material_budget.is_some() {
            return Err(publication_input_error(
                "bounded cancelled assignments require their original loaded owner",
            ));
        }
        self.complete_cancelled_assignment_material(admission)?;
        self.actor_refresh_assignment_material()
    }

    pub fn cancelled_segment_loaded_assignment_material(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        if self.replay_material_budget.is_none() {
            return Err(publication_input_error(
                "loaded cancelled assignments require their original capacity pool",
            ));
        }
        self.complete_cancelled_assignment_material(admission)?;
        self.actor_refresh_loaded_assignment_material()
    }

    fn complete_cancelled_assignment_material(
        &mut self,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<(), SemanticTransitionError> {
        if self.prepared_segment.is_some() || self.captured.is_some() {
            return Err(publication_input_error(
                "cancelled instruction still owns prepared execution",
            ));
        }
        let claim = self.instruction_admission(admission)?;
        if !matches!(
            claim.state,
            InstructionState::Admitted
                | InstructionState::Released
                | InstructionState::BuildEntered
                | InstructionState::Bound
                | InstructionState::Cancelled
        ) || claim.retirement_cancellation.is_none()
        {
            return Err(publication_input_error(
                "cancelled checkpoint requires the original whole non-submission proof",
            ));
        }
        require_cancelled_instruction_material(claim)?;
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let (last, previous) = program
            .admissions
            .split_last()
            .ok_or_else(invalid_program)?;
        if !Arc::ptr_eq(&last.issuance, &admission.issuance)
            || previous
                .iter()
                .any(|claim| claim.state != InstructionState::Completed)
        {
            return Err(invalid_program());
        }
        for previous in previous {
            require_retired_instruction_expense(previous)?;
        }
        self.instruction_admission_mut(admission)?.state = InstructionState::Cancelled;
        Ok(())
    }

    /// Verify terminal instruction custody before allocating a Session or
    /// restoring a model. Returned expense counts original charged Proposals,
    /// not executions, outcomes, or a refundable continuation allowance.
    pub fn verify_cancelled_segment_assignment_material(
        program_bytes: &[u8],
        initial_index: Identity256,
        bytes: &[u8],
    ) -> Result<
        Option<(
            u64,
            u64,
            SemanticPublishedIdentity,
            u64,
            Identity256,
            Identity256,
        )>,
        SemanticTransitionError,
    > {
        let program = ActorRefreshProgram::decode(program_bytes)?;
        let (_, admissions, _) = decode_assignment_material(&program, initial_index, bytes)?;
        let Some(claim) = admissions
            .last()
            .filter(|claim| claim.state == InstructionState::Cancelled)
        else {
            return Ok(None);
        };
        let source = claim.checkpoint_source.ok_or_else(invalid_program)?;
        // The source may already include charged work from retired sibling
        // trajectories. Preserve that cumulative expense without pretending
        // their admissions belong to this physical native branch.
        let charged = source
            .proposal_spent
            .checked_add(instruction_proposals(&program, claim))
            .filter(|charged| {
                instruction_expense_capacity(&program, claim)
                    .is_ok_and(|capacity| *charged <= capacity)
            })
            .ok_or_else(invalid_program)?;
        Ok(Some((
            claim.first,
            claim.end,
            claim.parent,
            charged,
            source.native,
            source.checkpoint,
        )))
    }

    /// Authenticate the original reader before any admitted instruction.
    pub(super) fn require_initial_model_context_checkpoint_boundary(
        &self,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_original_consumer_step(parent)?;
        self.checked_reader(parent)?;
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        if !parent.active
            || self.prepared_segment.is_some()
            || !program.admissions.is_empty()
            || program.original_rng.is_some()
            || !program.main_origins.is_empty()
            || self.actor_refresh_initial.is_none()
            || parent.header.abi != 1
            || parent.header.terminal != 0
            || parent
                .directory
                .iter()
                .find(|range| (range.role, range.index) == (30, 0))
                .is_none_or(|range| range.length_bytes != size_of::<IntentQueueHeader>() as u64)
            || parent
                .directory
                .iter()
                .find(|range| (range.role, range.index) == (32, 0))
                .is_none_or(|range| {
                    range.length_bytes != size_of::<AcknowledgementQueueHeader>() as u64
                })
        {
            return Err(publication_input_error(
                "checkpoint is not its original pre-instruction source boundary",
            ));
        }
        Ok(())
    }

    /// Authenticate attachment presence from the native instruction lifecycle,
    /// complete current publication and sealed delivery queues, never an ordinal.
    fn verify_evaluation_program_checkpoint_presence(
        native: &mut SemanticReplayLoadedMaterial,
        origin: &SemanticEvaluationForkOrigin,
        initial_index: Identity256,
        assignments: Option<&[u8]>,
        original_program: &[u8],
        attachment_present: bool,
    ) -> Result<(), SemanticTransitionError> {
        if attachment_present {
            return Err(invalid_program());
        }
        let assignments = assignments.ok_or_else(invalid_program)?;
        let budget = native
            .bytes
            .budget()
            .ok_or(SemanticTransitionError::NotBound)?
            .clone();
        let bytes = original_program
            .len()
            .checked_mul(12)
            .and_then(|bytes| bytes.checked_add(assignments.len().checked_mul(12)?))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let _scratch = budget.reserve_auxiliary(bytes)?;
        let _parse = budget.reserve_json_parse(original_program)?;
        let program = ActorRefreshProgram::decode(original_program)?;
        let (rng, admissions, actor_assignments) =
            decode_assignment_material(&program, initial_index, assignments)?;
        let scope = (origin.member, origin.intervention());
        if admissions.is_empty()
            || admissions.iter().any(|claim| {
                let evaluation = instruction_evaluation(&program, claim);
                evaluation.is_some_and(|original| original != scope)
                    || (evaluation.is_none() && claim.state != InstructionState::Completed)
            })
        {
            return Err(invalid_program());
        }
        let original_count = admissions
            .iter()
            .take_while(|claim| instruction_evaluation(&program, claim).is_none())
            .count();
        if original_count == 0
            || admissions[original_count..]
                .iter()
                .any(|claim| instruction_evaluation(&program, claim).is_none())
        {
            return Err(invalid_program());
        }
        let original_assignments: Vec<_> = actor_assignments
            .iter()
            .filter(|assignment| {
                program.steps[assignment.ordinal as usize]
                    .evaluation
                    .is_none()
            })
            .collect();
        let prefix = assignment_material(
            program.identity,
            initial_index,
            rng,
            &admissions[..original_count],
            &original_assignments,
        );
        if origin.source_program_assignments_digest()
            != Some(Identity256::from_bytes(Sha256::digest(&prefix).into()))
        {
            return Err(publication_input_error(
                "evaluation checkpoint changed its original donor assignment history",
            ));
        }
        let evaluation_start = program
            .steps
            .iter()
            .find(|step| step.evaluation == Some(scope))
            .ok_or_else(invalid_program)?
            .ordinal as usize;
        require_completed_instruction_prefix(
            &program,
            evaluation_start,
            &admissions[..original_count],
        )?;
        let current = &admissions[original_count..];
        if current.len() > 1 {
            return Err(invalid_program());
        }
        let completed = current
            .first()
            .filter(|claim| claim.state == InstructionState::Completed)
            .map_or(0, |claim| instruction_proposals(&program, claim));
        let material = native.complete()?;
        let header = material.bank.header;
        let basis = origin.rng_basis();
        if header.abi != 1
            || header.model_generation != u64::from(material.bank.state.model_generation)
            || header.stream_serial != material.bank.state.stream_serial
            || header.family_id != u64::from(material.bank.state.family_id)
            || header.model_generation != u64::from(basis.model_generation)
            || header.stream_serial != basis.stream_serial
            || header.family_id != u64::from(basis.family_id)
            || u64::from(basis.proposal).checked_add(completed) != Some(header.proposal)
        {
            return Err(publication_input_error(
                "evaluation checkpoint changed its original local random cursor",
            ));
        }
        Ok(())
    }

    pub fn verify_model_context_checkpoint_presence(
        native: &mut SemanticReplayLoadedMaterial,
        model_context: Option<&SemanticFrozenModelContext>,
        initial_index: Identity256,
        assignments: Option<&[u8]>,
        original_program: &[u8],
        attachment_present: bool,
    ) -> Result<(), SemanticTransitionError> {
        let budget = native
            .bytes
            .budget()
            .ok_or(SemanticTransitionError::NotBound)?
            .clone();
        let state = native.checkpoint_proof()?;
        state.require_frozen_model_context(model_context)?;
        match model_context {
            None => {
                return if attachment_present {
                    Err(publication_input_error(
                        "ordinary checkpoint has no dormant model context",
                    ))
                } else {
                    Ok(())
                };
            }
            Some(context) => {
                context.require_budget(&budget)?;
                let saved = context
                    .catalogue
                    .context_for_header(&state.owner.codebook_header)?;
                if saved.member != context.member {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                match context.member {
                    0 => {}
                    1 => {
                        return if attachment_present {
                            Err(publication_input_error(
                                "selected target checkpoint cannot contain a dormant attachment",
                            ))
                        } else {
                            Ok(())
                        };
                    }
                    member => {
                        let origin = state
                            .evaluation_fork_origin()
                            .ok_or(SemanticTransitionError::ObservationMismatch)?
                            .clone();
                        if origin.member != member {
                            return Err(SemanticTransitionError::ObservationMismatch);
                        }
                        return Self::verify_evaluation_program_checkpoint_presence(
                            native,
                            &origin,
                            initial_index,
                            assignments,
                            original_program,
                            attachment_present,
                        );
                    }
                }
            }
        }
        let _program_parse = budget.reserve_json_parse(original_program)?;
        let scratch = original_program
            .len()
            .checked_mul(12)
            .and_then(|bytes| bytes.checked_add(assignments.map_or(0, <[u8]>::len).checked_mul(8)?))
            .and_then(|bytes| bytes.checked_add(native.as_bytes().len().checked_mul(6)?))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        budget.require_host_allocation(scratch)?;
        let _scratch = budget.reserve_auxiliary(scratch)?;
        let program = ActorRefreshProgram::decode(original_program)?;
        let (rng, admissions, actor_assignments) =
            match assignments.filter(|bytes| !bytes.is_empty()) {
                Some(bytes) => decode_assignment_material(&program, initial_index, bytes)?,
                None => (None, Vec::new(), Vec::new()),
            };
        let material = native.complete()?;
        let header = material.bank.header;
        if header.abi != 1
            || header.publication_word == 0
            || header.model_generation != u64::from(material.bank.state.model_generation)
            || header.stream_serial != material.bank.state.stream_serial
            || header.family_id != u64::from(material.bank.state.family_id)
        {
            return Err(invalid_program());
        }
        let entries = material
            .ranges
            .iter()
            .find(|item| (item.range.role, item.range.index) == (30, 0))
            .ok_or_else(invalid_program)?;
        let payload = material
            .ranges
            .iter()
            .find(|item| (item.range.role, item.range.index) == (31, 0))
            .ok_or_else(invalid_program)?;
        let acknowledgements = material
            .ranges
            .iter()
            .find(|item| (item.range.role, item.range.index) == (32, 0))
            .ok_or_else(invalid_program)?;
        let intents = output_intents_from_ranges(entries, payload)?;
        let acknowledged =
            acknowledgement_records(&acknowledgements.bytes, acknowledgements.capacity)?;
        if admissions.is_empty() {
            if attachment_present
                || !actor_assignments.is_empty()
                || header.terminal != 0
                || !intents.is_empty()
                || !acknowledged.is_empty()
                || rng.is_some_and(|rng| {
                    header.stream_serial != rng.stream_serial
                        || header.family_id != u64::from(rng.family_id)
                        || header.proposal != u64::from(rng.proposal)
                })
            {
                return Err(publication_input_error(
                    "initial checkpoint predates any dormant model context",
                ));
            }
            return Ok(());
        }
        let first = program
            .steps
            .iter()
            .find(|step| step.kind.is_some())
            .ok_or_else(invalid_program)?;
        let claim = admissions.last().ok_or_else(invalid_program)?;
        let next = program
            .steps
            .get(usize::try_from(claim.end).map_err(|_| invalid_program())?)
            .ok_or_else(invalid_program)?;
        let source_steps = program
            .steps
            .get(claim.first as usize..claim.end as usize)
            .ok_or_else(invalid_program)?;
        let original_rng = rng.ok_or_else(invalid_program)?;
        if admissions.len() != 1
            || claim.state != InstructionState::Completed
            || claim.first != first.ordinal
            || source_steps.len() < 2
            || source_steps
                .last()
                .is_none_or(|step| step.kind != Some(SemanticTransitionKind::Recompute))
            || source_steps[..source_steps.len() - 1].iter().any(|step| {
                step.kind != Some(SemanticTransitionKind::Proposal) || step.branch.is_some()
            })
            || next.ordinal != claim.end
            || next.segment_begin != next.ordinal
            || next.request != first.request.checked_add(1).ok_or_else(invalid_program)?
            || next.segment_end != next.ordinal.checked_add(1).ok_or_else(invalid_program)?
            || next.kind != Some(SemanticTransitionKind::Recompute)
            || next.branch.is_some()
            || header.stream_serial != original_rng.stream_serial
            || header.family_id != u64::from(original_rng.family_id)
            || header.proposal
                != u64::from(original_rng.proposal)
                    .checked_add(next.before_proposals)
                    .ok_or_else(invalid_program)?
            || actor_assignments.len() as u64 != instruction_proposals(&program, claim)
            || actor_assignments.iter().any(|assignment| {
                assignment.ordinal < claim.first || assignment.ordinal >= claim.end
            })
            || intents.is_empty()
            || intents.len() != acknowledged.len()
            || intents.iter().any(|intent| {
                !acknowledged
                    .iter()
                    .any(|record| record.stable_identity == intent.stable_identity)
            })
        {
            return Err(invalid_program());
        }
        require_retired_instruction_expense(claim)?;
        require_completed_instruction_prefix(&program, claim.end as usize, &admissions)?;
        if !attachment_present {
            return Err(publication_input_error(
                "completed source checkpoint requires its saved dormant model context",
            ));
        }
        Ok(())
    }

    /// Restore every native-issued assignment from the authenticated checkpoint.
    /// An unavailable episode remains unresolved, not dropped or authenticated.
    /// Actor preparation joins its complete original replay before any draw.
    pub fn restore_actor_refresh_assignments<B: AsRef<[u8]>>(
        &mut self,
        bytes: &[u8],
        episodes: &[(SemanticReplayMaterial, Identity256, B)],
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_quiescent()?;
        let program = self
            .actor_refresh_program
            .as_ref()
            .ok_or_else(invalid_program)?;
        let initial = self
            .actor_refresh_initial
            .as_ref()
            .ok_or_else(invalid_program)?;
        if !program.main_origins.is_empty()
            || !program.admissions.is_empty()
            || self.prepared_segment.is_some()
        {
            return Err(invalid_program());
        }
        let (rng, admissions, mut assignments) =
            decode_assignment_material(program, initial.index, bytes)?;
        if admissions
            .iter()
            .any(|claim| claim.state == InstructionState::Cancelled)
        {
            return Err(publication_input_error(
                "cancelled checkpoint has no live actor execution",
            ));
        }
        for assignment in &mut assignments {
            let mut matches = 0usize;
            for (material, identity, bytes) in episodes {
                if *identity != assignment.batch {
                    continue;
                }
                let (actual, actor) =
                    actor_refresh::initial_episode_actor(material, *identity, bytes.as_ref())?;
                if actual == assignment.origin && actor {
                    matches += 1;
                }
            }
            if matches > 1 {
                return Err(invalid_program());
            }
            assignment.replay = if matches == 1 {
                AssignmentReplayProof::Joined
            } else {
                AssignmentReplayProof::Unresolved
            };
        }
        let program = self
            .actor_refresh_program
            .as_mut()
            .expect("checked original program");
        program.main_origins = assignments;
        program.original_rng = rng;
        program.admissions = admissions;
        Ok(())
    }
}
