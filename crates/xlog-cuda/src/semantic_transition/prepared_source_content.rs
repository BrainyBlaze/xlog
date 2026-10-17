//! Original SOURCE allocations and aliases retained before numerical admission.

use super::resident_model_memory::ResidentModelMemoryOwner;
use super::*;

/// Original decoded inputs used by the same private-child constructor.
/// None of these metadata values grants numerical access or a native allowance.
pub struct SemanticSourceContentGeometrySeed {
    pub material: SemanticReplayMaterial,
    pub proof: SemanticReplayStateMaterialProof,
    pub admission: Arc<SemanticReplayStateMaterialAdmission>,
    pub capacities: crate::SemanticHypergraphCapacities,
    pub admission_limits: crate::SemanticAdmissionLimits,
    pub program: Option<Arc<crate::SemanticProgramAdmission>>,
    pub task: Arc<SemanticTaskEvaluationSpec>,
    /// Lifetime custody for the original task/compiler metadata allocations.
    /// This owner is not a byte-capacity or execution-work authority.
    pub metadata: Arc<dyn Send + Sync>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticSourceContentGeometryKind {
    Historical,
    Current,
}

/// An immutable declaration owned by the original admitted instruction.
/// The late physical recipe can bind this declaration only once.
#[derive(Clone)]
pub struct SemanticPreparedSourceContentGeometry {
    pub(super) owner: Arc<SourceContentGeometryOwner>,
}

/// One original Update's input generation, shared with its CURRENT children.
/// Before the native allocator binds it this is metadata, not a device owner.
#[derive(Clone)]
pub struct SemanticModelInputGenerationPort {
    owner: Arc<ModelInputGenerationDeclaration>,
}

struct ModelInputGenerationGeometry {
    issuer: Arc<()>,
    instruction: Arc<[u8]>,
    parent: SemanticPublishedIdentity,
    memory: ModelMemoryGeometry,
    layouts: BTreeMap<(u64, u64), SemanticTensorLayout>,
    initial: Vec<std::sync::Weak<ModelGenerationOwner>>,
    budget: SemanticReplayMaterialBudget,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

struct ModelInputGenerationDeclaration {
    geometry: Arc<ModelInputGenerationGeometry>,
    step_index: usize,
    preceding: Option<std::sync::Weak<ModelInputGenerationDeclaration>>,
    preceding_index: Option<usize>,
    scope: Arc<()>,
    binding: Mutex<Option<ModelInputGenerationBinding>>,
}

struct ModelInputGenerationBinding {
    scope: Arc<()>,
    input: Vec<std::sync::Weak<ModelGenerationOwner>>,
    output: Vec<std::sync::Weak<ModelGenerationOwner>>,
}

pub(super) struct OriginalModelInputGenerations {
    geometry: Arc<ModelInputGenerationGeometry>,
    ports: Vec<SemanticModelInputGenerationPort>,
}

impl SemanticModelInputGenerationPort {
    fn require_target_binding(
        &self,
        inputs: &PreparedStepInputs,
        scope: &Arc<()>,
    ) -> Result<(), SemanticTransitionError> {
        let binding =
            self.owner.binding.lock().map_err(|_| {
                publication_input_error("model input generation custody is poisoned")
            })?;
        let binding = binding.as_ref().ok_or(SemanticTransitionError::NotBound)?;
        if !Arc::ptr_eq(&binding.scope, scope) || binding.input.len() != inputs.model_slots.len() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for (slots, expected) in inputs.model_slots.iter().zip(&binding.input) {
            let expected = expected
                .upgrade()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let actual = inputs
                .storage
                .allocations
                .get(slots[0])
                .and_then(PublicationAllocation::model_owner)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if slots[0] != slots[1] || !Arc::ptr_eq(&actual, &expected) {
                return Err(publication_input_error(
                    "CURRENT child changed its original TARGET input model owner",
                ));
            }
        }
        Ok(())
    }

    pub fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }

    pub fn material_budget(&self) -> SemanticReplayMaterialBudget {
        self.owner.geometry.budget.clone()
    }

    pub fn step_index(&self) -> usize {
        self.owner.step_index
    }

    pub fn preceding_update(&self) -> Option<usize> {
        self.owner.preceding_index
    }

    pub fn require_original_instruction(
        &self,
        session: &SemanticTransitionSession,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<(), SemanticTransitionError> {
        let original = session.instruction_admission(admission)?;
        if !Arc::ptr_eq(&self.owner.geometry.issuer, &session.publication_issuer)
            || !Arc::ptr_eq(&self.owner.geometry.instruction, &original.instruction)
            || self.owner.geometry.parent != original.parent
        {
            return Err(publication_input_error(
                "model input generation belongs to another original instruction",
            ));
        }
        Ok(())
    }

    pub fn model_allocation_layouts(
        &self,
    ) -> impl Iterator<Item = (SemanticSourceGeometryAllocation, u64)> + '_ {
        self.owner
            .geometry
            .memory
            .allocation_bytes
            .iter()
            .copied()
            .enumerate()
            .map(|(index, span_bytes)| {
                (
                    SemanticSourceGeometryAllocation {
                        scope: Arc::clone(&self.owner.scope),
                        port: SourceGeometryAllocationPort::Model(index),
                        prepared_section: None,
                        span_bytes,
                    },
                    span_bytes,
                )
            })
    }

    pub fn model_storage_layouts(&self) -> &[SemanticModelStorage] {
        &self.owner.geometry.memory.storages
    }

    pub fn model_view_layouts(
        &self,
    ) -> impl Iterator<Item = (SemanticTensorLayout, u64, u64)> + '_ {
        self.owner.geometry.memory.views.iter().map(|view| {
            (
                self.owner.geometry.layouts[&(view.role, view.index)],
                view.storage,
                view.byte_offset,
            )
        })
    }

    /// Valid foreign backings compare unequal; an unbound future is not a match.
    pub fn allocation_matches(
        &self,
        declared: &SemanticSourceGeometryAllocation,
        actual: &DeviceAllocationProvenance,
    ) -> Result<bool, SemanticTransitionError> {
        let SourceGeometryAllocationPort::Model(index) = declared.port else {
            return Err(publication_input_error(
                "model input comparison requires its original model allocation port",
            ));
        };
        if !Arc::ptr_eq(&declared.scope, &self.owner.scope)
            || self.owner.geometry.memory.allocation_bytes.get(index) != Some(&declared.span_bytes)
        {
            return Err(publication_input_error(
                "model input allocation belongs to another original generation declaration",
            ));
        }
        let binding =
            self.owner.binding.lock().map_err(|_| {
                publication_input_error("model input generation custody is poisoned")
            })?;
        let owner = binding
            .as_ref()
            .and_then(|binding| binding.input.get(index))
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| {
                publication_input_error(
                    "model input generation has no live original native binding",
                )
            })?;
        let expected = owner
            .allocation
            .view()
            .allocation_provenance()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        Ok(expected.same_allocation(actual)
            && expected.allocation_bytes() == actual.allocation_bytes()
            && expected.byte_offset() == actual.byte_offset()
            && expected.view_bytes() == actual.view_bytes())
    }
}

impl OriginalModelInputGenerations {
    fn new(
        session: &SemanticTransitionSession,
        parent: &SemanticPublishedLease,
        instruction: Arc<[u8]>,
        updates: impl Iterator<Item = usize> + Clone,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        let storage = session
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let count = updates.clone().count();
        let model_count = storage.model_memory.allocation_bytes.len();
        let node = OwnedPublicationMaterial::tree_node_bound(
            size_of::<((u64, u64), SemanticTensorLayout)>(),
            0,
        )?;
        let layout_count = storage
            .layouts
            .keys()
            .filter(|key| matches!(key.0, 18..=25))
            .count();
        let mut bytes = size_of::<Self>()
            .checked_add(size_of::<ModelInputGenerationGeometry>() + 4 * size_of::<usize>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for (count, width) in [
            (
                model_count,
                size_of::<u64>()
                    + size_of::<[usize; 2]>()
                    + size_of::<std::sync::Weak<ModelGenerationOwner>>(),
            ),
            (
                storage.model_memory.storages.len(),
                size_of::<SemanticModelStorage>(),
            ),
            (
                storage.model_memory.views.len(),
                size_of::<SemanticModelView>(),
            ),
            (layout_count, node),
            (
                count,
                size_of::<ModelInputGenerationDeclaration>()
                    + 4 * size_of::<usize>()
                    + size_of::<SemanticModelInputGenerationPort>(),
            ),
        ] {
            bytes = count
                .checked_mul(width)
                .and_then(|extent| bytes.checked_add(extent))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let bindings = count
            .checked_mul(model_count)
            .and_then(|count| {
                count.checked_mul(2 * size_of::<std::sync::Weak<ModelGenerationOwner>>())
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        bytes = bytes
            .checked_add(bindings)
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = Arc::new(budget.reserve_auxiliary(
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?);
        let slots = storage.model_slots_for_directory(&parent.directory)?;
        if slots.len() != model_count {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut initial = Vec::with_capacity(model_count);
        for slots in &slots {
            if slots[0] != slots[1] {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let owner = storage
                .allocations
                .get(slots[0])
                .and_then(PublicationAllocation::model_owner)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            initial.push(Arc::downgrade(&owner));
        }
        let mut layouts = BTreeMap::new();
        for (key, layout) in &storage.layouts {
            if matches!(key.0, 18..=25) {
                layouts.insert(*key, *layout);
            }
        }
        let geometry = Arc::new(ModelInputGenerationGeometry {
            issuer: Arc::clone(&session.publication_issuer),
            instruction,
            parent: parent.identity,
            memory: storage.model_memory.clone(),
            layouts,
            initial,
            budget: budget.clone(),
            _reservation: reservation,
        });
        let mut ports: Vec<SemanticModelInputGenerationPort> = Vec::with_capacity(count);
        for step_index in updates {
            let preceding = ports.last().map(|port| Arc::downgrade(&port.owner));
            let preceding_index = ports
                .last()
                .map(SemanticModelInputGenerationPort::step_index);
            ports.push(SemanticModelInputGenerationPort {
                owner: Arc::new(ModelInputGenerationDeclaration {
                    geometry: Arc::clone(&geometry),
                    step_index,
                    preceding,
                    preceding_index,
                    scope: Arc::new(()),
                    binding: Mutex::new(None),
                }),
            });
        }
        Ok(Arc::new(Self { geometry, ports }))
    }

    pub(super) fn require_instruction(
        &self,
        instruction: &Arc<[u8]>,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.geometry.instruction, instruction) {
            return Err(publication_input_error(
                "model input generations changed their original instruction",
            ));
        }
        Ok(())
    }

    fn port(
        &self,
        step_index: usize,
    ) -> Result<SemanticModelInputGenerationPort, SemanticTransitionError> {
        self.ports
            .iter()
            .find(|port| port.step_index() == step_index)
            .cloned()
            .ok_or_else(|| {
                publication_input_error(
                    "model input generation requires an original declared Update",
                )
            })
    }

    fn bind_segment(
        &self,
        session: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        let build = session
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let admission = build
            .program_admission
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        self.require_instruction(&session.instruction_admission(admission)?.instruction)?;
        if !Arc::ptr_eq(&build.issuer, &self.geometry.issuer) || build.submitted || build.capturing
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        for port in &self.ports {
            if build.transitions.get(port.step_index()) != Some(&SemanticTransitionKind::Update) {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let step = session
                .steps
                .get(&build.tokens[port.step_index()])
                .ok_or(SemanticTransitionError::NotBound)?;
            let inputs = step
                .inputs
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let prepared = step
                .prepared
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            let output_slots = prepared.branches[0]
                .model_update
                .as_ref()
                .map_or(&[][..], |update| update.pending_slots.as_slice());
            let other_slots = prepared.branches[1]
                .model_update
                .as_ref()
                .map_or(&[][..], |update| update.pending_slots.as_slice());
            if inputs.storage.model_memory != self.geometry.memory
                || self
                    .geometry
                    .layouts
                    .iter()
                    .any(|(key, layout)| inputs.storage.layouts.get(key) != Some(layout))
                || inputs.model_slots.len() != self.geometry.initial.len()
                || output_slots.len() != self.geometry.initial.len()
                || output_slots != other_slots
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            let previous = port
                .owner
                .preceding
                .as_ref()
                .map(|previous| {
                    previous
                        .upgrade()
                        .ok_or(SemanticTransitionError::ObservationMismatch)
                })
                .transpose()?;
            let previous_binding = previous
                .as_ref()
                .map(|previous| {
                    previous.binding.lock().map_err(|_| {
                        publication_input_error("preceding model output custody is poisoned")
                    })
                })
                .transpose()?;
            let expected = match &previous_binding {
                Some(previous) => {
                    &previous
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?
                        .output
                }
                None => &self.geometry.initial,
            };
            let mut binding = port.owner.binding.lock().map_err(|_| {
                publication_input_error("model input generation custody is poisoned")
            })?;
            if binding.is_some() {
                return Err(publication_input_error(
                    "model input generation is bound once by its original allocator",
                ));
            }
            let mut input = Vec::with_capacity(expected.len());
            let mut output = Vec::with_capacity(expected.len());
            for ((slots, expected), output_slot) in
                inputs.model_slots.iter().zip(expected).zip(output_slots)
            {
                let expected = expected
                    .upgrade()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let actual = inputs
                    .storage
                    .allocations
                    .get(slots[0])
                    .and_then(PublicationAllocation::model_owner)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let pending = inputs
                    .storage
                    .allocations
                    .get(*output_slot)
                    .and_then(PublicationAllocation::model_owner)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if slots[0] != slots[1]
                    || !Arc::ptr_eq(&actual, &expected)
                    || Arc::ptr_eq(&actual, &pending)
                {
                    return Err(publication_input_error(
                        "Update model input is not its original parent or preceding pending output",
                    ));
                }
                input.push(Arc::downgrade(&actual));
                output.push(Arc::downgrade(&pending));
            }
            *binding = Some(ModelInputGenerationBinding {
                scope: Arc::clone(&build.scope),
                input,
                output,
            });
        }
        Ok(())
    }
}

pub(super) struct SourceContentGeometryOwner {
    pub(super) work: SemanticColdModelWork,
    pub(super) instruction: Arc<[u8]>,
    allowance: Arc<Mutex<ColdNativeAllowance>>,
    pub(super) kind: SemanticSourceContentGeometryKind,
    step_index: usize,
    ordinal: u64,
    bank: Option<usize>,
    member: crate::semantic_training_view::FrozenPolicyGroupMember,
    pub(super) seed: Arc<SemanticSourceContentGeometrySeed>,
    pub(super) publication: Arc<SourcePublicationGeometry>,
    scope: Arc<()>,
    pub(super) recipe: Mutex<Option<std::sync::Weak<PreparedSourceContentOwner>>>,
    current: Mutex<Option<std::sync::Weak<actor_refresh::PreparedActorRefreshOwner>>>,
    reader: Mutex<Option<(Arc<()>, u64)>>,
    pub(super) occurrences:
        Mutex<Option<Arc<super::source_content_occurrences::FrozenSourceContentOccurrences>>>,
    uniform: Mutex<Option<[SemanticPolicyNumericalDomainLayout; 3]>>,
    input_generation: Option<SemanticModelInputGenerationPort>,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

pub(super) struct OriginalSourceGeometryRoster {
    instruction: Arc<[u8]>,
    handles: Arc<[SemanticPreparedSourceContentGeometry]>,
    seeds: Vec<(u64, Arc<SemanticSourceContentGeometrySeed>)>,
    input_generations: Arc<OriginalModelInputGenerations>,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceGeometryAllocationPort {
    Model(usize),
    Publication(usize),
    Input(usize),
    Header,
    Source,
    Feedback(usize),
    Uniform(usize),
}

/// Original physical allocation geometry, before any backing is allocated.
/// This is not a device allocation and cannot be exported or read numerically.
#[derive(Clone)]
pub struct SemanticSourceGeometryAllocation {
    scope: Arc<()>,
    port: SourceGeometryAllocationPort,
    prepared_section: Option<(SemanticTransitionKind, usize)>,
    span_bytes: u64,
}

/// An original typed view and its complete backing, without a device address.
#[derive(Clone)]
pub struct SemanticSourceGeometryTensor {
    pub layout: SemanticTensorLayout,
    pub logical_begin: u64,
    pub logical_end: u64,
    pub allocation: SemanticSourceGeometryAllocation,
    pub byte_offset: u64,
}

impl SemanticSourceGeometryAllocation {
    fn declared(
        scope: &Arc<()>,
        port: SourceGeometryAllocationPort,
        prepared_section: Option<(SemanticTransitionKind, usize)>,
        span_bytes: u64,
    ) -> Self {
        Self {
            scope: Arc::clone(scope),
            port,
            prepared_section,
            span_bytes,
        }
    }

    pub(super) fn restored_prefill(
        scope: &Arc<()>,
        model: bool,
        index: usize,
        span_bytes: u64,
    ) -> Self {
        Self::declared(
            scope,
            if model {
                SourceGeometryAllocationPort::Model(index)
            } else {
                SourceGeometryAllocationPort::Publication(index)
            },
            None,
            span_bytes,
        )
    }

    pub fn span_bytes(&self) -> u64 {
        self.span_bytes
    }

    pub fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.scope, &other.scope)
            && self.port == other.port
            && self.prepared_section == other.prepared_section
    }
}

impl SourceContentGeometryOwner {
    fn require_geometry_section(
        &self,
        section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<(), SemanticTransitionError> {
        match (self.kind, section) {
            (SemanticSourceContentGeometryKind::Historical, None) => Ok(()),
            (
                SemanticSourceContentGeometryKind::Current,
                Some((SemanticTransitionKind::Recompute | SemanticTransitionKind::Proposal, bank)),
            ) if bank < 2 => Ok(()),
            _ => Err(publication_input_error(
                "content geometry requires its original historical or prepared child section",
            )),
        }
    }

    fn allocation_section(
        &self,
        port: SourceGeometryAllocationPort,
        section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<Option<(SemanticTransitionKind, usize)>, SemanticTransitionError> {
        if matches!(port, SourceGeometryAllocationPort::Model(_)) {
            return Ok(None);
        }
        self.require_geometry_section(section)?;
        Ok(match port {
            // The child publication storage is shared across Steps. Its real
            // bank-selected slot already identifies prefix/cache sharing.
            SourceGeometryAllocationPort::Publication(_) => None,
            // These allocations belong to a Step, not either branch bank.
            SourceGeometryAllocationPort::Input(_)
            | SourceGeometryAllocationPort::Header
            | SourceGeometryAllocationPort::Source
            | SourceGeometryAllocationPort::Feedback(_) => section.map(|(kind, _)| (kind, 0)),
            // Uniform outputs retain their exact native Proposal bank.
            SourceGeometryAllocationPort::Uniform(_) => section,
            SourceGeometryAllocationPort::Model(_) => unreachable!(),
        })
    }

    pub(super) fn require_allocation_section(
        &self,
        allocation: &SemanticSourceGeometryAllocation,
        section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<(), SemanticTransitionError> {
        self.require_geometry_section(section)?;
        self.require_allocation(allocation, allocation.span_bytes)?;
        if allocation.prepared_section != self.allocation_section(allocation.port, section)? {
            return Err(publication_input_error(
                "content allocation belongs to another original prepared child section",
            ));
        }
        if self.kind == SemanticSourceContentGeometryKind::Current
            && matches!(allocation.port, SourceGeometryAllocationPort::Uniform(_))
            && section.map(|(kind, _)| kind) != Some(SemanticTransitionKind::Proposal)
        {
            return Err(publication_input_error(
                "uniform geometry requires its original prepared Proposal bank",
            ));
        }
        Ok(())
    }

    fn model_geometry(
        &self,
    ) -> (
        &ModelMemoryGeometry,
        &BTreeMap<(u64, u64), SemanticTensorLayout>,
    ) {
        match &self.input_generation {
            Some(port) => (&port.owner.geometry.memory, &port.owner.geometry.layouts),
            None => (
                &self.seed.proof.owner.model_memory,
                &self.seed.proof.owner.layouts,
            ),
        }
    }

    fn model_scope(&self) -> &Arc<()> {
        self.input_generation
            .as_ref()
            .map_or(&self.scope, |port| &port.owner.scope)
    }

    pub(super) fn is_uniform_allocation(
        &self,
        allocation: &SemanticSourceGeometryAllocation,
    ) -> bool {
        matches!(allocation.port, SourceGeometryAllocationPort::Uniform(_))
    }

    fn bind_uniform_domain_layouts(
        &self,
        layouts: [SemanticPolicyNumericalDomainLayout; 3],
    ) -> Result<(), SemanticTransitionError> {
        if self.kind != SemanticSourceContentGeometryKind::Historical {
            return Err(publication_input_error(
                "uniform domains changed their original allocation geometry",
            ));
        }
        for (actual, expected) in layouts.iter().zip(&self.publication.uniform_layouts) {
            let bytes = u64::try_from(tensor_layout_bytes(expected)?)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            if actual.layout != *expected || actual.allocation.allocation_bytes() != bytes {
                return Err(publication_input_error(
                    "uniform domains changed their original allocation geometry",
                ));
            }
        }
        let mut original = self
            .uniform
            .lock()
            .map_err(|_| publication_input_error("original uniform domain custody is poisoned"))?;
        if original.as_ref().is_some_and(|original| {
            original.iter().zip(&layouts).any(|(original, actual)| {
                !original.allocation.same_allocation(&actual.allocation)
                    || original.allocation.byte_offset() != actual.allocation.byte_offset()
                    || original.allocation.view_bytes() != actual.allocation.view_bytes()
            })
        }) {
            return Err(publication_input_error(
                "uniform domains replaced their original allocations",
            ));
        }
        if original.is_none() {
            *original = Some(layouts);
        }
        Ok(())
    }

    pub(super) fn require_prepared_allocation(
        &self,
        session: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        section: (SemanticTransitionKind, usize),
        declared: &SemanticSourceGeometryAllocation,
        descriptor: &super::source_content_occurrences::SemanticSourceTensorDescriptor,
        tensor: &PreparedSemanticTensor,
    ) -> Result<(), SemanticTransitionError> {
        self.require_allocation_section(declared, Some(section))?;
        self.classify_tensor(declared, descriptor, Some(section))?;
        self.require_current_section(session, step, section)?;
        if matches!(declared.port, SourceGeometryAllocationPort::Model(_)) {
            let actual = tensor.native_allocation.as_ref().ok_or_else(|| {
                publication_input_error(
                    "CURRENT model backing has no original native allocation identity",
                )
            })?;
            let port = self
                .input_generation
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            if !port.allocation_matches(declared, actual)? {
                return Err(publication_input_error(
                    "CURRENT model backing differs from its original TARGET input generation",
                ));
            }
            return Ok(());
        }
        let owner = &session.steps[&step.token];
        let inputs = owner
            .inputs
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let prepared = owner
            .prepared
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let view = match declared.port {
            SourceGeometryAllocationPort::Publication(index) => inputs
                .storage
                .allocations
                .get(index)
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .slice()?
                .view(),
            SourceGeometryAllocationPort::Input(index) => {
                let plan = self
                    .publication
                    .inputs
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let actual = inputs
                    .plans
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if (actual.role, actual.index, actual.layout)
                    != (plan.role, plan.index, plan.layout)
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                inputs.views[section.1]
                    .get(&(plan.role, plan.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .clone()
            }
            SourceGeometryAllocationPort::Header => {
                inputs.field_view(PublicationBankField::PrefixExtent)?
            }
            SourceGeometryAllocationPort::Source => {
                inputs.field_view(PublicationBankField::Source)?
            }
            SourceGeometryAllocationPort::Feedback(index) => owner
                .feedback
                .first()
                .and_then(|feedback| feedback.content.tensors.get(index))
                .and_then(|tensor| tensor.source.clone())
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
            SourceGeometryAllocationPort::Uniform(index) => {
                let branch = &prepared.branches[section.1];
                let buffers = branch
                    .policy_buffers
                    .as_ref()
                    .or_else(|| branch.policy.as_ref().map(|policy| &policy.buffers))
                    .ok_or(SemanticTransitionError::NotBound)?;
                let layouts =
                    policy_uniform_domain_layouts(buffers, session.codebooks.input_cells)?;
                let expected = layouts
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if expected.layout != self.publication.uniform_layouts[index]
                    || tensor.native_allocation.as_ref().is_none_or(|actual| {
                        !expected.allocation.same_allocation(actual)
                            || expected.allocation.byte_offset() != actual.byte_offset()
                            || expected.allocation.view_bytes() != actual.view_bytes()
                    })
                {
                    return Err(publication_input_error(
                        "CURRENT uniform domain changed its original native output allocation",
                    ));
                }
                // SAFETY: the byte view retains the original F64 output;
                // it is used only for authentic allocation/address comparison.
                unsafe { buffers.numerical.uniform.outputs[index].view().cast::<u8>() }
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
            }
            SourceGeometryAllocationPort::Model(_) => unreachable!(),
        };
        let expected = view
            .allocation_provenance()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let base = (*view.device_ptr())
            .checked_sub(expected.byte_offset())
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if expected.allocation_bytes() != declared.span_bytes
            || base.checked_add(descriptor.byte_offset) != Some(tensor.data)
            || tensor.native_allocation.as_ref().is_some_and(|actual| {
                !expected.same_allocation(actual)
                    || expected.allocation_bytes() != actual.allocation_bytes()
            })
        {
            return Err(publication_input_error(
                "CURRENT content changed its original prepared allocation or typed view address",
            ));
        }
        Ok(())
    }

    fn require_current_child(
        &self,
        session: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        let build = session
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let staged = session
            .actor_refresh
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let actor = &staged.proof.inner;
        if self.kind != SemanticSourceContentGeometryKind::Current
            || build.transitions
                != [
                    SemanticTransitionKind::Recompute,
                    SemanticTransitionKind::Proposal,
                ]
            || self.bank != Some(staged.target_bank)
            || self
                .current
                .lock()
                .map_err(|_| {
                    publication_input_error("original actor geometry custody is poisoned")
                })?
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .is_none_or(|original| !Arc::ptr_eq(&original, actor))
            || !Arc::ptr_eq(&build.actor_refresh_execution, &actor.execution)
            || !Arc::ptr_eq(&build.cancelled, &actor.cancelled)
        {
            return Err(publication_input_error(
                "CURRENT content changed its original prepared child admission",
            ));
        }
        Ok(())
    }

    pub(super) fn require_current_section(
        &self,
        session: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        section: (SemanticTransitionKind, usize),
    ) -> Result<(), SemanticTransitionError> {
        self.require_current_child(session)?;
        let build = session
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let position = build
            .tokens
            .iter()
            .position(|token| *token == step.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let prepared = session
            .steps
            .get(&step.token)
            .and_then(|owner| owner.prepared.as_ref())
            .ok_or(SemanticTransitionError::NotBound)?;
        if section.1 > 1
            || build.transitions.get(position) != Some(&section.0)
            || prepared
                .model_work
                .as_ref()
                .and_then(|work| work.capture_bank)
                != Some(section.1)
            || prepared.model_content_recorded & (1 << section.1) == 0
        {
            return Err(publication_input_error(
                "CURRENT content changed its original child admission, prepared operation or bank",
            ));
        }
        Ok(())
    }

    fn classify_tensor(
        &self,
        allocation: &SemanticSourceGeometryAllocation,
        descriptor: &super::source_content_occurrences::SemanticSourceTensorDescriptor,
        section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<(bool, bool), SemanticTransitionError> {
        self.require_allocation_section(allocation, section)?;
        let matches = |original: SemanticSourceGeometryTensor| {
            let mut layout = original.layout;
            if matches!(
                allocation.port,
                SourceGeometryAllocationPort::Header
                    | SourceGeometryAllocationPort::Source
                    | SourceGeometryAllocationPort::Uniform(_)
            ) {
                // Uniform output ports keep their original producer identity;
                // the capture roster supplies its own complete ordered indices.
                layout.index = descriptor.layout.index;
            }
            original.allocation.same_handle(allocation)
                && layout == descriptor.layout
                && original.logical_begin == descriptor.logical_begin
                && original.logical_end == descriptor.logical_end
                && original.byte_offset == descriptor.byte_offset
        };
        match allocation.port {
            SourceGeometryAllocationPort::Model(index) => {
                let (memory, layouts) = self.model_geometry();
                for view in &memory.views {
                    let storage = &memory.storages[usize::try_from(view.storage)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?];
                    if storage.allocation != index as u64 {
                        continue;
                    }
                    let layout = *layouts
                        .get(&(view.role, view.index))
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let offset = storage
                        .byte_offset
                        .checked_add(view.byte_offset)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?;
                    let end = if layout.logical_axis == u64::MAX {
                        0
                    } else {
                        layout.dimensions[layout.logical_axis as usize]
                    };
                    if matches(self.tensor_geometry(
                        layout,
                        0,
                        end,
                        allocation.port,
                        offset,
                        section,
                    )?) {
                        return Ok((true, false));
                    }
                }
            }
            SourceGeometryAllocationPort::Publication(_)
            | SourceGeometryAllocationPort::Input(_)
            | SourceGeometryAllocationPort::Header
            | SourceGeometryAllocationPort::Source => {
                for original in self.input_tensor_layouts(section)? {
                    if matches(original?) {
                        return Ok((true, false));
                    }
                }
            }
            SourceGeometryAllocationPort::Feedback(index) => {
                let layout = *self
                    .publication
                    .feedback_layouts
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if matches(self.tensor_geometry(layout, 0, 0, allocation.port, 0, section)?) {
                    return Ok((false, true));
                }
            }
            SourceGeometryAllocationPort::Uniform(index) => {
                let layout = *self
                    .publication
                    .uniform_layouts
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                if matches(self.tensor_geometry(layout, 0, 0, allocation.port, 0, section)?) {
                    return Ok((false, false));
                }
            }
        }
        Err(publication_input_error(
            "content occurrence changed its original typed geometry port",
        ))
    }
    fn input_tensor_layouts(
        &self,
        section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<
        impl Iterator<Item = Result<SemanticSourceGeometryTensor, SemanticTransitionError>> + '_,
        SemanticTransitionError,
    > {
        self.require_geometry_section(section)?;
        let original = &self.seed.proof.owner;
        let bank = section.map_or(
            (original.header.publication_word & 1) as usize,
            |(_, bank)| bank,
        );
        let inputs = self
            .publication
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, plan)| matches!(plan.role, 1 | 3 | 4..=13))
            .map(move |(index, plan)| {
                let range = self.publication.storage.bank_templates[bank]
                    .iter()
                    .find(|range| (range.role, range.index) == (plan.role, plan.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let layout = if plan.role == 1 {
                    committed_prefix_layout(range, original.contract.prefix_capacity)?
                } else {
                    plan.layout
                };
                let aliased = matches!(plan.role, 1 | 4 | 5);
                let port = if aliased {
                    SourceGeometryAllocationPort::Publication(plan.banks[bank].storage_slot)
                } else {
                    SourceGeometryAllocationPort::Input(index)
                };
                let export = publication_export_span(
                    &PublicationRange {
                        offset_bytes: 0,
                        ..*range
                    },
                    &layout,
                    plan.banks[bank].span.len(),
                )?;
                let byte_offset = if aliased {
                    plan.banks[bank]
                        .span
                        .start
                        .checked_add(export.start)
                        .ok_or(SemanticTransitionError::GenerationExhausted)?
                } else {
                    export.start
                };
                let byte_offset = u64::try_from(byte_offset)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
                self.tensor_geometry(
                    layout,
                    range.logical_begin,
                    range.logical_end,
                    port,
                    byte_offset,
                    section,
                )
            });
        let fields = [
            PublicationBankField::Source,
            PublicationBankField::PrefixExtent,
            PublicationBankField::RingHead,
            PublicationBankField::Terminal,
        ]
        .into_iter()
        .map(move |field| {
            let (span, dimensions, rank) = field.geometry();
            let mut shape = [0; 4];
            shape[..rank].copy_from_slice(&dimensions[..rank]);
            let layout = canonical_tensor_layout(SemanticTensorLayout {
                role: 0,
                index: 0,
                scalar_type: 3,
                element_bytes: 8,
                rank: rank as u64,
                dimensions: shape,
                logical_axis: u64::MAX,
                strides_bytes: [0; 4],
            })?;
            let (port, offset) = if matches!(field, PublicationBankField::Source) {
                (SourceGeometryAllocationPort::Source, 0)
            } else {
                (
                    SourceGeometryAllocationPort::Header,
                    u64::try_from(span.start)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
            };
            self.tensor_geometry(layout, 0, 0, port, offset, section)
        });
        Ok(inputs.chain(fields))
    }
    pub(super) fn matches_reader(
        &self,
        issuer: &Arc<()>,
        token: u64,
    ) -> Result<bool, SemanticTransitionError> {
        Ok(self
            .reader
            .lock()
            .map_err(|_| publication_input_error("original content reader custody is poisoned"))?
            .as_ref()
            .is_some_and(|(original, original_token)| {
                Arc::ptr_eq(original, issuer) && *original_token == token
            }))
    }

    pub(super) fn require_allocation(
        &self,
        allocation: &SemanticSourceGeometryAllocation,
        span: u64,
    ) -> Result<(), SemanticTransitionError> {
        let scope = if matches!(allocation.port, SourceGeometryAllocationPort::Model(_)) {
            self.model_scope()
        } else {
            &self.scope
        };
        if !Arc::ptr_eq(scope, &allocation.scope) || allocation.span_bytes != span {
            return Err(publication_input_error(
                "content allocation is not its original pre-factory geometry",
            ));
        }
        Ok(())
    }

    pub(super) fn bound_allocation(
        &self,
        allocation: &SemanticSourceGeometryAllocation,
    ) -> Result<Option<DeviceAllocationProvenance>, SemanticTransitionError> {
        self.require_allocation(allocation, allocation.span_bytes)?;
        let recipe = self
            .recipe
            .lock()
            .map_err(|_| publication_input_error("original content recipe custody is poisoned"))?
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| {
                publication_input_error(
                    "original content geometry has not bound its physical producer",
                )
            })?;
        let storage = recipe
            .storage
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let inputs = recipe
            .inputs
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let bank = (self.seed.proof.owner.header.publication_word & 1) as usize;
        let provenance = match allocation.port {
            SourceGeometryAllocationPort::Model(index) => {
                let range = recipe
                    .directory
                    .iter()
                    .find(|range| {
                        matches!(range.role, 18..=25)
                            && storage
                                .model_memory
                                .location(range.role, range.index)
                                .is_ok_and(|(allocation, _)| allocation == index)
                    })
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                let slot = usize::try_from(range.storage_slot)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                storage
                    .allocations
                    .get(slot)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .slice()?
                    .view()
                    .allocation_provenance()
            }
            SourceGeometryAllocationPort::Publication(index) => storage
                .allocations
                .get(index)
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .slice()?
                .view()
                .allocation_provenance(),
            SourceGeometryAllocationPort::Input(index) => {
                let plan = self
                    .publication
                    .inputs
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                inputs.views[bank]
                    .get(&(plan.role, plan.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .allocation_provenance()
            }
            SourceGeometryAllocationPort::Header => inputs.header.view().allocation_provenance(),
            SourceGeometryAllocationPort::Source => inputs.source.view().allocation_provenance(),
            SourceGeometryAllocationPort::Feedback(index) => recipe
                .feedback_geometry
                .lock()
                .map_err(|_| publication_input_error("original feedback custody is poisoned"))?
                .as_ref()
                .and_then(|tensors| tensors.get(index))
                .and_then(|tensor| tensor.native_allocation.clone()),
            SourceGeometryAllocationPort::Uniform(index) => self
                .uniform
                .lock()
                .map_err(|_| {
                    publication_input_error("original uniform domain custody is poisoned")
                })?
                .as_ref()
                .and_then(|layouts| layouts.get(index))
                .map(|layout| layout.allocation.clone()),
        };
        if provenance
            .as_ref()
            .is_some_and(|value| value.allocation_bytes() != allocation.span_bytes)
        {
            return Err(publication_input_error(
                "physical content producer changed its declared full allocation span",
            ));
        }
        Ok(provenance)
    }
    fn tensor_geometry(
        &self,
        layout: SemanticTensorLayout,
        logical_begin: u64,
        logical_end: u64,
        port: SourceGeometryAllocationPort,
        byte_offset: u64,
        section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<SemanticSourceGeometryTensor, SemanticTransitionError> {
        let bytes = u64::try_from(tensor_layout_bytes(&layout)?)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let span_bytes = match port {
            SourceGeometryAllocationPort::Model(index) => *self
                .model_geometry()
                .0
                .allocation_bytes
                .get(index)
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
            SourceGeometryAllocationPort::Publication(index) => u64::try_from(
                *self
                    .publication
                    .storage
                    .allocation_bytes
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
            )
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            SourceGeometryAllocationPort::Input(index) => u64::try_from(
                self.publication
                    .inputs
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .banks[0]
                    .span
                    .len(),
            )
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            SourceGeometryAllocationPort::Header => size_of::<PublicationHeader>() as u64,
            SourceGeometryAllocationPort::Source => size_of::<[SourceSlot; 32]>() as u64,
            SourceGeometryAllocationPort::Feedback(index) => u64::try_from(tensor_layout_bytes(
                self.publication
                    .feedback_layouts
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
            )?)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            SourceGeometryAllocationPort::Uniform(index) => u64::try_from(tensor_layout_bytes(
                self.publication
                    .uniform_layouts
                    .get(index)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
            )?)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        };
        if byte_offset
            .checked_add(bytes)
            .is_none_or(|end| end > span_bytes)
        {
            return Err(publication_input_error(
                "original geometry view exceeds its complete backing",
            ));
        }
        Ok(SemanticSourceGeometryTensor {
            layout,
            logical_begin,
            logical_end,
            byte_offset,
            allocation: SemanticSourceGeometryAllocation::declared(
                if matches!(port, SourceGeometryAllocationPort::Model(_)) {
                    self.model_scope()
                } else {
                    &self.scope
                },
                port,
                self.allocation_section(port, section)?,
                span_bytes,
            ),
        })
    }
}

impl SemanticPreparedSourceContentGeometry {
    pub fn model_input_generation_port(
        &self,
    ) -> Result<SemanticModelInputGenerationPort, SemanticTransitionError> {
        self.owner.input_generation.clone().ok_or_else(|| {
            publication_input_error("historical SOURCE model has no CURRENT input generation port")
        })
    }

    pub(super) fn require_original_actor_refresh(
        &self,
        target: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        bank: usize,
        member: &crate::semantic_training_view::FrozenPolicyGroupMember,
        material: &SemanticReplayMaterial,
    ) -> Result<(), SemanticTransitionError> {
        let build = target
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let admission = build
            .program_admission
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let instruction = &target.instruction_admission(admission)?.instruction;
        let step_index = build
            .tokens
            .iter()
            .position(|token| *token == step.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if self.owner.kind != SemanticSourceContentGeometryKind::Current
            || self.owner.bank != Some(bank)
            || self.owner.step_index != step_index
            || self.owner.member.ordinal != member.ordinal
            || self.owner.member.identity != member.identity
            || self.owner.member.source_identity != member.source_identity
            || self.owner.member.bytes_identity != member.bytes_identity
            || self.owner.member.content_identity != member.content_identity
            || self.owner.member.origin != member.origin
            || !Arc::ptr_eq(&self.owner.instruction, instruction)
            || !Arc::ptr_eq(&self.owner.seed.material.owner, &material.owner)
            || target
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(actual, &self.owner.allowance))
            || self
                .owner
                .current
                .lock()
                .map_err(|_| {
                    publication_input_error("original actor geometry custody is poisoned")
                })?
                .is_some()
            || self
                .owner
                .occurrences
                .lock()
                .map_err(|_| {
                    publication_input_error("original content occurrence custody is poisoned")
                })?
                .is_none()
        {
            return Err(publication_input_error(
                "actor refresh changed its original declared Update, bank or member geometry",
            ));
        }
        let inputs = target
            .steps
            .get(&step.token)
            .and_then(|step| step.inputs.as_ref())
            .ok_or(SemanticTransitionError::NotBound)?;
        let (memory, layouts) = self.owner.model_geometry();
        if inputs.storage.model_memory != *memory
            || memory.views.iter().any(|view| {
                inputs.storage.layouts.get(&(view.role, view.index))
                    != layouts.get(&(view.role, view.index))
            })
        {
            return Err(publication_input_error("actor refresh CURRENT model geometry changed its original immutable layout and alias partition"));
        }
        self.model_input_generation_port()?
            .require_target_binding(inputs, &build.scope)?;
        Ok(())
    }

    pub(super) fn bind_original_actor_refresh(
        &self,
        proof: &Arc<actor_refresh::PreparedActorRefreshOwner>,
    ) -> Result<(), SemanticTransitionError> {
        let mut original =
            self.owner.current.lock().map_err(|_| {
                publication_input_error("original actor geometry custody is poisoned")
            })?;
        if original.is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        *original = Some(Arc::downgrade(proof));
        Ok(())
    }

    /// Join the original historical declaration to its already-bound producer.
    /// This grants neither recording nor admission and creates no new recipe.
    pub fn bound_source_content_for_admission(
        &self,
        original_work: &SemanticColdModelWork,
    ) -> Result<SemanticPreparedSourceContent, SemanticTransitionError> {
        if self.owner.kind != SemanticSourceContentGeometryKind::Historical
            || self.owner.bank.is_some()
            || !self.owner.work.same_invocation(original_work)
        {
            return Err(publication_input_error(
                "SOURCE admission requires its original historical declaration and report",
            ));
        }
        let owner = self
            .owner
            .recipe
            .lock()
            .map_err(|_| publication_input_error("original content recipe custody is poisoned"))?
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| {
                publication_input_error(
                    "SOURCE admission requires its original live bound content recipe",
                )
            })?;
        if !Arc::ptr_eq(&owner.geometry, &self.owner)
            || !Arc::ptr_eq(&owner.target_instruction, &self.owner.instruction)
            || !Arc::ptr_eq(&owner.allowance, &self.owner.allowance)
            || owner.ordinal != self.owner.ordinal
            || owner.origin != self.owner.member.origin
            || owner.material != self.owner.seed.material.predecessor_material_digest()
            || !self.owner.matches_reader(&owner.issuer, owner.token)?
        {
            return Err(publication_input_error(
                "SOURCE admission changed its original instruction, member, material or reader binding",
            ));
        }
        Ok(SemanticPreparedSourceContent { owner })
    }

    pub fn material_budget(&self) -> SemanticReplayMaterialBudget {
        self.owner.seed.proof.owner.budget.clone()
    }

    /// Original task metadata retained by the same decoded source owner.
    /// It grants neither a task-use token nor access to future numerical data.
    pub fn task_spec(&self) -> &SemanticTaskEvaluationSpec {
        &self.owner.seed.task
    }

    pub fn task_objective_law(&self) -> Result<SemanticTaskObjectiveLaw, SemanticTransitionError> {
        let task = self.task_spec();
        task.scoring.objective_law(
            task.program.editable_program().is_some(),
            u32::try_from(task.statement_records.len())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )
    }

    pub fn prefix_capacity(&self) -> u64 {
        self.owner.seed.proof.owner.contract.prefix_capacity
    }

    pub fn feedback_capacity(&self) -> u64 {
        self.owner.seed.proof.owner.contract.feedback_capacity
    }

    pub fn policy_layout(&self) -> &SemanticPolicyLayout {
        &self.owner.publication.policy_layout
    }

    pub fn policy_support_layout(&self) -> (&[std::ops::Range<usize>], usize) {
        (
            &self.owner.publication.support_layout,
            self.owner.publication.support_cells,
        )
    }

    pub fn policy_binding(&self) -> SemanticCatalogueBinding {
        SemanticCatalogueBinding {
            generation: CATALOGUE_GENERATION,
            digest: self.codebook_identity(21),
        }
    }

    pub fn model_catalogue_binding(&self) -> SemanticCatalogueBinding {
        if self.owner.seed.proof.owner.codebook_header[40] == 0 {
            return self.policy_binding();
        }
        SemanticCatalogueBinding {
            generation: CATALOGUE_GENERATION,
            digest: self.codebook_identity(25),
        }
    }

    pub(super) fn frozen_model_context(
        &self,
        catalogue: Option<&Arc<model_catalogue::FrozenModelCatalogue>>,
    ) -> Result<Option<SemanticFrozenModelContext>, SemanticTransitionError> {
        if self.owner.seed.proof.owner.codebook_header[40] == 0 {
            if catalogue.is_some() {
                return Err(publication_input_error(
                    "solo source geometry cannot replace a frozen model context",
                ));
            }
            return Ok(None);
        }
        catalogue
            .ok_or_else(|| {
                publication_input_error(
                    "source construction requires the original frozen model catalogue",
                )
            })?
            .context_for_header(&self.owner.seed.proof.owner.codebook_header)
            .map(Some)
    }

    pub fn schema_generation(&self) -> Identity256 {
        self.codebook_identity(10)
    }

    fn codebook_identity(&self, first_word: usize) -> Identity256 {
        let mut bytes = [0; 32];
        for (word, destination) in self.owner.seed.proof.owner.codebook_header
            [first_word..first_word + 4]
            .iter()
            .zip(bytes.chunks_exact_mut(8))
        {
            destination.copy_from_slice(&word.to_le_bytes());
        }
        Identity256::from_bytes(bytes)
    }

    pub fn feedback_schema(&self) -> &SemanticFeedbackSchema {
        &self.owner.publication.feedback_schema
    }

    pub fn structural_capacities(&self) -> (u64, u64, u64) {
        let contract = &self.owner.seed.proof.owner.contract;
        (
            contract.prefix_capacity,
            contract.feedback_capacity,
            contract.max_position,
        )
    }

    pub fn position_table_identity(&self) -> Identity256 {
        self.owner.seed.proof.owner.contract.table_identity
    }

    pub fn content_family(
        &self,
        phase: super::source_content_occurrences::SemanticSourceContentPhase,
        kind: super::source_content_occurrences::SemanticSourceContentKind,
        prepared_section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<
        super::source_content_occurrences::SemanticSourceContentFamily,
        SemanticTransitionError,
    > {
        use super::source_content_occurrences::{
            SemanticSourceContentFamily, SemanticSourceContentKind, SemanticSourceContentPhase,
        };
        match (self.owner.kind, prepared_section) {
            (SemanticSourceContentGeometryKind::Historical, None) => {
                Ok(SemanticSourceContentFamily::new(&self.owner, phase, kind))
            }
            (SemanticSourceContentGeometryKind::Current, Some((operation, bank)))
                if bank < 2
                    && matches!(
                        operation,
                        SemanticTransitionKind::Recompute | SemanticTransitionKind::Proposal
                    )
                    && kind != SemanticSourceContentKind::BindModel
                    && !(operation == SemanticTransitionKind::Recompute
                        && phase == SemanticSourceContentPhase::PackPolicy) =>
            {
                Ok(SemanticSourceContentFamily::for_prepared_section(
                    &self.owner,
                    phase,
                    kind,
                    (operation, bank),
                ))
            }
            _ => Err(publication_input_error(
                "content family changed its original historical or prepared child section",
            )),
        }
    }

    pub fn freeze_content_occurrences<
        A: super::source_content_occurrences::SourceContentAllocation,
    >(
        &self,
        events: Vec<super::source_content_occurrences::SemanticSourceContentEvent<A>>,
    ) -> Result<
        Vec<super::source_content_occurrences::SemanticSourceContentOccurrence>,
        SemanticTransitionError,
    > {
        let mut original = self.owner.occurrences.lock().map_err(|_| {
            publication_input_error("original content geometry custody is poisoned")
        })?;
        if original.is_some()
            || self
                .owner
                .reader
                .lock()
                .map_err(|_| {
                    publication_input_error("original content reader custody is poisoned")
                })?
                .is_some()
        {
            return Err(publication_input_error(
                "original content occurrences freeze once before physical binding",
            ));
        }
        let frozen = super::source_content_occurrences::FrozenSourceContentOccurrences::new(
            &self.owner,
            events,
        )?;
        let handles = frozen.occurrence_handles();
        *original = Some(frozen);
        Ok(handles)
    }
    pub fn kind(&self) -> SemanticSourceContentGeometryKind {
        self.owner.kind
    }
    pub fn step_index(&self) -> usize {
        self.owner.step_index
    }
    pub fn member_ordinal(&self) -> u64 {
        self.owner.ordinal
    }
    pub fn bank(&self) -> Option<usize> {
        self.owner.bank
    }
    pub fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }
    pub fn restore_material_mode(&self) -> Result<&'static str, SemanticTransitionError> {
        match self.owner.kind {
            SemanticSourceContentGeometryKind::Historical => Ok("checkpoint-model"),
            SemanticSourceContentGeometryKind::Current => Err(publication_input_error(
                "current geometry uses its original native-bank adoption, not checkpoint restoration",
            )),
        }
    }
    pub fn model_contract_material(
        &self,
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        self.owner.seed.proof.model_contract_material()
    }
    pub fn model_restore_context(
        &self,
    ) -> Result<
        (
            SemanticPublishedIdentity,
            u64,
            (u64, u64, u64),
            (Identity256, Identity256),
        ),
        SemanticTransitionError,
    > {
        if self.owner.kind == SemanticSourceContentGeometryKind::Current {
            return Err(publication_input_error(
                "CURRENT model context belongs to its original future actor producer",
            ));
        }
        Ok(self.owner.seed.proof.model_restore_context())
    }
    pub fn model_allocation_layouts(
        &self,
    ) -> impl Iterator<Item = (SemanticSourceGeometryAllocation, u64)> + '_ {
        self.owner
            .model_geometry()
            .0
            .allocation_bytes
            .iter()
            .copied()
            .enumerate()
            .map(|(index, span_bytes)| {
                (
                    SemanticSourceGeometryAllocation {
                        scope: Arc::clone(self.owner.model_scope()),
                        port: SourceGeometryAllocationPort::Model(index),
                        prepared_section: None,
                        span_bytes,
                    },
                    span_bytes,
                )
            })
    }
    pub fn model_storage_layouts(&self) -> &[SemanticModelStorage] {
        &self.owner.model_geometry().0.storages
    }
    pub fn model_view_layouts(
        &self,
    ) -> impl Iterator<Item = (SemanticTensorLayout, u64, u64)> + '_ {
        let (memory, layouts) = self.owner.model_geometry();
        memory.views.iter().map(move |view| {
            (
                layouts[&(view.role, view.index)],
                view.storage,
                view.byte_offset,
            )
        })
    }
    pub fn policy_tensor_layouts(
        &self,
        prepared_section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<[SemanticTensorLayout; 9], SemanticTransitionError> {
        self.owner.require_geometry_section(prepared_section)?;
        if prepared_section.is_some_and(|(kind, _)| kind != SemanticTransitionKind::Proposal) {
            return Err(publication_input_error(
                "policy geometry requires its original prepared Proposal bank",
            ));
        }
        // The actual nine producers are external. Geometry declares only their
        // typed rows; their original Capture/witness owns the late alias proof.
        Ok(self.owner.publication.policy_layouts)
    }

    pub fn policy_numerical_domain_layouts(
        &self,
        prepared_section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<[SemanticSourceGeometryTensor; 3], SemanticTransitionError> {
        self.owner.require_geometry_section(prepared_section)?;
        if prepared_section.is_some_and(|(kind, _)| kind != SemanticTransitionKind::Proposal) {
            return Err(publication_input_error(
                "uniform geometry requires its original prepared Proposal bank",
            ));
        }
        let tensor = |index| {
            self.owner.tensor_geometry(
                self.owner.publication.uniform_layouts[index],
                0,
                0,
                SourceGeometryAllocationPort::Uniform(index),
                0,
                prepared_section,
            )
        };
        Ok([tensor(0)?, tensor(1)?, tensor(2)?])
    }

    pub fn input_tensor_layouts(
        &self,
        prepared_section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<
        impl Iterator<Item = Result<SemanticSourceGeometryTensor, SemanticTransitionError>> + '_,
        SemanticTransitionError,
    > {
        self.owner.input_tensor_layouts(prepared_section)
    }
    pub fn feedback_tensor_layouts(
        &self,
        prepared_section: Option<(SemanticTransitionKind, usize)>,
    ) -> Result<[SemanticSourceGeometryTensor; 6], SemanticTransitionError> {
        self.owner.require_geometry_section(prepared_section)?;
        let mut tensors = self
            .owner
            .publication
            .feedback_layouts
            .iter()
            .copied()
            .enumerate()
            .map(|(index, layout)| {
                self.owner.tensor_geometry(
                    layout,
                    0,
                    0,
                    SourceGeometryAllocationPort::Feedback(index),
                    0,
                    prepared_section,
                )
            });
        Ok([
            tensors.next().expect("six original feedback layouts")?,
            tensors.next().expect("six original feedback layouts")?,
            tensors.next().expect("six original feedback layouts")?,
            tensors.next().expect("six original feedback layouts")?,
            tensors.next().expect("six original feedback layouts")?,
            tensors.next().expect("six original feedback layouts")?,
        ])
    }
    pub fn continuation_capacity(&self) -> Result<u64, SemanticTransitionError> {
        self.owner
            .seed
            .proof
            .owner
            .contract
            .prefix_capacity
            .checked_add(32)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }
    pub fn continuation_blocks(&self) -> Result<Option<usize>, SemanticTransitionError> {
        Ok(self.owner.seed.proof.owner.continuation_blocks)
    }
    pub(super) fn continuation_service_capacity(&self) -> Result<u64, SemanticTransitionError> {
        self.owner
            .seed
            .proof
            .owner
            .contract
            .feedback_capacity
            .checked_add(32)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }
}

impl SemanticTransitionSession {
    pub fn model_input_generation_port(
        &self,
        parent: &SemanticPublishedLease,
        admission: &SemanticSegmentInstructionAdmission,
        step_index: usize,
    ) -> Result<SemanticModelInputGenerationPort, SemanticTransitionError> {
        self.checked_reader(parent)?;
        let original = self.instruction_admission(admission)?;
        if original.parent != parent.identity || original.parent_token != parent.token {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let allowance = self
            .graph
            .cold_work_allowance()
            .ok_or(SemanticTransitionError::NotBound)?;
        let generations = allowance
            .lock()
            .map_err(|_| {
                publication_input_error("original model input generation allowance is poisoned")
            })?
            .model_input_generations(&original.instruction)?
            .ok_or(SemanticTransitionError::NotBound)?;
        let port = generations.port(step_index)?;
        port.require_original_instruction(self, admission)?;
        Ok(port)
    }

    pub(super) fn bind_prepared_model_input_generations(
        &mut self,
    ) -> Result<(), SemanticTransitionError> {
        let build = self
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let Some(admission) = &build.program_admission else {
            return Ok(());
        };
        let instruction = &self.instruction_admission(admission)?.instruction;
        let Some(allowance) = self.graph.cold_work_allowance() else {
            return Ok(());
        };
        let generations = {
            let allowance = allowance.lock().map_err(|_| {
                publication_input_error("original model input generation allowance is poisoned")
            })?;
            let original = allowance.model_input_generations(instruction)?;
            if original.is_none()
                && allowance.is_source_prefix()
                && build.transitions.contains(&SemanticTransitionKind::Update)
            {
                return Err(publication_input_error(
                    "original Update inputs require their complete early generation declarations",
                ));
            }
            original
        };
        if let Some(generations) = generations {
            if self
                .prepared_segment
                .as_ref()
                .expect("authenticated original segment")
                .model_input_generations
                .is_some()
            {
                return Err(publication_input_error(
                    "original prepared segment already retains its sole generation roster",
                ));
            }
            self.prepared_segment
                .as_mut()
                .expect("authenticated original segment")
                .model_input_generations = Some(Arc::clone(&generations));
            generations.bind_segment(self)?;
        }
        Ok(())
    }

    pub(super) fn prepare_prepared_content_retirement(
        &self,
    ) -> Result<Option<PreparedContentRetirement>, SemanticTransitionError> {
        let Some(staged) = &self.actor_refresh else {
            return Ok(None);
        };
        let geometry = &staged.proof.inner.content_geometry[staged.target_bank];
        geometry.owner.require_current_child(self)?;
        let frozen = geometry
            .owner
            .occurrences
            .lock()
            .map_err(|_| publication_input_error("original prepared content custody is poisoned"))?
            .as_ref()
            .map(Arc::clone)
            .ok_or(SemanticTransitionError::NotBound)?;
        let vector_bytes = if self.prepared_resources.len() == self.prepared_resources.capacity() {
            self.prepared_resources
                .len()
                .checked_add(1)
                .and_then(|count| count.checked_mul(size_of::<Arc<dyn Send + Sync>>()))
                .ok_or(SemanticTransitionError::GenerationExhausted)?
        } else {
            0
        };
        let bytes = size_of::<PreparedContentRetirement>()
            .checked_add(2 * size_of::<usize>())
            .and_then(|bytes| bytes.checked_add(vector_bytes))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = geometry.material_budget().reserve_auxiliary(
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        Ok(Some(PreparedContentRetirement {
            frozen,
            armed: false,
            _reservation: reservation,
        }))
    }

    pub fn source_content_geometry_member_ordinals<'a>(
        &'a self,
        parent: &SemanticPublishedLease,
        admission: &SemanticSegmentInstructionAdmission,
    ) -> Result<impl Iterator<Item = u64> + 'a, SemanticTransitionError> {
        self.checked_reader(parent)?;
        self.admitted_segment_steps(admission)?;
        let arena = self
            .training_views
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let (actors, edits) = arena.original_policy_group_members();
        Ok(actors
            .iter()
            .chain(edits)
            .enumerate()
            .filter_map(move |(index, member)| {
                (!actors
                    .iter()
                    .chain(edits)
                    .take(index)
                    .any(|previous| previous.ordinal == member.ordinal))
                .then_some(member.ordinal)
            }))
    }

    pub fn prepare_source_content_geometry(
        &mut self,
        parent: &SemanticPublishedLease,
        admission: &SemanticSegmentInstructionAdmission,
        work: &SemanticColdModelWork,
        seeds: Vec<(u64, SemanticSourceContentGeometrySeed)>,
    ) -> Result<Arc<[SemanticPreparedSourceContentGeometry]>, SemanticTransitionError> {
        self.require_original_admitted_cold_model_work(admission, parent, work)?;
        let allowance = self.original_source_geometry_allowance(work)?;
        let instruction = Arc::clone(&self.instruction_admission(admission)?.instruction);
        if let Some(original) = self.original_source_geometry_roster(work)? {
            original
                .input_generations
                .require_instruction(&instruction)?;
            if !Arc::ptr_eq(&original.instruction, &instruction)
                || original.seeds.len() != seeds.len()
                || original.seeds.iter().zip(&seeds).any(
                    |((ordinal, original), (actual_ordinal, actual))| {
                        ordinal != actual_ordinal
                            || original.proof.owner.source_digest
                                != actual.proof.owner.source_digest
                            || !Arc::ptr_eq(&original.admission, &actual.admission)
                            || !Arc::ptr_eq(&original.task, &actual.task)
                    },
                )
            {
                return Err(publication_input_error(
                    "original SOURCE geometry cannot be replaced",
                ));
            }
            return Ok(Arc::clone(&original.handles));
        }
        let steps = self.admitted_segment_steps(admission)?;
        let updates = steps
            .iter()
            .filter(|step| step.kind == Some(SemanticTransitionKind::Update))
            .count();
        let arena = self
            .training_views
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let (actors, edits) = arena.original_policy_group_members();
        for (ordinal, seed) in &seeds {
            let origin =
                crate::semantic_training_view::origin_record(seed.material.training_view_origin()?);
            if !actors
                .iter()
                .chain(edits)
                .any(|member| member.ordinal == *ordinal && member.origin == origin)
                || seeds.iter().filter(|(actual, _)| actual == ordinal).count() != 1
            {
                return Err(publication_input_error(
                    "SOURCE geometry changed its original complete member roster",
                ));
            }
            seed.proof.require_original_admission(&seed.admission)?;
            seed.material
                .require_predecessor_geometry(&seed.proof, &seed.admission)?;
            seed.material.require_checkpoint_proof_budget(&seed.proof)?;
            let budget = self
                .replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?;
            if !budget.same_pool(&seed.proof.owner.budget) {
                return Err(publication_input_error(
                    "SOURCE geometry changed its original material pool",
                ));
            }
        }
        if actors
            .iter()
            .chain(edits)
            .any(|member| !seeds.iter().any(|(ordinal, _)| *ordinal == member.ordinal))
        {
            return Err(publication_input_error(
                "SOURCE geometry omitted an original member",
            ));
        }
        let per_update = actors
            .len()
            .checked_mul(2)
            .and_then(|count| count.checked_add(edits.len()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let count = updates
            .checked_mul(per_update)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = count
            .checked_mul(
                size_of::<SourceContentGeometryOwner>()
                    + 4 * size_of::<usize>()
                    + 2 * size_of::<SemanticPreparedSourceContentGeometry>(),
            )
            .and_then(|bytes| {
                bytes.checked_add(seeds.len().checked_mul(
                    size_of::<SemanticSourceContentGeometrySeed>()
                        + 2 * size_of::<usize>()
                        + size_of::<(u64, Arc<SemanticSourceContentGeometrySeed>)>()
                        + size_of::<Arc<SourcePublicationGeometry>>(),
                )?)
            })
            .and_then(|bytes| {
                bytes
                    .checked_add(size_of::<OriginalSourceGeometryRoster>() + 2 * size_of::<usize>())
                    .and_then(|bytes| {
                        bytes.checked_add(
                            size_of::<SemanticReplayMaterialReservation>() + 4 * size_of::<usize>(),
                        )
                    })
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let reservation = Arc::new(budget.reserve_auxiliary(
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?);
        let input_generations = OriginalModelInputGenerations::new(
            self,
            parent,
            Arc::clone(&instruction),
            steps.iter().enumerate().filter_map(|(index, step)| {
                (step.kind == Some(SemanticTransitionKind::Update)).then_some(index)
            }),
            budget,
        )?;
        let mut original_seeds = Vec::with_capacity(seeds.len());
        let mut projections = Vec::with_capacity(seeds.len());
        for (ordinal, seed) in seeds {
            let geometry = Arc::new(SourcePublicationGeometry::new(&seed)?);
            original_seeds.push((ordinal, Arc::new(seed)));
            projections.push(geometry);
        }
        let mut handles = Vec::with_capacity(count);
        for (step_index, step) in steps.iter().enumerate() {
            if step.kind != Some(SemanticTransitionKind::Update) {
                continue;
            }
            for (kind, members) in [
                (SemanticSourceContentGeometryKind::Historical, edits),
                (SemanticSourceContentGeometryKind::Current, actors),
            ] {
                for member in members {
                    let index = original_seeds
                        .iter()
                        .position(|(ordinal, _)| *ordinal == member.ordinal)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    let banks: &[Option<usize>] =
                        if kind == SemanticSourceContentGeometryKind::Current {
                            &[Some(0), Some(1)]
                        } else {
                            &[None]
                        };
                    for &bank in banks {
                        handles.push(SemanticPreparedSourceContentGeometry {
                            owner: Arc::new(SourceContentGeometryOwner {
                                work: work.clone(),
                                instruction: Arc::clone(&instruction),
                                allowance: Arc::clone(&allowance),
                                kind,
                                step_index,
                                ordinal: member.ordinal,
                                bank,
                                member: *member,
                                seed: Arc::clone(&original_seeds[index].1),
                                publication: Arc::clone(&projections[index]),
                                scope: Arc::new(()),
                                recipe: Mutex::new(None),
                                current: Mutex::new(None),
                                reader: Mutex::new(None),
                                occurrences: Mutex::new(None),
                                uniform: Mutex::new(None),
                                input_generation: if kind
                                    == SemanticSourceContentGeometryKind::Current
                                {
                                    Some(input_generations.port(step_index)?)
                                } else {
                                    None
                                },
                                _reservation: Arc::clone(&reservation),
                            }),
                        });
                    }
                }
            }
        }
        let handles: Arc<[SemanticPreparedSourceContentGeometry]> = handles.into();
        let result = Arc::clone(&handles);
        let original = Arc::new(OriginalSourceGeometryRoster {
            instruction,
            handles,
            seeds: original_seeds,
            input_generations: Arc::clone(&input_generations),
            _reservation: reservation,
        });
        self.register_original_source_geometry_roster(work, original)?;
        allowance
            .lock()
            .map_err(|_| {
                publication_input_error("original model input generation allowance is poisoned")
            })?
            .register_model_input_generations(&input_generations)?;
        Ok(result)
    }
}

/// Pure projection of the original decoder's capacities through the canonical
/// publication allocator and input planner. It owns no device view or grant.
pub(super) struct SourcePublicationGeometry {
    proof: SemanticReplayStateMaterialProof,
    admission: Arc<SemanticReplayStateMaterialAdmission>,
    storage: PublicationStorageGeometry,
    inputs: Vec<StepInputPlan>,
    feedback_schema: SemanticFeedbackSchema,
    feedback_plan: FeedbackAllocationPlan,
    feedback_layouts: [SemanticTensorLayout; 6],
    pub(super) policy_layouts: [SemanticTensorLayout; 9],
    uniform_layouts: [SemanticTensorLayout; 3],
    policy_layout: SemanticPolicyLayout,
    support_layout: [std::ops::Range<usize>; COMPONENT_COUNT],
    support_cells: usize,
    pub(super) feedback_ceiling: [u64; 9],
    pub(super) snapshot_ceiling: [u64; 9],
    pub(super) contract_guard_ceiling: [u64; 9],
    pub(super) input_ceiling: [u64; 9],
    pub(super) input_dma_ceiling: [u64; 9],
    pub(super) resident_ceiling: [u64; 9],
    _reservation: SemanticReplayMaterialReservation,
}

impl SourcePublicationGeometry {
    pub(super) fn new(
        seed: &SemanticSourceContentGeometrySeed,
    ) -> Result<Self, SemanticTransitionError> {
        let proof = seed.proof.clone();
        let admission = Arc::clone(&seed.admission);
        seed.material
            .require_predecessor_geometry(&proof, &admission)?;
        proof.require_original_admission(&admission)?;
        let original = &proof.owner;
        let count = original.allocation_geometry.len();
        let tree_node = OwnedPublicationMaterial::tree_node_bound(size_of::<(u64, u64)>(), 0)?;
        let span = size_of::<(usize, u64, u64, usize, std::ops::Range<usize>)>();
        let per_input = size_of::<StepInputPlan>()
            .checked_add(
                tree_node
                    .checked_mul(2)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?,
            )
            .and_then(|bytes| bytes.checked_add(size_of::<(u64, u64)>()))
            .and_then(|bytes| bytes.checked_add(span.checked_mul(2)?))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let bytes = count
            .checked_mul(per_input)
            .and_then(|bytes| bytes.checked_add(size_of::<Self>()))
            .and_then(|bytes| {
                bytes.checked_add(original.model_memory.views.len().checked_mul(size_of::<(
                    SemanticStateRole,
                    u64,
                )>(
                ))?)
            })
            .and_then(|bytes| {
                bytes.checked_add(4 * size_of::<(usize, std::ops::Range<usize>, usize)>())
            })
            .and_then(|bytes| bytes.checked_add(18 * size_of::<SemanticPolicyFieldLayout>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = original.budget.reserve_auxiliary(
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let storage = publication_storage_geometry(
            &original.allocation_geometry,
            &original.layouts,
            &original.model_memory,
            &original.contract,
            Some(&original.budget),
        )?;
        let inputs = plan_step_inputs(
            &storage.bank_templates,
            &original.layouts,
            original.contract.prefix_capacity,
            &storage.allocation_bytes,
        )?;
        let directory = &storage.bank_templates[0];
        let contract = directory
            .iter()
            .find(|range| (range.role, range.index) == (44, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let model_ranges = directory
            .iter()
            .filter(|range| matches!(range.role, 18..=20))
            .count()
            .checked_add(1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let snapshot_ceiling = native_work_bound::model_snapshot_native_work_ceiling(
            model_ranges,
            usize::try_from(contract.length_bytes)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let contract_guard_ceiling =
            native_work_bound::model_contract_guard_geometry_native_work_ceiling(
                &original.contract,
                directory,
            )?;
        let allocation_bytes = |slot: usize| {
            storage
                .allocation_bytes
                .get(slot)
                .copied()
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or(SemanticTransitionError::ObservationMismatch)
        };
        let input_ceiling = native_work_bound::step_input_producer_geometry_native_work_ceiling(
            &original.contract,
            original.layouts.len(),
            storage.allocation_bytes.len(),
            PUBLICATION_ROLE_COUNT,
            &inputs,
            directory,
            allocation_bytes,
        )?;
        let input_dma_ceiling = native_work_bound::step_input_dma_ceiling(inputs.len())?;
        let keys = original
            .model_memory
            .views
            .iter()
            .map(|view| {
                Ok((
                    SemanticStateRole::from_code(view.role)
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                    view.index,
                ))
            })
            .collect::<Result<Vec<_>, SemanticTransitionError>>()?;
        let resident_ceiling =
            native_work_bound::resident_model_content_guard_geometry_native_work_ceiling(
                &original.model_memory,
                &original.layouts,
                directory,
                &keys,
                allocation_bytes,
            )?;
        let feedback_schema = SemanticFeedbackSchema::from_statement_lengths(
            original
                .allocation_geometry
                .iter()
                .filter(|range| range.role == 16)
                .map(|range| range.length)
                .chain(
                    (original.codebook_header[37] != 0)
                        .then_some(original.codebook_header[37] as usize),
                ),
        )?;
        let (capacities, _) = seed
            .capacities
            .native_work_geometry()
            .map_err(SemanticTransitionError::Semantic)?;
        let feedback_plan = FeedbackAllocationPlan::new(
            &feedback_schema,
            original.contract.feedback_capacity,
            capacities[2],
        )?;
        let feedback_layouts = feedback_plan.tensor_layouts(&feedback_schema)?;
        let mut feedback_output = [0; 9];
        for layout in &feedback_layouts {
            native_work_bound::add_native_work_ceiling(
                &mut feedback_output,
                native_work_bound::tensor_layout_native_work_ceiling(
                    layout,
                    0,
                    0,
                    tensor_layout_bytes(layout)?,
                    false,
                )?,
            )?;
        }
        let feedback_ceiling = native_work_bound::feedback_geometry_native_work_ceiling(
            directory,
            u64::try_from(feedback_plan.slots)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            u64::try_from(feedback_plan.support_capacity)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            &feedback_schema,
            capacities[1],
            capacities[3],
            &seed.task,
            feedback_output,
        )?;
        let (components, support_cells) =
            ActionCodebooks::component_geometry(&original.codebook_header, seed.program.is_some())?;
        let policy = SemanticPolicyLayout::from_components(&components)?;
        let mut support_layout = std::array::from_fn(|_| 0..0);
        for (component, span) in components.iter().zip(&mut support_layout) {
            let begin = usize::try_from(component.offset)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            let cardinality = usize::try_from(component.cardinality)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            *span = begin
                ..begin
                    .checked_add(cardinality)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let policy_layouts =
            policy_numerical_producer_layouts(support_cells, policy.parameter_cells)?;
        let uniform_layouts =
            policy_uniform_domain_tensor_layouts(support_cells, policy.parameter_cells)?;
        Ok(Self {
            proof,
            admission,
            storage,
            inputs,
            feedback_schema,
            feedback_plan,
            feedback_layouts,
            policy_layouts,
            uniform_layouts,
            policy_layout: policy,
            support_layout,
            support_cells,
            feedback_ceiling,
            snapshot_ceiling,
            contract_guard_ceiling,
            input_ceiling,
            input_dma_ceiling,
            resident_ceiling,
            _reservation: reservation,
        })
    }

    pub(super) fn require_original_allocation(
        &self,
        storage: &PublicationStorage,
        inputs: &PreparedStepInputs,
        directory: &[PublicationRange],
    ) -> Result<(), SemanticTransitionError> {
        self.proof.require_original_admission(&self.admission)?;
        let original = &self.proof.owner;
        if storage.model_memory != original.model_memory
            || storage.layouts != original.layouts
            || storage.contract_value.prefix_capacity != original.contract.prefix_capacity
            || storage.contract_value.feedback_capacity != original.contract.feedback_capacity
            || storage.allocations.len() != self.storage.allocation_bytes.len()
            || storage
                .allocations
                .iter()
                .zip(&self.storage.allocation_bytes)
                .any(|(actual, original)| actual.len() != *original)
            || inputs.plans.len() != self.inputs.len()
            || inputs
                .plans
                .iter()
                .zip(&self.inputs)
                .any(|(actual, original)| {
                    actual.role != original.role
                        || actual.index != original.index
                        || actual.layout != original.layout
                        || actual
                            .banks
                            .iter()
                            .zip(&original.banks)
                            .any(|(actual, original)| {
                                actual.storage_slot != original.storage_slot
                                    || actual.span != original.span
                            })
                })
            || directory.len() != self.storage.bank_templates[0].len()
            || directory
                .iter()
                .zip(&self.storage.bank_templates[0])
                .any(|(actual, original)| {
                    actual.role != original.role
                        || actual.index != original.index
                        || actual.storage_slot != original.storage_slot
                        || actual.offset_bytes != original.offset_bytes
                        || actual.length_bytes != original.length_bytes
                        || actual.logical_begin != original.logical_begin
                        || actual.logical_end != original.logical_end
                })
        {
            return Err(publication_input_error(
                "SOURCE allocation changed its original pre-factory geometry",
            ));
        }
        Ok(())
    }
}

/// One genuine SOURCE preparation. Structural aliases grant no numerical use.
#[derive(Clone)]
pub struct SemanticPreparedSourceContent {
    pub(super) owner: Arc<PreparedSourceContentOwner>,
}

pub(super) struct PreparedSourceContentOwner {
    pub(super) geometry: Arc<SourceContentGeometryOwner>,
    pub(super) issuer: Arc<()>,
    pub(super) token: u64,
    pub(super) identity: SemanticPublishedIdentity,
    pub(super) ordinal: u64,
    pub(super) material: Identity256,
    pub(super) origin: SemanticTrainingViewOriginRecord,
    target: SemanticPreparedStep,
    target_instruction: Arc<[u8]>,
    pub(super) allowance: Arc<Mutex<ColdNativeAllowance>>,
    pub(super) resident: std::sync::Weak<ResidentModelMemoryOwner>,
    storage: std::sync::Weak<PublicationStorage>,
    inputs: std::sync::Weak<PreparedStepInputs>,
    feedback: usize,
    directory: Vec<PublicationRange>,
    feedback_geometry: Mutex<Option<Vec<PreparedSemanticTensor>>>,
    feedback_ceiling: [u64; 9],
    pub(super) policy_layouts: [SemanticTensorLayout; 9],
    pub(super) continuation_capacity: u64,
    pub(super) continuation_blocks: Option<usize>,
    activation: Mutex<Option<SourceContentActivation>>,
    terminal: Mutex<Option<SourceContentTerminal>>,
    pub(super) consumer_stream: u64,
    ports_issued: std::sync::atomic::AtomicBool,
    ports: Mutex<Option<SourceContentPortsState>>,
    _metadata: Arc<SemanticReplayMaterialReservation>,
    destinations: Mutex<SourceDestinations>,
    pub(super) occurrences:
        Mutex<Option<Arc<super::source_content_occurrences::FrozenSourceContentOccurrences>>>,
}

struct SourceContentActivation {
    identity: super::cold_model_work::SemanticColdNativeWorkActivationIdentity,
    cursor: usize,
    commands: Vec<OriginalNativeCommand>,
    inputs: Vec<OriginalConsumerEdge>,
    output: Option<OriginalConsumerEdge>,
}

enum SourceContentPortsState {
    Pending(SourceContentPortsPlan),
    Issued {
        _reservation: Arc<SemanticReplayMaterialReservation>,
    },
}

struct SourceContentPortsPlan {
    header: PublicationHeader,
    contract: PublicationContract,
    schema: SemanticFeedbackSchema,
    terminal_state_allocation: DeviceAllocationProvenance,
    cache_geometry: Vec<SourceContentCachePort>,
    exports: Vec<SourceContentPortExport>,
    caches: Vec<SemanticPreparedSourceCache>,
    cursor: usize,
    storage: Arc<PublicationStorage>,
    aliases: Arc<()>,
    reservation: Arc<SemanticReplayMaterialReservation>,
}

struct SourceContentCachePort {
    layout: SemanticTensorLayout,
    logical_begin: u64,
    logical_end: u64,
    allocation: DeviceAllocationProvenance,
}

struct SourceContentPortExport {
    view: Option<DeviceMemoryView<u8>>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dtype: (u8, u8),
    tensor: Option<DlpackManagedTensor>,
}

impl SourceContentPortExport {
    fn tensor(
        view: DeviceMemoryView<u8>,
        layout: SemanticTensorLayout,
    ) -> Result<Self, SemanticTransitionError> {
        let rank = usize::try_from(layout.rank)
            .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
        if rank > layout.dimensions.len() || layout.element_bytes == 0 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mut shape = Vec::with_capacity(rank);
        let mut strides = Vec::with_capacity(rank);
        for (&dimension, &stride) in layout.dimensions[..rank]
            .iter()
            .zip(&layout.strides_bytes[..rank])
        {
            shape.push(
                i64::try_from(dimension)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            );
            strides.push(
                i64::try_from(stride / layout.element_bytes)
                    .map_err(|_| SemanticTransitionError::ObservationMismatch)?,
            );
        }
        let dtype = match layout.scalar_type {
            1 => (1, 8),
            2 => (1, 32),
            3 => (1, 64),
            4 => (2, 16),
            5 => (4, 16),
            6 => (2, 32),
            7 => (0, 64),
            8 => (6, 8),
            9 => (2, 64),
            _ => return Err(SemanticTransitionError::ObservationMismatch),
        };
        Ok(Self {
            view: Some(view),
            shape,
            strides,
            dtype,
            tensor: None,
        })
    }

    fn raw(
        view: DeviceMemoryView<u8>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: (u8, u8),
    ) -> Self {
        Self {
            view: Some(view),
            shape,
            strides,
            dtype,
            tensor: None,
        }
    }
}

enum SourceContentTerminal {
    FinalUse(SemanticRngBinding),
    Unsubmitted,
}

/// Issued only by the same Session after positive original terminal proof.
/// Dropping this receipt outside owner locks retires temporary physical custody.
pub struct SemanticSourceContentRetirement {
    owner: Arc<PreparedSourceContentOwner>,
}

pub(super) struct PreparedContentRetirement {
    frozen: Arc<super::source_content_occurrences::FrozenSourceContentOccurrences>,
    armed: bool,
    _reservation: SemanticReplayMaterialReservation,
}

impl PreparedContentRetirement {
    pub(super) fn arm(mut self) -> Arc<dyn Send + Sync> {
        self.armed = true;
        Arc::new(self)
    }
}

impl Drop for PreparedContentRetirement {
    fn drop(&mut self) {
        if self.armed {
            self.frozen.retire_physical_custody();
        }
    }
}

impl Drop for SemanticSourceContentRetirement {
    fn drop(&mut self) {
        let frozen = self
            .owner
            .occurrences
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(Arc::clone);
        let feedback = self
            .owner
            .feedback_geometry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let activation = self
            .owner
            .activation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let ports = self
            .owner
            .ports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let uniform = self
            .owner
            .geometry
            .uniform
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(frozen) = frozen {
            frozen.retire_physical_custody();
        }
        drop(feedback);
        drop(activation);
        drop(ports);
        drop(uniform);
    }
}

pub(super) struct PendingSourceContent {
    pub(super) occurrence: super::source_content_occurrences::SemanticSourceContentOccurrence,
    pub(super) claim: super::source_content_occurrences::SourceContentClaim,
    pub(super) index: usize,
    pub(super) consumer_stream: u64,
    pub(super) verify: bool,
}

struct SourceDestinations {
    banks: [Option<SourceDestination>; 2],
    sealed: bool,
}

pub(super) struct SourceDestination {
    pub(super) preparation: std::sync::Weak<PreparedImportedEditDelivery>,
    pub(super) issuer: Arc<()>,
    pub(super) scope: Arc<()>,
    pub(super) step: u64,
    pub(super) instruction: Arc<[u8]>,
    pub(super) updates: Vec<u64>,
    pub(super) members: Vec<crate::semantic_training_view::FrozenPolicyGroupMember>,
    pub(super) all_members: Vec<(
        u64,
        Vec<crate::semantic_training_view::FrozenPolicyGroupMember>,
    )>,
}

/// Sole original capsules, projected from the retained physical allocations.
pub struct SemanticPreparedSourceContentPorts {
    pub model_context: SemanticModelContext,
    pub prefix: (SemanticPublishedIdentity, DlpackManagedTensor),
    pub terminal_state: DlpackManagedTensor,
    pub terminal_state_allocation: DeviceAllocationProvenance,
    pub feedback: SemanticFeedback,
    pub caches: Vec<SemanticPreparedSourceCache>,
}

pub struct SemanticPreparedSourceCache {
    pub layout: SemanticTensorLayout,
    pub logical_begin: u64,
    pub logical_end: u64,
    pub tensor: DlpackManagedTensor,
    pub allocation: DeviceAllocationProvenance,
    pub allocation_tensor: DlpackManagedTensor,
}

impl SemanticPreparedSourceContent {
    pub fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }
}

impl PreparedSourceContentOwner {
    pub(super) fn matches_inputs(&self, inputs: &PreparedStepInputs) -> bool {
        self.inputs
            .upgrade()
            .is_some_and(|original| std::ptr::eq(&*original, inputs))
    }

    pub(super) fn require_feedback_content(
        &self,
        content: &TensorContentBuffers,
    ) -> Result<(), SemanticTransitionError> {
        let original = self.feedback_geometry.lock().map_err(|_| {
            publication_input_error("original SOURCE feedback geometry is poisoned")
        })?;
        let original = original
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if original.len() != content.tensors.len()
            || original
                .iter()
                .zip(&content.tensors)
                .any(|(original, actual)| !same_tensor_content_owner(original, actual))
        {
            return Err(publication_input_error(
                "SOURCE feedback seals changed their original producer owners",
            ));
        }
        Ok(())
    }
}

impl native_work_bound::FrozenColdSourceContent {
    pub(super) fn new(
        allowance: &Arc<Mutex<ColdNativeAllowance>>,
        instruction: &Arc<[u8]>,
        sources: &[SemanticPreparedSourceContent],
        target_roster: Option<
            &[(
                u64,
                Vec<crate::semantic_training_view::FrozenPolicyGroupMember>,
            )],
        >,
    ) -> Result<Self, SemanticTransitionError> {
        let mut recipes = Vec::with_capacity(sources.len());
        let mut ceiling = [0; 9];
        let mut groups = BTreeMap::<u64, Vec<&PreparedSourceContentOwner>>::new();
        let mut previous_update = None;
        let mut original_roster = target_roster.map(|roster| {
            roster
                .iter()
                .map(|(step, members)| {
                    (
                        *step,
                        members
                            .iter()
                            .map(|member| (member.ordinal, member.origin))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        });
        for source in sources {
            let owner = &source.owner;
            if !Arc::ptr_eq(allowance, &owner.allowance)
                || recipes
                    .iter()
                    .any(|recipe: &native_work_bound::FrozenSourceRecipe| {
                        recipe.owner.ptr_eq(&Arc::downgrade(owner))
                    })
                || owner
                    .activation
                    .lock()
                    .map_err(|_| publication_input_error("original SOURCE activation is poisoned"))?
                    .is_some()
            {
                return Err(publication_input_error("SOURCE union requires distinct unentered recipes from the same original report"));
            }
            let destinations = owner.destinations.lock().map_err(|_| {
                publication_input_error("original SOURCE destinations are poisoned")
            })?;
            if !destinations.sealed
                || destinations.banks.iter().any(|bank| {
                    bank.as_ref()
                        .is_none_or(|bank| bank.preparation.upgrade().is_none())
                })
            {
                return Err(publication_input_error(
                    "SOURCE union requires both original sealed TARGET preparation receipts",
                ));
            }
            let target = destinations.banks[0]
                .as_ref()
                .expect("sealed original destination");
            if !Arc::ptr_eq(&target.instruction, instruction) {
                return Err(publication_input_error(
                    "SOURCE union belongs to another original native instruction",
                ));
            }
            if let Some(original) = &original_roster {
                if original
                    != &target
                        .all_members
                        .iter()
                        .map(|(step, members)| {
                            (
                                *step,
                                members
                                    .iter()
                                    .map(|member| (member.ordinal, member.origin))
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>()
                {
                    return Err(publication_input_error(
                        "SOURCE receipts changed the complete original target member roster",
                    ));
                }
            } else {
                original_roster = Some(
                    target
                        .all_members
                        .iter()
                        .map(|(step, members)| {
                            (
                                *step,
                                members
                                    .iter()
                                    .map(|member| (member.ordinal, member.origin))
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
            }
            let update = target
                .updates
                .iter()
                .position(|token| *token == target.step)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            if previous_update.is_some_and(|previous| previous > update) {
                return Err(publication_input_error(
                    "SOURCE union changed the original Update order",
                ));
            }
            previous_update = Some(update);
            groups.entry(target.step).or_default().push(owner);
            let frozen = owner.occurrences.lock().map_err(|_| {
                publication_input_error("original SOURCE occurrence roster is poisoned")
            })?;
            let frozen = frozen.as_ref().ok_or_else(|| {
                publication_input_error("SOURCE union lacks its whole frozen expression roster")
            })?;
            frozen.require_original_recipe(owner)?;
            let storage = owner
                .storage
                .upgrade()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let inputs = owner
                .inputs
                .upgrade()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let resident = owner.resident.upgrade().ok_or_else(|| {
                publication_input_error("SOURCE union lost its original resident carrier")
            })?;
            let model_ranges = owner
                .directory
                .iter()
                .filter(|range| matches!(range.role, 18..=20))
                .count()
                .checked_add(1)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let contract = owner
                .directory
                .iter()
                .find(|range| range.role == 44 && range.index == 0)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let snapshot = native_work_bound::model_snapshot_native_work_ceiling(
                model_ranges,
                usize::try_from(contract.length_bytes)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )?;
            let guard = native_work_bound::model_contract_guard_native_work_ceiling(
                &storage,
                &owner.directory,
            )?;
            let input_native = native_work_bound::step_input_producer_native_work_ceiling(
                &storage,
                &inputs.plans,
                &owner.directory,
            )?;
            let input_dma = native_work_bound::step_input_dma_ceiling(inputs.plans.len())?;
            let resident_ceiling = resident.native_guard_ceiling()?;
            let mut dma = snapshot;
            native_work_bound::add_native_work_ceiling(&mut dma, input_dma)?;
            for producer in [
                snapshot,
                guard,
                input_native,
                input_dma,
                resident_ceiling,
                owner.feedback_ceiling,
                frozen.native_work_ceiling(),
            ] {
                native_work_bound::add_native_work_ceiling(&mut ceiling, producer)?;
            }
            recipes.push(native_work_bound::FrozenSourceRecipe {
                owner: Arc::downgrade(owner),
                resident: owner.resident.clone(),
                resident_ceiling,
                preparation_claimed: false,
                resident_claimed: false,
                dma_ceiling: dma,
                model_snapshot: Some(snapshot),
                model_contract_guard: Some(guard),
                step_input_preparation: Some((input_native, input_dma)),
            });
        }
        // Each original target training view supplies the complete member
        // roster. Tuple lengths and caller-provided ordinals do not define it.
        for owners in groups.values() {
            let destinations = owners[0].destinations.lock().map_err(|_| {
                publication_input_error("original SOURCE destinations are poisoned")
            })?;
            let target = destinations.banks[0]
                .as_ref()
                .expect("sealed original destination");
            if target.members.len() != owners.len()
                || target.members.iter().zip(owners).any(|(member, owner)| {
                    member.ordinal != owner.ordinal || member.origin != owner.origin
                })
            {
                return Err(publication_input_error(
                    "SOURCE union omitted or reordered an original target Edit member",
                ));
            }
        }
        if let Some(original) = original_roster {
            if original.iter().any(|(step, members)| {
                groups.get(step).map_or(!members.is_empty(), |owners| {
                    members.len() != owners.len()
                        || members
                            .iter()
                            .zip(owners)
                            .any(|((ordinal, origin), owner)| {
                                *ordinal != owner.ordinal || *origin != owner.origin
                            })
                })
            }) {
                return Err(publication_input_error(
                    "SOURCE union omitted an original target Update group",
                ));
            }
        }
        native_work_bound::native_work_ceiling_units(ceiling)?;
        Ok(Self { recipes, ceiling })
    }
}

/// Price the same canonical content producer from original structural owners.
/// No tensor bytes, model seals or numerical completion witness are read here.
pub(super) fn source_geometry_occurrence_native_work_ceiling<
    A: super::source_content_occurrences::SourceContentAllocation,
>(
    owner: &SourceContentGeometryOwner,
    event: &super::source_content_occurrences::SemanticSourceContentEvent<A>,
) -> Result<[u64; 9], SemanticTransitionError> {
    use super::source_content_occurrences::{SemanticSourceContentKind, SourceContentAllocation};
    let original = &owner.seed.proof.owner;
    let directory = &owner.publication.storage.bank_templates[0];
    let verify = event.family.kind() != SemanticSourceContentKind::Capture;
    let model = verify
        && event
            .tensors
            .iter()
            .all(|tensor| matches!(tensor.layout.role, 18..=20));
    let mut ceiling = [0; 9];
    if model {
        model_content_ranges(
            directory,
            &original.layouts,
            event
                .tensors
                .iter()
                .map(|tensor| {
                    Ok((
                        tensor.layout,
                        tensor.logical_begin,
                        tensor.logical_end,
                        tensor_layout_bytes(&tensor.layout)?,
                    ))
                })
                .collect::<Result<Vec<_>, SemanticTransitionError>>()?
                .into_iter(),
        )?;
        let contract = directory
            .iter()
            .find(|range| (range.role, range.index) == (44, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        native_work_bound::add_native_work_ceiling(
            &mut ceiling,
            native_work_bound::content_digest_ceiling(contract.length_bytes, 21, false)?,
        )?;
    }
    let mut input_guard = false;
    for descriptor in &event.tensors {
        let bytes = tensor_layout_bytes(&descriptor.layout)?;
        let allocation = event
            .allocations
            .iter()
            .find(|allocation| allocation.slot == descriptor.allocation_slot)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let (native_input, producer_sealed) =
            if let Some(declaration) = allocation.native_allocation.as_ref() {
                if declaration.native_provenance().is_some() {
                    return Err(publication_input_error(
                    "pre-factory native content requires its original geometry allocation handle",
                ));
                }
                let token = declaration
                    .original_geometry()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?;
                owner.require_allocation(token, allocation.span_bytes)?;
                owner.classify_tensor(token, descriptor, event.family.prepared_section())?
            } else {
                (false, false)
            };
        if !model && native_input {
            if !input_guard {
                let basis = native_work_bound::step_input_basis_geometry_native_work_ceiling(
                    &owner.publication.inputs,
                    |slot| {
                        owner
                            .publication
                            .storage
                            .allocation_bytes
                            .get(slot)
                            .copied()
                            .and_then(|bytes| u64::try_from(bytes).ok())
                            .ok_or(SemanticTransitionError::ObservationMismatch)
                    },
                )?;
                native_work_bound::add_native_work_ceiling(&mut ceiling, basis)?;
                input_guard = true;
            }
        } else {
            native_work_bound::add_native_work_ceiling(
                &mut ceiling,
                native_work_bound::tensor_layout_native_work_ceiling(
                    &descriptor.layout,
                    descriptor.logical_begin,
                    descriptor.logical_end,
                    bytes,
                    verify || producer_sealed,
                )?,
            )?;
        }
    }
    native_work_bound::native_work_ceiling_units(ceiling)?;
    Ok(ceiling)
}

impl SemanticTransitionSession {
    pub(super) fn guard_source_continuation_content(
        &mut self,
        binding: &TextBindingStorage,
        kind: super::source_content_occurrences::SourceNativeContentGuard,
    ) -> Result<(), SemanticTransitionError> {
        self.check_continuation_content(binding)?;
        self.retain_source_native_content_guard(&binding._witness, binding.consumer_stream, kind)?;
        self.guard_continuation_content(binding)
    }

    pub(super) fn retain_source_native_content_guard(
        &self,
        witness: &SemanticTensorContentWitness,
        stream: u64,
        kind: super::source_content_occurrences::SourceNativeContentGuard,
    ) -> Result<(), SemanticTransitionError> {
        let step = self
            .steps
            .get(&witness.reader_token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let Some(owner) = &step.source_content else {
            return Ok(());
        };
        // Source execution and backward consumers are not turned into cold
        // work by keeping structural recipe metadata after their preparation.
        let Some(work) = self.cold_native_work(witness.reader_token)? else {
            return Ok(());
        };
        let allowance = self
            .cold_native_allowance(witness.reader_token)?
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !allowance
            .lock()
            .map_err(|_| publication_input_error("original native allowance is poisoned"))?
            .is_source_prefix()
        {
            return Ok(());
        }
        if !Arc::ptr_eq(&owner.issuer, &witness.issuer)
            || owner.token != witness.reader_token
            || !Arc::ptr_eq(&step.content_witnesses, &witness._witness)
            || witness.index >= step.content.len()
            || stream != owner.consumer_stream
            || !Arc::ptr_eq(&allowance, &owner.allowance)
            || owner
                .activation
                .lock()
                .map_err(|_| publication_input_error("original SOURCE activation is poisoned"))?
                .as_ref()
                .is_none_or(|activation| activation.cursor != 4)
            || step.pending_source_content.is_some()
        {
            return Err(publication_input_error(
                "SOURCE native consumer changed its original owner, stream or finite admission",
            ));
        }
        allowance
            .lock()
            .map_err(|_| publication_input_error("original SOURCE allowance is poisoned"))?
            .require_source_content_admitted(owner)?;
        let borrowed = self.graph.borrowed_cold_work().is_some_and(|original| {
            original.device_ptr() == work.device_ptr() && original.len() == work.len()
        });
        self.graph
            .require_cold_work_owner(&work, borrowed)
            .map_err(SemanticTransitionError::Semantic)?;
        let frozen = owner
            .occurrences
            .lock()
            .map_err(|_| publication_input_error("original SOURCE occurrence custody is poisoned"))?
            .as_ref()
            .map(Arc::clone)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        frozen.require_original_recipe(owner)?;
        let claim = frozen.claim_native(kind, witness)?;
        step.content[witness.index].retain_source_native_guard_batch(
            &self.domain,
            &claim,
            &allowance,
        )
    }

    pub(super) fn complete_source_content_final_use(
        &self,
        witness: &SemanticTensorContentWitness,
        invocation: SemanticRngBinding,
    ) -> Result<(), SemanticTransitionError> {
        let Some(step) = self.steps.get(&witness.reader_token) else {
            return Err(SemanticTransitionError::ObservationMismatch);
        };
        let Some(owner) = &step.source_content else {
            return Ok(());
        };
        if !Arc::ptr_eq(&owner.issuer, &witness.issuer)
            || !Arc::ptr_eq(&owner.issuer, &self.publication_issuer)
            || owner.token != witness.reader_token
            || !Arc::ptr_eq(&step.content_witnesses, &witness._witness)
            || witness.index >= step.content.len()
            || step.pending_source_content.is_some()
            || owner
                .activation
                .lock()
                .map_err(|_| publication_input_error("original SOURCE activation is poisoned"))?
                .as_ref()
                .is_none_or(|activation| activation.cursor != 4)
        {
            return Err(publication_input_error(
                "SOURCE final use changed its original witness or still has unresolved preparation",
            ));
        }
        owner
            .occurrences
            .lock()
            .map_err(|_| publication_input_error("original SOURCE occurrence custody is poisoned"))?
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .require_complete()?;
        let mut terminal = owner
            .terminal
            .lock()
            .map_err(|_| publication_input_error("original SOURCE terminal custody is poisoned"))?;
        if terminal.is_some() {
            return Err(publication_input_error(
                "SOURCE terminal proof cannot replace its original final use",
            ));
        }
        *terminal = Some(SourceContentTerminal::FinalUse(invocation));
        Ok(())
    }

    pub fn retain_source_content_non_submission(
        &self,
        lease: &SemanticPublishedLease,
        recipe: &SemanticPreparedSourceContent,
        target: &SemanticTransitionSession,
        step: &SemanticPreparedStep,
        proof: &SemanticPreparedSegmentNonSubmission,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self.original_source_content(lease, recipe)?;
        target.require_cancelled_prepared_step_final_use(step, proof)?;
        if !owner.target.same_handle(step)
            || !Arc::ptr_eq(&step.issuer, &target.publication_issuer)
            || owner
                .activation
                .lock()
                .map_err(|_| publication_input_error("original SOURCE activation is poisoned"))?
                .is_some()
            || self.steps[&lease.token].pending_source_content.is_some()
            || self.steps[&lease.token].pending_cold_content.is_some()
            || owner.inputs.upgrade().is_none_or(|inputs| {
                inputs
                    .original_preparation
                    .lock()
                    .map_or(true, |original| original.is_some())
            })
        {
            return Err(publication_input_error(
                "SOURCE non-submission lost its original unentered preparation or target receipt",
            ));
        }
        let mut terminal = owner
            .terminal
            .lock()
            .map_err(|_| publication_input_error("original SOURCE terminal custody is poisoned"))?;
        if matches!(&*terminal, Some(SourceContentTerminal::Unsubmitted)) {
            return Ok(());
        }
        if terminal.is_some() {
            return Err(publication_input_error(
                "SOURCE terminal proof is already retained",
            ));
        }
        *terminal = Some(SourceContentTerminal::Unsubmitted);
        Ok(())
    }

    pub fn source_content_retirement(
        &self,
        lease: &SemanticPublishedLease,
        recipe: &SemanticPreparedSourceContent,
    ) -> Result<SemanticSourceContentRetirement, SemanticTransitionError> {
        let owner = self.original_source_content(lease, recipe)?;
        if owner
            .terminal
            .lock()
            .map_err(|_| publication_input_error("original SOURCE terminal custody is poisoned"))?
            .is_none()
            || self.steps[&lease.token].pending_source_content.is_some()
            || self.steps[&lease.token].pending_cold_content.is_some()
        {
            return Err(publication_input_error(
                "SOURCE custody retirement requires positive original native terminal proof",
            ));
        }
        Ok(SemanticSourceContentRetirement { owner })
    }

    pub fn activate_prepared_source_content(
        &mut self,
        lease: &SemanticPublishedLease,
        recipe: &SemanticPreparedSourceContent,
        stream: u64,
        work: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(), SemanticTransitionError> {
        let identity = self.require_cold_model_work_activation(work, region, lease)?;
        self.activate_prepared_source_content_original(lease, recipe, stream, identity)
    }

    pub fn activate_prepared_source_content_with_permit(
        &mut self,
        lease: &SemanticPublishedLease,
        recipe: &SemanticPreparedSourceContent,
        stream: u64,
        permit: &SemanticColdNativeWorkActivationPermit<'_>,
    ) -> Result<(), SemanticTransitionError> {
        let identity = permit.identity_for_callback(self, lease)?;
        self.activate_prepared_source_content_original(lease, recipe, stream, identity)
    }

    fn activate_prepared_source_content_original(
        &mut self,
        lease: &SemanticPublishedLease,
        recipe: &SemanticPreparedSourceContent,
        stream: u64,
        identity: SemanticColdNativeWorkActivationIdentity,
    ) -> Result<(), SemanticTransitionError> {
        self.require_original_cold_content_reader(lease)?;
        let original = self.original_source_content_binding(lease, recipe)?;
        if stream != original.consumer_stream
            || !Arc::ptr_eq(identity.allowance(), &original.allowance)
            || self.has_pending_state_restoration_except(None, None)
            || self.resident_model_activation.is_some()
            || self.replay_verification_pending.load(Ordering::Acquire)
            || self.has_pending_cold_model_work_completion(None)
            || self
                .readers
                .values()
                .any(|reader| reader.retirement_pending || reader.acquisition_pending)
            || self
                .steps
                .values()
                .any(|step| step.consumer_completion.is_some())
        {
            return Err(publication_input_error(
                "SOURCE activation lost its original live recording context",
            ));
        }
        let mut retained = original
            .activation
            .lock()
            .map_err(|_| publication_input_error("original SOURCE activation is poisoned"))?;
        if retained.is_none() {
            original
                .allowance
                .lock()
                .map_err(|_| publication_input_error("original SOURCE allowance is poisoned"))?
                .claim_source_preparation(&original)?;
            // Identity and cursor exist before the first fallible event or
            // command construction. A retry can fill only its unissued tail.
            *retained = Some(SourceContentActivation {
                identity,
                cursor: 0,
                commands: Vec::new(),
                inputs: Vec::new(),
                output: None,
            });
        } else if !retained
            .as_ref()
            .expect("retained original activation")
            .identity
            .same_context(&identity)
        {
            return Err(publication_input_error(
                "SOURCE activation changed its original admitted callback",
            ));
        }
        let activation = retained.as_mut().expect("retained original activation");
        while activation.commands.len() < 2 {
            activation
                .commands
                .push(OriginalNativeCommand::new(&self.domain)?);
        }
        let consumer = dlpack_consumer_stream(stream)?;
        while activation.inputs.len() < 2 {
            let writer = if activation.inputs.is_empty() {
                std::ptr::null_mut()
            } else {
                consumer
            };
            activation.inputs.push(OriginalConsumerEdge::new(
                &self.stream,
                writer,
                self.stream.cu_stream(),
            )?);
        }
        if activation.output.is_none() {
            activation.output = Some(OriginalConsumerEdge::new(
                &self.stream,
                self.stream.cu_stream(),
                consumer,
            )?);
        }
        while activation.cursor < 4 {
            match activation.cursor {
                0 => {
                    for edge in &mut activation.inputs {
                        edge.join()?;
                    }
                }
                1 => {
                    let inputs = original
                        .inputs
                        .upgrade()
                        .ok_or(SemanticTransitionError::ObservationMismatch)?;
                    self.resume_original_step_inputs(lease, inputs)?;
                }
                2 => {
                    let descriptor = self.descriptor();
                    let step = &self.steps[&lease.token];
                    let feedback = &step.feedback[original.feedback];
                    feedback
                        .content
                        .retain_source_feedback_batch(&self.domain, &original)?;
                    let task = &self
                        .task
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?
                        .1;
                    feedback.enqueue_with_original(
                        &self.domain,
                        &mut self.poisoned,
                        &self.graph,
                        self.publication
                            .as_ref()
                            .ok_or(SemanticTransitionError::NotBound)?,
                        task,
                        &self.readers[&lease.token].device,
                        descriptor,
                        Some((
                            &mut activation.commands,
                            activation.identity.work(),
                            &original.allowance,
                        )),
                    )?;
                }
                3 => activation
                    .output
                    .as_mut()
                    .expect("retained original output edge")
                    .join()?,
                _ => unreachable!("bounded original SOURCE activation cursor"),
            }
            activation.cursor += 1;
        }
        Ok(())
    }

    pub(super) fn require_pending_source_occurrence(
        &self,
        token: u64,
        occurrence: Option<&super::source_content_occurrences::SemanticSourceContentOccurrence>,
    ) -> Result<(), SemanticTransitionError> {
        match (
            self.steps[&token].pending_source_content.as_ref(),
            occurrence,
        ) {
            (Some(pending), Some(actual)) if pending.occurrence.same_handle(actual) => Ok(()),
            (None, None) => Ok(()),
            _ => Err(publication_input_error(
                "SOURCE content continuation changed its original occurrence",
            )),
        }
    }

    pub(super) fn prepare_source_content_claim(
        &self,
        lease: &SemanticPublishedLease,
        receiver: Option<&SemanticTensorContentWitness>,
        tensors: &[PreparedSemanticTensor],
        consumer_stream: u64,
        model: bool,
        occurrence: Option<&super::source_content_occurrences::SemanticSourceContentOccurrence>,
    ) -> Result<
        Option<(
            super::source_content_occurrences::SemanticSourceContentOccurrence,
            super::source_content_occurrences::SourceContentClaim,
        )>,
        SemanticTransitionError,
    > {
        use super::source_content_occurrences::SemanticSourceContentKind;
        let Some(original) = &self.steps[&lease.token].source_content else {
            return if occurrence.is_none() {
                Ok(None)
            } else {
                Err(publication_input_error(
                    "content occurrence has no original SOURCE recipe",
                ))
            };
        };
        if original.consumer_stream != consumer_stream {
            return Err(publication_input_error(
                "SOURCE content changed its fixed original consumer stream",
            ));
        }
        original
            .allowance
            .lock()
            .map_err(|_| publication_input_error("original SOURCE allowance is poisoned"))?
            .require_source_content_admitted(original)?;
        if original
            .activation
            .lock()
            .map_err(|_| publication_input_error("original SOURCE activation is poisoned"))?
            .as_ref()
            .is_none_or(|activation| activation.cursor != 4)
        {
            return Err(publication_input_error(
                "SOURCE content precedes its completed original input and feedback preparation",
            ));
        }
        let occurrence = occurrence.ok_or_else(|| {
            publication_input_error("SOURCE content requires its frozen original occurrence")
        })?;
        let verify = model || receiver.is_some();
        let expected_kind = if model {
            SemanticSourceContentKind::BindModel
        } else if verify {
            SemanticSourceContentKind::Verify
        } else {
            SemanticSourceContentKind::Capture
        };
        if occurrence.kind() != expected_kind {
            return Err(publication_input_error(
                "SOURCE content occurrence changed its original capture or verification kind",
            ));
        }
        if let Some(pending) = &self.steps[&lease.token].pending_source_content {
            if !pending.occurrence.same_handle(occurrence)
                || pending.consumer_stream != consumer_stream
                || pending.verify != verify
            {
                return Err(publication_input_error(
                    "SOURCE content cannot replace its pending original occurrence",
                ));
            }
            let expected = &self.steps[&lease.token].content[pending.index].tensors;
            if expected.len() != tensors.len()
                || expected
                    .iter()
                    .zip(tensors)
                    .any(|(expected, actual)| !same_tensor_content_owner(expected, actual))
            {
                return Err(publication_input_error(
                    "SOURCE content continuation changed its original producer allocations",
                ));
            }
            return Ok(Some((occurrence.clone(), pending.claim.clone())));
        }
        let frozen = original.occurrences.lock().map_err(|_| {
            publication_input_error("original SOURCE occurrence roster is poisoned")
        })?;
        let frozen = frozen.as_ref().ok_or_else(|| {
            publication_input_error("SOURCE content has no frozen original occurrence roster")
        })?;
        frozen.require_original_recipe(original)?;
        frozen.require_original_occurrence(occurrence)?;
        let bind_uniform = occurrence.uses_uniform_domains() && {
            let original = original.geometry.uniform.lock().map_err(|_| {
                publication_input_error("original uniform domain custody is poisoned")
            })?;
            original.is_none()
        };
        if bind_uniform {
            original
                .geometry
                .bind_uniform_domain_layouts(self.policy_numerical_domain_layouts(lease)?)?;
        }
        let inputs = original
            .inputs
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let feedback = &self.steps[&lease.token].feedback[original.feedback]
            .content
            .tensors;
        for tensor in tensors
            .iter()
            .filter(|tensor| !model && tensor.native_allocation.is_some())
        {
            if !inputs.owns_tensor(tensor)?
                && !feedback
                    .iter()
                    .any(|expected| same_tensor_content_owner(expected, tensor))
            {
                return Err(publication_input_error("SOURCE native content allocation is not its actual original input or feedback producer"));
            }
        }
        let retained_receiver;
        let receiver = if model {
            None
        } else if let Some(receiver) = receiver {
            Some(receiver)
        } else {
            let (issuer, token, index) = frozen.receiver_content_identity(occurrence)?;
            if !Arc::ptr_eq(&issuer, &self.publication_issuer)
                || token != lease.token
                || index >= self.steps[&token].content.len()
            {
                return Err(publication_input_error(
                    "SOURCE capture lost its previously authenticated original receiver",
                ));
            }
            retained_receiver = SemanticTensorContentWitness {
                issuer,
                reader_token: token,
                index,
                _witness: Arc::clone(&self.steps[&token].content_witnesses),
            };
            Some(&retained_receiver)
        };
        let claim = frozen.claim(occurrence, receiver, tensors)?;
        Ok(Some((occurrence.clone(), claim)))
    }

    pub fn source_content_family(
        &self,
        lease: &SemanticPublishedLease,
        recipe: &SemanticPreparedSourceContent,
        phase: super::source_content_occurrences::SemanticSourceContentPhase,
        kind: super::source_content_occurrences::SemanticSourceContentKind,
    ) -> Result<
        super::source_content_occurrences::SemanticSourceContentFamily,
        SemanticTransitionError,
    > {
        let original = self.original_source_content(lease, recipe)?;
        original
            .allowance
            .lock()
            .map_err(|_| publication_input_error("original SOURCE allowance is poisoned"))?
            .require_source_preparation()?;
        Ok(
            super::source_content_occurrences::SemanticSourceContentFamily::new(
                &original.geometry,
                phase,
                kind,
            ),
        )
    }

    pub fn freeze_source_content_occurrences<
        A: super::source_content_occurrences::SourceContentAllocation,
    >(
        &self,
        lease: &SemanticPublishedLease,
        recipe: &SemanticPreparedSourceContent,
        events: Vec<super::source_content_occurrences::SemanticSourceContentEvent<A>>,
    ) -> Result<
        Vec<super::source_content_occurrences::SemanticSourceContentOccurrence>,
        SemanticTransitionError,
    > {
        let original = self.original_source_content(lease, recipe)?;
        SemanticPreparedSourceContentGeometry {
            owner: Arc::clone(&original.geometry),
        }
        .freeze_content_occurrences(events)
    }

    /// Retrieve the sole already-frozen declaration after a fallible binding
    /// handoff. This never freezes, reprices or issues a new native owner.
    pub fn source_content_occurrences(
        &self,
        lease: &SemanticPublishedLease,
        recipe: &SemanticPreparedSourceContent,
    ) -> Result<
        Vec<super::source_content_occurrences::SemanticSourceContentOccurrence>,
        SemanticTransitionError,
    > {
        let original = self.original_source_content(lease, recipe)?;
        let frozen = original.occurrences.lock().map_err(|_| {
            publication_input_error("original SOURCE occurrence roster is poisoned")
        })?;
        Ok(frozen
            .as_ref()
            .ok_or_else(|| {
                publication_input_error("original SOURCE occurrence roster has not been frozen")
            })?
            .occurrence_handles())
    }

    /// Allocate the original future outputs, without commands, events, copies,
    /// guards or ordering edges. The existing resident carrier stays the owner.
    #[expect(
        clippy::too_many_arguments,
        reason = "original material, replay row, reader and resident owner remain distinct"
    )]
    pub fn prepare_source_content<B: AsRef<[u8]>>(
        &mut self,
        lease: &SemanticPublishedLease,
        resident: &SemanticPreparedResidentModelMemory,
        material: &SemanticReplayMaterial,
        ordinal: u64,
        row: &SemanticTrainingViewRow<B>,
        consumer_stream: u64,
        target: &SemanticTransitionSession,
        target_step: &SemanticPreparedStep,
        geometry: &SemanticPreparedSourceContentGeometry,
    ) -> Result<SemanticPreparedSourceContent, SemanticTransitionError> {
        self.checked_reader(lease)?;
        self.checked_step(lease)?;
        dlpack_consumer_stream(consumer_stream)?;
        if target.prepared_transition_kind(target_step)? != SemanticTransitionKind::Update {
            return Err(publication_input_error(
                "SOURCE recipe requires its original structural target Update",
            ));
        }
        let target_admission = target
            .prepared_segment
            .as_ref()
            .and_then(|build| build.program_admission.as_ref())
            .ok_or(SemanticTransitionError::NotBound)?;
        let target_instruction =
            Arc::clone(&target.instruction_admission(target_admission)?.instruction);
        let target_index = target
            .prepared_segment
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .tokens
            .iter()
            .position(|&token| token == target_step.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let declared = &geometry.owner;
        if declared.kind != SemanticSourceContentGeometryKind::Historical
            || declared.bank.is_some()
            || declared.step_index != target_index
            || declared.ordinal != ordinal
            || !Arc::ptr_eq(&declared.instruction, &target_instruction)
            || !Arc::ptr_eq(&declared.seed.material.owner, &material.owner)
            || target
                .graph
                .cold_work_allowance()
                .as_ref()
                .is_none_or(|original| !Arc::ptr_eq(original, &declared.allowance))
        {
            return Err(publication_input_error(
                "SOURCE recipe changed its original instruction, member or geometry owner",
            ));
        }
        let occurrence_owner = declared
            .occurrences
            .lock()
            .map_err(|_| publication_input_error("original content geometry custody is poisoned"))?
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| {
                publication_input_error(
                    "SOURCE recipe requires its original frozen content declarations",
                )
            })?;
        let mut recipe_binding = declared
            .recipe
            .lock()
            .map_err(|_| publication_input_error("original content recipe custody is poisoned"))?;
        let mut reader_binding = declared
            .reader
            .lock()
            .map_err(|_| publication_input_error("original content reader custody is poisoned"))?;
        if recipe_binding.is_some() || reader_binding.is_some() {
            return Err(publication_input_error(
                "SOURCE geometry binds its physical producer only once",
            ));
        }
        if material.transition_kind() != SemanticTransitionKind::Proposal {
            return Err(publication_input_error(
                "SOURCE content requires its original historical proposal",
            ));
        }
        let origin = crate::semantic_training_view::origin_record(material.training_view_origin()?);
        if row.origin.map(crate::semantic_training_view::origin_record) != Some(origin) {
            return Err(publication_input_error(
                "SOURCE content changed its original replay row",
            ));
        }
        if declared.member.origin != origin {
            return Err(publication_input_error(
                "SOURCE recipe changed its native frozen member origin",
            ));
        }
        let (material_identity, _) = self.verify_replay_source_material(lease, material)?;
        let allowance = self.cold_native_allowance(lease.token)?.ok_or_else(|| {
            publication_input_error("SOURCE content requires its original shared Prefix allowance")
        })?;
        if !Arc::ptr_eq(&allowance, &declared.allowance) {
            return Err(publication_input_error(
                "SOURCE physical child changed its original report allowance",
            ));
        }
        allowance
            .lock()
            .map_err(|_| publication_input_error("original SOURCE allowance is poisoned"))?
            .require_source_preparation()?;
        let resident = resident.source_content_owner(self, lease, consumer_stream)?;
        if self.steps[&lease.token].source_content.is_some() {
            return Err(publication_input_error(
                "SOURCE preparation cannot replace its original content recipe",
            ));
        }
        let storage = Arc::clone(
            self.publication
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        );
        let inputs = if let Some(inputs) = &self.steps[&lease.token].inputs {
            if inputs
                .original_preparation
                .lock()
                .map_err(|_| publication_input_error("original input preparation is poisoned"))?
                .is_some()
            {
                return Err(publication_input_error(
                    "SOURCE recipe must precede its original input preparation",
                ));
            }
            Arc::clone(inputs)
        } else {
            let inputs = Arc::new(PreparedStepInputs::allocate(
                &self.provider,
                Arc::clone(&storage),
                self.readers[&lease.token].device.view(),
                &lease.directory,
            )?);
            self.steps
                .get_mut(&lease.token)
                .expect("validated SOURCE step")
                .inputs = Some(Arc::clone(&inputs));
            inputs
        };
        declared
            .publication
            .require_original_allocation(&storage, &inputs, &lease.directory)?;
        let feedback = FeedbackBuffers::allocate(
            &self.provider,
            self.feedback_schema()?,
            storage.contract_value.feedback_capacity,
            self.descriptor().arena[4],
        )?;
        let feedback_ceiling =
            native_work_bound::feedback_native_work_ceiling(self, lease, &feedback)?;
        let policy_layouts = policy_numerical_producer_layouts(
            self.codebooks.input_cells,
            self.policy_layout()?.parameter_cells,
        )?;
        if feedback
            .content
            .tensors
            .iter()
            .map(|tensor| tensor.layout)
            .ne(declared.publication.feedback_layouts.iter().copied())
            || feedback_ceiling != declared.publication.feedback_ceiling
            || policy_layouts != declared.publication.policy_layouts
        {
            return Err(publication_input_error(
                "SOURCE physical feedback or policy producer changed its original geometry",
            ));
        }
        let continuation_capacity = storage
            .contract_value
            .feedback_capacity
            .checked_add(32)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let continuation_blocks = model_physical_parameter_roster(
            &storage.model_contract_record,
            storage.contract_value.model_contract_layout,
        )?
        .map(|parameters| parameters.len());
        let step = self
            .steps
            .get_mut(&lease.token)
            .expect("validated SOURCE step");
        let feedback_index = step.feedback.len();
        let owner_metadata_bytes = lease
            .directory
            .len()
            .checked_mul(size_of::<PublicationRange>())
            .and_then(|bytes| {
                feedback
                    .content
                    .tensors
                    .len()
                    .checked_mul(size_of::<PreparedSemanticTensor>())
                    .and_then(|feedback| bytes.checked_add(feedback))
            })
            .and_then(|bytes| {
                bytes.checked_add(size_of::<PreparedSourceContentOwner>() + 2 * size_of::<usize>())
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let owner_metadata = Arc::new(
            declared.seed.proof.owner.budget.reserve_auxiliary(
                u64::try_from(owner_metadata_bytes)
                    .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            )?,
        );
        let feedback_geometry = feedback.content.tensors.clone();
        step.feedback.push(feedback);
        let owner = Arc::new(PreparedSourceContentOwner {
            geometry: Arc::clone(declared),
            issuer: Arc::clone(&self.publication_issuer),
            token: lease.token,
            identity: lease.identity,
            ordinal,
            material: material_identity,
            origin,
            target: target_step.clone(),
            target_instruction,
            allowance,
            resident,
            storage: Arc::downgrade(&storage),
            inputs: Arc::downgrade(&inputs),
            feedback: feedback_index,
            directory: lease.directory.clone(),
            feedback_geometry: Mutex::new(Some(feedback_geometry)),
            feedback_ceiling,
            policy_layouts,
            continuation_capacity,
            continuation_blocks,
            activation: Mutex::new(None),
            terminal: Mutex::new(None),
            consumer_stream,
            ports_issued: std::sync::atomic::AtomicBool::new(false),
            ports: Mutex::new(None),
            _metadata: owner_metadata,
            destinations: Mutex::new(SourceDestinations {
                banks: std::array::from_fn(|_| None),
                sealed: false,
            }),
            occurrences: Mutex::new(Some(occurrence_owner)),
        });
        *recipe_binding = Some(Arc::downgrade(&owner));
        *reader_binding = Some((Arc::clone(&owner.issuer), owner.token));
        step.source_content = Some(Arc::clone(&owner));
        Ok(SemanticPreparedSourceContent { owner })
    }

    pub(super) fn original_source_content(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticPreparedSourceContent,
    ) -> Result<Arc<PreparedSourceContentOwner>, SemanticTransitionError> {
        self.checked_reader(lease)?;
        self.original_source_content_binding(lease, handle)
    }

    fn original_source_content_binding(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticPreparedSourceContent,
    ) -> Result<Arc<PreparedSourceContentOwner>, SemanticTransitionError> {
        let original = &handle.owner;
        if !Arc::ptr_eq(&original.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&original.issuer, &lease.issuer)
            || original.token != lease.token
            || original.identity != lease.identity
            || self
                .publication
                .as_ref()
                .is_none_or(|storage| !original.storage.ptr_eq(&Arc::downgrade(storage)))
            || self.steps.get(&lease.token).is_none_or(|step| {
                step.source_content
                    .as_ref()
                    .is_none_or(|owner| !Arc::ptr_eq(owner, original))
                    || step
                        .inputs
                        .as_ref()
                        .is_none_or(|inputs| !original.inputs.ptr_eq(&Arc::downgrade(inputs)))
                    || original.feedback >= step.feedback.len()
            })
        {
            return Err(publication_input_error(
                "SOURCE content changed its original reader or physical owners",
            ));
        }
        Ok(Arc::clone(original))
    }

    pub fn require_prepared_source_content(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticPreparedSourceContent,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self.original_source_content(lease, handle)?;
        if owner.resident.upgrade().is_none() {
            return Err(publication_input_error(
                "SOURCE content lost its original resident carrier",
            ));
        }
        Ok(())
    }

    pub(super) fn retain_source_content_destination(
        &self,
        lease: &SemanticPublishedLease,
        preparation: Arc<PreparedImportedEditDelivery>,
        instruction: Arc<[u8]>,
        updates: Vec<u64>,
        members: Vec<crate::semantic_training_view::FrozenPolicyGroupMember>,
        all_members: Vec<(
            u64,
            Vec<crate::semantic_training_view::FrozenPolicyGroupMember>,
        )>,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self
            .steps
            .get(&lease.token)
            .and_then(|step| step.source_content.as_ref())
            .ok_or_else(|| {
                publication_input_error("future Edit lost its original SOURCE recipe")
            })?;
        if preparation.bank > 1
            || !Arc::ptr_eq(&owner.target.issuer, &preparation.issuer)
            || !Arc::ptr_eq(&owner.target.scope, &preparation.scope)
            || owner.target.token != preparation.step
            || !Arc::ptr_eq(&owner.target_instruction, &instruction)
            || !Arc::ptr_eq(&owner.issuer, &preparation.source_issuer)
            || owner.token != preparation.source_token
            || owner.identity != preparation.source_identity
            || owner.ordinal != preparation.member_ordinal
            || owner.material != preparation.material
            || owner.origin != preparation.origin
            || !updates.contains(&preparation.step)
            || !members
                .iter()
                .any(|member| member.ordinal == owner.ordinal && member.origin == owner.origin)
            || all_members
                .iter()
                .map(|(step, _)| *step)
                .collect::<Vec<_>>()
                != updates
        {
            return Err(publication_input_error(
                "future Edit changed its original SOURCE recipe",
            ));
        }
        let mut destinations = owner.destinations.lock().map_err(|_| {
            publication_input_error("original SOURCE destination roster is poisoned")
        })?;
        let bank = preparation.bank;
        if destinations.sealed || destinations.banks[bank].is_some() {
            return Err(publication_input_error(
                "original SOURCE destination was already retained",
            ));
        }
        if let Some(other) = &destinations.banks[1 - bank] {
            if !Arc::ptr_eq(&other.issuer, &preparation.issuer)
                || !Arc::ptr_eq(&other.scope, &preparation.scope)
                || other.step != preparation.step
                || !Arc::ptr_eq(&other.instruction, &instruction)
                || other.updates != updates
                || other.members.len() != members.len()
                || other.members.iter().zip(&members).any(|(left, right)| {
                    left.ordinal != right.ordinal
                        || left.origin != right.origin
                        || left.identity != right.identity
                        || left.source_identity != right.source_identity
                        || left.bytes_identity != right.bytes_identity
                        || left.content_identity != right.content_identity
                })
            {
                return Err(publication_input_error(
                    "SOURCE banks changed their original target or frozen roster",
                ));
            }
        }
        destinations.banks[bank] = Some(SourceDestination {
            preparation: Arc::downgrade(&preparation),
            issuer: Arc::clone(&preparation.issuer),
            scope: Arc::clone(&preparation.scope),
            step: preparation.step,
            instruction,
            updates,
            members,
            all_members,
        });
        Ok(())
    }

    /// Seal only the complete structural roster; no numerical witness is issued.
    pub fn seal_prepared_source_content(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticPreparedSourceContent,
    ) -> Result<(), SemanticTransitionError> {
        let owner = self.original_source_content(lease, handle)?;
        self.require_prepared_source_content(lease, handle)?;
        let mut destinations = owner.destinations.lock().map_err(|_| {
            publication_input_error("original SOURCE destination roster is poisoned")
        })?;
        if !owner.ports_issued.load(Ordering::Acquire)
            || destinations.banks.iter().any(|bank| {
                bank.as_ref()
                    .is_none_or(|bank| bank.preparation.upgrade().is_none())
            })
        {
            return Err(publication_input_error(
                "SOURCE preparation lacks its original ports and both TARGET banks",
            ));
        }
        destinations.sealed = true;
        Ok(())
    }

    /// Issue each original physical alias once, with no producer or stream work.
    pub fn prepared_source_content_ports(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticPreparedSourceContent,
    ) -> Result<SemanticPreparedSourceContentPorts, SemanticTransitionError> {
        let original = Arc::clone(&handle.owner);
        self.original_source_content(lease, handle)?;
        let mut ports = original
            .ports
            .lock()
            .map_err(|_| publication_input_error("original SOURCE port custody is poisoned"))?;
        if original.ports_issued.load(Ordering::Acquire)
            || matches!(&*ports, Some(SourceContentPortsState::Issued { .. }))
        {
            return Err(publication_input_error(
                "original SOURCE ports were already issued",
            ));
        }
        if ports.is_none() {
            // All fallible geometry resolution precedes the first alias. The
            // installed plan owns each shape, view and completed capsule once.
            *ports = Some(SourceContentPortsState::Pending(
                self.prepare_source_content_ports(&original, lease)?,
            ));
        }
        let step = self
            .steps
            .get_mut(&lease.token)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        step.consumer_streams.insert(original.consumer_stream);
        let Some(SourceContentPortsState::Pending(plan)) = ports.as_mut() else {
            return Err(SemanticTransitionError::ObservationMismatch);
        };
        while plan.cursor < plan.exports.len() {
            let export = &mut plan.exports[plan.cursor];
            let retained_owner: Arc<dyn Send + Sync> = plan.reservation.clone();
            export.tensor = Some(export_owned_allocation(
                export.view.take().expect("original unissued SOURCE view"),
                std::mem::take(&mut export.shape),
                std::mem::take(&mut export.strides),
                export.dtype,
                self.provider.device().ordinal() as i32,
                Arc::clone(&plan.aliases),
                Some(Arc::clone(&plan.storage)),
                None,
                Some(retained_owner),
            ));
            plan.cursor += 1;
        }
        let Some(SourceContentPortsState::Pending(mut plan)) = ports.take() else {
            unreachable!("original SOURCE emission plan retained until handoff");
        };
        let mut tensors = plan.exports.into_iter().map(|mut export| {
            export
                .tensor
                .take()
                .expect("completed original SOURCE alias")
        });
        for cache in plan.cache_geometry {
            plan.caches.push(SemanticPreparedSourceCache {
                layout: cache.layout,
                logical_begin: cache.logical_begin,
                logical_end: cache.logical_end,
                tensor: tensors.next().expect("original SOURCE cache view"),
                allocation: cache.allocation,
                allocation_tensor: tensors.next().expect("original SOURCE cache backing"),
            });
        }
        let prefix = tensors.next().expect("original SOURCE prefix");
        let identity = tensors.next().expect("original SOURCE prefix identity");
        let extent = tensors.next().expect("original SOURCE prefix extent");
        let ring_head = tensors.next().expect("original SOURCE ring head");
        let source = tensors.next().expect("original SOURCE source slots");
        let terminal_state = tensors.next().expect("original SOURCE terminal state");
        let header = plan.header;
        let contract = plan.contract;
        let result = SemanticPreparedSourceContentPorts {
            model_context: SemanticModelContext {
                parent: lease.identity,
                descriptor_digest: header.descriptor_digest,
                numerical_state: (header.model_geometry_digest, header.model_numerical_digest),
                semantic_root: (
                    header.semantic_owner,
                    header.semantic_slot,
                    header.semantic_generation,
                    header.semantic_digest,
                    header.semantic_extents,
                ),
                prefix: (identity, extent),
                ring_head,
                source,
                generations: (
                    header.model_generation,
                    header.neural_bank,
                    header.neural_generation,
                    header.cache_generation,
                    header.authority_generation,
                ),
                topology_identity: contract.topology_identity,
                table_identity: contract.table_identity,
                capacities: (
                    contract.prefix_capacity,
                    contract.feedback_capacity,
                    contract.max_position,
                ),
            },
            prefix: (lease.identity, prefix),
            terminal_state,
            terminal_state_allocation: plan.terminal_state_allocation,
            feedback: SemanticFeedback {
                parent: lease.identity,
                schema: plan.schema,
                features: tensors.next().expect("six original feedback ports"),
                device_validity: tensors.next().expect("six original feedback ports"),
                query_receipts: tensors.next().expect("six original feedback ports"),
                original_statements: tensors.next().expect("six original feedback ports"),
                support_offsets: tensors.next().expect("six original feedback ports"),
                supports: tensors.next().expect("six original feedback ports"),
            },
            caches: plan.caches,
        };
        *ports = Some(SourceContentPortsState::Issued {
            _reservation: plan.reservation,
        });
        original.ports_issued.store(true, Ordering::Release);
        Ok(result)
    }

    fn prepare_source_content_ports(
        &self,
        original: &PreparedSourceContentOwner,
        lease: &SemanticPublishedLease,
    ) -> Result<SourceContentPortsPlan, SemanticTransitionError> {
        let header = lease.model_context_header()?;
        let storage = original
            .storage
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let inputs = original
            .inputs
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let contract = storage.contract_value;
        let range = |role, index| {
            lease
                .directory
                .iter()
                .find(|range| range.role == role && range.index == index)
                .copied()
                .ok_or(SemanticTransitionError::ObservationMismatch)
        };
        let prefix_range = range(SemanticStateRole::PrefixSource as u64, 0)?;
        if prefix_range.logical_end != header.prefix_extent {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let prefix_identity_range = range(SemanticStateRole::PrefixIdentity as u64, 0)?;
        if prefix_identity_range.length_bytes != 32 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let prefix_layout = committed_prefix_layout(&prefix_range, contract.prefix_capacity)?;
        let buffers = self
            .steps
            .get(&lease.token)
            .and_then(|step| step.feedback.get(original.feedback))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if buffers.content.tensors.len() != 6 {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let cache_count = lease
            .directory
            .iter()
            .filter(|range| matches!(range.role, 4..=13))
            .count();
        let export_count = cache_count
            .checked_mul(2)
            .and_then(|count| count.checked_add(12))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut rank_count = (prefix_layout.rank as usize)
            .checked_add(1)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        for field in [
            PublicationBankField::PrefixExtent,
            PublicationBankField::RingHead,
            PublicationBankField::Source,
            PublicationBankField::Terminal,
        ] {
            rank_count = rank_count
                .checked_add(field.geometry().2)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        for range in lease
            .directory
            .iter()
            .filter(|range| matches!(range.role, 4..=13))
        {
            let layout = storage
                .layouts
                .get(&(range.role, range.index))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            rank_count = rank_count
                .checked_add(layout.rank as usize)
                .and_then(|count| count.checked_add(1))
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        for tensor in &buffers.content.tensors {
            rank_count = rank_count
                .checked_add(tensor.layout.rank as usize)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let metadata_bytes = cache_count
            .checked_mul(
                size_of::<SourceContentCachePort>() + size_of::<SemanticPreparedSourceCache>(),
            )
            .and_then(|bytes| {
                export_count
                    .checked_mul(
                        size_of::<SourceContentPortExport>() + size_of::<PublishedTensorExport>(),
                    )
                    .and_then(|exports| bytes.checked_add(exports))
            })
            .and_then(|bytes| {
                rank_count
                    .checked_mul(2 * size_of::<i64>())
                    .and_then(|ranks| bytes.checked_add(ranks))
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    size_of::<SemanticReplayMaterialReservation>() + 2 * size_of::<usize>(),
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    OwnedPublicationMaterial::tree_node_bound(size_of::<u64>(), 0).ok()?,
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = Arc::new(
            original
                .geometry
                .seed
                .proof
                .owner
                .budget
                .reserve_auxiliary(
                    u64::try_from(metadata_bytes)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )?,
        );
        let mut cache_geometry = Vec::with_capacity(cache_count);
        let mut exports = Vec::with_capacity(export_count);
        for &range in lease
            .directory
            .iter()
            .filter(|range| matches!(range.role, 4..=13))
        {
            let layout = *storage
                .layouts
                .get(&(range.role, range.index))
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let view = self.source_content_range_view(&original, lease, range, layout)?;
            let allocation = view.allocation_provenance().ok_or_else(|| {
                publication_input_error("original SOURCE cache has no native backing owner")
            })?;
            let backing = view
                .allocation_view()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            let bytes = i64::try_from(backing.len())
                .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
            exports.push(SourceContentPortExport::tensor(view, layout)?);
            exports.push(SourceContentPortExport::raw(
                backing,
                vec![bytes],
                vec![1],
                (1, 8),
            ));
            cache_geometry.push(SourceContentCachePort {
                layout,
                logical_begin: range.logical_begin,
                logical_end: range.logical_end,
                allocation,
            });
        }
        let prefix_view =
            self.source_content_range_view(&original, lease, prefix_range, prefix_layout)?;
        exports.push(SourceContentPortExport::tensor(prefix_view, prefix_layout)?);
        let identity_view = inputs.views[(header.publication_word & 1) as usize]
            .get(&(prefix_identity_range.role, 0))
            .ok_or(SemanticTransitionError::ObservationMismatch)?
            .slice(0..32);
        exports.push(SourceContentPortExport::raw(
            identity_view,
            vec![32],
            vec![1],
            (1, 8),
        ));
        for field in [
            PublicationBankField::PrefixExtent,
            PublicationBankField::RingHead,
            PublicationBankField::Source,
            PublicationBankField::Terminal,
        ] {
            let (_, shape, strides) = field.layout();
            exports.push(SourceContentPortExport::raw(
                inputs.field_view(field)?,
                shape,
                strides,
                (1, 64),
            ));
        }
        let terminal_state_allocation = inputs
            .field_view(PublicationBankField::Terminal)?
            .allocation_provenance()
            .ok_or_else(|| {
                publication_input_error("original SOURCE terminal field lost its native allocation")
            })?;
        let schema = buffers.schema.clone();
        for tensor in &buffers.content.tensors {
            exports.push(SourceContentPortExport::tensor(
                tensor
                    .source
                    .as_ref()
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .clone(),
                tensor.layout,
            )?);
        }
        Ok(SourceContentPortsPlan {
            header,
            contract,
            schema,
            terminal_state_allocation,
            cache_geometry,
            exports,
            caches: Vec::with_capacity(cache_count),
            cursor: 0,
            storage,
            aliases: Arc::clone(&self.steps[&lease.token].aliases),
            reservation,
        })
    }

    fn source_content_range_view(
        &self,
        original: &PreparedSourceContentOwner,
        lease: &SemanticPublishedLease,
        range: PublicationRange,
        layout: SemanticTensorLayout,
    ) -> Result<DeviceMemoryView<u8>, SemanticTransitionError> {
        let inputs = original
            .inputs
            .upgrade()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let view = inputs.views[(lease.header.publication_word & 1) as usize]
            .get(&(range.role, range.index))
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        Ok(view.slice(publication_export_span(
            &PublicationRange {
                offset_bytes: 0,
                ..range
            },
            &layout,
            view.len(),
        )?))
    }
}
