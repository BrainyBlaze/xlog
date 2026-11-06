"""Type stubs for pyxlog._native (Rust/PyO3 bindings).

All tensor-valued attributes (``prob``, ``log_prob``, ``query_counts``, etc.)
are returned as DLPack capsule objects (PyCapsule with name ``"dltensor"``).
Consumers should call ``torch.from_dlpack(x)`` or the DLPack protocol on these
values.  The stubs represent them as ``Any`` because the capsule type is not
importable from Python.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Callable, Literal, NoReturn, Optional, Sequence, TypedDict, Union

_SealedReplayMaterialSource = tuple[
    tuple[int, int], tuple[tuple[str, int], ...],
    Callable[[str, int, SemanticMaterialWriter], SemanticMaterialBytes],
]
_ReplayMaterialSource = SemanticMaterialSource

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

__version__: str

class _ResourceDeviceCertificate(TypedDict):
    format: Literal["xlog.device-capacity/1"]
    observer_identity: str
    device_uuid: str
    total_physical_bytes: int
    process_id: int
    driver_version: str
    source: Literal["nvml-memory-v2"]
    whole_device: Literal[True]

def register_process_resource_observer(capsule: Any) -> None:
    """Pin the original ABI3 collector before native GPU allocation.

    The same capsule is idempotent; replacement, late attachment and an old ABI
    are errors. Native root lifetime evidence enters this one collector.
    """
    ...

def resource_observer_device_certificate(device_ordinal: int) -> _ResourceDeviceCertificate:
    """Project the original immutable whole-device NVML capacity certificate.

    The registered collector must precede CUDA. This cold actual-UUID lookup
    allocates no native GPU storage and is not an observation or admission flag.
    Compare U against every original operation/cleanup ceiling before model
    allocation. Signed native restore repeats that check without changing old
    certificates or expense.
    """
    ...

_Path = Union[str, PathLike[str]]

class NativeCausalModel:
    """Native decoder with canonical acquired definitions and atomic candidate training."""

    @staticmethod
    def from_checkpoint(
        directory: _Path, *, device: int, memory_budget_bytes: int, max_context: int
    ) -> NativeCausalModel: ...
    def save_snapshot(self, path: _Path, *, application_state: bytes) -> str: ...
    @staticmethod
    def from_snapshot(
        path: _Path, *, snapshot_identity: str, device: int, memory_budget_bytes: int,
    ) -> tuple[NativeCausalModel, bytes]: ...
    def forward(
        self, token_ids: Sequence[int], *, prediction_position: int,
        history_span: Optional[tuple[int, int]] = None,
    ) -> NativeLogits: ...
    def fork_trainable(self, *, down_projection_layer: int) -> NativeCausalModel: ...
    def fork_candidate(self) -> NativeCausalModel: ...
    def acquisition_language(
        self, *, insertion_layer: int = 17, language_version: int = 1,
        raw_token_ids: Optional[Sequence[int]] = None,
    ) -> str:
        """Exact native context/schema; v6 excludes impossible single-operation output plans."""
        ...
    def fork_acquired(
        self, definition_json: str, parameter_values: dict[str, Sequence[float]], *,
        insertion_layer: int = 17, train_down_projection: bool = True,
    ) -> NativeCausalModel: ...
    @property
    def adaptation_manifest(self) -> str: ...
    @property
    def function_definitions(self) -> str: ...
    def fit_binary(
        self, token_ids: Sequence[Sequence[int]], *, prediction_positions: Sequence[int],
        outcomes: Sequence[bool], negative_token_id: int, positive_token_id: int,
        history_spans: Optional[Sequence[tuple[int, int]]] = None,
    ) -> tuple[float, float, int]: ...
    def fit_tokens(
        self, token_ids: Sequence[Sequence[int]], *, prediction_positions: Sequence[int],
        target_ids: Sequence[int],
        history_spans: Optional[Sequence[tuple[int, int]]] = None,
    ) -> tuple[float, float, int]: ...
    @property
    def checkpoint_identity(self) -> str: ...
    @property
    def optimizer_steps(self) -> dict[str, int]: ...
    @property
    def max_context(self) -> int: ...
    @property
    def vocabulary_size(self) -> int: ...

class NativeAcquisitionMask:
    """Intersect byte-token masks with the selected native acquisition contract."""

    def __init__(self, language_json: str, token_bytes: Sequence[bytes]) -> None: ...
    def filter(self, prefix: bytes, allowed_mask: bytes) -> bytes: ...
    def completion_state(self, prefix: bytes) -> str:
        """Version4/5/6 prefix facts as JSON; no token selection or final admission."""

class _LogitSelection(TypedDict):
    raw_argmax: int
    raw_score: float
    selected: int
    selected_score: float
    logits_sha256: str

class NativeLogits:
    """One complete FP32 vocabulary row retaining its native allocation."""

    def read(self, token_ids: Sequence[int]) -> list[float]:
        """Copy requested logits in caller order; requires host-io."""
        ...
    def select_greedy(self, allowed_mask: Optional[bytes] = None) -> _LogitSelection:
        """CUDA maxima with minimum-ID ties and full-row SHA256; requires host-io."""
        ...
    def select_categorical(
        self, allowed_mask: Optional[bytes] = None, *, seed: int, sequence: int, position: int
    ) -> _LogitSelection:
        """CUDA categorical draw at temperature one with a retained Philox address."""
        ...
    @property
    def prediction_position(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __dlpack__(self, stream: Optional[int] = None) -> Any: ...

_PublishedIdentity = tuple[bytes, int, bytes]
_NativeIdentityWords = tuple[int, int, int, int]
_MeasurementAvailability = Literal["not_executed", "incomplete", "obtained"]

class _UpdateCanaryMeasurement(TypedDict):
    evaluator_abi: int
    kind: int
    row_ordinal: int
    availability: _MeasurementAvailability
    reason: int
    measurement_bits: Optional[int]
    lower_bound_bits: int
    upper_bound_bits: int
    memory_used: int
    memory_limit: int
    work_used: int
    work_limit: int
    obligation_positions: tuple[int, ...]
    protected_member_offset: int
    protected_member_count: int
    protected_positions: tuple[int, ...]
    row_identity: _NativeIdentityWords
    row_content_identity: _NativeIdentityWords
    task_identity: _NativeIdentityWords
    identity: _NativeIdentityWords

class _UpdateForwardReceiptMaterial(TypedDict):
    role: int
    generation: int
    work_sequence: int
    scalar_type: int
    dimensions: tuple[int, int, int]
    strides_bytes: tuple[int, int, int]
    model_geometry_digest: bytes
    logits_digest: bytes
    identity: bytes
    canonical_bytes: bytes

class _UpdateForwardReceipt(TypedDict):
    role: int
    availability: _MeasurementAvailability
    material: Optional[_UpdateForwardReceiptMaterial]

class _UpdateMeasurements(TypedDict):
    schema: Literal[1]
    predecessor: _PublishedIdentity
    successor: Optional[_PublishedIdentity]
    bank: int
    model_generation: int
    model_geometry_digest: bytes
    transition_status: int
    selection: dict[str, Any]
    canaries: tuple[_UpdateCanaryMeasurement, ...]
    forward_receipts: tuple[_UpdateForwardReceipt, _UpdateForwardReceipt]
    canary_refusal: Optional[dict[str, Any]]

class SemanticCompletedMaterial:
    """Native-owned original material; construction is not public."""

    @property
    def identity(self) -> bytes: ...
    @property
    def bytes(self) -> bytes: ...
    @property
    def bytes_sha256(self) -> bytes: ...

class SemanticCompletedActionProjection:
    """Immutable cold projection of one completed original Proposal."""

    @property
    def owner(self) -> SemanticCompletedMaterial: ...
    @property
    def initial_prefill(self) -> Optional[SemanticCompletedMaterial]: ...
    @property
    def pre_action_checkpoint(self) -> Optional[SemanticCompletedMaterial]: ...

class SemanticCompletedExecutionObservation:
    """Native-issued immutable joined result; no constructor or CUDA ownership."""

    @property
    def device_work(self) -> int: ...
    @property
    def model_work(self) -> int: ...
    @property
    def native_work(self) -> int: ...
    @property
    def model_calls(self) -> int: ...

class SemanticPrivateExecutionContinuation:
    """Native-issued custody of one original private execution resolver.

    Issued by bind_private_execution_continuation before segment effects.
    Only the retained native phase may invoke the original resolver.
    """

class SemanticPreparedActorRefresh:
    """Native-issued CURRENT actor member with two original embedded children.

    Create it only through the original Update's prepare_actor_refresh method.
    Both bank producers borrow the sole outer memory_scope and never submit a
    graph independently. Retain this owner through native final-use retirement.
    """
    def source_task(self, *, refresh_snapshot: Any) -> tuple[Any, ...]: ...
    def prepare_bank(self, bank: int, task_use: SemanticTransitionTaskUse, producer: Any) -> Any: ...
    def retirement_observation(
        self, bank: int, task_use: SemanticTransitionTaskUse,
        prepared: Any, steps: tuple[SemanticPreparedStep, ...],
    ) -> Literal["observed", "cancelled", "initializer_refused"]:
        """Cached native disposition only inside the original finish_segment.

        Exact original child owners are required; no outcomes or work are issued.
        """
        ...
    def require_current_target_basis(
        self, template_task_use: SemanticTransitionTaskUse,
        bank: int, child_step: SemanticPreparedStep,
    ) -> None:
        """Authenticate the exact target and issued child's CURRENT model owners.

        This retained metadata check does not read or execute device storage.
        """
        ...
    def group_update_vjp(self, update_step: SemanticPreparedStep, bank: int, *, consumer_stream: int) -> tuple[Any, ...]: ...
    def group_update_vjp_layouts(
        self, update_step: SemanticPreparedStep, bank: int, *, consumer_stream: int,
    ) -> tuple[_PolicyOutputLayout, ...]:
        """Nine original adjoint allocation layouts, without recording or capsules."""
        ...

class SemanticPreparedStep:
    """Session-issued original step; construction is not public."""

    def retirement_consumer_streams(self) -> tuple[int, ...]:
        """Same native joined final-use roster, only in original finish_segment."""
        ...
    def prepare_actor_refresh(
        self, member_ordinal: int, *, refresh_snapshot: Any,
        content_geometry: tuple[
            SemanticPreparedSourceContentGeometry, SemanticPreparedSourceContentGeometry
        ],
    ) -> SemanticPreparedActorRefresh: ...

    def record_saved_tensor_work(self, tensor_owner: SemanticTensorOwner) -> None: ...

    def model_work_buffer_port(self, *, consumer_stream: int) -> SemanticModelWorkBufferPort:
        """Original step counter shared by both banks; export only after finalization."""
        ...

    def register_gradient_delivery(
        self, bank: int, phase: _SemanticTensorInput,
        accumulation: _SemanticTensorInput,
        leaves: Sequence[tuple[
            Optional[Any], _SemanticTensorInput, Optional[_SemanticTensorInput],
            _SemanticTensorInput, _SemanticTensorInput,
        ]], *, consumer_stream: int,
        group_member: Optional[Union[SemanticRetainedReplayMember, SemanticPreparedTaskImport]] = None,
        actor_refresh: Optional[SemanticPreparedActorRefresh] = None,
        edit_only: bool = False,
    ) -> SemanticGradientDelivery:
        """Freeze one bank's original physical gradient delivery and contribution."""
        ...

    @property
    def requested_transition(self) -> Literal["proposal", "recompute", "update"]: ...

    def prepare_cold_model_work(
        self, prepare_callback: Callable[[], Any], *,
        initialize_callback: Callable[[], None], consumer_streams: Sequence[int]
    ) -> Any:
        """Prepare metadata once and retain the mandatory original initializer."""
        ...

    def initialize_cold_model_work(self) -> None:
        """Run only that retained initializer, once after the complete plan freeze."""
        ...

    def group_critic_scalar_layout(
        self, bank: int, *, consumer_stream: int,
    ) -> _PolicyOutputLayout:
        """The original FP32[1] critic destination, without reduction."""
        ...

    @property
    def execution_measurements(self) -> SemanticCompletedExecutionObservation:
        """Joined step execution only, not preparation or a physical peak."""
        ...

    def record_model_invocation(self) -> None:
        """Record the CUDA marker at the actual forward's original capture site."""
        ...

    @property
    def update_measurements(self) -> Optional[_UpdateMeasurements]:
        """Original accepted/refused Update observations before retirement."""
        ...

    def completed_model_binding(
        self, *, consumer_streams: Sequence[int]
    ) -> tuple[_PublishedIdentity, Optional[_PublishedIdentity], int, bytes, Any]: ...

    def completed_feedback_records(
        self, *, consumer_streams: Sequence[int],
    ) -> tuple[
        _PublishedIdentity, Optional[_PublishedIdentity],
        tuple[tuple[int, int, int, int, int, int, bytes], ...],
    ]: ...

    def save_checkpoint(
        self, *, consumer_streams: Sequence[int],
        snapshot_model_state: Callable[[SemanticMaterialWriter], SemanticMaterialBytes],
        model_material_capacity: int, model_material_working_bytes: int,
        original_writer: Optional[SemanticMaterialWriter] = None,
    ) -> SemanticMaterialBytes:
        """Seal this late Proposal's full pre-action custody before retirement."""
        ...

    def completed_action_projection(
        self, *, consumer_streams: Sequence[int]
    ) -> SemanticCompletedActionProjection: ...

    @property
    def control_measurements(self) -> dict[str, Any] | None: ...

_SemanticTensorLayout = tuple[
    int, int, int, int, int, int, tuple[int, ...], tuple[int, ...]
]
_SemanticTensorInput = tuple[_SemanticTensorLayout, int, int, Any, Optional[Any]]

class SemanticGradientDelivery:
    """Native-issued delivery retained through the bank's sole backward."""

    def activate(self) -> None: ...
    def record_reset(self) -> None: ...
    def finish_recording(self) -> None: ...

class SemanticRetainedReplayMember:
    """Original historical invocation retained through both target banks."""

    def group_update_vjp(
        self, update_step: SemanticPreparedStep, bank: int, *,
        consumer_stream: int, edit_only: bool = False,
    ) -> tuple[Any, ...]: ...
    def group_update_vjp_layouts(
        self, update_step: SemanticPreparedStep, bank: int, *,
        consumer_stream: int, edit_only: bool = False,
    ) -> tuple[_PolicyOutputLayout, ...]: ...
    def confirm_published_target(self) -> None: ...
    def confirm_unpublished_target(self, update_step: SemanticPreparedStep) -> None: ...
    def finish(self) -> None: ...
    def finish_without_published_update(self, update_step: SemanticPreparedStep) -> None: ...
    def finish_cancelled_private_segment(self) -> None: ...

_SourceModelContractMetadata = tuple[bytes, tuple[int, int, int, int, int]]
_SourceModelMemoryGeometry = tuple[
    tuple[int, ...], tuple[tuple[int, int, int], ...],
    tuple[tuple[
        int, int, int, int,
        tuple[int, int, int, int, tuple[int, ...], tuple[int, ...]],
    ], ...],
]

_SourceContentTensorDescriptor = tuple[
    _SemanticTensorLayout, int, int, int, Optional[int], int, int
]
_SourceContentAllocationDescriptor = tuple[
    int, int, Optional[Union[NativeTensorAllocation, SemanticSourceGeometryAllocation]]
]
_SourceContentAllocationLayout = tuple[int, int]

class SemanticSourceGeometryAllocation:
    """Original opaque allocation identity before a physical native export."""
    def span_bytes(self) -> int: ...
    def same_allocation(self, other: SemanticSourceGeometryAllocation) -> bool: ...

_SourceGeometryTensor = tuple[
    _SemanticTensorLayout, int, int, SemanticSourceGeometryAllocation, int
]
_SourceGeometryModelMemory = tuple[
    tuple[tuple[SemanticSourceGeometryAllocation, int], ...],
    tuple[tuple[int, int, int], ...],
    tuple[tuple[_SemanticTensorLayout, int, int], ...],
]

class SemanticModelInputGenerationPort:
    """Original per-Update input backing declaration shared by TARGET/CURRENT.

    No public constructor. Retain this owner with all projected metadata.
    Late native allocation binds the same declaration to the preceding Update
    output, or the admitted parent for the first Update. Shape never proves it.
    """
    def same_port(self, other: SemanticModelInputGenerationPort) -> bool: ...
    def step_index(self) -> int: ...
    def preceding_update(self) -> Optional[int]: ...
    def model_memory_layouts(self) -> _SourceGeometryModelMemory: ...
    def allocation_matches(
        self, declared: SemanticSourceGeometryAllocation, actual: NativeTensorAllocation,
    ) -> bool:
        """Unbound future is an error; a valid foreign physical owner is unequal."""
        ...

class SemanticPreparedSourceContentGeometry:
    """Original admitted SOURCE geometry before factories or tensor exports.

    CURRENT carries structural proof only. Its numerical identity belongs to
    the original Actor/bank; historical restore context cannot substitute it.
    """
    def kind(self) -> Literal["historical", "current"]: ...
    def step_index(self) -> int: ...
    def member_ordinal(self) -> int: ...
    def bank(self) -> Optional[int]: ...
    def restore_material_mode(self) -> Literal["checkpoint-model"]:
        """Historical only; CURRENT requires its genuine Actor materialization."""
        ...
    def source_task(self) -> tuple[Any, Any]: ...
    def task_training_domain(self) -> Optional[tuple[bytes, Any]]:
        """Same native domain identity and original child-task projection."""
        ...
    def task_objective_law(self) -> tuple[
        bytes, tuple[int, int], tuple[int, ...], tuple[bytes, int]
    ]: ...
    def learning_feedback_intervention(self) -> None:
        """Authenticated fresh-child baseline, checked again on original binding.

        Not a default for an absent phase or an ordinary restored phase owner.
        """
        ...
    def policy_layout(self) -> tuple[Any, ...]: ...
    def policy_support_layout(self) -> tuple[
        tuple[int, bytes], list[tuple[int, int]], int
    ]: ...
    def feedback_schema(self) -> tuple[
        int, bytes, bytes, int, int, int, tuple[tuple[str, int, int, bool], ...]
    ]: ...
    def prefix_capacity(self) -> int: ...
    def feedback_capacity(self) -> int: ...
    def schema_generation(self) -> bytes: ...
    def structural_capacities(self) -> tuple[int, int, int]: ...
    def position_table_identity(self) -> bytes: ...
    def model_input_generation_port(self) -> SemanticModelInputGenerationPort:
        """CURRENT only; same original per-Update port as its TARGET declaration."""
        ...
    def model_contract_material(self) -> SemanticMaterialBytes: ...
    def model_restore_context(self) -> tuple[
        _PublishedIdentity, int, tuple[int, int, int], bytes
    ]:
        """Independent original historical context; unavailable for CURRENT."""
        ...
    def model_memory_layouts(self) -> _SourceGeometryModelMemory: ...
    def input_tensor_layouts(
        self, *, prepared_kind: Optional[Literal["recompute", "proposal"]] = None,
        prepared_bank: Optional[Literal[0, 1]] = None,
    ) -> tuple[_SourceGeometryTensor, ...]: ...
    def feedback_tensor_layouts(
        self, *, prepared_kind: Optional[Literal["recompute", "proposal"]] = None,
        prepared_bank: Optional[Literal[0, 1]] = None,
    ) -> tuple[_SourceGeometryTensor, ...]: ...
    def policy_tensor_layouts(
        self, *, prepared_kind: Optional[Literal["recompute", "proposal"]] = None,
        prepared_bank: Optional[Literal[0, 1]] = None,
    ) -> tuple[_SemanticTensorLayout, ...]:
        """Typed external outputs only; no native allocation or version owner."""
        ...
    def policy_numerical_domain_layouts(
        self, *, prepared_kind: Optional[Literal["recompute", "proposal"]] = None,
        prepared_bank: Optional[Literal[0, 1]] = None,
    ) -> tuple[
        _SourceGeometryTensor, _SourceGeometryTensor, _SourceGeometryTensor
    ]:
        """Three genuine FP64 uniform allocations, not policy-logit snapshots.

        Historical omits both section keywords. CURRENT requires its original
        Proposal and bank; its two actual uniform buffer sets are distinct.
        """
        ...
    def continuation_capacity(self) -> int: ...
    def continuation_blocks(self) -> Optional[int]: ...

class SemanticPreparedSourceContent:
    """Original SOURCE structural ports, not numerical content authority.

    Created only by the original prepared import and retained with the same
    resident carrier. Getters return the original capsules without re-export.
    Retain this recipe while any returned structural metadata remains held,
    including consumed capsules; physical retirement does not release its
    projection receipt or grant later numerical access.
    """

    def model_context(self) -> tuple[Any, ...]: ...
    def prefix(self) -> tuple[_PublishedIdentity, Any]: ...
    def terminal_state(self) -> Any: ...
    def terminal_state_allocation(self) -> NativeTensorAllocation:
        """Original terminal view provenance, without exporting its full header."""
        ...
    def feedback(self) -> tuple[Any, ...]: ...
    def tensor(self, role: int, index: int) -> Any: ...
    def tensor_allocation(self, role: int, index: int) -> tuple[NativeTensorAllocation, Any]: ...
    def activate(self, *, consumer_stream: int) -> None:
        """Prepare original SOURCE contents under the same admitted recorder."""
        ...

class SemanticPreparedTaskImport:
    """Original SOURCE import before its sole numerical invocation.

    Created only by import_task with prepare_invocation. Metadata and future
    physical gradient bindings are not an Imported use or a policy outcome.
    """

    @property
    def task_use(self) -> SemanticTransitionTaskUse: ...
    @property
    def parent(self) -> SemanticPublishedParent: ...
    def prepare_source_content(
        self, *, consumer_stream: int,
        resident_memory: SemanticPreparedResidentModelMemory,
        content_geometry: SemanticPreparedSourceContentGeometry,
    ) -> SemanticPreparedSourceContent:
        """Reuse this original resident carrier; no second stage or effects."""
        ...
    def source_model_contract_metadata(self) -> _SourceModelContractMetadata: ...
    def source_model_memory_geometry(self) -> _SourceModelMemoryGeometry: ...
    def group_update_vjp_layouts(
        self, update_step: SemanticPreparedStep, bank: int, *,
        consumer_stream: int, edit_only: bool = False,
    ) -> tuple[_PolicyOutputLayout, ...]:
        """Future SOURCE requires its original registered edit_only=True delivery."""
        ...
    def group_update_vjp(
        self, update_step: SemanticPreparedStep, bank: int, *,
        consumer_stream: int, edit_only: bool = False,
    ) -> tuple[Any, ...]:
        """Requires genuine completion of this same original import."""
        ...
    def finish_cancelled_private_segment(self) -> None:
        """Acknowledge only its cached positive, unexecuted cancellation."""
        ...

class SemanticSegmentInstructionAdmission:
    """Native-issued original instruction custody; construction is not public."""

    def prepare(self, prepare_callback: Callable[[], Any]) -> Any: ...
    def parent_identity(self) -> _PublishedIdentity: ...
    def resolve_parent_handoff(self) -> None: ...
    @property
    def cancelled_checkpoint(self) -> Optional[SemanticMaterialBytes]:
        """Cached terminal capsule only after genuine cancellation retirement."""
        ...
    @property
    def preparation_result(self) -> Optional[tuple[
        int, int, int, int, int, tuple[int, int, int, int, int, int, int, int, int]
    ]]: ...
    @property
    def retirement_result(self) -> Optional[tuple[
        int, int, int, int, int, tuple[int, int, int, int, int, int, int, int, int]
    ]]: ...
    @property
    def budgets(self) -> tuple[tuple[int, int, int], ...]: ...

    @property
    def control_measurements(self) -> dict[str, Any] | None: ...

    @property
    def resource_interval(self) -> dict[str, Any] | None:
        """Original interval after native and external retirement; never finishes work."""
        ...

    @property
    def resource_retirement(self) -> SemanticSegmentResourceRetirement | None: ...

    def finish_retirement(self) -> dict[str, Any]:
        """After real cold/model teardown, finish only this original observer suffix."""
        ...

class SemanticSegmentRetirementPending(RuntimeError):
    """Continue the same native or observer retirement; completed callbacks cannot repeat."""

class SemanticSegmentResourceRetirement:
    """Original CPU-only interval owner; it retains no graph, model or producer."""

    @property
    def resource_interval(self) -> dict[str, Any] | None: ...

    def finish_retirement(self) -> dict[str, Any]:
        """Finish after the original native closes and external model owners release."""
        ...

class SemanticEvaluationInstructionAdmission:
    """Original public read-only Evaluation; retained across genuine Pending."""

    @property
    def cohort(self) -> SemanticEvaluationCohort:
        """The frozen cohort, available only in the preparation callback."""
        ...

    def execute(
        self, callback: Callable[[], SemanticCompletedModelEvaluation],
    ) -> SemanticCompletedModelEvaluation:
        """Continue this original callback and retire its actual joined expense."""
        ...

    @property
    def observation(self) -> dict[str, Any] | None:
        """Authentic observation, visible after complete native and observer retirement."""
        ...

    @property
    def cold_work_result(self) -> Optional[tuple[
        int, int, int, int, int, tuple[int, int, int, int, int, int, int, int, int]
    ]]: ...

    @property
    def resources_retired(self) -> bool: ...

class SemanticPrivateReplayChild:
    """One original selected replay child retained by its instruction owner."""

    def source_task(
        self, *, refresh_snapshot: Callable[[], Any]
    ) -> tuple[Any, Any]: ...

class SemanticColdModelWork:
    """Original registrar; numerical rights remain callback-scoped."""

    def restore_material_mode(self) -> Literal["phase-reset", "checkpoint-model"]:
        """Original decoded restoration mode, without invoking its factory."""
        ...
    def prepare_checkpoint_writer(
        self, *, region_ordinal: int, model_material_capacity: int,
        model_material_working_bytes: int,
        resident_read: SemanticResidentModelRead, codec_staging_bytes: int,
    ) -> SemanticMaterialWriter:
        """Original region's retained metadata-only writer, also for future parents.

        Pass this same object into the compiled snapshot and later callback.
        resident_read is the same already-bound opaque region occurrence;
        codec_staging_bytes is the original encoder's chunk extent, not a price.
        Inspection grants no storage allocation, numerical read or write.
        """
        ...
    def prepare_resident_model_restoration(
        self, memory: SemanticPreparedResidentModelMemory,
    ) -> SemanticPreparedResidentModelRestoration:
        """Same planned Snapshot0 producer and future native restore issuance."""
        ...
    def report_cursor(self) -> tuple[
        int, int, int, int, tuple[tuple[int, int], ...], tuple[int, ...], bool, bool,
    ]:
        """Native region bounds, declared cursor, physical slots, proven skips,
        skipped region ordinals, recording closure and actual report completion.

        Pending, Unknown and stopped live work cannot authorize continuation.
        The same original handle retains final native proof after Session release;
        a completed refusal may retain a partial declared prefix.
        """
        ...

    @property
    def plan_admitted(self) -> bool: ...
    @property
    def region_count(self) -> int: ...
    @property
    def consumer_stream(self) -> int: ...
    def evaluation_source_geometry(
        self,
    ) -> list[tuple[list[int], list[int], tuple[int, int]]]:
        """Read seventeen original port layouts without selection or device export."""
        ...
    def admit_plan(
        self,
        operations: tuple[tuple[int, tuple[int, ...]], ...],
        *,
        region_ends: tuple[int, ...] = (),
        source_content: tuple[
            Union[SemanticPreparedSourceContent, SemanticPreparedSourceContentGeometry], ...
        ] = (),
        resident_reads: tuple[tuple[int, SemanticResidentModelRead], ...] = (),
    ) -> tuple[int, int, int]:
        """Admit once before any original buffer/view export or numerical producer.

        Genuine source allocation/copy work and evaluation preparation remain
        part of the original native model-work ceiling.
        Historical geometry resolves its existing live recipe in original order;
        missing, expired and CURRENT bindings are rejected without another freeze.
        """
        ...
    def model_work_buffer_layout(self, *, consumer_stream: int) -> _NativeWorkBufferLayout:
        """Genuine finalized scratch geometry; unavailable before plan admission."""
        ...
    def model_work_buffer_port(self, *, consumer_stream: int) -> SemanticModelWorkBufferPort:
        """Original report counter owner without an early event extent or allocation."""
        ...
    def model_work_buffer(self, *, consumer_stream: int) -> Any: ...
    def begin(self) -> None: ...
    def end(self) -> None: ...
    def fail(self) -> None: ...
    def prepare_operation(
        self, kind: int, dimensions: tuple[int, ...], *, device_produced: bool = False,
    ) -> SemanticColdModelWorkOperation:
        """Hold the exact next original event before its effect, without charging it."""
        ...
    def prepare_operations(
        self, operations: tuple[tuple[int, tuple[int, ...], bool], ...],
    ) -> tuple[SemanticColdModelWorkOperation, ...]: ...
    def record_model_invocation(self) -> None: ...
    def record_model_work(self, kind: int, dimensions: tuple[int, ...]) -> None: ...
    def record_model_device_work(self, kind: int, upper_dimensions: tuple[int, ...]) -> int: ...

class SemanticColdModelWorkOperation:
    """Native-issued single-use effect admission, retained by its original recorder."""

    @property
    def slot(self) -> int: ...
    def enter(self) -> int: ...
    def complete(self) -> int: ...
    def unknown(self) -> None: ...
    def cancel_before_entry(self) -> None: ...

class SemanticEvaluationCohort:
    """Native-issued sealed original selection; no public constructor."""

class SemanticCompletedModelEvaluation:
    """Native-issued original completed result, without model/cohort/reader ownership."""

    @property
    def status(self) -> Literal[0, 1]: ...
    @property
    def loss_bits(self) -> tuple[int, int, int, int, int, int]: ...
    @property
    def model_work(self) -> int: ...
    @property
    def model_calls(self) -> int: ...
    @property
    def operation_count(self) -> int: ...
    @property
    def work_bound(self) -> int: ...
    @property
    def retained_allocation_bytes(self) -> int: ...

class SemanticModelEvaluationPending(RuntimeError):
    """Original submitted evaluation awaits resolution, never a second forward or finish.

    Retain its Runtime, objective, native evaluation and original output owners.
    Fatal CUDA quarantine and known incomplete work do not carry this signal.
    """

class SemanticModelEvaluation:
    """Single read-only invocation on an authentic published parent."""

    @property
    def cohort(self) -> SemanticEvaluationCohort: ...
    @property
    def consumer_stream(self) -> int: ...
    def training_view_layouts(self, *, consumer_stream: int) -> tuple[_TrainingViewLayout, ...]:
        """All seventeen original allocation-backed ports without preparation or export."""
        ...
    def model_work_buffer_layout(self, *, consumer_stream: int) -> _NativeWorkBufferLayout:
        """Actual numerical U64 scratch geometry, separate from cold-report scratch."""
        ...
    def model_work_buffer_port(self, *, consumer_stream: int) -> SemanticModelWorkBufferPort:
        """Distinct original capture counter, not the cold report counter."""
        ...
    def admit_plan(self, operations: tuple[tuple[int, tuple[int, ...]], ...]) -> tuple[int, int, int]:
        """Finalize the complete original capture plan before numerical effects."""
        ...
    def training_view(self, *, consumer_stream: int) -> tuple[Any, ...]: ...
    def model_work_buffer(self, *, consumer_stream: int) -> Any: ...
    def capture(
        self, enqueue: Callable[[], Any], *, memory_scope: Any,
        output_owner: SemanticTensorOwner, consumer_streams: Sequence[int],
    ) -> tuple[int, int, int]:
        """Retain the original capture and its work/event/call bounds without launching."""
        ...
    def launch(self) -> None: ...
    def resolve_launch(self) -> None: ...
    def record_model_invocation(self) -> None: ...
    def record_model_work(self, kind: int, dimensions: tuple[int, ...]) -> None: ...
    def record_model_device_work(self, kind: int, upper_dimensions: tuple[int, ...]) -> int: ...
    def finish(self, output_owner: SemanticTensorOwner, *, consumer_streams: Sequence[int]) -> SemanticCompletedModelEvaluation:
        """Incomplete expenditure raises without a result and retains the original owner/fence."""
        ...
    def resolve_completion(self) -> SemanticCompletedModelEvaluation:
        """Resolve the same submitted report and original streams without resubmitting.

        Returns the identical native-issued observation after successful issuance.
        A late unpoisoned completion failure raises SemanticModelEvaluationPending.
        """
        ...
    def cancel(self, *, consumer_streams: Sequence[int]) -> None: ...
    def cancellation_proof(self) -> SemanticCancelledModelEvaluation:
        """Known native non-submission; unavailable for Pending or Unknown."""
        ...

class SemanticCancelledModelEvaluation:
    """Original known cancel proof, not a successful numerical evaluation."""
    @property
    def capture_disposition(self) -> Literal[
        "not-entered", "entered-aborted", "recorded-and-cancelled",
    ]: ...
    def belongs_to(self, original: SemanticModelEvaluation) -> bool: ...

class SemanticTransitionColdTask:
    """Original admitted statement roster and its explicit ordered query selection."""

    def __init__(
        self, *, initial_theory: str, input_facts: str,
        observer_program: Optional[str], statements: tuple[tuple[str, int], ...],
        query_records: tuple[int, ...], task_ground: tuple[str, tuple[Any, ...]],
        training_canary_source: Optional[SemanticTransitionColdTask],
        capacities: tuple[int, int, int, int],
        admission_limits: tuple[int, int, int, int], device_ordinal: int,
        memory_bytes: int, provenance_capacity_records: int, prefix_capacity: int,
        feedback_capacity: int, pad_token: int, terminal_tokens: tuple[int, ...],
        final_intent_payload_bytes: int, intent_effect: bytes,
        intent_entry_capacity: int, intent_payload_capacity_bytes: int,
        acknowledgement_payload_capacity_bytes: int,
        authority_decisions_capacity_bytes: int, generations: tuple[int, int, int, int],
        training_cursor: int, training_rng: tuple[int, int, int, int], fuel: int,
        rng: tuple[int, int, int], material_source: Optional[SemanticMaterialSource] = None,
        private_replay_child: Any = None, prepared_actor_refresh: Any = None,
        actor_refresh_bank: Optional[int] = None,
    ) -> None: ...

    @property
    def task_content(self) -> tuple[bytes, bytes, bytes, tuple[int, ...]]: ...
    @property
    def statement_records(self) -> tuple[int, ...]:
        """Original selected record indices, preserving order and repeated selections."""
        ...
    @property
    def task_ground(self) -> tuple[str, tuple[Any, ...]]: ...
    def close(self) -> None: ...

class NativeTensorAllocation:
    """Native-issued physical allocation origin; no public constructor.

    Retains allocation lifetime, not reader access or immutable tensor content.
    """

    def same_allocation(self, other: NativeTensorAllocation) -> bool: ...
    @property
    def allocation_bytes(self) -> int: ...
    @property
    def byte_offset(self) -> int: ...
    @property
    def view_bytes(self) -> int: ...

class SemanticModelWorkBufferPort:
    """Native-issued original counter owner; no public constructor or early E."""

    @property
    def original_owner(self) -> Union[SemanticPreparedStep, SemanticColdModelWork, SemanticModelEvaluation]: ...
    @property
    def device_ordinal(self) -> int: ...
    @property
    def consumer_stream(self) -> int: ...
    @property
    def dtype(self) -> tuple[Literal[1], Literal[64]]: ...
    @property
    def row_width(self) -> Literal[3]: ...
    @property
    def finalized_extent(self) -> int:
        """Original admitted E after known initialization; no allocation or resume."""
        ...
    def same_counter(self, other: SemanticModelWorkBufferPort) -> bool: ...
    def export(self) -> tuple[
        NativeTensorAllocation, tuple[int, int], tuple[int, int],
        tuple[Literal[1], Literal[64]], int, int, Any,
    ]:
        """Same cached genuine finalized allocation/layout paired with its capsule.

        A consumed capsule is never replaced with a second counter import.
        """
        ...

_NativeWorkBufferLayout = tuple[
    NativeTensorAllocation, tuple[int, int], tuple[int, int],
    tuple[Literal[1], Literal[64]],
]
_TrainingViewLayout = tuple[
    NativeTensorAllocation, tuple[int, ...], tuple[int, ...], tuple[int, int],
]
_PolicyNumericalDomainLayout = tuple[
    NativeTensorAllocation, tuple[int, ...], tuple[int, ...],
    tuple[Literal[2], Literal[64]],
]
_PolicyNumericalDomainLayouts = tuple[
    _PolicyNumericalDomainLayout, _PolicyNumericalDomainLayout,
    _PolicyNumericalDomainLayout,
]
_PolicyOutputLayout = tuple[
    NativeTensorAllocation, tuple[int, ...], tuple[int, ...],
    tuple[Literal[2], Literal[32, 64]],
]

class SemanticTransitionTaskUse:
    """Native-issued task; no public constructor."""

    def prepare_initial_prefill(
        self, prepare_callback: Callable[[SemanticColdModelWork], Any],
        initialize_callback: Callable[[SemanticInitialPrefillStage, Any], Any], *,
        prefix_capacity: int, feedback_capacity: int, max_position: int,
        pad_token: int, provenance_capacity_records: int, model_generation: int,
        memory_capacity_bytes: int, other_external_cuda_bytes: int,
    ) -> tuple[SemanticInitialPrefillStage, Any]:
        """Prepare the original arena, then execute one pre-parent invocation.

        The first callback receives the genuine active cold-work owner and
        returns its prepared owner. The second receives the actual stage and
        that exact owner. Native returns the same stage and second result.
        Repeat only after SemanticInitialPrefillPending with identical callback
        objects and arguments; neither callback is entered again.
        """
        ...

    @property
    def training_arena_result(self) -> Optional[
        tuple[tuple[int, int, int, int, int, tuple[int, ...]], int]
    ]:
        """Original completed native work report and measured backing peak."""
        ...

    def learning_feedback_intervention(
        self, parent: Union[SemanticPublishedParent, SemanticPreparedStep],
    ) -> Optional[tuple[bytes, bool]]:
        """Original phase planning metadata, without a numerical model witness.

        Private evaluation requires its genuine entered Preparation callback;
        metadata does not grant admission, bind, capture or verify authority.
        """
        ...

    def cold_model_work_report_layouts(self) -> tuple[
        tuple[SemanticColdModelWork, _NativeWorkBufferLayout, int], ...,
    ]:
        """Complete original region roster, identical scratch layout and stream.

        Metadata only during the genuine callback. Future region handles retain
        identity without activation, buffer export or a fabricated parent.
        Lazy adoption must already have issued its original callback through
        active_cold_model_work; this getter never advances that transition.
        """
        ...

    @property
    def original_program(self) -> Optional[bytes]:
        """Exact immutable first-import Program bytes, never a regenerated schedule."""
        ...

    def task_ground(self) -> tuple[str, tuple[Any, ...]]:
        """Original tagged ground and full ordered verification bindings."""
        ...

    def task_actor_eligible(self) -> bool:
        """Original native ex-ante actor decision, including after restore.

        Available before the first Proposal, without a device read. Reading
        the decision grants no training rights and does not reclassify records.
        Missing or stale task bindings raise rather than returning False.
        """
        ...

    def policy_numerical_domain_layouts(
        self, parent: Union[SemanticPublishedParent, SemanticPreparedStep],
        bank: Optional[int],
    ) -> _PolicyNumericalDomainLayouts:
        """Original allocations, shapes, element strides and FP64 dtype.

        Pure structural projection: no tensor export, value read or numerical
        access. Retained import preparation and prepared banks already own these
        buffers; ordinary replay queries its original parent after policy bind.
        """
        ...

    def policy_numerical_domains(
        self, parent: Union[SemanticPublishedParent, SemanticPreparedStep],
        bank: Optional[int], consumer_stream: int, *,
        expected_layouts: _PolicyNumericalDomainLayouts,
    ) -> tuple[Any, Any, Any]:
        """Export the same three genuine domains after all nine snapshots.

        Requires the original projected allocation owners, offsets, spans and
        layouts. Existing numerical, recording and stream guards still apply.
        """
        ...

    @property
    def proposal_expense(self) -> tuple[int, int]:
        """Original whole-task capacity and irreversible nominal places spent.

        The bound is required at import. Expense is not an RNG coordinate
        or proof that an admitted operation completed.
        """
        ...

class SemanticInitialPrefillPending(RuntimeError):
    """Retain original preparation; retry only its unfinished native join."""

    task_use: SemanticTransitionTaskUse
    prepare_callback: Callable[[SemanticColdModelWork], Any]
    initialize_callback: Callable[[SemanticInitialPrefillStage, Any], Any]

class SemanticTaskReleasePending(RuntimeError):
    """Retain the original close handles and repeat only its unfinished native join."""

    operation: Literal["close_imported_task", "close_aborted_task", "close_aborted_publications", "close"]

class SemanticCompletedSegmentPending(RuntimeError):
    """Native execute/pending resolver handoff; not a cold getter retry signal.

    Retain the original Runtime and resolve its result, never resubmit.
    """

class SemanticPreparedSegmentPending(RuntimeError):
    """Retain the original controller/task use and call resolve_segment.

    The original attempt may still be pending. Never submit another graph,
    replace its callback or treat this signal as completed execution.
    """

class SemanticPreparedSegmentCancelled(RuntimeError):
    """Original native proof confirms no graph call; retire the construction.

    This is not an executed or skipped result roster and does not refund its
    original attempt, Program progression or nominal proposal expense.
    """

class SemanticTransitionCheckpointCancelled(RuntimeError):
    """Authenticated terminal checkpoint; no further request can be admitted."""

    checkpoint: SemanticMaterialBytes
    task_identity: bytes
    first_program_ordinal: int
    end_program_ordinal: int
    proposal_spent: int
    parent_identity: _PublishedIdentity

def restore_cancelled_checkpoint(
    checkpoint: SemanticMaterialBytes, *, program_bytes: bytes,
    material_source: _ReplayMaterialSource, max_resolved_material_bytes: int,
    snapshot: Any = None, learning_phase_issuer: Any = None,
    learning_phase_records: Any = None, learning_phase_record_limits: Any = None,
    learning_phase_fence: Any = None, learning_phase_checkpoint_issuer: Any = None,
) -> NoReturn:
    """Authenticate without CUDA and raise SemanticTransitionCheckpointCancelled.

    Invalid or nonterminal material raises ValueError instead. The original
    Program, original material source and saved resolved-memory cap, and
    independently pinned Session checkpoint issuer are required;
    the authenticated native task identity is returned only in the exception.
    """
    ...

class SemanticRetainedFinalUsePending(RuntimeError):
    """Retain the original target retirement and members until final use joins.

    Retry only the original native final-use owner. Do not repeat cleanup,
    historical import, or numerical work after this exception.
    """

class _OwnedCompilerDiagnostic:
    """Immutable original compiler payload; no public constructor.

    These exact native types are exported for authenticated error recognition.
    Acquire the projection workspace before str/encode/JSON and retain it with
    every derived object. Scalar extent getters share pre-acquired metadata:
    retain this payload while using those scalars. The returned workspace also
    retains the authentic payload, not only the original capacity pool.
    """
    def __str__(self) -> str: ...
    def text_utf8_bytes(self) -> int: ...
    def format_workspace_bytes(self) -> int: ...
    def reserve_projection_workspace(
        self, material_source: SemanticMaterialSource, allocation_bytes: int,
    ) -> SemanticMaterialReadWorkspace: ...

class _OwnedCompilerError(RuntimeError):
    """Original args[0] is _OwnedCompilerDiagnostic, not a copied string.

    Native custody survives reassignment of args. Standard traceback, cause,
    context and notes remain mutable; none grant early allocation retirement.
    """

class SemanticMaterialBytes:
    """Immutable byte-buffer view retaining the original bounded native allocation.

    No public constructor. Slices and read-only memoryviews share that owner;
    converting to a separate bytes object is not a bounded checkpoint handoff.
    """
    def __len__(self) -> int: ...
    def slice(self, begin: int, end: int) -> SemanticMaterialBytes: ...
    def range_equals(self, begin: int, end: int, source: Any) -> bool: ...
    def reserve_read_workspace(self, allocation_bytes: int) -> SemanticMaterialReadWorkspace: ...

class SemanticMaterialReadWorkspace:
    """Original-pool auxiliary lease; no public constructor or early release.

    Reserve a finite parser/container bound before creating those objects.
    Retain this issuer with every derived object until its last use. The issuer
    retains the same immutable material, including its original slice range.
    """
    ...

class SemanticPublicationMaterialProjection:
    """Immutable native projection retaining its original decoding/packing leases.

    No public constructor. Retain this owner with every derived field until its
    last use; cache logical_bytes retain their original guarded byte allocation.
    """
    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...

class SemanticMaterialSource:
    """Original process/store owner; no public constructor."""
    @property
    def store_identity(self) -> tuple[int, int]: ...
    def extend(self, material_source: _SealedReplayMaterialSource) -> None: ...
    def prepare_checkpoint_writer(self, capacity: int, working_bytes: int) -> SemanticMaterialWriter: ...
    def resolve_stored(self, sha256_hex: str, size: int) -> SemanticMaterialBytes: ...
    def reserve_read_workspace(self, allocation_bytes: int) -> SemanticMaterialReadWorkspace: ...

def prepare_material_source(
    material_source: _SealedReplayMaterialSource, max_material_bytes: int,
    max_material_store_bytes: int, max_resolved_material_bytes: int,
) -> SemanticMaterialSource: ...

class SemanticMaterialWriter:
    """Original pre-reserved producer output; no public constructor.

    Freeze a complete non-overlapping span partition before writing. Each span
    advances in order, while distinct spans may be produced in any order.
    Seal succeeds only once every byte in the declared extent is initialized.
    """
    @property
    def region_ordinal(self) -> int:
        """Original cold region; unavailable for writers without a cold report."""
        ...
    @property
    def model_material_capacity(self) -> int: ...
    @property
    def model_material_working_bytes(self) -> int: ...
    def declare_spans(self, extent: int, spans: tuple[tuple[int, int], ...]) -> None: ...
    def write(self, offset: int, source: Any) -> None: ...
    def seal(self) -> SemanticMaterialBytes: ...

class SemanticResidentModelRead:
    """Opaque original ordered occurrence; no constructor or numerical authority."""
    ...

class SemanticPreparedResidentModelRestoration:
    """Original planned snapshot issuance carried into the one native restore."""
    @property
    def training_domain(self) -> Any: ...
    @property
    def objective_law(self) -> tuple[Any, ...]: ...
    @property
    def feedback_intervention(self) -> tuple[Any, ...]: ...
    @property
    def restore_read(self) -> SemanticResidentModelRead: ...
    @property
    def snapshot_read(self) -> SemanticResidentModelRead: ...

class SemanticPreparedResidentModelMemory:
    """Original structural aliases with separately admitted numerical activation.

    Retain this carrier through cold use, then release it before its parent.
    Framework aliases retain their native generation independently. Preparing
    or inspecting this object grants no numerical access.
    """
    @property
    def memory(self) -> tuple[Any, Any, Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        """Cached original six-field envelope; genuine capsules issued once."""
        ...
    @property
    def model_contract_metadata(self) -> _SourceModelContractMetadata: ...
    def prepare_resident_read(self, work: SemanticColdModelWork) -> SemanticResidentModelRead:
        """Associate the real parent and exact region before sole admission."""
        ...
    def activate(self, *, consumer_stream: int) -> None:
        """Guard each original occurrence, or resume its identical pending owner."""
        ...

class SemanticPublishedParent:
    def prepare_resident_model_memory(self, *, consumer_stream: int) -> SemanticPreparedResidentModelMemory:
        """Structural aliases only, without guards, copies or stream effects."""
        ...
    def resident_model_memory(self, *, consumer_stream: int) -> tuple[Any, Any, Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        """Ordinary export with immediate original content activation."""
        ...
    def original_prepared_import(self) -> SemanticPreparedTaskImport:
        """The genuine retained owner for this same SOURCE parent and task."""
        ...
    def source_model_contract_metadata(self) -> _SourceModelContractMetadata:
        """Complete seal-authenticated host contract of this acquired parent.

        Returns record bytes and five layout coordinates without a device read.
        Planning metadata does not grant numerical or SOURCE import access.
        """
        ...
    def source_model_memory_geometry(self) -> _SourceModelMemoryGeometry:
        """Original full model allocation, storage, view and tensor geometry."""
        ...
    def resolve_model_context_preparation(self) -> bool:
        """Join only the original submitted context preparation; never submit a suffix."""
        ...

class SemanticTransitionSession:
    @staticmethod
    def inspect_publication_material(
        material: SemanticMaterialBytes,
    ) -> SemanticPublicationMaterialProjection: ...
    @staticmethod
    def checkpoint_referent(
        checkpoint: SemanticMaterialBytes, *, material_source: SemanticMaterialSource,
    ) -> bytes: ...
    @staticmethod
    def verify_checkpoint_referent(
        referent: bytes, checkpoint: SemanticMaterialBytes, *,
        material_source: SemanticMaterialSource,
    ) -> None: ...
    @property
    def prepared_task_import(self) -> Optional[SemanticPreparedTaskImport]:
        """Recover the identical retained owner, without repeating callbacks."""
        ...
    @property
    def learning_phase_checkpoint_issuer(self) -> Optional[bytes]:
        """Original Session signing issuer, also before the first learning phase.

        Read it from the live owner before durable publication and retain it in
        the trusted create-only store, never from the checkpoint being restored.
        This is not phase-completion evidence or a device observation.
        """
        ...

    @staticmethod
    def restore_checkpoint(
        checkpoint: SemanticMaterialBytes, *, device_ordinal: int, snapshot: Any,
        restore_model: Any, training_domain: Any, training_canary_source: Any,
        material_source: _ReplayMaterialSource, max_resolved_material_bytes: int,
        original_program: Optional[bytes] = None, referent: Any = None,
        resolve_checkpoint: Any = None, max_checkpoint_bytes: Any = None,
        max_total_checkpoint_bytes: Any = None, refresh_snapshot: Any = None,
        learning_phase_issuer: Any = None, learning_phase_records: Any = None,
        learning_phase_record_limits: Any = None, learning_phase_fence: Any = None,
        learning_phase_checkpoint_issuer: Any = None,
    ) -> Any: ...

class SemanticInitialPrefillReceipt:
    """One original output handoff, consumed by that task's bind_parent."""

class SemanticInitialPrefillStage:
    def source_layout(self) -> tuple[Any, ...]: ...
    def bind_model_content(
        self, *, tensors: Any, model_allocations: Any, consumer_stream: int,
    ) -> None: ...
    def record_input(self, *, token_tensor: Any, consumer_stream: int) -> None: ...
    def record_outputs(
        self, *, tensors: Any, consumer_stream: int,
    ) -> SemanticInitialPrefillReceipt: ...

class SemanticTensorOwner:
    """Actual original tensor storage and Python/autograd producer custody."""

    @property
    def learning_feedback_intervention(self) -> Optional[tuple[bytes, bool]]:
        """Genuine model-owner phase read, not cold planning metadata."""
        ...

_LearningPhase = Literal["alignment", "fast", "consolidation"]
_LearningPhaseView = tuple[
    int, int, Literal["preserve", "zero", "master-from-effective", "phase"], int, int
]

class SemanticLearningPhaseRecipe:
    def __init__(
        self, *, source: _LearningPhase, target: _LearningPhase,
        phase_index: int, completed_updates_index: int,
        views: Sequence[_LearningPhaseView],
    ) -> None: ...
    @property
    def recipe_digest(self) -> bytes: ...
    @property
    def source(self) -> _LearningPhase: ...
    @property
    def target(self) -> _LearningPhase: ...
    @property
    def phase_index(self) -> int: ...
    @property
    def completed_updates_index(self) -> int: ...
    @property
    def views(self) -> tuple[_LearningPhaseView, ...]: ...

class SemanticLearningPhaseTransition:
    """Native-issued private attempt retained through unknown completion."""
    @property
    def phase_id(self) -> bytes: ...
    @property
    def issuer_anchor(self) -> bytes: ...
    @property
    def phase_record_limits(self) -> tuple[int, int, int]: ...
    @property
    def status(self) -> Literal[
        "preparing", "unknown", "prepared", "refusal-pending", "refused", "committed"
    ]: ...
    @property
    def checkpoint_sha256(self) -> bytes: ...
    @property
    def refusal_outcome(self) -> Optional[tuple[Any, ...]]: ...
    @property
    def refusal_checkpoint_custody(self) -> Optional[
        tuple[bytes, tuple[bytes, bytes, bytes]]
    ]: ...
    def commit_checkpoint(self, durable_owner: Any) -> SemanticTransitionRestoredCheckpoint: ...
    def resolve_checkpoint(self) -> Optional[SemanticTransitionRestoredCheckpoint]: ...
    def continue_preparation(self) -> SemanticLearningPhaseTransition: ...
    def abandon_prepared(self) -> None: ...

class SemanticTransitionController:
    def prepare_learning_phase_transition(
        self, task_use: SemanticTransitionTaskUse, *,
        parent: SemanticPublishedParent, recipe: SemanticLearningPhaseRecipe,
        source_checkpoint: SemanticMaterialBytes,
        consumer_streams: Sequence[int], snapshot: Any,
        scientific_owner: Any, scientific_refusal_type: type[BaseException],
        source_model: Any,
        prepare_checkpoint_writer: Callable[
            [Any, SemanticColdModelWork], SemanticMaterialWriter
        ],
        execute_phase_instruction: Callable[..., Any],
        release_cancelled_phase_execution: Callable[..., Any],
        resource_observer: Any, feedback_interventions: Any,
        learning_grant_ref: Any, checkpoint_destination: Any,
        snapshot_model_state: Callable[[SemanticMaterialWriter], SemanticMaterialBytes],
        restore_model: Callable[..., Any],
        snapshot_restored_model: Callable[[Any, SemanticMaterialWriter], SemanticMaterialBytes],
        retire_restored_model: Callable[[Any], None],
        cancelled_retirement_context: Callable[..., Any],
        retire_source_model: Callable[[], None], refresh_snapshot: Callable[..., Any],
        phase_record_owner: Any, phase_record_limits: tuple[int, int, int],
        frozen_program_bytes: bytes, resolve_checkpoint: Optional[Callable[..., Any]] = None,
        max_checkpoint_bytes: Optional[int] = None,
        max_total_checkpoint_bytes: Optional[int] = None,
    ) -> SemanticLearningPhaseTransition:
        """Prepare metadata before looking up each original regional writer.

        The metadata callback receives the selected model and original work.
        Serialization consumes the returned writer separately and only once.
        """
        ...
    def pending_learning_phase_transition(self) -> Optional[SemanticLearningPhaseTransition]: ...
    def save_checkpoint(
        self, task_use: SemanticTransitionTaskUse, *,
        parent: SemanticPublishedParent, consumer_streams: Sequence[int],
        snapshot: Any,
        snapshot_model_state: Callable[[SemanticMaterialWriter], SemanticMaterialBytes],
        model_material_capacity: int, model_material_working_bytes: int,
        original_writer: Optional[SemanticMaterialWriter] = None,
    ) -> SemanticMaterialBytes:
        """Seal one original output and retain it across Unknown completion.

        Compiled cold paths pass their preissued region writer explicitly.
        The serializer receives that exact object, never a replacement writer.
        """
        ...
    def retain_model_tensor_owners(
        self, task_use: SemanticTransitionTaskUse, *, parent: Any,
        tensors: Any, consumer_stream: int, bank: Optional[int] = None,
    ) -> SemanticTensorOwner: ...
    def retain_tensor_owners(
        self, task_use: SemanticTransitionTaskUse, *, parent: Any,
        tensors: Any, consumer_stream: int,
    ) -> SemanticTensorOwner: ...
    def task_observation_roots(
        self, *, task_scope: Sequence[str], live_authorities: Any,
        statement_records: Sequence[int], allowed_support_records: Sequence[int],
        task_program_source: Optional[str], task_query_ordinals: Sequence[int],
        task_scoring: Sequence[int], admissible_truth_masks: Sequence[int],
        actor_eligible: bool, task_ground: Any, replay_rows: Any,
        replay_selection: Any, max_material_bytes: int,
        max_total_material_bytes: int, max_evidence_bytes: int,
        material_source: _ReplayMaterialSource, max_resolved_material_bytes: int,
    ) -> tuple[
        bytes, tuple[int, ...], tuple[int, ...],
        tuple[tuple[int, Optional[int], Union[
            tuple[Literal["admitted"], int],
            tuple[Literal["authenticated"], int, tuple[int, ...]],
        ]], ...],
    ]:
        """Original coverage, including the complete signed support reference."""
        ...

    @property
    def session(self) -> SemanticTransitionSession:
        """The identical retained Session owner, checked on its creator thread.

        Reading this carrier does not create a Session or perform native work.
        """
        ...

    def import_task(
        self, *, task_ref: Any, task_scope: Any, statement_records: Any,
        allowed_support_records: Any, task_program_source: Any,
        task_query_ordinals: Any, task_scoring: Any, task_priority_levels: Any,
        admissible_truth_masks: Any, actor_eligible: Any, task_ground: Any,
        live_authorities: Any,
        replay_rows: Any, training_objective: Any, training_domain: Any,
        training_canary_source: Any, replay_capacity: Any,
        max_material_bytes: Any, max_total_material_bytes: Any,
        max_evidence_bytes: Any, dependencies: Any, feedback_roots: Any,
        material_source: _ReplayMaterialSource, max_resolved_material_bytes: int,
        publication_grants: Any, inference_grants: Any, training_grants: Any,
        initial_sources: Any, source_mapping: Any, snapshot: Any,
        replay_selection: Any, replay_operation: Any, restore_invocation: Any,
        pack_policy: Any, finish_invocation: Any, refresh_snapshot: Any,
        proposal_capacity: int,
        original_program: Optional[bytes] = None,
        actor_refresh_initial_kit: Optional[tuple[bytes, tuple[Any, ...]]] = None,
        resolve_checkpoint: Any = None, max_checkpoint_bytes: Any = None,
        max_total_checkpoint_bytes: Any = None, retain_policy: bool = False,
        prepare_invocation: Optional[Callable[[SemanticTransitionTaskUse, SemanticPublishedParent], None]] = None,
    ) -> Union[SemanticTransitionTaskUse, tuple[SemanticTransitionTaskUse, Any], SemanticPreparedTaskImport]: ...

    def admit_training_arena(
        self, task_use: SemanticTransitionTaskUse, *, parent: SemanticPublishedParent,
        consumer_streams: Sequence[int], resource_observer: Any,
        snapshot: Any, replay_rows: Any,
        material_source: _ReplayMaterialSource, training_objective: Any,
        training_domain: Any, refresh_snapshot: Any,
        resolve_checkpoint: Any = None, max_checkpoint_bytes: Any = None,
        max_total_checkpoint_bytes: Any = None,
    ) -> SemanticPublishedParent:
        """Retain the original source and inherit its fixed payload pool."""
        ...

    def resolve_training_arena(
        self, *, parent: SemanticPublishedParent,
    ) -> SemanticPublishedParent: ...

    def acknowledge_replay_delivery(
        self, task_use: SemanticTransitionTaskUse, *, parent: SemanticPublishedParent,
        snapshot: Any, delivery_receipt: bytes, result_receipt: bytes,
        replay_row: Any, material_source: _ReplayMaterialSource,
    ) -> tuple[Any, ...]:
        """Use the original material source without replacing its retained cap."""
        ...

    def execute_task_import(
        self, prepared_import: SemanticPreparedTaskImport,
    ) -> tuple[SemanticTransitionTaskUse, SemanticRetainedReplayMember]:
        """Execute its original restore/pack/finalize once; cache the same result."""
        ...

    def admit_segment_instruction(
        self, task_use: SemanticTransitionTaskUse, *, parent: SemanticPublishedParent,
        instruction_bytes: bytes, first_program_ordinal: int, resource_observer: Any,
        checkpoint: Optional[bytes] = None,
    ) -> SemanticSegmentInstructionAdmission:
        """Ordinary admission requires its exact original pre-request checkpoint.

        An original private child uses its already-retained checkpoint custody.
        """
        ...

    def prepare_private_replay_child(
        self, task_use: SemanticTransitionTaskUse, *,
        instruction_admission: SemanticSegmentInstructionAdmission,
        step_index: int, replay_ordinal: int,
    ) -> SemanticPrivateReplayChild: ...

    def prepare_source_content_geometry(
        self, task_use: SemanticTransitionTaskUse, *,
        instruction_admission: SemanticSegmentInstructionAdmission,
    ) -> tuple[SemanticPreparedSourceContentGeometry, ...]:
        """Issue the original full SOURCE roster before numerical preparation."""
        ...

    def model_input_generation_port(
        self, task_use: SemanticTransitionTaskUse, *,
        instruction_admission: SemanticSegmentInstructionAdmission, step_index: int,
    ) -> SemanticModelInputGenerationPort:
        """TARGET input-generation port from the already-issued original roster.

        Shares the same native declaration with every CURRENT member/bank of
        this Update; does not allocate/export tensors or grant numerical access.
        """
        ...

    def finalize_segment_layout(
        self, task_use: SemanticTransitionTaskUse, *,
        layouts: tuple[tuple[SemanticPreparedStep, int], ...],
    ) -> None:
        """Freeze every original step's actual model-work capacity.

        Derived layout is finalized once before original initialization/capture.
        A refused allocation preserves this child and never permits a new plan.
        """
        ...

    def build_segment(
        self, task_use: SemanticTransitionTaskUse, *, transitions: Any,
        producer: Any,
        segment_capacity_bytes: int, other_external_cuda_bytes: int,
        instruction_admission: Optional[SemanticSegmentInstructionAdmission],
        consumer_streams: Sequence[int],
        execute_prepared_imports: Optional[Callable[[], None]] = None,
    ) -> None:
        """Prepare the whole roster, execute imports, then hand off and capture.

        The original allocation pool spans all stages through instantiation.
        Recording requires genuine completion of every original future import.
        """
        ...

    def close_imported_task(self, task_use: SemanticTransitionTaskUse) -> None: ...
    def close_aborted_task(self, task_use: SemanticTransitionTaskUse) -> None: ...
    def close_aborted_publications(
        self, releases: tuple[tuple[SemanticPublishedParent, Sequence[int]], ...],
    ) -> None:
        """Join only this complete original parent/consumer roster after abort."""
        ...

    def close(self, parent: SemanticPublishedParent) -> None:
        """Confirm terminal native release after the original parent and steps retire."""
        ...

    def prepare_model_evaluation(
        self, task_use: Any, parent: Any, *, model_work_capacity: Optional[int] = None,
        cohort: Optional[SemanticEvaluationCohort] = None,
    ) -> SemanticModelEvaluation: ...

    def execute_segment(
        self, task_use: SemanticTransitionTaskUse, *, operation: str,
        snapshot: Any, refresh_snapshot: Callable[[], Any],
    ) -> tuple[SemanticPublishedParent, tuple[
        tuple[SemanticPreparedStep, Optional[str], int, Optional[dict[str, Any]], Any, Optional[int]], ...
    ]]: ...

    def resolve_completed_segment(
        self, task_use: SemanticTransitionTaskUse, *, refresh_snapshot: Callable[[], Any],
    ) -> tuple[SemanticPublishedParent, tuple[
        tuple[SemanticPreparedStep, Optional[str], int, Optional[dict[str, Any]], Any, Optional[int]], ...
    ]]:
        """Deliver the same retained complete native roster with fresh rights; no execution."""
        ...

    def resolve_segment(
        self, task_use: SemanticTransitionTaskUse,
    ) -> tuple[SemanticPublishedParent, tuple[
        tuple[SemanticPreparedStep, Optional[str], int, Optional[dict[str, Any]], Any, Optional[int]], ...
    ]]:
        """Join only the original attempt and retained handoff, never relaunch.

        Uses its original authority-refresh callback. An unfinished original
        attempt raises SemanticPreparedSegmentPending. A proved pre-graph
        cancellation raises SemanticPreparedSegmentCancelled for retirement.
        """
        ...

    def admit_evaluation_instruction(
        self, task_use: SemanticTransitionTaskUse, *,
        parent: SemanticPublishedParent, instruction_bytes: bytes,
        program_ordinal: int, resource_observer: Any,
    ) -> SemanticEvaluationInstructionAdmission:
        """Admit the exact outer Program instruction, with the original observer capsule."""
        ...

    def bind_prepared_update_output(
        self, task_use: SemanticTransitionTaskUse, *, step: SemanticPreparedStep,
        bank: int, tensors: Any, model_allocations: Any, model_storages: Any,
        model_views: Any, numerical_admissibility: Any, slab_backing: Any,
        slab_bytes: int, baseline_forward: SemanticModelForwardWitness,
        candidate_forward: SemanticModelForwardWitness,
        canary_comparison: SemanticModelCanaryWitness,
        allocation_owner: SemanticTensorOwner, consumer_stream: int,
    ) -> None: ...

    def bind_prepared_control_output(
        self, task_use: SemanticTransitionTaskUse, *, step: SemanticPreparedStep,
        bank: int, tensors: Any, model_allocations: Any, model_storages: Any,
        model_views: Any, numerical_admissibility: Any, slab_backing: Any,
        slab_bytes: int, baseline_forward: SemanticModelForwardWitness,
        candidate_forward: SemanticModelForwardWitness,
        canary_comparison: SemanticModelCanaryWitness,
        allocation_owner: SemanticTensorOwner, consumer_stream: int,
    ) -> None: ...

    def prepare_family_cohorts(
        self, task_use: SemanticTransitionTaskUse, *, parent: SemanticPublishedParent,
        instruction_admission: SemanticSegmentInstructionAdmission,
        model_work: SemanticColdModelWork,
    ) -> dict[str, Any] | None: ...

# ---------------------------------------------------------------------------
# Native relation provenance
# ---------------------------------------------------------------------------

_RelationScalarType = Literal[
    "u32", "u64", "i32", "i64", "f32", "f64", "bool", "symbol"
]
_RelationScalarValue = Union[int, float, bool]

class _RelationRole(TypedDict):
    name: str
    sort: Optional[str]
    type: _RelationScalarType

class _OptionalRelationRoleFields(TypedDict, total=False):
    sort: Optional[str]
    type: _RelationScalarType

class _RelationRoleInput(_OptionalRelationRoleFields):
    name: str

class _RelationProvenanceSpan(TypedDict):
    start: int
    end: int

class _RelationProvenanceRecord(TypedDict):
    source: Optional[str]
    document: Optional[str]
    span: Optional[_RelationProvenanceSpan]
    content_hash: Optional[str]
    kind: Optional[str]
    polarity: Optional[str]

class _RelationProvenanceRecordInput(TypedDict, total=False):
    source: Optional[str]
    document: Optional[str]
    span: Optional[_RelationProvenanceSpan]
    content_hash: Optional[str]
    kind: Optional[str]
    polarity: Optional[str]

class _RelationExactCell(TypedDict):
    type: _RelationScalarType
    hex: str

class _RelationTupleFactInput(TypedDict):
    tuple: Sequence[_RelationScalarValue]
    provenance: Sequence[_RelationProvenanceRecordInput]

class _RelationExactFactInput(TypedDict):
    cells: Sequence[_RelationExactCell]
    provenance: Sequence[_RelationProvenanceRecordInput]

_RelationFactInput = Union[_RelationTupleFactInput, _RelationExactFactInput]

class _RelationFact(TypedDict):
    identity: str
    tuple: list[_RelationScalarValue]
    cells: list[_RelationExactCell]
    provenance: list[_RelationProvenanceRecord]

class _RelationSnapshot(TypedDict):
    relation: str
    metadata_present: bool
    row_count: int
    roles: list[_RelationRole]
    facts: list[_RelationFact]

class _RelationSessionEvidence(TypedDict):
    program_hash: str
    relations: dict[str, _RelationSnapshot]

class _RelationManifestPredicate(TypedDict):
    name: str
    arity: int
    schema_sha256: str

class _RelationManifestFact(TypedDict):
    identity: str
    cells: list[_RelationExactCell]
    provenance: list[_RelationProvenanceRecord]

class _RelationProvenanceManifest(TypedDict):
    format: Literal["xlog.relation-provenance"]
    version: Literal[1]
    predicate: _RelationManifestPredicate
    row_count: int
    metadata_present: bool
    roles: list[_RelationRole]
    facts: list[_RelationManifestFact]

class _RelationProvenanceExport(TypedDict):
    columns: list[Any]
    manifest: _RelationProvenanceManifest

class _OptionalRelationDeltaFields(TypedDict, total=False):
    insert_columns: Any
    delete_columns: Any
    insert_facts: Sequence[_RelationFactInput]

class _RelationDeltaUpdate(_OptionalRelationDeltaFields):
    name: str

# ---------------------------------------------------------------------------
# Joint constraint carrier
# ---------------------------------------------------------------------------

SOLVER_ABI_IDENTITY: str

class CarrierRefused(RuntimeError): ...

class SolverResourceExhausted(RuntimeError): ...

class RelationMetadataError(ValueError):
    """A relation role or fact provenance value violates the compiled schema."""
    ...

class JointConstraintCarrier:
    def __init__(
        self,
        device: int,
        entities: int,
        domain_lanes: int,
        candidates: int,
        labels: int,
        fuel_limit: int,
    ) -> None: ...
    def register_schema(self, catalog_sha: str, solver_identity: str) -> None: ...
    def bind_signatures(
        self, head_masks: list[int], tail_masks: list[int]
    ) -> None: ...
    def export_buffer(self, name: str) -> Any: ...
    def solve_label_feasibility(self, abstain_label: int) -> None: ...
    def solve_label_map_top2(self) -> None: ...
    def solve_components_exact(
        self, comp_offsets: list[int], comp_indices: list[int]
    ) -> None: ...
    def note_producer_stream(self, external_stream: int) -> None: ...
    def note_consumer_stream(self, external_stream: int) -> None: ...
    @property
    def fuel_spent(self) -> int: ...

# ---------------------------------------------------------------------------
# Logic (pure Datalog, no probabilities)
# ---------------------------------------------------------------------------

class LogicProgram:
    """Factory for compiling pure Datalog programs (no probabilistic facts)."""

    @staticmethod
    def compile(
        source: str,
        device: int = 0,
        memory_mb: int = 32768,
    ) -> CompiledLogicProgram: ...

    @staticmethod
    def compile_file(
        entrypoint: _Path,
        module_paths: Sequence[_Path] = (),
        device: int = 0,
        memory_mb: int = 32768,
    ) -> CompiledLogicProgram:
        """Compile an entry file and its complete transitive module closure."""
        ...

class CompiledLogicProgram:
    """A compiled GPU-resident Datalog program ready to evaluate."""

    def evaluate(
        self,
        dlpack_inputs: Optional[dict[str, Any]] = None,
        memory_mb: Optional[int] = None,
    ) -> LogicEvalResult:
        """Evaluate the program, optionally supplying DLPack input relations.

        *dlpack_inputs* maps relation name → sequence of DLPack column capsules.
        Returns a :class:`LogicEvalResult` containing one :class:`LogicQueryResult`
        per query atom in the program.
        """
        ...

    def memory_stats(self) -> dict[str, Any]:
        """Return memory-limit, current-allocation, and peak-memory diagnostics."""
        ...

    def rule_provenance(self) -> list[dict[str, Any]]:
        """Return rule ids, source kinds, trace hashes, and support relation ids."""
        ...

    def proof_traces(self) -> list[dict[str, Any]]:
        """Return direct query proof traces naming source facts and rule ids."""
        ...

    def session(self) -> LogicRelationSession:
        """Create a stateful session for incremental relation updates."""
        ...

    def evaluate_conditioned(
        self, prob_source: str, memory_mb: Optional[int] = None
    ) -> EpistemicEvalResult:
        """Run this epistemic program and condition an exact query on what it knows.

        Only facts declared in this program's own source feed the world view.
        Unlike :meth:`evaluate`, this method does NOT accept ``dlpack_inputs``:
        caller-supplied input relations are not consulted. A program that relies
        on such a relation ends up with no accepted world view here, and this
        method **raises** ``RuntimeError`` ("Unsupported epistemic construct:
        accepted GPU world-view evidence ... probabilistic evidence requires
        non-empty accepted GPU final output"). It does **not** fall back to the
        unconditioned prior. The guard is fail-closed by design: a conditioned
        query that silently became unconditioned would be indistinguishable from
        a successful one, which is the exact failure the trace counters exist to
        expose. Call :meth:`epistemic_evidence` first to detect the state without
        catching an exception -- it reports ``accepted_world_views == 0`` and
        does not raise.

        Both epistemic modes reach this surface: FAEEL programs and
        non-recursive ``#pragma epistemic_mode = g91`` programs both lower to a
        single-component epistemic plan and condition normally. Recursive G91
        shapes (positive ``possible`` cycles needing tuple-level compatibility)
        compile to a dedicated G91-compatibility plan and are rejected at
        planning, as are split, stratified and WFS plans.

        The trace of the returned result reports
        ``gpu_conditioned_evidence_facts`` (the total the engine validates), its
        per-class breakdown (``gpu_conditioned_know_evidence_facts``,
        ``gpu_conditioned_possible_evidence_facts``,
        ``gpu_conditioned_not_known_evidence_facts``,
        ``gpu_conditioned_not_possible_evidence_facts``),
        ``accepted_faeel_world_view_evidence_consumed`` /
        ``accepted_g91_world_view_evidence_consumed``, and positive GPU exact,
        PIR/CNF, and knowledge-compilation event counters. Conditioning reached
        the GPU exact path when the validated total and required GPU events are
        non-zero; a ``possible``-only or
        negated-evidence program conditions correctly with the ``know`` class at
        ``0``.

        ``log_z_e`` is log P(evidence): the exact log-probability of the
        conditioned evidence, obtained by weighted model counting over the
        compiled circuit. Query probabilities are ``exp(log_z_eq - log_z_e)``.
        For independent root facts this coincides with the log of the product of
        their priors -- measured on GPU, conditioning ``0.6::fact().
        query(fact()).`` on ``know fact()`` gives ``log_z_e == ln(0.6)``, and two
        known atoms each with prior 0.5 give ``log_z_e == ln(0.25)`` -- but those
        are illustrations of the independent-root case, not the definition.
        Evidence on a derived atom, on atoms sharing an ancestor, or negated
        evidence all depart from the product form.
        """
        ...

    def prepare_conditioned(
        self, prob_source: str, memory_mb: Optional[int] = None
    ) -> CompiledConditionedProgram:
        """Compile accepted evidence and an exact probabilistic circuit once.

        The returned handle can be evaluated repeatedly and supports atomic
        updates to independent probabilistic fact priors. It has the same
        source-only epistemic input limitation as :meth:`evaluate_conditioned`.
        """
        ...

    def epistemic_evidence(self) -> EpistemicEvidence:
        """Run this epistemic program and report what its world view accepted.

        Like :meth:`evaluate_conditioned`, this only ever sees facts declared
        in the program's own source; it does not accept caller-supplied input
        relations. A program that depends on such a relation reports
        ``accepted_world_views == 0`` (along with ``accepted_candidates == 0``
        and ``final_output_rows == 0``) here, without raising -- while
        :meth:`evaluate_conditioned` on that same program raises. The operator
        censuses ``know_operator_count`` and ``possible_operator_count`` come
        from the plan rather than the execution, so they stay non-zero: it is the
        accepted/consumed family that goes to zero, not every counter.
        """
        ...

class CompiledConditionedProgram:
    """A reusable exact circuit conditioned on one accepted epistemic world view.

    Evaluations and weight updates on the same prepared circuit do not overlap.
    Only independent probabilistic fact entries may be changed.
    If a failed device update cannot be rolled back, this circuit and every clone
    become permanently invalid and all later operations raise ``RuntimeError``.
    """

    def evaluate(self) -> EpistemicEvalResult:
        """Evaluate the current priors without recompiling source or structure."""
        ...

    def set_fact_probabilities(self, mapping: dict[int, float]) -> None:
        """Atomically update independent fact priors by CNF variable id.

        The entire mapping is validated before device mutation. Invalid ids,
        non-finite or out-of-range probabilities, annotated-disjunction choices,
        and compiler-introduced variables reject the complete batch.
        Each id is its :meth:`prob_var_map` list index. Index ``0`` is padding,
        never a mutable variable id.
        Updating a fact fixed by evidence preserves that assignment while changing
        its prior and the resulting evidence likelihood.
        A device-write failure is rolled back. If rollback also fails, the shared
        circuit is permanently invalidated and every later operation raises.
        """
        ...

    def prob_var_map(self) -> list[dict[str, Any]]:
        """Return the current CNF-variable map, including updated fact priors.

        The returned list is indexed by CNF variable id; index ``0`` is unused
        padding and has ``kind == "other"``. Waiting for the native shared state
        does not hold the Python GIL; the returned Python dictionaries are built
        after the native snapshot completes.
        """
        ...

class LogicRelationSession:
    """Persistent relation session for incremental Datalog evaluation.

    If a delta operation fails before commit but after preparation takes ownership
    of cached derived state, the authoritative relation rows and evidence remain
    unchanged, but XLOG discards the derived cache and retained runtime. The next
    `evaluate()` rebuilds them.
    """

    def put_relation(self, name: str, dlpack_columns: Any) -> None:
        """Snapshot DLPack columns into a persistent session relation."""
        ...

    def put_relation_rows(self, name: str, rows: Sequence[Sequence[str]]) -> None:
        """Parse typed lexical rows and upload them into a persistent relation."""
        ...

    def put_relation_with_provenance(
        self,
        name: str,
        dlpack_columns: Any,
        *,
        roles: Sequence[_RelationRoleInput],
        facts: Sequence[_RelationFactInput],
    ) -> _RelationSnapshot:
        """Snapshot a relation and atomically bind roles and whole-fact evidence."""
        ...

    def put_relation_from_manifest(
        self,
        name: str,
        dlpack_columns: Any,
        manifest: _RelationProvenanceManifest,
    ) -> _RelationSnapshot:
        """Snapshot from a version-1 manifest, consuming imported DLPack capsules."""
        ...

    def relation(self, name: str) -> RelationEvidence:
        """Return a stable evidence snapshot; raise KeyError when not stored."""
        ...

    def evidence(self, name: Optional[str] = None) -> _RelationSessionEvidence:
        """Return deterministic evidence; a missing named relation raises KeyError."""
        ...

    def evaluate(self, memory_mb: Optional[int] = None) -> LogicEvalResult:
        """Evaluate the program against all currently stored relations."""
        ...

    def insert_relation(
        self,
        name: str,
        dlpack_columns: Any,
        *,
        facts: Optional[Sequence[_RelationFactInput]] = None,
    ) -> dict[str, Any]:
        """Insert DLPack rows into a stored relation through the delta path."""
        ...

    def delete_relation(self, name: str, dlpack_columns: Any) -> dict[str, Any]:
        """Delete DLPack rows from a stored relation through the delta path."""
        ...

    def apply_relation_delta(
        self,
        name: str,
        insert_columns: Optional[Any] = None,
        delete_columns: Optional[Any] = None,
        *,
        insert_facts: Optional[Sequence[_RelationFactInput]] = None,
    ) -> dict[str, Any]:
        """Apply insert and/or delete rows to a stored relation."""
        ...

    def apply_relation_delta_batch(
        self, updates: Sequence[_RelationDeltaUpdate]
    ) -> dict[str, Any]:
        """Apply a batch of relation deltas with device-side coalescing.

        Each update dictionary contains ``name`` plus optional
        ``insert_columns`` and ``delete_columns`` DLPack column sequences. The
        dictionaries reject unknown keys. A fully canceled batch is a no-op and
        emits no callback or generation increment. Returned stats include ``input_delta_count``,
        ``coalesced_insert_rows``, ``coalesced_delete_rows``, and
        ``canceled_rows``.
        """
        ...

    def apply_relation_delta_debug(
        self,
        updates: Sequence[_RelationDeltaUpdate],
        check_equivalence: bool = False,
    ) -> dict[str, Any]:
        """Apply a delta batch and return changed_relation_names, debug_trace, and optional equivalent_to_full_recompute."""
        ...

    def delta_stats(self) -> dict[str, Any]:
        """Return statistics from the most recent relation delta update."""
        ...

    def rule_provenance(self) -> list[dict[str, Any]]:
        """Return rule ids, source kinds, trace hashes, and support relation ids."""
        ...

    def proof_traces(self) -> list[dict[str, Any]]:
        """Return direct query proof traces naming source facts and rule ids."""
        ...

    def register_relation_callback(self, callback: Any) -> int:
        """Register a session-level relation mutation callback.

        The callable receives one metadata-only payload dictionary after each
        successful relation delta commit. Returns a callback id for
        ``unregister_relation_callback``.
        """
        ...

    def unregister_relation_callback(self, callback_id: int) -> bool:
        """Unregister a relation callback by id. Returns True when removed."""
        ...

    def host_transfer_stats(self) -> dict[str, int]:
        """Return ``{dtoh_bytes: int, ...}`` transfer statistics."""
        ...

    def join_index_cache_stats(self) -> dict[str, int]:
        """Return persistent hash-index cache telemetry for this session."""
        ...

    def wcoj_dispatch_stats(self) -> dict[str, Any]:
        """Multiway/Free-Join dispatch telemetry for this session.

        Keys: ``free_join_dispatch_count``,
        ``factorized_delta_dispatch_count``,
        ``wcoj_groupby_fusion_dispatch_count``, ``wcoj_error_decline_count``,
        and nested ``wcoj_fallback`` route counts plus ``total``.
        Counters accumulate across evaluates within this session.
        """
        ...

    def reset_host_transfer_stats(self) -> None:
        """Reset all host-transfer statistics."""
        ...

    def set_strict_deterministic_d2h(self, enabled: bool) -> None:
        """Reject deterministic device-to-host transfers while enabled."""
        ...

    def strict_deterministic_d2h_enabled(self) -> bool:
        """Return whether the deterministic transfer gate is enabled."""
        ...

    def deterministic_d2h_violation_count(self) -> int:
        """Return the number of transfers rejected by the deterministic gate."""
        ...

    def reset_deterministic_d2h_violations(self) -> None:
        """Reset the deterministic transfer-gate violation counter."""
        ...

    def cuda_graph_stats(self) -> dict[str, int]:
        """Return CUDA Graph capture, launch, fallback, and cache-hit counters."""
        ...

    def neural_hot_loop_diagnostics(self) -> dict[str, Any]:
        """Return unified nn/4 hot-loop diagnostics.

        Keys include ``post_load_dtoh_bytes``, ``post_load_htod_bytes``,
        ``control_plane_bytes_per_iteration``, ``scalar_sync_checks``,
        ``cuda_graph``, and ``circuit_cache``.
        """
        ...

    def memory_stats(self) -> dict[str, Any]:
        """Return memory-limit, current-allocation, and peak-memory diagnostics."""
        ...

    def export_relation(self, name: str) -> list[Any]:
        """Export the named relation as a list of DLPack column capsules."""
        ...

    def export_relation_rows(self, name: str) -> list[list[str]]:
        """Download one stored or materialized relation as typed lexical rows."""
        ...

    def export_relation_with_provenance(
        self, name: str
    ) -> _RelationProvenanceExport:
        """Export DLPack columns with their exact version-1 provenance manifest."""
        ...

    def remove_relation(self, name: str) -> bool:
        """Remove the named relation.  Returns True if it existed."""
        ...

    def clear_relations(self) -> None:
        """Remove all stored relations from the session."""
        ...

class RelationEvidence:
    """Immutable native evidence snapshot for one relation."""

    def provenance(self) -> _RelationSnapshot:
        """Return the relation snapshot captured when this object was created."""
        ...

class LogicQueryResult:
    """Result for one query atom from a Datalog evaluation."""

    relation_name: str
    """Name of the queried relation."""
    columns: list[str]
    """Column names; empty for 0-arity (boolean) queries."""
    sort_labels: list[str]
    """Per-column sort labels; follows query output variable names."""
    tensors: list[Any]
    """DLPack column capsules; empty when *columns* is empty."""
    num_rows: int
    """Number of result rows (0 for false boolean queries)."""
    is_true: bool
    """True iff this is a 0-arity query with at least one result row."""

class LogicEvalResult:
    """Aggregated result from one :meth:`CompiledLogicProgram.evaluate` call."""

    queries: list[LogicQueryResult]

class EpistemicEvalResult:
    """Exact probabilities conditioned on an accepted epistemic world view.

    Result of :meth:`CompiledLogicProgram.evaluate_conditioned`. ``prob`` and
    ``log_prob`` are DLPack capsules over device memory, like
    :class:`EvalResult`. ``trace`` carries the production-path counters of the
    epistemic-to-probability adapter: they are the evidence that conditioning
    actually happened on the GPU.
    """

    atoms: list[str]
    """Query atom strings in evaluation order."""
    prob: Any
    """DLPack f64 tensor of per-query probabilities."""
    log_prob: Any
    """DLPack f64 tensor of per-query log-probabilities."""
    log_z_e: float
    """Exact log Z_E (natural log): log P(evidence), the log-probability of the
    conditioned evidence under the probabilistic program's distribution, computed
    by weighted model counting over the compiled circuit. Query probabilities are
    ``exp(log_z_eq - log_z_e)``. For independent root facts this coincides with
    the log of the product of their priors -- measured on GPU, ``ln(0.6)`` for one
    known atom at prior 0.6 and ``ln(0.25)`` for two known atoms each at prior 0.5
    -- but that is the independent-root special case, not the definition."""
    trace: dict[str, int]
    """Production-path counters, including ``gpu_conditioned_evidence_facts``
    (the validated total), its per-class breakdown
    (``gpu_conditioned_know_evidence_facts``,
    ``gpu_conditioned_possible_evidence_facts``,
    ``gpu_conditioned_not_known_evidence_facts``,
    ``gpu_conditioned_not_possible_evidence_facts``),
    ``accepted_faeel_world_view_evidence_consumed`` /
    ``accepted_g91_world_view_evidence_consumed``. Direct results report positive
    required GPU exact, PIR/CNF, and knowledge-compilation event counters;
    prepared results report a positive prepared-circuit reuse counter. Every
    result also exposes
    ``gpu_conditioned_circuit_preparation_compiles`` (actual GPU circuit compiler
    invocations), ``gpu_conditioned_circuit_materializations``,
    ``gpu_conditioned_circuit_disk_cache_restores``,
    ``gpu_conditioned_circuit_gpu_cache_hits``, and the process-local
    ``gpu_conditioned_circuit_generation`` / cache-slot identity. Direct
    non-reuse results carry zeroes for these reuse fields; generation ``0`` is
    the sentinel that no prepared circuit identity is attached."""

class EpistemicEvidence:
    """Counters of one accepted epistemic GPU execution.

    Result of :meth:`CompiledLogicProgram.epistemic_evidence`.
    ``accepted_world_views == 0`` means the program ran but nothing was
    accepted; :meth:`CompiledLogicProgram.evaluate_conditioned` on that same
    program raises ``RuntimeError`` rather than returning an unconditioned
    result, so this class is the non-raising way to detect the state.
    ``know_operator_count`` and ``possible_operator_count`` are plan-level
    censuses and stay non-zero even when nothing is accepted.
    """

    epistemic_mode: str
    know_operator_count: int
    possible_operator_count: int
    accepted_candidates: int
    rejected_candidates: int
    accepted_world_views: int
    final_output_rows: int

# ---------------------------------------------------------------------------
# Program (probabilistic / neural-symbolic)
# ---------------------------------------------------------------------------

class Program:
    """Factory for compiling probabilistic / neural-symbolic programs."""

    @staticmethod
    def compile(
        source: str,
        device: int = 0,
        memory_mb: int = 32768,
        prob_engine: Optional[str] = None,
    ) -> CompiledProgram:
        """Compile a ProbLog/DeepProbLog source string.

        Parameters
        ----------
        source:
            Full program source text.
        device:
            CUDA device ordinal (default 0).
        memory_mb:
            GPU memory budget in MiB (default 32768).
        prob_engine:
            Override the inference engine: ``"exact_ddnnf"`` / ``"exact"`` /
            ``"ddnnf"`` for exact d-DNNF inference, ``"mc"`` for Monte Carlo.
            When *None* the engine is inferred from the program source.
        """
        ...

class CompiledProgram:
    """A compiled probabilistic / neural-symbolic program.

    After compilation, register any neural networks with
    :meth:`register_network` / :meth:`register_embedding` before calling
    :meth:`evaluate` or :meth:`forward_backward`.
    """

    # ------------------------------------------------------------------
    # Probabilistic evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        return_grads: bool = False,
        samples: Optional[int] = None,
        seed: Optional[int] = None,
        confidence: float = 0.95,
        max_nonmonotone_iterations: int = 1024,
        sampling_method: Optional[str] = None,
        memory_mb: Optional[int] = None,
        allow_cpu_oracle: bool = False,
    ) -> EvalResult:
        """Evaluate the program and return probabilities (host-side).

        For exact programs: *samples* / *seed* are not supported and must be
        ``None``.  Set *return_grads=True* to also compute marginal gradients.

        For MC programs: *return_grads* must be ``False``.  *sampling_method*
        is ``"rejection"`` or ``"evidence_clamping"``.  Programs rejected by
        the GPU-resident MC engine (negation, aggregates, ...) fail closed
        unless *allow_cpu_oracle=True*, in which case the labeled CPU oracle
        runs and ``EvalResult.mc_engine`` is ``"cpu-oracle"``.
        """
        ...

    def prob_var_map(self) -> list[dict]:
        """Which probabilistic fact each CNF variable stands for.

        The returned list's length is the CNF encoder's variable *capacity*,
        not the number of CNF variables in use and not the number of random
        variables in the program — a real fraction of entries are
        ``{"kind": "other"}`` padding (do not use ``len()`` of the result as a
        variable count). Entry ``i`` describes CNF variable ``i`` — the same
        position ``i`` that the ``grad_true`` / ``grad_false`` vectors of
        ``evaluate(return_grads=True)`` use (index ``0`` is unused padding,
        since CNF variables are 1-indexed). Each entry is one of:

        - ``{"kind": "fact", "atom": str, "prob": float}`` for a plain
          probabilistic fact. ``prob`` is the Bernoulli weight actually
          assigned to this CNF variable, so ``prob * (1 - prob)`` is the
          correct Jacobian to convert ``grad_true`` / ``grad_false`` at this
          position into a derivative with respect to ``prob``.
        - ``{"kind": "choice", "atoms": list[str], "probs": list[float],
          "choice_index": int, "prob": float}`` for one Bernoulli decision of
          an annotated disjunction's chain. ``atoms`` / ``probs`` are the
          disjunction's *declared, marginal* probabilities (display context
          only). ``prob`` is the *conditional* Bernoulli parameter actually
          assigned to this CNF variable's weight; use
          ``prob * (1 - prob)`` — not
          ``probs[choice_index] * (1 - probs[choice_index])`` — as the
          Jacobian for ``grad_true`` / ``grad_false`` at this position.
        - ``{"kind": "other"}`` for a variable introduced by compilation that
          is not a source of randomness (this also covers unused capacity
          padding slots — see above; the two are indistinguishable here).

        Only available for the exact engine; raises ``ValueError`` for
        Monte Carlo programs, and also for exact programs compiled through
        the GPU count-lift fast path (count aggregates without evidence or
        disjunctions), since that path never builds a CNF encoding and has
        no variable map to report — this does not mean the program has no
        probabilistic facts.
        """
        ...

    def evaluate_device(
        self,
        samples: Optional[int] = None,
        seed: Optional[int] = None,
        confidence: float = 0.95,
        max_nonmonotone_iterations: int = 1024,
        sampling_method: Optional[str] = None,
        memory_mb: Optional[int] = None,
    ) -> McDeviceEvalResult:
        """GPU-native MC evaluation — result counts stay on the device.

        Only valid for programs compiled with ``prob_engine="mc"``.
        """
        ...

    def rule_provenance(self) -> list[dict[str, Any]]:
        """Return rule ids, source kinds, trace hashes, and support relation ids."""
        ...

    def proof_traces(self) -> list[dict[str, Any]]:
        """Return direct query proof traces naming source facts and rule ids."""
        ...

    def host_transfer_stats(self) -> dict[str, int]:
        """Return ``{dtoh_bytes: int, ...}`` transfer statistics."""
        ...

    def reset_host_transfer_stats(self) -> None:
        """Reset all host-transfer statistics."""
        ...

    def cuda_graph_stats(self) -> dict[str, int]:
        """Return CUDA Graph capture, launch, fallback, and cache-hit counters."""
        ...

    def memory_stats(self) -> dict[str, Any]:
        """Return memory-limit, current-allocation, and peak-memory diagnostics."""
        ...

    # ------------------------------------------------------------------
    # NLL loss helpers (host-side scalars)
    # ------------------------------------------------------------------

    def nll_loss(self, query: str) -> float:
        """Compute NLL loss ``-log P(query)`` for a single query."""
        ...

    def nll_loss_batch(self, queries: list[str]) -> float:
        """Sum of NLL losses for a list of queries."""
        ...

    def nll_loss_mean(self, queries: list[str]) -> float:
        """Mean NLL loss for a non-empty list of queries."""
        ...

    def nll_loss_tensor(self, query: str) -> Any:
        """NLL loss as a PyTorch scalar tensor (supports autograd)."""
        ...

    def nll_loss_batch_tensor(self, queries: list[str]) -> Any:
        """Batch NLL loss sum as a PyTorch scalar tensor."""
        ...

    def evaluate_loss(self, queries: list[str]) -> float:
        """Mean NLL loss over *queries* without updating parameters."""
        ...

    # ------------------------------------------------------------------
    # Neural network registration
    # ------------------------------------------------------------------

    def register_network(
        self,
        name: str,
        module: Any,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        batching: bool = True,
        k: Optional[int] = None,
        det: bool = False,
        cache: bool = True,
        cache_size: int = 10000,
        *,
        arity: Optional[int] = None,
        arg_sorts: Optional[Sequence[int]] = None,
        artifact_hash: Optional[str] = None,
    ) -> None:
        """Register a PyTorch classification network declared via ``nn()``.

        Parameters
        ----------
        name:
            Must match an ``nn()`` declaration in the program source.
        module:
            A ``torch.nn.Module`` instance.
        optimizer:
            A PyTorch optimizer (e.g. ``torch.optim.Adam``).
        scheduler:
            Optional learning-rate scheduler.
        batching:
            Batch inputs for GPU efficiency (default ``True``).
        k:
            Top-*k* sampling: only consider the top *k* class outputs.
        det:
            Deterministic mode: use argmax instead of sampling.
        cache:
            Cache network outputs (default ``True``).
        cache_size:
            Maximum number of cache entries (default 10000).
        """
        ...

    # Top-level pyxlog wraps register_network with nn/4 lineage metadata:
    # checkpoint_hash, split_hashes, calibration_metrics, cuda_device,
    # influence_audit {registration, records}, nn4_lineage,
    # record_nn4_influence, changed_acceptance.

    def register_embedding(
        self,
        name: str,
        module_or_tensor: Any,
        trainable: bool = True,
    ) -> None:
        """Register an embedding for an embedding-form ``nn()`` declaration.

        *module_or_tensor* may be a ``torch.nn.Embedding`` (trainable) or a
        2-D ``torch.Tensor`` (frozen; *trainable* must be ``False``).
        """
        ...

    # ------------------------------------------------------------------
    # Neural network / tensor-source accessors
    # ------------------------------------------------------------------

    def network_names(self) -> list[str]:
        """Names of all registered neural networks."""
        ...

    def declared_network_names(self) -> list[str]:
        """Names of all networks declared via ``nn()`` in the program."""
        ...

    def has_neural_predicate(self, name: str) -> bool:
        """Return ``True`` if *name* is declared via ``nn()``."""
        ...

    def neural_predicate_info(self, predicate: str) -> dict[str, Any]:
        """Return metadata dict ``{network: str, labels: list[str] | None}``."""
        ...

    def network_metadata(self, name: str) -> dict[str, Any]:
        """Registered metadata plus the actual declarations for a network.

        Returns ``{arity, arg_sorts, artifact_hash, declared: [{predicate,
        predicate_arity, input_arity, labels}]}``. Classification networks
        only; embedding-declared names are refused (they carry no registration
        metadata by design).
        """
        ...

    def label_to_index(self, predicate: str, label: str) -> int:
        """Resolve a class label to its index in the declared label list."""
        ...

    def forward_embedding(self, name: str, ids: list[int]) -> Any:
        """Look up embedding vectors for a list of integer IDs.

        Returns a PyTorch tensor with shape ``[len(ids), dim]``.
        """
        ...

    def template_cache_size(self) -> int:
        """Number of cached circuit templates."""
        ...

    def template_compile_count(self) -> int:
        """Number of times template compilation has been executed."""
        ...

    def neural_cache_stats(self) -> dict[str, Any]:
        """Return circuit-cache and registered-network cache telemetry."""
        ...

    def deterministic_topk(self, values: Any, k: int) -> dict[str, Any]:
        """Stable top-k over a 1-D tensor, resolving ties by lower index."""
        ...

    def set_batch_queries(self, enabled: bool = True) -> None:
        """Enable or disable multi-query batching for training."""
        ...

    # ------------------------------------------------------------------
    # Tensor source management
    # ------------------------------------------------------------------

    def add_tensor_source(self, name: str, tensor: Any) -> None:
        """Add a named tensor source (e.g. training images).

        *tensor* must be a PyTorch tensor; the first dimension is treated as
        the sample count.
        """
        ...

    def set_active_tensor_source(self, name: str) -> None:
        """Set the active tensor source by name."""
        ...

    def active_tensor_source(self) -> Optional[str]:
        """Name of the currently active tensor source, or ``None``."""
        ...

    def active_tensor_source_size(self) -> int:
        """Number of samples in the active tensor source."""
        ...

    def tensor_source_names(self) -> list[str]:
        """Names of all registered tensor sources."""
        ...

    def has_tensor_source(self, name: str) -> bool:
        """Return ``True`` if the named tensor source exists."""
        ...

    # ------------------------------------------------------------------
    # Training controls
    # ------------------------------------------------------------------

    def set_train_mode(self, train: bool) -> None:
        """Switch all registered networks between train / eval mode."""
        ...

    def zero_grad(self) -> None:
        """Zero gradients for all registered optimizers."""
        ...

    def optimizer_step(self) -> None:
        """Call ``step()`` on all registered optimizers."""
        ...

    def clip_grad_norms(self, max_norm: float) -> None:
        """Clip gradient norms via ``torch.nn.utils.clip_grad_norm_``."""
        ...

    def scheduler_step(self, network_name: Optional[str] = None) -> None:
        """Step learning-rate scheduler(s).

        If *network_name* is given, only that network's scheduler is stepped;
        otherwise all schedulers are stepped.
        """
        ...

    def get_lr(self, network_name: str) -> float:
        """Return the current learning rate for a registered network."""
        ...

    def set_lr(self, network_name: str, lr: float) -> None:
        """Set the learning rate for all parameter groups of a network."""
        ...

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------

    def forward_backward(self, query: str, expected: bool = True) -> float:
        """Forward + backward pass; returns the scalar NLL loss.

        Calls ``zero_grad()`` before invoking this, ``optimizer_step()`` after.
        """
        ...

    def forward_backward_tensor(self, query: str, expected: bool = True) -> Any:
        """Forward + backward pass; returns the NLL loss as a CUDA tensor."""
        ...

    def belnap_loss(
        self,
        pro: Any,
        contra: Any,
        quarantine: Any,
        pro_reward: float = 1.0,
        contra_penalty: float = 1.0,
        quarantine_penalty: float = 1.0,
        reduction: str = "mean",
    ) -> dict[str, Any]:
        """Belnap/CFR-oriented loss terms for pro, contra, and quarantine channels."""
        ...

    def semantic_loss_tensor(
        self,
        violations: Any,
        weight: float = 1.0,
        reduction: str = "mean",
    ) -> Any:
        """Non-negative semantic violation loss."""
        ...

    def mse_loss_tensor(
        self,
        pred: Any,
        target: Any,
        weight: float = 1.0,
        reduction: str = "mean",
    ) -> Any:
        """Weighted MSE tensor loss."""
        ...

    def infoloss_tensor(
        self,
        prob: Any,
        weight: float = 1.0,
        eps: float = ...,
        reduction: str = "mean",
    ) -> Any:
        """Weighted information loss ``-log(prob)`` with clamping."""
        ...

    # ------------------------------------------------------------------
    # Training loop helpers
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        queries: list[str],
        batch_size: int = 32,
        max_grad_norm: Optional[float] = None,
    ) -> EpochStats:
        """Run one training epoch over *queries* and return statistics."""
        ...

    def train_epoch_tensor(
        self,
        queries: list[str],
        batch_size: int = 32,
        max_grad_norm: Optional[float] = None,
    ) -> EpochStats:
        """GPU-native training epoch (no per-query ``.item()`` sync)."""
        ...

# ---------------------------------------------------------------------------
# Probabilistic evaluation result types
# ---------------------------------------------------------------------------

class EvalResult:
    """Result from :meth:`CompiledProgram.evaluate` (host-side tensors)."""

    atoms: list[str]
    """Query atom strings in evaluation order."""
    prob: Any
    """DLPack f64 tensor of per-query probabilities."""
    log_prob: Any
    """DLPack f64 tensor of per-query log-probabilities."""
    num_vars: int
    """Number of probabilistic variables in the compiled circuit."""
    log_z_e: Optional[float]
    """Exact log-evidence log Z_E (natural log). None for Monte Carlo results."""
    grad_true: Optional[list[Any]]
    """Per-query gradients for the true label (exact engine, return_grads=True)."""
    grad_false: Optional[list[Any]]
    """Per-query gradients for the false label (exact engine, return_grads=True)."""
    approx: bool
    """True when MC inference was used."""
    stderr: Optional[Any]
    """DLPack f64 tensor of per-query standard errors (MC only)."""
    ci_low: Optional[Any]
    """DLPack f64 tensor of lower confidence-interval bounds (MC only)."""
    ci_high: Optional[Any]
    """DLPack f64 tensor of upper confidence-interval bounds (MC only)."""
    samples: Optional[int]
    """Total MC samples drawn (MC only)."""
    evidence_samples: Optional[int]
    """MC samples satisfying the evidence (MC only)."""
    seed: Optional[int]
    """RNG seed used (MC only)."""
    confidence: Optional[float]
    """Confidence level for the CI (MC only)."""
    nonmonotone_semantics: Optional[str]
    """Semantics used for non-monotone cycles (MC only)."""
    nonmonotone_sccs: Optional[int]
    nonmonotone_cycles: Optional[int]
    nonmonotone_iteration_limit_hits: Optional[int]
    sampling_method: Optional[str]
    mc_engine: Optional[str]
    """MC only: ``"gpu-resident"`` (production megakernel engine) or
    ``"cpu-oracle"`` (explicit opt-in via *allow_cpu_oracle*)."""

class McDeviceEvalResult:
    """Device-resident MC result from :meth:`CompiledProgram.evaluate_device`."""

    query_counts: Any
    """DLPack i32 tensor of per-query satisfying-sample counts (CUDA)."""
    evidence_count: Any
    """DLPack i32 tensor with shape [1] — evidence satisfying count (CUDA)."""
    total_samples: int
    seed: int
    confidence: float
    nonmonotone_semantics: str
    nonmonotone_sccs: int
    nonmonotone_cycles: int
    nonmonotone_iteration_limit_hits: int
    sampling_method: str
    resident_no_host_certified: bool
    resident_no_host_policy_result: str
    resident_no_host_tracked_dtoh_calls: int
    resident_no_host_tracked_htod_calls: int
    resident_no_host_host_loop_iterations: int
    resident_no_host_per_sample_host_launches: int
    resident_no_host_untracked_metadata_reads: int
    resident_no_host_engine_launches: int
    resident_no_host_host_fixpoint_iterations: int
    resident_no_host_per_operator_host_allocations: int

class DifferentiableProofTraceMap:
    """XLOG differentiable proof traces keyed by stable proof ids."""

    def insert(
        self,
        answer_key: str,
        clause_id: str,
        support_atoms: list[str],
        initial_weight: float,
    ) -> int:
        """Insert one differentiable proof trace and return its stable proof id."""
        ...

    def trace(self, proof_id: int) -> Optional[dict[str, Any]]:
        """Return one exported proof trace or ``None``."""
        ...

    def traces(self) -> list[dict[str, Any]]:
        """Return all exported proof traces with weights and gradients."""
        ...

    def accumulate_binary_logistic_gradients(
        self,
        targets: list[tuple[str, float]],
    ) -> float:
        """Accumulate binary-logistic gradients grouped by answer key."""
        ...

    def apply_gradients(self, learning_rate: float) -> None:
        """Apply accumulated proof-trace gradients to symbolic weights."""
        ...

# ---------------------------------------------------------------------------
# Training infrastructure
# ---------------------------------------------------------------------------

class EpochStats:
    """Statistics for a single training epoch."""

    avg_loss: float
    """Average loss across all batches."""
    num_batches: int
    """Number of batches processed."""
    total_queries: int
    """Total number of queries processed."""

class TrainingHistory:
    """Loss history accumulated across epochs and batches."""

    epoch_losses: list[float]
    """Loss at the end of each epoch."""
    epoch_times: list[float]
    """Wall-clock time (seconds) for each epoch."""
    batch_losses: list[float]
    """Loss for each batch across all epochs."""
    stopped_early: bool
    """True if early stopping triggered due to validation loss plateau."""

def train_model(
    program: CompiledProgram,
    queries: list[str],
    epochs: int = 10,
    batch_size: int = 32,
    log_iter: int = 100,
    shuffle: bool = True,
    max_grad_norm: Optional[float] = None,
    val_queries: Optional[list[str]] = None,
    patience: Optional[int] = None,
) -> TrainingHistory:
    """Run the full training loop for *epochs* epochs.

    Supports early stopping when *val_queries* and *patience* are both provided.
    """
    ...

def train_model_tensor(
    program: CompiledProgram,
    queries: list[str],
    epochs: int = 10,
    batch_size: int = 32,
    log_iter: int = 100,
    shuffle: bool = True,
    max_grad_norm: Optional[float] = None,
    val_queries: Optional[list[str]] = None,
    patience: Optional[int] = None,
) -> TrainingHistory:
    """GPU-native training loop — loss stays on the device; single ``.item()`` per batch."""
    ...

# ---------------------------------------------------------------------------
# ILP (Inductive Logic Programming)
# ---------------------------------------------------------------------------

class IlpProgramFactory:
    """Factory for compiling ILP programs."""

    @staticmethod
    def compile(
        source: str,
        device: int = 0,
        memory_mb: int = 512,
        max_active_rules: Optional[int] = None,
    ) -> CompiledIlpProgram:
        """Compile an ILP program source string."""
        ...

class CompiledIlpProgram:
    """A compiled GPU-resident ILP program."""

    # ------------------------------------------------------------------
    # Variants / diagnostics
    # ------------------------------------------------------------------

    def compile_variant(self, source: str) -> CompiledIlpProgram:
        """Compile ``source`` on this program's CUDA provider instead of a new one.

        The result is a separate program with its own relation store, in the
        state ``IlpProgramFactory.compile(source, ...)`` would produce; only
        the per-compile CUDA provider setup is skipped.

        Because the provider is shared, so are its device-memory budget, its
        streams and its host-transfer counters (a variant's ``fact_exists``
        bumps the counter this program's ``d2h_transfer_count`` reads).
        Relation overrides uploaded with ``put_relation`` are NOT carried over.
        The variant uses this program's ``max_active_rules``.
        """
        ...

    def compile_timing_ms(self) -> dict[str, float]:
        """Wall-clock breakdown of the compile that produced this program.

        Keys: ``provider`` (absent for variants), ``frontend``, ``facts``,
        ``execute``; values in milliseconds. Diagnostic only — the set of
        phase names may change.
        """
        ...

    # ------------------------------------------------------------------
    # Candidate management
    # ------------------------------------------------------------------

    def set_candidate_map(self, candidates: list[tuple[int, int, int]]) -> None:
        """Upload the ``(i, j, k)`` → candidate-index mapping.  Call once per attempt."""
        ...

    def candidate_map_len(self) -> int:
        """Return the number of entries in the current candidate map (0 if not set)."""
        ...

    # ------------------------------------------------------------------
    # Rule mask APIs
    # ------------------------------------------------------------------

    def set_rule_mask(
        self,
        name: str,
        mask_hard_flat: Any,
        mask_soft_flat: Any,
        schema_size: int,
    ) -> None:
        """Set a dense rule mask (DLPack hard + soft flat tensors)."""
        ...

    def set_rule_mask_sparse(
        self,
        name: str,
        candidate_ids: list[int],
        soft_probs_dlpack: Any,
        budget: int,
        allow_recursive: bool = False,
    ) -> None:
        """Set a sparse rule mask via top-k selection from a DLPack soft-probability tensor."""
        ...

    def set_rule_mask_sparse_selected(
        self,
        name: str,
        selected_candidate_ids: list[int],
        selected_soft_probs_dlpack: Any,
        allow_recursive: bool = False,
    ) -> None:
        """Set a sparse mask from pre-selected candidate IDs and DLPack soft probabilities."""
        ...

    def set_rule_mask_sparse_selected_device(
        self,
        name: str,
        selected_candidate_ids_dlpack: Any,
        selected_soft_probs_dlpack: Any,
        allow_recursive: bool = False,
    ) -> None:
        """Device-resident variant of :meth:`set_rule_mask_sparse_selected`.

        Candidate IDs stay on the GPU; Rust resolves them against the candidate
        order from :meth:`set_candidate_map`.
        """
        ...

    def debug_ilp_mask_kind(self, name: str) -> Optional[str]:
        """Return a human-readable string describing the current mask kind."""
        ...

    # ------------------------------------------------------------------
    # Relation upload
    # ------------------------------------------------------------------

    def put_relation(self, name: str, dlpack_columns: Any) -> None:
        """Upload a relation as a sequence of DLPack column capsules (zero-copy)."""
        ...

    # ------------------------------------------------------------------
    # COO / memory configuration
    # ------------------------------------------------------------------

    def set_coo_chunk_budget(self, bytes: int) -> None:
        """Set the per-chunk temp allocation budget in bytes (default 16 MiB)."""
        ...

    def set_coo_memory_cap(self, bytes: int) -> None:
        """Deprecated alias for :meth:`set_coo_chunk_budget`."""
        ...

    def set_strict_zero_dtoh(self, strict: bool) -> None:
        """Raise instead of falling back to the chunked COO path when ``True``."""
        ...

    # ------------------------------------------------------------------
    # Loss / gradient computation
    # ------------------------------------------------------------------

    def compute_ilp_loss_grad_gpu(
        self,
        positives: list[tuple[str, list[int]]],
        negatives: list[tuple[str, list[int]]],
        cand_probs_obj: Any,
    ) -> tuple[Any, Any]:
        """Compute ILP loss and gradient on the GPU.

        Returns a ``(loss_capsule, grad_capsule)`` pair of DLPack tensors.
        """
        ...

    def compute_ilp_loss_grad_gpu_relations(
        self,
        positives_by_relation: Any,
        negatives_by_relation: Any,
        cand_probs_obj: Any,
    ) -> tuple[Any, Any]:
        """Relation-keyed variant of :meth:`compute_ilp_loss_grad_gpu`.

        *positives_by_relation* / *negatives_by_relation* are dicts mapping
        relation name → sequence of DLPack column capsules.
        """
        ...

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> None:
        """Run the ILP fixpoint evaluation for the current rule masks."""
        ...

    def reset_runtime(self) -> None:
        """Reset all mutable runtime state (ILP registry, store, caches)."""
        ...

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def get_tagged_results(self) -> list[tuple[int, int, int, int]]:
        """Return tagged ``(i, j, k, count)`` results from the last evaluation."""
        ...

    def fact_exists(self, relation: str, values: list[int]) -> bool:
        """Return ``True`` if the specified fact tuple exists in *relation*."""
        ...

    def relation_facts(self, rel_name: str) -> list[list[int]]:
        """Return all facts in *rel_name* as a list of int-lists."""
        ...

    def sample_false_positives(
        self,
        head_rel: str,
        exclude: list[tuple[str, list[int]]],
        max_n: int,
    ) -> list[list[int]]:
        """Sample up to *max_n* false-positive tuples from the head relation."""
        ...

    def tagged_entries_containing_fact(
        self,
        relation: str,
        values: list[int],
    ) -> list[tuple[int, int, int]]:
        """Return ``(i, j, k)`` tagged entries whose result contains the specified fact."""
        ...

    def batch_fact_membership(
        self,
        relation: str,
        facts: list[list[int]],
    ) -> list[bool]:
        """Return a boolean mask indicating which facts exist in *relation*."""
        ...

    def batch_fact_membership_device(
        self,
        relation: str,
        facts: list[list[int]],
    ) -> Any:
        """Device-resident membership test — returns a DLPack boolean tensor."""
        ...

    def batch_tagged_credit(
        self,
        relation: str,
        facts: list[list[int]],
    ) -> list[list[tuple[int, int, int]]]:
        """Return tagged ``(i, j, k)`` entries crediting each fact in *facts*."""
        ...

    def batch_tagged_credit_device(
        self,
        relation: str,
        facts: list[list[int]],
    ) -> IlpTaggedCreditDeviceResult:
        """Device-resident tagged-credit query — all result buffers stay on GPU."""
        ...

    # ------------------------------------------------------------------
    # Schema / metadata
    # ------------------------------------------------------------------

    def ilp_schema_size(self) -> int:
        """Number of relations in the ILP schema."""
        ...

    def ilp_relation_names(self) -> list[str]:
        """Names of all relations in the ILP schema."""
        ...

    def relation_type_annotations(self) -> list[tuple[str, list[str]]]:
        """Return ``[(name, [type_str, ...])]`` for all predicates."""
        ...

    def valid_candidates(
        self,
        mask_name: str,
        allow_recursive: bool = False,
    ) -> list[dict[str, Any]]:
        """Return valid candidate dicts for *mask_name*.

        Each dict has keys ``{id, i, j, k, left_name, right_name, head_name}``.
        """
        ...

    def commit_induced_rule(self, rule_source: str) -> None:
        """Append *rule_source* to the base program and recompile."""
        ...

    # ------------------------------------------------------------------
    # Transfer statistics
    # ------------------------------------------------------------------

    def d2h_transfer_count(self) -> int:
        """Number of device-to-host transfers since last reset."""
        ...

    def reset_d2h_transfer_count(self) -> None:
        """Reset the D2H transfer counter to zero."""
        ...

    def host_transfer_stats(self) -> dict[str, int]:
        """Return ``{dtoh_bytes: int, ...}`` transfer statistics."""
        ...

    def reset_host_transfer_stats(self) -> None:
        """Reset all host-transfer statistics."""
        ...

class IlpTaggedCreditDeviceResult:
    """Device-resident tagged-credit result from :meth:`CompiledIlpProgram.batch_tagged_credit_device`."""

    fact_row_offsets: Any
    """DLPack tensor: start offset in the flat entry arrays for each fact."""
    entry_indices: Any
    """DLPack tensor: flat entry indices."""
    entry_i: Any
    """DLPack tensor: ``i`` component of each tagged entry."""
    entry_j: Any
    """DLPack tensor: ``j`` component of each tagged entry."""
    entry_k: Any
    """DLPack tensor: ``k`` component of each tagged entry."""

# ---------------------------------------------------------------------------
# DLPack / Arrow utilities
# ---------------------------------------------------------------------------

def dlpack_roundtrip(
    tensor: Any,
    device: int,
    memory_mb: int,
) -> Any:
    """Import a DLPack tensor, copy through CUDA, and re-export as DLPack.

    Primarily used for testing the DLPack import/export pipeline.
    """
    ...

def dlpack_is_cuda(tensor: Any) -> bool:
    """Return True when a DLPack capsule is backed by CUDA memory."""
    ...

def intern_symbols(symbols: list[str]) -> list[int]:
    """Intern strings in XLOG's canonical registry and return their IDs."""
    ...

def resolve_symbols(symbol_ids: list[int]) -> list[str]:
    """Resolve canonical symbol IDs, rejecting any unknown identifier."""
    ...

# The following two functions are only present when pyxlog is compiled with
# ``--features arrow-device-import``.  They are included here unconditionally
# so that type checkers can reference them; at runtime they may be absent.

def export_arrow_device(
    dlpack_columns: Any,
    device: int = 0,
    memory_mb: int = 32768,
) -> Any:
    """Export DLPack columns as an Arrow C Device Array capsule (zero-copy).

    Requires the ``arrow-device-import`` feature.
    """
    ...

def import_arrow_device(
    device_array: Any,
    device: int = 0,
    memory_mb: int = 32768,
) -> tuple[list[Any], list[str], int]:
    """Import an Arrow C Device Array capsule as DLPack columns (zero-copy).

    Returns ``(column_capsules, column_names, num_rows)``.
    Requires the ``arrow-device-import`` feature.
    """
    ...


class SemanticModelForwardWitness:
    """Opaque receipt issued by finishing an original full forward stream."""
    pass

class SemanticModelCanaryWitness:
    """Opaque receipt issued by the original paired canary comparison."""
    pass
