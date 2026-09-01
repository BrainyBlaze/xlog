"""Type stubs for pyxlog._native (Rust/PyO3 bindings).

All tensor-valued attributes (``prob``, ``log_prob``, ``query_counts``, etc.)
are returned as DLPack capsule objects (PyCapsule with name ``"dltensor"``).
Consumers should call ``torch.from_dlpack(x)`` or the DLPack protocol on these
values.  The stubs represent them as ``Any`` because the capsule type is not
importable from Python.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Literal, Optional, Sequence, TypedDict, Union

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

__version__: str

_Path = Union[str, PathLike[str]]

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
