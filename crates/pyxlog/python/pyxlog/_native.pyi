"""Type stubs for pyxlog._native (Rust/PyO3 bindings).

All tensor-valued attributes (``prob``, ``log_prob``, ``query_counts``, etc.)
are returned as DLPack capsule objects (PyCapsule with name ``"dltensor"``).
Consumers should call ``torch.from_dlpack(x)`` or the DLPack protocol on these
values.  The stubs represent them as ``Any`` because the capsule type is not
importable from Python.
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

__version__: str

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

class CompiledLogicProgram:
    """A compiled GPU-resident Datalog program ready to evaluate."""

    def evaluate(
        self,
        dlpack_inputs: Optional[dict[str, Any]] = None,
    ) -> LogicEvalResult:
        """Evaluate the program, optionally supplying DLPack input relations.

        *dlpack_inputs* maps relation name → sequence of DLPack column capsules.
        Returns a :class:`LogicEvalResult` containing one :class:`LogicQueryResult`
        per query atom in the program.
        """
        ...

    def session(self) -> LogicRelationSession:
        """Create a stateful session for incremental relation updates."""
        ...

class LogicRelationSession:
    """Persistent relation session for incremental Datalog evaluation."""

    def put_relation(self, name: str, dlpack_columns: Any) -> None:
        """Upload a relation as a sequence of DLPack column capsules."""
        ...

    def evaluate(self) -> LogicEvalResult:
        """Evaluate the program against all currently stored relations."""
        ...

    def export_relation(self, name: str) -> list[Any]:
        """Export the named relation as a list of DLPack column capsules."""
        ...

    def remove_relation(self, name: str) -> bool:
        """Remove the named relation.  Returns True if it existed."""
        ...

    def clear_relations(self) -> None:
        """Remove all stored relations from the session."""
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
    ) -> EvalResult:
        """Evaluate the program and return probabilities (host-side).

        For exact programs: *samples* / *seed* are not supported and must be
        ``None``.  Set *return_grads=True* to also compute marginal gradients.

        For MC programs: *return_grads* must be ``False``.  *sampling_method*
        is ``"rejection"`` or ``"evidence_clamping"``.
        """
        ...

    def evaluate_device(
        self,
        samples: Optional[int] = None,
        seed: Optional[int] = None,
        confidence: float = 0.95,
        max_nonmonotone_iterations: int = 1024,
        sampling_method: Optional[str] = None,
    ) -> McDeviceEvalResult:
        """GPU-native MC evaluation — result counts stay on the device.

        Only valid for programs compiled with ``prob_engine="mc"``.
        """
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
