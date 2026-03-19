"""dILP trainer types — see docs/plans/2026-02-26-dilp-hardening-design.md"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer configuration for dILP training. Frozen after creation."""

    # Budget
    global_step_limit: int = 1000
    step_budget_per_attempt: int = 150
    max_attempts: int = 5

    # Adaptive temperature
    tau_start: float = 2.0
    tau_floor: float = 0.05
    plateau_window: int = 10
    plateau_threshold: float = 0.01
    warmup_increment: float = 0.3
    trap_disc_threshold: float = 0.85
    trap_progress_window: int = 15

    # Entropy regularization
    entropy_weight_start: float = 0.1
    entropy_weight_end: float = 0.0

    # Multi-start
    random_reserve_fraction: float = 0.2
    narrowing_top_fraction: float = 0.5

    # Hard-negative mining
    max_mined_negatives: int = 0

    # Scaling
    max_active_rules: int = 32
    debug_dense_mask: bool = False  # beta: sparse default; True = dense fallback

    # Recursion
    allow_recursive_candidates: bool = False

    # Guarantees
    check_ambiguity: bool = False
    exhaustive_ambiguity: bool = False
    max_novel_rate: float = 0.0
    protected_relations: tuple[str, ...] = ()
    holdout_strategy: str = "kfold"
    holdout_threshold: float = 0.95
    holdout_folds: int = 5
    typed_schema_required: bool = False
    waiver_untyped: bool = False

    # Reproducibility
    seed: int | None = None
    deterministic: bool = False
    strict_gpu_native: bool = True

    # Compilation
    device: int = 0
    memory_mb: int = 512

    # Observability
    telemetry_level: int = 0
    max_telemetry_steps: int = 1000
    telemetry_sink: Path | None = None

    # Artifact telemetry persistence
    persist_telemetry: bool = False
    telemetry_persist_limit: int = 100

    # Numeric stability
    max_numeric_failures: int = 3


class PromotionStatus(Enum):
    PROMOTED = "promoted"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    COMMIT_FAILED = "commit_failed"
    NOT_CONVERGED = "not_converged"
    GATE_FAILED = "gate_failed"


@dataclass
class CandidateMapEntry:
    id: int
    i: int
    j: int
    k: int
    left_name: str
    right_name: str
    head_name: str


@dataclass
class StepRecord:
    step: int
    loss: float
    argmax_rule: str
    discreteness: float
    temperature: float
    entropy: float
    stable_count: int
    forward_p95_us: float = 0.0
    active_candidates: int = 0


@dataclass
class TrainTelemetry:
    steps: list[StepRecord] = field(default_factory=list)
    step_timings: dict | None = None


@dataclass
class ArtifactMetadata:
    pyxlog_version: str = ""
    git_sha: str | None = None
    cuda_version: str = ""
    device_name: str = ""
    candidate_map_hash: str = ""
    config_hash: str = ""
    timestamp_utc: str = ""


@dataclass
class LearnedArtifact:
    candidate_map: list[CandidateMapEntry] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    soft_probs: list[float] = field(default_factory=list)
    selected_hard: list[int] = field(default_factory=list)
    discovered_rule: str = ""
    config_snapshot: TrainConfig | None = None
    telemetry: TrainTelemetry = field(default_factory=TrainTelemetry)
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)
    strict_gpu_native: bool = False
    compat_materialized: bool = True
    _compat_exporter: Callable[[], "LearnedArtifact"] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def export_compat_artifact(self) -> "LearnedArtifact":
        """Materialize a host-shaped compatibility artifact explicitly."""
        if self.compat_materialized:
            return self
        if self._compat_exporter is None:
            raise RuntimeError(
                "strict artifact has no compatibility exporter; "
                "call export_compat_artifact() on the originating TrainResult"
            )
        exported = self._compat_exporter()
        if not exported.compat_materialized:
            raise RuntimeError("export_compat_artifact() returned a non-materialized artifact")
        return exported

    def save(self, path: Path) -> None:
        """Save artifact to JSON file."""
        if not self.compat_materialized:
            raise RuntimeError(
                "strict GPU-native artifacts cannot be saved directly; "
                "call export_compat_artifact() first"
            )
        import json
        import hashlib
        import dataclasses

        map_data = [
            {"id": c.id, "i": c.i, "j": c.j, "k": c.k,
             "left_name": c.left_name, "right_name": c.right_name,
             "head_name": c.head_name}
            for c in self.candidate_map
        ]
        map_str = json.dumps(map_data, sort_keys=True)
        candidate_map_hash = hashlib.sha256(map_str.encode()).hexdigest()

        # Compute config_hash from config_snapshot if present
        config_hash = ""
        if self.config_snapshot:
            config_dict = dataclasses.asdict(self.config_snapshot)
            config_str = json.dumps(config_dict, sort_keys=True, default=str)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        # Bounded telemetry snapshot (only when config says persist)
        telemetry_snapshot = None
        if (self.config_snapshot
                and self.config_snapshot.persist_telemetry
                and self.telemetry.steps):
            limit = self.config_snapshot.telemetry_persist_limit
            truncated_steps = self.telemetry.steps[-limit:]
            telemetry_snapshot = {
                "steps": [dataclasses.asdict(s) for s in truncated_steps],
                "step_timings": self.telemetry.step_timings,
            }

        data = {
            "schema_version": "beta-v2",
            "discovered_rule": self.discovered_rule,
            "candidate_map": map_data,
            "telemetry_snapshot": telemetry_snapshot,
            "logits": self.logits,
            "soft_probs": self.soft_probs,
            "selected_hard": self.selected_hard,
            "metadata": {
                "pyxlog_version": self.metadata.pyxlog_version,
                "git_sha": self.metadata.git_sha,
                "cuda_version": self.metadata.cuda_version,
                "device_name": self.metadata.device_name,
                "candidate_map_hash": candidate_map_hash,
                "config_hash": config_hash,
                "timestamp_utc": self.metadata.timestamp_utc,
            },
            "config_snapshot": dataclasses.asdict(self.config_snapshot) if self.config_snapshot else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path, verify_hash: bool = False) -> "LearnedArtifact":
        """Load artifact from JSON file."""
        import json
        import hashlib

        with open(path) as f:
            data = json.load(f)

        candidate_map = [
            CandidateMapEntry(**c) for c in data["candidate_map"]
        ]

        if verify_hash:
            map_str = json.dumps(data["candidate_map"], sort_keys=True)
            computed = hashlib.sha256(map_str.encode()).hexdigest()
            stored = data.get("metadata", {}).get("candidate_map_hash", "")
            if computed != stored:
                raise ValueError(
                    f"Candidate map hash mismatch: computed {computed}, stored {stored}"
                )

        # Schema version compatibility check (always validated)
        stored_version = data.get("schema_version", "")
        _SUPPORTED_VERSIONS = {"beta-v1", "beta-v2"}
        if stored_version and stored_version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Incompatible schema version: {stored_version} "
                f"(supported: {sorted(_SUPPORTED_VERSIONS)})"
            )

        meta_data = data.get("metadata", {})
        metadata = ArtifactMetadata(
            pyxlog_version=meta_data.get("pyxlog_version", ""),
            git_sha=meta_data.get("git_sha"),
            cuda_version=meta_data.get("cuda_version", ""),
            device_name=meta_data.get("device_name", ""),
            candidate_map_hash=meta_data.get("candidate_map_hash", ""),
            config_hash=meta_data.get("config_hash", ""),
            timestamp_utc=meta_data.get("timestamp_utc", ""),
        )

        # Restore config_snapshot if present
        config_snapshot = None
        raw_config = data.get("config_snapshot")
        if raw_config is not None:
            # Fix types that dataclasses.asdict + json.dump changed:
            # - Path fields serialized as strings
            # - tuple fields serialized as lists
            if raw_config.get("telemetry_sink") is not None:
                raw_config["telemetry_sink"] = Path(raw_config["telemetry_sink"])
            if "protected_relations" in raw_config:
                raw_config["protected_relations"] = tuple(
                    raw_config["protected_relations"]
                )
            config_snapshot = TrainConfig(**raw_config)

        # Restore telemetry from snapshot if present (v2+)
        telemetry = TrainTelemetry()
        raw_telemetry = data.get("telemetry_snapshot")
        if raw_telemetry is not None:
            restored_steps = [
                StepRecord(**s) for s in raw_telemetry.get("steps", [])
            ]
            telemetry = TrainTelemetry(
                steps=restored_steps,
                step_timings=raw_telemetry.get("step_timings"),
            )

        return cls(
            candidate_map=candidate_map,
            logits=data.get("logits", []),
            soft_probs=data.get("soft_probs", []),
            selected_hard=data.get("selected_hard", []),
            discovered_rule=data.get("discovered_rule", ""),
            config_snapshot=config_snapshot,
            telemetry=telemetry,
            metadata=metadata,
            strict_gpu_native=False,
            compat_materialized=True,
        )


@dataclass
class StrictLearnedArtifact:
    candidate_map: list[CandidateMapEntry] = field(default_factory=list)
    config_snapshot: TrainConfig | None = None
    telemetry: TrainTelemetry = field(default_factory=TrainTelemetry)
    strict_gpu_native: bool = True
    compat_materialized: bool = False
    _compat_exporter: Callable[[], LearnedArtifact] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def export_compat_artifact(self) -> LearnedArtifact:
        """Materialize a host-shaped compatibility artifact explicitly."""
        if self._compat_exporter is None:
            raise RuntimeError(
                "strict artifact has no compatibility exporter; "
                "call export_compat_artifact() only on strict GPU-native artifacts"
            )
        exported = self._compat_exporter()
        if not exported.compat_materialized:
            raise RuntimeError("export_compat_artifact() returned a non-materialized artifact")
        return exported

    def save(self, path: str) -> None:
        raise RuntimeError(
            "strict GPU-native artifacts cannot be saved directly; "
            "call export_compat_artifact() first"
        )


@dataclass
class TrainResult:
    converged: bool = False
    discovered_rule: str | None = None
    attempt_count: int = 0
    total_steps: int = 0
    precision: float = 0.0
    recall: float = 0.0
    holdout_f1: float | None = None
    confidence_margin: float | None = None
    top_k_concentration: float | None = None
    rule_frequency: float = 0.0
    holdout_variance: float = 0.0
    single_attempt: bool = True
    ambiguous_alternatives: list[str] | None = None
    artifact: LearnedArtifact = field(default_factory=LearnedArtifact)
    strict_gpu_native: bool = False
    compat_materialized: bool = True
    _compat_exporter: Callable[[], "TrainResult"] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def export_compat_result(self) -> "TrainResult":
        """Materialize compatibility-only host summaries explicitly."""
        if self.compat_materialized:
            return self
        if self._compat_exporter is None:
            raise RuntimeError(
                "strict GPU-native result has no compatibility exporter; "
                "call train_only(...).export_compat_result() only on strict results"
            )
        exported = self._compat_exporter()
        if not exported.compat_materialized:
            raise RuntimeError("export_compat_result() returned a non-materialized result")
        return exported


@dataclass
class StrictTrainResult:
    attempt_count: int = 0
    total_steps: int = 0
    single_attempt: bool = True
    winner_candidate_id: int | None = None
    discovered_rule: str | None = None
    artifact: StrictLearnedArtifact = field(default_factory=StrictLearnedArtifact)
    strict_gpu_native: bool = True
    compat_materialized: bool = False
    _compat_exporter: Callable[[], TrainResult] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def export_compat_result(self) -> TrainResult:
        """Materialize compatibility-only host summaries explicitly."""
        if self._compat_exporter is None:
            raise RuntimeError(
                "strict GPU-native result has no compatibility exporter; "
                "call export_compat_result() only on strict results"
            )
        exported = self._compat_exporter()
        if not exported.compat_materialized:
            raise RuntimeError("export_compat_result() returned a non-materialized result")
        return exported


@dataclass
class GateResult:
    name: str = ""
    passed: bool = False
    detail: str = ""


@dataclass
class PromotionResult:
    status: PromotionStatus = PromotionStatus.NOT_CONVERGED
    gates: list[GateResult] = field(default_factory=list)
    novel_count: int | None = None
    novel_rate: float | None = None
    novel_examples: list[str] | None = None
    committed_source: str | None = None
    ambiguous_alternatives: list[str] | None = None
    artifact: LearnedArtifact = field(default_factory=LearnedArtifact)
