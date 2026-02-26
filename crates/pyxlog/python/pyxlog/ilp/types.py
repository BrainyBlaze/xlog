"""dILP trainer types — see docs/plans/2026-02-26-dilp-hardening-design.md"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


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
    max_mined_negatives: int = 10

    # Scaling
    max_active_rules: int = 32
    debug_dense_mask: bool = True   # alpha: dense; beta: sparse default

    # Recursion
    allow_recursive_candidates: bool = False

    # Guarantees
    check_ambiguity: bool = False
    exhaustive_ambiguity: bool = False
    max_novel_rate: float = 0.0
    protected_relations: tuple[str, ...] = ()
    holdout_strategy: str = "loo"
    waiver_untyped: bool = False

    # Reproducibility
    seed: int | None = None
    deterministic: bool = False

    # Compilation
    device: int = 0
    memory_mb: int = 512

    # Observability
    telemetry_level: int = 0
    max_telemetry_steps: int = 1000
    telemetry_sink: Path | None = None

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


@dataclass
class TrainResult:
    converged: bool = False
    discovered_rule: str | None = None
    attempt_count: int = 0
    total_steps: int = 0
    precision: float = 0.0
    recall: float = 0.0
    holdout_f1: float | None = None
    confidence_margin: float = 0.0
    top_k_concentration: float = 0.0
    rule_frequency: float = 0.0
    holdout_variance: float = 0.0
    single_attempt: bool = True
    ambiguous_alternatives: list[str] | None = None
    artifact: LearnedArtifact = field(default_factory=LearnedArtifact)


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
    artifact: LearnedArtifact = field(default_factory=LearnedArtifact)
