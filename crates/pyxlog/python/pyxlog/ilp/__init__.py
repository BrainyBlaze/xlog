"""dILP trainer module."""
from pyxlog.ilp.backend import DenseMaskBackend, SparseMaskBackend
from pyxlog.ilp.entropy import entropy_weight_at_step, normalized_entropy
from pyxlog.ilp.exact_induce import ExactInductionResult, ScoredCandidate, induce_exact
from pyxlog.ilp.exceptions import IlpCandidateError, IlpConfigError, IlpTrainingError
from pyxlog.ilp.holdout import loo_holdout_f1
from pyxlog.ilp.temperature import AdaptiveTempController, TempMode
from pyxlog.ilp.promoter import train_and_promote
from pyxlog.ilp.trainer import train_on_compiled_relations, train_only
from pyxlog.ilp.types import (
    ArtifactMetadata,
    CandidateMapEntry,
    GateResult,
    LearnedArtifact,
    PromotionResult,
    PromotionStatus,
    StepRecord,
    StrictLearnedArtifact,
    StrictTrainResult,
    TrainConfig,
    TrainResult,
    TrainTelemetry,
)

__all__ = [
    "ExactInductionResult",
    "ScoredCandidate",
    "induce_exact",
    "AdaptiveTempController",
    "TempMode",
    "DenseMaskBackend",
    "SparseMaskBackend",
    "normalized_entropy",
    "entropy_weight_at_step",
    "IlpConfigError",
    "IlpCandidateError",
    "IlpTrainingError",
    "TrainConfig",
    "TrainResult",
    "StrictTrainResult",
    "TrainTelemetry",
    "StepRecord",
    "LearnedArtifact",
    "StrictLearnedArtifact",
    "ArtifactMetadata",
    "CandidateMapEntry",
    "PromotionStatus",
    "PromotionResult",
    "GateResult",
    "train_only",
    "train_on_compiled_relations",
    "train_and_promote",
    "loo_holdout_f1",
]
