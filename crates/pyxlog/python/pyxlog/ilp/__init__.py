"""dILP trainer module."""
from pyxlog.ilp.entropy import entropy_weight_at_step, normalized_entropy
from pyxlog.ilp.exceptions import IlpCandidateError, IlpConfigError, IlpTrainingError
from pyxlog.ilp.temperature import AdaptiveTempController, TempMode
from pyxlog.ilp.trainer import train_only
from pyxlog.ilp.types import (
    ArtifactMetadata,
    CandidateMapEntry,
    GateResult,
    LearnedArtifact,
    PromotionResult,
    PromotionStatus,
    StepRecord,
    TrainConfig,
    TrainResult,
    TrainTelemetry,
)

__all__ = [
    "AdaptiveTempController",
    "TempMode",
    "normalized_entropy",
    "entropy_weight_at_step",
    "IlpConfigError",
    "IlpCandidateError",
    "IlpTrainingError",
    "TrainConfig",
    "TrainResult",
    "TrainTelemetry",
    "StepRecord",
    "LearnedArtifact",
    "ArtifactMetadata",
    "CandidateMapEntry",
    "PromotionStatus",
    "PromotionResult",
    "GateResult",
    "train_only",
]
