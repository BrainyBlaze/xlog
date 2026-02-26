"""dILP exception taxonomy."""


class IlpConfigError(ValueError):
    """Invalid TrainConfig or empty examples. Raised before GPU work."""


class IlpCandidateError(ValueError):
    """No valid candidates for the given mask. Cannot train."""


class IlpTrainingError(RuntimeError):
    """CUDA or numerical failure during training.

    Attributes:
        context: dict with keys: attempt, step, C, k, device_name,
                 allocated_bytes, terminal_reason.
    """

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}
