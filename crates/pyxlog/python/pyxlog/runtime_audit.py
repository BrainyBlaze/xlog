"""Runtime audit contracts for CUDA-style nn/4 scoring loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class HostMaterializationError(RuntimeError):
    """Raised when an audited hot loop materializes device scores on host."""


@dataclass
class AuditViolation:
    operation: str
    detail: str


@dataclass
class CudaAuditSummary:
    d2h_transfers: int = 0
    h2d_transfers: int = 0
    scalar_extractions: int = 0
    score_row_downloads: int = 0
    violations: list[AuditViolation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


class CudaExecutionAudit:
    """Context manager that records host transfers and scalar extraction traps."""

    def __init__(self, *, forbid_host_materialization: bool = False) -> None:
        self.forbid_host_materialization = forbid_host_materialization
        self.d2h_transfers = 0
        self.h2d_transfers = 0
        self.scalar_extractions = 0
        self.score_row_downloads = 0
        self.violations: list[AuditViolation] = []
        self._score_tensor_ids: set[int] = set()
        self._torch = None
        self._originals: dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> "CudaExecutionAudit":
        import torch

        self._torch = torch
        self._originals = {
            "cpu": torch.Tensor.cpu,
            "tolist": torch.Tensor.tolist,
            "item": torch.Tensor.item,
        }
        audit = self

        def audited_cpu(tensor, *args, **kwargs):
            audit._record_host_materialization("cpu", tensor)
            return audit._originals["cpu"](tensor, *args, **kwargs)

        def audited_tolist(tensor, *args, **kwargs):
            audit._record_host_materialization("tolist", tensor)
            return audit._originals["tolist"](tensor, *args, **kwargs)

        def audited_item(tensor, *args, **kwargs):
            audit.scalar_extractions += 1
            audit._violate("item", "scalar extraction inside audited hot loop")
            return audit._originals["item"](tensor, *args, **kwargs)

        torch.Tensor.cpu = audited_cpu
        torch.Tensor.tolist = audited_tolist
        torch.Tensor.item = audited_item
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._torch is not None:
            self._torch.Tensor.cpu = self._originals["cpu"]
            self._torch.Tensor.tolist = self._originals["tolist"]
            self._torch.Tensor.item = self._originals["item"]
        return False

    def record_nn4_scores(self, name: str, scores: Any, *, device_resident: bool) -> None:
        """Register an nn/4 score tensor as part of the audited hot path."""

        self._score_tensor_ids.add(id(scores))
        if not device_resident:
            self.d2h_transfers += 1
            self._violate(name, "nn/4 scores were not device-resident")

    def record_h2d_transfer(self, detail: str = "explicit host-to-device transfer") -> None:
        self.h2d_transfers += 1
        self._violate("h2d", detail)

    def summary(self) -> CudaAuditSummary:
        return CudaAuditSummary(
            d2h_transfers=self.d2h_transfers,
            h2d_transfers=self.h2d_transfers,
            scalar_extractions=self.scalar_extractions,
            score_row_downloads=self.score_row_downloads,
            violations=list(self.violations),
        )

    def _record_host_materialization(self, operation: str, tensor: Any) -> None:
        self.d2h_transfers += 1
        if id(tensor) in self._score_tensor_ids or getattr(tensor, "numel", lambda: 0)() > 1:
            self.score_row_downloads += 1
        self._violate(operation, "host materialization inside audited hot loop")

    def _violate(self, operation: str, detail: str) -> None:
        self.violations.append(AuditViolation(operation=operation, detail=detail))
        if self.forbid_host_materialization:
            raise HostMaterializationError(f"{operation}: {detail}")
