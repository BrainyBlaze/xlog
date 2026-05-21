"""Reusable learned-rule inventory helpers for dILP promotion outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyxlog.ilp.types import CandidateMapEntry, GateResult, LearnedArtifact


@dataclass
class RuleInventoryClause:
    """One selected or rejected learned clause entry."""

    id: int | str
    rule: str
    status: str
    score: float | None = None
    weight: float | None = None
    neural_predicate: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "rule": self.rule,
            "status": self.status,
        }
        if self.score is not None:
            data["score"] = self.score
        if self.weight is not None:
            data["weight"] = self.weight
        if self.neural_predicate is not None:
            data["neural_predicate"] = self.neural_predicate
        return data


@dataclass
class RuleInventory:
    """Machine-readable audit record for learned symbolic rules."""

    schema_version: int = 1
    selected_clauses: list[RuleInventoryClause] = field(default_factory=list)
    rejected_clauses: list[RuleInventoryClause] = field(default_factory=list)
    training_fold: str | None = None
    excluded_held_out_domains: list[str] = field(default_factory=list)
    kernel_checksums: dict[str, str | None] = field(default_factory=dict)
    kernel_mutated: bool = False
    promotion_status: str | None = None
    gates: list[dict[str, Any]] = field(default_factory=list)
    training_objective: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "selected_clauses": [clause.to_dict() for clause in self.selected_clauses],
            "rejected_clauses": [clause.to_dict() for clause in self.rejected_clauses],
            "excluded_held_out_domains": list(self.excluded_held_out_domains),
            "kernel_checksums": dict(self.kernel_checksums),
            "kernel_mutated": self.kernel_mutated,
            "gates": list(self.gates),
        }
        if self.training_fold is not None:
            data["training_fold"] = self.training_fold
        if self.promotion_status is not None:
            data["promotion_status"] = self.promotion_status
        if self.training_objective is not None:
            data["training_objective"] = self.training_objective
        return data


def build_rule_inventory(
    artifact: LearnedArtifact,
    *,
    training_fold: str | None = None,
    held_out_domains: tuple[str, ...] | list[str] = (),
    base_kernel_checksum_before: str | None = None,
    base_kernel_checksum_after: str | None = None,
    promotion_status: str | None = None,
    gates: list[GateResult] | None = None,
    training_objective: str | None = None,
) -> RuleInventory:
    """Build an auditable inventory from a learned artifact."""

    selected_ids = set(artifact.selected_hard)
    if not selected_ids and artifact.discovered_rule:
        for entry in artifact.candidate_map:
            if _candidate_rule(entry) == artifact.discovered_rule:
                selected_ids.add(entry.id)
                break

    selected: list[RuleInventoryClause] = []
    rejected: list[RuleInventoryClause] = []
    for index, entry in enumerate(artifact.candidate_map):
        clause = RuleInventoryClause(
            id=entry.id,
            rule=_candidate_rule(entry),
            status="selected" if entry.id in selected_ids else "rejected",
            score=_at(artifact.soft_probs, index),
            weight=_at(artifact.logits, index),
        )
        if entry.id in selected_ids:
            selected.append(clause)
        else:
            rejected.append(clause)

    kernel_mutated = (
        base_kernel_checksum_before is not None
        and base_kernel_checksum_after is not None
        and base_kernel_checksum_before != base_kernel_checksum_after
    )
    return RuleInventory(
        selected_clauses=selected,
        rejected_clauses=rejected,
        training_fold=training_fold,
        excluded_held_out_domains=list(held_out_domains),
        kernel_checksums={
            "before": base_kernel_checksum_before,
            "after": base_kernel_checksum_after,
        },
        kernel_mutated=kernel_mutated,
        promotion_status=promotion_status,
        gates=[
            {"name": gate.name, "passed": gate.passed, "detail": gate.detail}
            for gate in (gates or [])
        ],
        training_objective=training_objective,
    )


def _candidate_rule(entry: CandidateMapEntry) -> str:
    return (
        f"{entry.head_name}(X, Y) :- "
        f"{entry.left_name}(X, Z), {entry.right_name}(Z, Y)."
    )


def _at(values: list[float], index: int) -> float | None:
    if 0 <= index < len(values):
        return float(values[index])
    return None
