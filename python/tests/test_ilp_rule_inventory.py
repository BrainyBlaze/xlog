"""Regression coverage for UCR-XLOG-003."""

import pytest

from pyxlog.ilp import (
    CandidateMapEntry,
    LearnedArtifact,
    PromotionStatus,
    TrainConfig,
    TrainResult,
)
from pyxlog.ilp import promoter


class _TrialProgramStub:
    def evaluate(self) -> None:
        return None

    def fact_exists(self, relation: str, values: list[int]) -> bool:
        return relation == "reach" and values in ([1, 3], [2, 4])

    def relation_facts(self, relation: str) -> list[list[int]]:
        if relation == "reach":
            return [[1, 3], [2, 4]]
        return []

    def relation_type_annotations(self) -> list[tuple[str, list[str]]]:
        return [
            ("edge", ["u32", "u32"]),
            ("reach", ["u32", "u32"]),
        ]


class _FactoryStub:
    @staticmethod
    def compile(*_args, **_kwargs) -> _TrialProgramStub:
        return _TrialProgramStub()


def test_train_and_promote_emits_held_out_safe_rule_inventory(monkeypatch) -> None:
    artifact = LearnedArtifact(
        candidate_map=[
            CandidateMapEntry(0, 0, 0, 1, "edge", "edge", "reach"),
            CandidateMapEntry(1, 1, 1, 1, "skip", "skip", "reach"),
        ],
        logits=[2.0, -1.0],
        soft_probs=[0.88, 0.12],
        selected_hard=[0],
        discovered_rule="reach(X, Y) :- edge(X, Z), edge(Z, Y).",
    )
    fake_train_result = TrainResult(
        converged=True,
        discovered_rule=artifact.discovered_rule,
        attempt_count=1,
        total_steps=12,
        precision=1.0,
        recall=1.0,
        holdout_f1=1.0,
        artifact=artifact,
    )

    monkeypatch.setattr(promoter, "train_only", lambda *args, **kwargs: fake_train_result)
    monkeypatch.setattr(promoter.pyxlog, "IlpProgramFactory", _FactoryStub)

    result = promoter.train_and_promote(
        source="learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).",
        mask_name="W",
        positives=[("reach", [1, 3]), ("reach", [2, 4])],
        negatives=[],
        config=TrainConfig(strict_gpu_native=False, typed_schema_required=True),
        holdout_positives=[("reach", [2, 4])],
        holdout_negatives=[],
        training_fold="train-fold-0",
        held_out_domains=("cybersecurity_intrusion",),
        base_kernel_checksum_before="sha256:stable",
        base_kernel_checksum_after="sha256:stable",
    )

    assert result.status is PromotionStatus.PROMOTED
    assert result.rule_inventory is not None
    inventory = result.rule_inventory.to_dict()
    assert inventory["training_fold"] == "train-fold-0"
    assert inventory["excluded_held_out_domains"] == ["cybersecurity_intrusion"]
    assert inventory["kernel_checksums"] == {
        "before": "sha256:stable",
        "after": "sha256:stable",
    }
    assert inventory["kernel_mutated"] is False
    assert inventory["selected_clauses"][0]["rule"] == artifact.discovered_rule
    assert inventory["selected_clauses"][0]["score"] == pytest.approx(0.88)
    assert inventory["rejected_clauses"][0]["id"] == 1
