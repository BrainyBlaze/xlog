# python/tests/test_ilp_artifact.py
"""Tests for artifact save/load."""
import json
import pytest
from pathlib import Path

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig, LearnedArtifact

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]


def test_save_load_roundtrip(tmp_path):
    """Strict artifacts require explicit export before save/load roundtrip."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(SOURCE, "W_reach", POS, [], config)

    path = tmp_path / "artifact.json"
    with pytest.raises(RuntimeError, match="export_compat_artifact"):
        result.artifact.save(path)

    compat = result.export_compat_result()
    assert compat.converged
    compat.artifact.save(path)
    assert path.exists()

    loaded = LearnedArtifact.load(path)
    assert loaded.discovered_rule == compat.artifact.discovered_rule
    assert len(loaded.candidate_map) == len(compat.artifact.candidate_map)
    assert loaded.logits == pytest.approx(compat.artifact.logits, abs=1e-6)
    assert loaded.telemetry.steps == []  # telemetry not persisted


def test_save_produces_valid_json(tmp_path):
    """Explicitly exported compat artifact is valid JSON."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(SOURCE, "W_reach", POS, [], config)
    path = tmp_path / "artifact.json"
    result.export_compat_result().artifact.save(path)

    with open(path) as f:
        data = json.load(f)
    assert "discovered_rule" in data
    assert "candidate_map" in data
    assert "metadata" in data


def test_load_rejects_incompatible_hash(tmp_path):
    """Loading exported compat artifact with wrong hash raises when verify_hash=True."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(SOURCE, "W_reach", POS, [], config)
    path = tmp_path / "artifact.json"
    result.export_compat_result().artifact.save(path)

    # Corrupt the hash
    with open(path) as f:
        data = json.load(f)
    data["metadata"]["candidate_map_hash"] = "bogus"
    with open(path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="hash"):
        LearnedArtifact.load(path, verify_hash=True)


def test_config_snapshot_roundtrip(tmp_path):
    """Config snapshot survives explicit compat save/load roundtrip."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        holdout_threshold=0.9, typed_schema_required=True,
        protected_relations=("edge",),
        deterministic=True,
    )
    result = train_only(SOURCE, "W_reach", POS, [], config)

    path = tmp_path / "artifact.json"
    compat = result.export_compat_result()
    assert compat.converged
    compat.artifact.save(path)

    loaded = LearnedArtifact.load(path)
    assert loaded.config_snapshot is not None, "config_snapshot should be restored"
    assert loaded.config_snapshot == config


def test_schema_v1_loads_in_v2_code(tmp_path):
    """A beta-v1 artifact must load cleanly in v2 code."""
    import json
    v1_data = {
        "schema_version": "beta-v1",
        "discovered_rule": "reach(X,Y) :- edge(X,Z), edge(Z,Y).",
        "candidate_map": [
            {"id": 0, "i": 0, "j": 0, "k": 1,
             "left_name": "edge", "right_name": "edge", "head_name": "reach"}
        ],
        "logits": [1.5],
        "soft_probs": [0.82],
        "selected_hard": [0],
        "metadata": {
            "pyxlog_version": "0.4.0",
            "cuda_version": "12.6",
            "device_name": "test",
            "candidate_map_hash": "",
            "config_hash": "",
            "timestamp_utc": "2026-01-01T00:00:00Z",
        },
        "config_snapshot": None,
    }
    path = tmp_path / "v1_artifact.json"
    with open(path, "w") as f:
        json.dump(v1_data, f)

    from pyxlog.ilp.types import LearnedArtifact
    art = LearnedArtifact.load(path)
    assert art.discovered_rule == "reach(X,Y) :- edge(X,Z), edge(Z,Y)."
    assert len(art.candidate_map) == 1
    # Telemetry should be empty (v1 has no telemetry_snapshot)
    assert len(art.telemetry.steps) == 0


def test_schema_v2_roundtrip(tmp_path):
    """Save as v2, load back, verify fields."""
    from pyxlog.ilp.types import (
        LearnedArtifact, CandidateMapEntry, ArtifactMetadata
    )
    art = LearnedArtifact(
        candidate_map=[CandidateMapEntry(
            id=0, i=0, j=0, k=1,
            left_name="edge", right_name="edge", head_name="reach"
        )],
        logits=[2.0],
        soft_probs=[0.88],
        selected_hard=[0],
        discovered_rule="test rule",
        metadata=ArtifactMetadata(pyxlog_version="0.5.0"),
    )
    path = tmp_path / "v2_artifact.json"
    art.save(path)

    import json
    with open(path) as f:
        data = json.load(f)
    assert data["schema_version"] == "beta-v2"

    art2 = LearnedArtifact.load(path)
    assert art2.discovered_rule == "test rule"
    assert art2.logits == [2.0]


def test_telemetry_persistence_roundtrip(tmp_path):
    """Telemetry snapshot persists and restores when enabled."""
    from pyxlog.ilp.types import (
        LearnedArtifact, CandidateMapEntry, ArtifactMetadata,
        TrainConfig, TrainTelemetry, StepRecord,
    )
    config = TrainConfig(persist_telemetry=True, telemetry_persist_limit=5)
    steps = [
        StepRecord(step=i, loss=1.0 / (i + 1), argmax_rule="r",
                   discreteness=0.9, temperature=1.0, entropy=0.5,
                   stable_count=i)
        for i in range(10)
    ]
    art = LearnedArtifact(
        candidate_map=[CandidateMapEntry(
            id=0, i=0, j=0, k=1,
            left_name="e", right_name="e", head_name="r"
        )],
        discovered_rule="test",
        config_snapshot=config,
        telemetry=TrainTelemetry(steps=steps, step_timings={"total": 1.5}),
        metadata=ArtifactMetadata(pyxlog_version="0.5.0"),
    )
    path = tmp_path / "telem_artifact.json"
    art.save(path)

    art2 = LearnedArtifact.load(path)
    # Should have last 5 steps only (truncated by persist_limit)
    assert len(art2.telemetry.steps) == 5
    assert art2.telemetry.steps[0].step == 5  # last 5 of 0..9
    assert art2.telemetry.step_timings == {"total": 1.5}


def test_telemetry_not_persisted_by_default(tmp_path):
    """Telemetry is NOT saved when persist_telemetry=False (default)."""
    from pyxlog.ilp.types import (
        LearnedArtifact, CandidateMapEntry, ArtifactMetadata,
        TrainConfig, TrainTelemetry, StepRecord,
    )
    config = TrainConfig()  # persist_telemetry defaults to False
    steps = [StepRecord(step=0, loss=1.0, argmax_rule="r",
                        discreteness=0.9, temperature=1.0, entropy=0.5,
                        stable_count=0)]
    art = LearnedArtifact(
        candidate_map=[CandidateMapEntry(
            id=0, i=0, j=0, k=1,
            left_name="e", right_name="e", head_name="r"
        )],
        discovered_rule="test",
        config_snapshot=config,
        telemetry=TrainTelemetry(steps=steps),
        metadata=ArtifactMetadata(pyxlog_version="0.5.0"),
    )
    path = tmp_path / "no_telem.json"
    art.save(path)

    import json
    with open(path) as f:
        data = json.load(f)
    assert data.get("telemetry_snapshot") is None

    art2 = LearnedArtifact.load(path)
    assert len(art2.telemetry.steps) == 0


def test_telemetry_persist_limit_truncation(tmp_path):
    """persist_limit truncates to last N steps."""
    from pyxlog.ilp.types import (
        LearnedArtifact, CandidateMapEntry, ArtifactMetadata,
        TrainConfig, TrainTelemetry, StepRecord,
    )
    config = TrainConfig(persist_telemetry=True, telemetry_persist_limit=3)
    steps = [
        StepRecord(step=i, loss=float(i), argmax_rule="r",
                   discreteness=0.9, temperature=1.0, entropy=0.5,
                   stable_count=0)
        for i in range(100)
    ]
    art = LearnedArtifact(
        candidate_map=[CandidateMapEntry(
            id=0, i=0, j=0, k=1,
            left_name="e", right_name="e", head_name="r"
        )],
        discovered_rule="test",
        config_snapshot=config,
        telemetry=TrainTelemetry(steps=steps),
        metadata=ArtifactMetadata(pyxlog_version="0.5.0"),
    )
    path = tmp_path / "truncated.json"
    art.save(path)

    art2 = LearnedArtifact.load(path)
    assert len(art2.telemetry.steps) == 3
    assert art2.telemetry.steps[0].step == 97  # last 3 of 0..99
