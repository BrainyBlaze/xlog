from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def adaptive_reoptimization_evidence_dir() -> Path:
    matches = []
    for path in sorted((ROOT / "docs-internal" / "evidence").glob("*adaptive-reoptimization")):
        measurements_path = path / "measurements.json"
        if not measurements_path.exists() or not (path / "README.md").exists():
            continue
        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        if (
            measurements.get("deterministic_fixture", {}).get(
                "adapted_output_matches_baseline"
            )
            is True
            and measurements.get("rollback_fixture", {}).get("rolled_back") == 1
            and measurements.get("transfer_budget", {}).get(
                "tracked_data_plane_dtoh_calls"
            )
            == 0
        ):
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


EVIDENCE_DIR = adaptive_reoptimization_evidence_dir()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_adaptive_reoptimization_reuses_executor_and_stats_paths() -> None:
    config = read("crates/xlog-core/src/config.rs")
    executor = read("crates/xlog-runtime/src/executor/mod.rs")
    dispatch = read("crates/xlog-runtime/src/executor/node_dispatch.rs")
    optimizer_doc = read("python/tests/contract_docs/query-optimizer.md")

    assert "XLOG_ADAPTIVE_REOPT" in config
    assert "with_adaptive_reoptimization" in config
    assert "resolved_adaptive_reoptimization_min_misplan_ratio" in config
    assert "AdaptiveReoptimizationStats" in executor
    assert "execute_plan_with_adaptive_candidate" in executor
    assert "adaptive_reoptimization_decision" in executor
    assert "restore_store_snapshot" in executor
    assert "restore_stats_snapshot" in executor
    assert "diff_full_row" in executor
    assert "record_adaptive_dtoh_delta" in executor
    assert "download_column" not in executor.split("fn buffers_gpu_set_equivalent", 1)[1].split(
        "fn record_adaptive_dtoh_delta", 1
    )[0]
    assert "record_adaptive_join_observation" in dispatch
    assert "Adaptive Runtime Re-Optimization" in optimizer_doc


def test_adaptive_reoptimization_evidence_is_present() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    assert "Adaptive Runtime Re-Optimization" in readme
    assert "deterministic decisions" in readme
    assert "no tracked" in readme
    assert measurements["deterministic_fixture"]["adapted_output_matches_baseline"] is True
    assert measurements["deterministic_fixture"]["adopted"] == 1
    assert measurements["deterministic_fixture"]["decision_replays"] == 100
    assert measurements["deterministic_fixture"]["data_plane_dtoh_calls"] == 0
    assert measurements["rollback_fixture"]["rolled_back"] == 1
    assert measurements["rollback_fixture"]["candidate_output_mismatch_diagnostic"] is True
    assert measurements["performance_or_blocker"]["correctness_blocker_removed"] is True
