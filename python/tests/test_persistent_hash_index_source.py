from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def persistent_hash_index_evidence_dir() -> Path:
    matches = []
    for path in sorted((ROOT / "docs" / "evidence").iterdir()):
        measurements_path = path / "measurements.json"
        if not measurements_path.exists() or not (path / "README.md").exists():
            continue
        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        repeated = measurements.get("repeated_session_fixture", {})
        performance = measurements.get("performance_fixture", {})
        transfer_budget = performance.get("transfer_budget", {})
        if (
            measurements.get("manager_key", {}).get("includes_relation_id") is True
            and measurements.get("invalidation_fixture", {}).get("entries_after_mutation") == 0
            and measurements.get("budget_fixture", {}).get("evictions") == 1
            and measurements.get("background_fixture", {}).get("background_build_requests") == 1
            and measurements.get("background_fixture", {}).get("background_builds_deferred") == 1
            and repeated.get("builds") == 1
            and repeated.get("hits", 0) >= 1
            and repeated.get("tracked_dtoh_calls") == 0
            and repeated.get("tracked_htod_calls") == 0
            and performance.get("speedup_ratio", 0) >= performance.get("target_speedup_ratio", 1.5)
            and transfer_budget.get("cached_tracked_dtoh_calls") == 0
            and transfer_budget.get("cached_tracked_htod_calls") == 0
            and measurements.get("performance_or_blocker", {}).get("speedup_gate_claimed") is True
        ):
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


EVIDENCE_DIR = persistent_hash_index_evidence_dir()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_persistent_hash_index_extends_existing_cache_path() -> None:
    config = read("crates/xlog-core/src/config.rs")
    cache = read("crates/xlog-runtime/src/executor/join_cache.rs")
    executor = read("crates/xlog-runtime/src/executor/mod.rs")
    dispatch = read("crates/xlog-runtime/src/executor/node_dispatch.rs")
    docs = read("docs/architecture/adaptive-indexing.md")

    assert "XLOG_PERSISTENT_HASH_INDEXES" in config
    assert "with_persistent_hash_indexes" in config
    assert "with_persistent_hash_index_background_build" in config
    assert "JoinIndexSchemaSignature" in cache
    assert "device_ordinal" in cache
    assert "JoinIndexCacheStats" in cache
    assert "stale_rejections" in cache
    assert "background_build_requests" in cache
    assert "join_index_cache_stats" in executor
    assert "JoinIndexKey::new" in dispatch
    assert "resolved_persistent_hash_indexes" in dispatch
    assert "record_background_build_request" in dispatch
    assert "record_background_build_deferred" in dispatch
    assert "Persistent Hash Index Manager" in docs


def test_persistent_hash_index_background_builds_use_recorded_provider_path() -> None:
    cache = read("crates/xlog-runtime/src/executor/join_cache.rs")
    dispatch = read("crates/xlog-runtime/src/executor/node_dispatch.rs")
    provider = read("crates/xlog-cuda/src/provider/relational.rs")

    assert "background_builds_deferred" in cache
    assert "build_join_index_v2_background" in dispatch
    assert "build_join_index_v2_background" in provider
    assert "build_join_index_v2_recorded" in provider
    assert "recorded_op_stream_or_init" in provider
    assert "pack_keys_gpu_on_stream" in provider
    assert "build_hash_table_v2_on_stream" in provider


def test_persistent_hash_index_evidence_is_present() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    assert "Persistent Hash Index Manager" in readme
    assert "manager API" in readme
    assert "transfer budget" in readme
    assert measurements["repeated_session_fixture"]["builds"] == 1
    assert measurements["repeated_session_fixture"]["hits"] >= 1
    assert measurements["repeated_session_fixture"]["tracked_dtoh_calls"] == 0
    assert measurements["repeated_session_fixture"]["tracked_htod_calls"] == 0
    assert measurements["invalidation_fixture"]["entries_after_mutation"] == 0
    assert measurements["budget_fixture"]["evictions"] == 1
    assert measurements["background_fixture"]["background_build_requests"] == 1
    assert measurements["background_fixture"]["background_builds_deferred"] == 1


def test_persistent_hash_index_records_required_speedup_fixture() -> None:
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    performance = measurements["performance_fixture"]
    assert performance["speedup_ratio"] >= 1.5
    assert measurements["performance_or_blocker"]["speedup_gate_claimed"] is True
    assert performance["cached"]["median_seconds"] < performance["uncached"]["median_seconds"]
    assert performance["cached"]["output_rows"] == performance["uncached"]["output_rows"]
    assert performance["transfer_budget"]["cached_tracked_dtoh_calls"] == 0
    assert performance["transfer_budget"]["cached_tracked_htod_calls"] == 0


def test_persistent_hash_index_pyxlog_session_reuse_is_exposed() -> None:
    pyxlog_logic = read("crates/pyxlog/src/logic.rs")
    pyxlog_lib = read("crates/pyxlog/src/lib.rs")
    gpu_logic = read("crates/xlog-gpu/src/logic.rs")
    stub = read("crates/pyxlog/python/pyxlog/_native.pyi")
    docs = read("docs/architecture/python-bindings.md")

    assert "LogicSessionRuntime" in gpu_logic
    assert "apply_relation_deltas_with_session_runtime" in gpu_logic
    assert "join_index_cache_stats" in pyxlog_logic
    assert "session_runtime" in pyxlog_lib
    assert "def join_index_cache_stats(self) -> dict[str, int]:" in stub
    assert "join_index_cache_stats()" in docs
