from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "2026-05-19-v086-persistent-hash-index"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v086_persistent_hash_index_extends_existing_cache_path() -> None:
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


def test_v086_persistent_hash_index_background_builds_use_recorded_provider_path() -> None:
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


def test_v086_persistent_hash_index_evidence_is_present() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    assert "G086_INDEX" in readme
    assert "M086_INDEX.1" in readme
    assert "M086_INDEX.6" in readme
    assert measurements["repeated_session_fixture"]["builds"] == 1
    assert measurements["repeated_session_fixture"]["hits"] >= 1
    assert measurements["repeated_session_fixture"]["tracked_dtoh_calls"] == 0
    assert measurements["repeated_session_fixture"]["tracked_htod_calls"] == 0
    assert measurements["invalidation_fixture"]["entries_after_mutation"] == 0
    assert measurements["budget_fixture"]["evictions"] == 1
    assert measurements["background_fixture"]["background_build_requests"] == 1
    assert measurements["background_fixture"]["background_builds_deferred"] == 1


def test_v086_persistent_hash_index_records_required_speedup_fixture() -> None:
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    performance = measurements["performance_fixture"]
    assert performance["speedup_ratio"] >= 1.5
    assert measurements["performance_or_blocker"]["speedup_gate_claimed"] is True
    assert performance["cached"]["median_seconds"] < performance["uncached"]["median_seconds"]
    assert performance["cached"]["output_rows"] == performance["uncached"]["output_rows"]
    assert performance["transfer_budget"]["cached_tracked_dtoh_calls"] == 0
    assert performance["transfer_budget"]["cached_tracked_htod_calls"] == 0
