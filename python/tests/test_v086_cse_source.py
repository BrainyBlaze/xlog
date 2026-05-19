from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "2026-05-19-v086-cse"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v086_cse_runtime_uses_existing_executor_path() -> None:
    config = read("crates/xlog-core/src/config.rs")
    executor = read("crates/xlog-runtime/src/executor/mod.rs")
    dispatch = read("crates/xlog-runtime/src/executor/node_dispatch.rs")
    optimizer_doc = read("docs/architecture/query-optimizer.md")

    assert "XLOG_CSE" in config
    assert "with_common_subexpression_elimination" in config
    assert "common_subexpression_cache: HashMap<CommonSubexpressionKey, CudaBuffer>" in executor
    assert "CommonSubexpressionStats" in executor
    assert "generation" in executor
    assert "negation_or_difference_boundary" in executor
    assert "aggregate_boundary" in executor
    assert "provenance_or_tensor_boundary" in executor
    assert "specialized_dispatch_boundary" in executor
    assert "execute_node_uncached" in dispatch
    assert "common_subexpression_cache.insert" in dispatch
    assert "Common Subexpression Elimination" in optimizer_doc


def test_v086_cse_evidence_is_present() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    assert "G086_CSE" in readme
    assert "M086_CSE.1" in readme
    assert "M086_CSE.6" in readme
    assert measurements["deterministic_fixture"]["output_parity"] is True
    assert measurements["deterministic_fixture"]["duplicate_subplan_reduction_percent"] >= 30.0
    assert measurements["deterministic_fixture"]["added_dtoh_calls"] == 0
    assert measurements["unsafe_rejections"]["aggregate_boundary"] is True
    assert measurements["unsafe_rejections"]["negation_or_difference_boundary"] is True
    assert measurements["unsafe_rejections"]["provenance_or_tensor_boundary"] is True
    assert measurements["unsafe_rejections"]["specialized_dispatch_boundary"] is True
