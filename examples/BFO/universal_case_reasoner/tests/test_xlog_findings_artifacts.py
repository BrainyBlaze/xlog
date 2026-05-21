from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COMPONENTS = {
    "parser",
    "compiler",
    "pyxlog",
    "nn/4",
    "CUDA runtime",
    "deltas",
    "sessions",
    "cache",
    "aggregates",
    "diagnostics",
    "docs",
}
REQUIRED_SEVERITIES = {"blocker", "major", "minor", "improvement"}
REQUIRED_FIELDS = {
    "id",
    "title",
    "component",
    "severity",
    "expected_behavior",
    "actual_behavior",
    "minimal_reproducer_command",
    "evidence_artifact_path",
    "proposed_fix",
    "suggested_regression_test",
}


def test_xlog_findings_bundle_is_complete_and_reproducible() -> None:
    findings = ROOT / "XLOG_FINDINGS.md"
    ledger = ROOT / "xlog_issue_ledger.json"
    proposed_fixes = ROOT / "proposed_fixes.md"
    repro_dir = ROOT / "repro"

    assert findings.exists()
    assert proposed_fixes.exists()
    assert repro_dir.is_dir()

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    issues = payload["issues"]
    assert len(issues) >= 5
    assert payload["project"] == "BFO Universal Case Reasoner"
    assert payload["focus"] == [
        "cross-domain transfer",
        "reusable BFO kernel",
        "nn/4 CUDA ranking",
        "leave-one-domain-out generalization",
    ]

    seen_components = set()
    for issue in issues:
        missing = REQUIRED_FIELDS - set(issue)
        assert not missing, (issue.get("id"), missing)
        assert issue["component"] in REQUIRED_COMPONENTS
        assert issue["severity"] in REQUIRED_SEVERITIES
        seen_components.add(issue["component"])

        command = issue["minimal_reproducer_command"]
        assert command.startswith(("python ", "cargo run ", "xlog-cli "))
        repro_path = ROOT / issue["reproducer_path"]
        evidence_path = ROOT / issue["evidence_artifact_path"]
        assert repro_path.exists(), issue["id"]
        assert evidence_path.exists(), issue["id"]

        regression = issue["suggested_regression_test"]
        assert regression["repo_location"].startswith(("python/tests/", "crates/", "examples/"))
        assert regression["assertion"]

    assert {"nn/4", "compiler", "pyxlog", "CUDA runtime", "diagnostics"} <= seen_components

    findings_text = findings.read_text(encoding="utf-8")
    fixes_text = proposed_fixes.read_text(encoding="utf-8")
    for prompt in [
        "What did this project reveal about XLOG?",
        "What broke, flaked, or required workaround code?",
        "What XLOG feature is missing or weak?",
        "What should be fixed in core XLOG vs pyxlog vs the example?",
        "What minimal test should enter upstream?",
    ]:
        assert prompt in findings_text
    for issue in issues:
        assert issue["id"] in findings_text
        assert issue["id"] in fixes_text
