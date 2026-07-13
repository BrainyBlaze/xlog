from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def chain_shared_memory_profile_evidence_dir() -> Path:
    matches = []
    for path in sorted((ROOT / "docs-internal" / "evidence").iterdir()):
        measurements_path = path / "measurements.json"
        if not measurements_path.exists() or not (path / "README.md").exists():
            continue
        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        if (
            measurements.get("profile_trigger_pass") is True
            and measurements.get("hot_to_small_median_ratio", 0.0) >= 20.0
            and measurements.get("chain_hot", {}).get("dtoh_calls") == 2
        ):
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


EVIDENCE_DIR = chain_shared_memory_profile_evidence_dir()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_chain_shared_memory_profile_script_uses_native_exact_path() -> None:
    script = read("scripts/measure_chain_shared_memory_profile.py")

    assert "induce_exact(prog, backend=\"native\"" in script
    assert "first query argument appears in every left row" in script
    assert "XLOG_ILP_EXACT_CHAIN_SMEM" in script
    assert "speedup_ratio" in script
    assert "regression_percent" in script
    assert "prog.d2h_transfer_count() != 2" in script


def test_chain_shared_memory_profile_trigger_evidence_is_present() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    assert "profile trigger" in readme
    assert "native exact-induction path" in readme
    assert "hot/small median ratio" in readme
    assert measurements["chain_hot"]["dtoh_calls"] == 2
    assert measurements["small"]["dtoh_calls"] == 2
    assert measurements["hot_to_small_median_ratio"] >= 20.0
