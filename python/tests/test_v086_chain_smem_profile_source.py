from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "2026-05-19-v086-chain-smem-profile"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_chain_smem_profile_script_uses_native_exact_path() -> None:
    script = read("scripts/measure_v086_chain_smem.py")

    assert "induce_exact(prog, backend=\"native\"" in script
    assert "qx=1 appears in every left row" in script
    assert "hot_to_small_median_ratio" in script
    assert "prog.d2h_transfer_count() != 2" in script


def test_chain_smem_profile_trigger_evidence_is_present() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    assert "G086_CHAIN_SMEM" in readme
    assert "PROFILE_TRIGGER_ONLY" in readme
    assert "M086_CHAIN.1" in readme
    assert "M086_CHAIN.2" in readme
    assert measurements["chain_hot"]["dtoh_calls"] == 2
    assert measurements["small"]["dtoh_calls"] == 2
    assert measurements["hot_to_small_median_ratio"] >= 20.0
