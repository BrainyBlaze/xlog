from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def chain_shared_memory_evidence_dir() -> Path:
    matches = []
    for path in sorted((ROOT / "docs" / "evidence").iterdir()):
        measurements_path = path / "measurements.json"
        if not measurements_path.exists() or not (path / "README.md").exists():
            continue
        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        chain_hot = measurements.get("chain_hot", {})
        transfer_budget = measurements.get("transfer_budget", {})
        if (
            chain_hot.get("parity") is True
            and chain_hot.get("speedup_ratio", 0.0) >= 1.2
            and transfer_budget.get("added_dtoh_calls") == 0
        ):
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


EVIDENCE_DIR = chain_shared_memory_evidence_dir()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_chain_shared_memory_kernel_and_provider_contracts() -> None:
    kernels = read("crates/xlog-cuda/kernels/ilp_exact.cu")
    provider = read("crates/xlog-cuda/src/provider/ilp_exact.rs")
    names = read("crates/xlog-cuda/src/provider/mod.rs")
    manifest = read("crates/xlog-cuda/src/kernel_manifest_data.rs")

    assert "ilp_exact_count_chain_smem" in kernels
    assert "extern __shared__" in kernels
    assert "ilp_exact_score_chain_smem" in kernels
    assert "ilp_exact_score_chain_smem_u32" in kernels

    assert "XLOG_ILP_EXACT_CHAIN_SMEM" in provider
    assert "ilp_exact_chain_smem_enabled" in provider
    assert "chain_smem_shared_bytes" in provider
    assert "ILP_EXACT_SCORE_CHAIN_SMEM" in provider
    assert "ILP_EXACT_SCORE_CHAIN_SMEM_U32" in provider

    assert 'pub const ILP_EXACT_SCORE_CHAIN_SMEM: &str = "ilp_exact_score_chain_smem";' in names
    assert 'pub const ILP_EXACT_SCORE_CHAIN_SMEM_U32: &str = "ilp_exact_score_chain_smem_u32";' in names
    assert '"ilp_exact_score_chain_smem"' in manifest
    assert '"ilp_exact_score_chain_smem_u32"' in manifest


def test_chain_shared_memory_evidence_is_present() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    assert "shared-memory kernel preserves strict" in readme
    assert "median runtime improves" in readme
    assert "no added" in readme
    assert measurements["chain_hot"]["parity"] is True
    assert measurements["chain_hot"]["speedup_ratio"] >= 1.2
    assert measurements["chain_hot"]["baseline"]["result_signature"]
    assert measurements["chain_hot"]["baseline"]["result_signature"] == (
        next(
            value
            for key, value in measurements["chain_hot"].items()
            if key not in {"baseline", "parity", "speedup_ratio"}
        )["result_signature"]
    )
    assert measurements["small"]["regression_percent"] <= 5.0
    assert measurements["fallback"]["non_chain_uses_baseline_logic"] is True
    assert measurements["transfer_budget"]["added_dtoh_calls"] == 0
