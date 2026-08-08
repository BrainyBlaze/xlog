from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "docs-internal/evidence"

# EVIDENCE-GATED: only the evidence-package test below needs this tree; the
# other tests in this module are pure source-contract checks and must keep
# running on a clean clone, so the gate is per-test, not a module-level
# `pytestmark`.
_EVIDENCE_SKIP_REASON = (
    "no local evidence workspace at docs-internal/evidence -- that tree is a "
    "local-only agent workspace excluded by .gitignore (the '/docs-internal/' "
    "entry, labelled 'Local-only agent workspaces'), so it is deliberately not "
    "shipped and is absent on a clean clone. When it IS present this test "
    "checks that some 'docs-internal/evidence/*-bridge' directory holds a "
    "README.md naming LearnedBridge next to a runtime_probe.json, and that the "
    "README records the gradient smoke, Belnap helper, deterministic top-k, "
    "neural cache telemetry and repeated-query speedup results. Re-run the "
    "bridge evidence job locally to recreate the package and exercise it."
)


def test_bridge_public_surface_is_stubbed_and_documented() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "python/tests/contract_docs/python-bindings.md").read_text()

    for needle in [
        "def deterministic_topk(",
        "def neural_cache_stats(",
        "def belnap_loss(",
        "def semantic_loss_tensor(",
        "def mse_loss_tensor(",
        "def infoloss_tensor(",
    ]:
        assert needle in native_stub
        assert needle.split("(")[0].removeprefix("def ") in docs


def test_bridge_native_helpers_keep_semantics_in_python_ml_layer() -> None:
    neural_rs = (ROOT / "crates/pyxlog/src/neural.rs").read_text()
    lib_rs = (ROOT / "crates/pyxlog/src/lib.rs").read_text()
    stage4_sources = [
        (ROOT / "crates/xlog-runtime/src/executor/rewrite.rs").read_text(),
        (ROOT / "crates/xlog-gpu/src/logic.rs").read_text(),
    ]

    for needle in [
        "belnap_loss",
        "semantic_loss_tensor",
        "mse_loss_tensor",
        "infoloss_tensor",
        "deterministic_topk",
        "neural_cache_stats",
        "circuit_cache_hits",
        "circuit_cache_misses",
    ]:
        assert needle in neural_rs or needle in lib_rs

    for source in stage4_sources:
        assert "belnap_loss" not in source
        assert "contra_penalty" not in source


def test_bridge_reuses_registered_network_output_modes() -> None:
    neural_rs = (ROOT / "crates/pyxlog/src/neural.rs").read_text()

    assert "NetworkHandle" in neural_rs
    assert "fn apply_network_output_mode(" in neural_rs
    assert "if k == Some(0)" in neural_rs
    assert "handle.det { Some(1) } else { handle.k }" in neural_rs

    for fn_name in [
        "fn forward_backward_direct_tensor",
        "fn forward_backward_complex_tensor",
        "fn forward_backward_batch_complex_tensor",
    ]:
        start = neural_rs.index(fn_name)
        end = neural_rs.find("\n    fn ", start + 1)
        if end == -1:
            end = len(neural_rs)
        body = neural_rs[start:end]
        assert "apply_network_output_mode(py" in body


@pytest.mark.skipif(not EVIDENCE_ROOT.is_dir(), reason=_EVIDENCE_SKIP_REASON)
def test_bridge_has_evidence_package() -> None:
    evidence = next(
        (
            path / "README.md"
            for path in sorted(EVIDENCE_ROOT.glob("*-bridge"))
            if (path / "README.md").exists()
            and "LearnedBridge" in (path / "README.md").read_text()
        ),
        None,
    )
    assert evidence is not None, (
        "docs-internal/evidence exists but holds no '*-bridge' directory with a "
        "README.md mentioning LearnedBridge -- the bridge evidence package is "
        "missing or incomplete in this local workspace"
    )
    probe = evidence.parent / "runtime_probe.json"

    assert evidence.exists()
    assert probe.exists()

    text = evidence.read_text()
    for needle in [
        "gradient smoke",
        "LearnedBridge-shaped",
        "Belnap helper",
        "deterministic top-k",
        "neural cache telemetry",
        "repeated-query speedup",
        "LearnedBridge",
    ]:
        assert needle in text
