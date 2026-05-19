from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v080_bridge_public_surface_is_stubbed_and_documented() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()

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


def test_v080_bridge_native_helpers_keep_semantics_in_python_ml_layer() -> None:
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


def test_v080_bridge_reuses_registered_network_output_modes() -> None:
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


def test_v080_bridge_has_evidence_package() -> None:
    evidence = ROOT / "docs/evidence/2026-05-18-v080-bridge/README.md"
    probe = ROOT / "docs/evidence/2026-05-18-v080-bridge/runtime_probe.json"

    assert evidence.exists()
    assert probe.exists()

    text = evidence.read_text()
    for needle in [
        "M080_BRIDGE.1",
        "M080_BRIDGE.2",
        "M080_BRIDGE.3",
        "M080_BRIDGE.4",
        "M080_BRIDGE.5",
        "M080_BRIDGE.6",
        "LearnedBridge",
    ]:
        assert needle in text
