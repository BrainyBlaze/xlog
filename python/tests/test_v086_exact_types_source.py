from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "2026-05-19-v086-exact-types"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_exact_type_dispatch_uses_existing_native_engine() -> None:
    provider = read("crates/xlog-cuda/src/provider/ilp_exact.rs")
    induce = read("crates/xlog-induce/src/lib.rs")
    kernels = read("crates/xlog-cuda/kernels/ilp_exact.cu")
    manifest = read("crates/xlog-cuda/src/kernel_manifest_data.rs")
    names = read("crates/xlog-cuda/src/provider/mod.rs")
    py_bridge = read("crates/pyxlog/src/ilp_exact.rs")

    assert "ExactPairLayout" in provider
    assert "ScalarType::U32 => ExactPairLayout::U32" in provider
    assert "ScalarType::Symbol => ExactPairLayout::Symbol" in provider
    assert "column_as_u32_view" in provider
    assert "ILP_EXACT_SCORE_U32" in provider
    assert "download_column" not in provider

    assert "ExactPairType" in induce
    assert "ScalarType::U32 => ExactPairType::U32" in induce
    assert "ScalarType::Symbol => ExactPairType::Symbol" in induce
    assert "provider.ilp_exact_score" in induce

    assert "ilp_exact_matches_t" in kernels
    assert "ilp_exact_score_u32" in kernels
    assert '"ilp_exact_score_u32"' in manifest
    assert 'pub const ILP_EXACT_SCORE_U32: &str = "ilp_exact_score_u32";' in names
    assert "induce_exact_engine(&self.provider, &request)" in py_bridge


def test_exact_type_dispatch_preserves_symbol_schema_at_dlpack_boundary() -> None:
    core_types = read("crates/xlog-core/src/types.rs")
    dlpack = read("crates/xlog-cuda/src/dlpack.rs")

    assert "(ScalarType::Symbol, ScalarType::U32)" in core_types
    assert "(ScalarType::Symbol, ScalarType::I32)" in core_types
    assert "!ScalarType::U32.dlpack_compatible(ScalarType::Symbol)" in core_types
    assert "from_dlpack_tensors_with_schema" in dlpack
    assert "self.buffer_from_columns(columns, num_rows.unwrap_or(0), schema)" in dlpack


def test_exact_type_dispatch_evidence_is_present() -> None:
    readme = (EVIDENCE_DIR / "README.md").read_text(encoding="utf-8")
    measurements = (EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8")

    assert "G086_EXACT_TYPES" in readme
    assert "M086_EXACT_TYPES.1" in readme
    assert "M086_EXACT_TYPES.2" in readme
    assert "M086_EXACT_TYPES.6" in readme
    assert '"u32"' in measurements
    assert '"symbol"' in measurements
    assert '"dtoh_calls": 2' in measurements
