from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def exact_types_evidence_dir() -> Path:
    matches = []
    for path in sorted((ROOT / "docs-internal" / "evidence").iterdir()):
        measurements_path = path / "measurements.json"
        if not measurements_path.exists() or not (path / "README.md").exists():
            continue
        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        if (
            measurements.get("provider_typed_tests_passed") == 7
            and measurements.get("core_dlpack_compatibility_tests_passed") == 1
            and "symbol" in measurements
        ):
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


EVIDENCE_DIR = exact_types_evidence_dir()


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
    measurements = json.loads((EVIDENCE_DIR / "measurements.json").read_text(encoding="utf-8"))

    assert "Native Exact-Induction U32/Symbol Dispatch" in readme
    assert "schema identity is preserved" in readme
    assert "count-array" in readme
    assert measurements["u32"]["3"]["parity"] is True
    assert measurements["symbol"]["3"]["parity"] is True
    assert measurements["u32"]["3"]["dtoh_calls"] == 2
