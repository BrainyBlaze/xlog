import inspect
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

# `docs/architecture/python-bindings.md` was retired from git tracking in
# c872ef23 ("chore(docs): retire the internal documentation tree from
# tracking") and does not exist anywhere in this checkout. The live,
# currently-maintained pyxlog Python reference is `docs/reference/python.mdx`.
# `4a42bf35` separately moved the tested contract documents that back the
# public API guard tests to `python/tests/contract_docs/`; this test targets
# both of those live surfaces instead of the retired path.
DOCS_PATH = ROOT / "docs/reference/python.mdx"
CONTRACT_DOCS_PATH = ROOT / "python/tests/contract_docs/python-bindings.md"


def test_xlog_pyxlog_004_epistemic_conditioned_api() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    init_stub = (ROOT / "crates/pyxlog/python/pyxlog/__init__.pyi").read_text()
    docs = DOCS_PATH.read_text()
    contract_docs = CONTRACT_DOCS_PATH.read_text()

    # `_native.pyi` is the sole authoritative declaration site, so the full list is
    # checked there and in both maintained documentation surfaces.
    for needle in [
        "EpistemicEvalResult",
        "EpistemicEvidence",
        "evaluate_conditioned",
        "epistemic_evidence",
        "log_z_e",
        "gpu_conditioned_know_evidence_facts",
        "cpu_only_probability_recomputations",
    ]:
        assert needle in native_stub, f"{needle} missing from _native.pyi"
        assert needle in docs, f"{needle} missing from {DOCS_PATH}"
        assert needle in contract_docs, f"{needle} missing from {CONTRACT_DOCS_PATH}"

    # `__init__.pyi` only re-exports names from `_native`; it carries no method
    # signatures, and asserting method names there would only ever check a comment.
    for needle in ["EpistemicEvalResult", "EpistemicEvidence"]:
        assert needle in init_stub, f"{needle} missing from __init__.pyi"


def test_epistemic_methods_and_result_types_bind_to_native_module() -> None:
    """Text-guard tests above only check substrings in stub/doc files: renaming
    `evaluate_conditioned` in Rust would leave them green. This binds the
    declared surface to the actual compiled extension, mirroring the idiom
    `test_relation_provenance_public_api.py` uses for `LogicRelationSession`
    (`vars(cls)[method_name]` + `inspect.ismethoddescriptor`). It needs the
    built wheel but not CUDA, and degrades gracefully (skips) when the native
    module is absent, same as this repo's other no-GPU native-binding checks.
    """
    native = pytest.importorskip("pyxlog._native")

    for method_name in ("evaluate_conditioned", "epistemic_evidence"):
        descriptor = vars(native.CompiledLogicProgram)[method_name]
        assert inspect.ismethoddescriptor(descriptor)
        assert descriptor.__objclass__ is native.CompiledLogicProgram

    for class_name in ("EpistemicEvalResult", "EpistemicEvidence"):
        assert inspect.isclass(getattr(native, class_name))
