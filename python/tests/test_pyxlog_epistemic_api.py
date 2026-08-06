from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# NOTE: the task plan named `docs/architecture/python-bindings.md` as the doc
# to extend. That path was retired from git tracking in c872ef23 ("chore(docs):
# retire the internal documentation tree from tracking"), well before this
# feature branch existed, and does not exist anywhere in this checkout. The
# live, currently-maintained pyxlog Python reference is
# `docs/reference/python.mdx`; this test (and the documentation change it
# guards) targets that file instead.
DOCS_PATH = ROOT / "docs/reference/python.mdx"


def test_xlog_pyxlog_004_epistemic_conditioned_api() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    init_stub = (ROOT / "crates/pyxlog/python/pyxlog/__init__.pyi").read_text()
    docs = DOCS_PATH.read_text()

    # `_native.pyi` is the sole authoritative declaration site, so the full list is
    # checked there and in the live reference.
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

    # `__init__.pyi` only re-exports names from `_native`; it carries no method
    # signatures, and asserting method names there would only ever check a comment.
    for needle in ["EpistemicEvalResult", "EpistemicEvidence"]:
        assert needle in init_stub, f"{needle} missing from __init__.pyi"
