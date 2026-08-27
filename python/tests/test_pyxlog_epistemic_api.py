import inspect

import pytest

def _named_parameters(signature: inspect.Signature) -> list[str]:
    """Parameter names of an unbound native method, without the receiver.

    `inspect.signature` on a class-level method descriptor keeps the `$self`
    slot from the generated `__text_signature__`; drop it so the assertion is
    about the Python-visible arguments only.
    """
    names = list(signature.parameters)
    if names and names[0] == "self":
        names = names[1:]
    return names


def test_epistemic_methods_and_result_types_bind_to_native_module() -> None:
    """Bind the public contract to the compiled extension without CUDA."""
    native = pytest.importorskip("pyxlog._native")

    for method_name in ("evaluate_conditioned", "epistemic_evidence"):
        descriptor = vars(native.CompiledLogicProgram)[method_name]
        assert inspect.ismethoddescriptor(descriptor)
        assert descriptor.__objclass__ is native.CompiledLogicProgram

    # Existence alone would stay green through a signature drift between the stub
    # and the native module -- the exact class of bug the stub exists to prevent.
    conditioned = inspect.signature(native.CompiledLogicProgram.evaluate_conditioned)
    assert _named_parameters(conditioned) == ["prob_source", "memory_mb"]
    assert conditioned.parameters["memory_mb"].default is None
    # The diagnostic counterpart takes no arguments at all.
    evidence_sig = inspect.signature(native.CompiledLogicProgram.epistemic_evidence)
    assert _named_parameters(evidence_sig) == []

    for class_name in ("EpistemicEvalResult", "EpistemicEvidence"):
        assert inspect.isclass(getattr(native, class_name))

    # The result types are the documented contract; assert their attributes exist
    # so a dropped or renamed getter fails here rather than at a user's call site.
    for attr in ("atoms", "prob", "log_prob", "log_z_e", "trace"):
        assert attr in vars(
            native.EpistemicEvalResult
        ), f"{attr} missing from EpistemicEvalResult"
    for attr in (
        "epistemic_mode",
        "know_operator_count",
        "possible_operator_count",
        "accepted_candidates",
        "rejected_candidates",
        "accepted_world_views",
        "final_output_rows",
    ):
        assert attr in vars(
            native.EpistemicEvidence
        ), f"{attr} missing from EpistemicEvidence"
