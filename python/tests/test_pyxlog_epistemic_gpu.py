"""GPU-only: accepted epistemic evidence must condition the exact probability path.

Without CUDA these tests skip. A skipped run proves nothing: the acceptance
protocol counts skip markers separately.
"""

import math

import pytest

pyxlog = pytest.importorskip("pyxlog")
pytest.importorskip("torch")

KNOWN_FACT = """
pred fact().
pred accepted().

fact().

accepted() :- know fact().
"""

PROB_SOURCE = """
0.6::fact().
query(fact()).
"""


def _compile_or_skip(source: str):
    try:
        return pyxlog.LogicProgram.compile(source)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CUDA runtime unavailable: {exc!r}")


def _first_prob(result) -> float:
    from torch.utils.dlpack import from_dlpack

    return float(from_dlpack(result.prob).cpu().reshape(-1)[0])


def test_epistemic_evidence_reports_an_accepted_world_view() -> None:
    program = _compile_or_skip(KNOWN_FACT)
    evidence = program.epistemic_evidence()

    assert evidence.epistemic_mode == "faeel"
    assert evidence.know_operator_count == 1
    assert evidence.accepted_world_views >= 1


def test_know_evidence_conditions_the_exact_query() -> None:
    program = _compile_or_skip(KNOWN_FACT)
    result = program.evaluate_conditioned(PROB_SOURCE)

    # The unconditioned program gives 0.6; conditioning on `know fact()` must
    # drive it to certainty. If this is 0.6, the broadcast did not reach the
    # circuit and the whole surface is decorative.
    assert abs(_first_prob(result) - 1.0) < 1e-9

    # log Z_E of the conditioned program is the log-evidence of what is known:
    # fact() had prior 0.6, so ln(0.6). Measured, not predicted.
    assert abs(result.log_z_e - math.log(0.6)) < 1e-9

    trace = result.trace
    assert trace["gpu_conditioned_know_evidence_facts"] == 1
    assert trace["gpu_conditioned_evidence_facts"] == 1
    assert trace["accepted_faeel_world_view_evidence_consumed"] == 1
    assert trace["gpu_knowledge_compilation_end_to_end_runs"] == 1
    assert trace["gpu_exact_query_evaluations"] == 1
    assert trace["cpu_only_probability_recomputations"] == 0
    assert trace["fixture_circuit_evaluations"] == 0


TUPLE_KEY = """
#pragma epistemic_mode = faeel

pred pair(u32, u32).
pred link(u32, u32).
pred matched(u32, u32).

pair(1, 2). pair(3, 3). pair(2, 9).
link(1, 2). link(3, 3).

matched(X, Y) :- pair(X, Y), know link(X, Y).

?- matched(X, Y).
"""

TUPLE_KEY_PROB = """
0.5::link(1, 2).
0.5::link(3, 3).
0.5::link(9, 9).
query(link(1, 2)).
query(link(3, 3)).
query(link(9, 9)).
"""


def test_only_known_atoms_are_conditioned() -> None:
    """The sharpest check: a fact outside the world view must keep its prior.

    link(1,2) and link(3,3) are known, link(9,9) is not. If all three came back
    at 1.0 the adapter would be conditioning on everything; if all three stayed
    at 0.5 it would be conditioning on nothing. Measured on GPU: 1.0, 1.0, 0.5.
    """
    program = _compile_or_skip(TUPLE_KEY)
    result = program.evaluate_conditioned(TUPLE_KEY_PROB)

    from torch.utils.dlpack import from_dlpack

    probs = [float(x) for x in from_dlpack(result.prob).cpu().reshape(-1).tolist()]
    assert len(probs) == 3
    assert abs(probs[0] - 1.0) < 1e-9
    assert abs(probs[1] - 1.0) < 1e-9
    assert abs(probs[2] - 0.5) < 1e-9

    # Two known facts at prior 0.5 each: ln(0.25) = -2 ln 2.
    assert abs(result.log_z_e - math.log(0.25)) < 1e-9
    assert result.trace["gpu_conditioned_know_evidence_facts"] == 2
    assert result.trace["cpu_only_probability_recomputations"] == 0


def test_unconditioned_baseline_differs() -> None:
    """The same probabilistic program, without evidence, must give 0.6."""
    # Only compile goes inside the try: it is what requires a live CUDA device.
    # Wrapping evaluate() here too would turn a real computation failure into
    # "SKIPPED: CUDA unavailable" on the single paid GPU run.
    try:
        plain = pyxlog.Program.compile(PROB_SOURCE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CUDA runtime unavailable: {exc!r}")
    baseline = plain.evaluate()

    from torch.utils.dlpack import from_dlpack

    value = float(from_dlpack(baseline.prob).cpu().reshape(-1)[0])
    assert abs(value - 0.6) < 1e-9


def test_ordinary_program_is_rejected() -> None:
    program = _compile_or_skip("pred a(u32).\na(1).\n?- a(X).\n")
    with pytest.raises(RuntimeError, match="epistemic"):
        program.epistemic_evidence()
