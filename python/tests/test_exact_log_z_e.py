"""EvalResult must expose the exact log-evidence log Z_E."""

import math

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

WET_WITH_EVIDENCE = """
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
query(rain()).
"""

WET_WITHOUT_EVIDENCE = """
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
query(wet()).
"""


def test_exact_result_exposes_log_evidence():
    # P(wet) = 1 - 0.3 * 0.8 = 0.76
    program = pyxlog.Program.compile(WET_WITH_EVIDENCE)
    result = program.evaluate()

    assert result.log_z_e == pytest.approx(math.log(0.76), abs=1e-9)


def test_log_evidence_is_zero_when_nothing_is_observed():
    program = pyxlog.Program.compile(WET_WITHOUT_EVIDENCE)
    result = program.evaluate()

    assert result.log_z_e == pytest.approx(0.0, abs=1e-9)

    # log_z_e alone is satisfiable by a stub that always returns Some(0.0);
    # without evidence, evaluate() must also still run the real WMC path
    # and report the correct query probability, not a hardwired 0.0.
    # P(wet) = 1 - 0.3 * 0.8 = 0.76
    wet_prob = torch.from_dlpack(result.prob)[0].item()
    assert wet_prob == pytest.approx(0.76, abs=1e-6)


def test_log_evidence_survives_gradient_mode():
    program = pyxlog.Program.compile(WET_WITH_EVIDENCE)
    result = program.evaluate(return_grads=True)

    assert result.log_z_e == pytest.approx(math.log(0.76), abs=1e-9)


def test_monte_carlo_result_has_no_log_evidence():
    program = pyxlog.Program.compile(WET_WITHOUT_EVIDENCE, prob_engine="mc")
    result = program.evaluate(samples=1000)

    assert result.log_z_e is None
