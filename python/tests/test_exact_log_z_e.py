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

# PR #180 is motivated by evidence with more than one atom, but the existing
# tests above only ever declared a single evidence(...) fact. rain/sprinkler
# and sunny are independent random variables (sunny has no rule connecting it
# to wet), so the joint evidence probability factors cleanly:
#   P(wet=true, sunny=false) = P(wet=true) * P(sunny=false)
#                             = (1 - (1-0.7)*(1-0.2)) * (1 - 0.4)
#                             = (1 - 0.3*0.8) * 0.6
#                             = 0.76 * 0.6
#                             = 0.456
WET_AND_SUNNY_WITH_TWO_ATOM_EVIDENCE = """
0.7::rain().
0.2::sprinkler().
0.4::sunny().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
evidence(sunny(), false).
query(rain()).
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


def test_exact_result_exposes_log_evidence_with_multiple_evidence_atoms():
    # rain/sprinkler and sunny are independent, so the joint evidence
    # probability factors: P(wet=true, sunny=false) = 0.76 * 0.6 = 0.456
    # (see WET_AND_SUNNY_WITH_TWO_ATOM_EVIDENCE's derivation above).
    program = pyxlog.Program.compile(WET_AND_SUNNY_WITH_TWO_ATOM_EVIDENCE)
    result = program.evaluate()

    assert result.log_z_e == pytest.approx(math.log(0.456), abs=1e-9)

    # Cross-check log_z_e against an independent computation from the query
    # probability: P(rain=true, wet=true, sunny=false)
    #   = P(rain=true) * P(wet=true | rain=true) * P(sunny=false)
    #   = 0.7 * 1.0 * 0.6 = 0.42   (wet is deterministically true given rain)
    # and P(rain=true | evidence) = P(rain=true, evidence) / P(evidence)
    #   = 0.42 / 0.456
    rain_given_evidence = torch.from_dlpack(result.prob)[0].item()
    assert rain_given_evidence == pytest.approx(0.42 / 0.456, abs=1e-9)


def test_monte_carlo_result_has_no_log_evidence():
    program = pyxlog.Program.compile(WET_WITHOUT_EVIDENCE, prob_engine="mc")
    result = program.evaluate(samples=1000)

    assert result.log_z_e is None
