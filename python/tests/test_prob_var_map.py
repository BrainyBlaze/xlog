"""CompiledProgram must expose which fact each CNF variable stands for."""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

TWO_FACTS = """
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
query(wet()).
"""

WITH_DISJUNCTION = """
0.6::ctx(left); 0.4::ctx(right).
0.9::flag().
seen() :- ctx(left), flag().
query(seen()).
"""


def test_map_has_one_entry_per_cnf_variable():
    program = pyxlog.Program.compile(TWO_FACTS)
    result = program.evaluate()
    var_map = program.prob_var_map()

    assert len(var_map) == result.num_vars


def test_simple_probabilistic_facts_are_named_with_their_probability():
    program = pyxlog.Program.compile(TWO_FACTS)
    var_map = program.prob_var_map()

    facts = {e["atom"]: e["prob"] for e in var_map if e["kind"] == "fact"}
    assert facts["rain()"] == pytest.approx(0.7)
    assert facts["sprinkler()"] == pytest.approx(0.2)


def test_annotated_disjunction_variables_carry_their_choices():
    program = pyxlog.Program.compile(WITH_DISJUNCTION)
    var_map = program.prob_var_map()

    choices = [e for e in var_map if e["kind"] == "choice"]
    assert choices, "the disjunction must contribute at least one choice variable"
    assert "ctx(left)" in choices[0]["atoms"]
    assert choices[0]["probs"][0] == pytest.approx(0.6)


def test_map_index_matches_gradient_position():
    """Позиция i в карте — это переменная i+1, то есть позиция i в grad_true."""
    program = pyxlog.Program.compile(TWO_FACTS)
    result = program.evaluate(return_grads=True)
    var_map = program.prob_var_map()

    grads = torch.from_dlpack(result.grad_true[0])
    assert grads.shape[0] == len(var_map)
