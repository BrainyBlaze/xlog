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


def test_map_positions_line_up_with_the_gradient_vector():
    """Позиция i в карте — это переменная i (не i+1); слот 0 не используется.

    Ненулевые градиенты в grad_true могут появиться только у переменных, которые
    сама схема кодирования делает источником случайности (probabilistic facts и
    choice-переменные annotated disjunction) — вспомогательные Tseitin-переменные
    компиляции (AND/OR-гейты) градиента не получают. Поэтому обратное включение
    гарантировано схемой кодирования и ловит сдвиг на единицу: если бы карта была
    сдвинута, какая-то позиция с ненулевым градиентом оказалась бы отмечена в карте
    как "other" вместо "fact"/"choice".

    Равенство множеств (а не только включение) тут не гарантировано: вероятностный
    факт с вырожденными условиями (например, если по структуре программы его вклад
    в выбранный query не воздействует ни на один достижимый путь) мог бы получить
    нулевой градиент, не будучи от этого меньше фактом. Поэтому проверяем только
    направление nonzero ⊆ known, а не строгое равенство.
    """
    program = pyxlog.Program.compile(TWO_FACTS)
    result = program.evaluate(return_grads=True)
    var_map = program.prob_var_map()

    grads = torch.from_dlpack(result.grad_true[0])
    assert grads.shape[0] == len(var_map)
    assert var_map[0]["kind"] == "other", "variable 0 does not exist; slot 0 is padding"

    known = {i for i, e in enumerate(var_map) if e["kind"] != "other"}
    nonzero = {i for i in range(grads.shape[0]) if grads[i].item() != 0.0}
    assert nonzero <= known, (
        "every CNF variable with a nonzero gradient must be a fact/choice the map "
        "knows about at that same position; a one-off shift would violate this"
    )
