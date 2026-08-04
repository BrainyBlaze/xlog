"""CompiledProgram must expose which fact each CNF variable stands for."""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


def two_facts(r: float, s: float) -> str:
    """Та же форма программы, что и ``TWO_FACTS``, но с параметризованными
    вероятностями двух фактов — нужна для пересборки программы с возмущённым
    ``p`` при проверке конечной разностью."""
    return f"""
{r}::rain().
{s}::sprinkler().
wet() :- rain().
wet() :- sprinkler().
query(wet()).
"""


TWO_FACTS = two_facts(0.7, 0.2)

WITH_DISJUNCTION = """
0.6::ctx(left); 0.4::ctx(right).
0.9::flag().
seen() :- ctx(left), flag().
query(seen()).
"""

# Какому именованному аргументу two_facts(r, s) соответствует каждый факт по
# его строке атома — нужно, чтобы возмущать ровно один факт за раз, оставляя
# остальные без изменений (см. Требование 1 брифа).
_ATOM_TO_PARAM = {"rain()": "r", "sprinkler()": "s"}

# eps и относительный допуск взяты дословно из брифа: 1e-5 — та же величина,
# что уже сходилась в анализе на GPU в репозитории anchor
# (tests/test_free_energy_grad.py::test_gradient_matches_a_finite_difference);
# 1e-4 — требуемый относительный допуск, его нельзя ослаблять.
_EPS = 1e-5
_REL_TOL = 1e-4


def test_map_length_matches_the_gradient_vectors():
    """prob_var_map() and result.num_vars both report the CNF encoder's
    variable *capacity* (3 * PIR node count at compile time), not the number
    of CNF variables actually in use and not the number of random variables
    in the program — most slots in a real program are unused padding. This
    test only checks that the two report the *same* capacity, i.e. that
    var_map stays index-aligned with grad_true/grad_false (which are also
    allocated at that capacity). It is not a claim that len(var_map) counts
    CNF variables or random variables.
    """
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
    """Диагностичная версия: прошлый прогон на GPU (RTX 4090, pyxlog 0.11.0,
    exact_ddnnf) уронил этот тест, но команда `pytest ... | tail -35` обрезала
    именно текст падения — причина неизвестна. Поэтому при падении тест обязан
    сам напечатать всё, что нужно для диагноза без повторного (платного) захода
    на под: полную карту переменных и явную проверку конкурирующей гипотезы —
    что ctx(left) мог быть понижен в независимый "fact"-лист вместо choice-цепочки
    annotated disjunction.

    Содержательные утверждения (choice-запись существует, "ctx(left)" есть в её
    atoms, probs[0] ≈ 0.6) не ослаблены: если экспорт annotated disjunction
    действительно неполон, тест обязан остаться красным.
    """
    program = pyxlog.Program.compile(WITH_DISJUNCTION)
    result = program.evaluate()
    var_map = program.prob_var_map()

    dump = "\n".join(f"  [{i}] {entry!r}" for i, entry in enumerate(var_map))
    diagnostics = (
        f"result.num_vars = {result.num_vars}, len(var_map) = {len(var_map)}\n"
        f"полная карта переменных (позиция: запись):\n{dump}"
    )

    # Конкурирующая гипотеза из брифа: annotated disjunction была понижена до
    # простых независимых листьев, и тогда ctx(left) попал бы в карту как
    # kind == "fact", а не как часть choice-цепочки.
    left_as_fact = [
        e for e in var_map if e.get("kind") == "fact" and e.get("atom") == "ctx(left)"
    ]
    assert not left_as_fact, (
        "конкурирующая гипотеза подтвердилась: ctx(left) экспортирован с "
        f"kind == 'fact' ({left_as_fact!r}), т.е. annotated disjunction была "
        "понижена до независимых вероятностных листьев вместо одной choice-"
        f"переменной с choices=[(ctx(left), 0.6), (ctx(right), 0.4)].\n{diagnostics}"
    )

    choices = [e for e in var_map if e.get("kind") == "choice"]
    assert choices, (
        "в карте нет ни одной записи с kind == 'choice' — annotated disjunction "
        f"вообще не представлена; ctx(left)-как-fact тоже не найден.\n{diagnostics}"
    )

    atoms0 = choices[0].get("atoms")
    assert "ctx(left)" in (atoms0 or []), (
        f"'ctx(left)' не найден в atoms первой choice-записи (atoms = {atoms0!r}).\n"
        "Если элементы этого списка выглядят как 'ctx(sym#<N>)' вместо 'ctx(left)' "
        "/ 'ctx(right)' — это не отсутствие choice-переменной в карте, а потеря "
        "читаемого имени символьного аргумента при форматировании GroundAtom -> str: "
        "atom_to_string в crates/pyxlog/src/program.rs печатает Value::Symbol как "
        "format!(\"sym#{}\", sym) по интернированному id, не вызывая "
        "xlog_core::symbol::resolve(id), хотя эта функция для обратного разрешения "
        f"существует (crates/xlog-core/src/symbol.rs).\n{diagnostics}"
    )
    assert choices[0].get("probs", [None])[0] == pytest.approx(0.6), (
        f"choices[0]['probs'][0] = {choices[0].get('probs')!r}, ожидалось ~0.6 "
        f"(0.6::ctx(left); 0.4::ctx(right)).\n{diagnostics}"
    )


def test_map_positions_line_up_with_the_gradient_vector():
    """Позиция i в карте — это переменная i (не i+1); слот 0 не используется.

    Предыдущая версия этого теста опиралась на посылку "ненулевой градиент
    бывает только у вероятностных переменных" (nonzero ⊆ known). Прогон на
    GPU (RTX 4090, pyxlog 0.11.0, exact_ddnnf) эту посылку опроверг: у
    TWO_FACTS вспомогательная переменная wet() (производная от rain()/
    sprinkler() через OR) тоже получает ненулевой градиент. Это дефект теста,
    а не карты, поэтому инвариант nonzero ⊆ known здесь не восстанавливается
    ни в каком виде.

    Вместо этого — прямая проверка выравнивания через конечную разность по
    каждому вероятностному факту. Для TWO_FACTS: P(wet) = 1 − (1−r)(1−s);
    при r=0.7, s=0.2 получаем P=0.76, ∂logP/∂r=(1−s)/P≈1.0526,
    ∂logP/∂s=(1−r)/P≈0.3947 — производные различаются больше чем в два раза,
    поэтому сдвиг или перестановка позиций карты этот тест ломают.

    Якобиан p·(1−p) для перевода grad_true (∂logP/∂log w_true при
    p = w_true/(w_true+w_false)) в ∂logP/∂p установлен эталонным тестом
    движка (crates/xlog-prob/tests/exact_ddnnf_gpu_grads.rs: для
    `p::rain(). dry() :- not rain().` grad_true = −p, grad_false = +p) и
    независимо подтверждён конечной разностью на GPU в репозитории anchor
    (tests/test_free_energy_grad.py::test_gradient_matches_a_finite_difference).
    """
    program = pyxlog.Program.compile(TWO_FACTS)
    result = program.evaluate(return_grads=True)
    var_map = program.prob_var_map()

    grads = torch.from_dlpack(result.grad_true[0])
    assert grads.shape[0] == len(var_map), (
        f"grad_true[0] имеет длину {grads.shape[0]}, а var_map — {len(var_map)}; "
        "обе структуры обязаны индексироваться одним и тем же номером CNF-"
        "переменной, иначе сравнение по позициям бессмысленно"
    )
    assert var_map[0]["kind"] == "other", (
        f"var_map[0] = {var_map[0]!r}, а ожидался паддинг {{'kind': 'other'}}: "
        "переменная 0 не существует, CNF-переменные нумеруются с единицы, и слот "
        "0 карты обязан быть неиспользуемой заглушкой"
    )

    fact_entries = [(i, e) for i, e in enumerate(var_map) if e["kind"] == "fact"]
    assert fact_entries, (
        f"в карте TWO_FACTS не нашлось ни одной записи с kind == 'fact'; "
        f"полная карта: {var_map!r}"
    )

    for i, entry in fact_entries:
        atom = entry["atom"]
        p = entry["prob"]

        param = _ATOM_TO_PARAM.get(atom)
        assert param is not None, (
            f"на позиции {i} карта называет факт {atom!r}, для которого тест не "
            "знает, как пересобрать TWO_FACTS с возмущённой вероятностью "
            f"(известны только {sorted(_ATOM_TO_PARAM)}); допишите _ATOM_TO_PARAM, "
            "если TWO_FACTS была изменена"
        )

        base_kwargs = {"r": 0.7, "s": 0.2}
        base_kwargs[param] = p + _EPS
        source_plus = two_facts(**base_kwargs)
        base_kwargs[param] = p - _EPS
        source_minus = two_facts(**base_kwargs)

        log_p_plus = torch.from_dlpack(
            pyxlog.Program.compile(source_plus).evaluate().log_prob
        )[0].item()
        log_p_minus = torch.from_dlpack(
            pyxlog.Program.compile(source_minus).evaluate().log_prob
        )[0].item()
        finite_diff = (log_p_plus - log_p_minus) / (2.0 * _EPS)

        engine_grad = grads[i].item() / (p * (1.0 - p))

        denom = max(abs(finite_diff), 1e-12)
        rel_err = abs(engine_grad - finite_diff) / denom
        assert rel_err <= _REL_TOL, (
            f"позиция {i} (атом {atom!r}, p={p!r}): из движка "
            f"grad_true[{i}]/(p*(1-p)) = {engine_grad!r}, конечная разность "
            f"(log_prob(p+eps) - log_prob(p-eps))/(2*eps) = {finite_diff!r} "
            f"(log_p_plus={log_p_plus!r}, log_p_minus={log_p_minus!r}, eps={_EPS}); "
            f"относительная ошибка {rel_err!r} превышает допуск {_REL_TOL}. "
            "Если атом на этой позиции не тот, что ожидался по TWO_FACTS "
            "(rain() на позиции 1, sprinkler() на позиции 2) — карта сдвинута "
            "или переставлена относительно grad_true"
        )
