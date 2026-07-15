import math
import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.join_bodies import prepare_extension
from pyxlog.ilp.neural_credit import CandidateSpec, credit_nll


def test_credit_is_sum_of_prob_times_score_hand_computed() -> None:
    """Два факта, два кандидата: реляционный покрывает только факт 0, нейро джойнит
    факту 0 событие 0, факту 1 — события 1 и 2. Значение сверяется с ручным счётом."""
    idx = prepare_extension([[0], [1, 2]], device="cpu", num_rows=3)
    specs = [
        CandidateSpec(cid=0, left="he", right="tag", is_neural=False,
                      witness_index=None,
                      binary_cover=torch.tensor([1.0, 0.0])),
        CandidateSpec(cid=1, left="he", right="sal", is_neural=True,
                      witness_index=idx, binary_cover=None),
    ]
    p = torch.tensor([0.6, 0.4])
    p_event = torch.tensor([0.9, 0.5, 0.5])
    is_pos = torch.tensor([True, False])

    loss = credit_nll(p, specs, p_event, is_pos)

    s_neural = torch.tensor([0.9, 1 - 0.5 * 0.5])          # noisy-OR по свидетелям
    credit0 = 0.6 * 1.0 + 0.4 * 0.9                         # позитивный факт
    credit1 = 0.6 * 0.0 + 0.4 * 0.75                        # негативный факт
    expected = (-math.log(credit0) - math.log(1 - credit1)) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_gamma_sharpens_only_the_neural_column() -> None:
    idx = prepare_extension([[0]], device="cpu", num_rows=1)
    spec = CandidateSpec(cid=0, left="he", right="sal", is_neural=True,
                         witness_index=idx, binary_cover=None)
    p = torch.tensor([1.0])
    p_event = torch.tensor([0.8])
    is_pos = torch.tensor([True])
    l1 = credit_nll(p, [spec], p_event, is_pos, gamma=1.0)
    l2 = credit_nll(p, [spec], p_event, is_pos, gamma=2.0)
    assert l2.item() > l1.item()        # 0.8^2 < 0.8 -> кредит ниже -> loss выше


class _FakeProg:
    """valid_candidates + relation_facts достаточно, чтобы построить спеки без CUDA."""

    def __init__(self) -> None:
        self._facts = {
            "has_event": [[0, 0], [0, 1], [1, 2]],   # (edge, ev)
            "sal": [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]],
            "tag": [[0, 1], [1, 0], [2, 1]],
        }

    def valid_candidates(self, mask_name):
        names = ["has_event", "sal", "tag"]
        cands, cid = [], 0
        for i, ln in enumerate(names):
            for j, rn in enumerate(names):
                cands.append({"id": cid, "i": i, "j": j, "k": 3,
                              "left_name": ln, "right_name": rn, "head_name": "plastic"})
                cid += 1
        return cands

    def relation_facts(self, name):
        return self._facts[name]


def test_enumerate_specs_builds_neural_and_relational_columns() -> None:
    from pyxlog.ilp.neural_credit import enumerate_specs

    facts = [(0, 1), (1, 0), (2, 1)]                       # (edge, label)
    specs = enumerate_specs(_FakeProg(), "W", facts,
                            neural_relations={"sal": 3}, device="cpu")
    by_names = {(s.left, s.right): s for s in specs}
    neural = by_names[("has_event", "sal")]
    assert neural.is_neural and neural.witness_index.num_bindings == len(facts)
    relational = by_names[("has_event", "tag")]
    # факт (0,1): у ребра 0 события {0,1}, tag даёт (0,1) -> покрыт; (1,0): ребро 1
    # имеет событие 2, tag(2,1) != (2,0) -> не покрыт; (2,1): у ребра 2 событий нет.
    assert relational.binary_cover.tolist() == [1.0, 0.0, 0.0]


def test_a_neural_relation_in_the_left_slot_is_refused() -> None:
    from pyxlog.ilp.neural_credit import enumerate_specs

    with pytest.raises(ValueError, match="left slot"):
        enumerate_specs(_FakeProg(), "W", [(0, 1)],
                        neural_relations={"has_event": 3}, device="cpu")
