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
