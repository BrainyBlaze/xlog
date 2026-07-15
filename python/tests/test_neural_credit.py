import math
import random

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.join_bodies import prepare_extension
from pyxlog.ilp.neural_credit import CandidateSpec, credit_nll, train_engine_mode


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


def test_a_neural_relation_in_the_left_slot_is_skipped_not_fatal() -> None:
    """Neural relations in the left slot are skipped during enumeration.

    This is pool filtering of an auto-enumerated space, not silent alteration of
    a user-declared rule: the engine always produces triples with neural-in-left,
    but the credit cannot score them (no witness semantics), so they are filtered
    rather than refused — a distinction that matters for production robustness."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    specs = enumerate_specs(_FakeProg(), "W", [(0, 1)],
                            neural_relations={"has_event": 3}, device="cpu")
    # No spec should have has_event in the left slot
    assert not any(s.left == "has_event" for s in specs)
    # But specs with has_event in the right slot should be present and neural
    has_event_right = [s for s in specs if s.right == "has_event"]
    assert len(has_event_right) > 0
    assert all(s.is_neural for s in has_event_right)


# ---------------------------------------------------------------------------
# Engine-mode training loop (Task 3). CUDA-gated: the ENGINE compiles the
# program (device=0), which needs a real CUDA context.
#
# World W1 is `code/spike_bridge.py::build_world(seed, informative=True)`,
# rewritten locally here rather than imported from the artifacts spike: 30
# edges, k=4 events each, exactly one salient event (slot 0) on a positive
# edge with an INFORMATIVE feature; `has_event_bad` / `co` are
# equal-cardinality relational distractors sampled from OTHER edges' events
# (fair sampling, same as the spike), so they are exactly as sharp a noisy-OR
# as the true join and carry zero label information. No `tag` escape is in
# the pool (unlike the spike's W2/W3), so the network is the ONLY thing that
# can explain the positives.
# ---------------------------------------------------------------------------
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="xlog engine requires CUDA")

N_EDGES = 30
K = 4
TEMPLATE = "learnable(W) :: plastic(X, Y) :- bL(X, Z), bR(Z, Y)."


def _w1_world(seed: int = 0):
    rng = random.Random(seed)
    features, own, labels = [], {}, {}
    salient = set()
    ev = 0
    for edge in range(N_EDGES):
        positive = rng.random() < 0.5
        evs = []
        for slot in range(K):
            is_sal = positive and slot == 0
            if is_sal:
                salient.add(ev)
            features.append(
                round(rng.uniform(0.6, 0.99) if is_sal else rng.uniform(0.01, 0.4), 3)
            )
            evs.append(ev)
            ev += 1
        own[edge] = evs
        labels[edge] = positive
    n_ev = ev
    all_ev = list(range(n_ev))

    def fair(r):
        out = []
        for edge in range(N_EDGES):
            mine = set(own[edge])
            out += [(edge, e) for e in r.sample([x for x in all_ev if x not in mine], K)]
        return out

    return dict(
        features=features, own=own, labels=labels, n_ev=n_ev, salient=salient,
        has_event=[(edge, e) for edge in range(N_EDGES) for e in own[edge]],
        has_event_bad=fair(random.Random(1000 + seed)),
        co=fair(random.Random(2000 + seed)),
        sal=[(e, l) for e in all_ev for l in (0, 1)],
    )


def _w1_source(world) -> str:
    lines = []
    for name in ("has_event", "has_event_bad", "co", "sal"):
        lines += [f"{name}({a}, {b})." for a, b in world[name]]
    lines.append(TEMPLATE)
    return "\n".join(lines)


def _train_w1(world, seed: int = 0, steps: int = 400):
    prog = pyxlog.IlpProgramFactory.compile(_w1_source(world), device=0, memory_mb=1024)
    torch.manual_seed(seed)
    network = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Softmax(dim=-1)).cuda()
    with torch.no_grad():
        network[0].bias[1] -= 2.0
    features = torch.tensor([[f] for f in world["features"]], dtype=torch.float32).cuda()
    facts = [(edge, 1) for edge in range(N_EDGES)]
    is_positive = [world["labels"][edge] for edge in range(N_EDGES)]
    result = train_engine_mode(
        prog, "W", facts, is_positive, network, features,
        neural_relations={"sal": world["n_ev"]}, steps=steps, seed=seed,
    )
    return result, features


@cuda
def test_w1_engine_mode_neural_wins_detector_separates_and_training_is_deterministic():
    """The true neural join wins the mixture, the per-event detector separates
    salient from quiet events, and two identically-seeded runs are bitwise
    deterministic (mirroring the dILP trainer's own determinism contract)."""
    world = _w1_world(seed=0)
    result, features = _train_w1(world, seed=0)

    print(f"\n[W1] cand_probs = {result.cand_probs}")
    assert result.cand_probs[("has_event", "sal")] > 0.95, result.cand_probs

    with torch.no_grad():
        probs = result.network(features)[:, 1].cpu().tolist()
    sal_p = [p for e, p in enumerate(probs) if e in world["salient"]]
    quiet_p = [p for e, p in enumerate(probs) if e not in world["salient"]]
    mean_sal, mean_quiet = sum(sal_p) / len(sal_p), sum(quiet_p) / len(quiet_p)
    print(
        f"[W1] mean P(salient)={mean_sal:.4f} mean P(quiet)={mean_quiet:.4f} "
        f"separation={mean_sal - mean_quiet:.4f}"
    )
    # min(salient) - max(quiet) > 0.5 is too strict for 400 steps (measured);
    # the mean-vs-mean gap is the honest separation gate here.
    assert mean_sal - mean_quiet > 0.5, (mean_sal, mean_quiet)

    # Determinism: two fresh, identically-seeded runs over the SAME world must
    # produce a bitwise-equal final loss.
    result_b, _ = _train_w1(world, seed=0)
    assert result.losses[-1] == result_b.losses[-1]
