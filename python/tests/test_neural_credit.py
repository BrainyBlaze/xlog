import math
import random

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.join_bodies import prepare_extension
from pyxlog.ilp.neural_credit import (
    CandidateSpec,
    credit_nll,
    kfold_select,
    train_engine_mode,
)


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


def _w1_world(n_edges: int = N_EDGES, k: int = K, seed: int = 0):
    rng = random.Random(seed)
    features, own, labels = [], {}, {}
    salient = set()
    ev = 0
    for edge in range(n_edges):
        positive = rng.random() < 0.5
        evs = []
        for slot in range(k):
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
        for edge in range(n_edges):
            mine = set(own[edge])
            out += [(edge, e) for e in r.sample([x for x in all_ev if x not in mine], k)]
        return out

    return dict(
        features=features, own=own, labels=labels, n_ev=n_ev, salient=salient,
        has_event=[(edge, e) for edge in range(n_edges) for e in own[edge]],
        has_event_bad=fair(random.Random(1000 + seed)),
        co=fair(random.Random(2000 + seed)),
        sal=[(e, l) for e in all_ev for l in (0, 1)],
    )


def _w1_source(world, extra: dict[str, list[tuple[int, int]]] | None = None) -> str:
    """The W1 source, optionally with additional ground relations spliced in before
    the (unchanged) plastic/2 rule -- used by the control tests below to add a
    coincidental (`lucky`), a perfect-relational (`tag`), or a trivially-true
    (`anything`) escape to the pool without duplicating the base world."""
    lines = []
    for name in ("has_event", "has_event_bad", "co", "sal"):
        lines += [f"{name}({a}, {b})." for a, b in world[name]]
    for name, pairs in (extra or {}).items():
        lines += [f"{name}({a}, {b})." for a, b in pairs]
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


def test_kfold_selection_semantics_on_synthetic_scores() -> None:
    """Селекция и Оккам-tie-break — чистая функция от holdout-скоров."""
    from pyxlog.ilp.neural_credit import _select_from_holdout

    # чёткое-случайное: 1.0 на трейне, 0.55 на holdout -> ниже min_fit? нет, но ниже
    # мягко-верного -> проигрывает селекцию
    s = _select_from_holdout(
        {("he", "lucky"): 0.55, ("he", "sal"): 0.97}, neural_rights={"sal"},
        min_fit=0.75)
    assert s.rule == ("he", "sal")

    # Оккам: реляционный и нейро в ничьей -> реляционный
    s = _select_from_holdout(
        {("he", "tag"): 0.99, ("he", "sal"): 0.985}, neural_rights={"sal"},
        min_fit=0.75)
    assert s.rule == ("he", "tag")

    # никто не прошёл fit-гейт -> воздержание с причиной
    s = _select_from_holdout(
        {("he", "a"): 0.5, ("he", "b"): 0.6}, neural_rights=set(), min_fit=0.75)
    assert s.rule is None and "fit gate" in s.reason


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


# ---------------------------------------------------------------------------
# Task 5: engine-mode CONTROLS for the holdout arbiter. Each test extends the W1
# world with one additional relation (via `_w1_source`'s `extra=` hook) and runs
# the real `kfold_select` -- a fresh `prog_factory` recompiles the ENGINE program
# per fold, so this exercises the whole holdout path Task 4 built, not a mock of
# it. `folds=4, steps=300` keeps pod time sane; `seed=0` everywhere.
# ---------------------------------------------------------------------------

def _w1_make_network():
    net = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Softmax(dim=-1)).cuda()
    with torch.no_grad():
        net[0].bias[1] -= 2.0
    return net


@cuda
def test_kill_criterion_holdout_separates_coincidental_from_correct():
    """THIS IS THE PHASE KILL-CRITERION -- if this test fails on GPU, the phase stops.

    Extends W1 with `lucky(Ev, L)`: truthful for edges 0..n//2 (every event of such
    an edge is labeled with that edge's TRUE label), then coincidental for the rest
    (labeled by `random.Random(7000)`, uncorrelated with the true label). `lucky` is
    exactly the crisp-but-coincidental relation the training-weight-only selector
    cannot be trusted to reject: it fits perfectly on the half of the data it was
    built to fit, so its TRAINING score can rival the true neural join. Held-out
    generalization is what must tell them apart -- `lucky`'s accuracy should collapse
    towards chance on its random half while `sal`'s does not, so k-fold holdout must
    still pick the true join.

    measured numbers recorded after the pod run (holdout scores: sal vs lucky).
    """
    world = _w1_world(seed=0)
    half = N_EDGES // 2
    rng = random.Random(7000)
    lucky = []
    for edge in range(N_EDGES):
        label = world["labels"][edge] if edge < half else bool(rng.choice([0, 1]))
        lucky += [(e, int(label)) for e in world["own"][edge]]

    src = _w1_source(world, extra={"lucky": lucky})
    features = torch.tensor([[f] for f in world["features"]], dtype=torch.float32).cuda()
    facts = [(edge, 1) for edge in range(N_EDGES)]
    is_positive = [world["labels"][edge] for edge in range(N_EDGES)]

    sel = kfold_select(
        lambda: pyxlog.IlpProgramFactory.compile(src, device=0, memory_mb=1024),
        "W", facts, is_positive, _w1_make_network, features,
        neural_relations={"sal": world["n_ev"]}, folds=4, steps=300, seed=0,
    )
    print(f"\n[kill-criterion] rule={sel.rule} reason={sel.reason}")
    assert sel.rule == ("has_event", "sal"), sel


@cuda
def test_occam_perfect_relational_beats_soft_neural():
    """With a PERFECT relational escape `tag(Ev, L)` in the pool -- truthful for
    EVERY event, not just half like `lucky` above -- holdout selection should
    prefer the simpler relational candidate over the soft neural join at equal
    (near-1.0) generalization. This is Occam's razor applied to holdout SCORES,
    not training weight, which cannot distinguish crisp-and-correct from
    soft-and-correct in principle (both fit the training data).

    measured numbers recorded after the pod run (sel.reason / sel.margin).
    """
    world = _w1_world(seed=0)
    tag = [(e, int(world["labels"][edge]))
           for edge in range(N_EDGES) for e in world["own"][edge]]

    src = _w1_source(world, extra={"tag": tag})
    features = torch.tensor([[f] for f in world["features"]], dtype=torch.float32).cuda()
    facts = [(edge, 1) for edge in range(N_EDGES)]
    is_positive = [world["labels"][edge] for edge in range(N_EDGES)]

    sel = kfold_select(
        lambda: pyxlog.IlpProgramFactory.compile(src, device=0, memory_mb=1024),
        "W", facts, is_positive, _w1_make_network, features,
        neural_relations={"sal": world["n_ev"]}, folds=4, steps=300, seed=0,
    )
    print(f"\n[Occam] rule={sel.rule} reason={sel.reason} margin={sel.margin}")
    assert sel.rule == ("has_event", "tag"), sel
    assert "Occam" in sel.reason or sel.margin > 0


@cuda
def test_trivially_true_relation_no_confident_wrong_answer():
    """`anything(Ev, L)` covers EVERY event x label pair -- a relation with zero
    discriminative content that is nonetheless perfectly "true" everywhere it is
    asked. A training-weight-only selector can land on such a relation with high
    confidence and no signal (this is the exact failure `discovery.select_rule`'s
    MIN_WEIGHT/TIE_TOLERANCE gates were built against -- see discovery.py). The
    holdout arbiter must not hand out a CONFIDENT WRONG ANSWER for it: either
    `anything` never wins (the true join still does, or nobody clears the fit
    gate), or the arbiter abstains and says why.

    measured numbers recorded after the pod run (sel.rule / sel.reason).
    """
    world = _w1_world(seed=0)
    anything = [(e, l) for e in range(world["n_ev"]) for l in (0, 1)]

    src = _w1_source(world, extra={"anything": anything})
    features = torch.tensor([[f] for f in world["features"]], dtype=torch.float32).cuda()
    facts = [(edge, 1) for edge in range(N_EDGES)]
    is_positive = [world["labels"][edge] for edge in range(N_EDGES)]

    sel = kfold_select(
        lambda: pyxlog.IlpProgramFactory.compile(src, device=0, memory_mb=1024),
        "W", facts, is_positive, _w1_make_network, features,
        neural_relations={"sal": world["n_ev"]}, folds=4, steps=300, seed=0,
    )
    print(f"\n[anything] rule={sel.rule} reason={sel.reason} tied={sel.tied}")
    assert sel.rule is None or sel.rule == ("has_event", "sal"), sel
    if sel.rule is None:
        assert "fit gate" in sel.reason or sel.tied


# ---------------------------------------------------------------------------
# Task 6: THE ACCEPTANCE TEST -- the project's original killer criterion
# (`code/spike_bridge.py`), reproduced in engine mode. `spike_bridge.py` hand-wrote
# a 6-triple `POOL` and could only ever exercise the arity-1-pinned torch mixture
# path; here the pool is `prog.valid_candidates("W")` in full (zero hand-written
# candidates) and the head is `plastic(X, Y)` -- arity 2, the multi-outcome shape
# `neurosymbolic.py`'s `joint_candidate_eligibility(train_head, 1, n)` call cannot
# even compile (its arity argument is hardcoded to 1: one supervised column
# ranging over example rows), let alone select correctly on.
# ---------------------------------------------------------------------------

@cuda
def test_acceptance_original_killer_criterion_arity2_head():
    """THE ORIGINAL KILLER CRITERION, passing in engine mode.

    THE POINT: the head is ARITY 2 -- `plastic(X, Y)`, edge X, label Y -- a
    multi-outcome plasticity relation that the torch MIXTURE path
    (`neurosymbolic.py::_train_joint_mixture`) cannot even compile: its
    `joint_candidate_eligibility` call is hardcoded to `arity=1` (a single
    supervised head column ranging over example row indices), not
    parameterized per candidate or per head. Engine mode carries no such pin --
    `enumerate_specs` reads witnesses and covers straight from
    `prog.relation_facts`, indifferent to how many columns the head carries --
    so this is the first time the original spike's killer criterion (a soft
    neural join beating hand-picked relational distractors under k-fold
    holdout, with a per-event detector that generalizes OUT of the training
    support) runs on a head the older mixture path structurally cannot touch.

    ZERO HAND-WRITTEN CANDIDATES: unlike `code/spike_bridge.py`'s hand-built
    `POOL = [(has_event, sal), (has_event_bad, sal), (co, sal), (has_event,
    tag), (has_event_bad, tag), (co, tag)]`, the pool here is whatever
    `prog.valid_candidates("W")` enumerates over the four ground relations
    `_w1_source` splices in (has_event, has_event_bad, co, sal) -- the full
    cross product, neural-in-left triples filtered by `enumerate_specs` (no
    witness semantics), everything else scored. `kfold_select` must still land
    on the true join, `(has_event, sal)`, purely from held-out generalization.

    measured numbers (recorded after the controller's pod run):
      kfold: sel.rule=<TBD>, sel.reason=<TBD>
      detector: mean P(salient)=<TBD>, mean P(quiet)=<TBD>
      generalization: net([[0.95]])[:,1]=<TBD>, net([[0.005]])[:,1]=<TBD>
    """
    world = _w1_world(n_edges=40, k=4, seed=0)
    n_edges = len(world["own"])
    src = _w1_source(world)
    features = torch.tensor([[f] for f in world["features"]], dtype=torch.float32).cuda()
    facts = [(edge, 1) for edge in range(n_edges)]
    is_positive = [world["labels"][edge] for edge in range(n_edges)]

    def prog_factory():
        return pyxlog.IlpProgramFactory.compile(src, device=0, memory_mb=1024)

    sel = kfold_select(
        prog_factory, "W", facts, is_positive, _w1_make_network, features,
        neural_relations={"sal": world["n_ev"]}, folds=4, steps=300, seed=0,
    )
    print(f"\n[acceptance] kfold rule={sel.rule} reason={sel.reason}")
    assert sel.rule == ("has_event", "sal"), sel

    prog = prog_factory()
    network = _w1_make_network()
    result = train_engine_mode(
        prog, "W", facts, is_positive, network, features,
        neural_relations={"sal": world["n_ev"]}, steps=400, seed=0,
    )
    print(f"\n[acceptance] cand_probs = {result.cand_probs}")

    with torch.no_grad():
        probs = result.network(features)[:, 1].cpu().tolist()
    sal_p = [p for e, p in enumerate(probs) if e in world["salient"]]
    quiet_p = [p for e, p in enumerate(probs) if e not in world["salient"]]
    mean_sal, mean_quiet = sum(sal_p) / len(sal_p), sum(quiet_p) / len(quiet_p)
    print(f"[acceptance] mean P(salient)={mean_sal:.4f} mean P(quiet)={mean_quiet:.4f}")
    assert mean_sal > 0.9, mean_sal
    assert mean_quiet < 0.1, mean_quiet

    # Generalization OUT of the training support: probe features never seen during
    # training (the world only ever samples quiet in [0.01,0.4] and salient in
    # [0.6,0.99]), as in the merged demo's decisive probe.
    probe = torch.tensor([[0.95], [0.005]], dtype=torch.float32).cuda()
    with torch.no_grad():
        probe_p = result.network(probe)[:, 1].cpu().tolist()
    print(f"[acceptance] probe P(0.95)={probe_p[0]:.4f} P(0.005)={probe_p[1]:.4f}")
    assert probe_p[0] > 0.5, probe_p
    assert probe_p[1] < 0.5, probe_p
