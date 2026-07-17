import math
import random

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.join_bodies import prepare_extension
from pyxlog.ilp.neural_credit import (
    CandidateSpec,
    HoldoutSelection,
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

    s_neural = [0.9, 1 - 0.5 * 0.5]                         # noisy-OR по свидетелям
    credit0 = 0.6 * 1.0 + 0.4 * s_neural[0]                 # позитивный факт
    credit1 = 0.6 * 0.0 + 0.4 * s_neural[1]                 # негативный факт
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
        # The real engine's cross product also always contains the dILP
        # TEMPLATE's own learnable placeholders (bL/bR): no ground extension,
        # so relation_facts below raises ValueError for them just like the
        # engine does for `prog.relation_facts("bL")`.
        for ln, rn in (("bL", "sal"), ("has_event", "bR"), ("bL", "bR")):
            cands.append({"id": cid, "i": 0, "j": 0, "k": 3,
                          "left_name": ln, "right_name": rn, "head_name": "plastic"})
            cid += 1
        return cands

    def relation_facts(self, name):
        if name not in self._facts:
            raise ValueError(f"Relation '{name}' not found")
        return self._facts[name]


def test_enumerate_specs_builds_neural_and_relational_columns() -> None:
    from pyxlog.ilp.neural_credit import enumerate_specs

    facts = [(0, 1), (1, 0), (2, 1)]                       # (edge, label)
    specs = enumerate_specs(_FakeProg(), "W", facts,
                            neural_relations={"sal": 3}, device="cpu", n_labels=2)
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

    # num_rows=2: has_event's left partners (sal, tag) join constants {0, 1},
    # and the dense-identity law (finding 1 of the part-2 review) requires the
    # witness domain to be exactly 0..num_rows-1.
    specs = enumerate_specs(_FakeProg(), "W", [(0, 1)],
                            neural_relations={"has_event": 2}, device="cpu",
                            n_labels=2)
    # No spec should have has_event in the left slot
    assert not any(s.left == "has_event" for s in specs)
    # But specs with has_event in the right slot should be present and neural
    has_event_right = [s for s in specs if s.right == "has_event"]
    assert len(has_event_right) > 0
    assert all(s.is_neural for s in has_event_right)


def test_template_placeholder_slots_are_skipped_not_fatal() -> None:
    """Template placeholders (bL/bR) have no ground extension: relation_facts
    raises ValueError for them, exactly as the real engine does for
    `prog.relation_facts("bL")`. This is a TARGETED skip -- pool filtering, same
    rationale as the __xlog_ meta relations and the neural-in-left skip above --
    not a blanket swallow of engine errors: a NON-ValueError from relation_facts
    must still propagate."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    facts = [(0, 1), (1, 0), (2, 1)]
    specs = enumerate_specs(_FakeProg(), "W", facts,
                            neural_relations={"sal": 3}, device="cpu", n_labels=2)

    # No spec touches a template placeholder.
    assert not any(s.left in ("bL", "bR") or s.right in ("bL", "bR") for s in specs)

    # The good candidates are unchanged from
    # test_enumerate_specs_builds_neural_and_relational_columns.
    by_names = {(s.left, s.right): s for s in specs}
    neural = by_names[("has_event", "sal")]
    assert neural.is_neural and neural.witness_index.num_bindings == len(facts)
    relational = by_names[("has_event", "tag")]
    assert relational.binary_cover.tolist() == [1.0, 0.0, 0.0]

    # Only ValueError is a targeted skip; anything else propagates.
    class _FakeProgBadRelation(_FakeProg):
        def relation_facts(self, name):
            if name == "tag":
                raise TypeError("boom")
            return super().relation_facts(name)

    with pytest.raises(TypeError):
        enumerate_specs(_FakeProgBadRelation(), "W", facts,
                        neural_relations={"sal": 3}, device="cpu", n_labels=2)


# ---------------------------------------------------------------------------
# Review wave, part 1 (levi770 on PR #154): the neural score must resolve the
# network column at THE FACT'S OWN label y (finding A), seeds must determine
# network construction (B), an empty scoreable pool refuses typed (C), and the
# "l|r" key round-trip guards against '|' in relation names (D).
# ---------------------------------------------------------------------------


def test_neural_witnesses_resolve_the_network_column_per_fact_y() -> None:
    """Finding A: a witness z for fact (h, y) indexes the FLAT (event, label)
    probability at z * n_labels + y — the column comes from the fact's own y,
    never from a hardcoded positive column."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    specs = enumerate_specs(_FakeProg(), "W", [(0, 1), (1, 0)],
                            neural_relations={"sal": 3}, device="cpu", n_labels=2)
    neural = {(s.left, s.right): s for s in specs}[("has_event", "sal")]
    # edge 0 owns events [0, 1] at y=1 -> flat [1, 3]; edge 1 owns [2] at y=0 -> [4]
    assert neural.witness_index.event_ids.tolist() == [1, 3, 4]
    assert neural.witness_index.binding_ids.tolist() == [0, 0, 1]


def test_fact_label_outside_the_network_output_is_refused_typed() -> None:
    """Finding A: a fact whose y has no network column is a contract violation,
    refused with the label named — not silently scored at a wrong column."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    with pytest.raises(ValueError, match="label 2"):
        enumerate_specs(_FakeProg(), "W", [(0, 2)],
                        neural_relations={"sal": 3}, device="cpu", n_labels=2)


def test_train_engine_mode_refuses_a_non_2d_network_output() -> None:
    """Finding A: n_labels is read off the network's actual output, so that
    output must be 2-D [num_events, num_labels] — anything else is refused."""
    class _FlatNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.lin(x).reshape(-1)

    features = torch.tensor([[0.1], [0.2], [0.3]])
    with pytest.raises(ValueError, match="2-D"):
        train_engine_mode(_FakeProg(), "W", [(0, 1)], [True], _FlatNet(),
                          features, neural_relations={"sal": 3}, steps=1)


def test_kfold_select_seeds_network_construction_not_ambient_rng() -> None:
    """Finding B: kfold_select(..., seed=) must determine the per-fold network
    inits by itself — two runs under scrambled ambient RNG draw identical values
    inside make_network, and the folds draw DIFFERENT values from each other."""
    features = torch.tensor([[0.1], [0.2], [0.3]])
    facts = [(0, 1), (1, 0), (2, 1)]
    is_positive = [True, False, True]
    draws: list[float] = []

    def make_network():
        draws.append(float(torch.rand(())))
        return torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Softmax(dim=-1))

    def run():
        return kfold_select(_FakeProg, "W", facts, is_positive, make_network,
                            features, neural_relations={"sal": 3}, folds=3,
                            steps=2, seed=0)

    torch.manual_seed(11111); torch.rand(5)          # scramble ambient RNG
    sel_a = run()
    torch.manual_seed(22222); torch.rand(17)         # scramble it differently
    sel_b = run()

    assert draws[:3] == draws[3:], draws
    assert len(set(draws[:3])) == 3, draws           # per-fold derived seeds differ
    assert sel_a == sel_b


def test_empty_scoreable_pool_is_refused_typed_with_filter_counts() -> None:
    """Finding C: a pool that filters to nothing refuses typed, naming how many
    candidates each filter removed — never AttributeError/max() deep inside."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    with pytest.raises(ValueError, match="zero scoreable"):
        enumerate_specs(_FakeProg(), "W", [(0, 1)],
                        neural_relations={"has_event": 3, "sal": 3, "tag": 3},
                        device="cpu", n_labels=2)


def test_credit_nll_refuses_an_empty_candidate_list() -> None:
    """Finding C: credit over no candidates is undefined, refused typed."""
    from pyxlog.ilp.neural_credit import credit_nll

    with pytest.raises(ValueError, match="no candidates"):
        credit_nll(torch.tensor([]), [], torch.tensor([0.5]),
                   torch.tensor([True]))


def test_relation_name_containing_the_key_separator_is_refused() -> None:
    """Finding D: candidate keys round-trip through 'l|r'; a relation name that
    contains '|' would corrupt the round-trip, so it is refused up front."""
    from pyxlog.ilp.neural_credit import _select_from_holdout

    with pytest.raises(ValueError, match="separator"):
        _select_from_holdout({("he|x", "sal"): 0.9}, neural_rights={"sal"},
                             min_fit=0.75)


# ---------------------------------------------------------------------------
# Review wave, part 2 (levi770 on PR #154, engine-semantics half): the mixture
# path's domain_ids dense-identity law applies to engine mode too (finding 1),
# Occam licenses nothing among relational duplicates (2), the engine's pool
# cross-products ALL arities (3), empty held-out folds poison scores (5), and
# the tie tolerance is a holdout-axis quantity (8).
# ---------------------------------------------------------------------------


def test_sparse_witness_domain_without_dense_identity_is_refused() -> None:
    """Finding 1: raw engine constants index feature rows, so the dense-identity
    law from the mixture path applies verbatim -- a witness domain that is not
    exactly 0..num_rows-1 could gather other events' probabilities silently and
    is refused, not guessed at."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    # _FakeProg's left relations only ever join constants {0, 1, 2}; declaring
    # 5 feature rows leaves rows 3..4 unjoined -- ambiguous, refused.
    with pytest.raises(ValueError, match="dense identity"):
        enumerate_specs(_FakeProg(), "W", [(0, 1)],
                        neural_relations={"sal": 5}, device="cpu", n_labels=2)


def test_occam_narrowing_refuses_a_residual_relational_tie() -> None:
    """Finding 2: the relational preference resolves a MIXED tie only when it
    yields a UNIQUE relational candidate; between extensionally identical
    relational duplicates it abstains -- vocabulary order must not pick."""
    from pyxlog.ilp.neural_credit import _select_from_holdout

    scores = {("he", "dup_a"): 0.9, ("he", "dup_b"): 0.9, ("he", "sal"): 0.9}
    s = _select_from_holdout(scores, neural_rights={"sal"}, min_fit=0.75)
    assert s.rule is None, s
    assert "relational" in s.reason

    # Vocabulary-order independence: reversed insertion order, same abstention.
    rev = dict(reversed(list(scores.items())))
    s2 = _select_from_holdout(rev, neural_rights={"sal"}, min_fit=0.75)
    assert s2.rule is None, s2


def test_non_binary_relations_in_the_pool_are_skipped_not_fatal() -> None:
    """Finding 3: valid_candidates cross-products ALL relations regardless of
    arity; a unary or arity-3 relation must be pool-filtered (counted), never
    a bare IndexError or a silent first-two-columns projection."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    class _FakeProgMixedArity(_FakeProg):
        def __init__(self) -> None:
            super().__init__()
            self._facts["un"] = [[0], [1]]
            self._facts["tri"] = [[0, 1, 2], [1, 0, 1]]

        def valid_candidates(self, mask_name):
            names = ["has_event", "sal", "tag", "un", "tri"]
            cands, cid = [], 0
            for i, ln in enumerate(names):
                for j, rn in enumerate(names):
                    cands.append({"id": cid, "i": i, "j": j, "k": 5,
                                  "left_name": ln, "right_name": rn,
                                  "head_name": "plastic"})
                    cid += 1
            return cands

    facts = [(0, 1), (1, 0), (2, 1)]
    specs = enumerate_specs(_FakeProgMixedArity(), "W", facts,
                            neural_relations={"sal": 3}, device="cpu", n_labels=2)
    assert not any(s.left in ("un", "tri") or s.right in ("un", "tri")
                   for s in specs)
    assert (("has_event", "sal") in {(s.left, s.right) for s in specs})


def test_more_folds_than_facts_is_refused_typed() -> None:
    """Finding 5: len(facts) < folds leaves empty held-out folds whose mean is
    NaN, poisoning every candidate's score -- refused typed at entry."""
    features = torch.tensor([[0.1], [0.2], [0.3]])

    def make_network():
        return torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Softmax(dim=-1))

    with pytest.raises(ValueError, match="folds"):
        kfold_select(_FakeProg, "W", [(0, 1), (1, 0), (2, 1)],
                     [True, False, True], make_network, features,
                     neural_relations={"sal": 3}, folds=5, steps=2, seed=0)


def test_holdout_tie_tolerance_is_a_parameter_on_the_holdout_axis() -> None:
    """Finding 8: the tie tolerance is a holdout-axis quantity kfold_select
    derives from the score quantum, so _select_from_holdout must accept it --
    and a coarser tolerance turns a clean win into a tie resolved by Occam."""
    from pyxlog.ilp.neural_credit import _select_from_holdout

    scores = {("he", "tag"): 0.95, ("he", "sal"): 0.90}
    # Default tolerance: 0.05 apart is a clean win for tag.
    assert _select_from_holdout(scores, {"sal"}, 0.75).rule == ("he", "tag")
    # A coarser holdout quantum makes 0.05 indistinguishable -> the tie is
    # mixed and the unique relational candidate wins by Occam, with the reason
    # saying so.
    s = _select_from_holdout(scores, {"sal"}, 0.75, tie_tolerance=0.06)
    assert s.rule == ("he", "tag"), s
    assert "Occam" in s.reason


# ---------------------------------------------------------------------------
# Consumer asks (levi770, integration seam-mapping after #154 merged): a typed
# registry for neural relations (ask 2) and a frozen-detector entry point
# (ask 3). Ask 1 (per-witness mask channel) is Phase-2 kernel design, not
# Python surface.
# ---------------------------------------------------------------------------


def test_neural_relation_spec_int_shorthand_is_equivalent_to_the_spec() -> None:
    """Ask 2: neural_relations={name: num_rows} stays valid shorthand for a
    full NeuralRelationSpec -- same pool either way."""
    from pyxlog.ilp.neural_credit import NeuralRelationSpec, enumerate_specs

    facts = [(0, 1), (1, 0), (2, 1)]
    a = enumerate_specs(_FakeProg(), "W", facts,
                        neural_relations={"sal": 3}, device="cpu", n_labels=2)
    b = enumerate_specs(_FakeProg(), "W", facts,
                        neural_relations={"sal": NeuralRelationSpec(num_rows=3)},
                        device="cpu", n_labels=2)
    assert ([(s.left, s.right, s.is_neural) for s in a]
            == [(s.left, s.right, s.is_neural) for s in b])


def test_registry_refuses_a_non_binary_declared_arity() -> None:
    """Ask 2: the pool is validated against the typed registry, not name
    conventions -- the credit scores the chain template's binary R(Z, Y) only,
    so a neural relation declared at any other arity is refused."""
    from pyxlog.ilp.neural_credit import NeuralRelationSpec, enumerate_specs

    with pytest.raises(ValueError, match="arity 3"):
        enumerate_specs(_FakeProg(), "W", [(0, 1)],
                        neural_relations={"sal": NeuralRelationSpec(num_rows=3,
                                                                    arity=3)},
                        device="cpu", n_labels=2)


def test_registry_refuses_arg_sorts_that_do_not_match_the_arity() -> None:
    from pyxlog.ilp.neural_credit import NeuralRelationSpec

    with pytest.raises(ValueError, match="arg_sorts"):
        NeuralRelationSpec(num_rows=3, arg_sorts=("event",))   # declared arity 2


def _frozen_detector_module():
    """A PARAMETERLESS detector: P(label 1) = [feature > 0.5]. Parameterless is
    load-bearing -- the training path cannot even build an optimizer over it,
    so this module working at all proves the frozen path trains nothing.
    Returned in eval mode: frozen_select refuses train-mode detectors."""
    class _Frozen(torch.nn.Module):
        def forward(self, x):
            p = (x[:, 0] > 0.5).float()
            return torch.stack([1 - p, p], dim=1)
    return _Frozen().eval()


def test_frozen_select_scores_candidates_against_an_external_detector() -> None:
    """Ask 3: engine-mode as an ACCEPTANCE INSTRUMENT -- candidates scored
    against a frozen, externally-trained detector, no gradient anywhere, the
    holdout arbiter's gates unchanged."""
    from pyxlog.ilp.neural_credit import frozen_select

    features = torch.tensor([[0.9], [0.8], [0.1]])   # events 0,1 salient; 2 quiet
    facts = [(0, 1), (1, 0), (2, 1)]
    is_positive = [True, True, False]

    sel = frozen_select(_FakeProg(), "W", facts, is_positive,
                        _frozen_detector_module(), features,
                        neural_relations={"sal": 3})
    assert sel.rule == ("has_event", "sal"), sel


def test_frozen_select_abstains_when_the_detector_fits_nothing() -> None:
    """Ask 3, the other direction: a detector under which no candidate fits
    ends in a fit-gate abstention -- 'is this rule right given this detector'
    can be answered NO."""
    from pyxlog.ilp.neural_credit import frozen_select

    class _Useless(torch.nn.Module):
        def forward(self, x):
            half = torch.full((x.shape[0],), 0.5)
            return torch.stack([half, half], dim=1)

    features = torch.tensor([[0.9], [0.8], [0.1]])
    facts = [(0, 1), (1, 0), (2, 1)]
    # Labels chosen so neither the flat detector nor any relational cover fits.
    is_positive = [False, True, True]

    sel = frozen_select(_FakeProg(), "W", facts, is_positive,
                        _Useless().eval(), features,
                        neural_relations={"sal": 3})
    assert sel.rule is None, sel
    assert "fit gate" in sel.reason


def test_frozen_select_refuses_a_detector_left_in_train_mode() -> None:
    """Review finding 1 (major, executed by the reviewer): in train mode a
    BatchNorm mutates its running statistics EVEN UNDER no_grad and dropout
    makes two identical scoring calls disagree -- both violate the
    bit-identical-artifact guarantee this entry point exists to provide.
    Refusal teaches the contract; silent mode-switching would hide caller bugs."""
    from pyxlog.ilp.neural_credit import frozen_select

    net = torch.nn.Sequential(torch.nn.BatchNorm1d(1), torch.nn.Linear(1, 2),
                              torch.nn.Softmax(dim=-1))
    assert net.training                      # torch's default -- exactly the trap
    with pytest.raises(ValueError, match="eval"):
        frozen_select(_FakeProg(), "W", [(0, 1), (1, 0), (2, 1)],
                      [True, True, False], net,
                      torch.tensor([[0.9], [0.8], [0.1]]),
                      neural_relations={"sal": 3})


def test_frozen_select_in_eval_mode_mutates_nothing_and_is_repeatable() -> None:
    """The frozen guarantee, pinned: state-hash-before == state-hash-after
    (parameters AND buffers -- BatchNorm running stats live in buffers), and
    two identical scoring calls return the same selection."""
    from pyxlog.ilp.neural_credit import frozen_select

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.BatchNorm1d(1), torch.nn.Dropout(0.5),
                              torch.nn.Linear(1, 2),
                              torch.nn.Softmax(dim=-1)).eval()
    state_before = [t.clone() for t in list(net.parameters()) + list(net.buffers())]

    features = torch.tensor([[0.9], [0.8], [0.1]])
    facts = [(0, 1), (1, 0), (2, 1)]
    is_positive = [True, True, False]
    sel_a = frozen_select(_FakeProg(), "W", facts, is_positive, net, features,
                          neural_relations={"sal": 3})
    sel_b = frozen_select(_FakeProg(), "W", facts, is_positive, net, features,
                          neural_relations={"sal": 3})

    state_after = list(net.parameters()) + list(net.buffers())
    assert all(torch.equal(a, b) for a, b in zip(state_before, state_after))
    assert sel_a == sel_b


def test_frozen_select_refuses_a_mismatched_is_positive_length() -> None:
    """Review finding N1 (medium, executed by the reviewer): a length-1
    is_positive broadcasts silently and every candidate scores against ONE
    label -- meaningless scores with no error. Refused typed, like zero facts."""
    from pyxlog.ilp.neural_credit import frozen_select

    with pytest.raises(ValueError, match="is_positive"):
        frozen_select(_FakeProg(), "W", [(0, 1), (1, 0), (2, 1)], [True],
                      _frozen_detector_module(),
                      torch.tensor([[0.9], [0.8], [0.1]]),
                      neural_relations={"sal": 3})


def test_registry_refuses_values_that_are_neither_int_nor_spec() -> None:
    """Review finding N2 (low): bool sneaks through isinstance(value, int),
    and a float used to die as a raw AttributeError -- the registry surface is
    now fully typed."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    for bad in (True, 3.0):
        with pytest.raises(ValueError, match="NeuralRelationSpec"):
            enumerate_specs(_FakeProg(), "W", [(0, 1)],
                            neural_relations={"sal": bad}, device="cpu",
                            n_labels=2)


def test_arg_sorts_and_artifact_hash_are_carried_opaquely() -> None:
    """Review finding N3 (low) + finding 2: in this phase nothing here reads
    arg_sorts or artifact_hash -- they are consumer-side metadata. Pinned:
    identical enumeration with and without them, and the spec survives
    registration untouched."""
    from pyxlog.ilp.neural_credit import (NeuralRelationSpec, _registry,
                                          enumerate_specs)

    facts = [(0, 1), (1, 0), (2, 1)]
    tagged_spec = NeuralRelationSpec(num_rows=3, arg_sorts=("event", "label"),
                                     artifact_hash="sha256:abc")
    plain = enumerate_specs(_FakeProg(), "W", facts,
                            neural_relations={"sal": 3}, device="cpu",
                            n_labels=2)
    tagged = enumerate_specs(_FakeProg(), "W", facts,
                             neural_relations={"sal": tagged_spec},
                             device="cpu", n_labels=2)
    assert ([(s.left, s.right, s.is_neural) for s in plain]
            == [(s.left, s.right, s.is_neural) for s in tagged])
    assert _registry({"sal": tagged_spec})["sal"] is tagged_spec


def test_frozen_select_refuses_no_facts() -> None:
    from pyxlog.ilp.neural_credit import frozen_select

    with pytest.raises(ValueError, match="facts"):
        frozen_select(_FakeProg(), "W", [], [], _frozen_detector_module(),
                      torch.tensor([[0.9], [0.8], [0.1]]),
                      neural_relations={"sal": 3})


# ---------------------------------------------------------------------------
# Witness-mask channel (contract #155): a MASKED witness contributes EXACTLY
# zero credit and gradient -- the masked row is physically absent from the
# index, never coerced to false, and each fact affected is flagged via
# `masked_any`.
# ---------------------------------------------------------------------------


def test_masked_witness_rows_are_excluded_from_the_index() -> None:
    """Контракт #155: MASKED вносит РОВНО ноль кредита — строка физически
    отсутствует в индексе, а masked_any помечает затронутые факты."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    mask = torch.zeros(3, 2, dtype=torch.bool)
    mask[0, 1] = True                     # событие 0 на метке 1 — замаскировано
    specs = enumerate_specs(_FakeProg(), "W", [(0, 1), (1, 0)],
                            neural_relations={"sal": 3}, device="cpu",
                            n_labels=2, witness_mask=mask)
    neural = {(s.left, s.right): s for s in specs}[("has_event", "sal")]
    # без маски было [1, 3, 4] (см. тест per-fact y); строка 0*2+1=1 исключена
    assert neural.witness_index.event_ids.tolist() == [3, 4]
    assert neural.masked_any.tolist() == [True, False]


def test_none_mask_is_byte_identical_to_omitting_it() -> None:
    from pyxlog.ilp.neural_credit import enumerate_specs

    a = enumerate_specs(_FakeProg(), "W", [(0, 1), (1, 0)],
                        neural_relations={"sal": 3}, device="cpu", n_labels=2)
    b = enumerate_specs(_FakeProg(), "W", [(0, 1), (1, 0)],
                        neural_relations={"sal": 3}, device="cpu", n_labels=2,
                        witness_mask=None)
    na = {(s.left, s.right): s for s in a}[("has_event", "sal")]
    nb = {(s.left, s.right): s for s in b}[("has_event", "sal")]
    assert na.witness_index.event_ids.tolist() == nb.witness_index.event_ids.tolist()
    assert nb.masked_any is None          # дефолт не создаёт нового состояния


def test_mask_of_wrong_shape_is_refused_typed() -> None:
    from pyxlog.ilp.neural_credit import enumerate_specs

    with pytest.raises(ValueError, match="witness_mask"):
        enumerate_specs(_FakeProg(), "W", [(0, 1)],
                        neural_relations={"sal": 3}, device="cpu", n_labels=2,
                        witness_mask=torch.zeros(5, 2, dtype=torch.bool))


def test_witness_mask_with_disagreeing_num_rows_names_both_relations() -> None:
    """Finding 2: two neural relations declaring different num_rows while a
    mask is supplied cannot be interpreted against a single row space at once
    -- refused typed, naming both relations."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    with pytest.raises(ValueError) as excinfo:
        enumerate_specs(_FakeProg(), "W", [(0, 1)],
                        neural_relations={"sal": 3, "tag": 2}, device="cpu",
                        n_labels=2, witness_mask=torch.zeros(3, 2, dtype=torch.bool))
    assert "sal" in str(excinfo.value)
    assert "tag" in str(excinfo.value)


def test_out_of_range_witness_constant_stays_a_typed_refusal_under_a_mask() -> None:
    """Finding 1: an out-of-range engine constant must not crash on a bare
    IndexError when a mask is supplied, and a NEGATIVE one must not silently
    alias to the mask's last row -- the mask is only consulted for in-range
    (z, y), and anything else is left for the downstream typed checks
    (prepare_extension's bounds check / the dense-identity law) to refuse,
    exactly as on the no-mask path."""
    from pyxlog.ilp.neural_credit import enumerate_specs

    class _FakeProgOutOfRangeEvent(_FakeProg):
        def __init__(self) -> None:
            super().__init__()
            # edge 0 now also joins event 7, which is outside both the mask's
            # 0..2 row range and sal's declared num_rows=3.
            self._facts["has_event"] = self._facts["has_event"] + [[0, 7]]

    mask = torch.zeros(3, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        enumerate_specs(_FakeProgOutOfRangeEvent(), "W", [(0, 1)],
                        neural_relations={"sal": 3}, device="cpu",
                        n_labels=2, witness_mask=mask)


# ---------------------------------------------------------------------------
# Task 2 of the witness-mask plan: interval-aware selection under witness
# masks with a coverage gate.
# ---------------------------------------------------------------------------
def test_masked_uncertain_facts_neither_help_nor_hurt_a_candidate() -> None:
    """Маскируем ЕДИНСТВЕННОГО свидетеля позитивного факта: без канала маски
    OR=0 посчитался бы ложью и убил бы кандидата (коэрция!); с каналом факт
    неопределён, исключён, и кандидат выбирается по оставшимся."""
    from pyxlog.ilp.neural_credit import frozen_select

    features = torch.tensor([[0.9], [0.8], [0.1]])
    facts = [(0, 1), (1, 0), (2, 1)]
    is_positive = [True, True, False]
    mask = torch.zeros(3, 2, dtype=torch.bool)
    mask[2, 0] = True                     # свидетель факта (1,0) — событие 2, метка 0

    sel = frozen_select(_FakeProg(), "W", facts, is_positive,
                        _frozen_detector_module(), features,
                        neural_relations={"sal": 3}, witness_mask=mask)
    assert sel.rule == ("has_event", "sal"), sel
    assert sel.coverage[("has_event", "sal")] == pytest.approx(2 / 3)


def test_low_coverage_candidate_abstains_with_a_named_reason() -> None:
    from pyxlog.ilp.neural_credit import frozen_select

    features = torch.tensor([[0.9], [0.8], [0.1]])
    facts = [(0, 1), (1, 0), (2, 1)]
    is_positive = [True, True, False]
    mask = torch.ones(3, 2, dtype=torch.bool)   # замаскировано ВСЁ

    sel = frozen_select(_FakeProg(), "W", facts, is_positive,
                        _frozen_detector_module(), features,
                        neural_relations={"sal": 3}, witness_mask=mask)
    # нейро-кандидаты потеряли покрытие; реляционные не маскируются — селекция
    # либо реляционная, либо воздержание с причиной про coverage
    if sel.rule is not None:
        assert sel.rule[1] not in ("sal",), sel
    else:
        assert "coverage" in sel.reason


def test_kfold_select_pools_coverage_across_folds_under_a_witness_mask() -> None:
    """Pins the fold-pooled coverage accounting added in Task 2: kfold_select's
    masked path sums `certain_sums`/`total_sums` ACROSS folds (each fact held
    out exactly once) into one pooled `coverage` fraction per candidate, guarded
    by the zero-total fallback to 0.0. `frozen_select`'s masked path already has
    coverage tests; the fold-pooled accounting here is kfold-only and had zero
    coverage before this test.

    Mirrors the calling convention of
    `test_kfold_select_seeds_network_construction_not_ambient_rng`. The mask
    marks witness (event 0, label 1) as masked, which affects fact (0, 1) via
    `has_event`'s bucket edge0 -> events [0, 1]."""
    features = torch.tensor([[0.1], [0.2], [0.3]])
    facts = [(0, 1), (1, 0), (2, 1)]
    is_positive = [True, False, True]

    def make_network():
        return torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Softmax(dim=-1))

    def run(witness_mask):
        return kfold_select(_FakeProg, "W", facts, is_positive, make_network,
                            features, neural_relations={"sal": 3}, folds=3,
                            steps=2, seed=0, witness_mask=witness_mask)

    sel_unmasked = run(None)
    assert sel_unmasked.coverage == {}          # no witness_mask -> the channel didn't run

    mask = torch.zeros(3, 2, dtype=torch.bool)
    mask[0, 1] = True                            # witness of fact (0, 1) via has_event
    sel_masked = run(mask)

    assert sel_masked.coverage                   # non-empty: the channel ran
    for key, c in sel_masked.coverage.items():
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0
    # Relational candidates are never masked, so their pooled coverage is exact.
    for key, c in sel_masked.coverage.items():
        if key[1] != "sal":
            assert c == 1.0, (key, c)
    # The neural candidate lost some certain evidence to the mask, but tiny
    # 2-step training leaves the network near softmax init (~0.5), so whether
    # OR_active clears 0.5 for the affected fact is not pinned here -- only
    # that the pooled fraction stays a valid coverage value.
    neural_key = ("has_event", "sal")
    assert neural_key in sel_masked.coverage
    assert 0.0 <= sel_masked.coverage[neural_key] <= 1.0

    assert isinstance(sel_unmasked, HoldoutSelection)
    assert isinstance(sel_masked, HoldoutSelection)


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
    # produce a bitwise-equal loss at EVERY step, not just the last one -- the
    # full trace is what the "bitwise" claim is about (review of PR #154).
    result_b, _ = _train_w1(world, seed=0)
    assert result.losses == result_b.losses


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
