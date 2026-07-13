"""The join extension must come FROM THE ENGINE. If this test passed while the map
were supplied from Python, it would prove nothing -- so it asserts against the
engine's own answer for facts the engine compiled.

API NOTE (correction found on the GPU box, 2026-07-13): the object the trainer's
mixture holds is `pyxlog.Program.compile(...)` -> `CompiledProgram`, whose only
read surface is `evaluate*` -- it does not expose facts at all (`EvalResult.atoms`
comes back empty for ground facts, and it has no `batch_fact_membership`).
`pyxlog.IlpProgramFactory.compile(...)` -> `CompiledIlpProgram` does expose facts,
via `relation_facts(name) -> list[list[int]]`, a direct enumeration of the
relation's extension. `read_join_extension` is built on that call.
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.join_bodies import JoinBody, read_join_extension
from pyxlog.ilp.neurosymbolic import (
    NeuralBodySpec,
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

_SOURCE = """
    pre_before_post(0, 0). pre_before_post(1, 0).
    pre_before_post(2, 1).
    pre_before_post(4, 2).
    pred pre_before_post(i64, i64).
"""

_JB = JoinBody(
    neural_predicate="saliency", network="sal_net", join_var="Ev",
    relation="pre_before_post", event_arg=0, head_arg=1,
)


def test_extension_is_read_from_the_engine() -> None:
    prog = pyxlog.IlpProgramFactory.compile(_SOURCE, device=0, memory_mb=64)
    prog.evaluate()
    ext = read_join_extension(prog, _JB, num_bindings=4)
    assert ext == [[0, 1], [2], [4], []]     # edge 3 joins nothing -> empty


def test_an_edge_with_no_joined_events_gets_an_empty_extension() -> None:
    prog = pyxlog.IlpProgramFactory.compile(_SOURCE, device=0, memory_mb=64)
    prog.evaluate()
    ext = read_join_extension(prog, _JB, num_bindings=4)
    assert ext[3] == []


def test_a_head_binding_outside_the_range_is_ignored() -> None:
    """The engine's relation may carry tuples whose head binding lies outside the
    supervised range 0..num_bindings-1 (here: edge 2, with num_bindings=2). Those
    tuples must be dropped, not crash and not shift the buckets."""
    prog = pyxlog.IlpProgramFactory.compile(_SOURCE, device=0, memory_mb=64)
    prog.evaluate()
    ext = read_join_extension(prog, _JB, num_bindings=2)
    assert ext == [[0, 1], [2]]     # the (4, 2) tuple is out of range -> ignored


# ---------------------------------------------------------------------------
# Task 3: a neural-JOIN candidate competes inside the multi-candidate mixture
# ---------------------------------------------------------------------------

_EVENT_FEATURES = [0.9, 0.1, 0.2, 0.15, 0.85, 0.1]
_PRE = {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}
# The distractor is EQUAL-CARDINALITY with pre_before_post (2, 2, 1, 1 events per
# edge) and takes each edge's events from OTHER edges. That is the plan's honest-
# distractor contract, and it is load-bearing, not cosmetic: with the unbalanced
# 1-event-per-edge distractor {0: [5], 1: [4], 2: [1], 3: [0]} the world has TWO
# exact global optima -- pre_before_post with a correct detector, and
# post_before_pre with an INVERTED one (that distractor is a bijection edge->event
# whose features anti-correlate perfectly with the labels, so p(0.1)=1, p(0.85)=0,
# p(0.9)=0 fits it to loss 0.0003). Which optimum SGD lands in is then decided by
# the init, not by the mechanism; measured on this branch, the inverted one won.
# With the balanced distractor no function of the event feature fits it (events 1
# and 5 share the feature 0.1 but would need opposite probabilities), so the
# correct relation + the correct per-event detector is the UNIQUE optimum.
_POST = {0: [2, 3], 1: [0, 1], 2: [5], 3: [4]}


def _world_source(
    candidates: str | None = None, extra_facts: str = "", extra_preds: str = ""
) -> str:
    """The two-relation world. ``candidates`` overrides the trainable_rule block so a
    test can vary ONLY the rule shape (body length, label term) against the same
    engine-owned facts."""
    pre = "\n".join(f"    pre_before_post({e}, {k})." for k in sorted(_PRE) for e in _PRE[k])
    post = "\n".join(f"    post_before_pre({e}, {k})." for k in sorted(_POST) for e in _POST[k])
    if candidates is None:
        candidates = """
        trainable_rule(cand_pre, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
        trainable_rule(cand_post, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
"""
    return f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{pre}
{post}
{extra_facts}
        pred pre_before_post(i64, i64).
        pred post_before_pre(i64, i64).
{extra_preds}
        pred plastic(i64).
{candidates}
        train(plastic, binary_cross_entropy).
    """


def _targets() -> list[float]:
    return [1.0 if any(_EVENT_FEATURES[e] > 0.5 for e in _PRE[k]) else 0.0 for k in sorted(_PRE)]


def test_a_stage_b_candidate_trains_in_the_multi_candidate_mixture() -> None:
    """Before this task it died with KeyError 'cand_pre' at neurosymbolic.py:530."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[f] for f in _EVENT_FEATURES], dtype=torch.float32)

    result = train_neurosymbolic_program(
        _world_source(),
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        examples=[{"targets": torch.tensor(_targets(), dtype=torch.float32)}],
        config=NeuroSymbolicTrainingConfig(steps=300, learning_rate=0.1),
    )

    # it trained, and gradient reached the PER-EVENT detector
    assert result.losses[-1] < result.losses[0]
    assert result.neural_parameter_grads["sal_net"] > 0.0
    # the correct timing relation won
    w = result.symbolic_rule_weights
    assert w["cand_pre"] > 0.7 and w["cand_post"] < 0.3, w


def _net_and_feats():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[f] for f in _EVENT_FEATURES], dtype=torch.float32)
    return net, feats


def _train(source: str, net, feats, steps: int = 300, lr: float = 0.1):
    return train_neurosymbolic_program(
        source,
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        examples=[{"targets": torch.tensor(_targets(), dtype=torch.float32)}],
        config=NeuroSymbolicTrainingConfig(steps=steps, learning_rate=lr),
    )


def test_a_longer_body_is_rejected_not_silently_trained() -> None:
    """`high_degree(E)` is a third conjunct. The join machinery has no mask for it, so
    the candidate must be REJECTED. Masking it as if the conjunct were absent would
    train `plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E)` — a rule
    nobody wrote — and report the result as if the written rule had trained."""
    net, feats = _net_and_feats()
    source = _world_source(
        candidates="""
        trainable_rule(cand_pre, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E), high_degree(E).
        trainable_rule(cand_post, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
""",
        extra_facts="    high_degree(0).\n    high_degree(1).\n    high_degree(2).\n    high_degree(3).",
        extra_preds="        pred high_degree(i64).",
    )
    with pytest.raises(ValueError) as exc:
        _train(source, net, feats, steps=5)
    assert "cand_pre" in str(exc.value)


def test_a_negated_join_relation_is_rejected_not_trained_as_its_inverse() -> None:
    """`not pre_before_post(Ev, E)` is a NEGATED literal, i.e. the complement of the
    join relation. Counting parenthesized atoms sees two atoms and cannot see the
    `not` at all, so before this fix the candidate parsed as the join shape and was
    masked with the extension of `pre_before_post` — training, and then reporting, the
    exact INVERSE of the rule that was written. It must be rejected."""
    net, feats = _net_and_feats()
    source = _world_source(
        candidates="""
        trainable_rule(cand_pre, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), not pre_before_post(Ev, E).
        trainable_rule(cand_post, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
"""
    )
    with pytest.raises(ValueError) as exc:
        _train(source, net, feats, steps=5)
    assert "cand_pre" in str(exc.value)


def test_a_comparison_literal_is_rejected_not_silently_dropped() -> None:
    """`Ev < 3` is a Comparison body literal, not an atom. An atom count cannot see it,
    so before this fix the body parsed as the two-literal join shape and the comparison
    was silently DISCARDED."""
    net, feats = _net_and_feats()
    source = _world_source(
        candidates="""
        trainable_rule(cand_pre, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E), Ev < 3.
        trainable_rule(cand_post, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
"""
    )
    with pytest.raises(ValueError) as exc:
        _train(source, net, feats, steps=5)
    assert "cand_pre" in str(exc.value)


def test_a_join_candidate_is_never_trained_on_an_engine_mask_alone() -> None:
    """Today a join candidate reaches the join path only because the engine emits no
    eligibility mask for it (its mask is keyed by the FIRST neural group's predicate,
    which for a join candidate yields a junk key). That is an upstream BUG, and the
    join path must not depend on it: if the keying is fixed, a join candidate arrives
    WITH a hard-filters-only (all-True) mask and — if routing were decided by "no mask"
    — would silently train as an always-true relational candidate, with no gradient to
    the per-event detector and no error.

    This simulates that upstream fix by wrapping the compiled program so
    `joint_candidate_eligibility` DOES return an all-True mask for every candidate, and
    asserts the join path still owns the candidate: the detector still gets gradient and
    the correct relation still wins. Before the routing fix, `sal_net` receives NO
    gradient here."""
    net, feats = _net_and_feats()
    real_program_cls = pyxlog.Program

    class _Fixed:
        """Delegates everything to the real program, except that the eligibility read
        also emits a mask for candidates the engine currently skips."""

        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_inner"), name)

        def joint_candidate_eligibility(self, head, lo, n):
            inner = object.__getattribute__(self, "_inner")
            out = list(inner.joint_candidate_eligibility(head, lo, n))
            present = {guard_pred for guard_pred, _ in out}
            for rule_id in ("cand_pre", "cand_post"):
                guard_pred = f"nsr_guard_{rule_id}"
                if guard_pred not in present:
                    out.append((guard_pred, [True] * n))   # hard filters only
            return out

    class _Factory:
        @staticmethod
        def compile(*args, **kwargs):
            return _Fixed(real_program_cls.compile(*args, **kwargs))

    pyxlog.Program = _Factory
    try:
        result = _train(_world_source(), net, feats, steps=300)
    finally:
        pyxlog.Program = real_program_cls

    assert result.neural_parameter_grads["sal_net"] > 0.0
    w = result.symbolic_rule_weights
    assert w["cand_pre"] > 0.7 and w["cand_post"] < 0.3, w


def test_the_join_index_is_built_once_per_candidate_not_once_per_step() -> None:
    """I2: the extension is STATIC, so its device index is built OUTSIDE the hot loop.
    A per-step rebuild would be a host->device copy every step and would not fail any
    other test, so it is pinned by counting the calls across a real multi-step run:
    two join candidates, many steps, exactly two `prepare_extension` calls."""
    import pyxlog.ilp.neurosymbolic as ns

    net, feats = _net_and_feats()
    real_prepare = ns.prepare_extension
    calls = []

    def counting_prepare(extension, device):
        calls.append(len(extension))
        return real_prepare(extension, device)

    ns.prepare_extension = counting_prepare
    try:
        _train(_world_source(), net, feats, steps=25)
    finally:
        ns.prepare_extension = real_prepare

    assert len(calls) == 2, calls      # once per join candidate, NOT once per step


def test_a_purely_relational_existential_join_is_still_rejected() -> None:
    """`plastic(E) :- pre_before_post(Ev, E).` has an existential Ev that is NOT a
    neural predicate's input. It was rejected before this branch and must still be:
    the join machinery only ever fires for the neural-join shape."""
    net, feats = _net_and_feats()
    source = _world_source(
        candidates="""
        trainable_rule(cand_pre, weight=0.0) :: plastic(E) :- pre_before_post(Ev, E).
        trainable_rule(cand_post, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
"""
    )
    with pytest.raises(ValueError) as exc:
        _train(source, net, feats, steps=5)
    assert "Ev" in str(exc.value)


def test_a_network_named_like_a_candidate_is_refused() -> None:
    """`neural_parameter_grads` is ONE flat map keyed both by nn/4 network name (join
    bodies) and by candidate rule id (NeuralBodySpec bodies). A network named exactly
    like a neural-bodied candidate would have its gradient silently overwritten, so the
    collision is refused rather than reported as a wrong number."""
    net, feats = _net_and_feats()
    source = _world_source(
        candidates="""
        trainable_rule(sal_net, weight=0.0) :: plastic(E) :- high_degree(E).
        trainable_rule(cand_pre, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
""",
        extra_facts="    high_degree(0).\n    high_degree(1).\n    high_degree(2).\n    high_degree(3).",
        extra_preds="        pred high_degree(i64).",
    )
    with pytest.raises(ValueError, match="must not collide"):
        train_neurosymbolic_program(
            source,
            networks={"sal_net": net},
            domain_inputs={"sal_net": feats},
            examples=[{"targets": torch.tensor(_targets(), dtype=torch.float32)}],
            neural_bodies={"sal_net": NeuralBodySpec(features=torch.rand(4, 3))},
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.1),
        )


def test_the_positive_label_column_comes_from_the_rule_not_a_hardcoded_index() -> None:
    """`strengthen` happens to be column 1, so a hardcoded `[:, 1]` passes every other
    test in this file. Here `cand_low` names the OTHER label (`low`, column 0), and the
    mask is pinned analytically: with learning_rate=0 the guards stay at their declared
    logit 0.0 (sigmoid 0.5), so the head probability is exactly
    ``1 - (1 - 0.5*m_low)(1 - 0.5*m_str)``. A hardcoded column-1 mask gives different
    numbers, and the test asserts it does."""
    net, feats = _net_and_feats()
    source = _world_source(
        candidates="""
        trainable_rule(cand_low, weight=0.0) :: plastic(E) :- saliency(Ev, low), pre_before_post(Ev, E).
        trainable_rule(cand_str, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
"""
    )
    result = _train(source, net, feats, steps=1, lr=0.0)

    with torch.no_grad():
        device = next(net.parameters()).device
        p = net(feats.to(device)).cpu()      # [n_events, 2]; column 0 = low, 1 = strengthen

    def noisy_or(col: int, ext: dict[int, list[int]], k: int) -> float:
        q = 1.0
        for e in ext[k]:
            q *= 1.0 - float(p[e, col])
        return 1.0 - q

    def head(low_col: int) -> list[float]:
        # low_col is the column the cand_low mask reads: 0 is the rule's label,
        # 1 is what a hardcoded index would (wrongly) read.
        return [
            1.0
            - (1.0 - 0.5 * noisy_or(low_col, _PRE, k))
            * (1.0 - 0.5 * noisy_or(1, _POST, k))
            for k in sorted(_PRE)
        ]

    got = result.query_probabilities
    from_the_rule = head(0)
    from_a_hardcoded_index = head(1)
    assert got == pytest.approx(from_the_rule, abs=1e-5), (got, from_the_rule)
    # and the two really are distinguishable, so this test can fail
    assert got != pytest.approx(from_a_hardcoded_index, abs=1e-3)


def _sparse_source() -> str:
    facts = "\n".join(
        f"    pbp({e}, {k})." for k, evs in {0: [0, 2], 1: [4, 6]}.items() for e in evs
    )
    return f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{facts}
        pred pbp(i64, i64).
        pred plastic(i64).
        trainable_rule(c_a, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pbp(Ev, E).
        trainable_rule(c_b, weight=0.0) :: plastic(E) :- saliency(Ev, low), pbp(Ev, E).
        train(plastic, binary_cross_entropy).
    """


def test_a_sparse_join_domain_without_domain_ids_is_refused() -> None:
    """domain_inputs carries no labels, so "which row is which constant" is a
    CONVENTION. Omitting `domain_ids` states the DENSE one (row j = constant j) -- which
    on the domain {0, 2, 4, 6} is simply false: constant 6 has no row. Before R3 that was
    either a CUDA device-side assert (which poisons the context, so every later op in the
    process fails) or, on a padded feature tensor, a silent read of a DIFFERENT row than
    the circuit's, disagreeing with it by 0.307. It is now a typed refusal that NAMES the
    constant with no row."""
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[0.9], [0.1], [0.2], [0.15]], dtype=torch.float32)

    with pytest.raises(ValueError, match="not in domain_ids"):
        train_neurosymbolic_program(
            _sparse_source(),
            networks={"sal_net": net},
            domain_inputs={"sal_net": feats},
            examples=[{"targets": torch.tensor([1.0, 0.0], dtype=torch.float32)}],
            config=NeuroSymbolicTrainingConfig(steps=5, learning_rate=0.1),
        )


def test_a_sparse_join_domain_with_explicit_domain_ids_trains() -> None:
    """The other half of the new contract: SAY which row holds which constant and the
    sparse domain is no longer special -- the mixture trains on it, and gradient still
    reaches the per-event detector. (That the numbers it computes are the exact
    circuit's is pinned by the sparse semantics anchor.)"""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[0.9], [0.1], [0.2], [0.15]], dtype=torch.float32)

    result = train_neurosymbolic_program(
        _sparse_source(),
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        domain_ids={"sal_net": [0, 2, 4, 6]},
        examples=[{"targets": torch.tensor([1.0, 0.0], dtype=torch.float32)}],
        config=NeuroSymbolicTrainingConfig(steps=100, learning_rate=0.1),
    )
    assert result.losses[-1] < result.losses[0]
    assert result.neural_parameter_grads["sal_net"] > 0.0


def test_unsorted_domain_ids_are_refused() -> None:
    """One row per constant, in a stable ascending layout, is a stated requirement on the
    caller's feature tensor -- unsorted (or duplicated) ids are refused rather than
    guessed at. (What reconciles the two engines is that both resolve a constant through
    this same id list; the ordering is a layout rule, not the reconciliation.)"""
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[0.9], [0.1], [0.2], [0.15]], dtype=torch.float32)

    with pytest.raises(ValueError, match="strictly increasing"):
        train_neurosymbolic_program(
            _sparse_source(),
            networks={"sal_net": net},
            domain_inputs={"sal_net": feats},
            domain_ids={"sal_net": [0, 4, 2, 6]},
            examples=[{"targets": torch.tensor([1.0, 0.0], dtype=torch.float32)}],
            config=NeuroSymbolicTrainingConfig(steps=5, learning_rate=0.1),
        )


def test_domain_ids_must_have_one_id_per_feature_row() -> None:
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[0.9], [0.1], [0.2], [0.15]], dtype=torch.float32)

    with pytest.raises(ValueError, match="exactly one id per row"):
        train_neurosymbolic_program(
            _sparse_source(),
            networks={"sal_net": net},
            domain_inputs={"sal_net": feats},
            domain_ids={"sal_net": [0, 2, 4]},          # 3 ids, 4 rows
            examples=[{"targets": torch.tensor([1.0, 0.0], dtype=torch.float32)}],
            config=NeuroSymbolicTrainingConfig(steps=5, learning_rate=0.1),
        )


def test_domain_ids_for_a_network_with_no_domain_inputs_is_refused() -> None:
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[0.9], [0.1], [0.2], [0.15]], dtype=torch.float32)

    with pytest.raises(ValueError, match="no domain_inputs"):
        train_neurosymbolic_program(
            _sparse_source(),
            networks={"sal_net": net},
            domain_inputs={"sal_net": feats},
            domain_ids={"sal_net": [0, 2, 4, 6], "ghost_net": [0]},
            examples=[{"targets": torch.tensor([1.0, 0.0], dtype=torch.float32)}],
            config=NeuroSymbolicTrainingConfig(steps=5, learning_rate=0.1),
        )
