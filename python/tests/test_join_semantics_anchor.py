"""SEMANTICS ANCHOR. The joint mixture is a SURROGATE (a torch noisy-OR), while the
single-rule path compiles an EXACT d-DNNF circuit whose provenance OR-aggregates the
joined events. On a world small enough for the circuit, the two must agree.

If they diverge, our claim collapses -- so this test, not the flagship, is what
licenses us to say the torch-side path computes Stage-B semantics.

API NOTE. The plan's sketch predates the module: the extension is read with
``read_join_extension(ilp_program, jb, num_bindings)`` off a ``CompiledIlpProgram``
(``pyxlog.Program`` cannot enumerate facts at all -- see test_join_bodies_engine.py),
and the read handle is opened by the trainer's OWN helpers (``_read_only_source`` +
``_open_join_read_handle``), reused here rather than re-implemented, so the anchor
compares against exactly the handle the mixture reads from.
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.join_bodies import (
    JoinBody,
    noisy_or_over_extension,
    read_join_extension,
    translate_extension_to_rows,
)
from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    _open_join_read_handle,
    _read_only_source,
    train_neurosymbolic_program,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

# 6 events -- inside the exact compiler's ceiling (it caps around 6-7)
_EF = [0.9, 0.1, 0.2, 0.15, 0.85, 0.1]
_EDGES = {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}

_JB = JoinBody(
    neural_predicate="saliency",
    network="sal_net",
    join_var="Event",
    relation="pre_before_post",
    event_arg=0,
    head_arg=1,
)


def _source() -> str:
    facts = "\n".join(
        f"    pre_before_post({e}, {k})." for k in sorted(_EDGES) for e in _EDGES[k]
    )
    return f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{facts}
        pred pre_before_post(i64, i64).
        pred plastic(i64).
        trainable_rule(rule_plastic, weight=0.0) :: plastic(Edge) :- saliency(Event, strengthen), pre_before_post(Event, Edge).
        train(plastic, binary_cross_entropy).
    """


def test_torch_side_or_reproduces_the_exact_circuit() -> None:
    source = _source()
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[f] for f in _EF], dtype=torch.float32)
    targets = [
        1.0 if any(_EF[e] > 0.5 for e in _EDGES[k]) else 0.0 for k in sorted(_EDGES)
    ]

    # ONE trainable_rule -> the EXACT circuit path (Stage B), not the mixture.
    config = NeuroSymbolicTrainingConfig(steps=120, learning_rate=0.15)
    result = train_neurosymbolic_program(
        source,
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        examples=[{"targets": torch.tensor(targets, dtype=torch.float32)}],
        config=config,
    )
    exact = torch.tensor(result.query_probabilities, dtype=torch.float64)

    # Now recompute the SAME quantity with our torch-side formula, using the SAME
    # trained network and the SAME learned guard:
    #     P(plastic(E)) = sigma(w) * (1 - PROD_{e in ext(E)} (1 - p_sal(e)))
    #
    # The extension comes from the ENGINE (relation_facts), never from the _EDGES
    # dict the facts were written from: if it came from Python, the OR would be
    # Python's aggregation over a caller-supplied hint, and this test would be
    # comparing the circuit against a hand-written answer instead of against the
    # logic's own.
    ilp_read = _open_join_read_handle(_read_only_source(source), config)
    ext = read_join_extension(ilp_read, _JB, num_bindings=len(_EDGES))
    assert ext == [_EDGES[k] for k in sorted(_EDGES)], "engine extension != planted graph"

    # Which column is "strengthen" is the ENGINE's answer, not a hardcoded 1.
    label_reader = pyxlog.Program.compile(
        _read_only_source(source), device=config.device, memory_mb=config.gpu_memory_mb
    )
    positive = int(label_reader.label_to_index("saliency", "strengthen"))

    with torch.no_grad():
        dev = next(net.parameters()).device
        p_event = net(feats.to(dev))[:, positive]
        or_ = noisy_or_over_extension(p_event, ext, dev).double().cpu()
    # symbolic_rule_weights is ALREADY sigma(w) (neurosymbolic.py: learned_weights =
    # sigmoid(logit)), so it multiplies the OR directly -- no second sigmoid.
    guard = float(result.symbolic_rule_weights["rule_plastic"])
    ours = guard * or_

    deviation = float((ours - exact).abs().max())
    assert torch.allclose(ours, exact, atol=1e-4), (
        f"\nours ={ours.tolist()}\nexact={exact.tolist()}\nmax|dev|={deviation:.3e}"
    )
    print(f"\nexact (d-DNNF circuit) = {exact.tolist()}")
    print(f"ours  (torch-side OR)  = {ours.tolist()}")
    print(f"guard sigma(w)         = {guard!r}")
    print(f"max abs deviation      = {deviation:.3e}")


# ---------------------------------------------------------------------------
# THE SPARSE ANCHOR. The dense anchor above cannot see the indexing convention at
# all: with ids 0..5 the constant IS its own row, so rank indexing (the circuit)
# and identity indexing (our torch path) coincide. That coincidence is exactly
# what hid the bug -- so the anchor is re-run here on a SPARSE domain, where the
# two are only reconciled by `domain_ids` -- the one constant->row map both the
# circuit and the torch path resolve through.
# ---------------------------------------------------------------------------

_SPARSE_IDS = [0, 2, 4, 6, 8, 10]                     # the join domain: NOT 0..5
_SPARSE_EDGES = {0: [0, 2], 1: [4, 6], 2: [8], 3: [10]}

_SPARSE_JB = JoinBody(
    neural_predicate="saliency",
    network="sal_net",
    join_var="Event",
    relation="pbp",
    event_arg=0,
    head_arg=1,
)


def _sparse_source() -> str:
    facts = "\n".join(
        f"    pbp({e}, {k})." for k in sorted(_SPARSE_EDGES) for e in _SPARSE_EDGES[k]
    )
    return f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{facts}
        pred pbp(i64, i64).
        pred plastic(i64).
        trainable_rule(rule_plastic, weight=0.0) :: plastic(Edge) :- saliency(Event, strengthen), pbp(Event, Edge).
        train(plastic, binary_cross_entropy).
    """


def test_torch_side_or_reproduces_the_exact_circuit_on_a_sparse_domain() -> None:
    """The acceptance test for R3. Same claim as the dense anchor, on a domain whose
    constants are NOT their own row numbers: both engines read the row `domain_ids`
    says holds the constant they grounded at. If R3 reconciles the two, they agree to
    1e-4 here too."""
    source = _sparse_source()
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[f] for f in _EF], dtype=torch.float32)   # 6 rows, sparse ids
    row_of = {c: j for j, c in enumerate(_SPARSE_IDS)}
    targets = [
        1.0 if any(_EF[row_of[e]] > 0.5 for e in _SPARSE_EDGES[k]) else 0.0
        for k in sorted(_SPARSE_EDGES)
    ]

    config = NeuroSymbolicTrainingConfig(steps=120, learning_rate=0.15)
    result = train_neurosymbolic_program(
        source,
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        domain_ids={"sal_net": _SPARSE_IDS},
        examples=[{"targets": torch.tensor(targets, dtype=torch.float32)}],
        config=config,
    )
    exact = torch.tensor(result.query_probabilities, dtype=torch.float64)

    # The extension still comes from the ENGINE, in RAW constants; `domain_ids` only
    # says which ROW holds which constant, and the translation is the branch's own.
    ilp_read = _open_join_read_handle(_read_only_source(source), config)
    ext = read_join_extension(ilp_read, _SPARSE_JB, num_bindings=len(_SPARSE_EDGES))
    assert ext == [_SPARSE_EDGES[k] for k in sorted(_SPARSE_EDGES)]
    rows = translate_extension_to_rows(ext, _SPARSE_IDS, network="sal_net")
    assert rows == [[0, 1], [2, 3], [4], [5]]

    label_reader = pyxlog.Program.compile(
        _read_only_source(source), device=config.device, memory_mb=config.gpu_memory_mb
    )
    positive = int(label_reader.label_to_index("saliency", "strengthen"))

    with torch.no_grad():
        dev = next(net.parameters()).device
        p_event = net(feats.to(dev))[:, positive]
        or_ = noisy_or_over_extension(p_event, rows, dev).double().cpu()
    guard = float(result.symbolic_rule_weights["rule_plastic"])
    ours = guard * or_

    deviation = float((ours - exact).abs().max())
    print(f"\n[sparse] exact (d-DNNF circuit) = {exact.tolist()}")
    print(f"[sparse] ours  (torch-side OR)  = {ours.tolist()}")
    print(f"[sparse] guard sigma(w)         = {guard!r}")
    print(f"[sparse] max abs deviation      = {deviation:.3e}")
    assert torch.allclose(ours, exact, atol=1e-4), (
        f"\nours ={ours.tolist()}\nexact={exact.tolist()}\nmax|dev|={deviation:.3e}"
    )


# ---------------------------------------------------------------------------
# THE SUPERSET ANCHOR. The sparse anchor still hands the circuit a `domain_ids`
# list that is EXACTLY the join relation's domain, so "row j" and "the j-th
# constant the relation joins" still coincide. That is the last coincidence
# propping the two conventions up. Here the caller supplies features for MORE
# constants than the relation joins (entirely natural: features for every event,
# but only some events are pre->post). The relation's own domain is then a strict
# SUBSET of the ids, the ranks SHIFT, and rank-indexing reads the wrong rows.
# The row must be looked up BY CONSTANT, through `domain_ids`, in BOTH engines.
# ---------------------------------------------------------------------------

# Features for constants 0..5; the relation never joins constant 0.
_SUPER_IDS = [0, 1, 2, 3, 4, 5]
_SUPER_EF = [0.02, 0.90, 0.03, 0.85, 0.05, 0.10]
_SUPER_EDGES = {0: [1, 2], 1: [3], 2: [4, 5]}    # joins {1,2,3,4,5}: 0 is absent

_SUPER_JB = JoinBody(
    neural_predicate="saliency",
    network="sal_net",
    join_var="Event",
    relation="pbp",
    event_arg=0,
    head_arg=1,
)


def _superset_source() -> str:
    facts = "\n".join(
        f"    pbp({e}, {k})." for k in sorted(_SUPER_EDGES) for e in _SUPER_EDGES[k]
    )
    return f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{facts}
        pred pbp(i64, i64).
        pred plastic(i64).
        trainable_rule(rule_plastic, weight=0.0) :: plastic(Edge) :- saliency(Event, strengthen), pbp(Event, Edge).
        train(plastic, binary_cross_entropy).
    """


def test_torch_side_or_reproduces_the_exact_circuit_on_a_superset_domain() -> None:
    """The circuit must read the feature row `domain_ids` assigns to the constant it
    grounded at -- not the rank of that constant inside the relation's own domain.
    The world is discriminating: edge 0 and edge 1 join salient events, edge 2 joins
    two quiet ones, so a row mix-up moves the probabilities well past 1e-4."""
    source = _superset_source()
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[f] for f in _SUPER_EF], dtype=torch.float32)
    row_of = {c: j for j, c in enumerate(_SUPER_IDS)}
    targets = [
        1.0 if any(_SUPER_EF[row_of[e]] > 0.5 for e in _SUPER_EDGES[k]) else 0.0
        for k in sorted(_SUPER_EDGES)
    ]
    assert set(targets) == {0.0, 1.0}, "the world must discriminate, or it proves nothing"

    config = NeuroSymbolicTrainingConfig(steps=120, learning_rate=0.15)
    result = train_neurosymbolic_program(
        source,
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        domain_ids={"sal_net": _SUPER_IDS},
        examples=[{"targets": torch.tensor(targets, dtype=torch.float32)}],
        config=config,
    )
    exact = torch.tensor(result.query_probabilities, dtype=torch.float64)

    ilp_read = _open_join_read_handle(_read_only_source(source), config)
    ext = read_join_extension(ilp_read, _SUPER_JB, num_bindings=len(_SUPER_EDGES))
    assert ext == [_SUPER_EDGES[k] for k in sorted(_SUPER_EDGES)]
    rows = translate_extension_to_rows(ext, _SUPER_IDS, network="sal_net")
    assert rows == [[1, 2], [3], [4, 5]]      # the ids ARE the rows here; the RANKS are not

    label_reader = pyxlog.Program.compile(
        _read_only_source(source), device=config.device, memory_mb=config.gpu_memory_mb
    )
    positive = int(label_reader.label_to_index("saliency", "strengthen"))

    with torch.no_grad():
        dev = next(net.parameters()).device
        p_event = net(feats.to(dev))[:, positive]
        or_ = noisy_or_over_extension(p_event, rows, dev).double().cpu()
    guard = float(result.symbolic_rule_weights["rule_plastic"])
    ours = guard * or_

    deviation = float((ours - exact).abs().max())
    print(f"\n[superset] exact (d-DNNF circuit) = {exact.tolist()}")
    print(f"[superset] ours  (torch-side OR)  = {ours.tolist()}")
    print(f"[superset] guard sigma(w)         = {guard!r}")
    print(f"[superset] max abs deviation      = {deviation:.3e}")
    assert torch.allclose(ours, exact, atol=1e-4), (
        f"\nours ={ours.tolist()}\nexact={exact.tolist()}\nmax|dev|={deviation:.3e}"
    )


def test_a_joined_constant_absent_from_domain_ids_is_refused() -> None:
    """A constant the relation joins but `domain_ids` never mentions has NO feature
    row. The circuit must say so, naming it -- never silently ground on some other
    constant's row."""
    source = """
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
        pbp(1, 0).
        pbp(7, 1).
        pred pbp(i64, i64).
        pred plastic(i64).
        trainable_rule(rule_plastic, weight=0.0) :: plastic(Edge) :- saliency(Event, strengthen), pbp(Event, Edge).
        train(plastic, binary_cross_entropy).
    """
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[0.1], [0.9], [0.5]], dtype=torch.float32)

    with pytest.raises(ValueError, match="7"):
        train_neurosymbolic_program(
            source,
            networks={"sal_net": net},
            domain_inputs={"sal_net": feats},
            domain_ids={"sal_net": [0, 1, 2]},          # 7 is joined, but has no row
            examples=[{"targets": torch.tensor([1.0, 0.0], dtype=torch.float32)}],
            config=NeuroSymbolicTrainingConfig(steps=1),
        )


def test_a_mixture_trains_on_a_sparse_domain() -> None:
    """The multi-candidate mixture (2+ same-head join candidates) on the SAME sparse
    world. This is what the old dense-range stopgap refused outright; with explicit
    `domain_ids` it must simply train."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[f] for f in _EF], dtype=torch.float32)
    row_of = {c: j for j, c in enumerate(_SPARSE_IDS)}
    facts = "\n".join(
        f"    pbp({e}, {k})." for k in sorted(_SPARSE_EDGES) for e in _SPARSE_EDGES[k]
    )
    source = f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{facts}
        pred pbp(i64, i64).
        pred plastic(i64).
        trainable_rule(c_a, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pbp(Ev, E).
        trainable_rule(c_b, weight=0.0) :: plastic(E) :- saliency(Ev, low), pbp(Ev, E).
        train(plastic, binary_cross_entropy).
    """
    targets = [
        1.0 if any(_EF[row_of[e]] > 0.5 for e in _SPARSE_EDGES[k]) else 0.0
        for k in sorted(_SPARSE_EDGES)
    ]

    result = train_neurosymbolic_program(
        source,
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        domain_ids={"sal_net": _SPARSE_IDS},
        examples=[{"targets": torch.tensor(targets, dtype=torch.float32)}],
        config=NeuroSymbolicTrainingConfig(steps=200, learning_rate=0.1),
    )
    assert result.losses[-1] < result.losses[0]
    assert result.neural_parameter_grads["sal_net"] > 0.0
    assert len(result.query_probabilities) == len(targets)
