"""Carrier binding law: Python sees the exact device allocations, never
copies, and every session misstep is a typed refusal.

Exercised through the real PyO3 surface on live CUDA with torch as the
DLPack consumer: exported views alias one device allocation (a write
through one view is visible through a fresh export of the same buffer);
the cold-path session lifecycle refuses typed at each precondition
(unregistered schema, duplicate registration, rebinding, shape
mismatch, out-of-range abstain); the on-device existential semantics
match the engine's own proof (plural domain holds a label feasible,
empty signature intersection cuts it, abstain stays unconditionally
feasible); and the session fuel meter saturates across Python calls —
a retry repeats the identical typed refusal.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("live CUDA required", allow_module_level=True)

import pyxlog
from torch.utils.dlpack import from_dlpack

CATALOG_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

ENTITIES = 3
LANES = 1
CANDIDATES = 2
LABELS = 4
ABSTAIN = 3


def _carrier(fuel_limit=64):
    return pyxlog.JointConstraintCarrier(
        0, ENTITIES, LANES, CANDIDATES, LABELS, fuel_limit
    )


def _registered(fuel_limit=64):
    carrier = _carrier(fuel_limit)
    carrier.register_schema(CATALOG_SHA, pyxlog.SOLVER_ABI_IDENTITY)
    # Signature masks are labels x lanes u64 words. Head side: labels
    # 0/1 accept sort-bit 0, label 2 accepts only sort-bit 1, abstain
    # label 3 accepts nothing by signature. Tail side: labels 0/2
    # accept sort-bit 0, label 1 accepts only sort-bit 2.
    head = [0b01, 0b01, 0b10, 0b00]
    tail = [0b01, 0b100, 0b01, 0b00]
    carrier.bind_signatures(head, tail)
    return carrier


def test_exported_views_alias_one_allocation() -> None:
    carrier = _carrier()
    scores_a = from_dlpack(carrier.export_buffer("scores"))
    assert scores_a.shape == (CANDIDATES, LABELS)
    assert scores_a.dtype == torch.float32
    assert scores_a.is_cuda
    scores_a.fill_(0.0)
    scores_a[1, 2] = 7.5
    scores_b = from_dlpack(carrier.export_buffer("scores"))
    torch.cuda.synchronize()
    assert scores_b[1, 2].item() == 7.5, "fresh export must alias the same bytes"

    counts = from_dlpack(carrier.export_buffer("logical_counts"))
    assert counts.shape == (1, 4)


def test_session_lifecycle_refuses_typed() -> None:
    carrier = _carrier()
    with pytest.raises(pyxlog.CarrierRefused, match="not registered"):
        carrier.bind_signatures([0] * LABELS, [0] * LABELS)
    with pytest.raises(pyxlog.CarrierRefused, match="not registered"):
        carrier.solve_label_feasibility(ABSTAIN)

    carrier.register_schema(CATALOG_SHA, pyxlog.SOLVER_ABI_IDENTITY)
    with pytest.raises(pyxlog.CarrierRefused, match="already registered"):
        carrier.register_schema(CATALOG_SHA, pyxlog.SOLVER_ABI_IDENTITY)
    with pytest.raises(pyxlog.CarrierRefused, match="head"):
        carrier.bind_signatures([0] * (LABELS - 1), [0] * LABELS)

    carrier.bind_signatures([0] * LABELS, [0] * LABELS)
    with pytest.raises(pyxlog.CarrierRefused, match="already bound"):
        carrier.bind_signatures([0] * LABELS, [0] * LABELS)
    with pytest.raises(pyxlog.CarrierRefused, match="abstain"):
        carrier.solve_label_feasibility(LABELS)

    with pytest.raises(ValueError, match="valid names"):
        carrier.export_buffer("nonsense")


def test_existential_feasibility_matches_engine_semantics() -> None:
    carrier = _registered()
    domains = from_dlpack(carrier.export_buffer("domains"))
    constraints = from_dlpack(carrier.export_buffer("constraints"))
    # Entity 0: plural domain {sort0, sort1}; entity 1: {sort0};
    # entity 2: {sort2}.
    domains[0, 0] = 0b011
    domains[1, 0] = 0b001
    domains[2, 0] = 0b100
    # Candidate 0 = (head e0, tail e1); candidate 1 = (head e1, tail e1).
    constraints_host = torch.tensor(
        [[0, 1], [1, 1]], dtype=torch.int32, device="cuda"
    )
    constraints.copy_(constraints_host)
    torch.cuda.synchronize()

    carrier.solve_label_feasibility(ABSTAIN)
    torch.cuda.synchronize()

    feasible = from_dlpack(carrier.export_buffer("feasible_sets"))
    sets = feasible[:, 0].cpu()
    # Candidate 0: label 0 (head bit0 x tail bit0) holds via e0's
    # plural domain and e1; label 1 needs tail sort-bit 2 — e1 lacks
    # it; label 2 needs head sort-bit 1 — e0's PLURAL domain holds it
    # existentially; abstain 3 always feasible.
    assert sets[0].item() & 0b0001, "label 0 must stay feasible"
    assert not sets[0].item() & 0b0010, "label 1 must be cut (tail empty)"
    assert sets[0].item() & 0b0100, "plural head domain holds label 2"
    assert sets[0].item() & 0b1000, "abstain is unconditional"
    # Candidate 1: head e1 = {sort0} only — label 2 cut; label 0 holds.
    assert sets[1].item() & 0b0001
    assert not sets[1].item() & 0b0100, "singleton head domain cuts label 2"
    assert sets[1].item() & 0b1000

    counts = from_dlpack(carrier.export_buffer("outputs")).cpu()
    assert counts[0, 0].item() == 3
    assert counts[1, 0].item() == 2


def test_top2_stage_order_and_feasible_max_through_python() -> None:
    import struct

    carrier = _registered()
    with pytest.raises(pyxlog.CarrierRefused, match="feasibility"):
        carrier.solve_label_map_top2()

    domains = from_dlpack(carrier.export_buffer("domains"))
    constraints = from_dlpack(carrier.export_buffer("constraints"))
    scores = from_dlpack(carrier.export_buffer("scores"))
    domains[0, 0] = 0b001
    domains[1, 0] = 0b001
    constraints.copy_(
        torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="cuda")
    )
    # Candidate 0: label 1 is INFEASIBLE for these domains (tail needs
    # sort-bit 2) but carries the highest raw score — it must be
    # ignored; feasible label 0 (score 2.0) must win over abstain
    # (score 0.5), margin 1.5.
    scores.copy_(
        torch.tensor(
            [[2.0, 9.0, 1.0, 0.5], [1.0, 9.0, 1.0, 1.0]],
            dtype=torch.float32,
            device="cuda",
        )
    )
    torch.cuda.synchronize()

    carrier.solve_label_feasibility(ABSTAIN)
    carrier.solve_label_map_top2()
    torch.cuda.synchronize()

    map_results = from_dlpack(carrier.export_buffer("map_results")).cpu()
    best, ambiguous, _, margin_bits = (int(v) for v in map_results[0])
    assert best == 0, "best is the FEASIBLE maximum, not the raw one"
    assert ambiguous == 0
    assert struct.unpack("f", struct.pack("I", margin_bits))[0] == 1.5
    # Candidate 1: feasible labels 0 and abstain tie at 1.0 — typed
    # ambiguity, never a unique emission.
    assert int(map_results[1][1]) == 1, "tied maximum must flag ambiguity"
    assert carrier.fuel_spent == 2 * CANDIDATES * LABELS


def test_producer_event_replaces_host_sync() -> None:
    import struct

    carrier = _registered()
    domains = from_dlpack(carrier.export_buffer("domains"))
    constraints = from_dlpack(carrier.export_buffer("constraints"))
    scores = from_dlpack(carrier.export_buffer("scores"))
    # Producer writes go on a DEDICATED stream (torch's default
    # stream has raw handle 0, indistinguishable from a null handle,
    # and is refused); the noted event replaces every host barrier
    # between producer and solve.
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        domains[0, 0] = 0b001
        domains[1, 0] = 0b001
        constraints.copy_(
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device="cuda")
        )
        scores.copy_(
            torch.tensor(
                [[2.0, 9.0, 1.0, 0.5], [0.25, 9.0, 0.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
            )
        )
        carrier.note_producer_stream(producer.cuda_stream)
    carrier.solve_label_feasibility(ABSTAIN)
    carrier.solve_label_map_top2()
    torch.cuda.synchronize()  # readback only — not producer ordering

    map_results = from_dlpack(carrier.export_buffer("map_results")).cpu()
    best, ambiguous, _, margin_bits = (int(v) for v in map_results[0])
    assert best == 0 and ambiguous == 0
    assert struct.unpack("f", struct.pack("I", margin_bits))[0] == 1.5

    with pytest.raises(pyxlog.CarrierRefused, match="null producer stream"):
        carrier.note_producer_stream(0)


def test_consumer_events_order_solve_before_external_stream_reads() -> None:
    carrier = _registered()
    domains = from_dlpack(carrier.export_buffer("domains"))
    constraints = from_dlpack(carrier.export_buffer("constraints"))
    feasible = from_dlpack(carrier.export_buffer("feasible_sets"))

    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        domains.copy_(
            torch.tensor([[0b011], [0b001], [0b100]], dtype=torch.int64, device="cuda")
        )
        constraints.copy_(
            torch.tensor([[0, 1], [1, 1]], dtype=torch.int32, device="cuda")
        )
        carrier.note_producer_stream(producer.cuda_stream)

    consumers = [torch.cuda.Stream(), torch.cuda.Stream()]
    for consumer in consumers:
        carrier.note_consumer_stream(consumer.cuda_stream)

    carrier.solve_label_feasibility(ABSTAIN)

    # Each clone is enqueued after the wait that the carrier placed on
    # that external stream. Synchronizing only the consumer streams is
    # therefore sufficient; there is no host/global solve barrier.
    snapshots = []
    for consumer in consumers:
        with torch.cuda.stream(consumer):
            snapshots.append(feasible.clone())
    for consumer in consumers:
        consumer.synchronize()

    assert [int(snapshot[0, 0]) for snapshot in snapshots] == [0b1101, 0b1101]
    assert [int(snapshot[1, 0]) for snapshot in snapshots] == [0b1001, 0b1001]

    with pytest.raises(pyxlog.CarrierRefused, match="null consumer stream"):
        carrier.note_consumer_stream(0)


def test_component_exact_rejects_greedy_infeasible_joint() -> None:
    import struct

    carrier = _carrier(fuel_limit=64)
    carrier.register_schema(CATALOG_SHA, pyxlog.SOLVER_ABI_IDENTITY)
    # Labels: L0 wants tail sort0, L1 infeasible-by-head trap with the
    # highest raw score, L2 wants tail sort1, L3 = unconstrained
    # abstain. Both candidates share tail entity 1 (plural domain).
    carrier.bind_signatures(
        [0b01, 0b10, 0b01, 0b00], [0b01, 0b01, 0b10, 0b00]
    )
    domains = from_dlpack(carrier.export_buffer("domains"))
    constraints = from_dlpack(carrier.export_buffer("constraints"))
    scores = from_dlpack(carrier.export_buffer("scores"))
    domains[0, 0] = 0b01
    domains[1, 0] = 0b011
    domains[2, 0] = 0b01
    constraints.copy_(
        torch.tensor([[0, 1], [2, 1]], dtype=torch.int32, device="cuda")
    )
    # Greedy per-candidate maxima are c0=L2 (5.0) and c1=L0 (5.0),
    # which pin shared entity 1 to DIFFERENT sorts — jointly
    # infeasible; the exact optimum is (L2, L2) with total 9.0.
    scores.copy_(
        torch.tensor(
            [[1.0, 9.0, 5.0, 0.5], [5.0, 9.0, 4.0, 0.5]],
            dtype=torch.float32,
            device="cuda",
        )
    )
    torch.cuda.synchronize()
    carrier.solve_label_feasibility(ABSTAIN)
    carrier.solve_label_map_top2()

    with pytest.raises(pyxlog.CarrierRefused, match="offsets"):
        carrier.solve_components_exact([1, 2], [0, 1])

    carrier.solve_components_exact([0, 2], [0, 1])
    torch.cuda.synchronize()

    status = from_dlpack(carrier.export_buffer("solve_status")).cpu()
    assert int(status[0, 0]) == 2 and int(status[1, 0]) == 2
    map_results = from_dlpack(carrier.export_buffer("map_results")).cpu()
    for row in (0, 1):
        best, ambiguous, total_bits, margin_bits = (
            int(v) for v in map_results[row]
        )
        assert best == 2, "exact stage must pick the consistent optimum"
        assert ambiguous == 0
        assert struct.unpack("f", struct.pack("I", total_bits))[0] == 9.0
        assert struct.unpack("f", struct.pack("I", margin_bits))[0] == 3.0
    # Device-measured exact accounting (carrier 47792ce07): the meter
    # reconciles to actual expansions and refunds the unspent upfront
    # authorization — pin-scoped update predicted by the replay map.
    assert carrier.fuel_spent == 25


def test_fuel_saturates_across_python_calls() -> None:
    # Budget covers exactly one solve (candidates x labels = 8).
    carrier = _registered(fuel_limit=CANDIDATES * LABELS)
    carrier.solve_label_feasibility(ABSTAIN)
    assert carrier.fuel_spent == CANDIDATES * LABELS
    with pytest.raises(pyxlog.SolverResourceExhausted, match="8 of 8"):
        carrier.solve_label_feasibility(ABSTAIN)
    with pytest.raises(pyxlog.SolverResourceExhausted, match="8 of 8"):
        carrier.solve_label_feasibility(ABSTAIN)
    assert carrier.fuel_spent == CANDIDATES * LABELS, "refused charge not applied"
