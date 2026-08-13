"""GPU-only reusable conditioned-circuit behavior through the installed wheel."""

from __future__ import annotations

import sys
import threading
import time

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()

LOGIC_SOURCE = """
pred fact().
pred accepted().
fact().
accepted() :- know fact().
"""

PROB_SOURCE = """
0.5::target().
0.6::fact().
query(target()).
query(fact()).
"""

REUSE_TRACE_KEYS = (
    "gpu_conditioned_circuit_reuses",
    "gpu_conditioned_circuit_preparation_compiles",
    "gpu_conditioned_circuit_materializations",
    "gpu_conditioned_circuit_disk_cache_restores",
    "gpu_conditioned_circuit_gpu_cache_hits",
    "gpu_conditioned_circuit_generation",
    "gpu_conditioned_circuit_cache_slot",
)


def _values(capsule) -> list[float]:
    return [
        float(value)
        for value in torch.utils.dlpack.from_dlpack(capsule).cpu().reshape(-1).tolist()
    ]


def _snapshot(result) -> tuple[list[str], list[float], list[float], float]:
    return result.atoms, _values(result.prob), _values(result.log_prob), result.log_z_e


def test_prepared_conditioned_program_reuses_one_circuit_across_priors(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("XLOG_CIRCUIT_CACHE_DIR", str(tmp_path / "circuit-cache"))
    logic = pyxlog.LogicProgram.compile(LOGIC_SOURCE)
    prepared = logic.prepare_conditioned(PROB_SOURCE)
    variable_map = prepared.prob_var_map()
    target_var = next(
        index
        for index, entry in enumerate(variable_map)
        if entry["kind"] == "fact" and entry["atom"].replace(" ", "") == "target()"
    )
    evidence_var = next(
        index
        for index, entry in enumerate(variable_map)
        if entry["kind"] == "fact" and entry["atom"].replace(" ", "") == "fact()"
    )
    direct_trace = logic.evaluate_conditioned(PROB_SOURCE).trace
    assert all(key in direct_trace for key in REUSE_TRACE_KEYS)
    assert direct_trace["gpu_conditioned_circuit_generation"] == 0
    assert direct_trace["gpu_conditioned_circuit_cache_slot"] == 0
    circuit_identity = None

    for target_prior, evidence_prior in (
        (0.5, 0.6),
        (0.9, 0.6),
        (0.1, 0.6),
        (0.1, 0.2),
    ):
        prepared.set_fact_probabilities(
            {target_var: target_prior, evidence_var: evidence_prior}
        )
        evaluated = prepared.evaluate()
        actual = _snapshot(evaluated)
        fresh_source = PROB_SOURCE.replace(
            "0.5::target().", f"{target_prior}::target()."
        ).replace("0.6::fact().", f"{evidence_prior}::fact().")
        expected = _snapshot(logic.evaluate_conditioned(fresh_source))

        assert actual[0] == expected[0]
        assert actual[1] == pytest.approx(expected[1], abs=1e-9)
        assert actual[2] == pytest.approx(expected[2], abs=1e-9)
        assert actual[3] == pytest.approx(expected[3], abs=1e-9)

        variable_map = prepared.prob_var_map()
        assert variable_map[target_var]["prob"] == pytest.approx(target_prior)
        assert variable_map[evidence_var]["prob"] == pytest.approx(evidence_prior)

        trace = evaluated.trace
        assert trace["gpu_exact_source_compiles"] == 0
        assert trace["gpu_exact_program_compiles"] == 0
        assert trace["gpu_conditioned_circuit_reuses"] == 1
        assert trace["gpu_conditioned_circuit_preparation_compiles"] == 1
        assert trace["gpu_conditioned_circuit_materializations"] == 1
        assert trace["gpu_conditioned_circuit_disk_cache_restores"] == 0
        assert trace["gpu_conditioned_circuit_gpu_cache_hits"] == 0
        current_identity = (
            trace["gpu_conditioned_circuit_generation"],
            trace["gpu_conditioned_circuit_cache_slot"],
        )
        assert current_identity[0] > 0
        if circuit_identity is None:
            circuit_identity = current_identity
        else:
            assert current_identity == circuit_identity

    restored = logic.prepare_conditioned(PROB_SOURCE).evaluate()
    restored_trace = restored.trace
    assert restored_trace["gpu_exact_source_compiles"] == 0
    assert restored_trace["gpu_exact_program_compiles"] == 0
    assert restored_trace["gpu_conditioned_circuit_preparation_compiles"] == 0
    assert restored_trace["gpu_conditioned_circuit_materializations"] == 1
    assert restored_trace["gpu_conditioned_circuit_disk_cache_restores"] == 1
    assert restored_trace["gpu_conditioned_circuit_gpu_cache_hits"] == 0
    restored_identity = (
        restored_trace["gpu_conditioned_circuit_generation"],
        restored_trace["gpu_conditioned_circuit_cache_slot"],
    )
    assert restored_identity != circuit_identity


def test_prepared_conditioned_program_rejects_invalid_updates_atomically() -> None:
    logic = pyxlog.LogicProgram.compile(LOGIC_SOURCE)
    prepared = logic.prepare_conditioned(PROB_SOURCE)
    variable_map = prepared.prob_var_map()
    target_var = next(
        index
        for index, entry in enumerate(variable_map)
        if entry["kind"] == "fact" and entry["atom"].replace(" ", "") == "target()"
    )
    other_var = next(
        index
        for index, entry in enumerate(variable_map)
        if index != 0 and entry["kind"] == "other"
    )
    before = variable_map[target_var]["prob"]

    for updates in (
        {target_var: float("nan")},
        {target_var: -0.1},
        {target_var: 1.1},
        {target_var: 0.8, 0: 0.2},
        {target_var: 0.8, other_var: 0.2},
        {target_var: 0.8, len(variable_map) + 10: 0.2},
    ):
        with pytest.raises((ValueError, RuntimeError)):
            prepared.set_fact_probabilities(updates)
        assert prepared.prob_var_map()[target_var]["prob"] == pytest.approx(before)


def test_prepared_conditioned_program_rejects_unsupported_engines_and_device_budget() -> None:
    logic = pyxlog.LogicProgram.compile(LOGIC_SOURCE)
    with pytest.raises((ValueError, RuntimeError), match="exact"):
        logic.prepare_conditioned("#pragma prob_engine = mc\n" + PROB_SOURCE)
    count_lift_source = "".join(
        f"0.5::edge(1, {index}).\n" for index in range(1, 18)
    ) + """
0.6::fact().
out_degree(X, count(Y)) :- edge(X, Y).
query(out_degree(1, 8)).
"""
    with pytest.raises((ValueError, RuntimeError), match="count-lift"):
        logic.prepare_conditioned(count_lift_source)
    with pytest.raises((ValueError, RuntimeError), match="memory|budget"):
        logic.prepare_conditioned(PROB_SOURCE, memory_mb=0)


def test_prepared_conditioned_program_rejects_choice_variable_updates() -> None:
    logic = pyxlog.LogicProgram.compile(LOGIC_SOURCE)
    prepared = logic.prepare_conditioned(
        "0.5::side(left); 0.5::side(right).\n0.6::fact().\nquery(side(left)).\n"
    )
    choice_var = next(
        index
        for index, entry in enumerate(prepared.prob_var_map())
        if entry["kind"] == "choice"
    )
    with pytest.raises((ValueError, RuntimeError), match="choice"):
        prepared.set_fact_probabilities({choice_var: 0.2})


def test_prepared_prob_var_map_releases_gil_while_waiting_for_shared_state() -> None:
    fact_count = 512
    source = "\n".join(f"0.5::weight({index})." for index in range(fact_count))
    source += "\n0.6::fact().\nquery(fact()).\n"
    logic = pyxlog.LogicProgram.compile(LOGIC_SOURCE)
    prepared = logic.prepare_conditioned(source)
    updates = {
        index: 0.75
        for index, entry in enumerate(prepared.prob_var_map())
        if entry["kind"] == "fact" and entry["atom"].startswith("weight(")
    }
    assert len(updates) == fact_count

    setter_entered = threading.Event()
    setter_done = threading.Event()
    setter_errors: list[Exception] = []

    def update_all_facts() -> None:
        try:
            setter_entered.set()
            prepared.set_fact_probabilities(updates)
        except Exception as exc:  # pragma: no cover - surfaced on test thread
            setter_errors.append(exc)
        finally:
            setter_done.set()

    setter = threading.Thread(target=update_all_facts, daemon=True)
    setter.start()
    assert setter_entered.wait(timeout=10)
    time.sleep(0.005)
    assert not setter_done.is_set(), "bulk update finished before lock-contention witness"

    observer_ready = threading.Event()
    begin_map = threading.Event()
    map_returned = threading.Event()
    observer_saw_return: list[bool] = []

    def observe() -> None:
        observer_ready.set()
        assert begin_map.wait(timeout=30)
        observer_saw_return.append(map_returned.is_set())

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    assert observer_ready.wait(timeout=10)

    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(60.0)
    try:
        begin_map.set()
        variable_map = prepared.prob_var_map()
    finally:
        map_returned.set()
        sys.setswitchinterval(previous_switch_interval)

    observer.join(timeout=10)
    setter.join(timeout=30)
    assert not observer.is_alive()
    assert not setter.is_alive()
    assert not setter_errors
    assert observer_saw_return == [False], "observer ran only after prob_var_map returned"
    assert len(variable_map) > fact_count
