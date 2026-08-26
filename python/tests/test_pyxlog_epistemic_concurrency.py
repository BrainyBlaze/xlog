"""GPU epistemic calls must not monopolize the CPython interpreter lock."""

from __future__ import annotations

import math
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


_JOIN_WIDTH = 2_048
_PROB_SOURCE = """
0.6::observed().
query(observed()).
"""

_MC_SAMPLES = 1_000_000


@pytest.fixture(scope="module")
def long_epistemic_program():
    left = "\n".join(f"left({index})." for index in range(_JOIN_WIDTH))
    right = "\n".join(f"right({index})." for index in range(_JOIN_WIDTH))
    source = f"""
pred left(u32).
pred right(u32).
pred pair(u32, u32).
pred observed().
pred accepted().

{left}
{right}
observed().

pair(X, Y) :- left(X), right(Y).
accepted() :- pair({_JOIN_WIDTH - 1}, {_JOIN_WIDTH - 1}), know observed().
"""
    return pyxlog.LogicProgram.compile(source, device=0, memory_mb=1_024)


@pytest.fixture(scope="module")
def long_logic_program():
    left = "\n".join(f"left({index})." for index in range(_JOIN_WIDTH))
    right = "\n".join(f"right({index})." for index in range(_JOIN_WIDTH))
    source = f"""
pred left(u32).
pred right(u32).
pred pair(u32, u32).

{left}
{right}
pair(X, Y) :- left(X), right(Y).
?- pair(X, Y).
"""
    return pyxlog.LogicProgram.compile(source, device=0, memory_mb=1_024)


@pytest.fixture(scope="module")
def long_mc_program():
    return pyxlog.Program.compile(_PROB_SOURCE, prob_engine="mc")


def _call_while_observer_waits(operation: Callable[[], Any]) -> Any:
    observer_started = threading.Event()
    stop_observer = threading.Event()
    observer_iterations = [0]

    def observe() -> None:
        observer_started.set()
        while not stop_observer.is_set():
            observer_iterations[0] += 1
            time.sleep(0)

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    assert observer_started.wait(timeout=10)
    deadline = time.monotonic() + 10
    while observer_iterations[0] == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert observer_iterations[0] > 0

    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(60.0)
    try:
        iterations_before_call = observer_iterations[0]
        result = operation()
        iterations_after_call = observer_iterations[0]
    finally:
        sys.setswitchinterval(previous_switch_interval)
        stop_observer.set()

    observer.join(timeout=10)
    assert not observer.is_alive()
    assert iterations_after_call > iterations_before_call, (
        "observer made no progress while the native call held the GIL"
    )
    return result


def test_epistemic_evidence_releases_gil(long_epistemic_program) -> None:
    evidence = _call_while_observer_waits(long_epistemic_program.epistemic_evidence)

    assert evidence.accepted_world_views >= 1
    assert evidence.final_output_rows == 1


def test_evaluate_conditioned_releases_gil(long_epistemic_program) -> None:
    result = _call_while_observer_waits(
        lambda: long_epistemic_program.evaluate_conditioned(_PROB_SOURCE)
    )

    assert result.log_z_e == pytest.approx(math.log(0.6), abs=1e-9)
    assert result.trace["gpu_conditioned_evidence_facts"] == 1


def test_compiled_logic_evaluate_releases_gil(long_logic_program) -> None:
    result = _call_while_observer_waits(long_logic_program.evaluate)

    assert len(result.queries) == 1
    assert result.queries[0].relation_name == "pair"


def test_logic_session_evaluate_releases_gil(long_logic_program) -> None:
    session = long_logic_program.session()
    result = _call_while_observer_waits(session.evaluate)

    assert len(result.queries) == 1
    assert result.queries[0].relation_name == "pair"


def test_probabilistic_evaluate_releases_gil(long_mc_program) -> None:
    result = _call_while_observer_waits(
        lambda: long_mc_program.evaluate(samples=_MC_SAMPLES, seed=17)
    )

    assert result.samples == _MC_SAMPLES
    assert result.seed == 17


def test_probabilistic_device_evaluate_releases_gil(long_mc_program) -> None:
    result = _call_while_observer_waits(
        lambda: long_mc_program.evaluate_device(samples=_MC_SAMPLES, seed=17)
    )

    assert result.total_samples == _MC_SAMPLES
    assert result.seed == 17
