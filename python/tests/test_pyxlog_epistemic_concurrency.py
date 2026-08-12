"""GPU epistemic calls must not monopolize the CPython interpreter lock."""

from __future__ import annotations

import math
import sys
import threading
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


def _call_while_observer_waits(operation: Callable[[], Any]) -> Any:
    rendezvous = threading.Barrier(2)
    observer_waiting = threading.Event()
    begin_call = threading.Event()
    call_returned = threading.Event()
    observer_progressed = threading.Event()
    observer_saw_return: list[bool] = []
    observer_errors: list[Exception] = []

    def observe() -> None:
        try:
            rendezvous.wait(timeout=10)
            observer_waiting.set()
            if not begin_call.wait(timeout=30):
                raise AssertionError("native call was never started")
            observer_saw_return.append(call_returned.is_set())
            observer_progressed.set()
        except Exception as exc:  # pragma: no cover - surfaced on the test thread
            observer_errors.append(exc)

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    rendezvous.wait(timeout=10)
    assert observer_waiting.wait(timeout=10)

    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(60.0)
    try:
        begin_call.set()
        result = operation()
    finally:
        call_returned.set()
        sys.setswitchinterval(previous_switch_interval)

    observer.join(timeout=10)
    assert not observer.is_alive()
    assert not observer_errors
    assert observer_progressed.is_set()
    assert observer_saw_return == [False], "observer ran only after the native call returned"
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
