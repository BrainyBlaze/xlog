import gc
import os

import pytest

import pyxlog


def test_provider_memory_stats_preserves_transient_peak():
    row_count = 256
    facts = "\n".join(f"edge({row}, {row + 1})." for row in range(row_count))
    source = f"""
pred edge(i32, i32).
pred reach(i32, i32).
{facts}
reach(X, Y) :- edge(X, Y).
?- reach(X, Y).
"""
    try:
        program = pyxlog.LogicProgram.compile(source, device=0, memory_mb=64)
    except RuntimeError as error:
        if os.environ.get("XLOG_REQUIRE_CUDA") == "1":
            raise RuntimeError(
                f"XLOG_REQUIRE_CUDA=1 but pyxlog CUDA initialization failed: {error}"
            ) from error
        pytest.skip(f"pyxlog CUDA unavailable: {error}")

    result = program.evaluate()
    assert result.queries[0].num_rows == row_count
    live_stats = program.memory_stats()

    del result
    gc.collect()
    settled_stats = program.memory_stats()

    assert settled_stats["allocated_bytes"] < live_stats["allocated_bytes"]
    assert settled_stats["peak_memory_bytes"] >= live_stats["allocated_bytes"]
    assert settled_stats["peak_memory_bytes"] > settled_stats["allocated_bytes"]
