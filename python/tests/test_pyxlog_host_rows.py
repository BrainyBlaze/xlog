from __future__ import annotations

import os

import pytest


pyxlog = pytest.importorskip("pyxlog")


def _compile_or_skip(entrypoint, tmp_path):
    try:
        return pyxlog.LogicProgram.compile_file(
            entrypoint,
            module_paths=[tmp_path],
            device=0,
            memory_mb=512,
        )
    except RuntimeError as exc:
        if os.environ.get("XLOG_REQUIRE_CUDA") == "1":
            raise
        pytest.skip(f"CUDA is unavailable to Pyxlog: {exc}")


def test_session_host_rows_keep_exact_evaluation_inside_zero_transfer_window(
    tmp_path,
) -> None:
    entrypoint = tmp_path / "main.xlog"
    entrypoint.write_text(
        "pred source(symbol, u32, u64, i32, i64, f32, f64, bool).\n"
        "pred result(symbol, u32, u64, i32, i64, f32, f64, bool).\n"
        "result(A, B, C, D, E, F, G, H) :- source(A, B, C, D, E, F, G, H).\n"
        "?- result(A, B, C, D, E, F, G, H).\n",
        encoding="utf-8",
    )
    compiled = _compile_or_skip(entrypoint, tmp_path)
    session = compiled.session()

    with pytest.raises(ValueError, match="lexical row arity"):
        session.put_relation_rows("source", [["too", "short"]])
    with pytest.raises(ValueError, match="not valid F32"):
        session.put_relation_rows(
            "source",
            [["alpha", "1", "2", "-3", "-4", "nan", "2.5", "true"]],
        )
    with pytest.raises(ValueError, match="not valid Bool"):
        session.put_relation_rows(
            "source",
            [["alpha", "1", "2", "-3", "-4", "1.5", "2.5", "yes"]],
        )

    session.put_relation_rows(
        "source",
        [
            ["alpha", "1", "2", "-3", "-4", "1.5", "2.5", "true"],
            ["beta", "5", "6", "-7", "-8", "-1.25", "-2.75", "false"],
        ],
    )

    session.reset_host_transfer_stats()
    session.set_strict_deterministic_d2h(True)
    session.reset_deterministic_d2h_violations()
    evaluated = session.evaluate()
    stats = session.host_transfer_stats()

    assert len(evaluated.queries) == 1
    assert stats == {
        "dtoh_bytes": 0,
        "htod_bytes": 0,
        "dtoh_calls": 0,
        "htod_calls": 0,
    }
    assert session.strict_deterministic_d2h_enabled() is True
    assert session.deterministic_d2h_violation_count() == 0

    session.set_strict_deterministic_d2h(False)
    assert session.export_relation_rows("__xlog_query_0") == [
        ["alpha", "1", "2", "-3", "-4", "1.5", "2.5", "true"],
        ["beta", "5", "6", "-7", "-8", "-1.25", "-2.75", "false"],
    ]
