"""Calculated guard operands retain dependency order on the native GPU."""

from __future__ import annotations

import pytest

pyxlog = pytest.importorskip("pyxlog")


@pytest.mark.parametrize("scalar_type", ["u64", "i64"])
@pytest.mark.parametrize("guard_first", [False, True])
def test_calculated_guards_run_before_dependent_division(
    tmp_path, scalar_type, guard_first
):
    computed = "A > 0, A is N / D" if guard_first else "A is N / D, A > 0"
    source = tmp_path / "guarded.xlog"
    source.write_text(
        f"pred input(symbol,{scalar_type},{scalar_type}).\n"
        f"pred result(symbol,{scalar_type}).\n"
        f"result(K,Q) :- input(K,N,D), D > 0, {computed}, "
        f"Q is cast(100,{scalar_type}) / A, Q > 0.\n"
        "?- result(K,Q).\n"
    )
    compiled = pyxlog.LogicProgram.compile_file(
        source, module_paths=[tmp_path], device=0, memory_mb=512
    )
    session = compiled.session()
    session.put_relation_rows(
        "input",
        [
            ["accepted", "25", "5"],
            ["input_zero", "10", "0"],
            ["computed_zero", "1", "2"],
            ["result_zero", "101", "1"],
        ],
    )
    session.reset_host_transfer_stats()
    session.set_strict_deterministic_d2h(True)
    session.reset_deterministic_d2h_violations()
    session.evaluate()
    assert session.host_transfer_stats() == {
        "htod_bytes": 0,
        "dtoh_bytes": 0,
        "htod_calls": 0,
        "dtoh_calls": 0,
    }
    assert session.deterministic_d2h_violation_count() == 0
    session.set_strict_deterministic_d2h(False)
    assert session.export_relation_rows("__xlog_query_0") == [["accepted", "20"]]
