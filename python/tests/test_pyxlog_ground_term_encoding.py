"""Production-wheel coverage for ILP ground-term fact loading."""

import pytest

pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


def test_ilp_factory_loads_all_supported_ground_term_encodings() -> None:
    program = pyxlog.IlpProgramFactory.compile(
        r'''
            pred integers(u32, u64, i32, i64).
            pred floats(f32, f64).
            pred booleans(bool, bool).
            pred symbols(symbol, symbol, symbol).

            integers(42, 43, -44, -45).
            floats(1.5, 2.25).
            booleans(true, 0).
            symbols("hello", hello, world).
        ''',
        device=0,
        memory_mb=256,
    )

    assert dict(program.relation_type_annotations()) == {
        "booleans": ["bool", "bool"],
        "floats": ["f32", "f64"],
        "integers": ["u32", "u64", "i32", "i64"],
        "symbols": ["symbol", "symbol", "symbol"],
    }
    assert program.relation_facts("integers") == [[42, 43, -44, -45]]
    assert program.relation_facts("booleans") == [[1, 0]]

    symbol_ids = program.relation_facts("symbols")
    assert len(symbol_ids) == 1
    assert symbol_ids[0][0] == symbol_ids[0][1]
    assert symbol_ids[0][0] != symbol_ids[0][2]


def test_ilp_factory_reports_shared_encoder_context() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            r"^Execution error: Failed to encode fact for predicate invalid at column 0: "
            r"Fact cannot contain variable X$"
        ),
    ):
        pyxlog.IlpProgramFactory.compile(
            "pred invalid(u32). invalid(X).",
            device=0,
            memory_mb=256,
        )
